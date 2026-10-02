"""GoEmotions as a second sentiment corpus with rater-level labels.

DynaSent is the only widely used English sentiment corpus that releases every
crowdworker's judgement for every item, so a replication on a second corpus needs one
whose sentiment distribution can be *derived* from rater-level labels. GoEmotions
(Demszky et al., 2020) releases 211,225 individual ratings of 58,011 Reddit comments: each
rater marks one or more of 27 emotions or ``neutral``, or flags the comment as very
unclear. Its authors group the emotions by sentiment into positive, negative and
ambiguous (``sentiment_mapping.json`` in their release), which is the bridge used here.

Each *rating*, not each item, is mapped onto the four-way space of the main study, so the
target stays a distribution over raters exactly as DynaSent's is:

    positive and negative emotions both marked   -> mixed
    only positive (with or without ambiguous)    -> positive
    only negative (with or without ambiguous)    -> negative
    neutral                                       -> neutral
    only ambiguous emotions (surprise, curiosity, confusion, realization)
                                                  -> neutral, or dropped (``ambiguous_as``)
    flagged very unclear, or nothing marked       -> dropped

The ambiguous group carries no polarity of its own, which is why it is folded into
"no sentiment" by default and why the alternative is a parameter rather than a comment:
it is the one derivation choice a reader might make differently, and the build records
which one produced it.

Three facts about the release shape the rest of the pipeline and are checked at load:

* Items carry three raters, and five where the first three agreed on no emotion, so the
  denominator of ``p_human`` varies per item. The preference builder is called with a
  per-item denominator for that reason.
* The simplified release's train/dev/test files list the item ids of the official split
  and cover 54,263 of the 58,011 items; the other 3,748 are items on which no emotion
  reached two raters, the high-disagreement end of the corpus. They belong to no official
  split and join the training-side pool, where the ambiguity-stratified held-out set can
  draw on them.
* Short comments repeat ("Thank you", "Happy cake day"), 1.4 percent of items, above the
  resampling guard. One item per normalised text is kept, preferring the official test
  split, then dev, so no text is scored on one side of the split and trained on the other.
"""

from __future__ import annotations

import csv
import hashlib
import json
import ssl
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

from ..labels import MIXED, NEGATIVE, NEUTRAL, NO_MAJORITY, POSITIVE
from .dynasent import Item
from .integrity import _hash64, normalise_text

RELEASE = "https://storage.googleapis.com/gresearch/goemotions/data/full_dataset"
REPO = ("https://raw.githubusercontent.com/google-research/google-research/master/"
        "goemotions/data")

#: Raw ratings, one row per (item, rater), and the official split ids.
RAW_FILES = ("goemotions_1.csv", "goemotions_2.csv", "goemotions_3.csv")
SPLIT_FILES = {"train": "train.tsv", "dev": "dev.tsv", "test": "test.tsv"}
MAPPING_FILE = "sentiment_mapping.json"

#: The authors' sentiment grouping, copied so a build does not depend on a network file;
#: ``check_mapping`` compares it with the downloaded one when that is present.
SENTIMENT_GROUPS: Mapping[str, tuple[str, ...]] = {
    "positive": ("amusement", "excitement", "joy", "love", "desire", "optimism", "caring",
                 "pride", "admiration", "gratitude", "relief", "approval"),
    "negative": ("fear", "nervousness", "remorse", "embarrassment", "disappointment",
                 "sadness", "grief", "disgust", "anger", "annoyance", "disapproval"),
    "ambiguous": ("realization", "surprise", "curiosity", "confusion"),
}
EMOTION_GROUP: dict[str, str] = {e: g for g, emotions in SENTIMENT_GROUPS.items()
                                 for e in emotions}
EMOTIONS: tuple[str, ...] = tuple(sorted(EMOTION_GROUP)) + ("neutral",)

AMBIGUOUS_AS = ("neutral", "drop")
#: An item needs this many usable ratings. Two raters cannot express a majority short of
#: unanimity, and one cannot express disagreement at all.
MIN_RATERS = 3
#: Round names, which also serve as strata for the held-out split and as GR-DPO groups.
ROUNDS = {"train": "go_train", "dev": "go_dev", "test": "go_test", None: "go_unsplit"}
#: When two items share a normalised text, the one in the earlier split is kept.
DEDUP_PRIORITY = ("go_test", "go_dev", "go_train", "go_unsplit")


def check_mapping(path: Path) -> None:
    """Refuse a build whose released grouping differs from the one coded here."""
    released = json.loads(path.read_text())
    coded = {g: sorted(v) for g, v in SENTIMENT_GROUPS.items()}
    if {g: sorted(v) for g, v in released.items()} != coded:
        raise ValueError(f"{path} does not match the sentiment grouping this loader uses; "
                         f"released {released}")


def rater_sentiment(marked: Iterable[str], *, ambiguous_as: str = "neutral") -> str | None:
    """One rater's emotions mapped to a sentiment label, or ``None`` for no usable rating."""
    if ambiguous_as not in AMBIGUOUS_AS:
        raise ValueError(f"ambiguous_as must be one of {AMBIGUOUS_AS}")
    marked = set(marked)
    unknown = marked - set(EMOTIONS)
    if unknown:
        raise ValueError(f"unknown emotion labels {sorted(unknown)}")
    if not marked:
        return None
    groups = {EMOTION_GROUP.get(e, "neutral") for e in marked}
    positive, negative = "positive" in groups, "negative" in groups
    if positive and negative:
        return MIXED
    if positive:
        return POSITIVE
    if negative:
        return NEGATIVE
    if "neutral" in groups:
        return NEUTRAL
    return NEUTRAL if ambiguous_as == "neutral" else None


def gold_from_votes(votes: Mapping[str, int]) -> str:
    """Majority label, or ``no-majority`` unless one label has more than half the votes.

    The rule the NLI loader uses, which is DynaSent's three of five at five raters and two
    of three at three.
    """
    total = sum(votes.values())
    if not total:
        return NO_MAJORITY
    top = max(votes, key=lambda k: votes[k])
    return top if votes[top] / total > 0.5 else NO_MAJORITY


def band_of(votes: Mapping[str, int], gold: str) -> str:
    """Unanimous, majority or no majority: comparable across three and five raters."""
    if gold == NO_MAJORITY:
        return "no-majority"
    return "unanimous" if max(votes.values()) == sum(votes.values()) else "majority"


# --------------------------------------------------------------------------- download

def _ssl_context() -> ssl.SSLContext | None:
    try:
        import certifi
    except ImportError:
        return None
    return ssl.create_default_context(cafile=certifi.where())


def download(cache_dir: Path) -> dict[str, str]:
    """Fetch the release once; return the sha256 of every file for the build manifest.

    A file already present is used as-is, so a copy placed by hand is picked up. Each file
    is checked to be what it claims before it is trusted, because a failed fetch from a
    file host comes back as an HTML page with a success status.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    wanted = {name: f"{RELEASE}/{name}" for name in RAW_FILES}
    wanted.update({name: f"{REPO}/{name}" for name in (*SPLIT_FILES.values(), MAPPING_FILE)})
    digests = {}
    for name, url in wanted.items():
        dest = cache_dir / name
        if not dest.exists():
            request = urllib.request.Request(url, headers={"User-Agent": "sentalign/1.0"})
            with urllib.request.urlopen(request, context=_ssl_context()) as src:
                dest.write_bytes(src.read())
        head = dest.read_bytes()[:64].lstrip()
        if head[:1] == b"<":
            dest.rename(dest.with_suffix(dest.suffix + ".rejected"))
            raise ValueError(f"{name}: the host returned HTML, not the file; fetch it by "
                             f"hand into {dest}:\n    wget -O {dest} '{url}'")
        if name in RAW_FILES and not head.startswith(b"text,id,"):
            raise ValueError(f"{name}: unexpected header {head[:40]!r}")
        digests[name] = hashlib.sha256(dest.read_bytes()).hexdigest()
    check_mapping(cache_dir / MAPPING_FILE)
    return digests


# ------------------------------------------------------------------------------- load

def iter_ratings(cache_dir: Path) -> Iterator[dict]:
    """Every raw rating row, with the emotion columns checked against the grouping."""
    for name in RAW_FILES:
        with (cache_dir / name).open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            columns = set(reader.fieldnames or ())
            missing = set(EMOTIONS) - columns
            if missing:
                raise KeyError(f"{name} lacks emotion columns {sorted(missing)}")
            yield from reader


def official_split(cache_dir: Path) -> dict[str, str]:
    """Item id to official split, from the simplified release's three files."""
    split_of: dict[str, str] = {}
    for split, name in SPLIT_FILES.items():
        with (cache_dir / name).open(newline="", encoding="utf-8") as fh:
            for row in csv.reader(fh, delimiter="\t", quoting=csv.QUOTE_NONE):
                if len(row) != 3:
                    raise ValueError(f"{name}: expected text, labels, id; got {len(row)} "
                                     f"fields")
                item_id = row[2].strip()
                if split_of.setdefault(item_id, split) != split:
                    raise ValueError(f"{item_id} is in two official splits")
    return split_of


@dataclass
class LoadReport:
    ratings: int = 0
    ratings_unusable: int = 0
    ratings_ambiguous_only: int = 0
    items: int = 0
    items_too_few_raters: int = 0
    items_duplicate_text: int = 0
    raters_per_item: Counter = field(default_factory=Counter)
    ambiguous_as: str = "neutral"

    def as_dict(self) -> dict:
        out = dict(self.__dict__)
        out["raters_per_item"] = {str(k): v for k, v in sorted(self.raters_per_item.items())}
        return out


def items_from_ratings(rows: Iterable[Mapping[str, str]], split_of: Mapping[str, str], *,
                       ambiguous_as: str = "neutral", min_raters: int = MIN_RATERS,
                       report: LoadReport | None = None) -> list[Item]:
    """Group ratings by item and build one ``Item`` per comment with enough raters."""
    report = report if report is not None else LoadReport(ambiguous_as=ambiguous_as)
    texts: dict[str, str] = {}
    votes: dict[str, Counter] = defaultdict(Counter)
    ambiguous_only: Counter = Counter()
    for row in rows:
        report.ratings += 1
        item_id = row["id"]
        if texts.setdefault(item_id, row["text"]) != row["text"]:
            raise ValueError(f"{item_id} carries two different texts")
        marked = [e for e in EMOTIONS if str(row[e]).strip() == "1"]
        if str(row.get("example_very_unclear", "")).strip().lower() == "true":
            marked = []
        label = rater_sentiment(marked, ambiguous_as=ambiguous_as)
        only_ambiguous = bool(marked) and all(EMOTION_GROUP.get(e) == "ambiguous"
                                              for e in marked)
        if only_ambiguous:
            report.ratings_ambiguous_only += 1
            ambiguous_only[item_id] += 1
        if label is None:
            report.ratings_unusable += 1
            votes.setdefault(item_id, Counter())
            continue
        votes[item_id][label] += 1

    items: list[Item] = []
    for item_id in sorted(texts):
        counts = {y: votes[item_id].get(y, 0) for y in (POSITIVE, NEGATIVE, NEUTRAL, MIXED)}
        n = sum(counts.values())
        if n < min_raters:
            report.items_too_few_raters += 1
            continue
        report.raters_per_item[n] += 1
        gold = gold_from_votes(counts)
        items.append(Item(
            text_id=item_id, text=texts[item_id], votes=counts, gold=gold,
            round=ROUNDS[split_of.get(item_id)], source="goemotions",
            meta={"band": band_of(counts, gold), "n_annotators": n,
                  "ambiguous_only_ratings": ambiguous_only[item_id]}))
    report.items = len(items)
    return items


def dedupe_by_text(items: Sequence[Item], *, seed: int = 0,
                   report: LoadReport | None = None) -> list[Item]:
    """One item per normalised text: earliest split in ``DEDUP_PRIORITY``, then by hash."""
    rank = {name: i for i, name in enumerate(DEDUP_PRIORITY)}
    keep: dict[str, Item] = {}
    for item in items:
        key = normalise_text(item.text)
        current = keep.get(key)
        if current is None or ((rank[item.round], _hash64(item.text_id, seed))
                               < (rank[current.round], _hash64(current.text_id, seed))):
            keep[key] = item
    kept_ids = {i.text_id for i in keep.values()}
    if report is not None:
        report.items_duplicate_text = len(items) - len(kept_ids)
    return [i for i in items if i.text_id in kept_ids]


def load_goemotions(cache_dir: Path, *, ambiguous_as: str = "neutral",
                    min_raters: int = MIN_RATERS) -> tuple[dict[str, list[Item]], LoadReport]:
    """Items by official split, with the unsplit high-disagreement items in ``unsplit``."""
    report = LoadReport(ambiguous_as=ambiguous_as)
    items = items_from_ratings(iter_ratings(cache_dir), official_split(cache_dir),
                               ambiguous_as=ambiguous_as, min_raters=min_raters,
                               report=report)
    items = dedupe_by_text(items, report=report)
    by_split: dict[str, list[Item]] = {name: [] for name in ROUNDS.values()}
    for item in items:
        by_split[item.round].append(item)
    return by_split, report


# ----------------------------------------------------------------------------- splits

#: Held out of the training-side pool for the ambiguity-stratified evaluation. Larger than
#: DynaSent's 6 percent because the pool is less than half the size, and the held-out set
#: needs a few hundred no-majority items to be read by band.
AMBIG_FRACTION = 0.10
AMBIG_SEED = 20260923
#: Protected in this order when two evaluation sets share an item: the official test
#: split, the official dev split that selects checkpoints, then the derived set.
EVAL_PRIORITY = ("go_test", "go_dev", "ambig_eval")


def build_splits(by_split: Mapping[str, Sequence[Item]], *,
                 ambig_fraction: float = AMBIG_FRACTION, seed: int = AMBIG_SEED,
                 check_near: bool = True, near_threshold: float = 0.85):
    """Training pool, ambiguity-stratified held-out set and the official dev and test.

    The pool is the official training split plus the items no official split contains.
    The held-out set is drawn from it by seeded hash within round and agreement band, the
    same rule as on DynaSent, and every evaluation set is then audited against training
    and against each other; contaminated evaluation items are dropped, never training
    items, so the arms of a build all learn from the same rows.
    """
    from .integrity import assert_no_resampling, audit_splits, stratified_holdout
    from .splits import SplitBundle, drop_leaking_items

    pool = [*by_split["go_train"], *by_split["go_unsplit"]]
    held_out = stratified_holdout([i.text_id for i in pool],
                                  [f"{i.round}:{i.agreement_band}" for i in pool],
                                  ambig_fraction, seed)
    train = [i for i in pool if i.text_id not in held_out]
    ambig_eval = [i for i in pool if i.text_id in held_out]
    official = {"go_test": list(by_split["go_test"]), "go_dev": list(by_split["go_dev"])}
    for name, items in [("train", train), ("ambig_eval", ambig_eval), *official.items()]:
        assert_no_resampling([(i.text_id, i.text) for i in items], name)

    eval_names = list(EVAL_PRIORITY)
    targets = {"train": [(i.text_id, i.text) for i in train],
               "ambig_eval": [(i.text_id, i.text) for i in ambig_eval],
               **{k: [(i.text_id, i.text) for i in v] for k, v in official.items()}}
    directions = [("train", n) for n in eval_names]
    directions += [(a, b) for k, a in enumerate(eval_names) for b in eval_names[k + 1:]]
    integrity = audit_splits(targets, near_threshold=near_threshold, check_near=check_near,
                             raise_on_leak=False, directions=directions)
    bundle = SplitBundle(train, ambig_eval, official, integrity, {
        "source": "GoEmotions (Demszky et al., 2020), raw ratings and official split ids",
        "pool": "official train split plus items in no official split",
        "ambig_fraction": ambig_fraction, "holdout_seed": seed,
        "selection": "deterministic blake2b hash rank within round x agreement band",
        "near_duplicate_threshold": near_threshold if check_near else None,
    })
    if not integrity.clean:
        bundle = drop_leaking_items(bundle, near_threshold=near_threshold,
                                    check_near=check_near, priority=EVAL_PRIORITY)
    return bundle
