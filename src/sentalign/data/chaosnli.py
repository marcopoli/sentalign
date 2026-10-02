"""NLI sources: SNLI/MNLI for training targets, ChaosNLI for a dense evaluation target.

This exists to answer one question the sentiment study cannot. DynaSent gives five
annotators per item, so ``p_human`` is quantised to multiples of 0.2 and a JSD against it
is dominated by sampling noise in the *target*. That is the most likely reason a single
temperature fitted to a hard-label model matches every distributional method on the
sentiment data: with a five-sample target there is little per-item shape to learn beyond
an entropy level, and one scalar sets the entropy level.

ChaosNLI re-annotates SNLI and MNLI items with **100** annotators each, so the target has
real, measurable shape. The design is therefore:

* **train** on SNLI/MNLI items whose five ``annotator_labels`` give a coarse target, which
  keeps the training signal comparable to the sentiment study;
* **evaluate** on ChaosNLI, where the target is precise.

Any item that appears in ChaosNLI is removed from the training pool, so the dense
evaluation set is never trained on.

Formats are validated rather than trusted. Both releases are static files whose schemas
are documented, but a silently changed field would corrupt every distribution in the
study without raising, so each parser asserts the shape it depends on.
"""

from __future__ import annotations

import hashlib
import io
import json
import ssl
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterator, Sequence

from ..labels import CONTRADICTION, ENTAILMENT, NEUTRAL, NO_MAJORITY
from .dynasent import Item

#: Label surfaces used by SNLI/MNLI ``annotator_labels`` and ``gold_label``.
NLI_OPTIONS: tuple[str, ...] = (ENTAILMENT, NEUTRAL, CONTRADICTION)

#: ChaosNLI abbreviates the same three labels in ``label_counter``.
CHAOS_ABBREV = {"e": ENTAILMENT, "n": NEUTRAL, "c": CONTRADICTION}

#: Annotators per item in each source. SNLI/MNLI validation files carry five; ChaosNLI
#: carries one hundred. Both are asserted at load time.
N_SOURCE_ANNOTATORS = 5
N_CHAOS_ANNOTATORS = 100

#: Release archives. URLs are overridable because a moved file should be a configuration
#: problem, not a code change.
SOURCES: dict[str, dict] = {
    "snli": {
        "url": "https://nlp.stanford.edu/projects/snli/snli_1.0.zip",
        "members": ("snli_1.0/snli_1.0_dev.jsonl", "snli_1.0/snli_1.0_test.jsonl"),
    },
    "mnli": {
        "url": "https://cims.nyu.edu/~sbowman/multinli/multinli_1.0.zip",
        "members": ("multinli_1.0/multinli_1.0_dev_matched.jsonl",
                    "multinli_1.0/multinli_1.0_dev_mismatched.jsonl"),
    },
    "chaosnli": {
        "url": "https://www.dropbox.com/s/h4j7dqszmpt2679/chaosNLI_v1.0.zip?dl=1",
        # The archive extracts to chaosNLI_v1.0/, not data/chaosNLI_v1.0/;
        # the repo's download_data.sh unzips it *into* its data directory.
        "members": ("chaosNLI_v1.0/chaosNLI_snli.jsonl",
                    "chaosNLI_v1.0/chaosNLI_mnli_m.jsonl"),
    },
}


def _ssl_context() -> ssl.SSLContext | None:
    try:
        import certifi
    except ImportError:
        return None
    return ssl.create_default_context(cafile=certifi.where())


def download(name: str, cache_dir: Path, url: str | None = None) -> Path:
    """Fetch one release archive once and record its digest beside it.

    A file already present is used as-is, so an archive fetched by hand (which is often
    the only way past a file host that serves an interstitial page to scripts) is picked
    up without touching the code. The bytes are checked to actually be a zip: Dropbox and
    friends answer with HTML on failure, and a 9 KB "download page" saved as ``.zip``
    otherwise surfaces hundreds of lines later as an unreadable archive.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / f"{name}.zip"
    if not dest.exists():
        target = url or SOURCES[name]["url"]
        request = urllib.request.Request(target, headers={"User-Agent": "sentalign/1.0"})
        with urllib.request.urlopen(request, context=_ssl_context()) as src:
            dest.write_bytes(src.read())
    head = dest.read_bytes()[:4]
    if not head.startswith(b"PK"):
        size = dest.stat().st_size
        dest.rename(dest.with_suffix(".zip.rejected"))
        raise ValueError(
            f"{name}: downloaded {size:,} bytes that are not a zip (starts with "
            f"{head!r}); the host most likely returned an HTML page. Fetch it by hand "
            f"and save it as {dest}, then re-run:\n"
            f"    wget -O {dest} '{url or SOURCES[name]['url']}'")
    (cache_dir / f"{name}.zip.sha256").write_text(
        hashlib.sha256(dest.read_bytes()).hexdigest() + "\n")
    return dest


def iter_jsonl(archive: Path, member: str) -> Iterator[dict]:
    with zipfile.ZipFile(archive) as zf:
        names = set(zf.namelist())
        if member not in names:
            candidates = sorted(n for n in names if n.endswith(".jsonl"))
            raise KeyError(
                f"{archive.name} has no member {member!r}; .jsonl members present: "
                f"{candidates[:12]}")
        with zf.open(member) as fh:
            for line in io.TextIOWrapper(fh, encoding="utf-8"):
                line = line.strip()
                if line:
                    yield json.loads(line)


def format_text(premise: str, hypothesis: str) -> str:
    """The two NLI fields flattened into the single ``Item.text`` the pipeline expects.

    The label space's template supplies the instruction and the ``Answer:`` anchor, so
    this carries only the premise/hypothesis block.
    """
    return f"Premise: {premise.strip()}\nHypothesis: {hypothesis.strip()}"


def _gold_from_votes(votes: dict[str, int]) -> str:
    """Majority label, or ``no-majority`` when no option holds more than half the votes.

    This is DynaSent's rule generalised: three of five is the smallest strict majority
    there, and ``> 0.5`` reproduces it exactly while remaining meaningful at 100 votes.
    """
    total = sum(votes.values())
    if not total:
        return NO_MAJORITY
    top = max(votes, key=lambda k: votes[k])
    return top if votes[top] / total > 0.5 else NO_MAJORITY


def _band(votes: dict[str, int], gold: str) -> str:
    """Coarse agreement stratum, so 100-annotator items do not form singleton groups."""
    if gold == NO_MAJORITY:
        return "no-majority"
    total = sum(votes.values())
    frac = max(votes.values()) / total
    for edge, name in ((0.6, "weak"), (0.8, "moderate"), (1.01, "strong")):
        if frac < edge:
            return name
    return "strong"


def source_items(archive: Path, member: str, source: str) -> list[Item]:
    """SNLI/MNLI validation items, whose five ``annotator_labels`` give a coarse target."""
    items: list[Item] = []
    for record in iter_jsonl(archive, member):
        labels = record.get("annotator_labels")
        if labels is None:
            raise KeyError(
                f"{member}: record has no 'annotator_labels'; this loader needs the "
                f"original release files, not a simplified redistribution")
        votes = {opt: 0 for opt in NLI_OPTIONS}
        for label in labels:
            if label in votes:
                votes[label] += 1
        if sum(votes.values()) == 0:
            continue                      # every annotator marked '-' (no consensus tag)
        pair_id = record.get("pairID") or record.get("pair_id")
        if not pair_id:
            raise KeyError(f"{member}: record has no 'pairID'")
        gold = _gold_from_votes(votes)
        items.append(Item(
            text_id=str(pair_id),
            text=format_text(record["sentence1"], record["sentence2"]),
            votes=votes,
            gold=gold,
            round=source,
            source=source,
            meta={"band": _band(votes, gold), "n_annotators": sum(votes.values())},
        ))
    return items


def chaos_items(archive: Path, member: str, source: str) -> list[Item]:
    """ChaosNLI items, whose 100 annotators give a precisely estimated target."""
    items: list[Item] = []
    for record in iter_jsonl(archive, member):
        counter = record.get("label_counter")
        example = record.get("example")
        if counter is None or example is None:
            raise KeyError(
                f"{member}: record lacks 'label_counter'/'example'; got keys "
                f"{sorted(record)[:12]}")
        votes = {opt: 0 for opt in NLI_OPTIONS}
        for key, count in counter.items():
            label = CHAOS_ABBREV.get(key, key)
            if label in votes:
                votes[label] += int(count)
        total = sum(votes.values())
        if total != N_CHAOS_ANNOTATORS:
            raise ValueError(
                f"{member}: {record.get('uid')} has {total} annotations, expected "
                f"{N_CHAOS_ANNOTATORS}; the release format has changed")
        gold = _gold_from_votes(votes)
        items.append(Item(
            text_id=str(record["uid"]),
            text=format_text(example["premise"], example["hypothesis"]),
            votes=votes,
            gold=gold,
            round=source,
            source=source,
            meta={"band": _band(votes, gold), "n_annotators": total,
                  "entropy_reported": record.get("entropy")},
        ))
    return items


#: Scriptable mirror of the ChaosNLI MNLI portion. The upstream archive lives behind a
#: Dropbox share that now answers automated requests with an HTML interstitial, so the
#: documented URL cannot be fetched from a build script. This mirror carries the same
#: ``label_counter`` (100 annotations) and additionally keeps ``old_labels``, the five
#: original MNLI annotations for the *same* items. That pairing is worth more here than
#: the missing SNLI portion: it gives a coarse and a dense target over one set of items,
#: which is exactly the comparison this study exists to make.
HF_CHAOS_MNLI = "metaeval/chaos-mnli-ambiguity"


def chaos_items_from_hf(dataset: str = HF_CHAOS_MNLI,
                        source: str = "chaos_mnli") -> tuple[list[Item], list[Item]]:
    """Return ``(dense, coarse)`` item lists over the same MNLI examples.

    ``dense`` uses the 100 ChaosNLI annotations; ``coarse`` uses the five original MNLI
    ones. Both carry the same ``text_id``, so a caller that trains on one and evaluates
    on the other must split by id first.
    """
    from datasets import load_dataset

    rows = load_dataset(dataset, split="train")
    dense: list[Item] = []
    coarse: list[Item] = []
    for row in rows:
        counter = row["label_counter"]
        votes = {opt: 0 for opt in NLI_OPTIONS}
        for key, count in dict(counter).items():
            label = CHAOS_ABBREV.get(key, key)
            if label in votes:
                # A label nobody chose is absent from the original JSON object, and the
                # mirror's struct schema fills that hole with null rather than 0. Coercing
                # it to zero is the whole difference between "no annotator picked this"
                # and a crash, and silently dropping it would renormalise the target.
                votes[label] += int(count) if count is not None else 0
        total = sum(votes.values())
        if total != N_CHAOS_ANNOTATORS:
            raise ValueError(
                f"{dataset}: {row.get('uid')} has {total} annotations, expected "
                f"{N_CHAOS_ANNOTATORS}")
        text = format_text(row["premise"], row["hypothesis"])
        uid = str(row["uid"])
        gold = _gold_from_votes(votes)
        dense.append(Item(text_id=uid, text=text, votes=votes, gold=gold,
                          round=source, source=source,
                          meta={"band": _band(votes, gold), "n_annotators": total,
                                "target": "dense"}))

        old = {opt: 0 for opt in NLI_OPTIONS}
        for label in row.get("old_labels") or []:
            if label in old:
                old[label] += 1
        if sum(old.values()):
            old_gold = _gold_from_votes(old)
            coarse.append(Item(text_id=uid, text=text, votes=old, gold=old_gold,
                               round=f"{source}_coarse", source=f"{source}_coarse",
                               meta={"band": _band(old, old_gold),
                                     "n_annotators": sum(old.values()),
                                     "target": "coarse"}))
    return dense, coarse


def load_nli(cache_dir: Path, chaos_only: bool = False,
             train_target: str = "dense") -> dict[str, list[Item]]:
    """Assemble the NLI study, adapting to whichever sources are reachable.

    The evaluation is always the *paired* one: the same items carrying a dense
    100-annotator target and the coarse 5-annotator target they originally had. Scoring
    one set of predictions against both is what separates "the model learned the
    distribution" from "the five-sample target was too noisy to tell", which is the
    question the sentiment study could not answer.

    Training volume comes from whichever source is available:

    * SNLI/MNLI validation items, five annotators, with every evaluation id held out.
      Preferred, because it is the same coarse regime the sentiment study trained in and
      it is an order of magnitude larger.
    * Failing that (``chaos_only``, or the archives being unreachable), half of the
      paired items, split by id. Smaller, and the targets are dense, which is a different
      regime; callers record which one produced a build.

    Returns ``train``, ``chaos_dense``, ``chaos_coarse`` and ``regime``.
    """
    dense, coarse = chaos_items_from_hf()
    by_id = {it.text_id: it for it in coarse}

    def paired(items: Sequence[Item]) -> tuple[list[Item], list[Item]]:
        kept = [it for it in items if it.text_id in by_id]
        return kept, [by_id[it.text_id] for it in kept]

    if not chaos_only:
        try:
            held_out = {it.text_id for it in dense}
            train: list[Item] = []
            for name in ("snli", "mnli"):
                archive = download(name, cache_dir)
                for member in SOURCES[name]["members"]:
                    train.extend(source_items(archive, member, name))
            kept = [it for it in train if it.text_id not in held_out]
            if kept:
                d, c = paired(dense)
                return {"train": kept, "chaos_dense": d, "chaos_coarse": c,
                        "regime": "coarse_train_dense_eval"}
        except Exception as exc:                      # noqa: BLE001 - reported, not hidden
            print(f"  SNLI/MNLI unavailable ({type(exc).__name__}: {str(exc)[:120]}); "
                  f"falling back to a split of the paired items")

    ordered = sorted(dense, key=lambda i: i.text_id)
    cut = len(ordered) // 2
    train_items = ordered[:cut]
    eval_items = ordered[cut:]
    if train_target == "coarse":
        # The same training items carrying their original five annotations. Paired with
        # the dense build this isolates annotation density from item count, so a null
        # cannot be blamed on one arm simply having seen more data.
        train_items = [by_id[i.text_id] for i in train_items if i.text_id in by_id]
    d, c = paired(eval_items)
    return {"train": train_items, "chaos_dense": d, "chaos_coarse": c,
            "regime": f"{train_target}_train_dense_eval"}
