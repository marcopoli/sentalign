"""DynaSent v1.1 loader that preserves the per-item annotator distribution.

DynaSent is the primary corpus because it is the only widely used sentiment resource
that releases **all five crowdworker judgements for every item**, which is the input the
whole design depends on (RESEARCH_DESIGN.md §2). The Hugging Face mirror
``dynabench/dynasent`` is a loading script and no longer works under ``datasets>=3``,
so we read the official release archive directly.

Verified against ``dynasent-v1.1.zip`` (SHA recorded in ``ARCHIVE``):

    split                       n        <=3/5 agreement
    round01 train           94,459               49.6%
    round01 dev/test    3,600 / 3,600             0.0%   (filtered to >=4/5)
    round02 train           18,535               40.2%
    round02 dev/test      720 / 720               0.0%   (filtered to >=4/5)
    sst-dev-validated        1,101               26.8%

The official dev/test sets are filtered to >=4/5 agreement, so they cannot exercise H3.
That is why ``sentalign.data.splits`` carves an ambiguity-stratified held-out set out of
the *training* rounds instead.
"""

from __future__ import annotations

import hashlib
import io
import json
import ssl
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

from ..labels import ANNOTATOR_OPTIONS, NO_MAJORITY, LabelSpace

ARCHIVE_URL = "https://raw.githubusercontent.com/cgpotts/dynasent/main/dynasent-v1.1.zip"
ARCHIVE_NAME = "dynasent-v1.1.zip"
N_ANNOTATORS = 5
MAJORITY_THRESHOLD = 3  # DynaSent's own rule: >=3/5 for a gold label to exist

FILES: dict[str, str] = {
    "r1_train": "dynasent-v1.1/dynasent-v1.1-round01-yelp-train.jsonl",
    "r1_dev": "dynasent-v1.1/dynasent-v1.1-round01-yelp-dev.jsonl",
    "r1_test": "dynasent-v1.1/dynasent-v1.1-round01-yelp-test.jsonl",
    "r2_train": "dynasent-v1.1/dynasent-v1.1-round02-dynabench-train.jsonl",
    "r2_dev": "dynasent-v1.1/dynasent-v1.1-round02-dynabench-dev.jsonl",
    "r2_test": "dynasent-v1.1/dynasent-v1.1-round02-dynabench-test.jsonl",
    "sst_dev_validated": "dynasent-v1.1/sst-dev-validated.jsonl",
}


@dataclass(frozen=True)
class Item:
    """One sentence with its full annotator distribution.

    ``votes`` is the raw count per annotator option; ``p_human`` is its normalisation.
    ``gold`` is the majority option or ``no-majority``. Keeping votes rather than only
    the normalised distribution matters: the preference builder needs integer counts to
    apply an exact minimum-margin threshold without floating-point ties.
    """

    text_id: str
    text: str
    votes: dict[str, int]
    gold: str
    round: str
    source: str
    annotators: dict[str, tuple[str, ...]] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    @property
    def n_votes(self) -> int:
        return sum(self.votes.values())

    @property
    def p_human(self) -> dict[str, float]:
        n = self.n_votes
        return {k: v / n for k, v in self.votes.items()} if n else {}

    @property
    def majority_votes(self) -> int:
        return max(self.votes.values()) if self.votes else 0

    @property
    def agreement(self) -> float:
        """Fraction of annotators choosing the modal option, in [1/|options|, 1]."""
        return self.majority_votes / self.n_votes if self.n_votes else 0.0

    @property
    def agreement_band(self) -> str:
        """Stratum used for group-robust training and for the H3 analysis.

        ``meta["band"]`` overrides the derived value. Nothing in DynaSent sets it, so
        those items are unchanged; it exists for densely annotated sources, where
        "43of100" would spread the items over a hundred singleton strata instead of the
        handful the group-robust arms and the H3 analysis expect.
        """
        if self.meta.get("band"):
            return str(self.meta["band"])
        if self.gold == NO_MAJORITY:
            return "no-majority"
        return f"{self.majority_votes}of{self.n_votes}"

    @property
    def entropy(self) -> float:
        """Shannon entropy of the human distribution, in nats. 0 = unanimous."""
        import math

        return -sum(p * math.log(p) for p in self.p_human.values() if p > 0)

    def distribution_vector(self, space: LabelSpace) -> list[float]:
        """``p_human`` projected onto ``space``, renormalised over in-space mass.

        Returns a uniform vector when no annotator chose an in-space option, which only
        happens for a collapsed space where every vote fell outside it.
        """
        raw = [0.0] * space.size
        for option, count in self.votes.items():
            try:
                raw[space.index(option)] += count
            except KeyError:
                continue  # option not represented in this space
        total = sum(raw)
        if total == 0:
            return [1.0 / space.size] * space.size
        return [v / total for v in raw]

    def to_dict(self) -> dict:
        return {
            "text_id": self.text_id, "text": self.text, "votes": self.votes,
            "gold": self.gold, "round": self.round, "source": self.source,
            "agreement": self.agreement, "agreement_band": self.agreement_band,
            "entropy": self.entropy, "meta": self.meta,
        }


def _ssl_context() -> ssl.SSLContext | None:
    try:
        import certifi
    except ImportError:
        return None
    return ssl.create_default_context(cafile=certifi.where())


def download_archive(cache_dir: Path) -> Path:
    """Fetch the official release archive once and record its digest."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / ARCHIVE_NAME
    if not dest.exists():
        with urllib.request.urlopen(ARCHIVE_URL, context=_ssl_context()) as src:
            payload = src.read()
        dest.write_bytes(payload)
    digest = hashlib.sha256(dest.read_bytes()).hexdigest()
    (cache_dir / f"{ARCHIVE_NAME}.sha256").write_text(digest + "\n")
    return dest


def _iter_jsonl(archive: Path, member: str) -> Iterator[dict]:
    with zipfile.ZipFile(archive) as zf:
        with zf.open(member) as fh:
            for line in io.TextIOWrapper(fh, encoding="utf-8"):
                line = line.strip()
                if line:
                    yield json.loads(line)


def _to_item(record: dict, split: str) -> Item:
    dist = record["label_distribution"]
    votes = {opt: len(dist.get(opt, [])) for opt in ANNOTATOR_OPTIONS}
    annotators = {opt: tuple(dist.get(opt, [])) for opt in ANNOTATOR_OPTIONS}

    gold = record.get("gold_label")
    top = max(votes.values()) if votes else 0
    if gold is None or top < MAJORITY_THRESHOLD:
        # DynaSent writes null (round data) or "No Majority" for these; unify the name
        # so downstream code never has to test for three spellings of the same state.
        gold = NO_MAJORITY

    meta = {k: record[k] for k in
            ("review_rating", "model_0_label", "model_1_label", "sst_label",
             "sentence_author", "has_prompt", "review_id")
            if k in record}
    if record.get("has_prompt") and "prompt_data" in record:
        meta["prompt_sentence"] = record["prompt_data"].get("prompt_sentence")
        meta["prompt_review_rating"] = record["prompt_data"].get("review_rating")
    for key in ("model_0_probs", "model_1_probs"):
        if key in record:
            meta[key] = record[key]

    return Item(
        text_id=record["text_id"],
        text=record["sentence"],
        votes=votes,
        gold=gold,
        round=split.split("_")[0],
        source=split,
        annotators=annotators,
        meta=meta,
    )


def load_split(split: str, cache_dir: Path) -> list[Item]:
    """Load one named split (see ``FILES``) as ``Item`` objects."""
    if split not in FILES:
        raise KeyError(f"unknown split {split!r}; choose from {sorted(FILES)}")
    archive = download_archive(cache_dir)
    items = [_to_item(rec, split) for rec in _iter_jsonl(archive, FILES[split])]
    bad = [it for it in items if it.n_votes != N_ANNOTATORS]
    if bad:
        raise ValueError(
            f"{split}: {len(bad)} items do not have exactly {N_ANNOTATORS} annotators "
            f"(first: {bad[0].text_id}); the release format has changed"
        )
    return items


def load_all(cache_dir: Path, splits: Sequence[str] | None = None) -> dict[str, list[Item]]:
    return {s: load_split(s, cache_dir) for s in (splits or FILES)}


def counterfactual_pairs(cache_dir: Path) -> list[dict]:
    """Round-2 items paired with the Yelp sentence a worker was asked to rewrite.

    16,911 of the 18,535 round-2 training items are human rewrites of a real sentence,
    carrying the original review's star rating. That gives a naturally occurring
    perturbation set for the counterfactual-sensitivity analysis, complementing the
    minimal edits of IMDb-CAD (Kaushik et al., 2020).
    """
    pairs = []
    for it in load_split("r2_train", cache_dir):
        original = it.meta.get("prompt_sentence")
        if not original or original.strip() == it.text.strip():
            continue
        pairs.append({
            "text_id": it.text_id,
            "original": original,
            "rewritten": it.text,
            "rewritten_gold": it.gold,
            "original_review_rating": it.meta.get("prompt_review_rating"),
            "agreement_band": it.agreement_band,
        })
    return pairs
