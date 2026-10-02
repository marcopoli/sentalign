"""Zero-shot transfer and counterfactual-robustness evaluation sets.

None of these are trained on. They exist to answer the question the v1 cross-domain
design could not: v1 trained on binary IMDB and evaluated on three-way Sentiment140 and
back, so a model that had never emitted "neutral" was capped near 67% for structural
reasons and the cells measured label-space mismatch rather than generalisation
(AUDIT.md S2-7). Every set here is mapped into the *same* label space the model was
trained on, and where a set is coarser the mapping is stated and its cost reported.

    TweetEval-sentiment      register shift (SemEval-2017 Task 4A), 12,284 test items
    Financial PhraseBank     domain shift, plus its own annotator-agreement gradient
    SST-dev-validated        a classic benchmark's items with 5 fresh annotators
    IMDb-CAD                 human minimal edits that flip sentiment
    DynaSent R2 prompt pairs 16,899 human rewrites of a real Yelp sentence

Financial PhraseBank is worth singling out: it ships four subsets defined by annotator
agreement (>=50%, >=66%, >=75%, 100%). That gives an independent replication of H3 on a
different domain, different annotator pool, and different task framing: which is a much
stronger test than repeating the analysis on more DynaSent.
"""

from __future__ import annotations

import csv
import io
import json
import ssl
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from ..labels import NEGATIVE, NEUTRAL, POSITIVE, LabelSpace

CAD_BASE = ("https://raw.githubusercontent.com/acmi-lab/"
            "counterfactually-augmented-data/master/sentiment")

FINANCIAL_SUBSETS = ("sentences_50agree", "sentences_66agree",
                     "sentences_75agree", "sentences_allagree")


@dataclass(frozen=True)
class TransferItem:
    text_id: str
    text: str
    label: str
    dataset: str
    group: str = "default"
    meta: dict | None = None


def _ssl_context() -> ssl.SSLContext | None:
    try:
        import certifi
    except ImportError:
        return None
    return ssl.create_default_context(cafile=certifi.where())


def _get(url: str) -> bytes:
    with urllib.request.urlopen(url, context=_ssl_context()) as fh:
        return fh.read()


def load_tweeteval(space: LabelSpace, split: str = "test") -> list[TransferItem]:
    """TweetEval sentiment (Barbieri et al., 2020). SemEval-2017 Task 4A.

    Labels are 0=negative, 1=neutral, 2=positive. There is no ``mixed`` class, so in a
    four-way space this set can only exercise three of the four labels; that is reported
    rather than papered over by remapping.
    """
    from datasets import load_dataset

    mapping = {0: NEGATIVE, 1: NEUTRAL, 2: POSITIVE}
    ds = load_dataset("cardiffnlp/tweet_eval", "sentiment", split=split)
    return [
        TransferItem(text_id=f"tweeteval-{split}-{i}", text=row["text"],
                     label=space.normalise(mapping[row["label"]]), dataset="tweeteval")
        for i, row in enumerate(ds) if space.contains(mapping[row["label"]])
    ]


def load_financial_phrasebank(space: LabelSpace,
                              subsets: Sequence[str] = FINANCIAL_SUBSETS
                              ) -> list[TransferItem]:
    """Financial PhraseBank (Malo et al., 2014), grouped by annotator agreement.

    ``group`` carries the agreement subset, so the same worst-group and stratified
    analyses used on DynaSent apply directly and H3 gets an out-of-domain replication.
    """
    from datasets import load_dataset

    mapping = {0: NEGATIVE, 1: NEUTRAL, 2: POSITIVE}
    items: list[TransferItem] = []
    for subset in subsets:
        ds = load_dataset("takala/financial_phrasebank", subset,
                          split="train", trust_remote_code=True)
        for i, row in enumerate(ds):
            label = mapping[row["label"]]
            if not space.contains(label):
                continue
            items.append(TransferItem(
                text_id=f"fpb-{subset}-{i}", text=row["sentence"],
                label=space.normalise(label), dataset="financial_phrasebank",
                group=subset))
    return items


def load_imdb_cad(space: LabelSpace, cache_dir: Path) -> list[dict]:
    """IMDb counterfactually-augmented data (Kaushik, Hovy & Lipton, 2020).

    Returns ``(original, edited)`` pairs where a human made the minimal edit that flips
    the sentiment. Sensitivity to those edits separates a model that reads sentiment from
    one that reads topic: the distinction a saturated IMDB accuracy cannot make.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    pairs: list[dict] = []
    for split in ("train", "dev", "test"):
        orig_path = cache_dir / f"cad_orig_{split}.tsv"
        new_path = cache_dir / f"cad_new_{split}.tsv"
        for path, folder in ((orig_path, "orig"), (new_path, "new")):
            if not path.exists():
                path.write_bytes(_get(f"{CAD_BASE}/{folder}/{split}.tsv"))

        def read(path: Path) -> list[dict]:
            with path.open(encoding="utf-8") as fh:
                return list(csv.DictReader(fh, delimiter="\t"))

        originals, edited = read(orig_path), read(new_path)
        if len(originals) != len(edited):
            raise ValueError(f"CAD {split}: {len(originals)} originals vs {len(edited)} edits")
        for i, (o, e) in enumerate(zip(originals, edited, strict=True)):
            o_label = o["Sentiment"].strip().lower()
            e_label = e["Sentiment"].strip().lower()
            if not (space.contains(o_label) and space.contains(e_label)):
                continue
            pairs.append({
                "pair_id": f"cad-{split}-{i}",
                "original_text": o["Text"], "edited_text": e["Text"],
                "original_label": space.normalise(o_label),
                "edited_label": space.normalise(e_label),
                "should_flip": o_label != e_label,
                "split": split,
            })
    return pairs


def load_dynasent_rewrites(cache_dir: Path) -> list[dict]:
    """DynaSent round-2 rewrites paired with the Yelp sentence they were written from."""
    from .dynasent import counterfactual_pairs

    return [
        {"pair_id": p["text_id"], "original_text": p["original"],
         "edited_text": p["rewritten"], "edited_label": p["rewritten_gold"],
         "original_review_rating": p["original_review_rating"],
         "should_flip": None, "split": "r2_train"}
        for p in counterfactual_pairs(cache_dir)
    ]


def save_transfer_set(items: Sequence[TransferItem], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for item in items:
            fh.write(json.dumps({
                "text_id": item.text_id, "text": item.text, "label": item.label,
                "dataset": item.dataset, "group": item.group,
            }, ensure_ascii=False) + "\n")
    return len(items)
