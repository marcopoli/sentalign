"""Split construction, including the ambiguity-stratified held-out set.

DynaSent's official dev/test sets are filtered to >=4/5 annotator agreement and balanced
three ways. That is the right call for a leaderboard, but it means the official
evaluation contains **no ambiguous items at all**, verified: the r1/r2 test bands are
{4of5, 5of5} only. H3 asks whether the benefit of preference optimization scales with
human label variation, and the official test sets cannot answer it.

So we carve ``dynasent-ambig`` out of the *training* rounds, stratified by agreement
band, hold it out from every training arm, and release the item ids. It is a new split
of an existing corpus, and the paper says so.

Order of operations is fixed and enforced: **split, then audit, then balance**. v1
balanced first by resampling with replacement and never audited, which put 100% of its
neutral test class into training (AUDIT.md S1-1).
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

from ..labels import NO_MAJORITY, LabelSpace
from .dynasent import Item, load_split
from .integrity import (IntegrityReport, _hash64, assert_no_resampling,
                        audit_splits, stratified_holdout)

TRAIN_ROUNDS = ("r1_train", "r2_train")
OFFICIAL_EVAL = ("r1_dev", "r1_test", "r2_dev", "r2_test", "sst_dev_validated")

#: Fraction of each training round held out for the ambiguity-stratified evaluation.
AMBIG_FRACTION = 0.06
AMBIG_SEED = 20260819


@dataclass
class SplitBundle:
    """Everything the experiment trains and evaluates on, plus the audit that cleared it."""

    train: list[Item]
    ambig_eval: list[Item]
    official: dict[str, list[Item]]
    integrity: IntegrityReport
    provenance: dict

    def sizes(self) -> dict[str, int]:
        out = {"train": len(self.train), "ambig_eval": len(self.ambig_eval)}
        out.update({k: len(v) for k, v in self.official.items()})
        return out

    def band_table(self) -> dict[str, Counter]:
        table = {"train": Counter(i.agreement_band for i in self.train),
                 "ambig_eval": Counter(i.agreement_band for i in self.ambig_eval)}
        for name, items in self.official.items():
            table[name] = Counter(i.agreement_band for i in items)
        return table

    def save_manifest(self, path: Path) -> None:
        """Write the item ids so the derived split is reproducible without our code."""
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "provenance": self.provenance,
            "sizes": self.sizes(),
            "band_table": {k: dict(v) for k, v in self.band_table().items()},
            "ambig_eval_ids": sorted(i.text_id for i in self.ambig_eval),
            "integrity": {
                "clean": self.integrity.clean,
                "duplicate_rates": self.integrity.duplicate_rates,
                "overlaps": [
                    {"from": o.split_a, "to": o.split_b,
                     "exact": len(o.exact), "near": len(o.near)}
                    for o in self.integrity.overlaps
                ],
            },
        }
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def build_splits(
    cache_dir: Path,
    *,
    ambig_fraction: float = AMBIG_FRACTION,
    seed: int = AMBIG_SEED,
    check_near_duplicates: bool = True,
    near_threshold: float = 0.8,
) -> SplitBundle:
    """Build training / ambiguity-eval / official-eval splits and audit them.

    Near-duplicate checking matters here specifically: 16,899 round-2 items are human
    rewrites of a round-1 Yelp sentence, so a rewrite and its source can straddle the
    holdout boundary. Exact matching would miss it.
    """
    rounds = {name: load_split(name, cache_dir) for name in TRAIN_ROUNDS}
    pooled = [item for items in rounds.values() for item in items]

    keys = [i.text_id for i in pooled]
    strata = [f"{i.round}:{i.agreement_band}" for i in pooled]
    held_out = stratified_holdout(keys, strata, ambig_fraction, seed)

    train = [i for i in pooled if i.text_id not in held_out]
    ambig_eval = [i for i in pooled if i.text_id in held_out]

    official = {name: load_split(name, cache_dir) for name in OFFICIAL_EVAL}

    for name, items in [("train", train), ("ambig_eval", ambig_eval), *official.items()]:
        assert_no_resampling([(i.text_id, i.text) for i in items], name)

    audit_targets = {
        "train": [(i.text_id, i.text) for i in train],
        "ambig_eval": [(i.text_id, i.text) for i in ambig_eval],
    }
    for name, items in official.items():
        audit_targets[name] = [(i.text_id, i.text) for i in items]

    # Only the informative directions: "how much of each evaluation set was already
    # seen during training". The reverse direction answers nothing and costs as much.
    eval_names = ["ambig_eval", *official]
    directions = [("train", name) for name in eval_names]
    directions += [(a, b) for i, a in enumerate(eval_names) for b in eval_names[i + 1:]]

    integrity = audit_splits(
        audit_targets,
        near_threshold=near_threshold,
        check_near=check_near_duplicates,
        raise_on_leak=False,   # inspected below; we want the report either way
        directions=directions,
    )

    provenance = {
        "source": "DynaSent v1.1 (Potts et al., 2021), official release archive",
        "train_rounds": list(TRAIN_ROUNDS),
        "ambig_fraction": ambig_fraction,
        "holdout_seed": seed,
        "selection": "deterministic blake2b hash rank within round x agreement band",
        "near_duplicate_threshold": near_threshold if check_near_duplicates else None,
    }
    return SplitBundle(train, ambig_eval, official, integrity, provenance)


#: Evaluation sets are protected in this order when two of them share an item: the
#: official test sets first (they are what published numbers are compared against), then
#: the SST re-validation, then the dev sets, then our derived ambiguity split. Training
#: data is never modified: shrinking it would change what each arm learned and make runs
#: incomparable across builds.
EVAL_PRIORITY = ("r1_test", "r2_test", "sst_dev_validated", "r1_dev", "r2_dev", "ambig_eval")


def drop_leaking_items(bundle: SplitBundle, *, near_threshold: float = 0.85,
                       check_near: bool = True,
                       priority: Sequence[str] = EVAL_PRIORITY) -> SplitBundle:
    """Remove evaluation items that overlap training data or a higher-priority eval set.

    Two kinds of contamination are resolved:

    *   **train -> eval.** Items seen during training are dropped from evaluation.
    *   **eval -> eval.** DynaSent's own splits share items: the same generic Yelp
        sentence ("Great food!") was harvested independently into more than one round.
        Measured on v1.1: ~1.0% of round-1 dev and ~0.9% of round-1 test appear verbatim
        in the training rounds. Where two evaluation sets collide, the item is kept in
        the higher-priority one and dropped from the other, so no item is scored twice.

    ``priority`` names the evaluation sets in the order they are protected. It defaults to
    DynaSent's; another corpus passes its own, and a set it does not name is refused rather
    than silently left out of the audit, which would let its contamination through.
    """
    ambig = list(bundle.ambig_eval)
    official = {k: list(v) for k, v in bundle.official.items()}
    pools: dict[str, list] = {"ambig_eval": ambig, **official}

    def rebuild(pools: dict[str, list]) -> dict[str, list[tuple[str, str]]]:
        targets = {"train": [(i.text_id, i.text) for i in bundle.train]}
        targets.update({k: [(i.text_id, i.text) for i in v] for k, v in pools.items()})
        return targets

    unranked = sorted(set(pools) - set(priority))
    if unranked:
        raise ValueError(f"evaluation sets {unranked} have no place in the leakage "
                         f"priority {tuple(priority)}; they would escape the audit")
    eval_names = [n for n in priority if n in pools]
    directions = [("train", n) for n in eval_names]
    directions += [(a, b) for i, a in enumerate(eval_names) for b in eval_names[i + 1:]]

    report = audit_splits(rebuild(pools), near_threshold=near_threshold,
                          check_near=check_near, raise_on_leak=False,
                          directions=directions)

    dropped: dict[str, set[str]] = {name: set() for name in pools}
    for overlap in report.overlaps:
        victim = overlap.split_b          # directions are ordered source -> victim
        if victim not in dropped:
            continue
        dropped[victim].update(item_id for _, item_id in overlap.exact)
        dropped[victim].update(item_id for _, item_id, _ in overlap.near)

    total_dropped = sum(len(v) for v in dropped.values())
    if not total_dropped:
        return bundle

    pools = {name: [i for i in items if i.text_id not in dropped[name]]
             for name, items in pools.items()}
    ambig = pools.pop("ambig_eval")
    official = pools

    integrity = audit_splits(rebuild({"ambig_eval": ambig, **official}),
                             near_threshold=near_threshold, check_near=check_near,
                             raise_on_leak=True, directions=directions)

    provenance = dict(bundle.provenance)
    provenance["dropped_contaminated_eval_items"] = total_dropped
    provenance["dropped_by_split"] = {k: len(v) for k, v in dropped.items() if v}
    provenance["eval_priority"] = list(priority)
    return SplitBundle(bundle.train, ambig, official, integrity, provenance)


def stratified_subsample(
    items: Sequence[Item],
    n: int,
    *,
    seed: int = 101,
    strata: Callable[[Item], str] | None = None,
) -> list[Item]:
    """Take ``n`` items while preserving the composition the hypotheses depend on.

    The full training pool is 106,215 items. Two epochs over all of it is roughly four
    hours for the 1.2B model alone, which does not fit a 150 GPU-hour programme across
    three models, ten arms, and five seeds. Subsampling is therefore a stated compute
    decision, and a data-scaling ablation over {3k, 8k, 16k, 32k} reports whether the
    conclusions depend on it.

    Stratification is by agreement band crossed with gold label, because the agreement
    distribution is exactly what the label-variation hypotheses are about: a uniform
    random sample would preserve it only in expectation, and at n = 8,000 the
    no-majority and mixed cells are small enough for that to matter.

    Selection within a stratum is by seeded hash of the item id rather than by shuffling,
    so the sample is stable under reordering of the input and nested across sizes: the
    3,000-item sample is a subset of the 8,000-item sample, which makes the scaling
    ablation a clean nesting rather than four unrelated draws.
    """
    items = list(items)
    if n >= len(items):
        return items
    key = strata or (lambda it: f"{it.agreement_band}|{it.gold}")

    buckets: dict[str, list[Item]] = defaultdict(list)
    for item in items:
        buckets[key(item)].append(item)

    # Largest-remainder allocation, so small strata are not rounded out of existence.
    fraction = n / len(items)
    quotas: dict[str, int] = {}
    remainders: list[tuple[float, str]] = []
    for name, members in buckets.items():
        exact = len(members) * fraction
        quotas[name] = int(exact)
        remainders.append((exact - int(exact), name))
    shortfall = n - sum(quotas.values())
    for _, name in sorted(remainders, reverse=True)[:max(0, shortfall)]:
        quotas[name] += 1

    chosen: list[Item] = []
    for name, members in buckets.items():
        ranked = sorted(members, key=lambda it: _hash64(it.text_id, seed))
        chosen.extend(ranked[:quotas[name]])
    chosen.sort(key=lambda it: _hash64(it.text_id, seed))
    return chosen


def subsample_report(full: Sequence[Item], sample: Sequence[Item]) -> str:
    """Composition of the sample against the pool, for the data appendix."""
    def dist(items: Sequence[Item]) -> dict[str, float]:
        counts = Counter(f"{i.agreement_band}" for i in items)
        total = max(len(items), 1)
        return {k: v / total for k, v in counts.items()}

    a, b = dist(full), dist(sample)
    lines = [f"  {'band':<14}{'pool':>9}{'sample':>9}{'delta':>9}"]
    for band in sorted(set(a) | set(b)):
        lines.append(f"  {band:<14}{100 * a.get(band, 0):>8.2f}%"
                     f"{100 * b.get(band, 0):>8.2f}%"
                     f"{100 * (b.get(band, 0) - a.get(band, 0)):>+8.2f}%")
    return "\n".join(lines)


def class_weights(items: Sequence[Item], space: LabelSpace) -> dict[str, float]:
    """Inverse-frequency weights over gold labels.

    This is how class imbalance gets handled. DynaSent round 1 is 45,076 neutral against
    14,021 negative. Reweighting the loss leaves the data untouched; resampling rows to
    balance is what produced the v1 leak.
    """
    counts = Counter(space.normalise(i.gold) for i in items
                     if i.gold != NO_MAJORITY and space.contains(i.gold))
    total = sum(counts.values())
    if not total:
        return {y: 1.0 for y in space.labels}
    return {y: total / (space.size * counts[y]) if counts[y] else 0.0 for y in space.labels}


def stratify_report(items: Iterable[Item]) -> str:
    by_band: dict[str, Counter] = defaultdict(Counter)
    for item in items:
        by_band[item.agreement_band][item.gold] += 1
    lines = []
    for band in sorted(by_band):
        counts = by_band[band]
        total = sum(counts.values())
        detail = " ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        lines.append(f"  {band:<12} n={total:>7,}  {detail}")
    return "\n".join(lines)
