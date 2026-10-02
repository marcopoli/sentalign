"""Leakage auditing. Every dataset build must pass through ``audit_splits``.

This module exists because of AUDIT.md S1-1: the v1 pipeline oversampled 139 unique
tweets into 120,000 rows *before* splitting, putting 100% of its "held-out" neutral
class into training. Nothing in that pipeline could have noticed. So the rule here is
structural rather than advisory, ``audit_splits`` raises by default, and a build that
wants to tolerate overlap has to say so explicitly and record why.

Two kinds of overlap are checked:

*   **exact**: identical normalised text across splits. Catches resampling-with-
    replacement and naive concatenation.
*   **near**: high Jaccard similarity over character 5-grams, found with a MinHash /
    banded-LSH index. Catches boilerplate reviews, retweets, and the
    minimal-edit rewrites that DynaSent round 2 is full of: a round-2 rewrite of a
    round-1 sentence is a genuine leak if the two land on opposite sides of a split.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable, Mapping, Sequence

if TYPE_CHECKING:
    import numpy as np

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]")


class LeakageError(RuntimeError):
    """Raised when a dataset build would emit overlapping splits."""


def normalise_text(text: str) -> str:
    """Aggressive normalisation for duplicate detection only: never for modelling."""
    text = unicodedata.normalize("NFKC", text).lower()
    text = _PUNCT.sub(" ", text)
    return _WS.sub(" ", text).strip()


def _shingles(text: str, k: int = 5) -> set[str]:
    norm = normalise_text(text)
    if len(norm) < k:
        return {norm} if norm else set()
    return {norm[i:i + k] for i in range(len(norm) - k + 1)}


def _hash64(value: str, seed: int) -> int:
    digest = hashlib.blake2b(f"{seed}:{value}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big")


def _base_hashes(shingles: Iterable[str]) -> "np.ndarray":
    """One 64-bit hash per shingle, as a uint64 array."""
    import numpy as np

    return np.fromiter(
        (int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "big")
         for s in shingles),
        dtype=np.uint64,
    )


class MinHashIndex:
    """Banded-LSH index over character shingles.

    ``n_perm`` permutations in ``n_bands`` bands; two documents become candidates when
    any band matches. With the defaults (64 perms, 16 bands of 4) the detection
    probability at Jaccard 0.8 is ~0.99 and at 0.5 is ~0.20, which is the right shape:
    we want near-duplicates flagged and merely topically similar items ignored.
    """

    def __init__(self, n_perm: int = 64, n_bands: int = 16, shingle_k: int = 5,
                 seed: int = 0) -> None:
        if n_perm % n_bands:
            raise ValueError("n_perm must be divisible by n_bands")
        import numpy as np

        self.n_perm, self.n_bands = n_perm, n_bands
        self.rows_per_band = n_perm // n_bands
        self.shingle_k = shingle_k
        self._buckets: list[dict[bytes, list[str]]] = [defaultdict(list) for _ in range(n_bands)]
        self._signatures: dict[str, "np.ndarray"] = {}
        self._shingles: dict[str, set[str]] = {}
        # Affine permutation family h_i(x) = a_i * x + b_i over uint64 with wraparound.
        # Hashing each shingle once and permuting in numpy is what makes the audit
        # tractable: calling a cryptographic hash n_perm times per shingle is ~100x
        # slower and does not finish on a 100k-row split.
        rng = np.random.default_rng(seed)
        self._a = (rng.integers(0, 2**63 - 1, size=n_perm, dtype=np.int64)
                   .astype(np.uint64) | np.uint64(1))     # odd multipliers are bijective
        self._b = rng.integers(0, 2**63 - 1, size=n_perm, dtype=np.int64).astype(np.uint64)
        self._np = np

    def signature(self, text: str) -> "np.ndarray":
        np = self._np
        sh = _shingles(text, self.shingle_k)
        if not sh:
            return np.zeros(self.n_perm, dtype=np.uint64)
        base = _base_hashes(sh)
        with np.errstate(over="ignore"):   # wraparound is the point
            perm = base[:, None] * self._a[None, :] + self._b[None, :]
        return perm.min(axis=0)

    def add(self, key: str, text: str) -> None:
        sig = self.signature(text)
        self._signatures[key] = sig
        self._shingles[key] = _shingles(text, self.shingle_k)
        for b in range(self.n_bands):
            band = sig[b * self.rows_per_band:(b + 1) * self.rows_per_band].tobytes()
            self._buckets[b][band].append(key)

    def query(self, text: str, threshold: float = 0.8,
              signature: "np.ndarray | None" = None) -> list[tuple[str, float]]:
        """Keys whose true Jaccard with ``text`` is at least ``threshold``.

        Pass ``signature`` when the probe is itself indexed elsewhere; recomputing it is
        the single most expensive step in a cross-split audit.
        """
        sig = self.signature(text) if signature is None else signature
        candidates: set[str] = set()
        for b in range(self.n_bands):
            band = sig[b * self.rows_per_band:(b + 1) * self.rows_per_band].tobytes()
            candidates.update(self._buckets[b].get(band, ()))
        if not candidates:
            return []
        probe = _shingles(text, self.shingle_k)
        hits = []
        for key in candidates:
            other = self._shingles[key]
            union = len(probe | other)
            jac = len(probe & other) / union if union else 0.0
            if jac >= threshold:
                hits.append((key, jac))
        return sorted(hits, key=lambda kv: -kv[1])


@dataclass
class OverlapReport:
    split_a: str
    split_b: str
    n_a: int
    n_b: int
    exact: list[tuple[str, str]] = field(default_factory=list)
    near: list[tuple[str, str, float]] = field(default_factory=list)

    @property
    def exact_rate(self) -> float:
        return len(self.exact) / self.n_b if self.n_b else 0.0

    @property
    def near_rate(self) -> float:
        return len(self.near) / self.n_b if self.n_b else 0.0

    @property
    def clean(self) -> bool:
        return not self.exact and not self.near

    def summary(self) -> str:
        return (f"{self.split_a} -> {self.split_b}: "
                f"exact {len(self.exact)}/{self.n_b} ({100 * self.exact_rate:.2f}%), "
                f"near {len(self.near)}/{self.n_b} ({100 * self.near_rate:.2f}%)")


@dataclass
class IntegrityReport:
    overlaps: list[OverlapReport]
    duplicate_rates: dict[str, float]
    split_sizes: dict[str, int]

    @property
    def clean(self) -> bool:
        return all(o.clean for o in self.overlaps)

    def summary(self) -> str:
        lines = ["split sizes: " + ", ".join(
            f"{k}={v:,}" for k, v in sorted(self.split_sizes.items()))]
        lines += [f"  within-split duplicate rate: {k}={100 * v:.2f}%"
                  for k, v in sorted(self.duplicate_rates.items())]
        lines += ["  " + o.summary() for o in self.overlaps]
        lines.append(f"  VERDICT: {'clean' if self.clean else 'LEAKING'}")
        return "\n".join(lines)


def audit_splits(
    splits: Mapping[str, Sequence[tuple[str, str]]],
    *,
    near_threshold: float = 0.8,
    check_near: bool = True,
    max_report: int = 50,
    raise_on_leak: bool = True,
    directions: Sequence[tuple[str, str]] | None = None,
) -> IntegrityReport:
    """Check ordered pairs of splits for exact and near duplicates.

    ``splits`` maps a split name to ``(item_id, text)`` pairs. Comparisons are directed:
    ``train -> test`` asks "how much of test was seen in train", which is the quantity
    that matters and the one v1 never computed. ``directions`` restricts the check to
    specific ordered pairs; the default checks all of them, which is O(n_splits^2) and
    wasteful once the informative direction is known.
    """
    normalised = {name: {i: normalise_text(t) for i, t in rows} for name, rows in splits.items()}
    sizes = {name: len(rows) for name, rows in splits.items()}

    duplicate_rates = {}
    for name, mapping in normalised.items():
        seen: set[str] = set()
        dupes = 0
        for text in mapping.values():
            if text in seen:
                dupes += 1
            seen.add(text)
        duplicate_rates[name] = dupes / len(mapping) if mapping else 0.0

    names = list(splits)
    wanted = ([(a, b) for a in names for b in names if a != b] if directions is None
              else [tuple(d) for d in directions])
    unknown = {s for pair in wanted for s in pair} - set(names)
    if unknown:
        raise KeyError(f"directions reference unknown splits: {sorted(unknown)}")

    indices: dict[str, MinHashIndex] = {}
    signatures: dict[str, dict[str, "np.ndarray"]] = {}
    if check_near:
        needed = {a for a, _ in wanted} | {b for _, b in wanted}
        for name in needed:
            idx = MinHashIndex()
            sigs = {}
            for item_id, text in splits[name]:
                idx.add(item_id, text)
                sigs[item_id] = idx._signatures[item_id]
            indices[name] = idx
            signatures[name] = sigs

    overlaps = []
    for a, b in wanted:
        by_text_a = defaultdict(list)
        for item_id, text in normalised[a].items():
            by_text_a[text].append(item_id)

        report = OverlapReport(a, b, sizes[a], sizes[b])
        for item_id, text in normalised[b].items():
            if text in by_text_a:
                report.exact.append((by_text_a[text][0], item_id))
                if len(report.exact) >= max_report * 20:
                    break
        if check_near:
            exact_b = {pair[1] for pair in report.exact}
            for item_id, text in splits[b]:
                if item_id in exact_b:
                    continue
                hits = indices[a].query(text, near_threshold,
                                        signature=signatures[b].get(item_id))
                if hits:
                    report.near.append((hits[0][0], item_id, hits[0][1]))
                if len(report.near) >= max_report * 20:
                    break
        overlaps.append(report)

    result = IntegrityReport(overlaps, duplicate_rates, sizes)
    if raise_on_leak and not result.clean:
        offenders = [o.summary() for o in result.overlaps if not o.clean]
        raise LeakageError(
            "split overlap detected; refusing to emit the dataset:\n  "
            + "\n  ".join(offenders[:10])
            + "\n(pass raise_on_leak=False and record the justification if this is intended)"
        )
    return result


def assert_no_resampling(rows: Iterable[tuple[str, str]], name: str = "split") -> None:
    """Fail loudly on the exact v1 failure mode: the same text repeated many times.

    A within-split duplicate rate above ~1% on naturally occurring text is almost always
    sampling with replacement rather than a property of the corpus.
    """
    texts = [normalise_text(t) for _, t in rows]
    if not texts:
        return
    unique = len(set(texts))
    rate = 1 - unique / len(texts)
    if rate > 0.01:
        raise LeakageError(
            f"{name}: {len(texts):,} rows but only {unique:,} unique texts "
            f"({100 * rate:.1f}% duplicated). This is the AUDIT.md S1-1 failure mode: "
            "balance classes by reweighting the loss, not by resampling rows."
        )


def stratified_holdout(
    keys: Sequence[str],
    strata: Sequence[str],
    fraction: float,
    seed: int,
) -> set[str]:
    """Deterministically select ``fraction`` of ``keys`` within each stratum.

    Selection is by a seeded hash of the key rather than by shuffling a list, so the
    held-out set is stable when upstream data is added or reordered: a property the
    released item ids need in order to stay meaningful.
    """
    if not 0 < fraction < 1:
        raise ValueError("fraction must be in (0, 1)")
    by_stratum: dict[str, list[str]] = defaultdict(list)
    for key, stratum in zip(keys, strata, strict=True):
        by_stratum[stratum].append(key)

    chosen: set[str] = set()
    for members in by_stratum.values():
        if not members:
            continue
        ranked = sorted(members, key=lambda k: _hash64(k, seed))
        chosen.update(ranked[:max(1, round(len(members) * fraction))])
    return chosen
