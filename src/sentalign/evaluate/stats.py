"""Significance testing for the method comparison.

v1 reported mean +/- std over three seeds and drew conclusions from it (AUDIT.md S2-8).
Effects in the preference-optimization literature are routinely 0.3-1.0 macro-F1 points
while seed-to-seed spread on a 3.6k test set is comparable, so that protocol cannot
distinguish a method from a lucky initialisation.

What this module provides instead:

*   **Paired bootstrap** over test items (Koehn, 2004). Every arm is evaluated on the
    same items, so resampling items jointly removes the between-item variance that
    dominates an unpaired comparison.
*   **Seed-aware resampling.** The unit of replication is a (seed, item) pair. Bootstrap
    over items alone treats five runs of the same method as five times the evidence;
    ``paired_bootstrap`` resamples seeds and items together so the CI reflects both.
*   **Holm-Bonferroni.** Eight objectives are each compared to one SFT baseline, so the
    family-wise error rate at alpha=0.05 uncorrected is ~34%. Holm is uniformly more
    powerful than Bonferroni and needs no independence assumption.
*   **Effect sizes.** A p-value on a 3.6k test set says almost nothing about whether an
    effect is worth a practitioner's GPU time; the paired mean difference and its CI do.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np

Metric = Callable[[np.ndarray, np.ndarray], float]


@dataclass(frozen=True)
class ComparisonResult:
    """One method-versus-baseline comparison."""

    name: str
    baseline_score: float
    method_score: float
    difference: float
    ci_low: float
    ci_high: float
    p_value: float
    p_adjusted: float | None = None
    n_items: int = 0
    n_seeds: int = 1

    @property
    def significant(self) -> bool:
        p = self.p_adjusted if self.p_adjusted is not None else self.p_value
        return p < 0.05

    @property
    def ci_excludes_zero(self) -> bool:
        return self.ci_low > 0 or self.ci_high < 0

    def format_row(self) -> str:
        p = self.p_adjusted if self.p_adjusted is not None else self.p_value
        stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        return (f"{self.name:<12} {self.method_score:7.4f}  "
                f"{self.difference:+7.4f} [{self.ci_low:+.4f}, {self.ci_high:+.4f}]  "
                f"p={p:.4f}{stars}")


def bootstrap_ci(
    values: np.ndarray, statistic: Callable[[np.ndarray], float] = np.mean,
    n_resamples: int = 10_000, alpha: float = 0.05, seed: int = 0,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI. Returns ``(point, low, high)``."""
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    idx = rng.integers(0, n, size=(n_resamples, n))
    draws = np.array([statistic(values[i]) for i in idx])
    return (float(statistic(values)),
            float(np.quantile(draws, alpha / 2)),
            float(np.quantile(draws, 1 - alpha / 2)))


def paired_bootstrap(
    baseline: np.ndarray,
    method: np.ndarray,
    *,
    seeds_baseline: np.ndarray | None = None,
    seeds_method: np.ndarray | None = None,
    n_resamples: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
    name: str = "method",
    alternative: str = "two-sided",
) -> ComparisonResult:
    """Paired bootstrap over per-item scores.

    ``baseline`` and ``method`` are per-item scores (per-item correctness, per-item NLL,
    per-item JSD ...) aligned on the same items. When seed arrays are supplied the arrays
    are ``(n_seeds * n_items,)`` long and seeds are resampled with items.

    The p-value is the two-sided bootstrap proportion of resamples whose mean difference
    has the opposite sign to the observed one, doubled and clipped: the standard
    percentile test. It is *not* a permutation test: with per-item scores that are not
    exchangeable between arms, permutation would test the wrong null.
    """
    baseline = np.asarray(baseline, dtype=np.float64)
    method = np.asarray(method, dtype=np.float64)
    if baseline.shape != method.shape:
        raise ValueError(f"shape mismatch: {baseline.shape} vs {method.shape}")

    diff = method - baseline
    rng = np.random.default_rng(seed)

    if seeds_baseline is not None:
        seeds_baseline = np.asarray(seeds_baseline)
        if seeds_method is not None and not np.array_equal(seeds_baseline,
                                                           np.asarray(seeds_method)):
            raise ValueError("baseline and method must be aligned on the same seed order")
        unique_seeds = np.unique(seeds_baseline)
        by_seed = {s: np.flatnonzero(seeds_baseline == s) for s in unique_seeds}
        n_items = len(by_seed[unique_seeds[0]])
        draws = np.empty(n_resamples)
        for r in range(n_resamples):
            chosen = rng.choice(unique_seeds, size=len(unique_seeds), replace=True)
            item_idx = rng.integers(0, n_items, size=n_items)
            acc = [diff[by_seed[s][item_idx]] for s in chosen]
            draws[r] = float(np.concatenate(acc).mean())
        n_seeds, n_report = len(unique_seeds), n_items
    else:
        n = len(diff)
        idx = rng.integers(0, n, size=(n_resamples, n))
        draws = diff[idx].mean(axis=1)
        n_seeds, n_report = 1, n

    observed = float(diff.mean())
    if alternative == "two-sided":
        p = 2 * min((draws <= 0).mean(), (draws >= 0).mean())
    elif alternative == "greater":
        p = float((draws <= 0).mean())
    elif alternative == "less":
        p = float((draws >= 0).mean())
    else:
        raise ValueError(f"unknown alternative {alternative!r}")
    # A bootstrap p-value can never be below 1/n_resamples; reporting 0.0 overstates it.
    p = float(min(1.0, max(p, 1.0 / n_resamples)))

    return ComparisonResult(
        name=name,
        baseline_score=float(baseline.mean()),
        method_score=float(method.mean()),
        difference=observed,
        ci_low=float(np.quantile(draws, alpha / 2)),
        ci_high=float(np.quantile(draws, 1 - alpha / 2)),
        p_value=p,
        n_items=n_report,
        n_seeds=n_seeds,
    )


def holm_bonferroni(results: Sequence[ComparisonResult],
                    alpha: float = 0.05) -> list[ComparisonResult]:
    """Holm step-down correction across a family of comparisons.

    Uniformly more powerful than Bonferroni at the same FWER and valid under arbitrary
    dependence, which matters here because the arms share a base model and a test set.
    Adjusted p-values are made monotone so a later hypothesis can never report a smaller
    adjusted p than an earlier, more significant one.
    """
    order = sorted(range(len(results)), key=lambda i: results[i].p_value)
    m = len(results)
    adjusted = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        value = min(1.0, (m - rank) * results[idx].p_value)
        running = max(running, value)
        adjusted[idx] = running
    return [
        ComparisonResult(
            name=r.name, baseline_score=r.baseline_score, method_score=r.method_score,
            difference=r.difference, ci_low=r.ci_low, ci_high=r.ci_high,
            p_value=r.p_value, p_adjusted=adjusted[i], n_items=r.n_items, n_seeds=r.n_seeds,
        )
        for i, r in enumerate(results)
    ]


def compare_family(
    baseline_scores: np.ndarray,
    method_scores: Mapping[str, np.ndarray],
    *,
    seeds: np.ndarray | None = None,
    n_resamples: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> list[ComparisonResult]:
    """Compare every method to one baseline and correct the family."""
    raw = [
        paired_bootstrap(baseline_scores, scores, seeds_baseline=seeds, seeds_method=seeds,
                         n_resamples=n_resamples, alpha=alpha, seed=seed + i, name=name)
        for i, (name, scores) in enumerate(sorted(method_scores.items()))
    ]
    return holm_bonferroni(raw, alpha=alpha)


def mcnemar_exact(baseline_correct: np.ndarray, method_correct: np.ndarray) -> float:
    """Exact McNemar p-value on paired binary outcomes.

    Reported alongside the bootstrap for the accuracy metric specifically: it is the
    classical test for this design, it makes no distributional assumption, and reviewers
    in this area expect to see it.
    """
    from math import comb

    b = int(np.sum((baseline_correct == 1) & (method_correct == 0)))
    c = int(np.sum((baseline_correct == 0) & (method_correct == 1)))
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    return float(min(1.0, 2 * tail))


def required_seeds(
    effect_size: float, item_sd: float, n_items: int, seed_sd: float,
    power: float = 0.8, alpha: float = 0.05,
) -> int:
    """Seeds needed to detect ``effect_size`` at the given power.

    Used to fix the seed count *before* running the grid rather than justifying it
    afterwards. The variance of a per-arm mean is ``seed_sd^2 + item_sd^2 / n_items``;
    the paired difference of two arms has twice that (conservative: it ignores the
    positive correlation the pairing induces, so the answer errs high).
    """
    from math import ceil, sqrt

    z_alpha = 1.959963985  # two-sided 0.05
    z_beta = {0.8: 0.8416212336, 0.9: 1.2815515655, 0.95: 1.6448536270}.get(power)
    if z_beta is None:
        raise ValueError("power must be one of 0.80, 0.90, 0.95")
    if effect_size <= 0:
        raise ValueError("effect_size must be positive")
    per_seed_var = 2 * (seed_sd ** 2 + item_sd ** 2 / max(n_items, 1))
    return max(2, ceil(per_seed_var * ((z_alpha + z_beta) / effect_size) ** 2))


def required_seeds_paired(
    effect_size: float,
    item_sd: float,
    n_items: int,
    seed_sd: float,
    *,
    item_correlation: float = 0.9,
    shared_reference: bool = True,
    power: float = 0.8,
    alpha: float = 0.05,
) -> int:
    """Seeds needed under the paired design this study actually uses.

    ``required_seeds`` deliberately ignores pairing and therefore answers the question
    "how many seeds would an unpaired comparison need", which is the right number to
    quote about the literature and the wrong number to design against. Two features of
    the design reduce the variance of the paired difference:

    ``item_correlation``
        Two arms fine-tuned from the same checkpoint agree on most items, so the
        per-item difference has variance ``2 * item_sd^2 * (1 - rho)`` rather than
        ``2 * item_sd^2``. At rho = 0.9 that is a fivefold reduction.

    ``shared_reference``
        All arms for a given model and seed start from that model and seed's own SFT
        checkpoint, so the initialisation component of seed variance is common to both
        sides of the comparison and cancels. We credit half of the seed variance to it,
        which is conservative.

    Both assumptions are checkable after the fact from the released per-item predictions,
    and the paper reports the realised correlation alongside the planned one.
    """
    from math import ceil

    if not 0.0 <= item_correlation < 1.0:
        raise ValueError("item_correlation must lie in [0, 1)")
    z_alpha = 1.959963985
    z_beta = {0.8: 0.8416212336, 0.9: 1.2815515655, 0.95: 1.6448536270}.get(power)
    if z_beta is None:
        raise ValueError("power must be one of 0.80, 0.90, 0.95")
    if effect_size <= 0:
        raise ValueError("effect_size must be positive")

    item_term = 2 * item_sd ** 2 * (1 - item_correlation) / max(n_items, 1)
    seed_term = 2 * seed_sd ** 2 * (0.5 if shared_reference else 1.0)
    variance = item_term + seed_term
    return max(2, ceil(variance * ((z_alpha + z_beta) / effect_size) ** 2))


def observed_item_correlation(baseline: np.ndarray, method: np.ndarray) -> float:
    """Realised per-item correlation between two arms, for reporting against the plan."""
    baseline = np.asarray(baseline, dtype=np.float64)
    method = np.asarray(method, dtype=np.float64)
    if baseline.std() == 0 or method.std() == 0:
        return float("nan")
    return float(np.corrcoef(baseline, method)[0, 1])


def format_table(results: Sequence[ComparisonResult], metric: str = "macro_f1",
                 lower_is_better: bool = False) -> str:
    """Rank best first and name the column after the metric actually compared.

    Both defaults were wrong for a divergence. The header hard-coded ``macro_f1`` while
    the numbers were whatever ``--metric`` selected, and the sort put the largest delta
    first, which for JSD is the worst arm. A table captioned macro-F1, sorted worst to
    best, containing Jensen-Shannon divergences is three ways misleading at once.
    """
    header = (f"{'method':<12} {metric:>7}  {'Δ vs SFT':>7} {'95% CI':>21}  "
              f"{'p (Holm)':>12}")
    lines = [header, "-" * len(header)]
    lines += [r.format_row() for r in
              sorted(results, key=lambda r: r.difference if lower_is_better
                     else -r.difference)]
    return "\n".join(lines)
