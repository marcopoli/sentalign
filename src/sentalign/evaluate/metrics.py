"""Metric suite.

Depends only on numpy, so it is unit-testable without a GPU or a model.

Design notes that are arguments, not preferences:

*   **Accuracy and calibration are computed from the same probabilities.** v1 took
    predictions from free generation and probabilities from constrained label scoring
    (AUDIT.md S2-6). An ECE over a distribution that never produced the reported
    predictions is not a calibration measurement of the reported system.
*   **ECE uses equal-mass bins.** Equal-width bins are the common choice and are badly
    biased when confidence piles up near 1.0, which is exactly what fine-tuned
    classifiers do. Both are reported; equal-mass is the headline.
*   **Human-distribution fit is a first-class metric.** With five annotators per item the
    target is a distribution, and a model that puts 0.6 on positive for an item where
    3/5 humans said positive is *right* in a way accuracy cannot express.
*   **Selective prediction** (AURC) is included because it is the number an edge
    deployment actually cares about: a model that knows when to abstain is worth more
    than one point of macro-F1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

EPS = 1e-12

# numpy renamed trapz -> trapezoid in 2.0; clusters often pin 1.x.
_trapz = getattr(np, "trapezoid", None) or np.trapz


# --------------------------------------------------------------------------------------
# Accuracy
# --------------------------------------------------------------------------------------

def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def per_class_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    cm = confusion_matrix(y_true, y_pred, n_classes)
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    denom = 2 * tp + fp + fn
    return np.where(denom > 0, 2 * tp / np.maximum(denom, EPS), 0.0)


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    return float(per_class_f1(y_true, y_pred, n_classes).mean())


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_true == y_pred).mean()) if len(y_true) else 0.0


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    """Mean per-class recall. Reported because DynaSent round 1 is 48% neutral, so plain
    accuracy rewards a model that simply learns the prior."""
    recalls = []
    for c in range(n_classes):
        mask = y_true == c
        if mask.any():
            recalls.append(float((y_pred[mask] == c).mean()))
    return float(np.mean(recalls)) if recalls else 0.0


# --------------------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------------------

def expected_calibration_error(
    probs: np.ndarray, y_true: np.ndarray, n_bins: int = 15, strategy: str = "quantile"
) -> float:
    """|confidence - accuracy| averaged over bins, weighted by bin mass.

    ``strategy='quantile'`` gives equal-mass bins (the headline number);
    ``'uniform'`` gives the conventional equal-width bins, reported alongside.
    """
    if len(y_true) == 0:
        return float("nan")
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == y_true).astype(float)

    if strategy == "quantile":
        edges = np.quantile(conf, np.linspace(0, 1, n_bins + 1))
        edges = np.unique(edges)
        if len(edges) < 2:
            return float(abs(conf.mean() - correct.mean()))
    elif strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    else:
        raise ValueError(f"unknown strategy {strategy!r}")

    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, len(edges) - 2)
    ece = 0.0
    for b in range(len(edges) - 1):
        mask = idx == b
        if not mask.any():
            continue
        ece += mask.mean() * abs(conf[mask].mean() - correct[mask].mean())
    return float(ece)


def brier_score(probs: np.ndarray, y_true: np.ndarray, n_classes: int) -> float:
    """Multiclass Brier score: mean squared error against the one-hot target."""
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(y_true)), y_true] = 1.0
    return float(((probs - onehot) ** 2).sum(axis=1).mean())


def negative_log_likelihood(probs: np.ndarray, y_true: np.ndarray) -> float:
    return float(-np.log(np.clip(probs[np.arange(len(y_true)), y_true], EPS, 1.0)).mean())


def fit_temperature(
    logits: np.ndarray, y_true: np.ndarray, bounds: tuple[float, float] = (0.05, 10.0),
    tol: float = 1e-4,
) -> float:
    """Temperature minimising NLL, by golden-section search on a 1-D convex problem.

    Reported because temperature scaling (Guo et al., 2017) is what a practitioner would
    do anyway: the interesting question is not whether an objective is calibrated
    out of the box, but whether it is still better *after* the cheap fix. Several
    "calibration" results in the PO literature vanish under this control.
    """
    inv_phi = (np.sqrt(5) - 1) / 2

    def nll(t: float) -> float:
        z = logits / t
        z = z - z.max(axis=1, keepdims=True)
        logZ = np.log(np.exp(z).sum(axis=1))
        return float(-(z[np.arange(len(y_true)), y_true] - logZ).mean())

    a, b = bounds
    c, d = b - inv_phi * (b - a), a + inv_phi * (b - a)
    while abs(b - a) > tol:
        if nll(c) < nll(d):
            b, d = d, c
            c = b - inv_phi * (b - a)
        else:
            a, c = c, d
            d = a + inv_phi * (b - a)
    return float((a + b) / 2)


def fit_temperature_to_distribution(
    logits: np.ndarray, p_human: np.ndarray, bounds: tuple[float, float] = (0.05, 10.0),
    tol: float = 1e-4,
) -> float:
    """Temperature minimising cross-entropy to the *annotator distribution*.

    ``fit_temperature`` calibrates against hard labels, which is what a practitioner
    chasing accuracy would do. Someone chasing the annotator distribution would fit this
    instead, and it is the stronger control: on the sentiment data a single temperature
    fitted this way to a plain hard-label SFT model matched every distributional method,
    which is the first thing a reviewer will try. Reporting it is not optional.

    Golden-section search, as in ``fit_temperature``; the objective is convex in log t
    for the same reason.
    """
    inv_phi = (np.sqrt(5) - 1) / 2

    def cross_entropy(t: float) -> float:
        z = logits / t
        z = z - z.max(axis=1, keepdims=True)
        log_q = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
        return float(-(p_human * log_q).sum(axis=1).mean())

    a, b = bounds
    c, d = b - inv_phi * (b - a), a + inv_phi * (b - a)
    while abs(b - a) > tol:
        if cross_entropy(c) < cross_entropy(d):
            b, d = d, c
            c = b - inv_phi * (b - a)
        else:
            a, c = c, d
            d = a + inv_phi * (b - a)
    return float((a + b) / 2)


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    z = logits / temperature
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


# --------------------------------------------------------------------------------------
# Fit to the human label distribution
# --------------------------------------------------------------------------------------

def jensen_shannon(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Row-wise JS divergence in bits. Bounded in [0, 1], symmetric, defined when a
    human distribution has zeros: which KL is not, and every DynaSent item has."""
    p = np.clip(p, EPS, 1.0); p = p / p.sum(axis=1, keepdims=True)
    q = np.clip(q, EPS, 1.0); q = q / q.sum(axis=1, keepdims=True)
    m = 0.5 * (p + q)
    kl = lambda a, b: (a * (np.log2(a) - np.log2(b))).sum(axis=1)
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def human_distribution_metrics(probs: np.ndarray, p_human: np.ndarray) -> dict[str, float]:
    """How close the model's distribution is to the annotators' distribution.

    ``top_agreement`` is the fraction of items where the model's argmax matches the
    human modal label: the accuracy-flavoured projection of the same comparison, kept
    so the distributional numbers can be read against something familiar.
    """
    jsd = jensen_shannon(probs, p_human)
    ce = -(p_human * np.log(np.clip(probs, EPS, 1.0))).sum(axis=1)
    human_entropy = -(np.where(p_human > 0, p_human * np.log(np.clip(p_human, EPS, 1)), 0)).sum(axis=1)
    return {
        "jsd_human": float(jsd.mean()),
        "jsd_human_median": float(np.median(jsd)),
        "cross_entropy_human": float(ce.mean()),
        "excess_ce_human": float((ce - human_entropy).mean()),  # 0 iff model == humans
        "top_agreement": float((probs.argmax(1) == p_human.argmax(1)).mean()),
    }


# --------------------------------------------------------------------------------------
# Selective prediction
# --------------------------------------------------------------------------------------

def risk_coverage_curve(
    probs: np.ndarray, y_true: np.ndarray, confidence: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Selective risk as a function of coverage, ordering items by confidence."""
    conf = probs.max(axis=1) if confidence is None else confidence
    correct = (probs.argmax(axis=1) == y_true).astype(float)
    order = np.argsort(-conf, kind="stable")
    errors = 1.0 - correct[order]
    n = len(errors)
    coverage = np.arange(1, n + 1) / n
    risk = np.cumsum(errors) / np.arange(1, n + 1)
    return coverage, risk


def aurc(probs: np.ndarray, y_true: np.ndarray, confidence: np.ndarray | None = None) -> float:
    """Area under the risk-coverage curve. Lower is better; a perfect ranker of its own
    errors approaches the optimal AURC, not zero."""
    if len(y_true) == 0:
        return float("nan")
    coverage, risk = risk_coverage_curve(probs, y_true, confidence)
    if len(coverage) <= 1:
        return float(risk[0])
    return float(_trapz(risk, coverage) / (coverage[-1] - coverage[0] + EPS))


def accuracy_at_coverage(
    probs: np.ndarray, y_true: np.ndarray, coverage: float,
    confidence: np.ndarray | None = None,
) -> float:
    """Accuracy on the most-confident ``coverage`` fraction of items."""
    if not 0 < coverage <= 1:
        raise ValueError("coverage must lie in (0, 1]")
    conf = probs.max(axis=1) if confidence is None else confidence
    order = np.argsort(-conf, kind="stable")[:max(1, int(round(coverage * len(y_true))))]
    return float((probs.argmax(axis=1)[order] == y_true[order]).mean())


# --------------------------------------------------------------------------------------
# Group robustness and behaviour
# --------------------------------------------------------------------------------------

def group_metrics(
    probs: np.ndarray, y_true: np.ndarray, groups: Sequence[str], n_classes: int,
    min_size: int = 30,
) -> dict[str, float | dict[str, float]]:
    """Per-group macro-F1 plus the worst-group and the spread.

    Groups smaller than ``min_size`` are excluded from the worst-group statistic: with
    5-10 items the minimum over groups is a measure of sampling noise, and reporting it
    as robustness is how group-robust methods get credited for variance.
    """
    y_pred = probs.argmax(axis=1)
    groups = np.asarray(groups)
    per_group, eligible = {}, {}
    for g in np.unique(groups):
        mask = groups == g
        score = macro_f1(y_true[mask], y_pred[mask], n_classes)
        per_group[str(g)] = score
        if mask.sum() >= min_size:
            eligible[str(g)] = score
    if not eligible:
        return {"per_group_f1": per_group, "worst_group_f1": float("nan"),
                "group_gap": float("nan"), "n_eligible_groups": 0}
    values = list(eligible.values())
    return {
        "per_group_f1": per_group,
        "worst_group_f1": float(min(values)),
        "group_gap": float(max(values) - min(values)),
        "n_eligible_groups": len(eligible),
    }


def label_prior_drift(probs: np.ndarray, reference_prior: np.ndarray) -> float:
    """Total-variation distance between the model's marginal predictions and a reference.

    Preference optimization is known to shift a model's output marginal; on a classifier
    that shift is directly visible and directly harmful, so it gets its own number rather
    than being left to leak into macro-F1.
    """
    predicted = np.bincount(probs.argmax(axis=1), minlength=len(reference_prior))
    predicted = predicted / max(predicted.sum(), 1)
    return float(0.5 * np.abs(predicted - reference_prior).sum())


def counterfactual_flip_rate(
    pred_original: np.ndarray, pred_edited: np.ndarray, should_flip: np.ndarray
) -> dict[str, float]:
    """Sensitivity to human edits that change sentiment, and stability to edits that do not.

    ``consistency`` is the joint rate of getting both right, which is the quantity that
    matters: a model can score well on either half alone by being uniformly rigid or
    uniformly jumpy.
    """
    flipped = pred_original != pred_edited
    should_flip = should_flip.astype(bool)
    out = {}
    if should_flip.any():
        out["sensitivity"] = float(flipped[should_flip].mean())
    if (~should_flip).any():
        out["stability"] = float((~flipped[~should_flip]).mean())
    correct = np.where(should_flip, flipped, ~flipped)
    out["consistency"] = float(correct.mean())
    return out


# --------------------------------------------------------------------------------------
# Aggregate
# --------------------------------------------------------------------------------------

@dataclass
class EvalResult:
    n: int
    metrics: dict[str, float] = field(default_factory=dict)
    groups: dict[str, float] = field(default_factory=dict)
    extra: dict = field(default_factory=dict)

    def flat(self) -> dict[str, float]:
        out = dict(self.metrics)
        out.update({f"group/{k}": v for k, v in self.groups.items()
                    if isinstance(v, (int, float))})
        return out


def evaluate_predictions(
    logits: np.ndarray,
    y_true: np.ndarray,
    n_classes: int,
    *,
    p_human: np.ndarray | None = None,
    groups: Sequence[str] | None = None,
    parse_failures: int = 0,
    reference_prior: np.ndarray | None = None,
    temperature_fit_logits: np.ndarray | None = None,
    temperature_fit_labels: np.ndarray | None = None,
) -> EvalResult:
    """Compute the full metric suite from one set of logits.

    ``logits`` are the unnormalised verbalizer scores, so probabilities, predictions,
    calibration, and temperature scaling all derive from a single computation: the
    invariant this module exists to enforce.

    Temperature is fitted on *held-out* logits when ``temperature_fit_logits`` is given.
    Fitting on the test set is the standard shortcut and it silently inflates the
    temperature-scaled numbers.
    """
    logits = np.asarray(logits, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.int64)
    probs = apply_temperature(logits, 1.0)
    y_pred = probs.argmax(axis=1)

    metrics: dict[str, float] = {
        "n": float(len(y_true)),
        "accuracy": accuracy(y_true, y_pred),
        "macro_f1": macro_f1(y_true, y_pred, n_classes),
        "balanced_accuracy": balanced_accuracy(y_true, y_pred, n_classes),
        "ece": expected_calibration_error(probs, y_true, strategy="quantile"),
        "ece_uniform": expected_calibration_error(probs, y_true, strategy="uniform"),
        "brier": brier_score(probs, y_true, n_classes),
        "nll": negative_log_likelihood(probs, y_true),
        "aurc": aurc(probs, y_true),
        "acc@50": accuracy_at_coverage(probs, y_true, 0.5),
        "acc@80": accuracy_at_coverage(probs, y_true, 0.8),
        "acc@90": accuracy_at_coverage(probs, y_true, 0.9),
        "parse_failure_rate": parse_failures / max(len(y_true) + parse_failures, 1),
    }
    for c, f1 in enumerate(per_class_f1(y_true, y_pred, n_classes)):
        metrics[f"f1_class{c}"] = float(f1)

    fit_logits = temperature_fit_logits if temperature_fit_logits is not None else logits
    fit_labels = temperature_fit_labels if temperature_fit_labels is not None else y_true
    temperature = fit_temperature(np.asarray(fit_logits, dtype=np.float64),
                                  np.asarray(fit_labels, dtype=np.int64))
    scaled = apply_temperature(logits, temperature)
    metrics["temperature"] = temperature
    metrics["ece_ts"] = expected_calibration_error(scaled, y_true, strategy="quantile")
    metrics["brier_ts"] = brier_score(scaled, y_true, n_classes)
    metrics["nll_ts"] = negative_log_likelihood(scaled, y_true)

    if p_human is not None:
        metrics.update(human_distribution_metrics(probs, np.asarray(p_human, dtype=np.float64)))
    if reference_prior is not None:
        metrics["label_prior_drift"] = label_prior_drift(probs, np.asarray(reference_prior))

    groups_out: dict = {}
    if groups is not None:
        groups_out = group_metrics(probs, y_true, groups, n_classes)

    return EvalResult(n=len(y_true), metrics=metrics, groups=groups_out,
                      extra={"temperature": temperature})
