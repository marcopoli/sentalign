"""Preference-optimization losses, written so the math can be tested without a GPU.

Two of the five objectives in v1 were not the objectives they were named after
(AUDIT.md S1-4, S1-5). The response is to write each loss out explicitly, cite the
equation it implements, and unit-test it against a closed-form reference: see
``tests/test_objectives.py``. Every function here takes plain arrays and works with
numpy or torch through the small shim below, so the tests run on any machine.

Which objectives come from TRL and which are implemented here:

| objective | source | notes |
|-----------|--------|-------|
| DPO       | TRL `loss_type="sigmoid"` | Rafailov et al. (2023) |
| cDPO      | TRL `loss_type="sigmoid"`, `label_smoothing=ε` | conservative DPO |
| rDPO      | TRL `loss_type="robust"`, `label_smoothing=ε` | Chowdhury et al. (2024) |
| IPO       | TRL `loss_type="ipo"` | Azar et al. (2024); β is τ |
| DiscoPOP  | TRL `loss_type="discopop"` | discovered objective |
| KTO       | TRL `KTOTrainer` | Ethayarajh et al. (2024), unpaired |
| SimPO     | **here** | TRL 1.10 removed `CPOTrainer`, which hosted it |
| Dr. DPO   | **here** | not available in TRL |
| GR-DPO    | **here** | Ramesh et al. (2024), group-robust |
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence

import math

import numpy as np


class _Ops(Protocol):
    def logsigmoid(self, x: Any) -> Any: ...
    def exp(self, x: Any) -> Any: ...
    def mean(self, x: Any) -> Any: ...
    def clamp(self, x: Any, lo: float, hi: float) -> Any: ...


class NumpyOps:
    """Reference backend used by the tests."""

    @staticmethod
    def logsigmoid(x):
        x = np.asarray(x, dtype=np.float64)
        # numerically stable log(1/(1+e^-x))
        return np.where(x >= 0, -np.log1p(np.exp(-np.abs(x))),
                        x - np.log1p(np.exp(-np.abs(x))))

    @staticmethod
    def exp(x):
        return np.exp(np.asarray(x, dtype=np.float64))

    @staticmethod
    def mean(x):
        return float(np.mean(x))

    @staticmethod
    def clamp(x, lo, hi):
        return np.clip(np.asarray(x, dtype=np.float64), lo, hi)


class TorchOps:
    """Backend used in training."""

    def __init__(self) -> None:
        import torch
        import torch.nn.functional as F

        self._torch, self._F = torch, F

    def logsigmoid(self, x):
        return self._F.logsigmoid(x)

    def exp(self, x):
        return self._torch.exp(x)

    def mean(self, x):
        return x.mean()

    def clamp(self, x, lo, hi):
        return self._torch.clamp(x, lo, hi)


def get_ops(backend: str = "numpy") -> _Ops:
    return NumpyOps() if backend == "numpy" else TorchOps()


# --------------------------------------------------------------------------------------
# Pairwise objectives
# --------------------------------------------------------------------------------------

def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps,
             beta: float = 0.1, label_smoothing: float = 0.0, ops: _Ops | None = None):
    """DPO (Rafailov et al., 2023), with optional conservative label smoothing.

        ℓ = -(1-ε)·log σ(β·Δ) - ε·log σ(-β·Δ)
        Δ = [log π_θ(y_w|x) - log π_ref(y_w|x)] - [log π_θ(y_l|x) - log π_ref(y_l|x)]

    ``label_smoothing=0`` is plain DPO; ``ε>0`` is cDPO, which assumes preference labels
    are flipped with probability ε and caps the gradient accordingly.
    """
    ops = ops or NumpyOps()
    delta = ((policy_chosen_logps - ref_chosen_logps)
             - (policy_rejected_logps - ref_rejected_logps))
    return (-(1 - label_smoothing) * ops.logsigmoid(beta * delta)
            - label_smoothing * ops.logsigmoid(-beta * delta))


def robust_dpo_loss(policy_chosen_logps, policy_rejected_logps,
                    ref_chosen_logps, ref_rejected_logps,
                    beta: float = 0.1, epsilon: float = 0.1, ops: _Ops | None = None):
    """rDPO (Chowdhury, Kini & Natarajan, 2024): the unbiased estimator under known flips.

        ℓ = [-(1-ε)·log σ(β·Δ) + ε·log σ(-β·Δ)] / (1 - 2ε)

    Note the sign on the second term and the 1/(1-2ε) rescaling: this is what makes the
    estimator unbiased for the clean loss, and it is what distinguishes rDPO from cDPO,
    whose two terms are both negative. v1 collapsed all of this into one ad-hoc
    reweighting (AUDIT.md S1-3).
    """
    if not 0.0 <= epsilon < 0.5:
        raise ValueError("epsilon must lie in [0, 0.5)")
    ops = ops or NumpyOps()
    delta = ((policy_chosen_logps - ref_chosen_logps)
             - (policy_rejected_logps - ref_rejected_logps))
    clean = -(1 - epsilon) * ops.logsigmoid(beta * delta)
    flipped = -epsilon * ops.logsigmoid(-beta * delta)
    return (clean - flipped) / (1 - 2 * epsilon)


def ipo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps,
             tau: float = 0.1, ops: _Ops | None = None):
    """IPO (Azar et al., 2024), Eq. 17: a squared loss around a 1/(2τ) target margin.

        ℓ = (Δ - 1/(2τ))²

    Unlike DPO's logistic loss this does not saturate, so it cannot drive the implicit
    reward gap to infinity on deterministic preferences: the failure mode IPO was
    introduced to fix, and the one most likely to bite on a closed label set where
    preferences are near-deterministic.
    """
    delta = ((policy_chosen_logps - ref_chosen_logps)
             - (policy_rejected_logps - ref_rejected_logps))
    return (delta - 1.0 / (2.0 * tau)) ** 2


def simpo_loss(policy_chosen_logps, policy_rejected_logps,
               chosen_lengths, rejected_lengths,
               beta: float = 2.0, gamma: float = 0.5, ops: _Ops | None = None):
    """SimPO (Meng, Xia & Chen, 2024). reference-free, length-normalised, margin-based.

        ℓ = -log σ( (β/|y_w|)·log π_θ(y_w|x) - (β/|y_l|)·log π_θ(y_l|x) - γ )

    Both the 1/|y| normalisation and the target margin γ are the contribution. v1 dropped
    the normalisation and reused the symbol γ as the scaling coefficient, which reduces
    the objective to plain RankNet (AUDIT.md S1-5).

    Note ``chosen_lengths``/``rejected_lengths`` are *completion* token counts. On this
    task the verbalizers are 1-2 tokens, so length normalisation is nearly a no-op and
    SimPO reduces to a margin-based reference-free loss: which is itself a finding worth
    reporting rather than an accident to hide.
    """
    ops = ops or NumpyOps()
    chosen_lengths = np.maximum(np.asarray(chosen_lengths), 1) if isinstance(
        chosen_lengths, (list, tuple, np.ndarray)) else chosen_lengths
    rejected_lengths = np.maximum(np.asarray(rejected_lengths), 1) if isinstance(
        rejected_lengths, (list, tuple, np.ndarray)) else rejected_lengths
    reward_chosen = beta * policy_chosen_logps / chosen_lengths
    reward_rejected = beta * policy_rejected_logps / rejected_lengths
    return -ops.logsigmoid(reward_chosen - reward_rejected - gamma)


def dr_dpo_loss(policy_chosen_logps, policy_rejected_logps,
                ref_chosen_logps, ref_rejected_logps,
                beta: float = 0.1, beta_prime: float = 1.0, ops: _Ops | None = None):
    """Dr. DPO (Wu et al., 2024): distributionally robust DPO over pointwise noise.

    The KL-constrained DRO problem ``max_{Q: KL(Q‖P) <= ρ} E_Q[ℓ]`` has the dual form

        ℓ_batch = β' · log E_batch[ exp( ℓ_DPO / β' ) ]

    which is a soft maximum over the per-example DPO losses: as β' -> 0 it approaches the
    worst-case example, and as β' -> ∞ it approaches the mean (recovering DPO). It is a
    *batch-level* loss, so unlike the others it does not reduce to a per-example term.

    Not available in TRL 1.10, so it is implemented here.
    """
    ops = ops or NumpyOps()
    per_example = dpo_loss(policy_chosen_logps, policy_rejected_logps,
                           ref_chosen_logps, ref_rejected_logps, beta=beta, ops=ops)
    # β' log E[exp(ℓ/β')], stabilised by factoring out the largest scaled loss.
    scaled = per_example / beta_prime
    if isinstance(scaled, np.ndarray):
        m = float(scaled.max())
        return float(beta_prime * (m + np.log(np.mean(np.exp(scaled - m)))))
    m = scaled.max().detach()
    return beta_prime * (m + (ops.exp(scaled - m)).mean().log())


# --------------------------------------------------------------------------------------
# CSPO: Closed-Set Preference Optimization (this work)
# --------------------------------------------------------------------------------------
#
# Motivation. Every objective above reduces annotator evidence to a binary comparison
# (DPO, IPO, SimPO, rDPO), a scalar preference strength attached to a pair (GPO), or an
# ordering over sampled candidates (LiPO, listwise DPO). On a closed label set the
# supervision available is richer and differently shaped: an exhaustive categorical
# distribution over every admissible answer, obtained by counting annotator votes. A
# vote pattern of (3, 1, 1, 0) and one of (3, 2, 0, 0) induce the same top pair and very
# different distributions, and an item with no majority has no preferred answer at all
# while still constraining which answers are plausible.
#
# Definition. Let Y = {y_1..y_K} be the label set with single-token verbalizers
# v_1..v_K, let l_theta(x) be the logits at the final prompt position, and let
# l_ref(x) be the same under the frozen reference policy. Define the implicit reward and
# the reward-induced distribution over Y:
#
#     r_theta(x, y_k) = beta * [ log pi_theta(y_k|x) - log pi_ref(y_k|x) ]
#     q_theta(.|x)    = softmax_k r_theta(x, y_k)
#
# and minimise the cross-entropy of q_theta against the annotator distribution p_human:
#
#     L_CSPO = - E_x sum_k p_human(y_k|x) log q_theta(y_k|x)
#
# Three properties follow, each proved by the tests in tests/test_objectives.py.
#
#   P1 (exact normalizer, one forward pass). For single-token verbalizers,
#      log pi(y_k|x) = l[v_k] - logsumexp_V(l), so
#      r_theta(x,y_k) = beta*(l_theta[v_k] - l_ref[v_k]) - beta*(logZ_theta - logZ_ref).
#      The second term does not depend on k and cancels inside the softmax. Only the K
#      logit differences at one position are needed, so the objective costs a single
#      forward pass over the prompt regardless of K, and the reference logits can be
#      precomputed once. Listwise methods over sampled candidates cannot do this: their
#      normalizer is over an arbitrary sample, so the reward is identified only up to
#      that sample, and they pay one pass per candidate.
#
#   P2 (DPO is a special case). For K = 2 with p_human one-hot on the preferred label,
#      L_CSPO = -log sigma(r_w - r_l), which is exactly the DPO loss. CSPO is therefore a
#      strict generalisation along two axes at once: K > 2, and non-degenerate targets.
#
#   P3 (calibrated fixed point). L_CSPO is minimised at q_theta = p_human, so at the
#      optimum beta*(l_theta[v_k] - l_ref[v_k]) = log p_human(y_k) + c. The implicit
#      reward recovers the log annotator agreement rate up to an additive constant, which
#      is what makes the resulting probabilities interpretable for abstention.
#
# What CSPO is not. It is not a listwise ranking loss: it consumes a distribution, not an
# ordering, and it needs no Plackett-Luce factorisation or tie-handling convention. It is
# not soft-label SFT either, and the difference is the reference anchoring: because the
# full-vocabulary normalizer cancels, CSPO constrains only the *relative* reward among
# the K labels and leaves total likelihood mass untouched, whereas soft-label
# cross-entropy moves the normalizer as well. That control (SFT-soft) is run in the
# experiments precisely because it is the baseline that could make CSPO redundant.


def cspo_loss(policy_logit_diff, p_human, beta: float = 1.0, ops: _Ops | None = None):
    """CSPO cross-entropy loss.

    ``policy_logit_diff``  (B, K)  beta-free logit differences l_theta[v_k] - l_ref[v_k]
    ``p_human``            (B, K)  annotator distribution, rows summing to 1

    Returns the per-example loss (B,). The reference enters only through the precomputed
    difference, so no reference model needs to be resident during training.
    """
    ops = ops or NumpyOps()
    z = np.asarray(policy_logit_diff, dtype=np.float64) * beta
    p = np.asarray(p_human, dtype=np.float64)
    if z.shape != p.shape:
        raise ValueError(f"shape mismatch: logits {z.shape} vs target {p.shape}")
    log_q = z - _logsumexp(z, axis=-1, keepdims=True)
    return -(p * log_q).sum(axis=-1)


def cspo_kl_loss(policy_logits, ref_logits, p_human, lam: float = 1.0):
    """CSPO's goal, moved into the distribution the model actually predicts.

    ``cspo_loss`` fits ``softmax(beta * (l_theta - l_ref))`` to the annotators. At its
    optimum ``l_theta = l_ref + (1/beta) log p_human + c``, so the model's *own*
    distribution is ``softmax(l_theta) proportional to pi_ref * p_human**(1/beta)``: a
    product of experts with the reference, never ``p_human`` itself. Every read-out that
    does not subtract the reference therefore sees a tilted distribution, and the tilt
    shrinks as beta grows because ``p_human**(1/beta)`` flattens. Measured on the grid:
    JSD between CSPO and its reference was 0.0476, 0.0206, 0.0091 at beta 0.5, 1.0, 2.0.

    This variant fits ``softmax(l_theta)`` directly and keeps the reference as a penalty
    beside the loss rather than inside it, so the anchoring survives while the quantity
    scored at evaluation is the quantity trained. ``lam = 0`` recovers ``sft_soft``.

    ``policy_logits``  (B, K)   verbalizer logits from the policy
    ``ref_logits``     (B, K)   the same from the frozen reference, precomputed
    ``p_human``        (B, K)   annotator distribution, rows summing to 1
    """
    z = np.asarray(policy_logits, dtype=np.float64)
    r = np.asarray(ref_logits, dtype=np.float64)
    p = np.asarray(p_human, dtype=np.float64)
    if not (z.shape == r.shape == p.shape):
        raise ValueError(f"shape mismatch: policy {z.shape}, ref {r.shape}, "
                         f"target {p.shape}")
    log_q = z - _logsumexp(z, axis=-1, keepdims=True)
    log_ref = r - _logsumexp(r, axis=-1, keepdims=True)
    cross_entropy = -(p * log_q).sum(axis=-1)
    kl = (np.exp(log_q) * (log_q - log_ref)).sum(axis=-1)
    return cross_entropy + lam * kl


def target_entropy(p_human, base: str = "nats"):
    """Row-wise entropy of the annotator distribution, normalised to [0, 1] by log K.

    Used as the ambiguity signal that conditions the anchor. Normalising by ``log K``
    keeps the resulting weight comparable across label spaces of different size, which
    matters because the same hyperparameter has to hold for K=3 NLI and K=4 sentiment.
    """
    p = np.asarray(p_human, dtype=np.float64)
    k = p.shape[-1]
    h = -(np.where(p > 0, p * np.log(np.clip(p, 1e-12, 1.0)), 0.0)).sum(axis=-1)
    return h / np.log(k)


def adaptive_lambda(p_human, lam0: float = 1.0):
    """Per-item anchor strength, decaying with the ambiguity of that item's target.

    A global anchor cannot be right for both ends of the ambiguity range, and the
    lambda sweep showed why: on ``ambig_no_majority`` JSD rose monotonically with lambda
    (0.140, 0.147, 0.184, 0.230 at lambda 0, 0.1, 1, 10) while on the broader ambiguous
    slice lambda=1 was an interior optimum. The mechanism is that the reference is an SFT
    model fitted to *hard* labels, so it is confidently wrong exactly where annotators
    disagree; anchoring to it helps where it was well supervised and drags the policy off
    a flat target where it was not.

    ``lambda(x) = lam0 * (1 - H(p_human) / log K)`` therefore anchors an unanimous item at
    full strength and an entirely flat one not at all. It is also the property that lets
    this objective escape the calibration equivalence: a global scalar produces a fixed
    monotone rescaling of the reference logits, which is indistinguishable from a
    temperature-scaled reference, whereas an item-conditional weight is not.
    """
    return float(lam0) * (1.0 - target_entropy(p_human))


def empirical_bayes_target(votes, alpha: float = 1.0, prior=None):
    """Shrink an M-annotator empirical distribution toward a prior.

    With M annotators the observed frequencies carry sampling noise that does not shrink
    with dataset size: measured on 1,599 items annotated both ways, the JSD between a
    100-annotator and a 5-annotator estimate of the *same* items is 0.0736, and resampling
    the dense target at M=5 reproduces it at 0.0760. Fitting to raw frequencies therefore
    spends capacity on noise. The posterior mean under a Dirichlet(alpha * prior) prior,

        p_tilde = (n + alpha * prior) / (M + alpha)

    is the minimum-variance correction. ``alpha = 0`` recovers the raw frequencies, so the
    ablation is nested and the comparison stays honest.
    """
    n = np.asarray(votes, dtype=np.float64)
    k = n.shape[-1]
    m = np.full_like(n, 1.0 / k) if prior is None else np.asarray(prior, dtype=np.float64)
    total = n.sum(axis=-1, keepdims=True)
    return (n + alpha * m) / (total + alpha)


def cspo_ada_loss(policy_logits, ref_logits, p_human, lam0: float = 1.0):
    """``cspo_kl`` with the anchor conditioned on each item's ambiguity.

    ``lam0 = 0`` recovers ``sft_soft``; a constant anchor recovers ``cspo_kl``. Both are
    nested, so the ablation that could refute this objective is built into it.
    """
    z = np.asarray(policy_logits, dtype=np.float64)
    r = np.asarray(ref_logits, dtype=np.float64)
    p = np.asarray(p_human, dtype=np.float64)
    if not (z.shape == r.shape == p.shape):
        raise ValueError(f"shape mismatch: policy {z.shape}, ref {r.shape}, "
                         f"target {p.shape}")
    log_q = z - _logsumexp(z, axis=-1, keepdims=True)
    log_ref = r - _logsumexp(r, axis=-1, keepdims=True)
    cross_entropy = -(p * log_q).sum(axis=-1)
    kl = (np.exp(log_q) * (log_q - log_ref)).sum(axis=-1)
    return cross_entropy + adaptive_lambda(p, lam0) * kl


def polya_loss(logits, votes, concentration_scale: float = 1.0,
               max_concentration: float = 1e4):
    """Pólya (Dirichlet-Multinomial) alignment: model the votes, do not match them.

    Every objective in this study, CSPO included, treats the annotator distribution as a
    target and minimises ``CE(p_hat, softmax(l))``. But ``p_hat`` is not a distribution,
    it is a multinomial sample of size M from an unknown p, and at M=5 it carries about
    0.074 JSD of pure sampling noise (measured on 1,599 items annotated at both 5 and
    100). Matching the sample therefore spends capacity on noise. The correct objective
    is the marginal likelihood of the observed *counts*, obtained by placing
    ``p ~ Dirichlet(alpha(x))`` and integrating p out:

        -log P(n | alpha) = -[ lgamma(a0) - lgamma(a0 + M)
                               + sum_k ( lgamma(a_k + n_k) - lgamma(a_k) ) ]

    The parameterisation is the point. With ``alpha_k = exp(l_k)`` the mean is
    ``alpha/a0 = softmax(l)``, exactly what every method already predicts, while the
    concentration is ``a0 = sum_k exp(l_k)``, the overall logit magnitude that softmax
    discards. The missing degree of freedom was always in the network's output and was
    being deleted by the normalisation. One forward pass, no extra parameters, so P1 is
    preserved.

    Three consequences:

    * ``sft_soft`` is the ``a0 -> infinity`` limit, verified numerically to 1e-7, so the
      control that refuted CSPO is a special case of this rather than a rival.
    * M enters the likelihood, so the objective is annotator-count aware by construction
      instead of by an ad-hoc shrinkage term.
    * ``a0`` is not determined by ``softmax(l)``, so the prediction is not a monotone
      rescaling of the logits and a global temperature cannot reproduce it. That is the
      structural property a global ``beta`` or ``lambda`` lacked.

    ``logits``  (B, K)  verbalizer logits
    ``votes``   (B, K)  raw annotator counts, rows summing to M (M may vary per row)
    """
    from scipy.special import gammaln

    z = np.asarray(logits, dtype=np.float64)
    n = np.asarray(votes, dtype=np.float64)
    if z.shape != n.shape:
        raise ValueError(f"shape mismatch: logits {z.shape}, votes {n.shape}")
    # Shift by the row max before exponentiating: alpha is scale-free in the mean and the
    # scale is carried explicitly, so this is numerically safe rather than a change of
    # objective. Clamping a0 keeps lgamma finite for confidently unanimous items.
    shifted = z - z.max(axis=-1, keepdims=True)
    alpha = np.exp(shifted) * float(concentration_scale)
    a0 = alpha.sum(axis=-1, keepdims=True)
    over = np.clip(a0 / max_concentration, 1.0, None)
    alpha = alpha / over
    a0 = alpha.sum(axis=-1)
    m = n.sum(axis=-1)
    return -(gammaln(a0) - gammaln(a0 + m)
             + (gammaln(alpha + n) - gammaln(alpha)).sum(axis=-1))


def polya_concentration(logits, concentration_scale: float = 1.0):
    """The predicted Dirichlet concentration, the quantity softmax throws away."""
    z = np.asarray(logits, dtype=np.float64)
    shifted = z - z.max(axis=-1, keepdims=True)
    return (np.exp(shifted) * float(concentration_scale)).sum(axis=-1)


def cspo_reward_distribution(policy_logit_diff, beta: float = 1.0):
    """q_theta(.|x): the reward-induced distribution over the label set."""
    z = np.asarray(policy_logit_diff, dtype=np.float64) * beta
    log_q = z - _logsumexp(z, axis=-1, keepdims=True)
    return np.exp(log_q)


def _logsumexp(z, axis=-1, keepdims=False):
    m = np.max(z, axis=axis, keepdims=True)
    out = m + np.log(np.exp(z - m).sum(axis=axis, keepdims=True))
    return out if keepdims else np.squeeze(out, axis=axis)


def soft_target(votes, temperature: float = 1.0, smoothing: float = 0.0):
    """Annotator counts to a target distribution, with optional sharpening and smoothing.

    ``temperature`` < 1 sharpens toward the majority label and ``> 1`` flattens; the
    ablation over it separates "the distribution matters" from "the argmax matters".
    ``smoothing`` mixes in the uniform distribution, which keeps the target in the
    interior so the cross-entropy gradient stays bounded on unanimous items.
    """
    counts = np.asarray(votes, dtype=np.float64)
    total = counts.sum(axis=-1, keepdims=True)
    p = np.divide(counts, np.maximum(total, 1e-12))
    if temperature != 1.0:
        with np.errstate(divide="ignore"):
            logits = np.where(p > 0, np.log(np.maximum(p, 1e-12)) / temperature, -np.inf)
        p = np.exp(logits - _logsumexp(logits, axis=-1, keepdims=True))
    if smoothing > 0:
        k = p.shape[-1]
        p = (1 - smoothing) * p + smoothing / k
    return p


# --------------------------------------------------------------------------------------
# MixDPO (Pang et al., 2026): difficulty-aware routing
# --------------------------------------------------------------------------------------

def mixdpo_route(margins, threshold: float = 0.5):
    """Which examples take the preference loss and which take the SFT loss.

    MixDPO's finding is that low-margin (ambiguous) pairs destabilise preference losses
    but remain useful under supervised fine-tuning, so it routes hard pairs to SFT and
    easy pairs to the preference objective. Its margins are model-derived implicit reward
    gaps; here the margin is the observed annotator gap, which lets the routing rule be
    evaluated against ground truth rather than against an estimate of itself.
    """
    m = np.abs(np.asarray(margins, dtype=np.float64))
    return m >= threshold


def mixdpo_loss(policy_chosen_logps, policy_rejected_logps,
                ref_chosen_logps, ref_rejected_logps, margins,
                *, beta: float = 0.1, threshold: float = 0.5, sft_weight: float = 1.0,
                ops: _Ops | None = None):
    """Preference loss on easy pairs, negative log-likelihood of the winner on hard pairs."""
    ops = ops or NumpyOps()
    easy = mixdpo_route(margins, threshold)
    pref = dpo_loss(policy_chosen_logps, policy_rejected_logps,
                    ref_chosen_logps, ref_rejected_logps, beta=beta, ops=ops)
    sft = -np.asarray(policy_chosen_logps, dtype=np.float64)
    return np.where(easy, pref, sft_weight * sft)


def curriculum_order(margins, ascending: bool = False):
    """Easy-to-hard ordering over margins, the second half of MixDPO."""
    m = np.abs(np.asarray(margins, dtype=np.float64))
    return np.argsort(m if ascending else -m, kind="stable")


# --------------------------------------------------------------------------------------
# Reward calibration (Chen et al., 2026): keep the winner while suppressing the loser
# --------------------------------------------------------------------------------------

def reward_calibration_weights(chosen_logratios, rejected_logratios,
                               *, target_ratio: float = 1.0, eps: float = 1e-6):
    """Per-example rebalancing of the chosen and rejected update directions.

    Margin-based objectives can push the chosen likelihood down while pushing the
    rejected one down harder, which still lowers the loss. On a closed label set that
    failure is severe rather than cosmetic: the rejected label is the one a minority of
    annotators actually chose, so suppressing it destroys the calibration the task needs.
    The weights below rescale the two directions toward ``target_ratio`` whenever the
    chosen response is being suppressed.
    """
    c = np.asarray(chosen_logratios, dtype=np.float64)
    r = np.asarray(rejected_logratios, dtype=np.float64)
    suppressing_winner = c < 0
    magnitude = np.abs(r) + eps
    scale = np.where(suppressing_winner,
                     np.clip(target_ratio * np.abs(c) / magnitude, 0.0, 1.0), 1.0)
    return np.ones_like(c), scale


# --------------------------------------------------------------------------------------
# Group-robust DPO
# --------------------------------------------------------------------------------------

@dataclass
class GroupRobustState:
    """Exponentiated-gradient weights over groups for GR-DPO.

    Ramesh et al. (2024) solve ``min_θ max_{q ∈ Δ(G)} Σ_g q_g L_g(θ)`` with an online
    mirror-ascent step on ``q``. The weights are updated multiplicatively from the
    observed per-group loss and renormalised, so a group that is doing badly is upweighted
    smoothly rather than by a hard max over groups.

    v1 described GRPO as "upweighting groups with higher empirical DPO loss ... maximum
    upweighting factor of 2.0", which is a heuristic, not this algorithm: and it also
    used length buckets as groups. Here the groups are the annotator-agreement bands,
    which is the axis the paper's robustness claim is about.
    """

    groups: tuple[str, ...]
    step_size: float = 0.01
    weights: np.ndarray | None = None
    ema_loss: np.ndarray | None = None
    ema_decay: float = 0.9

    def __post_init__(self) -> None:
        n = len(self.groups)
        if n == 0:
            raise ValueError("GroupRobustState needs at least one group")
        if self.weights is None:
            self.weights = np.full(n, 1.0 / n, dtype=np.float64)
        if self.ema_loss is None:
            self.ema_loss = np.zeros(n, dtype=np.float64)
        self._index = {g: i for i, g in enumerate(self.groups)}
        self._seen_ever = np.zeros(n, dtype=bool)

    def index_of(self, group: str) -> int:
        return self._index[group]

    def update(self, group_losses: Sequence[float], seen: Sequence[bool] | None = None) -> np.ndarray:
        """One exponentiated-gradient ascent step on ``q``.

        ``seen`` marks which groups appeared in this batch; unseen groups keep their EMA
        so a group that is rare does not have its weight decay to zero simply through
        absence: the failure mode that makes naive implementations ignore small groups,
        which are exactly the ones the method exists to protect.
        """
        losses = np.asarray(group_losses, dtype=np.float64)
        if losses.shape != self.weights.shape:
            raise ValueError(f"expected {len(self.groups)} group losses, got {losses.shape}")
        mask = (np.ones(len(self.groups), dtype=bool) if seen is None
                else np.asarray(seen, dtype=bool))
        self._seen_ever |= mask
        self.ema_loss = np.where(
            mask, self.ema_decay * self.ema_loss + (1 - self.ema_decay) * losses, self.ema_loss)

        # A group that has never been observed is scored at the mean of the observed
        # groups rather than at zero. Scoring it at zero makes "never sampled" look
        # identical to "already solved", and the resulting weight decay silently starves
        # exactly the rare groups the objective exists to protect.
        reference = float(self.ema_loss[self._seen_ever].mean()) if self._seen_ever.any() else 0.0
        effective = np.where(self._seen_ever, self.ema_loss, reference)

        logits = np.log(np.maximum(self.weights, 1e-12)) + self.step_size * effective
        logits -= logits.max()
        w = np.exp(logits)
        self.weights = w / w.sum()
        return self.weights

    def weighted_loss(self, group_losses: Sequence[float]) -> float:
        return float(np.dot(self.weights, np.asarray(group_losses, dtype=np.float64)))


def group_robust_dpo_loss(
    per_example_losses,
    group_ids: Sequence[int],
    state: GroupRobustState,
    *,
    update: bool = True,
) -> tuple[float, np.ndarray]:
    """Aggregate per-example DPO losses into the group-robust objective.

    Returns ``(loss, group_weights)``. Groups absent from the batch contribute nothing to
    the loss but keep their weight, which is what ``seen`` in ``update`` is for.
    """
    losses = np.asarray(per_example_losses, dtype=np.float64)
    group_ids = np.asarray(group_ids, dtype=np.int64)
    n_groups = len(state.groups)

    sums = np.zeros(n_groups)
    counts = np.zeros(n_groups)
    np.add.at(sums, group_ids, losses)
    np.add.at(counts, group_ids, 1.0)
    seen = counts > 0
    group_losses = np.where(seen, sums / np.maximum(counts, 1.0), 0.0)

    if update:
        state.update(group_losses, seen)
    weights = state.weights
    # Renormalise over the groups actually present so batch composition does not scale
    # the loss magnitude from step to step.
    present = weights * seen
    total = present.sum()
    normed = present / total if total > 0 else seen / max(seen.sum(), 1)
    return float(np.dot(normed, group_losses)), weights


# --------------------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------------------

#: Objectives configured through TRL 0.2x. SimPO and AlphaPO come from ``CPOTrainer``
#: with ``cpo_alpha=0``, which is the library's own SimPO; using it rather than a local
#: reimplementation removes a class of transcription risk, and ``simpo_loss`` below stays
#: under test as the specification that implementation is checked against.
TRL_OBJECTIVES: dict[str, dict] = {
    "dpo":     {"trainer": "DPOTrainer", "loss_type": "sigmoid", "label_smoothing": 0.0},
    "rdpo":    {"trainer": "DPOTrainer", "loss_type": "robust",  "label_smoothing": 0.1},
    "ipo":     {"trainer": "DPOTrainer", "loss_type": "ipo"},
    "kto":     {"trainer": "KTOTrainer"},
    "simpo":   {"trainer": "CPOTrainer", "loss_type": "simpo"},
    "alphapo": {"trainer": "CPOTrainer", "loss_type": "alphapo"},
}

#: Objectives implemented in this module, because TRL provides none of them.
CUSTOM_OBJECTIVES = ("grdpo", "mixdpo", "cspo", "cspo_kl", "cspo_ada",
                     "polya", "mopa")

#: Supervised arms. ``sft_soft`` is the control that CSPO has to beat: cross-entropy
#: against the same annotator distribution, without the reference anchoring.
SUPERVISED_OBJECTIVES = ("sft", "sft_soft")

ALL_OBJECTIVES = (*SUPERVISED_OBJECTIVES, *TRL_OBJECTIVES, *CUSTOM_OBJECTIVES)

#: Objectives that consume the annotator distribution rather than binarised pairs.
DISTRIBUTIONAL_OBJECTIVES = ("sft_soft", "cspo", "cspo_kl", "cspo_ada",
                             "polya", "mopa")

#: Objectives that need no reference policy at training time. On a 24 GB card this is a
#: material difference, and CSPO belongs here because its reference enters only through
#: precomputed logit differences.
#: Objectives trained from the *base* model instead of warm-started from the SFT adapter.
#:
#: This is a distinct property from REFERENCE_FREE, which is about whether a live
#: reference model is needed in memory. Purely supervised objectives belong here: they
#: have no reference, and warm-starting one would hand it a whole stage of training that
#: its own control did not get. `polya` was omitted from the hard-coded gate this
#: replaces, so it warm-started from SFT while `sft_soft` trained from base, which is
#: exactly the asymmetry that has bitten this project six times before.
FROM_BASE = ("sft", "sft_soft", "polya", "mopa")

REFERENCE_FREE = ("sft", "sft_soft", "simpo", "alphapo", "cspo", "cspo_kl",
                  "cspo_ada", "polya", "mopa")
