"""Verify each objective against its published equation.

v1 shipped a KTO that was ORPO and a SimPO with neither length normalisation nor a
margin (AUDIT.md S1-4, S1-5). Losses get tested here so that cannot recur silently.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from importlib.util import find_spec

needs_torch = pytest.mark.skipif(find_spec("torch") is None, reason="torch not installed")

from sentalign.train.objectives import (GroupRobustState, cspo_loss,
                                        cspo_reward_distribution, curriculum_order,
                                        dpo_loss, dr_dpo_loss, group_robust_dpo_loss,
                                        ipo_loss, mixdpo_loss, mixdpo_route,
                                        reward_calibration_weights, robust_dpo_loss,
                                        simpo_loss, soft_target)


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


# --------------------------------------------------------------------------------------
# DPO
# --------------------------------------------------------------------------------------

def test_dpo_matches_closed_form():
    pc, pr, rc, rr = -1.0, -3.0, -1.5, -2.0
    beta = 0.1
    delta = (pc - rc) - (pr - rr)          # 0.5 - (-1.0) = 1.5
    expected = -math.log(sigmoid(beta * delta))
    got = float(dpo_loss(np.array([pc]), np.array([pr]),
                         np.array([rc]), np.array([rr]), beta=beta)[0])
    assert got == pytest.approx(expected, rel=1e-10)


def test_dpo_is_decreasing_in_the_margin():
    """A larger policy-vs-reference advantage for the chosen answer must cost less."""
    base = dict(policy_rejected_logps=np.array([-3.0]), ref_chosen_logps=np.array([-2.0]),
                ref_rejected_logps=np.array([-2.0]), beta=0.1)
    losses = [float(dpo_loss(np.array([v]), **base)[0]) for v in (-4.0, -2.0, -1.0, 0.0)]
    assert losses == sorted(losses, reverse=True)


def test_cdpo_label_smoothing_bounds_the_loss():
    """cDPO must not diverge as the margin goes to -inf; plain DPO must."""
    far_wrong = dict(policy_chosen_logps=np.array([-50.0]),
                     policy_rejected_logps=np.array([0.0]),
                     ref_chosen_logps=np.array([0.0]),
                     ref_rejected_logps=np.array([0.0]), beta=1.0)
    plain = float(dpo_loss(**far_wrong)[0])
    smoothed = float(dpo_loss(**far_wrong, label_smoothing=0.1)[0])
    assert plain > 40
    assert smoothed < plain
    # Both terms of cDPO are negative log-sigmoids, so the loss stays positive.
    assert smoothed > 0


# --------------------------------------------------------------------------------------
# rDPO -- distinct from cDPO
# --------------------------------------------------------------------------------------

def test_rdpo_is_not_cdpo():
    """The sign of the second term and the 1/(1-2e) rescaling separate them."""
    args = dict(policy_chosen_logps=np.array([-1.0]), policy_rejected_logps=np.array([-3.0]),
                ref_chosen_logps=np.array([-1.5]), ref_rejected_logps=np.array([-2.0]),
                beta=0.1)
    cdpo = float(dpo_loss(**args, label_smoothing=0.1)[0])
    rdpo = float(robust_dpo_loss(**args, epsilon=0.1)[0])
    assert cdpo != pytest.approx(rdpo, rel=1e-6)


def test_rdpo_closed_form():
    pc, pr, rc, rr, beta, eps = -1.0, -3.0, -1.5, -2.0, 0.1, 0.2
    delta = (pc - rc) - (pr - rr)
    expected = (-(1 - eps) * math.log(sigmoid(beta * delta))
                + eps * math.log(sigmoid(-beta * delta))) / (1 - 2 * eps)
    got = float(robust_dpo_loss(np.array([pc]), np.array([pr]), np.array([rc]),
                                np.array([rr]), beta=beta, epsilon=eps)[0])
    assert got == pytest.approx(expected, rel=1e-10)


def test_rdpo_reduces_to_dpo_at_zero_noise():
    args = dict(policy_chosen_logps=np.array([-1.0, -2.0]),
                policy_rejected_logps=np.array([-3.0, -1.0]),
                ref_chosen_logps=np.array([-1.5, -1.5]),
                ref_rejected_logps=np.array([-2.0, -2.0]), beta=0.1)
    assert np.allclose(robust_dpo_loss(**args, epsilon=0.0), dpo_loss(**args))


def test_rdpo_rejects_invalid_epsilon():
    args = (np.array([-1.0]), np.array([-2.0]), np.array([-1.0]), np.array([-2.0]))
    with pytest.raises(ValueError):
        robust_dpo_loss(*args, epsilon=0.5)


# --------------------------------------------------------------------------------------
# IPO
# --------------------------------------------------------------------------------------

def test_ipo_is_minimised_at_the_target_margin():
    """IPO's optimum is Delta = 1/(2*tau), not Delta -> inf. This is the whole point."""
    tau = 0.1
    target = 1.0 / (2 * tau)
    at_target = float(ipo_loss(np.array([target]), np.array([0.0]),
                               np.array([0.0]), np.array([0.0]), tau=tau)[0])
    beyond = float(ipo_loss(np.array([target * 3]), np.array([0.0]),
                            np.array([0.0]), np.array([0.0]), tau=tau)[0])
    assert at_target == pytest.approx(0.0, abs=1e-12)
    assert beyond > at_target       # overshooting is penalised; DPO would reward it


def test_ipo_does_not_saturate():
    """Unlike DPO, IPO's gradient keeps growing, so it cannot be driven to infinity."""
    tau = 0.1
    losses = [float(ipo_loss(np.array([d]), np.array([0.0]), np.array([0.0]),
                             np.array([0.0]), tau=tau)[0]) for d in (20.0, 40.0, 80.0)]
    gaps = np.diff(losses)
    assert (gaps > 0).all() and gaps[1] > gaps[0]


# --------------------------------------------------------------------------------------
# SimPO -- the two components v1 dropped
# --------------------------------------------------------------------------------------

def test_simpo_closed_form_with_length_normalisation_and_margin():
    pc, pr = -4.0, -9.0
    lc, lr = 2, 3
    beta, gamma = 2.0, 0.5
    expected = -math.log(sigmoid(beta * pc / lc - beta * pr / lr - gamma))
    got = float(simpo_loss(np.array([pc]), np.array([pr]),
                           np.array([lc]), np.array([lr]), beta=beta, gamma=gamma)[0])
    assert got == pytest.approx(expected, rel=1e-10)


def test_simpo_length_normalisation_changes_the_ranking():
    """A long chosen answer and a short rejected one flip sign under normalisation.

    This is the concrete cost of v1's omission: without 1/|y| the objective simply
    prefers shorter sequences, which is the length bias SimPO was built to remove.
    """
    pc, pr = -6.0, -4.0        # chosen has lower total logprob ...
    lc, lr = 6, 2              # ... but far higher per-token logprob (-1.0 vs -2.0)
    normalised = float(simpo_loss(np.array([pc]), np.array([pr]), np.array([lc]),
                                  np.array([lr]), beta=2.0, gamma=0.0)[0])
    unnormalised = float(simpo_loss(np.array([pc]), np.array([pr]), np.array([1]),
                                    np.array([1]), beta=2.0, gamma=0.0)[0])
    assert normalised < math.log(2) < unnormalised


def test_simpo_margin_shifts_the_decision_boundary():
    """At zero reward difference the loss must equal -log sigma(-gamma), not -log 0.5."""
    for gamma in (0.0, 0.5, 1.0):
        loss = float(simpo_loss(np.array([-2.0]), np.array([-2.0]), np.array([1]),
                                np.array([1]), beta=2.0, gamma=gamma)[0])
        assert loss == pytest.approx(-math.log(sigmoid(-gamma)), rel=1e-10)


def test_simpo_needs_no_reference_model():
    """Signature check: SimPO is reference-free, so no reference logps are accepted."""
    import inspect

    params = set(inspect.signature(simpo_loss).parameters)
    assert not {p for p in params if "ref" in p}


# --------------------------------------------------------------------------------------
# Dr. DPO
# --------------------------------------------------------------------------------------

def test_dr_dpo_interpolates_between_worst_case_and_mean():
    pc = np.array([-1.0, -1.0, -8.0])
    pr = np.array([-3.0, -3.0, -1.0])
    rc = np.array([-2.0, -2.0, -2.0])
    rr = np.array([-2.0, -2.0, -2.0])
    per_example = dpo_loss(pc, pr, rc, rr, beta=1.0)
    small = dr_dpo_loss(pc, pr, rc, rr, beta=1.0, beta_prime=0.05)
    large = dr_dpo_loss(pc, pr, rc, rr, beta=1.0, beta_prime=100.0)
    assert small == pytest.approx(per_example.max(), rel=0.1)
    assert large == pytest.approx(per_example.mean(), rel=0.05)
    assert small > large


# --------------------------------------------------------------------------------------
# GR-DPO
# --------------------------------------------------------------------------------------

def test_group_weights_shift_towards_the_worst_group():
    state = GroupRobustState(groups=("easy", "hard"), step_size=1.0)
    start = state.weights.copy()
    for _ in range(20):
        state.update([0.1, 0.9])
    assert state.weights[1] > start[1]
    assert state.weights[1] > state.weights[0]
    assert state.weights.sum() == pytest.approx(1.0)


def test_absent_groups_keep_their_weight():
    """A rare group must not have its weight decay to zero simply by not appearing."""
    state = GroupRobustState(groups=("a", "b", "c"), step_size=0.5)
    for _ in range(30):
        group_robust_dpo_loss(np.array([1.0, 1.0]), [0, 1], state)
    assert state.weights[2] > 0.05


def test_group_robust_loss_is_between_min_and_max_group_loss():
    state = GroupRobustState(groups=("a", "b"), step_size=0.1)
    losses = np.array([0.2, 0.2, 1.8, 1.8])
    loss, weights = group_robust_dpo_loss(losses, [0, 0, 1, 1], state)
    assert 0.2 <= loss <= 1.8
    assert weights.sum() == pytest.approx(1.0)


def test_uniform_weights_recover_the_mean_over_groups():
    state = GroupRobustState(groups=("a", "b"), step_size=0.0)
    loss, _ = group_robust_dpo_loss(np.array([0.0, 0.0, 2.0, 2.0]), [0, 0, 1, 1],
                                    state, update=False)
    assert loss == pytest.approx(1.0)


# --------------------------------------------------------------------------------------
# CSPO: the three propositions the paper states
# --------------------------------------------------------------------------------------

def test_p1_full_vocabulary_normalizer_cancels():
    """P1: the objective depends only on the K logit differences.

    Adding any constant to every policy logit (equivalently, changing the full-vocabulary
    normalizer) must leave the loss unchanged. This is what licenses reading K logits off
    one forward pass instead of computing K normalised sequence log-probabilities.
    """
    diff = np.array([[1.5, -0.5, 0.25, 0.0]])
    target = np.array([[0.6, 0.2, 0.2, 0.0]])
    base = cspo_loss(diff, target, beta=1.0)
    for shift in (-5.0, 0.3, 12.0):
        assert cspo_loss(diff + shift, target, beta=1.0) == pytest.approx(base)


def test_p2_cspo_reduces_exactly_to_dpo():
    """P2: K=2 with a one-hot target recovers the DPO loss exactly."""
    beta = 0.1
    # single-token verbalizers, so the sequence log-prob is the token log-prob
    pc, pr, rc, rr = -1.0, -3.0, -1.5, -2.0
    logit_diff = np.array([[pc - rc, pr - rr]])         # chosen first
    one_hot = np.array([[1.0, 0.0]])

    got = float(cspo_loss(logit_diff, one_hot, beta=beta)[0])
    expected = float(dpo_loss(np.array([pc]), np.array([pr]),
                              np.array([rc]), np.array([rr]), beta=beta)[0])
    assert got == pytest.approx(expected, rel=1e-12)


def test_p2_holds_across_random_instances():
    rng = np.random.default_rng(11)
    for _ in range(50):
        pc, pr, rc, rr = rng.normal(0, 2, 4)
        beta = float(rng.uniform(0.05, 1.0))
        got = float(cspo_loss(np.array([[pc - rc, pr - rr]]), np.array([[1.0, 0.0]]),
                              beta=beta)[0])
        expected = float(dpo_loss(np.array([pc]), np.array([pr]), np.array([rc]),
                                  np.array([rr]), beta=beta)[0])
        assert got == pytest.approx(expected, rel=1e-10)


def test_p3_minimiser_matches_the_annotator_distribution():
    """P3: the loss is minimised where q_theta equals p_human, and the reward at that
    point is the log agreement rate up to an additive constant."""
    p = np.array([[0.6, 0.2, 0.2]])
    optimal = np.log(p) / 1.0                      # beta = 1 => reward = log p + c
    at_optimum = float(cspo_loss(optimal, p, beta=1.0)[0])
    assert np.allclose(cspo_reward_distribution(optimal, 1.0), p, atol=1e-9)

    rng = np.random.default_rng(12)
    for _ in range(200):
        perturbed = optimal + rng.normal(0, 0.3, optimal.shape)
        assert float(cspo_loss(perturbed, p, beta=1.0)[0]) >= at_optimum - 1e-9

    # The minimum equals the entropy of the human distribution, not zero: a model cannot
    # do better than reproducing genuine annotator disagreement.
    assert at_optimum == pytest.approx(float(-(p * np.log(p)).sum()), rel=1e-9)


def test_cspo_handles_zero_support_and_no_majority_items():
    """Labels no annotator chose contribute nothing; items with no majority are valid."""
    no_majority = np.array([[0.4, 0.4, 0.2, 0.0]])          # no label reaches 3/5
    loss = cspo_loss(np.array([[0.1, 0.1, 0.0, -9.0]]), no_majority, beta=1.0)
    assert np.isfinite(loss).all()

    # A label with zero human support may have any reward without affecting the loss
    # through its own term; it enters only through the normalizer.
    a = cspo_loss(np.array([[1.0, 0.0, 0.0, -50.0]]), no_majority, beta=1.0)
    b = cspo_loss(np.array([[1.0, 0.0, 0.0, -80.0]]), no_majority, beta=1.0)
    assert float(a[0]) == pytest.approx(float(b[0]), abs=1e-9)


def test_cspo_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="shape mismatch"):
        cspo_loss(np.zeros((2, 4)), np.zeros((2, 3)))


def test_soft_target_temperature_and_smoothing():
    votes = np.array([[3, 1, 1, 0]])
    assert np.allclose(soft_target(votes), [[0.6, 0.2, 0.2, 0.0]])

    sharp = soft_target(votes, temperature=0.25)
    flat = soft_target(votes, temperature=4.0)
    assert sharp[0, 0] > 0.6 > flat[0, 0]
    assert np.allclose(sharp.sum(-1), 1.0) and np.allclose(flat.sum(-1), 1.0)

    smoothed = soft_target(votes, smoothing=0.2)
    assert smoothed[0, 3] > 0.0                       # zero-support label gets mass
    assert np.allclose(smoothed.sum(-1), 1.0)

    # In the limit of extreme sharpening the target approaches the majority one-hot,
    # which is the ablation that isolates "distribution" from "argmax".
    assert soft_target(votes, temperature=0.01)[0, 0] == pytest.approx(1.0, abs=1e-6)


# --------------------------------------------------------------------------------------
# MixDPO and reward calibration
# --------------------------------------------------------------------------------------

def test_mixdpo_routes_hard_pairs_to_the_supervised_loss():
    margins = np.array([1.0, 0.2])                    # one easy, one hard
    routed = mixdpo_route(margins, threshold=0.5)
    assert routed.tolist() == [True, False]

    pc = np.array([-1.0, -1.0]); pr = np.array([-3.0, -3.0])
    rc = np.array([-1.5, -1.5]); rr = np.array([-2.0, -2.0])
    loss = mixdpo_loss(pc, pr, rc, rr, margins, beta=0.1, threshold=0.5)
    pref_only = dpo_loss(pc, pr, rc, rr, beta=0.1)
    assert loss[0] == pytest.approx(pref_only[0])     # easy pair keeps the preference loss
    assert loss[1] == pytest.approx(1.0)              # hard pair becomes -log p(chosen)


def test_curriculum_orders_easy_to_hard():
    order = curriculum_order(np.array([0.2, 1.0, 0.6]))
    assert order.tolist() == [1, 2, 0]


def test_reward_calibration_only_damps_when_the_winner_is_suppressed():
    # winner rising: no rescaling
    _, scale = reward_calibration_weights(np.array([0.5]), np.array([-0.5]))
    assert scale[0] == pytest.approx(1.0)

    # winner being pushed down while the loser is pushed down much harder: damp
    _, scale = reward_calibration_weights(np.array([-0.1]), np.array([-2.0]))
    assert 0.0 <= scale[0] < 1.0


# --------------------------------------------------------------------------------------
# CSPO-KL: the same anchoring, outside the softmax
# --------------------------------------------------------------------------------------

def test_cspo_kl_recovers_soft_cross_entropy_when_the_penalty_is_off():
    """lam=0 must be exactly sft_soft, or the variant is not a clean ablation of one term."""
    import numpy as np

    from sentalign.train.objectives import cspo_kl_loss, cspo_loss

    rng = np.random.default_rng(0)
    policy = rng.normal(size=(8, 4))
    reference = rng.normal(size=(8, 4))
    p_human = rng.dirichlet(np.ones(4), size=8)

    # cspo_loss with a zero reference and beta=1 is plain soft cross-entropy.
    soft = cspo_loss(policy, p_human, beta=1.0)
    assert np.allclose(cspo_kl_loss(policy, reference, p_human, lam=0.0), soft)


def test_cspo_kl_penalty_is_zero_only_at_the_reference_and_positive_elsewhere():
    import numpy as np

    from sentalign.train.objectives import cspo_kl_loss

    rng = np.random.default_rng(1)
    policy = rng.normal(size=(16, 4))
    p_human = rng.dirichlet(np.ones(4), size=16)

    at_reference = cspo_kl_loss(policy, policy, p_human, lam=3.0)
    without = cspo_kl_loss(policy, policy, p_human, lam=0.0)
    assert np.allclose(at_reference, without), "KL(q||q) must vanish"

    away = cspo_kl_loss(policy, rng.normal(size=(16, 4)), p_human, lam=3.0)
    assert (away >= without - 1e-12).all(), "KL is non-negative, so the loss cannot drop"
    assert (away > without + 1e-6).any()


def test_cspo_kl_optimum_is_the_annotator_distribution_itself():
    """The point of the variant. cspo_loss drives softmax(l) to pi_ref * p**(1/beta);
    this one drives softmax(l) to p_human, which is what evaluation scores."""
    import numpy as np

    from sentalign.train.objectives import cspo_kl_loss

    p_human = np.array([[0.5, 0.3, 0.15, 0.05]])
    reference = np.array([[2.0, -1.0, 0.5, 0.0]])          # deliberately not p_human

    exact = np.log(p_human)                                 # softmax(exact) == p_human
    best = cspo_kl_loss(exact, reference, p_human, lam=0.0)
    for _ in range(200):
        other = exact + np.random.default_rng().normal(scale=0.3, size=exact.shape)
        assert cspo_kl_loss(other, reference, p_human, lam=0.0) >= best - 1e-9


def test_the_adaptive_anchor_nests_both_arms_it_must_be_compared_against():
    """lam0=0 must be sft_soft and a constant-entropy batch must be cspo_kl, or the
    ablation that could refute this objective is not actually its ablation."""
    import numpy as np

    from sentalign.train.objectives import (adaptive_lambda, cspo_ada_loss,
                                            cspo_kl_loss, soft_target)

    rng = np.random.default_rng(0)
    z, r = rng.normal(size=(64, 4)), rng.normal(size=(64, 4))
    p = rng.dirichlet(np.ones(4), size=64)

    assert np.allclose(cspo_ada_loss(z, r, p, 0.0), cspo_kl_loss(z, r, p, 0.0))

    # Every row equally ambiguous: the adaptive weight is then a constant, so the two
    # objectives must agree for the matching constant.
    flat = np.tile(np.array([0.5, 0.5, 0.0, 0.0]), (64, 1))
    lam = adaptive_lambda(flat, 1.0)
    assert np.allclose(lam, lam[0])
    assert np.allclose(cspo_ada_loss(z, r, flat, 1.0), cspo_kl_loss(z, r, flat, lam[0]))


def test_the_anchor_decays_monotonically_with_ambiguity():
    """The mechanism: anchor hard where the hard-label reference is trustworthy, not at
    all where annotators disagree and the reference is confidently wrong."""
    import numpy as np

    from sentalign.train.objectives import adaptive_lambda

    p = np.array([[1.0, 0.0, 0.0],
                  [0.8, 0.2, 0.0],
                  [0.5, 0.5, 0.0],
                  [1 / 3, 1 / 3, 1 / 3]])
    lam = adaptive_lambda(p, 1.0)
    assert lam[0] == pytest.approx(1.0)
    assert lam[-1] == pytest.approx(0.0, abs=1e-9)
    assert np.all(np.diff(lam) < 0), f"anchor must decay with ambiguity, got {lam}"


def test_the_adaptive_weight_is_not_a_global_rescaling():
    """The whole point. A global scalar yields a fixed monotone rescaling of the
    reference logits, which is indistinguishable from a temperature-scaled reference.
    An item-conditional weight must actually vary across a batch of mixed ambiguity."""
    import numpy as np

    from sentalign.train.objectives import adaptive_lambda

    mixed = np.array([[1.0, 0.0, 0.0], [0.4, 0.35, 0.25]])
    lam = adaptive_lambda(mixed, 1.0)
    assert lam.std() > 0.3, "the anchor collapsed to a constant"


def test_empirical_bayes_shrinkage_reduces_target_variance_and_nests_the_raw_target():
    """alpha=0 must leave the frequencies untouched; alpha>0 must move a 5-vote target
    toward the prior and strictly lower its sampling variance."""
    import numpy as np

    from sentalign.train.objectives import empirical_bayes_target

    votes = np.array([[4.0, 1.0, 0.0]])
    assert np.allclose(empirical_bayes_target(votes, 0.0), [[0.8, 0.2, 0.0]])

    shrunk = empirical_bayes_target(votes, 2.0)[0]
    assert shrunk.sum() == pytest.approx(1.0)
    assert shrunk[2] > 0.0, "a zero-count label must get non-zero mass under a prior"
    assert shrunk[0] < 0.8, "the mode must move toward the prior"

    # Variance: resample a known p at M=5 and check shrinkage lowers the spread.
    rng = np.random.default_rng(0)
    p = np.array([0.45, 0.35, 0.20])
    draws = np.array([rng.multinomial(5, p) for _ in range(4000)], dtype=float)
    raw = empirical_bayes_target(draws, 0.0)
    eb = empirical_bayes_target(draws, 2.0)
    assert eb.var(axis=0).sum() < raw.var(axis=0).sum()


def test_polya_recovers_soft_cross_entropy_in_the_high_concentration_limit():
    """sft_soft is the a0 -> infinity limit, so the control that refuted CSPO is a
    special case of this objective rather than a rival. If this drifts, the ablation
    stops being an ablation."""
    import numpy as np

    from sentalign.train.objectives import polya_loss

    z = np.array([[2.0, 1.0, 0.0]])
    n = np.array([[3.0, 1.0, 1.0]])
    p = np.exp(z - z.max()) / np.exp(z - z.max()).sum()
    ce = float(-(n * np.log(p)).sum())
    got = float(polya_loss(z, n, concentration_scale=1e4, max_concentration=1e12)[0])
    assert got == pytest.approx(ce, abs=1e-2), (got, ce)


def test_polya_is_annotator_count_aware_where_frequency_matching_is_not():
    """The same observed frequencies from 5 and from 100 annotators are different
    evidence. Cross-entropy to p_hat cannot tell them apart; this must."""
    import numpy as np

    from sentalign.train.objectives import polya_loss

    freq = np.array([[0.6, 0.3, 0.1]])
    z = np.array([[0.5, 0.0, -0.5]])
    per_annotator = [float(polya_loss(z, freq * m)[0]) / m for m in (5, 20, 100)]
    assert per_annotator[0] > per_annotator[1] > per_annotator[2], per_annotator


def test_polya_separates_items_with_equal_means_but_different_agreement():
    """The capability softmax provably lacks: one logit vector cannot encode both the
    mean and an independent spread. Fit the concentration from 5-annotator counts drawn
    from two populations sharing a mean, and the recovered values must differ sharply."""
    import numpy as np

    from sentalign.train.objectives import polya_loss

    rng = np.random.default_rng(0)
    mu = np.array([0.4, 0.35, 0.25])
    z = np.log(mu)[None, :]

    def best_scale(true_a0):
        counts = np.array([rng.multinomial(5, rng.dirichlet(mu * true_a0))
                           for _ in range(500)], dtype=float)
        zz = np.repeat(z, len(counts), axis=0)
        return min((float(polya_loss(zz, counts, concentration_scale=s,
                                     max_concentration=1e9).mean()), s)
                   for s in np.logspace(-1, 3, 40))[1]

    volatile, consistent = best_scale(2.0), best_scale(200.0)
    assert volatile < consistent / 10, (volatile, consistent)


def test_the_concentration_is_the_magnitude_softmax_discards():
    """Two logit vectors with the same softmax but different scale must give the same
    mean and different concentration. That difference is the new degree of freedom."""
    import numpy as np

    from sentalign.train.objectives import polya_concentration

    small = np.array([[1.0, 0.5, 0.0]])
    large = small * 4.0
    soft = lambda z: np.exp(z - z.max()) / np.exp(z - z.max()).sum()
    assert not np.allclose(soft(small[0]), soft(large[0]))   # sanity: scale changes mean
    # Same mean, different scale, achieved by adding a constant (softmax-invariant).
    shifted = small + 7.0
    assert np.allclose(soft(small[0]), soft(shifted[0]))
    assert polya_concentration(small) == pytest.approx(polya_concentration(shifted))
    assert polya_concentration(small, 5.0) > polya_concentration(small, 1.0)


def test_the_mixture_reproduces_its_own_control_at_one_perspective():
    """A=1 initialised from the verbalizer rows must equal softmax(U h) exactly. If it
    does not, sft_soft stops being nested inside mopa and the ablation is not an
    ablation."""
    import numpy as np

    from sentalign.train.mixture import _logsumexp, mixture_log_probs

    rng = np.random.default_rng(0)
    h, u = rng.normal(size=(6, 8)), rng.normal(size=(3, 8))
    got = mixture_log_probs(h, u[None, :, :], np.zeros((1, 8)))
    want = (h @ u.T) - _logsumexp(h @ u.T, axis=-1, keepdims=True)
    assert np.allclose(got, want)


def test_no_temperature_reproduces_the_mixture():
    """The property every previous objective lacked. cspo, cspo_kl and polya all emit a
    monotone rescaling of one logit vector, which a temperature reproduces exactly; that
    is why all three collapsed to calibrated SFT. A mixture of softmaxes does not, and
    this pins it: the best temperature fit must leave real divergence, and the per-item
    entropy ordering must not be recoverable at any temperature."""
    import numpy as np

    from sentalign.evaluate.metrics import jensen_shannon
    from sentalign.train.mixture import _logsumexp, mixture_log_probs

    rng = np.random.default_rng(0)
    h, u = rng.normal(size=(300, 16)), rng.normal(size=(3, 16))
    base = h @ u.T
    weights = np.stack([u + rng.normal(size=u.shape) * 0.6 for _ in range(4)])
    q = np.exp(mixture_log_probs(h, weights, rng.normal(size=(4, 16)) * 0.5))

    def at(t):
        z = base / t
        return np.exp(z - _logsumexp(z, axis=-1, keepdims=True))

    best = min(float(jensen_shannon(at(t), q).mean()) for t in np.logspace(-1.5, 1.5, 200))
    assert best > 0.01, f"a temperature reproduced the mixture (JSD {best})"

    def rank(a, b):
        ra, rb = (np.argsort(np.argsort(x)).astype(float) for x in (a, b))
        ra -= ra.mean(); rb -= rb.mean()
        return float((ra * rb).sum() / np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))

    ent = lambda p: -(np.clip(p, 1e-12, 1) * np.log(np.clip(p, 1e-12, 1))).sum(-1)
    assert max(rank(ent(at(t)), ent(q)) for t in (0.5, 1.0, 2.0, 5.0)) < 0.95


@needs_torch
def test_the_torch_head_matches_the_reference_and_starts_as_the_base_model():
    import numpy as np
    import torch

    from sentalign.train.mixture import _logsumexp, make_mixture_head, mixture_log_probs

    rng = np.random.default_rng(0)
    rows = rng.normal(size=(3, 8))
    h = rng.normal(size=(5, 8))

    one = make_mixture_head(8, rows, n_perspectives=1)
    got = one(torch.tensor(h, dtype=torch.float32)).detach().numpy()
    want = (h @ rows.T) - _logsumexp(h @ rows.T, axis=-1, keepdims=True)
    assert np.allclose(got, want, atol=1e-5), "A=1 head is not the base readout"

    four = make_mixture_head(8, rows, n_perspectives=4, seed=0)
    torch_out = four(torch.tensor(h, dtype=torch.float32)).detach().numpy()
    ref = mixture_log_probs(h, four.weight.detach().numpy(),
                            four.gate.detach().numpy(),
                            bias=four.bias.detach().numpy(),
                            gate_bias=four.gate_bias.detach().numpy())
    assert np.allclose(torch_out, ref, atol=1e-5)
    assert np.allclose(np.exp(torch_out).sum(1), 1.0, atol=1e-5)
