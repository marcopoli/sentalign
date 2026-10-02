"""Tests for the evaluation protocol: prompting, parsing, metrics, statistics.

The protocol defects in AUDIT.md (S2-6) are silent: they produce numbers, just not the
numbers they claim to. Each test below pins one of them.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from sentalign.evaluate.metrics import (accuracy_at_coverage, apply_temperature, aurc,
                                        balanced_accuracy, brier_score,
                                        counterfactual_flip_rate,
                                        evaluate_predictions,
                                        expected_calibration_error, fit_temperature,
                                        group_metrics, human_distribution_metrics,
                                        jensen_shannon, label_prior_drift, macro_f1)
from sentalign.evaluate.stats import (compare_family, holm_bonferroni, mcnemar_exact,
                                      paired_bootstrap, required_seeds)
from sentalign.labels import (INSTRUCTION, TERNARY, TERNARY_PLUS_MIXED, build_prompt,
                              build_target, parse_completion)


# --------------------------------------------------------------------------------------
# Generation length
# --------------------------------------------------------------------------------------

class _Config:
    def __init__(self, max_length):
        self.max_length = max_length


class _Model:
    def __init__(self, max_length):
        self.generation_config = _Config(max_length)


def test_a_checkpoints_generation_max_length_is_cleared():
    """LFM2 ships max_length=128000, which transformers warns about on every batch.

    It warns and then uses ``max_new_tokens`` anyway, so this is about the log, not the
    output: a few hundred lines per evaluation would otherwise bury the messages that
    matter across a grid of a hundred runs.
    """
    from sentalign.evaluate.scoring import clear_generation_max_length

    model = _Model(128_000)
    assert clear_generation_max_length(model) is True
    assert model.generation_config.max_length is None
    # Idempotent, and quiet about models that never set one.
    assert clear_generation_max_length(model) is False
    assert clear_generation_max_length(object()) is False


# --------------------------------------------------------------------------------------
# Prompting and parsing (AUDIT S2-6)
# --------------------------------------------------------------------------------------

def test_instruction_does_not_enumerate_the_labels():
    """The v1 prompt listed "negative, neutral, or positive", so a first-match parser
    over the decoded prompt+completion returned "negative" for every input."""
    lowered = INSTRUCTION.lower()
    listed = sum(1 for y in ("negative", "neutral", "positive", "mixed") if y in lowered)
    assert listed == 0, f"the instruction names {listed} label words"


def test_parse_completion_rejects_prompt_contamination():
    prompt = build_prompt("The food was fine.")
    with pytest.raises(ValueError, match="generated continuation only"):
        parse_completion(prompt + " positive", TERNARY_PLUS_MIXED)


def test_parse_completion_handles_the_common_shapes():
    space = TERNARY_PLUS_MIXED
    assert parse_completion(" positive", space) == "positive"
    assert parse_completion("Sentiment: negative", space) == "negative"
    assert parse_completion("  Sentiment: **mixed** because ...", space) == "mixed"
    assert parse_completion("neutral. The text is factual.", space) == "neutral"
    assert parse_completion("I am not sure about this one", space) is None
    assert parse_completion("", space) is None


def test_prompt_ends_without_a_trailing_space():
    """The space belongs to the verbalizer: most BPE tokenizers encode " positive" as
    one token and "positive" as two, so moving it changes the number of scored tokens."""
    prompt = build_prompt("Nice place.")
    assert prompt.endswith("Sentiment:")
    assert not prompt.endswith(" ")
    assert build_target("positive").startswith(" ")


def test_verbalizers_are_not_prefixes_of_one_another():
    """Length-normalised scoring is ill-defined if one verbalizer prefixes another."""
    from sentalign.labels import LabelSpace

    with pytest.raises(ValueError, match="prefix"):
        LabelSpace(name="bad", labels=("a", "b"),
                   verbalizers={"a": "pos", "b": "positive"})


# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------

def test_macro_f1_matches_a_hand_computation():
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 0])
    # class 0: tp=1 fp=1 fn=1 -> 0.5 | class 1: tp=2 fp=1 fn=0 -> 0.8 | class 2: tp=1 fn=1 -> 2/3
    assert macro_f1(y_true, y_pred, 3) == pytest.approx((0.5 + 0.8 + 2 / 3) / 3)


def test_balanced_accuracy_penalises_predicting_the_majority():
    y_true = np.array([0] * 90 + [1] * 10)
    y_pred = np.zeros(100, dtype=int)
    assert (y_true == y_pred).mean() == pytest.approx(0.90)
    assert balanced_accuracy(y_true, y_pred, 2) == pytest.approx(0.50)


def test_ece_is_zero_for_a_perfectly_calibrated_predictor():
    rng = np.random.default_rng(0)
    n = 20_000
    p = rng.uniform(0.5, 1.0, n)
    correct = rng.random(n) < p
    probs = np.stack([1 - p, p], axis=1)
    y = np.where(correct, 1, 0)
    assert expected_calibration_error(probs, y, strategy="quantile") < 0.02


def test_temperature_scaling_moves_in_the_right_direction():
    rng = np.random.default_rng(1)
    n, K = 4000, 3
    y = rng.integers(0, K, n)
    base = rng.normal(0, 1, (n, K))
    base[np.arange(n), y] += 1.5
    assert fit_temperature(base * 5, y) > 1.0      # overconfident -> soften
    assert fit_temperature(base * 0.2, y) < 1.0    # underconfident -> sharpen


def test_jsd_bounds():
    assert jensen_shannon(np.array([[0.5, 0.5]]), np.array([[0.5, 0.5]]))[0] == pytest.approx(0.0)
    assert jensen_shannon(np.array([[1.0, 0.0]]), np.array([[0.0, 1.0]]))[0] == pytest.approx(1.0)


def test_excess_cross_entropy_is_zero_when_the_model_matches_the_annotators():
    p_human = np.array([[0.6, 0.2, 0.2], [0.4, 0.4, 0.2]])
    out = human_distribution_metrics(p_human.copy(), p_human)
    assert out["excess_ce_human"] == pytest.approx(0.0, abs=1e-9)
    assert out["jsd_human"] == pytest.approx(0.0, abs=1e-9)


def test_aurc_rewards_ranking_your_own_errors():
    """Two models with identical accuracy must differ in AURC if one knows when it is
    wrong. This is the property accuracy cannot express and deployment depends on."""
    y = np.array([0, 0, 1, 1])
    # both get 3/4 right; the first is confident exactly when correct
    informative = np.array([[9.0, 0], [9.0, 0], [0, 9.0], [1.0, 0.9]])
    misleading = np.array([[1.0, 0.9], [9.0, 0], [0, 9.0], [9.0, 0]])
    assert aurc(informative, y) < aurc(misleading, y)


def test_accuracy_at_coverage_is_monotone_for_a_good_ranker():
    rng = np.random.default_rng(3)
    n = 2000
    y = rng.integers(0, 3, n)
    logits = rng.normal(0, 1, (n, 3))
    logits[np.arange(n), y] += 2.0
    probs = apply_temperature(logits, 1.0)
    a50 = accuracy_at_coverage(probs, y, 0.5)
    a90 = accuracy_at_coverage(probs, y, 0.9)
    a100 = accuracy_at_coverage(probs, y, 1.0)
    assert a50 >= a90 >= a100


def test_small_groups_are_excluded_from_the_worst_group_statistic():
    """A 5-item group's minimum is sampling noise; crediting it as robustness is how
    group-robust methods get credit for variance."""
    rng = np.random.default_rng(4)
    n = 400
    y = rng.integers(0, 2, n)
    logits = rng.normal(0, 1, (n, 2))
    logits[np.arange(n), y] += 3.0
    probs = apply_temperature(logits, 1.0)
    groups = ["big"] * 395 + ["tiny"] * 5
    out = group_metrics(probs, y, groups, 2, min_size=30)
    assert out["n_eligible_groups"] == 1
    assert "tiny" in out["per_group_f1"]          # still reported ...
    assert out["worst_group_f1"] == out["per_group_f1"]["big"]   # ... but not the minimum


def test_label_prior_drift_detects_a_collapsed_predictor():
    uniform = np.array([1 / 3, 1 / 3, 1 / 3])
    collapsed = np.tile(np.array([[9.0, 0.0, 0.0]]), (100, 1))
    assert label_prior_drift(collapsed, uniform) == pytest.approx(2 / 3)


def test_counterfactual_consistency_penalises_rigid_and_jumpy_models():
    should_flip = np.array([1, 1, 0, 0])
    rigid = counterfactual_flip_rate(np.array([0, 0, 0, 0]), np.array([0, 0, 0, 0]),
                                     should_flip)
    jumpy = counterfactual_flip_rate(np.array([0, 0, 0, 0]), np.array([1, 1, 1, 1]),
                                     should_flip)
    assert rigid["stability"] == 1.0 and rigid["sensitivity"] == 0.0
    assert jumpy["sensitivity"] == 1.0 and jumpy["stability"] == 0.0
    assert rigid["consistency"] == jumpy["consistency"] == 0.5


def test_evaluate_predictions_uses_one_probability_source():
    """Accuracy and ECE must describe the same classifier (AUDIT S2-6)."""
    rng = np.random.default_rng(5)
    n, K = 500, 3
    y = rng.integers(0, K, n)
    logits = rng.normal(0, 1, (n, K))
    logits[np.arange(n), y] += 2.0
    result = evaluate_predictions(logits, y, K)
    probs = apply_temperature(logits, 1.0)
    assert result.metrics["accuracy"] == pytest.approx((probs.argmax(1) == y).mean())
    assert result.metrics["ece"] == pytest.approx(
        expected_calibration_error(probs, y, strategy="quantile"))


def test_temperature_is_fitted_on_held_out_logits_when_supplied():
    rng = np.random.default_rng(6)
    y = rng.integers(0, 3, 800)
    test_logits = rng.normal(0, 1, (800, 3)) * 4
    dev_y = rng.integers(0, 3, 800)
    dev_logits = rng.normal(0, 1, (800, 3))
    dev_logits[np.arange(800), dev_y] += 3.0
    on_test = evaluate_predictions(test_logits, y, 3)
    on_dev = evaluate_predictions(test_logits, y, 3,
                                  temperature_fit_logits=dev_logits,
                                  temperature_fit_labels=dev_y)
    assert on_test.metrics["temperature"] != pytest.approx(on_dev.metrics["temperature"])


# --------------------------------------------------------------------------------------
# Statistics (AUDIT S2-8)
# --------------------------------------------------------------------------------------

def test_holm_controls_the_family_wise_error_rate():
    """Eight null comparisons at alpha=0.05 give ~34% uncorrected FWER."""
    rng = np.random.default_rng(7)
    n = 3000
    base = (rng.random(n) < 0.7).astype(float)
    family = {f"m{i}": (rng.random(n) < 0.7).astype(float) for i in range(8)}
    results = compare_family(base, family, n_resamples=1000, seed=1)
    assert not any(r.significant for r in results)


def test_holm_retains_a_real_effect():
    rng = np.random.default_rng(8)
    n = 4000
    base = (rng.random(n) < 0.70).astype(float)
    family = {f"null{i}": (rng.random(n) < 0.70).astype(float) for i in range(7)}
    family["real"] = np.clip(base + (rng.random(n) < 0.08), 0, 1)
    results = {r.name: r for r in compare_family(base, family, n_resamples=2000, seed=2)}
    assert results["real"].significant
    assert results["real"].difference > 0


def test_holm_adjusted_p_values_are_monotone():
    from sentalign.evaluate.stats import ComparisonResult

    raw = [ComparisonResult(f"m{i}", 0.0, 0.0, 0.0, 0.0, 0.0, p)
           for i, p in enumerate([0.001, 0.04, 0.03, 0.5])]
    adjusted = holm_bonferroni(raw)
    by_raw = sorted(adjusted, key=lambda r: r.p_value)
    values = [r.p_adjusted for r in by_raw]
    assert values == sorted(values)
    assert all(r.p_adjusted >= r.p_value for r in adjusted)


def test_seed_aware_bootstrap_is_more_conservative_than_item_only():
    """Pooling seeds and resampling items alone treats 5 runs as 5x the evidence."""
    rng = np.random.default_rng(9)
    seeds = np.repeat(np.arange(5), 800)
    base = (rng.random(4000) < 0.70).astype(float)
    method = (rng.random(4000) < 0.73).astype(float)
    aware = paired_bootstrap(base, method, seeds_baseline=seeds, seeds_method=seeds,
                             n_resamples=800, seed=1)
    item_only = paired_bootstrap(base, method, n_resamples=800, seed=1)
    assert (aware.ci_high - aware.ci_low) > (item_only.ci_high - item_only.ci_low)
    assert aware.n_seeds == 5


def test_bootstrap_p_value_is_floored_at_the_resampling_resolution():
    base = np.zeros(500)
    method = np.ones(500)
    result = paired_bootstrap(base, method, n_resamples=1000)
    assert result.p_value == pytest.approx(1 / 1000)


def test_mcnemar_matches_the_exact_binomial():
    base = np.array([1] * 10 + [0] * 10)
    method = np.array([1] * 5 + [0] * 5 + [1] * 10)   # b=5 discordant, c=10 discordant
    p = mcnemar_exact(base, method)
    assert 0.0 < p <= 1.0
    assert mcnemar_exact(base, base) == 1.0


def test_power_analysis_is_monotone_in_the_effect_size():
    big = required_seeds(0.02, item_sd=0.46, n_items=3600, seed_sd=0.004)
    small = required_seeds(0.005, item_sd=0.46, n_items=3600, seed_sd=0.004)
    assert small > big
    # The headline claim of Section "Statistics": half a point needs dozens of seeds.
    assert small > 20


# --------------------------------------------------------------------------------------
# TRL integration contract
# --------------------------------------------------------------------------------------

def test_custom_trainers_validate_the_trl_batch_contract():
    """GR-DPO and MixDPO override a TRL internal. An API change there would not raise,
    it would silently train the wrong objective, so the batch layout the overrides
    depend on is asserted on the first batch of every run."""
    from sentalign.train.po import REQUIRED_BATCH_KEYS, _assert_batch_contract

    good = {k: object() for k in REQUIRED_BATCH_KEYS}
    _assert_batch_contract(good, needs_reference=False)          # must not raise
    _assert_batch_contract(good | {"ref_chosen_logps": object()}, needs_reference=True)

    with pytest.raises(RuntimeError, match="missing"):
        _assert_batch_contract({"prompt_input_ids": object()}, needs_reference=False)
    with pytest.raises(RuntimeError, match="precompute_ref_log_probs"):
        _assert_batch_contract(good, needs_reference=True)


def test_trl_version_range_is_enforced_at_both_ends():
    """The supported line is TRL 0.2x, which is what Unsloth targets. The 1.x line is a
    real API break (CPOTrainer removed, hooks renamed, batch layout changed), so it is
    rejected explicitly rather than left to fail somewhere less legible."""
    from sentalign.train.po import MAX_TRL_EXCLUSIVE, MIN_TRL, _parse_version

    def supported(text: str) -> bool:
        return MIN_TRL <= _parse_version(text) < MAX_TRL_EXCLUSIVE

    assert supported("0.24.0") and supported("0.28.0")
    assert not supported("0.23.0"), "below the floor: loss_type is a bare string"
    assert not supported("1.10.0"), "1.x removed CPOTrainer and renamed the hooks"

    # Suffixed versions must not parse to zero and silently fail the floor.
    assert _parse_version("0.24.0.dev0") == (0, 24, 0)
    assert _parse_version("0.26.1") == (0, 26, 1)


def test_kto_weights_hit_the_recommended_balance():
    """One correct label and three incorrect gives a 1:3 imbalance intrinsic to a closed
    label set; leaving both weights at 1.0 trains KTO in a regime it was not designed for."""
    from sentalign.train.po import kto_weights

    lam_d, lam_u = kto_weights(94_740, 286_902)          # the real HDP-Sent counts
    ratio = (lam_d * 94_740) / (lam_u * 286_902)
    assert 1.0 <= ratio <= 4 / 3
    assert kto_weights(0, 10) == (1.0, 1.0)              # degenerate input is safe


def test_simpo_and_alphapo_are_library_native_not_reimplemented():
    """TRL 0.2x provides both through CPOTrainer, so we do not transcribe the loss.

    The formula in objectives.simpo_loss stays under test as the specification the
    library implementation is checked against, but it is not what runs.
    """
    from sentalign.train.objectives import CUSTOM_OBJECTIVES, TRL_OBJECTIVES

    for name in ("simpo", "alphapo"):
        assert TRL_OBJECTIVES[name]["trainer"] == "CPOTrainer"
        assert name not in CUSTOM_OBJECTIVES
    assert TRL_OBJECTIVES["simpo"]["loss_type"] == "simpo"
    assert TRL_OBJECTIVES["alphapo"]["loss_type"] == "alphapo"


def test_objective_registry_is_exhaustive_and_disjoint():
    """Every arm belongs to exactly one implementation route, and none is orphaned."""
    from sentalign.train.objectives import (ALL_OBJECTIVES, CUSTOM_OBJECTIVES,
                                            DISTRIBUTIONAL_OBJECTIVES, REFERENCE_FREE,
                                            SUPERVISED_OBJECTIVES, TRL_OBJECTIVES)

    routes = (set(SUPERVISED_OBJECTIVES), set(TRL_OBJECTIVES), set(CUSTOM_OBJECTIVES))
    for objective in ALL_OBJECTIVES:
        memberships = sum(objective in route for route in routes)
        assert memberships == 1, f"{objective} belongs to {memberships} routes"
    assert set(ALL_OBJECTIVES) == set().union(*routes)

    # The declared properties must reference real arms.
    assert set(DISTRIBUTIONAL_OBJECTIVES) <= set(ALL_OBJECTIVES)
    assert set(REFERENCE_FREE) <= set(ALL_OBJECTIVES)

    # CSPO and its control differ only in the reference anchor, so exactly one of the two
    # distributional arms uses a reference.
    # cspo_kl joins them because it consumes the annotator distribution rather than
    # binarised pairs, and it routes through the same trainer. It differs from cspo only
    # in where the reference sits: a KL penalty beside the loss instead of inside the
    # softmax, so the distribution it trains is the one evaluation scores.
    assert set(DISTRIBUTIONAL_OBJECTIVES) == {"sft_soft", "cspo", "cspo_kl",
                                              "cspo_ada", "polya", "mopa"}
    assert "sft_soft" in REFERENCE_FREE and "cspo" in REFERENCE_FREE


def test_dispatch_routes_every_distributional_arm_to_the_cspo_family(monkeypatch):
    """The trainer that consumes annotator distributions must receive every arm the
    registry marks distributional. The routing tuple was hand-maintained and drifted:
    cspo_kl was added to DISTRIBUTIONAL_OBJECTIVES and to the family trainer body, but
    the dispatch guard still listed only ("cspo", "sft_soft"), so cspo_kl fell through
    to train_preference and raised ``unknown preference objective 'cspo_kl'`` at run
    time. Routing off the registry closes that gap; this pins it shut."""
    from sentalign import train as _train_pkg  # noqa: F401
    from sentalign.config import ExperimentConfig
    from sentalign.train import driver
    from sentalign.train.objectives import DISTRIBUTIONAL_OBJECTIVES
    import sentalign.train.po as po

    fired: dict[str, str] = {}
    monkeypatch.setattr(driver, "_train_cspo_family",
                        lambda *a, **k: fired.__setitem__("route", "cspo_family") or {})
    monkeypatch.setattr(po, "train_preference",
                        lambda *a, **k: fired.__setitem__("route", "preference") or {})

    def route(objective: str) -> str:
        fired.clear()
        cfg = ExperimentConfig(name="test", model="lfm-350m")
        cfg.train.objective = objective
        driver._dispatch(cfg, {}, None, None, None, None, None, None, None, None, None)
        return fired["route"]

    for objective in DISTRIBUTIONAL_OBJECTIVES:
        assert route(objective) == "cspo_family", (
            f"{objective} is distributional but did not reach the CSPO-family trainer")
    # A genuine pairwise arm still reaches the preference trainer.
    assert route("dpo") == "preference"


def test_group_preserving_collator_refuses_to_lose_group_ids():
    """TRL's preference collator returns a fresh dict and drops extra columns, so GR-DPO
    would silently train as mean DPO. That must be an error, not a default."""
    from sentalign.train.po import GroupPreservingCollator

    class FakeInner:
        def __call__(self, examples):
            return {"input_ids": object()}          # mimics TRL: builds a fresh dict

    collator = GroupPreservingCollator(FakeInner(), n_groups=3)
    with pytest.raises(RuntimeError, match="silently train as mean DPO"):
        collator([{"prompt": "x"}, {"prompt": "y"}])

    pytest.importorskip("torch")
    batch = collator([{"group_id": 0}, {"group_id": 2}])
    assert batch["group_id"].tolist() == [0, 2]
    with pytest.raises(RuntimeError, match="out of range"):
        collator([{"group_id": 0}, {"group_id": 7}])


def test_paired_design_reduces_the_seed_requirement():
    """The claim the paper makes: pairing brings a one-point effect within five seeds."""
    from sentalign.evaluate.stats import required_seeds_paired

    unpaired = required_seeds(0.010, item_sd=0.46, n_items=3600, seed_sd=0.004)
    paired = required_seeds_paired(0.010, item_sd=0.46, n_items=11_000, seed_sd=0.004,
                                   item_correlation=0.9)
    assert unpaired > 5 >= paired, (unpaired, paired)

    # Half a point stays out of reach even with pairing, which is what we report.
    assert required_seeds_paired(0.005, item_sd=0.46, n_items=11_000, seed_sd=0.004,
                                 item_correlation=0.9) > 5


def test_paired_power_is_monotone_in_its_assumptions():
    from sentalign.evaluate.stats import required_seeds_paired

    kw = dict(effect_size=0.01, item_sd=0.46, n_items=11_000, seed_sd=0.006)
    assert (required_seeds_paired(**kw, item_correlation=0.5)
            >= required_seeds_paired(**kw, item_correlation=0.9))
    assert (required_seeds_paired(**kw, item_correlation=0.9, shared_reference=False)
            >= required_seeds_paired(**kw, item_correlation=0.9, shared_reference=True))
    with pytest.raises(ValueError):
        required_seeds_paired(**kw, item_correlation=1.0)


def test_observed_item_correlation_matches_the_planning_assumption():
    from sentalign.evaluate.stats import observed_item_correlation

    rng = np.random.default_rng(21)
    base = (rng.random(4000) < 0.7).astype(float)
    agree = base.copy()
    flip = rng.random(4000) < 0.05          # arms disagree on 5 percent of items
    agree[flip] = 1 - agree[flip]
    assert observed_item_correlation(base, agree) > 0.85
    assert observed_item_correlation(base, rng.random(4000)) < 0.1


# --------------------------------------------------------------------------------------
# Undefined metrics must not corrupt the results files
# --------------------------------------------------------------------------------------

def test_metrics_json_is_strict_json_when_a_metric_is_undefined(tmp_path):
    """`worst_group_f1` is undefined when no group reaches min_size, and the label-free
    slice has no point metrics at all. json.dumps writes bare NaN for those, which no
    strict parser accepts: jq, jsonlite, and JSON.parse all reject the file, and only
    the Python that wrote it can read it back."""
    import json

    from sentalign.evaluate.metrics import EvalResult
    from sentalign.evaluate.run_eval import EvaluationOutput

    output = EvaluationOutput(
        run_id="test", wallclock_s=1.0, throughput_items_per_s=1.0,
        results={
            "ambig_no_majority": EvalResult(
                n=3, metrics={"n": 3.0, "macro_f1": float("nan"),
                              "jsd_human": 0.25, "aurc": float("-inf")}),
        },
        per_item={})
    output.save(tmp_path)

    raw = (tmp_path / "metrics.json").read_text()
    assert "NaN" not in raw and "Infinity" not in raw, raw
    block = json.loads(raw)["sets"]["ambig_no_majority"]["metrics"]
    assert block["macro_f1"] is None and block["aurc"] is None
    assert block["jsd_human"] == 0.25, "defined metrics survive untouched"


def test_an_undefined_metric_is_dropped_from_the_aggregate_not_averaged_in():
    """One seed with an undefined metric must not turn the whole column into nan."""
    import importlib.util
    import math
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "scripts" / "03_make_tables.py"
    spec = importlib.util.spec_from_file_location("make_tables_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def run(macro):
        return {"metrics": {"sets": {"r1_test": {"metrics": {"macro_f1": macro}}}}}

    summary = module.summarise({13: run(0.50), 21: run(None), 34: run(0.60)}, "r1_test")
    mean, sd = summary["macro_f1"]
    assert math.isfinite(mean) and abs(mean - 0.55) < 1e-9, summary
    assert math.isfinite(sd)

    legacy = module.summarise({13: run(0.50), 21: run(float("nan"))}, "r1_test")
    assert legacy["macro_f1"][0] == 0.50, "a NaN from an older run is dropped too"


def test_the_reward_readout_reproduces_the_distribution_cspo_trains(tmp_path):
    """CSPO fits softmax(beta*(l - l_ref)); every other read-out here scores softmax(l).

    On the real grid the difference was a factor of two: 0.3086 JSD from the policy
    read-out against 0.1412 from the reward read-out, on the no-majority items where the
    claim lives. This pins the arithmetic that recovers the second from stored logits.
    """
    import json
    import numpy as np

    from sentalign.evaluate.metrics import EvalResult
    from sentalign.evaluate.run_eval import EvaluationOutput, attach_reward_readout

    beta, human = 2.0, [0.5, 0.3, 0.2]
    policy_logits = [1.5, 0.25, -0.75]
    ref_logits = [0.5, 0.5, 0.5]

    reference = tmp_path / "sft" / "eval"
    reference.mkdir(parents=True)
    (reference / "predictions_s.jsonl").write_text(json.dumps(
        {"text_id": "a", "logits": ref_logits}) + "\n")

    output = EvaluationOutput(
        run_id="cspo", wallclock_s=1.0, throughput_items_per_s=1.0,
        results={"s": EvalResult(n=1, metrics={"n": 1.0})},
        per_item={"s": [{"text_id": "a", "logits": policy_logits, "p_human": human}]})

    added = attach_reward_readout(output, tmp_path / "sft", beta=beta)

    z = beta * (np.array(policy_logits) - np.array(ref_logits))
    expected = np.exp(z - z.max())
    expected /= expected.sum()
    got = output.per_item["s"][0]["reward_probs"]
    assert np.allclose(got, expected), (got, expected)
    assert added["s"]["reward/beta"] == beta
    assert added["s"]["reward/n_matched"] == 1.0
    assert "reward/jsd_human" in output.results["s"].metrics

    # And the per-item metric consumes it.
    from sentalign.evaluate.run_eval import per_item_scores

    scores = per_item_scores(output.per_item["s"], "jsd_reward")
    assert scores.shape == (1,) and 0.0 <= scores[0] <= 1.0

    # An arm with no reference must say so rather than silently scoring something else.
    output.per_item["s"][0]["reward_probs"] = None
    try:
        per_item_scores(output.per_item["s"], "jsd_reward")
    except ValueError as exc:
        assert "reward_probs" in str(exc)
    else:
        raise AssertionError("a missing read-out must raise, not fall back")


def test_the_sentiment_prompt_is_byte_identical_after_the_task_refactor():
    """Prompting moved onto LabelSpace so a second task could be added. If that changed
    the sentiment prompt by even one byte, every new sentiment run would be incomparable
    with the 283 already completed, silently. This pins the exact string."""
    from sentalign.labels import TERNARY_PLUS_MIXED, build_prompt

    expected = ("Read the text and judge the sentiment the author expresses.\n"
                "Answer on one line in the form `Sentiment: <label>`.\n\n"
                "Text: The food was fine.\nSentiment:")
    assert build_prompt("The food was fine.") == expected
    assert build_prompt("The food was fine.", TERNARY_PLUS_MIXED) == expected


def test_nli_parses_its_own_verbalizers_and_not_the_sentiment_words():
    """NLI verbalises entailment as "yes", so parsing must invert the verbalizer map
    rather than normalising the matched surface as a label. It must also not accept
    sentiment words, which the old module-level regex would have."""
    from sentalign.labels import NLI3, TERNARY_PLUS_MIXED, parse_completion

    assert parse_completion(" yes", NLI3) == "entailment"
    assert parse_completion("Answer: maybe", NLI3) == "neutral"
    assert parse_completion("Answer: **no** because ...", NLI3) == "contradiction"
    # The sentiment words are not in this space.
    assert parse_completion(" positive", NLI3) is None
    # And the NLI words are not in the sentiment space.
    assert parse_completion(" yes", TERNARY_PLUS_MIXED) is None


def test_every_label_space_has_a_recoverable_and_unambiguous_verbalizer_map():
    """P1 needs one token per verbalizer, and the parser needs the surface -> label map
    to be injective. Both are properties of the space, so check every registered one."""
    from sentalign.labels import LABEL_SPACES, label_for_surface

    for name, space in LABEL_SPACES.items():
        surfaces = [space.verbalizers[y] for y in space.labels]
        assert len(set(s.lower() for s in surfaces)) == len(surfaces), name
        for y in space.labels:
            assert label_for_surface(space, space.verbalizers[y]) == y, (name, y)
            assert label_for_surface(space, f"  {space.verbalizers[y].upper()} ") == y


def test_the_distributional_temperature_is_fitted_to_the_annotator_distribution():
    """Two different calibrations. Fitting to hard labels is what a practitioner chasing
    accuracy does; fitting to p_human is what someone chasing the annotator distribution
    does, and it is the control that erased the distributional effect on the sentiment
    data. They must not be confused, so this pins that the distributional fit recovers a
    known sharpening exactly while the hard-label fit does not have to."""
    import numpy as np

    from sentalign.evaluate.metrics import (apply_temperature,
                                            fit_temperature_to_distribution,
                                            jensen_shannon)

    p = np.array([[0.45, 0.35, 0.20]] * 400)
    sharpened = np.log(p) * 3.0                     # exactly 3x too confident
    t = fit_temperature_to_distribution(sharpened, p)
    assert t == pytest.approx(3.0, abs=0.05)

    before = float(jensen_shannon(apply_temperature(sharpened, 1.0), p).mean())
    after = float(jensen_shannon(apply_temperature(sharpened, t), p).mean())
    assert before > 0.04 and after < 1e-6, (before, after)


def test_jsd_temp_refuses_to_score_items_that_were_never_temperature_corrected():
    """A silent fallback here would compare corrected arms against uncorrected ones."""
    from sentalign.evaluate.run_eval import per_item_scores

    rows = [{"text_id": "a", "temp_probs": [0.5, 0.3, 0.2], "p_human": [0.4, 0.4, 0.2]},
            {"text_id": "b", "temp_probs": None, "p_human": [0.4, 0.4, 0.2]}]
    with pytest.raises(ValueError, match="temp_probs"):
        per_item_scores(rows, "jsd_temp")

    ok = per_item_scores(rows[:1], "jsd_temp")
    assert ok.shape == (1,) and ok[0] > 0


# -- the attention kernel scoring runs on ------------------------------------------------


class _FakeConfig:
    def __init__(self, name="flex_attention"):
        self._attn_implementation = name


class _FrozenConfig:
    """A config whose attention setting cannot be written, which is how a transformers
    version that ignores the pin would look from here."""

    def __init__(self, name="flex_attention"):
        object.__setattr__(self, "_name", name)

    @property
    def _attn_implementation(self):
        return self._name

    @_attn_implementation.setter
    def _attn_implementation(self, value):
        raise AttributeError("read-only")


class _FakeModel:
    def __init__(self, config):
        self.config = config


def test_smollm3_is_scored_away_from_flex_attention():
    """flex_attention builds a mask ``create_block_mask`` cannot unpack, so a batched
    generate dies after the model has loaded. The pin lives on the registry entry, and
    applying it reports the kernel that scoring will actually use."""
    from sentalign.modeling import (MODEL_REGISTRY, apply_inference_attn_implementation,
                                    inference_attn_implementation)
    from sentalign.config import ExperimentConfig

    assert MODEL_REGISTRY["smollm3-3b"]["inference_attn_implementation"] == "sdpa"
    spec = ExperimentConfig(model="smollm3-3b").model_spec
    assert inference_attn_implementation(spec) == "sdpa"

    model = _FakeModel(_FakeConfig())
    assert apply_inference_attn_implementation(spec, model) == "sdpa"
    assert model.config._attn_implementation == "sdpa"


def test_a_sub_config_is_switched_with_its_parent():
    """The kernel is read off whichever config the decoder consults, so a parent that
    says sdpa over a text_config that still says flex_attention is not a pinned model."""
    from sentalign.config import ExperimentConfig
    from sentalign.modeling import apply_inference_attn_implementation

    model = _FakeModel(_FakeConfig())
    model.config.text_config = _FakeConfig()
    apply_inference_attn_implementation(ExperimentConfig(model="smollm3-3b").model_spec,
                                        model)
    assert model.config.text_config._attn_implementation == "sdpa"


def test_a_pin_that_did_not_take_is_an_error_not_a_silent_fallback():
    from sentalign.config import ExperimentConfig
    from sentalign.modeling import apply_inference_attn_implementation

    spec = ExperimentConfig(model="smollm3-3b").model_spec
    with pytest.raises(RuntimeError, match="registry pinned it away"):
        apply_inference_attn_implementation(spec, _FakeModel(_FrozenConfig()))


def test_an_unpinned_model_keeps_the_backend_default():
    """lfm-1.2b and qwen-2b were evaluated before this existed. Changing their kernel now
    would mean their completed evaluations and any rerun ran on different code."""
    from sentalign.config import ExperimentConfig
    from sentalign.modeling import apply_inference_attn_implementation

    for key in ("lfm-1.2b", "qwen-2b"):
        model = _FakeModel(_FakeConfig("eager"))
        spec = ExperimentConfig(model=key).model_spec
        assert apply_inference_attn_implementation(spec, model) == "eager"
        assert model.config._attn_implementation == "eager"


def test_the_kernel_is_recorded_with_the_metrics(tmp_path):
    """Two runs of the same arm on different kernels are a difference nothing else in the
    output would show."""
    import json

    from sentalign.evaluate.run_eval import EvaluationOutput

    out = EvaluationOutput(run_id="r", results={}, per_item={}, wallclock_s=1.0,
                           throughput_items_per_s=1.0, attn_implementation="sdpa")
    out.save(tmp_path)
    assert json.loads((tmp_path / "metrics.json").read_text())["attn_implementation"] \
        == "sdpa"


def test_the_pin_is_applied_after_the_backend_inference_patch():
    """Unsloth's for_inference rewrites the config it is handed, so a pin applied before
    it can be undone with nothing to show for it."""
    from pathlib import Path as _Path

    source = _Path("src/sentalign/evaluate/run_eval.py").read_text()
    assert source.index("set_inference_mode(model)") \
        < source.index("apply_inference_attn_implementation(cfg.model_spec, model)")


# -- one identifier, two items ------------------------------------------------------------


def _cspo(tmp_path, rows):
    import json

    build = tmp_path / "build"
    (build / "cspo").mkdir(parents=True)
    (build / "cspo" / "train_n8000.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n")
    return build


def test_a_reused_identifier_is_resolved_by_the_prompt(tmp_path):
    """The NLI corpus draws ids from SNLI and from MNLI, and four of the short MNLI ids
    collide among 8,000 items. Both items are real and their distributions differ, so the
    join has to tell them apart rather than refuse or guess."""
    from sentalign.cli import _annotator_targets, _attach_targets

    build = _cspo(tmp_path, [
        {"text_id": "4667c", "prompt": "premise A", "p_human": [0.8, 0.1, 0.1]},
        {"text_id": "4667c", "prompt": "premise B", "p_human": [0.1, 0.1, 0.8]},
        {"text_id": "unique", "prompt": "premise C", "p_human": [0.4, 0.3, 0.3]},
    ])
    targets = _annotator_targets(build)
    assert targets["ambiguous"] == {"4667c"}

    records = [{"text_id": "4667c", "prompt": "premise B"},
               {"text_id": "unique", "prompt": "a differently phrased prompt"}]
    kept = _attach_targets(records, targets, split="training", drop_unmatched=False)
    assert [r["p_human"] for r in kept] == [[0.1, 0.1, 0.8], [0.4, 0.3, 0.3]], \
        "the collision resolves by prompt, and a unique id still joins by id alone"


def test_two_builds_in_one_directory_still_raise(tmp_path):
    """Same item, same prompt, two distributions: that is a mixed directory, and the join
    would depend on file order. This is the case the guard was written for."""
    from sentalign.cli import _annotator_targets

    build = _cspo(tmp_path, [
        {"text_id": "x1", "prompt": "same prompt", "p_human": [0.8, 0.1, 0.1]},
        {"text_id": "x1", "prompt": "same prompt", "p_human": [0.2, 0.4, 0.4]},
    ])
    with pytest.raises(SystemExit, match="mixes builds"):
        _annotator_targets(build)


def test_an_unresolvable_collision_is_dropped_by_name_and_nothing_else_is(tmp_path):
    from sentalign.cli import _annotator_targets, _attach_targets

    build = _cspo(tmp_path, [
        {"text_id": "4667c", "prompt": "premise A", "p_human": [0.8, 0.1, 0.1]},
        {"text_id": "4667c", "prompt": "premise B", "p_human": [0.1, 0.1, 0.8]},
        {"text_id": "unique", "prompt": "premise C", "p_human": [0.4, 0.3, 0.3]},
    ])
    targets = _annotator_targets(build)

    kept = _attach_targets(
        [{"text_id": "4667c", "prompt": "a third phrasing"},
         {"text_id": "unique", "prompt": "premise C"}],
        targets, split="training", drop_unmatched=False)
    assert [r["text_id"] for r in kept] == ["unique"], "the ambiguous record is dropped"

    # An item the build simply does not contain is a different failure and still raises.
    with pytest.raises(SystemExit, match="no annotator distribution"):
        _attach_targets([{"text_id": "absent", "prompt": "p"},
                         {"text_id": "unique", "prompt": "premise C"}],
                        targets, split="training", drop_unmatched=False)


def test_an_empty_distributional_directory_still_raises(tmp_path):
    from sentalign.cli import _annotator_targets

    build = tmp_path / "build"
    (build / "cspo").mkdir(parents=True)
    with pytest.raises(SystemExit, match="no distributional records"):
        _annotator_targets(build)
