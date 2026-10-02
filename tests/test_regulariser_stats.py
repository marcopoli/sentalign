"""The confirmatory statistics behind the regulariser claims.

Each test pins a property the paper's numbers depend on, checked by breaking that property
in ``scripts/06_regulariser_stats.py`` and watching the test fail.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "06_regulariser_stats.py"


@pytest.fixture(scope="module")
def stats():
    spec = importlib.util.spec_from_file_location("regulariser_stats", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_auroc_matches_the_pairwise_definition_with_ties(stats):
    rng = np.random.default_rng(7)
    checked = 0
    for _ in range(60):
        n = int(rng.integers(20, 80))
        conf = rng.integers(0, 6, n) / 5.0          # heavy ties on purpose
        correct = (rng.random(n) < 0.6).astype(float)
        if correct.min() == correct.max():
            continue
        pos, neg = conf[correct == 1], conf[correct == 0]
        brute = (((pos[:, None] > neg[None, :]).sum()
                  + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))
        assert stats.auroc(conf, correct) == pytest.approx(brute)
        checked += 1
    assert checked > 40


def test_e_aurc_is_zero_for_a_perfect_ranker_at_any_accuracy(stats):
    """The property that makes E-AURC accuracy-corrected. Raw AURC fails it: a perfect
    ranker at 60 percent accuracy has a larger AURC than one at 90 percent."""
    n = 200
    for accuracy in (0.6, 0.75, 0.9):
        k = int(accuracy * n)
        correct = np.array([1.0] * k + [0.0] * (n - k))
        confidence = np.linspace(1.0, 0.0, n)       # every correct item above every error
        assert stats.eaurc(confidence, correct) == pytest.approx(0.0, abs=1e-12)
    worst = stats.eaurc(np.linspace(0.0, 1.0, n), np.array([1.0] * 150 + [0.0] * 50))
    assert worst > 0.1


def test_the_planned_comparisons_are_declared_per_model(stats):
    """Each regularised arm meets its own twin, and SmolLM3 runs only the five-arm set."""
    full = (("kto-dreg1.0", "kto"), ("rdpo-dreg1.0", "rdpo"), ("ipo-dreg1.0", "ipo"))
    assert stats.repair_plan(stats.MODELS) == {
        "lfm-1.2b": full, "qwen-2b": full,
        "smollm3-3b": (("kto-dreg1.0", "kto"), ("rdpo-dreg1.0", "rdpo"))}
    assert stats.ablation_plan(stats.MODELS) == {
        "lfm-1.2b": (("kto-dreg1.0", "kto-dreg-hard"),),
        "qwen-2b": (("kto-dreg1.0", "kto-dreg-hard"),)}
    assert all(second == "sft" for pairs in stats.sft_plan(stats.MODELS).values()
               for _, second in pairs)


def test_the_average_endpoint_uses_only_seeds_scored_on_every_set(stats):
    sets = stats.SETS
    scores = {}
    for i, name in enumerate(sets):
        seeds = (13, 21) if name == sets[-1] else (13, 21, 34)
        scores[("m", name, "a")] = {s: {"auroc": 0.5 + 0.1 * i + s / 1000} for s in seeds}
    scores[("m", sets[0], "partial")] = {13: {"auroc": 0.9}}     # one set only
    mean = stats.average_over_sets(scores)

    assert ("m", stats.ENDPOINT, "partial") not in mean, "an arm missing a set has no endpoint"
    assert sorted(mean[("m", stats.ENDPOINT, "a")]) == [13, 21], "seed 34 lacks the last set"
    expected = np.mean([0.5 + 0.1 * i + 13 / 1000 for i in range(len(sets))])
    assert mean[("m", stats.ENDPOINT, "a")][13]["auroc"] == pytest.approx(expected)


def test_direction_is_read_per_metric(stats):
    seeds = (13, 21, 34)
    better = {s: {"eaurc": 0.05 + i * 1e-3, "auroc": 0.80 + i * 1e-3}
              for i, s in enumerate(seeds)}
    worse = {s: {"eaurc": 0.09 + i * 2e-3, "auroc": 0.70 + i * 2e-3}
             for i, s in enumerate(seeds)}
    scores = {("m", "ambig_eval", "a"): better, ("m", "ambig_eval", "b"): worse}
    assert stats.paired(scores, "m", "ambig_eval", "a", "b", "eaurc")["favours_first"]
    assert stats.paired(scores, "m", "ambig_eval", "a", "b", "auroc")["favours_first"]
    assert not stats.paired(scores, "m", "ambig_eval", "b", "a", "eaurc")["favours_first"]


def test_a_missing_planned_comparison_is_an_error_not_a_smaller_family(stats):
    with pytest.raises(SystemExit):
        stats.run_family({}, stats.repair_plan(("lfm-1.2b",)), ("auroc",), sets=stats.SETS)


def test_pooled_is_not_counted_as_an_independent_set(stats):
    assert "pooled" not in stats.SETS


def test_the_t_quantiles_match_scipy(stats):
    scipy_stats = pytest.importorskip("scipy.stats")
    for df, value in stats.T975.items():
        assert value == pytest.approx(scipy_stats.t.ppf(0.975, df), abs=5e-4)


def _row(model, first, second, metric, p, p_holm):
    return {"model": model, "set": "mean4", "first": first, "second": second,
            "metric": metric, "n": 5, "seeds": [13, 21, 34, 55, 89], "mean_diff": 0.1,
            "ci95": [0.09, 0.11], "t": 9.0, "p": p, "favours_first": True,
            "p_holm": p_holm}


def test_the_joint_correction_is_never_weaker_than_the_family_it_pools(stats):
    """Holm over a superset is at least as strict as Holm within each part of it. If the
    joint p ever came out below the family p, the paper's sceptical reading would be
    weaker than its protocol reading, which is the wrong way round and would flatter the
    result."""
    families = {
        "P1_primary_auroc": [_row("a", "kto-dreg1.0", "kto", "auroc", 0.0007, 0.0030),
                             _row("b", "kto-dreg1.0", "kto", "auroc", 0.0001, 0.0006)],
        "P2_secondary_eaurc": [_row("a", "kto-dreg1.0", "kto", "eaurc", 0.0005, 0.0025),
                               _row("b", "kto-dreg1.0", "kto", "eaurc", 0.0201, 0.0201)],
        "P3_vs_sft": [_row("a", "kto-dreg1.0", "sft", "eaurc", 0.0011, 0.0044),
                      _row("a", "kto-dreg1.0", "sft", "acc", 0.0400, 0.0800)],
    }
    # The family-level adjustment is recomputed with the same Holm the script uses, so the
    # comparison is between two corrections of the same p-values and not against a number
    # written into the fixture.
    for rows in families.values():
        adjusted = stats.selective.holm({i: r["p"] for i, r in enumerate(rows)})
        for i, r in enumerate(rows):
            r["p_holm"] = adjusted[i]
    joint = stats.joint_correction(families)
    assert joint["n_tests"] == 6
    for r in joint["rows"]:
        family_p = next(x["p_holm"] for x in families[r["family"]]
                        if x["model"] == r["model"] and x["metric"] == r["metric"]
                        and x["second"] == r["second"])
        assert r["p_joint"] >= family_p - 1e-12, r
    assert any(r["p_joint"] > next(x["p_holm"] for x in families[r["family"]]
                                   if x["model"] == r["model"] and x["metric"] == r["metric"]
                                   and x["second"] == r["second"]) + 1e-9
               for r in joint["rows"]), "pooling must actually cost something"


def test_the_joint_survivor_count_tracks_the_tests_it_pools(stats):
    """The manuscript quotes this count. Adding a test can only make the correction
    stricter, so a count that did not move when the pool grew would be a typed number
    rather than a measured one."""
    base = {
        "P1_primary_auroc": [_row("a", "kto-dreg1.0", "kto", "auroc", 0.001, 0.001)],
        "P2_secondary_eaurc": [_row("a", "kto-dreg1.0", "kto", "eaurc", 0.012, 0.012)],
        "P3_vs_sft": [_row("a", "kto-dreg1.0", "sft", "acc", 0.020, 0.020)],
    }
    small = stats.joint_correction(base)
    assert (small["n_tests"], small["n_survivors"]) == (3, 3)
    grown = dict(base)
    grown["P3_vs_sft"] = base["P3_vs_sft"] + [
        _row("b", "rdpo-dreg1.0", "sft", "acc", 0.030, 0.030),
        _row("c", "ipo-dreg1.0", "sft", "acc", 0.040, 0.040)]
    bigger = stats.joint_correction(grown)
    assert bigger["n_tests"] == 5
    assert bigger["n_survivors"] < small["n_tests"] + 2
    assert all(r["p_joint"] >= r["p"] for r in bigger["rows"])


def test_ece_is_the_equal_mass_definition_the_method_section_names(stats):
    """The package computes ECE over quantile bins because a fine-tuned classifier piles
    its confidences against 1.0 and equal-width bins then measure one bin. The analysis
    script had its own equal-width copy, so the reported numbers were not the statistic
    the paper described. One definition, checked against the other implementation.
    """
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from sentalign.evaluate.metrics import expected_calibration_error

    rng = np.random.default_rng(11)
    checked = 0
    for _ in range(40):
        n = int(rng.integers(120, 400))
        # Confidence piled up near 1.0, which is where the two definitions diverge.
        conf = 1.0 - rng.beta(1.0, 12.0, n) * 0.5
        correct = (rng.random(n) < conf * 0.85).astype(float)
        probs = np.stack([conf, 1.0 - conf], axis=1)
        y_true = np.where(correct == 1.0, 0, 1)
        theirs = expected_calibration_error(probs, y_true, n_bins=15, strategy="quantile")
        assert stats.ece(conf, correct) == pytest.approx(theirs, abs=1e-9)
        # And the equal-width statistic is genuinely a different number here, so the test
        # above is not satisfied by both definitions at once.
        uniform = expected_calibration_error(probs, y_true, n_bins=15, strategy="uniform")
        if abs(uniform - theirs) > 1e-4:
            checked += 1
    assert checked > 30, "the two binnings must actually differ on this data"
