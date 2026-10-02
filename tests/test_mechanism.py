"""The mechanism analysis measures what it says: exact order, correct answers, own baseline."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _script(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _row(margin: float, agreement: float, correct: bool) -> dict:
    z = np.array([0.0, -margin, -margin - 2.0, -margin - 4.0])
    p = np.exp(z - z.max()) / np.exp(z - z.max()).sum()
    return {"logits": z.tolist(), "probs": p.tolist(), "correct": correct,
            "p_human": [agreement, 1.0 - agreement, 0.0, 0.0]}


def test_spearman_handles_ties_as_average_ranks():
    stats = pytest.importorskip("scipy.stats")
    mech = _script("mechanism", "22_mechanism.py")
    rng = np.random.default_rng(3)
    x = rng.integers(0, 4, size=40).astype(float)
    y = x + rng.normal(size=40)
    assert mech.spearman(x, y) == pytest.approx(stats.spearmanr(x, y).correlation, abs=1e-12)


def test_agreement_sensitivity_reads_only_the_correct_answers():
    mech = _script("mechanism", "22_mechanism.py")
    reg = _script("regulariser_stats", "06_regulariser_stats.py")
    # Right answers are surer where more annotators agreed; the errors sit on unanimous
    # items at the lowest confidence. Over all items the correlation would be diluted.
    rows = [_row(m, a, True) for m, a in ((2, 0.6), (3, 0.6), (4, 0.8), (5, 0.8), (6, 1.0))]
    rows += [_row(0.5, 1.0, False), _row(1.0, 1.0, False)]
    got = mech.set_metrics(rows, auroc=reg.auroc)
    assert got["sensitivity"] == pytest.approx(mech.spearman([2, 3, 4, 5, 6],
                                                             [0.6, 0.6, 0.8, 0.8, 1.0]))
    everything = mech.spearman([2, 3, 4, 5, 6, 0.5, 1.0], [0.6, 0.6, 0.8, 0.8, 1.0, 1.0, 1.0])
    assert got["sensitivity"] > everything + 0.3
    assert got["auroc"] == 1.0


def test_agreement_sensitivity_reads_the_exact_order_of_saturated_items():
    mech = _script("mechanism", "22_mechanism.py")
    reg = _script("regulariser_stats", "06_regulariser_stats.py")
    rows = [_row(m, a, True) for m, a in ((40, 0.6), (45, 0.8), (50, 1.0))]
    rows.append(_row(38, 1.0, False))
    assert all(max(r["probs"]) == 1.0 for r in rows), "premise: every stored c is 1.0"
    got = mech.set_metrics(rows, auroc=reg.auroc)
    assert got["sensitivity"] == pytest.approx(1.0)
    assert got["log_odds"] > 36


def test_changes_are_taken_against_the_same_models_baseline():
    mech = _script("mechanism", "22_mechanism.py")
    means = {"a|sft": {"auroc": 0.8, "sensitivity": 0.3, "log_odds": 2.0},
             "a|kto": {"auroc": 0.7, "sensitivity": 0.1, "log_odds": 40.0},
             "a|dpo": {"auroc": 0.79, "sensitivity": 0.29, "log_odds": 20.0},
             "b|sft": {"auroc": 0.6, "sensitivity": 0.2, "log_odds": 1.0},
             "b|kto": {"auroc": 0.55, "sensitivity": 0.1, "log_odds": 30.0},
             "b|dpo": {"auroc": 0.61, "sensitivity": 0.22, "log_odds": 5.0}}
    result = mech.summarise(means, models=("a", "b"))
    cells = {(c["model"], c["arm"]): c for c in result["cells"]}
    assert result["n_cells"] == 4
    assert cells[("b", "kto")]["d_auroc"] == pytest.approx(-0.05)
    assert cells[("b", "kto")]["d_log_odds"] == pytest.approx(29.0)


def test_partial_correlation_matches_the_residual_definition():
    mech = _script("mechanism", "22_mechanism.py")
    rng = np.random.default_rng(7)
    z = rng.normal(size=60)
    x = z + rng.normal(size=60)
    y = 2 * z + x + rng.normal(size=60)

    def residual(v):
        return v - np.polyval(np.polyfit(z, v, 1), z)

    assert mech.partial(x, y, z) == pytest.approx(
        np.corrcoef(residual(x), residual(y))[0, 1], abs=1e-10)
