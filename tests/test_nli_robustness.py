"""The second-task table: what it may claim, and what it must not imply."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "12_nli_robustness.py"


@pytest.fixture(scope="module")
def nli():
    spec = importlib.util.spec_from_file_location("nli_robustness", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def arm(auroc, acc, n=5, spread=0.002):
    return {13 + i: {"auroc": auroc + i * spread, "acc": acc + i * spread,
                     "eaurc": 0.2, "ece": 0.1} for i in range(n)}


@pytest.fixture
def scores():
    return {"sft": arm(0.633, 0.611), "kto": arm(0.541, 0.609), "rdpo": arm(0.547, 0.601)}


def test_the_repair_panel_is_absent_until_the_regularised_arms_exist(nli, scores):
    """Half a replication must not read as a whole one. Rendering an empty panel B, or
    quietly dropping it without saying so, both invite the reader to assume the repair was
    tested on this task."""
    table, missing = nli.nli_table(scores, "lfm-1.2b")
    assert "A. Against SFT" in table and "B. Regularised" not in table
    assert "kto-dreg1.0" in missing and "rdpo-dreg1.0" in missing

    scores["kto-dreg1.0"] = arm(0.630, 0.620)
    table, missing = nli.nli_table(scores, "lfm-1.2b")
    assert "B. Regularised" in table and "KTO+reg" in table
    assert "kto-dreg1.0" not in missing


def test_the_degradation_is_reported_against_the_baseline_on_the_same_task(nli, scores):
    table, _ = nli.nli_table(scores, "lfm-1.2b")
    assert "-0.092" in table or "-0.091" in table, "KTO's AUROC loss against NLI SFT"
    assert "vs SFT" in table


def test_marks_read_the_corrected_p_not_the_raw_one(nli, scores):
    rows = nli.family(scores, [(a, "sft", "auroc") for a in ("kto", "rdpo")])
    assert all("p_holm" in r for r in rows)
    assert all(r["p_holm"] >= r["p"] for r in rows), "Holm can only raise a p-value"


def test_a_seed_missing_a_set_does_not_enter_the_mean(nli, tmp_path):
    """An arm averaged over a different subset of evaluation sets is a different quantity
    wearing the same name."""
    run = tmp_path / "nli__lfm-1.2b__kto__eps0.0__tau0.2__n8000__s13"
    (run / "eval").mkdir(parents=True)
    rows = "\n".join(json.dumps({"probs": [0.6, 0.4], "correct": bool(i % 2)})
                     for i in range(60))
    (run / "eval" / f"predictions_{nli.SETS[0]}.jsonl").write_text(rows)
    assert nli.load(tmp_path, "lfm-1.2b") == {}

    (run / "eval" / f"predictions_{nli.SETS[1]}.jsonl").write_text(rows)
    assert set(nli.load(tmp_path, "lfm-1.2b")) == {"kto"}


def test_too_few_seeds_is_no_comparison(nli):
    thin = {"sft": arm(0.63, 0.61, n=2), "kto": arm(0.54, 0.61, n=2)}
    assert nli.paired(thin, "kto", "sft", "auroc") is None


def test_a_cell_whose_raw_p_survives_but_whose_corrected_p_does_not_is_unmarked(nli):
    """The panel corrects twelve tests. A cell starred on its uncorrected p would claim a
    significance the table's own correction withdrew."""
    row = {"mean_diff": -0.02, "ci95": [-0.035, -0.005], "p": 0.02, "p_holm": 0.24}
    assert "$^{" not in nli._cell(row)
    row["p_holm"] = 0.004
    assert r"$^{**}$" in nli._cell(row)
