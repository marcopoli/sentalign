"""The lambda sensitivity table."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "14_lambda_sweep.py"


@pytest.fixture(scope="module")
def sweep():
    spec = importlib.util.spec_from_file_location("lambda_sweep", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cell(v):
    return {"auroc": v, "eaurc": 0.1, "acc": 0.7, "ece": 0.1}


def test_every_row_uses_the_seeds_all_rows_share(sweep):
    """lambda = 1 has five seeds and the sweep points three. Averaging the five-seed row
    beside three-seed rows compares estimates of different precision, so the two extra
    seeds must be dropped, and they are given a value here that would show if they were
    not."""
    mean = {("lfm-1.2b", "mean4", arm): {s: cell(0.7) for s in (13, 21, 34)}
            for arm, _ in sweep.ROWS}
    mean[("lfm-1.2b", "mean4", "kto-dreg1.0")].update({55: cell(0.0), 89: cell(0.0)})
    result = sweep.sweep(mean)
    assert result["seeds"] == [13, 21, 34]
    row = next(r for r in result["rows"] if r["arm"] == "kto-dreg1.0")
    assert row["auroc"] == pytest.approx(0.7)


def test_a_missing_sweep_point_is_an_error(sweep):
    mean = {("lfm-1.2b", "mean4", arm): {13: cell(0.7)} for arm, _ in sweep.ROWS
            if arm != "kto-dreg0.3"}
    with pytest.raises(SystemExit):
        sweep.sweep(mean)
