"""The risk-coverage figure: what it averages, and what it refuses to draw."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "11_risk_coverage_figure.py"


@pytest.fixture(scope="module")
def fig():
    spec = importlib.util.spec_from_file_location("risk_coverage_figure", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def risk_at_coverage(pairs, coverage):
    ordered = sorted(pairs, key=lambda t: -t[0])
    kept = max(1, int(round(coverage * len(ordered))))
    return 1.0 - float(np.mean([c for _, c in ordered[:kept]]))


def test_the_curve_averages_seeds_rather_than_pooling_their_items(fig):
    """Pooling items across seeds would draw one long run instead of the mean of five, and
    would understate the seed-to-seed spread the paper's statistics rest on."""
    good = {13: [(0.9, 1.0)] * 9 + [(0.1, 0.0)]}
    bad = {21: [(0.9, 0.0)] * 9 + [(0.1, 1.0)]}
    grid = np.array([1.0])
    mixed = fig.curve({**good, **bad}, grid, risk_at_coverage=risk_at_coverage)
    assert mixed[0] == pytest.approx(0.5)
    assert fig.curve(good, grid, risk_at_coverage=risk_at_coverage)[0] == pytest.approx(0.1)


def test_the_curve_is_evaluated_at_every_requested_coverage(fig):
    seeds = {13: [(0.9 - 0.01 * i, float(i < 8)) for i in range(10)]}
    grid = np.linspace(0.2, 1.0, 5)
    values = fig.curve(seeds, grid, risk_at_coverage=risk_at_coverage)
    assert len(values) == len(grid)
    assert values[0] <= values[-1], "abstaining on the least confident cannot raise risk here"


def test_a_missing_arm_stops_the_figure(fig, tmp_path, monkeypatch):
    """A curve silently absent from a panel reads as an arm that was never run, not as one
    whose files were not found."""
    monkeypatch.setattr(fig, "coverage_module", lambda: type("M", (), {
        "collect": staticmethod(lambda *a, **k: {}),
        "risk_at_coverage": staticmethod(risk_at_coverage)}))
    with pytest.raises(SystemExit):
        fig.main(["--runs", str(tmp_path), "--models", "lfm-1.2b",
                  "--out", str(tmp_path / "f.pdf")])
