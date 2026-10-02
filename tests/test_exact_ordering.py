"""The exact score separates what the stored confidence ties at 1.0."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _script(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_exact_score_orders_items_the_stored_confidence_ties():
    exact = _script("exact_ordering", "20_exact_ordering.py")
    stats = _script("regulariser_stats", "06_regulariser_stats.py")
    # Four items whose top-label margins (40, 45, 50, 55) all store a confidence of 1.0,
    # but whose true confidences are ordered; the two errors have the smaller margins.
    logits = [[0.0, -m, -m - 5.0, -m - 9.0] for m in (40.0, 45.0, 50.0, 55.0)]
    correct = [0.0, 0.0, 1.0, 1.0]
    stored = [float(np.max(np.exp(z) / np.exp(z).sum())) for z in map(np.array, logits)]
    assert stored == [1.0] * 4, "the premise: a double cannot tell these items apart"
    scores = [exact.exact_score(z) for z in logits]
    assert scores == sorted(scores), "strictly increasing in the true top probability"
    assert stats.auroc(stored, correct) == 0.5
    assert stats.auroc(scores, correct) == 1.0


def test_exact_score_agrees_with_the_stored_one_where_nothing_saturates():
    exact = _script("exact_ordering", "20_exact_ordering.py")
    rng = np.random.default_rng(0)
    for _ in range(50):
        z = rng.normal(size=4)
        p = np.exp(z) / np.exp(z).sum()
        assert np.isclose(exact.exact_score(z), -np.log((1 - p.max()) / p.max()))
