"""The deferral cost analysis: the paper's human-centred consequence, priced."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "15_deferral_cost.py"


@pytest.fixture(scope="module")
def dc():
    spec = importlib.util.spec_from_file_location("deferral_cost", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_threshold_search_finds_the_cheapest_policy(dc):
    """Against brute force over every threshold, including deferring nothing and
    everything, with heavy ties so the tie rule is exercised."""
    rng = np.random.default_rng(3)
    for _ in range(60):
        n = int(rng.integers(20, 90))
        conf = rng.integers(0, 8, n) / 7.0
        correct = (rng.random(n) < 0.3 + 0.6 * conf).astype(float)
        r = float(rng.choice([0.5, 2.0, 5.0, 20.0]))
        t = dc.choose_threshold(conf, correct, r)
        got = dc.policy_cost(conf, correct, t, r)["cost"]
        candidates = list(np.unique(conf)) + [np.inf]
        best = min(dc.policy_cost(conf, correct, c, r)["cost"] for c in candidates)
        assert got == pytest.approx(best)


def test_tied_confidences_are_deferred_or_kept_together(dc):
    """A model that puts the same confidence on everything has nothing to rank by, so its
    only choices are all or nothing, whatever order the items arrive in."""
    conf = [0.99] * 10
    correct = [1.0] * 7 + [0.0] * 3
    for r in (2.0, 5.0):
        t = dc.choose_threshold(conf, correct, r)
        deferred = dc.policy_cost(conf, correct, t, r)["deferred"]
        assert deferred in (0.0, 1.0)
    assert dc.policy_cost(conf, correct, dc.choose_threshold(conf, correct, 2.0),
                          2.0)["cost"] == pytest.approx(0.6), "0.3 * 2 beats reviewing all"
    assert dc.policy_cost(conf, correct, dc.choose_threshold(conf, correct, 5.0),
                          5.0)["cost"] == pytest.approx(1.0), "0.3 * 5 is worse than 1"


def test_a_perfect_ranker_pays_only_for_its_errors(dc):
    conf = [0.9] * 8 + [0.4] * 2
    correct = [1.0] * 8 + [0.0] * 2
    t = dc.choose_threshold(conf, correct, 10.0)
    out = dc.policy_cost(conf, correct, t, 10.0)
    assert out["cost"] == pytest.approx(0.2) and out["wrong_kept"] == 0.0


def test_the_split_depends_on_the_item_only(dc):
    """Every arm of a model must be split identically, or two arms would be compared on
    different items; and the two halves must partition the items."""
    keys = [f"ambig_eval/r1-{i:07d}" for i in range(2000)]
    halves = [dc.half(k) for k in keys]
    assert halves == [dc.half(k) for k in keys]
    assert 0.45 < np.mean(halves) < 0.55


def test_the_threshold_is_paid_for_on_items_it_was_not_chosen_on(dc):
    """Choosing and scoring on the same items rewards a model for noise in its own
    confidences. With every error in one half and none in the other, the threshold
    chosen on the clean half defers nothing, so the errors in the other half are paid in
    full; scoring in-sample would have hidden them."""
    items = []
    for i in range(4000):
        key = f"s/{i}"
        wrong = dc.half(key) == 0 and i % 4 == 0
        items.append((key, 0.6 if wrong else 0.9, 0.0 if wrong else 1.0))
    r = 10.0
    out = dc.cross_fitted(items, r)
    in_sample = dc.policy_cost([c for _, c, _ in items], [k for _, _, k in items],
                               dc.choose_threshold([c for _, c, _ in items],
                                                   [k for _, _, k in items], r), r)
    assert out["cost"] > in_sample["cost"] + 0.1


def test_a_threshold_between_two_groups_sits_between_them(dc):
    """Fitted on errors at 0.5 and hits at 0.9, the threshold must separate the two groups
    without claiming either boundary: a new item at 0.8 was never seen, and deferring it
    only because the fitting half had no item between 0.5 and 0.9 would charge the policy
    for a gap in its sample."""
    t = dc.choose_threshold([0.5] * 3 + [0.9] * 7, [0.0] * 3 + [1.0] * 7, 10.0)
    assert 0.5 < t < 0.9
    assert dc.policy_cost([0.8], [1.0], t, 10.0)["deferred"] == 0.0
