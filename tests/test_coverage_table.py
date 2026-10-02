"""Risk at a coverage threshold: the deployment translation of the ranking claim."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "10_coverage_table.py"


@pytest.fixture(scope="module")
def coverage():
    spec = importlib.util.spec_from_file_location("coverage_table", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_risk_is_measured_on_the_items_the_model_keeps(coverage):
    """Ten items, eight correct, and the two errors are the least confident. Abstaining on
    the least confident fifth must remove them, which is the whole point of the metric."""
    pairs = [(0.9 - 0.05 * i, 1.0) for i in range(8)] + [(0.2, 0.0), (0.1, 0.0)]
    assert coverage.risk_at_coverage(pairs, 1.0) == pytest.approx(0.2)
    assert coverage.risk_at_coverage(pairs, 0.8) == pytest.approx(0.0)

    # A model whose confidence carries no information gains nothing by abstaining.
    uninformative = [(0.5, 1.0)] * 8 + [(0.5, 0.0)] * 2
    assert coverage.risk_at_coverage(uninformative, 0.5) == pytest.approx(0.2)


def test_confidence_order_decides_what_is_kept_not_file_order(coverage):
    pairs = [(0.1, 0.0), (0.95, 1.0), (0.2, 0.0), (0.9, 1.0)]
    assert coverage.risk_at_coverage(pairs, 0.5) == pytest.approx(0.0)


def test_an_empty_set_is_an_error_not_a_perfect_score(coverage):
    with pytest.raises(ValueError):
        coverage.risk_at_coverage([], 0.8)


def test_an_arm_missing_an_evaluation_set_is_not_collected(coverage, tmp_path):
    """Pooling whatever sets happen to be present would compare arms on different items,
    and the difference would be attributed to the objective."""
    run = tmp_path / "main__lfm-1.2b__sft__eps0.0__tau0.2__n8000__s13"
    (run / "eval").mkdir(parents=True)
    row = json.dumps({"probs": [0.7, 0.3], "correct": True}) + "\n"
    for name in coverage.SETS[:-1]:
        (run / "eval" / f"predictions_{name}.jsonl").write_text(row)
    assert coverage.pooled_items(run) is None

    (run / "eval" / f"predictions_{coverage.SETS[-1]}.jsonl").write_text(row)
    assert len(coverage.pooled_items(run)) == len(coverage.SETS)


def test_seeds_are_averaged_not_pooled(coverage):
    """Pooling items across seeds first would treat five runs of one arm as one larger
    run, which is the resampling error this programme corrected once already."""
    good = [(0.9, 1.0)] * 9 + [(0.1, 0.0)]
    bad = [(0.9, 0.0)] * 9 + [(0.1, 1.0)]
    collected = {("lfm-1.2b", "sft"): {13: good, 21: bad}}
    text = coverage.coverage_table(collected, ["lfm-1.2b"], coverages=(1.0,),
                                   arms=(("sft", "SFT"),), seeds_required=2)
    assert "0.500" in text, "the mean of 0.1 and 0.9, not the risk of the pooled items"


def test_the_number_does_not_depend_on_the_order_rows_were_written_in(coverage):
    import random

    pairs = [(0.5, 1.0)] * 8 + [(0.5, 0.0)] * 2 + [(0.9, 1.0), (0.1, 0.0)]
    shuffled = pairs[:]
    random.Random(7).shuffle(shuffled)
    for c in (1.0, 0.8, 0.5, 0.25):
        assert coverage.risk_at_coverage(pairs, c) == pytest.approx(
            coverage.risk_at_coverage(shuffled, c))


def test_the_curves_are_pooled_over_the_same_declared_seeds(coverage, tmp_path):
    """The figure and the table say five seeds. One SFT arm has fifteen runs on disk, so
    without a filter the SFT curve is a fifteen-seed mean drawn beside five-seed curves.
    """
    from sentalign.config import SEEDS

    def write(arm, seed, correct):
        d = tmp_path / (f"main__lfm-1.2b__{arm}__eps0.0__tau0.2__n8000__s{seed}") / "eval"
        d.mkdir(parents=True)
        for name in coverage.SETS:
            rows = [{"correct": c, "probs": [0.9, 0.1]} for c in correct]
            (d / f"predictions_{name}.jsonl").write_text(
                "\n".join(json.dumps(r) for r in rows) + "\n")

    for seed in list(SEEDS):
        write("sft", seed, [True] * 40 + [False] * 20)
        write("kto", seed, [True] * 40 + [False] * 20)
    for seed in (101, 102, 103):
        write("sft", seed, [True] * 60)

    collected = coverage.collect(tmp_path, {"lfm-1.2b"})
    assert sorted(collected[("lfm-1.2b", "sft")]) == sorted(SEEDS)
    assert (sorted(collected[("lfm-1.2b", "sft")])
            == sorted(collected[("lfm-1.2b", "kto")])), "one seed set for every arm"


def test_an_arm_short_of_the_declared_seeds_is_not_rendered(coverage):
    """A two-seed row beside five-seed rows is the same comparability defect as a
    fifteen-seed one: this table's content is the comparison between curves, so a row that
    is estimated differently is not comparable to the rest. It is left out, not shortened.
    """
    from sentalign.config import SEEDS

    full = {s: [(0.9, 1.0)] * 20 + [(0.1, 0.0)] * 10 for s in SEEDS}
    short = {s: [(0.9, 1.0)] * 20 + [(0.1, 0.0)] * 10 for s in list(SEEDS)[:2]}
    collected = {("m", "sft"): full, ("m", "kto"): full, ("m", "simpo"): short}
    text = coverage.coverage_table(collected, ["m"], arms=(("sft", "SFT"),
                                                           ("kto", "KTO"),
                                                           ("simpo", "SimPO")))
    assert "SFT" in text and "KTO" in text
    assert "SimPO" not in text, "an arm with two of five seeds must not be rendered"
