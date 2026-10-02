"""The objective-landscape table: absolute levels, and what may be compared with what."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "09_landscape_table.py"


@pytest.fixture(scope="module")
def landscape():
    spec = importlib.util.spec_from_file_location("landscape_table", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def means():
    def cell(acc, auroc, ece, n=5):
        return {"seeds": list(range(n)), "acc": acc, "auroc": auroc, "ece": ece}
    return {
        ("lfm-1.2b", "sft"): cell(0.7235, 0.7672, 0.1048),
        ("lfm-1.2b", "kto"): cell(0.7215, 0.6261, 0.2754),
        ("lfm-1.2b", "simpo"): cell(0.7330, 0.7727, 0.1131),
        ("lfm-1.2b", "kto-dreg1.0"): cell(0.7441, 0.7799, 0.0765),
    }


def test_the_cells_are_the_computed_means(landscape, means):
    text = landscape.landscape_table(means, ["lfm-1.2b"])
    for value in ("0.724", "0.767", "0.105", "0.626", "0.780", "0.076"):
        assert value in text, value


def test_the_leading_value_is_bold_and_reads_each_metric_in_its_own_direction(landscape, means):
    text = landscape.landscape_table(means, ["lfm-1.2b"])
    assert r"\textbf{0.744}" in text, "highest accuracy"
    assert r"\textbf{0.780}" in text, "highest AUROC"
    assert r"\textbf{0.076}" in text, "lowest ECE, not the highest"
    assert r"\textbf{0.275}" not in text, "the worst ECE must not be marked as leading"


def test_a_row_with_too_few_seeds_is_marked_and_cannot_take_a_column(landscape, means):
    """An average over one seed and an average over five are not the same quantity. The
    third model is mid-programme, so this is the row the reader will meet."""
    means[("lfm-1.2b", "rdpo-dreg1.0")] = {"seeds": [13], "acc": 0.99, "auroc": 0.99,
                                           "ece": 0.001}
    text = landscape.landscape_table(means, ["lfm-1.2b"])
    assert r"0.990$^{\dagger}$" in text
    assert r"\textbf{0.990}" not in text, "a one-seed row took a column"
    assert r"\textbf{0.744}" in text, "the leader among complete rows is unchanged"


def test_every_row_has_the_same_number_of_columns(landscape, means):
    text = landscape.landscape_table(means, ["lfm-1.2b", "qwen-2b"])
    columns = text.split(r"\begin{tabular}{")[1].split("}")[0]
    assert len(columns) == 1 + 3 * 2
    for line in text.splitlines():
        if line.endswith(r"\\") and "&" in line and "multicolumn" not in line:
            assert line.count("&") == len(columns) - 1, line


def test_an_arm_no_model_ran_is_left_out_entirely(landscape, means):
    text = landscape.landscape_table(means, ["lfm-1.2b"])
    assert "MixDPO" not in text and "GR-DPO" not in text
    assert "KTO+reg" in text


def test_an_arm_the_table_does_not_render_cannot_take_a_column(landscape, means):
    """The run programme also holds exploratory objectives from a separate study. One of
    them has the lowest calibration error on both models, and scanning every directory for
    the leader left the ECE column with no bold cell and nothing to say why."""
    means[("lfm-1.2b", "cspo_kl")] = {"seeds": list(range(5)), "acc": 0.99,
                                      "auroc": 0.99, "ece": 0.001}
    text = landscape.landscape_table(means, ["lfm-1.2b"])
    assert "cspo" not in text
    assert r"\textbf{0.076}" in text, "the leader among rendered rows lost its mark"
    assert r"\textbf{0.744}" in text and r"\textbf{0.780}" in text


def test_an_empty_table_is_refused_rather_than_written(landscape, tmp_path):
    """A model key that matches nothing produced a valid, empty table that overwrote a
    correct one. An empty result here is a wrong invocation, not a finding."""
    (tmp_path / "runs").mkdir()
    with pytest.raises(SystemExit):
        landscape.main(["--runs", str(tmp_path / "runs"), "--models", "lfm-1.2b qwen-2b",
                        "--out", str(tmp_path / "out")])
    assert not (tmp_path / "out" / "table_landscape.tex").exists()


def test_the_table_is_sized_to_the_text_block(landscape, means):
    """Ten columns at full size overflow the page and the overflow is silent in the PDF."""
    assert r"\small" in landscape.landscape_table(means, ["lfm-1.2b"])
    wide = landscape.landscape_table(means, ["lfm-1.2b", "qwen-2b", "smollm3-3b"])
    assert r"\footnotesize" in wide and r"\tabcolsep" in wide


def test_every_arm_is_averaged_over_the_same_declared_seeds(landscape):
    """A seed-variance study left fifteen SFT runs on one model while every other arm has
    five. Averaging fifteen against five puts a better-estimated version of one objective
    beside the others and calls the difference an effect of the objective, which is the
    comparability defect this project keeps finding. The guard that existed marked an arm
    with *fewer* seeds than the design and said nothing about more.
    """
    from sentalign.config import SEEDS

    class Stub:
        ENDPOINT = "mean4"

        def load(self, runs, models):
            return {}, {}, []

        def average_over_sets(self, scores):
            def cell(v):
                return {"acc": v, "auroc": v, "ece": v}
            extra = {s: cell(1.0) for s in (101, 102, 103, 104, 105)}
            return {
                ("lfm-1.2b", "mean4", "sft"): {**{s: cell(0.5) for s in SEEDS}, **extra},
                ("lfm-1.2b", "mean4", "kto"): {s: cell(0.5) for s in SEEDS},
            }

    means = landscape.arm_means(Path("unused"), ["lfm-1.2b"], stats=Stub())
    assert means[("lfm-1.2b", "sft")]["seeds"] == sorted(SEEDS)
    assert means[("lfm-1.2b", "sft")]["acc"] == pytest.approx(0.5), (
        "the ten extra runs must not enter the mean")
    assert (means[("lfm-1.2b", "sft")]["seeds"]
            == means[("lfm-1.2b", "kto")]["seeds"]), "every arm, the same seeds"
