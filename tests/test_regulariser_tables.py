"""The LaTeX tables in the manuscript, generated from the statistics file.

A table is where a corrected analysis quietly becomes an uncorrected claim, so each test
breaks the corresponding property in ``scripts/08_regulariser_tables.py`` and watches it
fail.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "08_regulariser_tables.py"


@pytest.fixture(scope="module")
def tables():
    spec = importlib.util.spec_from_file_location("regulariser_tables", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def row(model, first, second, metric, diff, p, p_holm, name="mean4"):
    return {"model": model, "set": name, "first": first, "second": second,
            "metric": metric, "n": 5, "seeds": [13, 21, 34, 55, 89], "mean_diff": diff,
            "ci95": [diff - 0.01, diff + 0.01], "t": 9.0, "p": p,
            "favours_first": True, "p_holm": p_holm}


@pytest.fixture
def families():
    return {
        "models": ["lfm-1.2b"], "endpoint": "mean4",
        "sets": ["ambig_eval", "r1_test", "r2_test", "sst_dev_validated"],
        "P1_primary_auroc": [row("lfm-1.2b", "kto-dreg1.0", "kto", "auroc",
                                 0.1538, 0.0007, 0.0030)],
        "P2_secondary_eaurc": [row("lfm-1.2b", "kto-dreg1.0", "kto", "eaurc",
                                   -0.1147, 0.0005, 0.0020)],
        "P3_vs_sft": [row("lfm-1.2b", "kto-dreg1.0", "sft", "eaurc", -0.0108, 0.0011, 0.0125),
                      row("lfm-1.2b", "kto-dreg1.0", "sft", "acc", 0.0233, 0.0002, 0.0022)],
        "A1_soft_vs_hard": [row("lfm-1.2b", "kto-dreg1.0", "kto-dreg-hard", "auroc",
                                0.0058, 0.2999, 0.8997),
                            row("lfm-1.2b", "kto-dreg1.0", "kto-dreg-hard", "ece",
                                -0.0273, 0.2187, 0.8997)],
        "S1_per_set_repair": [row("lfm-1.2b", "kto-dreg1.0", "kto", metric, value, 0.001, holm,
                                  name=name)
                              for name, holm in (("ambig_eval", 0.0008), ("r1_test", 0.0372),
                                                 ("r2_test", 0.0372), ("sst_dev_validated", 0.0935))
                              for metric, value in (("auroc", 0.1169), ("eaurc", -0.0760))],
    }


def test_the_numbers_are_read_from_the_statistics_file(tables, families):
    """Not retyped: changing the file changes the table, which is the only guarantee that
    a rerun of the analysis reaches the paper."""
    text = tables.primary_table(families, ["lfm-1.2b"], "mean4")
    assert "+0.154" in text and "[+0.144, +0.164]" in text

    families["P1_primary_auroc"][0]["mean_diff"] = 0.0421
    changed = tables.primary_table(families, ["lfm-1.2b"], "mean4")
    assert "+0.042" in changed and "+0.154" not in changed


def test_significance_marks_follow_the_corrected_p(tables, families):
    """The per-set family corrects over 48 tests, where raw p-values below 0.05 routinely
    do not survive. A table starring the raw value contradicts the text beside it."""
    families["P1_primary_auroc"][0]["p"] = 0.0007          # would earn three stars raw
    families["P1_primary_auroc"][0]["p_holm"] = 0.0300     # survives only at 0.05
    text = tables.primary_table(families, ["lfm-1.2b"], "mean4")
    assert r"+0.154$^{*}$" in text
    assert r"$^{**}$" not in text.split(r"\midrule")[1].split("&")[2]
    assert "0.030" in text and "0.001" not in text


def test_the_ablation_is_reported_uncorrected_and_unmarked(tables, families):
    """It is declared with intervals and uncorrected p. Marking those cells would claim a
    correction that was never applied to them."""
    text = tables.ablation_table(families, "mean4")
    assert "$^{" not in text, "no significance marks in the ablation table"
    assert "0.300" in text and "0.900" not in text, "the uncorrected p is the one shown"


def test_a_planned_cell_with_no_data_stops_the_table(tables, families):
    families["P2_secondary_eaurc"] = []
    with pytest.raises(SystemExit):
        tables.primary_table(families, ["lfm-1.2b"], "mean4")
    text = tables.primary_table(families, ["lfm-1.2b"], "mean4", allow_missing=True)
    assert tables.PENDING in text


def test_model_names_come_from_the_registry(tables):
    from sentalign.modeling import MODEL_REGISTRY, display_name

    for key in ("lfm-1.2b", "qwen-2b", "smollm3-3b"):
        published = MODEL_REGISTRY[key]["hf_id"].split("/")[-1]
        assert tables.model_label(key) == display_name(key)
        assert published.startswith(tables.model_label(key)), "not the published name"
    assert display_name("smollm3-3b") == "SmolLM3-3B", "the -Base suffix is constant here"


def test_arm_names_are_the_ones_a_reader_knows(tables):
    assert tables.arm_label("kto-dreg1.0") == "KTO+reg"
    assert tables.arm_label("kto-dreg-hard") == "KTO+reg (hard)"
    assert tables.arm_label("rdpo") == "R-DPO"
    assert tables.arm_label("sft") == "SFT"
    assert "dreg" not in tables.arm_label("rdpo-dreg1.0")


def test_every_table_is_well_formed_latex(tables, families):
    for text in (tables.primary_table(families, ["lfm-1.2b"], "mean4"),
                 tables.versus_sft_table(families, ["lfm-1.2b"], "mean4"),
                 tables.ablation_table(families, "mean4"),
                 tables.per_set_table(families, families["sets"])):
        assert text.count(r"\begin{tabular}") == text.count(r"\end{tabular}") == 1
        assert text.rstrip().endswith(r"\end{table}")
        assert any(size in text for size in ("\\small", "\\footnotesize", "\\scriptsize")), \
            'the effect cells overflow a single-column text block at full size'
        columns = text.split(r"\begin{tabular}{")[1].split("}")[0]
        header = text.split(r"\toprule")[1].split(r"\\")[0]
        body = [line for line in text.splitlines()
                if line.endswith(r"\\") and "&" in line]
        for line in body:
            assert line.count("&") == len(columns) - 1, line
        assert header.count("&") == len(columns) - 1


def test_the_merged_table_names_the_family_each_panel_was_corrected_within(tables, families):
    """Two families with different corrections in one table is exactly where a reader can
    be misled about how strictly a p-value was adjusted."""
    text = tables.differences_table(families, ["lfm-1.2b"], "mean4")
    for key, panels in (("P1_primary_auroc", ("A.",)), ("P2_secondary_eaurc", ("B.",)),
                        ("P3_vs_sft", ("C.", "D."))):
        for panel in panels:
            header = next(l for l in text.splitlines() if l.startswith(f"\\multicolumn")
                          and panel in l)
            assert f"Holm over {len(families[key])}" in header, header
    for panel in ("A.", "B.", "C.", "D."):
        assert panel in text
    columns = text.split(r"\begin{tabular}{")[1].split("}")[0]
    for line in text.splitlines():
        if line.endswith(r"\\") and "&" in line and "multicolumn" not in line:
            assert line.count("&") == len(columns) - 1, line


def test_the_panel_headers_count_the_family_they_were_corrected_within(tables, families):
    """A panel that shows eight rows while claiming a correction over six misreports the
    test. The count has to come from the family, so the two cannot drift apart when a
    model is added.

    The fixture declares one twin comparison and a vs-SFT family of two, so a hardcoded
    "6" or "12" fails here; growing the family below moves the header with it.
    """
    text = tables.differences_table(families, ["lfm-1.2b"], "mean4")
    assert "Holm over 1" in text and "Holm over 2" in text
    assert "Holm over 6" not in text and "Holm over 12" not in text

    grown = dict(families)
    grown["P1_primary_auroc"] = families["P1_primary_auroc"] + [
        row("qwen-2b", "kto-dreg1.0", "kto", "auroc", 0.1470, 0.0004, 0.0009)]
    grown["P2_secondary_eaurc"] = families["P2_secondary_eaurc"] + [
        row("qwen-2b", "kto-dreg1.0", "kto", "eaurc", -0.1058, 0.0003, 0.0007)]
    grown["P3_vs_sft"] = families["P3_vs_sft"] + [
        row("qwen-2b", "kto-dreg1.0", "sft", "eaurc", -0.0103, 0.0090, 0.1080),
        row("qwen-2b", "kto-dreg1.0", "sft", "acc", 0.0285, 0.0027, 0.0262)]
    text = tables.differences_table(grown, ["lfm-1.2b", "qwen-2b"], "mean4")
    assert "Holm over 2" in text and "Holm over 4" in text
    assert "Holm over 1," not in text


def test_the_captions_count_their_families_too(tables, families):
    """The same drift, one level up: the caption of each standalone table names the family
    size, and a reader who checks it against the rows must find them equal."""
    assert "family of 1." in tables.primary_table(families, ["lfm-1.2b"], "mean4")
    assert "family of 2." in tables.versus_sft_table(families, ["lfm-1.2b"], "mean4")
    grown = dict(families)
    grown["P1_primary_auroc"] = families["P1_primary_auroc"] + [
        row("qwen-2b", "kto-dreg1.0", "kto", "auroc", 0.1470, 0.0004, 0.0009)]
    grown["P2_secondary_eaurc"] = families["P2_secondary_eaurc"] + [
        row("qwen-2b", "kto-dreg1.0", "kto", "eaurc", -0.1058, 0.0003, 0.0007)]
    assert "family of 2." in tables.primary_table(grown, ["lfm-1.2b", "qwen-2b"], "mean4")


def test_the_supplementary_caption_counts_its_own_family(tables, families):
    """The per-set caption said 48 while rendering 64 rows, for the same reason the panel
    headers did: a literal that was right for the two-model run and never moved. The paper
    claims every table states the family it was corrected within, so this one has to be
    computed too."""
    text = tables.per_set_table(families, ["ambig_eval", "r1_test", "r2_test",
                                           "sst_dev_validated"], allow_missing=True)
    n = len(families["S1_per_set_repair"])
    assert f"over all {n} tests" in text
    assert "over all 48 tests" not in text or n == 48
