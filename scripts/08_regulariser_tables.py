#!/usr/bin/env python3
"""LaTeX tables for the regulariser result, built from the statistics file.

Every number in the manuscript comes from ``results/regulariser_stats.json``, which
``scripts/06_regulariser_stats.py`` writes, so a table cannot disagree with the analysis
that produced it and a rerun of the statistics updates the paper by regenerating rather
than by retyping. Three properties are worth stating because the alternative is a table
that looks right:

*   Significance marks read ``p_holm``, never the uncorrected ``p``. The per-set family
    corrects over 48 tests, where several raw p-values below 0.05 do not survive, and a
    table starring those would contradict the text on the same page.
*   The soft-versus-hard ablation is reported with intervals and uncorrected p, exactly
    as it is declared, so its cells carry no stars at all.
*   A planned cell with no data stops the table. A family that shrinks silently is
    corrected less strictly than the one that was declared, and a table missing a row is
    the same claim made quieter.

Model names come from the registry rather than from this file, so a table cannot name a
checkpoint the runs did not use.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sentalign.modeling import MODEL_REGISTRY, display_name                      # noqa: E402

#: Objective surface forms. The arm name on disk is not the name a reader knows.
OBJECTIVE_LABEL = {"sft": "SFT", "kto": "KTO", "rdpo": "R-DPO", "ipo": "IPO",
                   "dpo": "DPO", "sft_soft": "SFT-soft", "grdpo": "GR-DPO",
                   "mixdpo": "MixDPO", "simpo": "SimPO", "alphapo": "AlphaPO"}
# Whether a higher value of the metric is the better one. Every header and panel title
# carries the arrow, so a reader never has to remember which way E-AURC or ECE runs.
HIGHER_BETTER = {"auroc": True, "eaurc": False, "acc": True, "ece": False}
ARROW = {True: r"$\uparrow$", False: r"$\downarrow$"}
METRIC_LABEL = {"auroc": r"$\Delta$AUROC", "eaurc": r"$\Delta$E-AURC",
                "acc": r"$\Delta$accuracy", "ece": r"$\Delta$ECE"}
METRIC_LABEL = {k: f"{v}{ARROW[HIGHER_BETTER[k]]}" for k, v in METRIC_LABEL.items()}
# For a difference, the arrow is the sign that favours the first-named arm.
DIRECTION_NOTE = (" An arrow gives the sign of a difference that favours the first-named "
                  "arm: $\\uparrow$ where higher is better, $\\downarrow$ where lower is.")
SET_LABEL = {"ambig_eval": "Ambiguous", "r1_test": "DynaSent R1",
             "r2_test": "DynaSent R2", "sst_dev_validated": "SST-dev"}
PENDING = r"\pending"


def model_label(key: str) -> str:
    return display_name(key)


def arm_label(arm: str) -> str:
    base, _, suffix = arm.partition("-dreg")
    name = OBJECTIVE_LABEL.get(base, base.upper())
    if not suffix:
        return name
    return f"{name}+reg (hard)" if suffix == "-hard" else f"{name}+reg"


def stars(p: float) -> str:
    return (r"$^{***}$" if p < 0.001 else r"$^{**}$" if p < 0.01
            else r"$^{*}$" if p < 0.05 else "")


def index_rows(rows) -> dict:
    return {(r["model"], r["set"], r["first"], r["second"], r["metric"]): r for r in rows}


def effect(row, *, adjusted: bool, digits: int = 3) -> str:
    """The difference with its interval, marked from the p-value that governs it."""
    mark = stars(row["p_holm"]) if adjusted else ""
    return (f"{row['mean_diff']:+.{digits}f}{mark} "
            f"[{row['ci95'][0]:+.{digits}f}, {row['ci95'][1]:+.{digits}f}]")


def p_value(row, *, adjusted: bool) -> str:
    p = row["p_holm"] if adjusted else row["p"]
    return r"$<$0.001" if p < 0.001 else f"{p:.3f}"


def take(index, key, *, allow_missing: bool):
    row = index.get(key)
    if row is None and not allow_missing:
        model, name, first, second, metric = key
        raise SystemExit(f"no result for {model} {name} {first} vs {second} {metric}; "
                         "rerun the statistics or pass --allow-missing")
    return row


def _table(caption: str, label: str, spec: str, header, body, size=r"\small") -> str:
    # The effect cells carry a difference, a mark and an interval, which overflows the
    # text block of a single-column journal at full size. Set by the generator rather
    # than left to the author to remember per table.
    return "\n".join([
        r"\begin{table}[t]", r"\centering", f"\\caption{{{caption}}}",
        f"\\label{{{label}}}", size, f"\\begin{{tabular}}{{{spec}}}", r"\toprule",
        " & ".join(header) + r" \\", r"\midrule", *body, r"\bottomrule",
        r"\end{tabular}", r"\end{table}"])


def _panel(index, plan, endpoint, metric, *, allow_missing) -> list[str]:
    """One metric's rows, as a panel. Two metrics side by side needs six columns and
    overflows a single-column text block; stacked, the same numbers fit at full size."""
    body = []
    for model, comparisons in plan:
        for i, (first, second) in enumerate(comparisons):
            row = take(index, (model, endpoint, first, second, metric),
                       allow_missing=allow_missing)
            body.append(" & ".join([
                model_label(model) if i == 0 else "",
                f"{arm_label(first)} vs {arm_label(second)}",
                effect(row, adjusted=True) if row else PENDING,
                p_value(row, adjusted=True) if row else PENDING]) + r" \\")
    return body


def _plan_from(rows, models) -> list:
    """The comparisons the statistics file actually contains, in its own order."""
    plan = []
    for model in models:
        pairs = []
        for r in rows:
            if r["model"] == model and (r["first"], r["second"]) not in pairs:
                pairs.append((r["first"], r["second"]))
        if pairs:
            plan.append((model, pairs))
    return plan


def primary_table(families, models, endpoint, *, allow_missing=False) -> str:
    auroc, eaurc = (index_rows(families["P1_primary_auroc"]),
                    index_rows(families["P2_secondary_eaurc"]))
    plan = _plan_from(families["P1_primary_auroc"], models)
    body = [r"\multicolumn{4}{@{}l}{\emph{Primary endpoint: AUROC of confidence against "
            r"correctness, $\uparrow$ higher is better}} \\"]
    body += _panel(auroc, plan, endpoint, "auroc", allow_missing=allow_missing)
    body += [r"\addlinespace",
             r"\multicolumn{4}{@{}l}{\emph{Secondary: E-AURC, accuracy-corrected "
             r"selective risk, $\downarrow$ lower is better}} \\"]
    body += _panel(eaurc, plan, endpoint, "eaurc", allow_missing=allow_missing)
    return _table(
        "Distributional regularisation of a preference objective, against the same "
        "objective without it. Per-seed mean over the four evaluation sets, five seeds, "
        "paired $t$ on seeds, Holm correction within each family of "
        f"{len(families['P1_primary_auroc'])}." + DIRECTION_NOTE,
        "tab:repair", "llll",
        ["Model", "Comparison", "Difference [95\\% CI]", "$p_{\\mathrm{Holm}}$"],
        body)


def versus_sft_table(families, models, endpoint, *, allow_missing=False) -> str:
    rows = index_rows(families["P3_vs_sft"])
    plan = _plan_from(families["P3_vs_sft"], models)
    body = [r"\multicolumn{4}{@{}l}{\emph{E-AURC, $\downarrow$ lower is better}} \\"]
    body += _panel(rows, plan, endpoint, "eaurc", allow_missing=allow_missing)
    body += [r"\addlinespace",
             r"\multicolumn{4}{@{}l}{\emph{Accuracy, $\uparrow$ higher is better}} \\"]
    body += _panel(rows, plan, endpoint, "acc", allow_missing=allow_missing)
    return _table(
        "Regularised arms against the supervised fine-tuning baseline they were trained "
        "from. Same endpoint and seeds as Table~\\ref{tab:repair}; Holm correction over "
        f"the whole family of {len(families['P3_vs_sft'])}." + DIRECTION_NOTE,
        "tab:vs-sft", "llll",
        ["Model", "Arm vs SFT", "Difference [95\\% CI]", "$p_{\\mathrm{Holm}}$"],
        body)


def differences_table(families, models, endpoint, *, allow_missing=False) -> str:
    """Every confirmatory difference in one table, as four panels.

    The families have separate Holm corrections, so each panel names the size of the family
    it was corrected within. That size is read from the family itself rather than written
    into the string: a panel that shows eight rows while claiming a correction over six is
    a misreported test, and the only way the two cannot drift apart is for one to be
    computed from the other. Panels C and D are two metrics of a single family, so both
    name its whole size.

    They are merged into one float because two tables of identical shape cost a float and a
    caption each without telling the reader anything the panel headers do not.
    """
    auroc = index_rows(families["P1_primary_auroc"])
    eaurc = index_rows(families["P2_secondary_eaurc"])
    sft = index_rows(families["P3_vs_sft"])
    twin_plan = _plan_from(families["P1_primary_auroc"], models)
    sft_plan_rows = _plan_from(families["P3_vs_sft"], models)
    n_auroc = len(families["P1_primary_auroc"])
    n_eaurc = len(families["P2_secondary_eaurc"])
    n_sft = len(families["P3_vs_sft"])
    # Short enough to fit the text block: the reading of each metric is in the caption.
    panels = (
        (f"A. Against the twin, AUROC $\\uparrow$ (higher better), Holm over {n_auroc}",
         auroc, twin_plan, "auroc"),
        (f"B. Against the twin, E-AURC $\\downarrow$ (lower better), Holm over {n_eaurc}",
         eaurc, twin_plan, "eaurc"),
        (f"C. Against SFT, E-AURC $\\downarrow$ (lower better), Holm over {n_sft}",
         sft, sft_plan_rows, "eaurc"),
        (f"D. Against SFT, accuracy $\\uparrow$ (higher better), Holm over {n_sft}",
         sft, sft_plan_rows, "acc"),
    )
    body = []
    for i, (title, index, plan, metric) in enumerate(panels):
        if i:
            body.append(r"\addlinespace")
        body.append(f"\\multicolumn{{4}}{{@{{}}l}}{{\\emph{{{title}}}}} \\\\")
        body += _panel(index, plan, endpoint, metric, allow_missing=allow_missing)
    return _table(
        "The confirmatory comparisons. Per-seed mean over the four evaluation sets, five "
        "seeds, paired $t$ on seeds, Holm correction within each declared family. AUROC is "
        "of confidence against correctness and E-AURC is accuracy-corrected selective "
        "risk. Panels A and B ask whether the anchor improves its own objective; "
        "panels C and D ask whether the result beats the supervised baseline it was "
        "trained from, and are the two metrics of one family, corrected together."
        + DIRECTION_NOTE,
        "tab:differences", "llll",
        ["Model", "Comparison", "Difference [95\\% CI]", "$p_{\\mathrm{Holm}}$"],
        body, size=r"\footnotesize")


def ablation_table(families, endpoint, *, allow_missing=False) -> str:
    """Soft against hard targets. Declared with intervals and uncorrected p, so no stars.
    Marking these would claim a correction that was never applied."""
    rows = families["A1_soft_vs_hard"]
    index = index_rows(rows)
    models, metrics = [], []
    for r in rows:
        if r["model"] not in models:
            models.append(r["model"])
        if r["metric"] not in metrics:
            metrics.append(r["metric"])
    body = []
    for model in models:
        for i, metric in enumerate(metrics):
            first, second = next((r["first"], r["second"]) for r in rows
                                 if r["model"] == model)
            row = take(index, (model, endpoint, first, second, metric),
                       allow_missing=allow_missing)
            body.append(" & ".join([
                model_label(model) if i == 0 else "", METRIC_LABEL[metric],
                effect(row, adjusted=False) if row else PENDING,
                p_value(row, adjusted=False) if row else PENDING]) + r" \\")
        body.append(r"\addlinespace")
    return _table(
        "The anchor towards the annotator distribution against the anchor towards the "
        "majority label, for KTO. Uncorrected $p$ with 95\\% intervals, as declared: this "
        "is a secondary question about the target, not one of the confirmatory families. "
        "The accuracy row was added after the results were known."
        + DIRECTION_NOTE.replace("first-named arm", "soft target"),
        "tab:soft-vs-hard", "llll",
        ["Model", "Metric", "Difference (soft $-$ hard) [95\\% CI]", "$p$"],
        body[:-1])


def per_set_table(families, sets, *, allow_missing=False) -> str:
    rows = families["S1_per_set_repair"]
    index = index_rows(rows)
    seen, body = [], []
    for r in rows:
        if (r["model"], r["first"], r["second"]) not in seen:
            seen.append((r["model"], r["first"], r["second"]))
    for model, first, second in seen:
        for i, name in enumerate(sets):
            a = take(index, (model, name, first, second, "auroc"), allow_missing=allow_missing)
            e = take(index, (model, name, first, second, "eaurc"), allow_missing=allow_missing)
            body.append(" & ".join([
                f"{model_label(model)}, {arm_label(first)} vs {arm_label(second)}"
                if i == 0 else "",
                SET_LABEL.get(name, name),
                effect(a, adjusted=True) if a else PENDING,
                p_value(a, adjusted=True) if a else PENDING,
                effect(e, adjusted=True) if e else PENDING,
                p_value(e, adjusted=True) if e else PENDING]) + r" \\")
        body.append(r"\addlinespace")
    return _table(
        "The repair comparisons on each evaluation set separately, "
        f"Holm-corrected over all {len(rows)} tests. The primary endpoint averages these "
        "four sets per seed." + DIRECTION_NOTE,
        "tab:repair-per-set", "llllll",
        ["Model and comparison", "Set", METRIC_LABEL["auroc"] + " [95\\% CI]",
         "$p_{\\mathrm{Holm}}$", METRIC_LABEL["eaurc"] + " [95\\% CI]",
         "$p_{\\mathrm{Holm}}$"],
        body[:-1], size=r"\scriptsize")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stats", type=Path, default=Path("results/regulariser_stats.json"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--allow-missing", action="store_true",
                    help=f"emit {PENDING} for a planned cell with no data")
    args = ap.parse_args(argv)

    families = json.loads(args.stats.read_text())
    models, endpoint = families["models"], families["endpoint"]
    sets = families["sets"]
    loose = args.allow_missing
    tables = {
        "table_repair.tex": primary_table(families, models, endpoint, allow_missing=loose),
        "table_differences.tex": differences_table(families, models, endpoint,
                                                   allow_missing=loose),
        "table_vs_sft.tex": versus_sft_table(families, models, endpoint, allow_missing=loose),
        "table_soft_vs_hard.tex": ablation_table(families, endpoint, allow_missing=loose),
        "table_repair_per_set.tex": per_set_table(families, sets, allow_missing=loose),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    for name, text in tables.items():
        (args.out / name).write_text(text + "\n")
        print(f"wrote {args.out / name}")
    print("\nRequires \\usepackage{booktabs} and \\newcommand{\\pending}{--} if any cell "
          "is pending.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
