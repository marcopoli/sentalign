#!/usr/bin/env python3
"""The supplementary tables that no other script writes.

Hyperparameters are read from the configuration the runs were launched with, the compute
budget from ``scripts/17``'s output, and the per-set comparisons against SFT, the
descriptive IPO+reg cell and the RQ4 placement from ``results/regulariser_stats.json``.
Formatting (arrows, marks, intervals) is ``scripts/08``'s, so a supplementary table reads
exactly like the main ones.

    python scripts/21_supplementary_tables.py --out results
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import fields
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sentalign.config import DataConfig, EvalConfig, TrainConfig   # noqa: E402
from sentalign.modeling import LoraConfigSpec                     # noqa: E402
from sentalign.plan import DREG                                   # noqa: E402


def _tables():
    spec = importlib.util.spec_from_file_location("regulariser_tables",
                                                  ROOT / "scripts" / "08_regulariser_tables.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["regulariser_tables"] = module
    spec.loader.exec_module(module)
    return module


OPTIMISER = {"adamw_8bit": "8-bit AdamW", "adamw_torch": "AdamW"}


def _sci(x: float) -> str:
    mantissa, exponent = f"{x:.0e}".split("e")
    return f"${mantissa} \\times 10^{{{int(exponent)}}}$"


def _default(cls, name):
    return next(f.default for f in fields(cls) if f.name == name)


def hyperparameters_table() -> str:
    t, d, lora = TrainConfig, DataConfig, LoraConfigSpec
    rows = [
        ("Adapters", "LoRA rank / scale / dropout",
         f"{_default(lora, 'r')} / {_default(lora, 'alpha')} / {_default(lora, 'dropout')}"),
        ("", "Base weights", "4-bit NF4, sequence length 256"),
        ("Optimisation", "Optimiser, schedule",
         f"{OPTIMISER[_default(t, 'optim')]}, {_default(t, 'lr_scheduler_type')}, warm-up "
         f"{_default(t, 'warmup_ratio')}"),
        ("", "Effective batch, epochs",
         f"{_default(t, 'per_device_batch_size') * _default(t, 'gradient_accumulation_steps')}"
         f", {_default(t, 'num_train_epochs'):g}"),
        ("", "Learning rate, SFT / preference",
         f"{_sci(_default(t, 'learning_rate'))} / {_sci(_default(t, 'po_learning_rate'))}"),
        ("", "Weight decay, gradient clip",
         f"{_default(t, 'weight_decay')}, {_default(t, 'max_grad_norm')}"),
        ("", "Checkpoint selection", "development macro-F$_1$ on "
         f"{_default(EvalConfig, 'selection_subset'):,} items every ".replace(",", "{,}")
         + f"{_default(t, 'eval_steps')} steps, no early stopping"),
        ("Data", "Training items (SFT) / pairs (preference; KTO twice as many completions)",
         f"{_default(d, 'train_subsample'):,} / {_default(d, 'max_train_pairs'):,}"
         .replace(",", "{,}")),
        ("", r"Pair threshold $\tau$", f"{_default(d, 'tau')}"),
        ("", "KTO desirable / undesirable at",
         f"$\\ge {_default(d, 'kto_desirable_at')}$ / $\\le {_default(d, 'kto_undesirable_at')}$"
         " share of annotators"),
        ("Objectives", r"$\beta$ (DPO, R-DPO, IPO, GR-DPO, MixDPO, KTO)", f"{_default(t, 'beta')}"),
        ("", r"R-DPO flip rate $\varepsilon$", f"{_default(t, 'label_smoothing')}"),
        ("", r"SimPO $\beta$ / $\gamma$",
         f"{_default(t, 'simpo_beta')} / {_default(t, 'simpo_gamma')}"),
        ("", r"AlphaPO $\alpha$", f"{_default(t, 'alphapo_alpha')}"),
        ("", "GR-DPO group step size", f"{_default(t, 'grdpo_step_size')}"),
        ("", "MixDPO margin threshold / supervised weight",
         f"{_default(t, 'mixdpo_threshold')} / {_default(t, 'mixdpo_sft_weight')}"),
        ("", "KTO class weights", "recomputed per run from its own subsample"),
        ("Anchor", r"Weight $\lambda$, target",
         f"{DREG['pref_distributional_lambda']:g}, annotator distribution"
         " (majority label in the ablation)"),
    ]
    body = [" & ".join(r) + r" \\" for r in rows]
    return "\n".join([
        r"\begin{table}[t]", r"\centering",
        "\\caption{Training configuration shared by every arm, read from the configuration "
        "the runs were launched with. Settings are fixed, not tuned per arm; no column "
        "carries a better direction.}",
        r"\label{tab:hyperparameters}", r"\footnotesize",
        r"\begin{tabular}{lp{5.4cm}p{6.2cm}}", r"\toprule", r"Group & Setting & Value \\", r"\midrule",
        *body, r"\bottomrule", r"\end{tabular}", r"\end{table}"])


def compute_table(budget: dict) -> str:
    body = [f"{name} & {g['runs']} & {g['gpu_hours']:.1f} \\\\"
            for name, g in sorted(budget["groups"].items())]
    body += [r"\midrule", f"Total & {budget['runs']} & {budget['gpu_hours']:.1f} \\\\"]
    return "\n".join([
        r"\begin{table}[t]", r"\centering",
        "\\caption{Runs reported and their training cost on one RTX 3090, by the analysis "
        "that reads them, from the run manifests. Counts, with no better direction.}",
        r"\label{tab:compute}", r"\footnotesize",
        r"\begin{tabular}{lrr}", r"\toprule", r"Analysis & Runs & GPU-hours \\", r"\midrule",
        *body, r"\bottomrule", r"\end{tabular}", r"\end{table}"])


def per_set_vs_sft_table(families, tables) -> str:
    rows = families["S2_per_set_vs_sft"]
    index = tables.index_rows(rows)
    seen, body = [], []
    for r in rows:
        if (r["model"], r["first"], r["second"]) not in seen:
            seen.append((r["model"], r["first"], r["second"]))
    for model, first, second in seen:
        for i, name in enumerate(families["sets"]):
            e = index[(model, name, first, second, "eaurc")]
            a = index[(model, name, first, second, "acc")]
            body.append(" & ".join([
                f"{tables.model_label(model)}, {tables.arm_label(first)} vs SFT" if i == 0 else "",
                tables.SET_LABEL.get(name, name),
                tables.effect(e, adjusted=True), tables.p_value(e, adjusted=True),
                tables.effect(a, adjusted=True), tables.p_value(a, adjusted=True)]) + r" \\")
        body.append(r"\addlinespace")
    return tables._table(
        "The comparisons against SFT on each evaluation set separately, Holm-corrected over "
        f"all {len(rows)} tests. The primary endpoint averages these four sets per seed."
        + tables.DIRECTION_NOTE,
        "tab:vs-sft-per-set", "llllll",
        ["Model and comparison", "Set", tables.METRIC_LABEL["eaurc"] + " [95\\% CI]",
         "$p_{\\mathrm{Holm}}$", tables.METRIC_LABEL["acc"] + " [95\\% CI]",
         "$p_{\\mathrm{Holm}}$"],
        body[:-1], size=r"\scriptsize")


def descriptive_table(families, key, caption, label, tables, size=r"\footnotesize") -> str:
    rows = families[key]
    body = []
    previous = None
    for r in rows:
        lead = tables.model_label(r["model"]) if r["model"] != previous else ""
        if previous and r["model"] != previous:
            body.append(r"\addlinespace")
        previous = r["model"]
        body.append(" & ".join([
            lead, f"{tables.arm_label(r['first'])} vs {tables.arm_label(r['second'])}",
            tables.METRIC_LABEL[r["metric"]], tables.effect(r, adjusted=False),
            tables.p_value(r, adjusted=False)]) + r" \\")
    return tables._table(caption + tables.DIRECTION_NOTE, label, "lllll",
                         ["Model", "Comparison", "Metric", "Difference [95\\% CI]", "$p$"],
                         body, size=size)


def placement_table(families, tables) -> str:
    """RQ4 placement, one row per comparison with both metrics side by side.

    One row per metric made the table taller than a page; the numbers are the same.
    """
    rows = families["D2_descriptive_placement"]
    cells, order = {}, []
    for r in rows:
        key = (r["model"], r["first"], r["second"])
        if key not in cells:
            order.append(key)
            cells[key] = {}
        cells[key][r["metric"]] = r
    body, previous = [], None
    for model, first, second in order:
        if previous and model != previous:
            body.append(r"\addlinespace")
        line = [tables.model_label(model) if model != previous else "",
                f"{tables.arm_label(first)} vs {tables.arm_label(second)}"]
        previous = model
        for metric in ("acc", "auroc"):
            r = cells[(model, first, second)].get(metric)
            line += ([tables.effect(r, adjusted=False), tables.p_value(r, adjusted=False)]
                     if r else [tables.PENDING, tables.PENDING])
        body.append(" & ".join(line) + r" \\")
    return tables._table(
        "RQ4: KTO+reg against every other arm of the landscape, on accuracy and AUROC. Not "
        "pre-specified, so 95\\% intervals and uncorrected $p$ only." + tables.DIRECTION_NOTE,
        "tab:placement", "llllll",
        ["Model", "Comparison", tables.METRIC_LABEL["acc"] + " [95\\% CI]", "$p$",
         tables.METRIC_LABEL["auroc"] + " [95\\% CI]", "$p$"],
        body, size=r"\scriptsize")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stats", type=Path, default=Path("results/regulariser_stats.json"))
    ap.add_argument("--budget", type=Path, default=Path("results/compute_budget.json"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args(argv)
    tables = _tables()
    families = json.loads(args.stats.read_text())
    out = {
        "table_hyperparameters.tex": hyperparameters_table(),
        "table_compute.tex": compute_table(json.loads(args.budget.read_text())),
        "table_vs_sft_per_set.tex": per_set_vs_sft_table(families, tables),
        "table_ipo_reg_smollm3.tex": descriptive_table(
            families, "D1_descriptive_ipo_reg",
            "IPO+reg on SmolLM3-3B, run after the confirmatory families were fixed and reported "
            "descriptively: per-seed mean over the four DynaSent evaluation sets, five seeds, "
            "seed-paired $t$ with 95\\% intervals and uncorrected $p$.",
            "tab:ipo-reg-smollm3", tables),
        "table_placement.tex": placement_table(families, tables),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    for name, text in out.items():
        (args.out / name).write_text(text + "\n")
        print(f"wrote {args.out / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
