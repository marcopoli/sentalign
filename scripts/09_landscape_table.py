#!/usr/bin/env python3
"""The objective landscape: every arm's absolute level, not just the differences.

``scripts/06_regulariser_stats.py`` answers whether a difference is real, and
``scripts/08_regulariser_tables.py`` typesets those differences. Neither reports where an
arm actually sits, which is what a reader needs to see that the regularised arms are not
merely better than their twins but competitive with the strongest unregularised ones.

The seed count is printed with every row rather than assumed. An arm with fewer seeds than
the design declares is marked, because an average over one seed and an average over five
are not the same quantity and a table that renders them alike invites the reader to read
the first as the second.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sentalign.config import SEEDS                                                # noqa: E402
from sentalign.modeling import MODEL_REGISTRY, display_name                       # noqa: E402

#: Arms in reporting order, grouped as the text discusses them.
GROUPS = (
    ("Reference", (("sft", "SFT"), ("sft_soft", "SFT-soft"))),
    ("Pairwise preference", (("dpo", "DPO"), ("cdpo", "cDPO"), ("rdpo", "R-DPO"),
                             ("ipo", "IPO"), ("grdpo", "GR-DPO"), ("mixdpo", "MixDPO"))),
    ("Reference-free", (("simpo", "SimPO"), ("alphapo", "AlphaPO"))),
    ("Unpaired", (("kto", "KTO"),)),
    ("With the anchor (this work)", (("kto-dreg1.0", "KTO+reg"), ("rdpo-dreg1.0", "R-DPO+reg"),
                                 ("ipo-dreg1.0", "IPO+reg"))),
)
METRICS = (("acc", "Acc", True), ("auroc", "AUROC", True), ("ece", "ECE", False))


def stats_module():
    path = ROOT / "scripts" / "06_regulariser_stats.py"
    spec = importlib.util.spec_from_file_location("regulariser_stats", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["regulariser_stats"] = module
    spec.loader.exec_module(module)
    return module


def arm_means(runs: Path, models, *, stats=None, seeds=SEEDS) -> dict:
    """Per-arm mean over seeds of the per-seed mean over the four evaluation sets.

    Only the seeds the design declares. The run directory holds more for some arms, a
    seed-variance study having added ten further SFT runs on one model, and averaging an
    arm over fifteen seeds beside arms averaged over five puts two different quantities in
    one column: the wider average is a better estimate of a different thing, and the
    comparison the table exists for is no longer between objectives alone. The earlier
    guard marked an arm with *fewer* seeds than the design, which is the same defect seen
    from one side only.
    """
    stats = stats or stats_module()
    scores, _, _ = stats.load(runs, set(models))
    mean = stats.average_over_sets(scores)
    declared = set(seeds)
    out: dict = {}
    for (model, endpoint, arm), per_seed in mean.items():
        if endpoint != stats.ENDPOINT:
            continue
        kept = {seed: value for seed, value in per_seed.items() if seed in declared}
        if not kept:
            continue
        out[(model, arm)] = {
            "seeds": sorted(kept),
            **{metric: float(np.mean([v[metric] for v in kept.values()]))
               for metric, _, _ in METRICS}}
    return out


def model_label(key: str) -> str:
    return display_name(key)


def best_arms(means, model, seeds_required: int) -> dict:
    """The leading arm per metric, over rows the table actually renders as complete.

    Two exclusions, both of which produced a wrong table when they were missing. A row
    with fewer seeds than the design declares would let an unfinished arm take a column.
    And an arm the table does not render would win invisibly: the run programme also
    contains exploratory objectives from a separate study, one of which holds the lowest
    calibration error on both models, so scanning every directory left the ECE column with
    no marked leader at all and no sign that anything was wrong.
    """
    shown = {arm for _, arms in GROUPS for arm, _ in arms}
    best = {}
    for metric, _, higher in METRICS:
        candidates = [(cell[metric], arm) for (m, arm), cell in means.items()
                      if m == model and arm in shown
                      and len(cell["seeds"]) >= seeds_required]
        if candidates:
            best[metric] = (max if higher else min)(candidates)[1]
    return best


def landscape_table(means, models, *, seeds_required: int = 5) -> str:
    header = ["Objective"] + [f"\\multicolumn{{3}}{{c}}{{{model_label(m)}}}" for m in models]
    sub = [""] + [label + (r"$\uparrow$" if higher else r"$\downarrow$")
                  for _ in models for _, label, higher in METRICS]
    spans = " ".join(f"\\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(len(models)))
    body = []
    leaders = {m: best_arms(means, m, seeds_required) for m in models}
    for group, arms in GROUPS:
        rendered = []
        for arm, label in arms:
            cells, present = [], False
            for model in models:
                cell = means.get((model, arm))
                if cell is None:
                    cells += ["--"] * len(METRICS)
                    continue
                present = True
                short = len(cell["seeds"]) < seeds_required
                for metric, _, _ in METRICS:
                    value = f"{cell[metric]:.3f}"
                    if leaders[model].get(metric) == arm:
                        value = f"\\textbf{{{value}}}"
                    if short:
                        value = f"{value}$^{{\\dagger}}$"
                    cells.append(value)
            if present:
                rendered.append(" & ".join([label] + cells) + r" \\")
        if rendered:
            body.append(f"\\multicolumn{{{1 + 3 * len(models)}}}{{@{{}}l}}"
                        f"{{\\emph{{{group}}}}} \\\\")
            body.extend(rendered)
            body.append(r"\addlinespace")
    # A caption that explains a mark no row carries describes a table that is not there.
    any_short = any(len(means[(m, a)]["seeds"]) < seeds_required
                    for m in models for _, arms in GROUPS for a, _ in arms if (m, a) in means)
    # Three models is ten columns, which overflows a single-column text block even at
    # footnotesize until the inter-column padding comes down with it.
    size = (r"\footnotesize" + "\n" + r"\setlength{\tabcolsep}{4pt}"
            if len(models) > 2 else r"\small")
    return "\n".join([
        r"\begin{table}[t]", r"\centering",
        "\\caption{Accuracy, ranking and calibration of every objective. Per-seed mean over the four evaluation "
        "sets, then averaged over seeds. $\\uparrow$ marks a column where higher is better "
        "and $\\downarrow$ one where lower is: accuracy and the AUROC of confidence "
        "against correctness are higher-better, ECE is lower-better. The leading value "
        "among the objectives reported here is in bold for each column."
        + (" $^{\\dagger}$ marks a row with fewer seeds than the design declares, which is "
           "reported rather than averaged into the rest." if any_short else "") + "}",
        r"\label{tab:landscape}", size,
        f"\\begin{{tabular}}{{l{'rrr' * len(models)}}}", r"\toprule",
        " & ".join(header) + r" \\", spans, " & ".join(sub) + r" \\", r"\midrule",
        *body[:-1], r"\bottomrule", r"\end{tabular}", r"\end{table}"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", type=Path, default=Path("runs_final"))
    ap.add_argument("--models", nargs="+", default=["lfm-1.2b", "qwen-2b"])
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--seeds-required", type=int, default=5)
    args = ap.parse_args(argv)

    means = arm_means(args.runs, args.models)
    if not means:
        # A table with no rows is not an empty result, it is a wrong invocation: an
        # unsplit shell variable put three model names into one argument once, and the
        # script wrote a valid, empty LaTeX table over a correct one.
        raise SystemExit(f"no scored runs for {args.models} under {args.runs}; "
                         "check the model keys")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "table_landscape.tex").write_text(
        landscape_table(means, args.models, seeds_required=args.seeds_required) + "\n")
    (args.out / "arm_means.json").write_text(json.dumps(
        {f"{m}|{a}": cell for (m, a), cell in sorted(means.items())}, indent=2) + "\n")
    for model in args.models:
        for arm in (a for _, arms in GROUPS for a, _ in arms):
            cell = means.get((model, arm))
            if cell:
                print(f"  {model:11} {arm:14} n={len(cell['seeds'])} "
                      + " ".join(f"{m}={cell[m]:.4f}" for m, _, _ in METRICS))
    print(f"wrote {args.out / 'table_landscape.tex'} and {args.out / 'arm_means.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
