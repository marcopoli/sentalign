#!/usr/bin/env python3
"""Sensitivity of the repair to the anchor weight, on the one cell where it was swept.

lambda = 1 was the first value run, on LFM2.5 KTO. A sweep over {0.1, 0.3, 3} on the same
cell followed at three seeds, before any other objective or model was regularised, and
lambda = 1 was then kept for every regularised arm in the paper. The sweep is therefore a
check on one cell that preceded the other seven comparisons, and the manuscript says so.

Every row is averaged over the seeds all rows share. lambda = 1 has five seeds and the
sweep points three, and a row averaged over five beside rows averaged over three is the
comparability defect this project has found twice, so the extra two are left out here.

    python scripts/14_lambda_sweep.py --runs runs_final --out results
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL = "lfm-1.2b"
#: (arm, label) in the order the table prints them; the unregularised twin is lambda = 0.
ROWS = (("kto", "0 (KTO)"), ("kto-dreg0.1", "0.1"), ("kto-dreg0.3", "0.3"),
        ("kto-dreg1.0", "1"), ("kto-dreg3.0", "3"), ("sft", "SFT"))
METRICS = (("auroc", "AUROC", True), ("eaurc", "E-AURC", False), ("acc", "Acc", True),
           ("ece", "ECE", False))


def _stats_module():
    spec = importlib.util.spec_from_file_location("regulariser_stats",
                                                  ROOT / "scripts" / "06_regulariser_stats.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["regulariser_stats"] = module
    spec.loader.exec_module(module)
    return module


def common_seeds(mean, model=MODEL, rows=ROWS, endpoint="mean4") -> list[int]:
    sets = [set(mean[(model, endpoint, arm)]) for arm, _ in rows if (model, endpoint, arm) in mean]
    if len(sets) != len(rows):
        missing = [arm for arm, _ in rows if (model, endpoint, arm) not in mean]
        raise SystemExit(f"arms without runs: {missing}")
    return sorted(set.intersection(*sets))


def sweep(mean, model=MODEL, rows=ROWS, endpoint="mean4") -> dict:
    seeds = common_seeds(mean, model, rows, endpoint)
    twin = mean[(model, endpoint, "kto")]
    out = {"seeds": seeds, "rows": []}
    for arm, label in rows:
        cell = mean[(model, endpoint, arm)]
        row = {"arm": arm, "label": label}
        for metric, _, _ in METRICS:
            row[metric] = float(np.mean([cell[s][metric] for s in seeds]))
        gains = [cell[s]["auroc"] - twin[s]["auroc"] for s in seeds]
        row["auroc_gain_min"] = float(min(gains))
        row["auroc_gain_positive"] = int(sum(g > 0 for g in gains))
        out["rows"].append(row)
    return out


def table(result) -> str:
    body = []
    for row in result["rows"]:
        if row["arm"] == "sft":
            body.append(r"\addlinespace")
        cells = [row["label"]] + [f"{row[m]:.3f}" for m, _, _ in METRICS]
        body.append(" & ".join(cells) + r" \\")
    words = {2: "two", 3: "three", 4: "four", 5: "five"}
    n = words.get(len(result["seeds"]), str(len(result["seeds"])))
    return "\n".join([
        r"\begin{table}[t]", r"\centering",
        "\\caption{Sensitivity to the anchor weight $\\lambda$, KTO on LFM2.5-1.2B. Per-seed "
        "mean over the four evaluation sets, averaged over the "
        f"{n} seeds every row shares; $\\lambda = 0$ is the twin without the anchor and SFT the "
        "supervised baseline. $\\uparrow$ marks a column where higher is better and "
        "$\\downarrow$ one where lower is.}",
        r"\label{tab:lambda}", r"\small",
        r"\begin{tabular}{lrrrr}", r"\toprule",
        "$\\lambda$ & " + " & ".join(label + (r"$\uparrow$" if higher else r"$\downarrow$")
                               for _, label, higher in METRICS) + r" \\",
        r"\midrule", *body, r"\bottomrule", r"\end{tabular}", r"\end{table}"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", type=Path, default=Path("runs_final"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args(argv)
    stats = _stats_module()
    scores, _, _ = stats.load(args.runs, {MODEL})
    result = sweep(stats.average_over_sets(scores))
    for row in result["rows"]:
        print(f"  lambda {row['label']:8s} " + " ".join(f"{m}={row[m]:.3f}"
                                                    for m, _, _ in METRICS)
              + f"  gain>0 on {row['auroc_gain_positive']}/{len(result['seeds'])}")
    (args.out / "lambda_sweep.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.out / "table_lambda.tex").write_text(table(result) + "\n")
    print(f"wrote {args.out / 'table_lambda.tex'} on seeds {result['seeds']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
