#!/usr/bin/env python3
"""What the ranking loss costs at a coverage threshold, which is how a deployment meets it.

AUROC is the right summary for a ranking claim and the wrong number to hand an engineer
who has to pick an abstention threshold. This script reports the error rate among the
items a model keeps when it abstains on the least confident ones, at full coverage and at
two operating points, pooled over the four evaluation sets within a seed and then averaged
over seeds.

The pooling is within a seed and the average is over seeds, matching the primary endpoint:
pooling across seeds first would treat five runs of one arm as one larger run.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sentalign.config import SEEDS                                 # noqa: E402
from sentalign.modeling import display_name                        # noqa: E402
from sentalign.confidence import order_score, top_probability   # noqa: E402

SETS = ("ambig_eval", "r1_test", "r2_test", "sst_dev_validated")
RUN_ID = re.compile(r"^(?P<name>[^_]+)__(?P<model>[^_]+)__(?P<objective>.+?)__"
                    r"eps(?P<eps>[\d.]+)__tau(?P<tau>[\d.]+)__n(?P<n>\w+?)__s(?P<seed>\d+)$")
ARMS = (("sft", "SFT"), ("simpo", "SimPO"), ("kto", "KTO"), ("kto-dreg1.0", "KTO+reg"),
        ("rdpo", "R-DPO"), ("rdpo-dreg1.0", "R-DPO+reg"))
COVERAGES = (1.0, 0.8, 0.5, 0.25)


def risk_at_coverage(pairs, coverage: float) -> float:
    """Error rate among the highest-confidence ``coverage`` fraction of items.

    Items tied at the threshold contribute their block's mean correctness rather than
    whichever of them the sort happened to place first, so the number is the expected risk
    under random tie-breaking and does not depend on the order rows were written in. This
    matters exactly where the paper's claim lives: a model whose confidences are nearly
    constant has no information to abstain on, and order-dependent tie-breaking would let
    it appear to gain from abstention anyway.

    At coverage 1 this is the plain error rate, so the column doubles as an accuracy check.
    """
    if not pairs:
        raise ValueError("no scored items")
    ordered = sorted(pairs, key=lambda t: -t[0])
    kept = max(1, int(round(coverage * len(ordered))))
    correct_sum, taken, i = 0.0, 0, 0
    while taken < kept:
        j = i
        while j < len(ordered) and ordered[j][0] == ordered[i][0]:
            j += 1
        block = ordered[i:j]
        take = min(len(block), kept - taken)
        correct_sum += take * float(np.mean([c for _, c in block]))
        taken += take
        i = j
    return 1.0 - correct_sum / kept


def pooled_items(run_dir: Path, sets=SETS):
    out = []
    for name in sets:
        path = run_dir / "eval" / f"predictions_{name}.jsonl"
        if not path.exists():
            return None                      # an arm scored on fewer sets is not this arm
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("correct") is not None:
                out.append((order_score(row), 1.0 if row["correct"] else 0.0))
    return out


def collect(runs: Path, models, arms=ARMS, sets=SETS, seeds=SEEDS) -> dict:
    """Pooled items per (model, arm, seed), restricted to the seeds the design declares.

    One model's SFT arm has fifteen runs on disk because a seed-variance study added ten
    more. Averaging that curve over fifteen seeds against five for every other curve draws
    a figure whose panels compare an objective with a better-estimated version of another
    objective, and the reader is told all curves are five-seed means. The filter is here
    rather than in the caller so the table, the figure and the JSON cannot disagree.
    """
    wanted = {arm for arm, _ in arms}
    declared = set(seeds)
    out: dict = {}
    for run_dir in sorted(runs.iterdir()):
        m = RUN_ID.match(run_dir.name)
        if not m or m["model"] not in models or m["objective"] not in wanted:
            continue
        if m["name"] != "main" or m["n"] != "8000" or m["eps"] != "0.0" or m["tau"] != "0.2":
            continue
        if int(m["seed"]) not in declared:
            continue
        items = pooled_items(run_dir, sets)
        if items:
            out.setdefault((m["model"], m["objective"]), {})[int(m["seed"])] = items
    return out


def coverage_table(collected, models, *, coverages=COVERAGES, arms=ARMS,
                   seeds_required=len(SEEDS)) -> str:
    """One row per arm that has the whole declared seed set.

    An arm still filling up its seeds is not rendered rather than rendered short. This
    table is read as a comparison between curves at a shared operating point, and a row
    averaged over two seeds beside rows averaged over five differs from them in how well it
    is estimated as well as in its objective. The landscape table can mark a short row with
    a dagger because it reports levels; here the whole content of a row is the comparison.
    """
    body = []
    for model in models:
        first = True
        for arm, label in arms:
            seeds = collected.get((model, arm))
            if not seeds:
                continue
            if len(seeds) < seeds_required:
                print(f"  skipping {model} {arm}: {len(seeds)} of {seeds_required} seeds")
                continue
            cells = [f"{np.mean([risk_at_coverage(p, c) for p in seeds.values()]):.3f}"
                     for c in coverages]
            body.append(" & ".join([display_name(model) if first else "", label] + cells)
                        + r" \\")
            first = False
        body.append(r"\addlinespace")
    # Every cell is an error rate, so lower is better throughout; the arrow sits on the
    # spanning header rather than on each coverage, where it would read as a coverage.
    headers = ["Model", "Objective"] + [f"{int(c * 100)}\\%" for c in coverages]
    span = (f"& & \\multicolumn{{{len(coverages)}}}{{c}}{{Error rate among kept items "
            f"$\\downarrow$, at coverage}} \\\\ \\cmidrule(lr){{3-{2 + len(coverages)}}}")
    return "\n".join([
        r"\begin{table}[t]", r"\centering",
        "\\caption{Error rate among the items a model keeps when it abstains on the least "
        "confident ones. Items are pooled over the four evaluation sets within a seed and "
        "averaged over seeds. The 100\\% column is the plain error rate. The KTO "
        "rows show how little a model gains by abstaining when its confidence says little "
        "about its errors. $\\downarrow$: lower is better.}",
        r"\label{tab:coverage}", r"\small",
        f"\\begin{{tabular}}{{ll{'r' * len(coverages)}}}", r"\toprule",
        span,
        " & ".join(headers) + r" \\", r"\midrule", *body[:-1], r"\bottomrule",
        r"\end{tabular}", r"\end{table}"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", type=Path, default=Path("runs_final"))
    ap.add_argument("--models", nargs="+", default=["lfm-1.2b", "qwen-2b"])
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args(argv)

    collected = collect(args.runs, set(args.models))
    if not collected:
        raise SystemExit(f"no scored runs for {args.models} under {args.runs}")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "table_coverage.tex").write_text(
        coverage_table(collected, args.models) + "\n")
    summary = {f"{model}|{arm}": {f"coverage_{c}": float(np.mean(
        [risk_at_coverage(p, c) for p in seeds.values()])) for c in COVERAGES}
        for (model, arm), seeds in sorted(collected.items())}
    (args.out / "coverage_risk.json").write_text(json.dumps(summary, indent=2) + "\n")
    for key, cell in summary.items():
        print(f"  {key:26} " + " ".join(f"{k.split('_')[1]}:{v:.4f}" for k, v in cell.items()))
    print(f"wrote {args.out / 'table_coverage.tex'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
