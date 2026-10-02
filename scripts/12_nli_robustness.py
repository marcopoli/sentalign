#!/usr/bin/env python3
"""The second task: does the same thing happen to the same objectives on NLI?

The sentiment study answers its questions on one corpus, one label space, and a
five-annotator target. This script runs the same analysis on the NLI task, whose targets
come from a hundred annotators and whose label space is three-way, so that the paper's
external-validity claim is a measurement rather than an assurance.

Two panels, and they are not the same claim. Panel A asks whether preference training
degrades confidence ranking there too, which the runs on disk can answer. Panel B asks
whether the regulariser repairs it there too, which needs the regularised arms; until they
exist the panel is absent and the script says so, rather than letting a reader infer that
the missing half was tested and omitted.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sentalign.modeling import display_name                        # noqa: E402
from sentalign.confidence import order_score, top_probability   # noqa: E402

SETS = ("ambig_eval", "chaos_coarse")
RUN_ID = re.compile(r"^(?P<name>[^_]+)__(?P<model>[^_]+)__(?P<objective>.+?)__"
                    r"eps(?P<eps>[\d.]+)__tau(?P<tau>[\d.]+)__n(?P<n>\w+?)__s(?P<seed>\d+)$")
BASELINE = "sft"
DEGRADATION_ARMS = (("kto", "KTO"), ("rdpo", "R-DPO"), ("ipo", "IPO"), ("dpo", "DPO"),
                    ("mixdpo", "MixDPO"), ("sft_soft", "SFT (soft targets)"))
REPAIR_ARMS = (("kto-dreg1.0", "KTO+reg", "kto"), ("rdpo-dreg1.0", "R-DPO+reg", "rdpo"))
METRICS = (("auroc", r"$\Delta$AUROC$\uparrow$"),
           ("acc", r"$\Delta$accuracy$\uparrow$"))
DIRECTION_NOTE = (" $\\uparrow$: higher is better, so a positive difference favours the "
                  "first-named arm.")


def _modules():
    out = []
    for name, filename in (("selective", "05_selective.py"),
                           ("regulariser_stats", "06_regulariser_stats.py")):
        spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        out.append(module)
    return out


def load(runs: Path, model: str, *, sets=SETS, stats=None, study: str = "nli") -> dict:
    """Per-seed mean over the task's evaluation sets, keyed by arm.

    ``study`` is the run_id prefix of the task's runs; the GoEmotions replication reads
    its own through this function with ``study="goemo"``.

    A seed enters only if it was scored on every set, for the same reason the sentiment
    endpoint requires it: an arm averaged over a different subset of sets is a different
    quantity wearing the same name.
    """
    stats = stats or _modules()[1]
    out: dict = {}
    for run_dir in sorted(runs.iterdir()):
        m = RUN_ID.match(run_dir.name)
        if not m or m["name"] != study or m["model"] != model:
            continue
        per_set = {}
        for name in sets:
            path = run_dir / "eval" / f"predictions_{name}.jsonl"
            if not path.exists():
                break
            conf, score, correct = [], [], []
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("correct") is not None:
                    conf.append(top_probability(row))
                    score.append(order_score(row))
                    correct.append(1.0 if row["correct"] else 0.0)
            if len(conf) < 50:
                break
            per_set[name] = {"auroc": stats.auroc(score, correct),
                             "eaurc": stats.eaurc(score, correct),
                             "acc": float(np.mean(correct)),
                             "ece": stats.ece(conf, correct)}
        if len(per_set) == len(sets):
            out.setdefault(m["objective"], {})[int(m["seed"])] = {
                metric: float(np.mean([per_set[s][metric] for s in sets]))
                for metric in ("auroc", "eaurc", "acc", "ece")}
    return out


def paired(scores, first: str, second: str, metric: str, *, selective=None, min_seeds=3):
    selective = selective or _modules()[0]
    a, b = scores.get(first, {}), scores.get(second, {})
    seeds = sorted(set(a) & set(b))
    if len(seeds) < min_seeds:
        return None
    diffs = [a[s][metric] - b[s][metric] for s in seeds]
    t, p = selective.paired_t(diffs)
    mean = float(np.mean(diffs))
    half = 2.776 * float(np.std(diffs, ddof=1)) / len(diffs) ** 0.5    # t_{.975, 4}
    return {"first": first, "second": second, "metric": metric, "n": len(diffs),
            "mean_diff": mean, "ci95": [mean - half, mean + half], "p": float(p)}


def family(scores, comparisons, *, selective=None) -> list:
    """Every comparison in one panel, Holm-corrected together."""
    selective = selective or _modules()[0]
    rows = [r for r in (paired(scores, first, second, metric, selective=selective)
                        for first, second, metric in comparisons) if r]
    adjusted = selective.holm({i: r["p"] for i, r in enumerate(rows)})
    for i, row in enumerate(rows):
        row["p_holm"] = adjusted[i]
    return rows


def _cell(row) -> str:
    mark = (r"$^{***}$" if row["p_holm"] < .001 else r"$^{**}$" if row["p_holm"] < .01
            else r"$^{*}$" if row["p_holm"] < .05 else "")
    return (f"{row['mean_diff']:+.3f}{mark} "
            f"[{row['ci95'][0]:+.3f}, {row['ci95'][1]:+.3f}]")


def nli_table(scores, model: str, *, selective=None) -> tuple[str, list]:
    """The table, and the arms that were declared but have no runs behind them."""
    labels = {arm: label for arm, label in DEGRADATION_ARMS}
    labels.update({arm: label for arm, label, _ in REPAIR_ARMS})
    missing = [arm for arm, _ in DEGRADATION_ARMS if arm not in scores]
    missing += [arm for arm, _, _ in REPAIR_ARMS if arm not in scores]

    degradation = family(scores, [(arm, BASELINE, metric) for arm, _ in DEGRADATION_ARMS
                                  for metric, _ in METRICS], selective=selective)
    repair = family(scores, [(arm, twin, metric) for arm, _, twin in REPAIR_ARMS
                             for metric, _ in METRICS], selective=selective)

    body = []
    # The panel headers state the size of the family each was corrected within, computed
    # from the family, for the same reason the sentiment tables do: the paper says every
    # table names its correction, and a header that says "the panel" names nothing.
    for title, rows in ((f"A. Against SFT on the same task, Holm over {len(degradation)}",
                         degradation),
                        (f"B. Regularised against its twin, Holm over {len(repair)}", repair)):
        if not rows:
            continue
        if body:
            body.append(r"\addlinespace")
        body.append(f"\\multicolumn{{4}}{{@{{}}l}}{{\\emph{{{title}}}}} \\\\")
        by_arm: dict = {}
        for row in rows:
            by_arm.setdefault(row["first"], {})[row["metric"]] = row
        for arm, metrics in by_arm.items():
            cells = [labels.get(arm, arm),
                     f"vs {labels.get(metrics[METRICS[0][0]]['second'], metrics[METRICS[0][0]]['second'])}"
                     if arm in {a for a, _, _ in REPAIR_ARMS} else "vs SFT"]
            for metric, _ in METRICS:
                cells.append(_cell(metrics[metric]) if metric in metrics else "--")
            body.append(" & ".join(cells) + r" \\")
    table = "\n".join([
        r"\begin{table}[t]", r"\centering",
        "\\caption{The second task. Same protocol on NLI, where the target comes from a "
        "hundred annotators instead of five and the label space is three-way. Per-seed "
        f"mean over the two NLI evaluation sets, five seeds, paired $t$ on seeds, "
        f"{display_name(model)}.{DIRECTION_NOTE}}}",
        r"\label{tab:nli}", r"\footnotesize",
        r"\begin{tabular}{ll" + "l" * len(METRICS) + "}", r"\toprule",
        " & ".join(["Arm", "Comparison"]
                   + [f"{label} [95\\% CI]" for _, label in METRICS]) + r" \\",
        r"\midrule", *body, r"\bottomrule",
        r"\end{tabular}", r"\end{table}"])
    return table, missing


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", type=Path, default=Path("runs_final"))
    ap.add_argument("--model", default="lfm-1.2b")
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args(argv)

    selective, stats = _modules()
    scores = load(args.runs, args.model, stats=stats)
    if BASELINE not in scores:
        raise SystemExit(f"no {BASELINE} runs for {args.model} on the NLI task")
    table, missing = nli_table(scores, args.model, selective=selective)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "table_nli.tex").write_text(table + "\n")
    (args.out / "nli_robustness.json").write_text(json.dumps(
        {"model": args.model, "sets": list(SETS),
         "arms": {arm: {"seeds": sorted(v), **{m: float(np.mean([x[m] for x in v.values()]))
                                               for m in ("auroc", "eaurc", "acc", "ece")}}
                  for arm, v in sorted(scores.items())}}, indent=2) + "\n")
    for arm, per_seed in sorted(scores.items()):
        print(f"  {arm:14} n={len(per_seed):>2} "
              f"auroc={np.mean([v['auroc'] for v in per_seed.values()]):.4f} "
              f"acc={np.mean([v['acc'] for v in per_seed.values()]):.4f}")
    if missing:
        print(f"declared but not run yet: {', '.join(missing)}")
    print(f"wrote {args.out / 'table_nli.tex'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
