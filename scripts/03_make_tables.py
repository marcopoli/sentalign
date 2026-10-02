"""Aggregate finished runs into the paper's tables.

    python scripts/03_make_tables.py --runs runs --out results

Reads every ``runs/*/eval/predictions_*.jsonl`` and ``runs/*/manifest.json``, pools seeds,
runs the paired bootstrap against each model's SFT arm with Holm correction, and emits
both a readable report and LaTeX table bodies for the manuscript.

The unit of resampling is the (seed, item) pair, not the item: pooling seeds and
bootstrapping items alone treats five runs of one method as five times the evidence.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sentalign.evaluate.run_eval import load_per_item, per_item_scores   # noqa: E402
from sentalign.evaluate.stats import (compare_family, format_table,       # noqa: E402
                                      mcnemar_exact, required_seeds)

#: Two things this pattern gets wrong if written casually, both of which made it match
#: nothing and the aggregation report "no completed runs":
#:
#:   * the `n{subsample}` segment. Run ids carry it, this pattern did not, so every run
#:     failed to parse and the whole programme aggregated to zero rows.
#:   * `[^_]+` for the objective. `sft_soft`, `encoder_hard`, and `encoder_soft` all
#:     contain an underscore, so a class-excluding pattern silently drops them while
#:     appearing to work for the others. It is anchored non-greedily against `__eps`
#:     instead.
RUN_ID = re.compile(
    r"^(?P<name>[^_]+)__(?P<model>[^_]+)__(?P<objective>.+?)"
    r"__eps(?P<eps>[\d.]+)__tau(?P<tau>[\d.]+)__n(?P<n>\d+|all)__s(?P<seed>\d+)$")

#: The tau at which the shared SFT anchor is trained. Comparisons at other tau or eps
#: values use this same anchor, because neither parameter reaches the SFT corpus.
BASELINE_TAU = 0.2

#: Per-item metrics where a smaller value is a better model, so "best first" reverses.
LOWER_IS_BETTER_ITEM_METRICS = {"nll", "jsd", "jsd_reward", "jsd_temp", "brier"}

#: Metrics where a *lower* value is better, so the sign of the reported delta flips.
LOWER_IS_BETTER = {"ece", "ece_ts", "brier", "brier_ts", "nll", "nll_ts", "aurc",
                   "jsd_human", "cross_entropy_human", "excess_ce_human",
                   "parse_failure_rate", "label_prior_drift"}

HEADLINE = ["macro_f1", "balanced_accuracy", "ece", "ece_ts", "brier", "aurc",
            "jsd_human", "acc@80", "parse_failure_rate"]


def parse_run_id(run_id: str) -> dict | None:
    m = RUN_ID.match(run_id)
    return m.groupdict() if m else None


def collect(runs_dir: Path, name: str | None = None) -> dict:
    """Index every completed run by (model, objective, eps, tau) -> {seed: run}.

    ``name`` restricts to one study prefix. Two studies can share a run_id tail while
    differing in dataset and label space, so mixing them in one table would compare
    numbers computed on different evaluation sets.
    """
    index: dict[tuple, dict[int, dict]] = defaultdict(dict)
    for run_dir in sorted(runs_dir.glob("*")):
        meta = parse_run_id(run_dir.name)
        if not meta or not (run_dir / "eval" / "metrics.json").exists():
            continue
        if name is not None and meta["name"] != name:
            continue
        metrics = json.loads((run_dir / "eval" / "metrics.json").read_text())
        manifest = {}
        if (run_dir / "manifest.json").exists():
            manifest = json.loads((run_dir / "manifest.json").read_text())
        # `n` belongs in the key. Without it the four data-scaling sizes share one slot
        # per seed and overwrite each other, so the scaling study would silently reduce
        # to whichever size happened to sort last.
        key = (meta["model"], meta["objective"], float(meta["eps"]), float(meta["tau"]),
               meta["n"])
        index[key][int(meta["seed"])] = {
            "dir": run_dir, "metrics": metrics, "manifest": manifest}
    return index


def pooled_scores(runs: dict[int, dict], eval_set: str,
                  metric: str) -> tuple[np.ndarray, np.ndarray]:
    """Per-item scores concatenated across seeds, with the matching seed labels.

    Items are ordered identically in every run (the evaluation sets are fixed), which is
    what makes the comparison paired. We assert it rather than assume it.
    """
    scores, seeds, reference_ids = [], [], None
    for seed in sorted(runs):
        path = runs[seed]["dir"] / "eval" / f"predictions_{eval_set}.jsonl"
        if not path.exists():
            continue
        rows = load_per_item(path)
        ids = [r["text_id"] for r in rows]
        if reference_ids is None:
            reference_ids = ids
        elif ids != reference_ids:
            raise ValueError(
                f"{eval_set}: item order differs between seeds; the paired bootstrap "
                f"requires aligned items (seed {seed})")
        scores.append(per_item_scores(rows, metric))
        seeds.append(np.full(len(rows), seed))
    if not scores:
        return np.array([]), np.array([])
    return np.concatenate(scores), np.concatenate(seeds)


def summarise(runs: dict[int, dict], eval_set: str) -> dict[str, tuple[float, float]]:
    """Mean and standard deviation across seeds for each headline metric."""
    out: dict[str, list[float]] = defaultdict(list)
    for run in runs.values():
        block = run["metrics"]["sets"].get(eval_set, {}).get("metrics", {})
        for metric in HEADLINE:
            value = block.get(metric)
            # `null` in the file, or a non-finite value from an older run: the metric is
            # undefined for this set, not zero. Averaging it in would propagate nan
            # through the whole column and print an empty cell in the paper.
            if isinstance(value, (int, float)) and not isinstance(value, bool) \
                    and math.isfinite(value):
                out[metric].append(float(value))
    return {k: (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0)
            for k, v in out.items()}


def latex_row(name: str, summary: dict, comparison=None) -> str:
    def cell(metric: str) -> str:
        if metric not in summary:
            return "\\pending"
        mean, sd = summary[metric]
        return f"{mean:.3f}\\,\\tiny{{$\\pm${sd:.3f}}}"

    parts = [name.upper(), cell("macro_f1")]
    if comparison is None:
        parts += ["--", "--"]
    else:
        p = comparison.p_adjusted if comparison.p_adjusted is not None else comparison.p_value
        stars = "$^{***}$" if p < 0.001 else "$^{**}$" if p < 0.01 else "$^{*}$" if p < 0.05 else ""
        parts += [f"{comparison.difference:+.3f}{stars}", f"{p:.3f}"]
    parts += [cell("ece"), cell("jsd_human"), cell("aurc")]
    return " & ".join(parts) + r" \\"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--eval-sets", nargs="+",
                    default=["r1_test", "r2_test", "ambig_eval"])
    ap.add_argument("--metric", default="correct",
                    choices=["correct", "nll", "jsd", "jsd_reward", "jsd_temp", "brier"],
                    help="per-item quantity the paired bootstrap resamples")
    ap.add_argument("--name", default=None,
                    help="restrict to one study prefix, e.g. nli")
    ap.add_argument("--resamples", type=int, default=10_000)
    args = ap.parse_args()

    index = collect(args.runs, args.name)

    # A programme that spans a code change is not one experiment. Surface it rather than
    # averaging across versions silently.
    fingerprints: dict[str, list[str]] = defaultdict(list)
    for runs in index.values():
        for seed, run in runs.items():
            env_path = run["dir"] / "environment.json"
            if env_path.exists():
                try:
                    fp = json.loads(env_path.read_text()).get("code_fingerprint")
                except json.JSONDecodeError:
                    continue
                if fp:
                    fingerprints[fp].append(run["dir"].name)
    if len(fingerprints) > 1:
        print("WARNING: runs span more than one version of the code.", file=sys.stderr)
        for fp, names in sorted(fingerprints.items(), key=lambda kv: -len(kv[1])):
            print(f"  {fp}  {len(names)} runs  e.g. {names[0]}", file=sys.stderr)
        print("  Re-run the minority group, or report the split explicitly.\n",
              file=sys.stderr)

    if not index:
        print(f"no completed runs with evaluations under {args.runs}/\n"
              f"expected e.g. {args.runs}/main__lfm-1.2b__dpo__eps0.0__tau0.2__n8000"
              f"__s13/eval/metrics.json", file=sys.stderr)
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    report: list[str] = []
    latex: dict[str, list[str]] = defaultdict(list)
    payload: dict = {}

    models = sorted({k[0] for k in index})
    for model in models:
        for eps, tau, n in sorted({(k[2], k[3], k[4]) for k in index if k[0] == model}):
            # The SFT anchor is tau- and eps-independent by construction: tau shapes
            # only the preference pairs and eps only flips them, while SFT reads
            # data/build/sft/, which has neither partition. The plan therefore trains
            # one anchor per (model, n), and looking for one at tau=0.4 or eps=0.1
            # discarded the entire noise ladder and margin ablation as "no SFT
            # baseline". This is the same fact `_default_reference` encodes.
            baseline_key = (model, "sft", 0.0, BASELINE_TAU, n)
            if baseline_key not in index:
                report.append(f"[skip] {model} n={n}: no SFT baseline at "
                              f"tau={BASELINE_TAU}")
                continue

            baseline_runs = index[baseline_key]
            for eval_set in args.eval_sets:
                # Arms are grouped by the seeds they share with the baseline, and each
                # group is compared against the baseline restricted to those same seeds.
                # Demanding the baseline's full seed set instead discarded every study
                # run at 3 seeds against a 5-seed main grid: the whole noise ladder,
                # margin ablation, and cspo ablation, reported only as a shape mismatch.
                by_seed_set: dict[tuple, dict] = defaultdict(dict)
                for (m, objective, e, t, nn), runs in sorted(index.items()):
                    if (m, e, t, nn) != (model, eps, tau, n) or objective == "sft":
                        continue
                    shared = tuple(sorted(set(runs) & set(baseline_runs)))
                    if not shared:
                        report.append(f"[skip] {model}/{objective}/{eval_set}: "
                                      f"no seed shared with the SFT baseline")
                        continue
                    by_seed_set[shared][objective] = runs

                for shared, arms in sorted(by_seed_set.items(), key=lambda kv: -len(kv[0])):
                    try:
                        base_scores, base_seeds = pooled_scores(
                            {s: baseline_runs[s] for s in shared}, eval_set, args.metric)
                    except ValueError as exc:
                        # The baseline itself lacks this metric: jsd_temp needs a fit
                        # split carrying p_human, and runs evaluated before that existed
                        # have no temp_probs. Skip the slice and say so, rather than
                        # killing every other slice in the report.
                        report.append(f"[skip] {model}/sft(baseline)/{eval_set}: {exc}")
                        continue
                    if base_scores.size == 0:
                        continue

                    methods, summaries = {}, {}
                    for objective, runs in sorted(arms.items()):
                        subset = {s: runs[s] for s in shared}
                        try:
                            scores, _ = pooled_scores(subset, eval_set, args.metric)
                        except ValueError as exc:
                            # jsd_reward exists only for reference-anchored arms.
                            report.append(f"[skip] {model}/{objective}/{eval_set}: "
                                          f"{exc}")
                            continue
                        if scores.shape != base_scores.shape:
                            # Same seeds, different item count: a real misalignment.
                            report.append(
                                f"[skip] {model}/{objective}/{eval_set}: "
                                f"{scores.shape} vs baseline {base_scores.shape} "
                                f"on seeds {list(shared)}")
                            continue
                        methods[objective] = scores
                        summaries[objective] = summarise(subset, eval_set)

                    if not methods:
                        continue
                    results = compare_family(base_scores, methods, seeds=base_seeds,
                                             n_resamples=args.resamples)

                    report.append(f"=== {model} | {eval_set} | eps={eps} tau={tau} "
                                  f"n={n} | per-item metric: {args.metric} ===")
                    report.append(f"  seeds: {list(shared)}  "
                                  f"items/seed: {results[0].n_items:,}")
                    report.append(format_table(
                        results, metric=args.metric,
                        lower_is_better=args.metric in LOWER_IS_BETTER_ITEM_METRICS))
                    report.append("")

                    latex_key = f"{model}_{eval_set}_eps{eps}_tau{tau}_n{n}"
                    latex[latex_key].append(latex_row(
                        "sft", summarise({s: baseline_runs[s] for s in shared}, eval_set)))
                    by_name = {r.name: r for r in results}
                    for objective in sorted(methods):
                        latex[latex_key].append(
                            latex_row(objective, summaries[objective], by_name[objective]))

                    payload[f"{model}|{eval_set}|eps{eps}|tau{tau}|n{n}|"
                            f"seeds{len(shared)}"] = [
                        {"objective": r.name, "delta": r.difference,
                         "ci": [r.ci_low, r.ci_high], "p": r.p_value,
                         "p_holm": r.p_adjusted, "n_items": r.n_items,
                         "n_seeds": r.n_seeds, "seeds": list(shared)}
                        for r in results]

    # What the current seed count can actually resolve.
    if payload:
        first = next(iter(payload.values()))
        n_items = first[0]["n_items"]
        n_seeds = first[0]["n_seeds"]
        report.append("=== resolvable effect size at this budget ===")
        for effect in (0.005, 0.010, 0.015, 0.020):
            need = required_seeds(effect, item_sd=0.46, n_items=n_items, seed_sd=0.004)
            verdict = "resolvable" if need <= n_seeds else f"needs {need} seeds"
            report.append(f"  {effect * 100:.1f} macro-F1 points: {verdict}")

    (args.out / "report.txt").write_text("\n".join(report) + "\n")
    (args.out / "comparisons.json").write_text(json.dumps(payload, indent=2) + "\n")
    for key, rows in latex.items():
        (args.out / f"table_{key}.tex").write_text("\n".join(rows) + "\n")

    print("\n".join(report))
    print(f"\nwrote {args.out}/report.txt, comparisons.json, and "
          f"{len(latex)} LaTeX table bodies")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
