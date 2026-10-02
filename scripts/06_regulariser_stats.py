"""Confirmatory statistics for the distributional regulariser.

Endpoint. Each seed's metric is averaged over the four independent held-out sets, and the
seed-paired test runs on that average: the four sets test one hypothesis, not four. This
endpoint was chosen on 2026-09-15, *after* the per-set results for lfm-1.2b and qwen-2b had
been seen and *before* any SmolLM3 run existed. The per-set families are still computed and
reported as the stricter supplementary analysis, whatever they show.

Families, fixed here in code. Holm is applied within each.

P1  primary      each regularised arm against its own unregularised twin, AUROC
P2  secondary    the same comparisons, E-AURC
P3  vs SFT       each regularised arm against the SFT baseline, E-AURC and accuracy
A1  ablation     soft against hard targets. A claim of *no* difference on ranking, which a
                 p-value cannot support, so it is a 95 percent interval with no correction.
                 Accuracy was added to A1 on 2026-10-01, after the results were known, to
                 test the text's account of where the anchor's accuracy gain comes from.
                 It is descriptive like the rest of A1 and changes no other row.
S1, S2           P1 with P2, and P3, on each set separately (supplementary)

The planned arms differ by model. SmolLM3-3B runs the five-arm confirmation set only, so it
has no IPO comparison and no soft-vs-hard ablation, and neither counts as missing.

AUROC of confidence against correctness is rank-only, and E-AURC subtracts the optimal
risk-coverage curve at the arm's own accuracy, so neither can be won by being accurate
rather than by knowing when it is wrong. Raw AURC is never reported for that reason.
``pooled`` is excluded: it is the concatenation of ambig_eval, r1_test and r2_test.

    python scripts/06_regulariser_stats.py --runs runs_final --out results/regulariser_stats.json
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sentalign.confidence import order_score, top_probability   # noqa: E402

_SELECTIVE = Path(__file__).resolve().with_name("05_selective.py")
_spec = importlib.util.spec_from_file_location("selective", _SELECTIVE)
selective = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(selective)

RUN_ID = selective.RUN_ID
SETS = ("ambig_eval", "r1_test", "r2_test", "sst_dev_validated")
ENDPOINT = "mean4"
MODELS = ("lfm-1.2b", "qwen-2b", "smollm3-3b")
REGULARISED = ("kto-dreg1.0", "rdpo-dreg1.0", "ipo-dreg1.0")
SOFT, HARD = "kto-dreg1.0", "kto-dreg-hard"
BASELINE = "sft"
#: The regularised arms each model was planned to run.
PLANNED = {"lfm-1.2b": REGULARISED, "qwen-2b": REGULARISED,
           "smollm3-3b": ("kto-dreg1.0", "rdpo-dreg1.0")}
ABLATION_MODELS = ("lfm-1.2b", "qwen-2b")
#: The one regularised cell run after the confirmatory families were fixed
#: (``plan.ipo_reg_completion``). It is reported with intervals and uncorrected p, never
#: enters PLANNED or the joint correction, and so cannot move a confirmatory p-value.
DESCRIPTIVE = {"smollm3-3b": (("ipo-dreg1.0", "ipo"), ("ipo-dreg1.0", BASELINE))}
DESCRIPTIVE_METRICS = ("auroc", "eaurc", "acc", "ece")
#: RQ4: where KTO+reg sits against every other arm of the landscape. Not pre-specified,
#: so uncorrected, and computed here so the manuscript quotes numbers rather than types them.
PLACEMENT_ARM = "kto-dreg1.0"
PLACEMENT_RIVALS = ("sft", "sft_soft", "dpo", "rdpo", "ipo", "grdpo", "mixdpo", "simpo",
                    "alphapo", "kto", "rdpo-dreg1.0", "ipo-dreg1.0")
#: Metrics where a larger value is better. The others are better when smaller.
HIGHER_IS_BETTER = frozenset({"auroc", "acc"})
#: Two-sided 97.5th percentile of Student t by degrees of freedom, for the intervals.
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
        8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145}


def twin(arm: str) -> str:
    """The unregularised arm a regularised one is compared with: ``kto-dreg1.0 -> kto``."""
    return arm.split("-dreg", 1)[0]


def repair_plan(models) -> dict:
    return {m: tuple((arm, twin(arm)) for arm in PLANNED[m]) for m in models}


def sft_plan(models) -> dict:
    return {m: tuple((arm, BASELINE) for arm in PLANNED[m]) for m in models}


def ablation_plan(models) -> dict:
    return {m: ((SOFT, HARD),) for m in models if m in ABLATION_MODELS}


def auroc(confidence, correct) -> float:
    """P(confidence of a correct item > confidence of an error), ties counted as a half."""
    conf = np.asarray(confidence, dtype=float)
    pos = np.asarray(correct, dtype=float) == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    _, inverse, counts = np.unique(conf, return_inverse=True, return_counts=True)
    average_rank = np.cumsum(counts) - (counts - 1) / 2.0
    ranks = average_rank[inverse]
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _aurc_of_order(errors_in_order: np.ndarray) -> float:
    return float((np.cumsum(errors_in_order) / np.arange(1, len(errors_in_order) + 1)).mean())


def eaurc(confidence, correct) -> float:
    """AURC minus the AURC of a perfect ranker with the same number of errors."""
    correct = np.asarray(correct, dtype=float)
    order = np.argsort(-np.asarray(confidence, dtype=float), kind="stable")
    observed = _aurc_of_order(1.0 - correct[order])
    ideal = _aurc_of_order(np.sort(1.0 - correct))
    return observed - ideal


def ece(confidence, correct, bins: int = 15) -> float:
    """Expected calibration error over equal-mass bins, as ``evaluate.metrics`` defines it.

    Equal mass, not equal width. A fine-tuned classifier piles its confidences up against
    1.0, so equal-width bins put almost every item in the last bin or two and the estimate
    reports the calibration of that bin rather than of the model; the package chose
    quantile edges for that reason and the paper describes them. This function had equal
    widths, so every ECE in the landscape table and in the soft-versus-hard ablation was a
    different statistic from the one the method section named.
    """
    conf = np.asarray(confidence, dtype=float)
    correct = np.asarray(correct, dtype=float)
    if conf.size == 0:
        return float("nan")
    edges = np.unique(np.quantile(conf, np.linspace(0.0, 1.0, bins + 1)))
    if len(edges) < 2:
        return float(abs(conf.mean() - correct.mean()))
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, len(edges) - 2)
    total = 0.0
    for b in range(len(edges) - 1):
        members = idx == b
        if members.any():
            total += members.mean() * abs(correct[members].mean() - conf[members].mean())
    return float(total)


def load(runs: Path, models, sets=SETS):
    """Per-seed metrics keyed by ``(model, set, arm)``, the item-set digests, exclusions."""
    scores: dict = defaultdict(dict)
    items: dict = defaultdict(set)
    excluded: list[str] = []
    for run_dir in sorted(runs.iterdir()):
        m = RUN_ID.match(run_dir.name)
        if not m or m["model"] not in models or m["name"] != "main":
            continue
        if m["n"] != "8000" or m["eps"] != "0.0" or m["tau"] != "0.2":
            continue
        manifest = run_dir / "manifest.json"
        if manifest.exists():
            record = json.loads(manifest.read_text())
            # Every trained arm evaluates on dev at least once, so an empty selection
            # history means the scored adapter was never chosen on dev macro-F1.
            if not record.get("baseline") and not record.get("selection_history"):
                excluded.append(run_dir.name)
                continue
        for name in sets:
            path = run_dir / "eval" / f"predictions_{name}.jsonl"
            if not path.exists():
                continue
            conf, score, correct, ids = [], [], [], []
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("correct") is None:
                    continue
                # Ranking metrics read the exact order of c (sentalign.confidence), which
                # a double stops resolving once c rounds to 1.0; calibration reads c.
                conf.append(top_probability(row))
                score.append(order_score(row))
                correct.append(1.0 if row["correct"] else 0.0)
                ids.append(row["text_id"])
            if len(conf) < 50:
                continue
            scores[(m["model"], name, m["objective"])][int(m["seed"])] = {
                "auroc": auroc(score, correct), "eaurc": eaurc(score, correct),
                "acc": float(np.mean(correct)), "ece": ece(conf, correct)}
            digest = hashlib.blake2b("\n".join(sorted(ids)).encode(), digest_size=8)
            items[(m["model"], name)].add(digest.hexdigest())
    return scores, items, excluded


def average_over_sets(scores, sets=SETS) -> dict:
    """Per-seed metrics averaged over ``sets``, keyed ``(model, ENDPOINT, arm)``.

    A seed enters only if it was scored on every set, and an arm only if every set exists.
    Averaging whatever subset a seed happens to have would give different seeds different
    endpoints, and the paired difference would then compare different quantities.
    """
    grouped: dict = defaultdict(list)
    for (model, name, arm), by_seed in scores.items():
        if name in sets:
            grouped[(model, arm)].append(by_seed)
    out: dict = {}
    for (model, arm), per_set in grouped.items():
        if len(per_set) != len(sets):
            continue
        seeds = sorted(set.intersection(*(set(p) for p in per_set)))
        out[(model, ENDPOINT, arm)] = {
            s: {k: float(np.mean([p[s][k] for p in per_set])) for k in per_set[0][s]}
            for s in seeds}
    return out


def paired(scores, model: str, name: str, first: str, second: str, metric: str):
    a = scores.get((model, name, first), {})
    b = scores.get((model, name, second), {})
    seeds = sorted(set(a) & set(b))
    if len(seeds) < 3:
        return None
    diffs = [a[s][metric] - b[s][metric] for s in seeds]
    t, p = selective.paired_t(diffs)
    mean = float(np.mean(diffs))
    half = T975[len(diffs) - 1] * float(np.std(diffs, ddof=1)) / len(diffs) ** 0.5
    return {"model": model, "set": name, "first": first, "second": second,
            "metric": metric, "n": len(diffs), "seeds": seeds, "mean_diff": mean,
            "ci95": [mean - half, mean + half], "t": float(t), "p": float(p),
            "favours_first": bool(mean > 0 if metric in HIGHER_IS_BETTER else mean < 0)}


def run_family(scores, plan: dict, metrics, *, sets, allow_missing: bool = False):
    """Every planned comparison, Holm-adjusted together.

    A planned comparison with no data is an error rather than a smaller family: a family
    that shrinks silently is corrected less strictly than the one that was declared.
    """
    rows, missing = [], []
    for model, comparisons in plan.items():
        for first, second in comparisons:
            for name in sets:
                for metric in metrics:
                    row = paired(scores, model, name, first, second, metric)
                    if row is None:
                        missing.append(f"{model} {name} {first} vs {second} {metric}")
                    else:
                        rows.append(row)
    if missing and not allow_missing:
        raise SystemExit("planned comparisons without data:\n  " + "\n  ".join(missing))
    adjusted = selective.holm({i: row["p"] for i, row in enumerate(rows)})
    for i, row in enumerate(rows):
        row["p_holm"] = adjusted[i]
    return rows, missing


CONFIRMATORY = ("P1_primary_auroc", "P2_secondary_eaurc", "P3_vs_sft")


def joint_correction(families: dict, keys=CONFIRMATORY, alpha: float = 0.05) -> dict:
    """Holm over every confirmatory test at once, as the sceptical reader would run it.

    Correcting within declared families is the protocol, and it is the weaker of the two
    readings: a family of eight is adjusted less strictly than the pool of thirty-two that
    contains it. The paper reports what survives the pool as well, so that a reader who
    disagrees with the family structure does not have to take the analysis apart to find
    out how much of the result depends on it.

    The count is computed here rather than counted by hand into the manuscript: it changes
    whenever a model or an arm is added, and a number typed once is a number that stops
    tracking the runs.
    """
    rows = [dict(row, family=key) for key in keys for row in families[key]]
    adjusted = selective.holm({i: row["p"] for i, row in enumerate(rows)})
    for i, row in enumerate(rows):
        row["p_joint"] = adjusted[i]
    survivors = [r for r in rows if r["p_joint"] < alpha]
    return {"alpha": alpha, "n_tests": len(rows), "n_survivors": len(survivors),
            "families": list(keys), "rows": rows}


def _mark(p: float) -> str:
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""


def _print(title: str, rows, adjusted: bool) -> None:
    print(f"\n{title}" + (f", Holm over {len(rows)}" if adjusted else ""))
    for r in rows:
        tail = (f"p={r['p']:.4f} holm={r['p_holm']:.4f}{_mark(r['p_holm'])}" if adjusted
                else f"p={r['p']:.4f}")
        print(f"  {r['model']:10} {r['set']:18} {r['first']:13} vs {r['second']:13} "
              f"{r['metric']:5} d={r['mean_diff']:+.4f} "
              f"[{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}] t={r['t']:+6.2f} n={r['n']} {tail}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=Path("runs_final"))
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--allow-missing", action="store_true")
    args = ap.parse_args()

    scores, items, excluded = load(args.runs, set(args.models))
    print("excluded (no checkpoint selection):", excluded or "none")
    for (model, name), digests in sorted(items.items()):
        if len(digests) != 1:
            raise SystemExit(f"{model} {name}: runs scored {len(digests)} different item sets")
    print("item sets identical within every model and set: yes")

    mean = average_over_sets(scores)
    endpoint, loose = (ENDPOINT,), args.allow_missing
    repair, versus_sft = repair_plan(args.models), sft_plan(args.models)
    families = {
        "P1_primary_auroc": run_family(mean, repair, ("auroc",), sets=endpoint,
                                       allow_missing=loose)[0],
        "P2_secondary_eaurc": run_family(mean, repair, ("eaurc",), sets=endpoint,
                                         allow_missing=loose)[0],
        "P3_vs_sft": run_family(mean, versus_sft, ("eaurc", "acc"), sets=endpoint,
                                allow_missing=loose)[0],
        "A1_soft_vs_hard": run_family(mean, ablation_plan(args.models),
                                      ("auroc", "eaurc", "ece", "acc"), sets=endpoint,
                                      allow_missing=loose)[0],
        "S1_per_set_repair": run_family(scores, repair, ("auroc", "eaurc"), sets=SETS,
                                        allow_missing=loose)[0],
        "S2_per_set_vs_sft": run_family(scores, versus_sft, ("eaurc", "acc"), sets=SETS,
                                        allow_missing=loose)[0],
    }
    _print("P1 primary: regularised arm vs its twin, mean over four sets, AUROC",
           families["P1_primary_auroc"], adjusted=True)
    _print("P2 secondary: same comparisons, E-AURC", families["P2_secondary_eaurc"],
           adjusted=True)
    _print(f"P3 against {BASELINE}, mean over four sets", families["P3_vs_sft"],
           adjusted=True)
    _print("A1 soft vs hard targets, mean over four sets: 95 percent intervals, no correction",
           families["A1_soft_vs_hard"], adjusted=False)
    _print("S1 supplementary: repair per set", families["S1_per_set_repair"], adjusted=True)
    _print("S2 supplementary: against SFT per set", families["S2_per_set_vs_sft"],
           adjusted=True)

    joint = joint_correction(families)
    descriptive = {m: plan for m, plan in DESCRIPTIVE.items() if m in args.models}
    if descriptive:
        rows = run_family(mean, descriptive, DESCRIPTIVE_METRICS, sets=endpoint,
                          allow_missing=loose)[0]
        # Uncorrected by declaration: a Holm value here would imply a family that was
        # never declared.
        families["D1_descriptive_ipo_reg"] = [
            {k: v for k, v in row.items() if k != "p_holm"} for row in rows]
        _print("D1 descriptive, declared after the families: 95 percent intervals, no "
               "correction", families["D1_descriptive_ipo_reg"], adjusted=False)
    placement = {m: tuple((PLACEMENT_ARM, rival) for rival in PLACEMENT_RIVALS
                          if (m, ENDPOINT, rival) in mean) for m in args.models}
    rows = run_family(mean, placement, ("acc", "auroc"), sets=endpoint, allow_missing=loose)[0]
    families["D2_descriptive_placement"] = [
        {k: v for k, v in row.items() if k != "p_holm"} for row in rows]
    _print("D2 descriptive, RQ4 placement of KTO+reg: no correction",
           families["D2_descriptive_placement"], adjusted=False)
    print(f"\nJoint Holm over all {joint['n_tests']} confirmatory tests "
          f"({', '.join(joint['families'])}): {joint['n_survivors']} survive at "
          f"alpha={joint['alpha']}")
    for r in joint["rows"]:
        if r["p_joint"] < joint["alpha"]:
            print(f"  {r['model']:10} {r['first']:13} vs {r['second']:13} {r['metric']:5} "
                  f"d={r['mean_diff']:+.4f} p={r['p']:.4f} joint={r['p_joint']:.4f}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({**families, "J_joint_confirmatory": joint,
                                        "excluded": excluded,
                                        "sets": list(SETS), "endpoint": ENDPOINT,
                                        "models": args.models}, indent=2) + "\n")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
