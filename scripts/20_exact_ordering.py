#!/usr/bin/env python3
"""How much of the ranking loss is saturation of the stored score, and how much is order.

The declared AUROC reads the confidence as stored, the largest softmax probability in double
precision. Under the objectives that inflate the decision margin, most items reach exactly
1.0 in that representation: the runner-up probability falls below the resolution of a double,
and the tied items are counted as a coin flip. Two things are then mixed in one number: the
score a person or a threshold can read has lost its resolution, and the model's own ordering
of items may have changed. This script separates them.

The exact score is computed from the stored logits as ``-log((1 - c) / c)``, that is minus
the log-sum-exp of the other labels' logits relative to the top one. It is a strictly
increasing function of the true largest probability, so it orders items exactly as ``c``
would in infinite precision, and it does not saturate. AUROC on it is the ordering the model
holds; AUROC on the stored ``c`` is the ordering a reader of the score can use. Both use the
same tie-aware AUROC as the declared analysis.

For each corpus the endpoint is the per-seed mean over its evaluation sets, as in the
declared analyses, and the key contrasts are recomputed on both scores with seed-paired t
and uncorrected p. This is a sensitivity analysis, outside the declared families.

    python scripts/20_exact_ordering.py --runs runs_final --out results
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

from sentalign.config import SEEDS                                    # noqa: E402
from sentalign.confidence import order_score                         # noqa: E402
from sentalign.modeling import display_name                          # noqa: E402

CORPORA = (
    ("DynaSent", "main", ("ambig_eval", "r1_test", "r2_test", "sst_dev_validated"),
     ("lfm-1.2b", "qwen-2b", "smollm3-3b")),
    ("GoEmotions", "goemo", ("ambig_eval", "go_test"), ("lfm-1.2b", "qwen-2b", "smollm3-3b")),
)
ARMS = ("sft", "kto", "rdpo", "kto-dreg1.0", "rdpo-dreg1.0")
CONTRASTS = (("kto", "sft"), ("rdpo", "sft"), ("kto-dreg1.0", "kto"), ("rdpo-dreg1.0", "rdpo"))
LABELS = {"sft": "SFT", "kto": "KTO", "rdpo": "R-DPO", "kto-dreg1.0": "KTO+reg",
          "rdpo-dreg1.0": "R-DPO+reg"}
T975_DF4 = 2.776


def _script(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def exact_score(logits) -> float:
    """``-log((1 - c) / c)`` from the logits, strictly increasing in the true top probability.

    The analysis-wide definition is ``sentalign.confidence.order_score``; this wrapper keeps
    the logits-only signature the tests use.
    """
    return order_score({"logits": list(np.asarray(logits, dtype=float))})


def scores_of(path: Path):
    stored, exact, correct = [], [], []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("correct") is None:
            continue
        stored.append(max(row["probs"]))
        exact.append(order_score(row))
        correct.append(1.0 if row["correct"] else 0.0)
    return np.array(stored), np.array(exact), np.array(correct)


def collect(runs: Path, *, seeds=SEEDS, stats=None) -> dict:
    """Per corpus, model, arm and seed: stored and exact AUROC and the tie share, set means."""
    stats = stats or _script("regulariser_stats", "06_regulariser_stats.py")
    out, missing = {}, []
    for corpus, study, sets, models in CORPORA:
        for model in models:
            for arm in ARMS:
                for seed in seeds:
                    run = runs / f"{study}__{model}__{arm}__eps0.0__tau0.2__n8000__s{seed}"
                    per_set = []
                    for name in sets:
                        path = run / "eval" / f"predictions_{name}.jsonl"
                        if not path.exists():
                            missing.append(f"{run.name}/{name}")
                            break
                        stored, exact, correct = scores_of(path)
                        per_set.append({"stored": stats.auroc(stored, correct),
                                        "exact": stats.auroc(exact, correct),
                                        "ties": float(np.mean(stored == 1.0))})
                    if len(per_set) == len(sets):
                        out[(corpus, model, arm, seed)] = {
                            k: float(np.mean([v[k] for v in per_set])) for k in per_set[0]}
    if missing:
        raise SystemExit("predictions missing:\n  " + "\n  ".join(missing))
    return out


TEMPERATURES = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)


def temperature_drift(runs: Path, *, seeds=SEEDS, stats=None) -> dict:
    """How far a temperature on the logits moves the exact AUROC, per DynaSent model and arm.

    With four labels a temperature is not exactly monotone in ``c``: it can reorder items
    through the non-top logits. This is the measured size of that effect, per-seed mean over
    the four DynaSent sets, as the largest minus the smallest AUROC across the temperatures.
    """
    stats = stats or _script("regulariser_stats", "06_regulariser_stats.py")
    corpus, study, sets, models = CORPORA[0]
    out = {}
    for model in models:
        for arm in ARMS:
            by_t = {t: [] for t in TEMPERATURES}
            for seed in seeds:
                run = runs / f"{study}__{model}__{arm}__eps0.0__tau0.2__n8000__s{seed}"
                per_t = {t: [] for t in TEMPERATURES}
                for name in sets:
                    rows = [json.loads(line) for line in
                            (run / "eval" / f"predictions_{name}.jsonl").read_text().splitlines()
                            if line.strip()]
                    rows = [r for r in rows if r.get("correct") is not None]
                    logits = np.array([r["logits"] for r in rows], dtype=float)
                    correct = np.array([1.0 if r["correct"] else 0.0 for r in rows])
                    for t in TEMPERATURES:
                        z = logits / t
                        z = z - z.max(axis=1, keepdims=True)
                        top = z.argmax(axis=1)
                        mask = np.ones_like(z, dtype=bool)
                        mask[np.arange(len(z)), top] = False
                        rest = np.where(mask, z, -np.inf)
                        m = rest.max(axis=1, keepdims=True)
                        score = -(m[:, 0] + np.log(np.exp(rest - m).sum(axis=1)))
                        per_t[t].append(stats.auroc(score, correct))
                for t in TEMPERATURES:
                    by_t[t].append(float(np.mean(per_t[t])))
            means = [float(np.mean(by_t[t])) for t in TEMPERATURES]
            out[f"{model}|{arm}"] = {"by_temperature": dict(zip(map(str, TEMPERATURES), means)),
                                     "drift": max(means) - min(means)}
    return out


def summarise(collected, *, seeds=SEEDS, selective=None) -> dict:
    selective = selective or _script("selective", "05_selective.py")
    means, contrasts = {}, []
    for corpus, _, _, models in CORPORA:
        for model in models:
            for arm in ARMS:
                cells = [collected[(corpus, model, arm, s)] for s in seeds]
                means[f"{corpus}|{model}|{arm}"] = {
                    k: float(np.mean([c[k] for c in cells])) for k in cells[0]}
            for first, second in CONTRASTS:
                row = {"corpus": corpus, "model": model, "first": first, "second": second}
                for score in ("stored", "exact"):
                    d = [collected[(corpus, model, first, s)][score]
                         - collected[(corpus, model, second, s)][score] for s in seeds]
                    half = T975_DF4 * float(np.std(d, ddof=1)) / len(d) ** 0.5
                    row[score] = {"diff": float(np.mean(d)),
                                  "ci95": [float(np.mean(d)) - half, float(np.mean(d)) + half],
                                  "p": float(selective.paired_t(d)[1])}
                contrasts.append(row)
    return {"means": means, "contrasts": contrasts}


def table(result, drift=None) -> str:
    body, previous = [], None
    for row in result["contrasts"]:
        corpus, model = row["corpus"], row["model"]
        if previous and corpus != previous[0]:
            body.append(r"\addlinespace")
        lead = [corpus if not previous or corpus != previous[0] else "",
                display_name(model) if (corpus, model) != previous else ""]
        previous = (corpus, model)
        s, e = row["stored"], row["exact"]
        ties = result["means"][f"{corpus}|{model}|{row['first']}"]["ties"]
        body.append(" & ".join(lead + [
            f"{LABELS[row['first']]} vs {LABELS[row['second']]}",
            f"{s['diff']:+.3f}", f"{e['diff']:+.3f} [{e['ci95'][0]:+.3f}, {e['ci95'][1]:+.3f}]",
            r"$<$0.001" if e["p"] < 0.001 else f"{e['p']:.3f}",
            f"{100 * ties:.0f}\\%"]) + r" \\")
    return "\n".join([
        r"\begin{table}[t]", r"\centering",
        "\\caption{Ranking results on the stored confidence and on its exact order. $\\Delta$AUROC "
        "on the confidence as stored in double precision, which rounds to exactly 1.0 once "
        "the runner-up probability falls below the resolution of a double, and on the exact "
        "order of $c$ computed from the logits, which every ranking metric in the paper uses; "
        "ties is the share of the first arm's items whose stored confidence is exactly 1.0. "
        "Per-seed mean over each corpus's evaluation sets, five seeds, seed-paired $t$, "
        "uncorrected. "
        + (f"Rescaling the logits by a temperature between {TEMPERATURES[0]:g} and "
           f"{TEMPERATURES[-1]:g} moves the exact AUROC of any DynaSent arm here by at most "
           f"{max(v['drift'] for v in drift.values()):.3f}. " if drift else "")
        + "$\\uparrow$: higher is better, so a positive difference favours the first-named "
        "arm; the tie share has no better direction.}",
        r"\label{tab:exact-order}", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lllrlrr}", r"\toprule",
        r"Corpus & Model & Comparison & $\Delta$AUROC$\uparrow$ stored & "
        r"$\Delta$AUROC$\uparrow$ exact [95\% CI] & $p$ exact & Ties \\",
        r"\midrule", *body, r"\bottomrule", r"\end{tabular}", r"\end{table}"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", type=Path, default=Path("runs_final"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args(argv)
    result = summarise(collect(args.runs))
    result["temperature_drift"] = temperature_drift(args.runs)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "exact_ordering.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.out / "table_exact_ordering.tex").write_text(
        table(result, result["temperature_drift"]) + "\n")
    for key, v in result["temperature_drift"].items():
        print(f"  temperature drift {key:24} {v['drift']:.4f}")
    for key, v in result["means"].items():
        print(f"  {key:30} stored={v['stored']:.3f} exact={v['exact']:.3f} ties={v['ties']:.2f}")
    for r in result["contrasts"]:
        print(f"  {r['corpus']:10} {r['model']:10} {r['first']:13} vs {r['second']:6} "
              f"stored {r['stored']['diff']:+.3f} (p={r['stored']['p']:.4f})  "
              f"exact {r['exact']['diff']:+.3f} (p={r['exact']['p']:.4f})")
    print(f"wrote {args.out / 'table_exact_ordering.tex'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
