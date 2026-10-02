#!/usr/bin/env python3
"""Where the damage lands: the ambiguity-stratified set split by annotator agreement.

The paper's human-centred argument is about the items people found hard, so it has to say
what preference training does on them rather than assume it. The ambiguity-stratified set
records each item's agreement band (5of5, 4of5, 3of5), and this script reports three
quantities per band, per arm, averaged over the declared seeds:

    error rate            whether the new objective is wrong more often there
    confident-error rate  the share of items that are wrong *and* above a confidence
                          threshold, which is what a person reading the score is misled by
    AUROC within band     whether confidence still ranks errors among items of equal
                          human difficulty

It is descriptive: none of these enters a declared family, and the manuscript says so.
The threshold makes the confident-error rate uninformative for an arm whose confidences
rarely reach it, so the share of items above the threshold is reported beside it and the
paper does not compare arms on the rate when that share differs by an order of magnitude.

    python scripts/13_agreement_bands.py --runs runs_final --out results/agreement_bands.json

The table the discussion cites is written beside the JSON as table_agreement_bands.tex.
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

from sentalign.config import SEEDS                                  # noqa: E402
from sentalign.confidence import order_score, top_probability   # noqa: E402

BANDS = ("5of5", "4of5", "3of5")
ARMS = ("sft", "kto", "kto-dreg1.0")
MODELS = ("lfm-1.2b", "qwen-2b", "smollm3-3b")
MODEL_NAMES = {"lfm-1.2b": "LFM2.5-1.2B", "qwen-2b": "Qwen3.5-2B", "smollm3-3b": "SmolLM3-3B"}
ARM_NAMES = {"sft": "SFT", "kto": "KTO", "kto-dreg1.0": "KTO+reg"}
BAND_NAMES = {"5of5": "5/5", "4of5": "4/5", "3of5": "3/5"}
SET = "ambig_eval"
THRESHOLD = 0.9


def _stats_module():
    spec = importlib.util.spec_from_file_location("regulariser_stats",
                                                  ROOT / "scripts" / "06_regulariser_stats.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["regulariser_stats"] = module
    spec.loader.exec_module(module)
    return module


def band_metrics(rows, *, auroc, threshold: float = THRESHOLD) -> dict:
    """Per-band metrics for one run's predictions."""
    out = {}
    for band in BANDS:
        sub = [r for r in rows if r.get("group") == band]
        if not sub:
            continue
        conf = np.array([top_probability(r) for r in sub], dtype=float)
        score = np.array([order_score(r) for r in sub], dtype=float)
        correct = np.array([1.0 if r["correct"] else 0.0 for r in sub])
        wrong = correct == 0.0
        out[band] = {
            "n": len(sub),
            "error": float(wrong.mean()),
            "confident_error": float((wrong & (conf > threshold)).mean()),
            "above_threshold": float((conf > threshold).mean()),
            "auroc": (float(auroc(score, correct))
                      if 0.0 < correct.mean() < 1.0 else float("nan")),
        }
    return out


def collect(runs: Path, *, models=MODELS, arms=ARMS, seeds=SEEDS, stats=None) -> dict:
    stats = stats or _stats_module()
    result: dict = {}
    for model in models:
        for arm in arms:
            per_seed = []
            for seed in seeds:
                path = (runs / f"main__{model}__{arm}__eps0.0__tau0.2__n8000__s{seed}"
                        / "eval" / f"predictions_{SET}.jsonl")
                if not path.exists():
                    raise SystemExit(f"missing {path}: the bands are averaged over the "
                                     "declared seeds or not reported")
                rows = [json.loads(line) for line in path.read_text().splitlines()
                        if line.strip()]
                per_seed.append(band_metrics(rows, auroc=stats.auroc))
            result[f"{model}|{arm}"] = {
                band: {key: float(np.mean([s[band][key] for s in per_seed]))
                       for key in per_seed[0][band]}
                for band in BANDS}
    return result


def table(result: dict, *, threshold: float = THRESHOLD, n_seeds: int = len(SEEDS)) -> str:
    """One row per model, band and arm, so every number sits beside the arm it belongs to."""
    body, sizes = [], set()
    for m, model in enumerate(MODELS):
        if m:
            body.append(r"\addlinespace")
        for b, band in enumerate(BANDS):
            for a, arm in enumerate(ARMS):
                v = result[f"{model}|{arm}"][band]
                sizes.add((band, int(round(v["n"]))))
                cells = [MODEL_NAMES[model] if b == 0 and a == 0 else "",
                         BAND_NAMES[band] if a == 0 else "", ARM_NAMES[arm],
                         f"{v['error']:.3f}", f"{v['above_threshold']:.2f}",
                         f"{v['confident_error']:.3f}", f"{v['auroc']:.3f}"]
                body.append(" & ".join(cells) + r" \\")
    counts = ", ".join(f"{BAND_NAMES[b]}: {n:,}".replace(",", "{,}")
                       for b, n in sorted(sizes, key=lambda t: BANDS.index(t[0])))
    words = {3: "three", 4: "four", 5: "five"}
    return "\n".join([
        r"\begin{table}[t]", r"\centering",
        "\\caption{Error and confidence by annotator agreement. The ambiguity-stratified set split by how many of "
        "the five annotators chose the majority label (items per band: "
        f"{counts}), averaged over {words.get(n_seeds, n_seeds)} seeds. Error is the error rate; "
        f"$>{threshold}$ is the share of items shown with confidence above {threshold}; "
        "wrong and $>" f"{threshold}" "$ is the share that are both, which is what a person "
        "reading the score is misled by; AUROC is computed within the band. $\\uparrow$ "
        "marks a column where higher is better and $\\downarrow$ one where lower is; the "
        f"share above {threshold} has no better direction, since it should be high where "
        "annotators agreed and low where they did not. Descriptive, outside the declared "
        "families.}",
        r"\label{tab:bands}", r"\footnotesize",
        r"\begin{tabular}{lllrrrr}", r"\toprule",
        "Model & Band & Arm & Error$\\downarrow$ & $>" f"{threshold}" "$ & Wrong and $>"
        f"{threshold}" "$ $\\downarrow$ & AUROC$\\uparrow$ \\\\",
        r"\midrule", *body, r"\bottomrule", r"\end{tabular}", r"\end{table}"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", type=Path, default=Path("runs_final"))
    ap.add_argument("--out", type=Path, default=Path("results/agreement_bands.json"))
    args = ap.parse_args(argv)
    result = collect(args.runs)
    for key, bands in result.items():
        print(f"  {key:24s} " + " | ".join(
            f"{b}: err {v['error']:.3f} confErr {v['confident_error']:.3f} "
            f">{THRESHOLD} {v['above_threshold']:.2f} AUROC {v['auroc']:.3f}"
            for b, v in bands.items()))
    args.out.write_text(json.dumps({"set": SET, "threshold": THRESHOLD,
                                    "seeds": list(SEEDS), "arms": result}, indent=2) + "\n")
    (args.out.parent / "table_agreement_bands.tex").write_text(table(result) + "\n")
    print(f"wrote {args.out} and {args.out.parent / 'table_agreement_bands.tex'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
