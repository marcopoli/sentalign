#!/usr/bin/env python3
"""GoEmotions at the operating points a deployment meets: kept-item error and review cost.

Table 7 reports the GoEmotions replication as differences in AUROC and accuracy. A reader
deciding whether to trust a model needs the same translation the DynaSent results get in
Tables 2 and 6: the error rate among the items a model keeps when it abstains on the least
confident half, and the cost of the cheapest deferral policy in units of one review. Both
are computed here with the functions that produce those tables, unchanged, so the two
corpora are priced the same way. Items are pooled over the two GoEmotions evaluation sets
within a seed; the deferral threshold is chosen on one half of the items and paid for on
the other. Descriptive, outside the declared families.

    python scripts/18_goemo_operating_points.py --runs runs_final --out results
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
from sentalign.confidence import top_probability                     # noqa: E402
from sentalign.modeling import display_name                          # noqa: E402

STUDY = "goemo"
SETS = ("ambig_eval", "go_test")
MODELS = ("lfm-1.2b", "qwen-2b", "smollm3-3b")
ARMS = (("sft", "SFT"), ("kto", "KTO"), ("kto-dreg1.0", "KTO+reg"),
        ("rdpo", "R-DPO"), ("rdpo-dreg1.0", "R-DPO+reg"))
COVERAGE = 0.5
RATIO = 10.0
#: The confidence a reviewer might learn to accept unread, as in the agreement-band table.
CONFIDENT = 0.9
#: Paired contrasts reported beside the table, as (first, second).
CONTRASTS = (("kto-dreg1.0", "kto"), ("rdpo-dreg1.0", "rdpo"), ("kto-dreg1.0", "sft"))


def _script(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _shown(run: Path) -> np.ndarray:
    """``c(x)`` per scored item, in the order ``load_items`` reads them."""
    out = []
    for name in SETS:
        for line in (run / "eval" / f"predictions_{name}.jsonl").read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get("correct") is not None:
                    out.append(top_probability(row))
    return np.array(out)


def collect(runs: Path, models=MODELS, arms=ARMS, seeds=SEEDS, *, deferral=None) -> dict:
    """Per (model, arm, seed): the ``(key, order score, correct)`` items that rank and
    defer, and the shown probability ``c`` of each item, which the threshold counts read.
    Every cell is required."""
    deferral = deferral or _script("deferral_cost", "15_deferral_cost.py")
    out, missing = {}, []
    for model in models:
        for arm, _ in arms:
            for seed in seeds:
                run = runs / f"{STUDY}__{model}__{arm}__eps0.0__tau0.2__n8000__s{seed}"
                items = deferral.load_items(run, sets=SETS) if run.exists() else None
                if not items:
                    missing.append(run.name)
                else:
                    out[(model, arm, seed)] = {"items": items, "shown": _shown(run)}
    if missing:
        raise SystemExit("runs not scored on every set:\n  " + "\n  ".join(missing))
    return out


def summarise(collected, models=MODELS, arms=ARMS, seeds=SEEDS, *, coverage=None,
              deferral=None) -> dict:
    coverage = coverage or _script("coverage_table", "10_coverage_table.py")
    deferral = deferral or _script("deferral_cost", "15_deferral_cost.py")
    per_seed = {}
    for key, cell in collected.items():
        items = cell["items"]
        pairs = [(c, k) for _, c, k in items]
        conf = cell["shown"]
        wrong = 1.0 - np.array([k for _, k in pairs])
        per_seed[key] = {"error_full": coverage.risk_at_coverage(pairs, 1.0),
                         "error_half": coverage.risk_at_coverage(pairs, COVERAGE),
                         "confident": float((conf > CONFIDENT).mean()),
                         "confident_error": float(((conf > CONFIDENT) * wrong).mean()),
                         **deferral.cross_fitted(items, RATIO)}
    means = {f"{m}|{a}": {k: float(np.mean([per_seed[(m, a, s)][k] for s in seeds]))
                          for k in per_seed[(m, a, seeds[0])]}
             for m in models for a, _ in arms}
    contrasts = {}
    present = {a for a, _ in arms}
    for m in models:
        for first, second in (c for c in CONTRASTS if set(c) <= present):
            d = np.array([per_seed[(m, first, s)]["cost"] - per_seed[(m, second, s)]["cost"]
                          for s in seeds])
            base = float(np.mean([per_seed[(m, second, s)]["cost"] for s in seeds]))
            contrasts[f"{m}|{first}-{second}"] = {
                "diff": float(d.mean()), "relative": float(d.mean() / base),
                "favours_first_on": int((d < 0).sum()), "n": len(d)}
    return {"means": means, "contrasts": contrasts}


def table(result, models=MODELS, arms=ARMS) -> str:
    body = []
    for i, model in enumerate(models):
        if i:
            body.append(r"\addlinespace")
        for j, (arm, label) in enumerate(arms):
            v = result["means"][f"{model}|{arm}"]
            body.append(" & ".join([
                display_name(model) if j == 0 else "", label,
                f"{v['error_full']:.3f}", f"{v['error_half']:.3f}",
                f"{v['confident']:.2f}", f"{v['confident_error']:.3f}",
                f"{v['cost']:.3f} ({100 * v['deferred']:.0f}\\%)"]) + r" \\")
    return "\n".join([
        r"\begin{table}[t]", r"\centering",
        "\\caption{GoEmotions at the operating points of a deployment. Error rate among all "
        "items (coverage 100\\%) and among the most confident half (50\\%); the share of "
        f"items shown with confidence above {CONFIDENT}, and the share that are both wrong and "
        "shown so, which is what a person trusting the score is misled by; and the expected "
        "cost per item of the cheapest deferral policy when an undetected error costs "
        f"$r = {RATIO:g}$ reviews, with the share deferred in brackets; the threshold is chosen on one half of the items and paid for "
        "on the other, as in Table~\\ref{tab:deferral}. Items pooled over the two GoEmotions "
        "evaluation sets within a seed, averaged over five seeds. Deferring every item costs "
        "$1$. $\\downarrow$: lower is better; the shares above the confidence threshold and "
        "deferred have no better direction. Descriptive, outside the declared families.}",
        r"\label{tab:goemo-operating}", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{llrrrrr}", r"\toprule",
        r"& & \multicolumn{2}{c}{Error $\downarrow$} & & & \\ \cmidrule(lr){3-4}",
        f"Model & Objective & 100\\% & 50\\% & $>{CONFIDENT}$ & Wrong and $>{CONFIDENT}$ "
        f"$\\downarrow$ & Cost $\\downarrow$ at $r = {RATIO:g}$ \\\\",
        r"\midrule", *body, r"\bottomrule", r"\end{tabular}", r"\end{table}"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", type=Path, default=Path("runs_final"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args(argv)
    result = summarise(collect(args.runs))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "table_goemo_operating.tex").write_text(table(result) + "\n")
    (args.out / "goemo_operating_points.json").write_text(json.dumps(
        {"sets": list(SETS), "coverage": COVERAGE, "ratio": RATIO, **result},
        indent=2) + "\n")
    for key, v in result["means"].items():
        print(f"  {key:24} err={v['error_full']:.3f} err@50={v['error_half']:.4f} "
              f">{CONFIDENT}={v['confident']:.3f} wrong&>{CONFIDENT}={v['confident_error']:.4f} "
              f"cost@{RATIO:g}={v['cost']:.4f} deferred={v['deferred']:.2f}")
    for key, c in result["contrasts"].items():
        print(f"  {key:32} {100 * c['relative']:+.1f}% lower on {c['favours_first_on']}/{c['n']}")
    print(f"wrote {args.out / 'table_goemo_operating.tex'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
