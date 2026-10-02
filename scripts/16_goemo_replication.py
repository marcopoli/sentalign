#!/usr/bin/env python3
"""The second sentiment corpus: do the damage and the repair reproduce on GoEmotions?

The families were declared with the runs (``sentalign.plan``, 23 September 2026), before
any GoEmotions result existed:

    A  KTO and R-DPO against SFT                       AUROC and accuracy
    B  KTO+reg against KTO, R-DPO+reg against R-DPO     AUROC and accuracy

each Holm-corrected within its panel over every model that ran, which is what
``--models`` defaults to. The
endpoint is the per-seed mean over the two GoEmotions evaluation sets, the held-out
ambiguity-stratified set and the official test split, for the reason the main endpoint
averages four sets: a seed contributes one number.

A declared cell without five seeds is an error, not a smaller family, exactly as in the
main analysis: a family that shrinks silently is corrected less strictly than the one that
was declared. ``--interim`` lifts that for a look at runs still in progress, and marks
the output as interim so it cannot be mistaken for the declared analysis.

    python scripts/16_goemo_replication.py --runs runs_final --out results
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
from sentalign.modeling import display_name                          # noqa: E402

STUDY = "goemo"
SETS = ("ambig_eval", "go_test")
MODELS = ("lfm-1.2b", "qwen-2b")
#: Every model the plan can run on GoEmotions: the two of ``goemo`` and the third of the
#: separate ``goemo-smollm3`` study.
CANDIDATES = (*MODELS, "smollm3-3b")
BASELINE = "sft"
DEGRADATION = (("kto", "KTO"), ("rdpo", "R-DPO"))
REPAIR = (("kto-dreg1.0", "KTO+reg", "kto"), ("rdpo-dreg1.0", "R-DPO+reg", "rdpo"))
METRICS = (("auroc", r"$\Delta$AUROC$\uparrow$"),
           ("acc", r"$\Delta$accuracy$\uparrow$"))
ARMS = (BASELINE, *(a for a, _ in DEGRADATION), *(a for a, _, _ in REPAIR))


def _nli():
    spec = importlib.util.spec_from_file_location("nli_robustness",
                                                  ROOT / "scripts" / "12_nli_robustness.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["nli_robustness"] = module
    spec.loader.exec_module(module)
    return module


def models_that_ran(runs: Path, candidates=CANDIDATES) -> list[str]:
    """The declaration corrects over every model that ran, so the family is read from disk.

    A default of the two planned models would let a third that ran drop out of the family
    unnoticed, and a family that shrinks is corrected less strictly than the declared one.
    """
    names = [d.name for d in runs.iterdir()]
    return [m for m in candidates if any(n.startswith(f"{STUDY}__{m}__") for n in names)]


def collect(runs: Path, models=MODELS, *, seeds=SEEDS, nli=None) -> dict:
    """Per model, per arm, per declared seed: the mean over the two evaluation sets."""
    nli = nli or _nli()
    out = {}
    for model in models:
        scores = nli.load(runs, model, sets=SETS, study=STUDY)
        out[model] = {arm: {s: v for s, v in per_seed.items() if s in seeds}
                      for arm, per_seed in scores.items() if arm in ARMS}
    return out


def shortfall(scores: dict, *, seeds=SEEDS) -> list[str]:
    """Declared cells without every declared seed."""
    return [f"{model}/{arm} has {len(scores[model].get(arm, {}))} of {len(seeds)} seeds"
            for model in scores for arm in ARMS
            if set(scores[model].get(arm, {})) != set(seeds)]


def panels(scores: dict, *, nli=None) -> dict:
    """The two declared families, each corrected once across every model."""
    nli = nli or _nli()
    selective = nli._modules()[0]
    out = {}
    for panel, comparisons in (
            ("A", [(arm, BASELINE) for arm, _ in DEGRADATION]),
            ("B", [(arm, twin) for arm, _, twin in REPAIR])):
        rows = []
        for model, per_arm in scores.items():
            for first, second in comparisons:
                for metric, _ in METRICS:
                    row = nli.paired(per_arm, first, second, metric, selective=selective)
                    if row is not None:
                        rows.append(row | {"model": model})
        adjusted = selective.holm({i: r["p"] for i, r in enumerate(rows)})
        for i, row in enumerate(rows):
            row["p_holm"] = adjusted[i]
        out[panel] = rows
    return out


def table(result: dict, *, interim: bool = False, nli=None) -> str:
    nli = nli or _nli()
    labels = dict(DEGRADATION) | {arm: label for arm, label, _ in REPAIR} | {"sft": "SFT"}
    body = []
    titles = {"A": "A. Against SFT", "B": "B. Anchored arm against its twin"}
    for panel in ("A", "B"):
        rows = result[panel]
        if not rows:
            continue
        if body:
            body.append(r"\addlinespace")
        body.append(f"\\multicolumn{{4}}{{@{{}}l}}{{\\emph{{{titles[panel]}, Holm over "
                    f"{len(rows)}}}}} \\\\")
        cells: dict = {}
        for row in rows:
            cells.setdefault((row["model"], row["first"], row["second"]),
                             {})[row["metric"]] = row
        previous = None
        for (model, first, second), metrics in cells.items():
            name = display_name(model) if model != previous else ""
            previous = model
            line = [name, f"{labels[first]} vs {labels[second]}"]
            line += [nli._cell(metrics[m]) if m in metrics else "--" for m, _ in METRICS]
            body.append(" & ".join(line) + r" \\")
    caption = ("The second sentiment corpus. GoEmotions rater labels mapped to the same four "
               "labels, same prompt and protocol. Per-seed mean over the held-out "
               "ambiguity-stratified set and the official test split, five seeds, paired "
               "$t$ on seeds, Holm correction within each panel across models."
               + nli.DIRECTION_NOTE)
    if interim:
        caption = "INTERIM, not the declared analysis: some cells lack seeds. " + caption
    return "\n".join([
        r"\begin{table}[t]", r"\centering", f"\\caption{{{caption}}}",
        r"\label{tab:goemo}", r"\footnotesize",
        r"\begin{tabular}{ll" + "l" * len(METRICS) + "}", r"\toprule",
        " & ".join(["Model", "Comparison"]
                   + [f"{label} [95\\% CI]" for _, label in METRICS]) + r" \\",
        r"\midrule", *body, r"\bottomrule", r"\end{tabular}", r"\end{table}"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", type=Path, default=Path("runs_final"))
    ap.add_argument("--models", nargs="+", default=None,
                    help="default: every model with a GoEmotions run, as declared")
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--interim", action="store_true",
                    help="report whatever has run, marked as interim")
    args = ap.parse_args(argv)

    nli = _nli()
    args.models = args.models or models_that_ran(args.runs)
    scores = collect(args.runs, args.models, nli=nli)
    missing = shortfall(scores)
    if missing and not args.interim:
        raise SystemExit("declared cells are incomplete (use --interim to look anyway):\n  "
                         + "\n  ".join(missing))
    result = panels(scores, nli=nli)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "table_goemo.tex").write_text(table(result, interim=bool(missing), nli=nli)
                                             + "\n")
    means = {model: {arm: {"seeds": sorted(v), **{
        m: float(np.mean([x[m] for x in v.values()])) for m in ("auroc", "eaurc", "acc", "ece")}}
        for arm, v in sorted(per_arm.items()) if v} for model, per_arm in scores.items()}
    (args.out / "goemo_replication.json").write_text(json.dumps(
        {"sets": list(SETS), "models": args.models, "interim": bool(missing),
         "missing": missing, "means": means, "panels": result}, indent=2) + "\n")
    for model, per_arm in means.items():
        for arm, v in per_arm.items():
            print(f"  {model:11} {arm:14} n={len(v['seeds'])} auroc={v['auroc']:.4f} "
                  f"acc={v['acc']:.4f} ece={v['ece']:.4f}")
    if missing:
        print("INTERIM: " + "; ".join(missing))
    print(f"wrote {args.out / 'table_goemo.tex'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
