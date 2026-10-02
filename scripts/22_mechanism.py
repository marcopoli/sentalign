#!/usr/bin/env python3
"""What the preference objectives change in the confidence when ranking is lost.

Proposition 1 says the preference losses leave the order of c unconstrained; it does not say
what changes when the order is lost. Two candidate descriptions are measured for every arm
and model of the landscape against its supervised baseline:

inflation              the change in the median log-odds of c, ``log(c / (1 - c))``, which
                       is how much more certain the model has become across the board;
agreement sensitivity  the change in the Spearman correlation between c and the share of
                       annotators who chose the majority label, computed on the correctly
                       answered items only. Restricting to correct answers keeps the measure
                       from restating correctness, which the AUROC already scores: it asks
                       whether the model is surer of the right answers people agreed on than
                       of the right answers they split over.

Both read the exact order of c (``sentalign.confidence.order_score``), as every ranking metric
in the paper does, and follow the declared endpoint: computed per evaluation set, averaged
over the four DynaSent sets within a seed, then over the five seeds. Across the arm-by-model
cells the script reports how closely each description tracks the change in AUROC, overall,
within each model, and for agreement sensitivity with inflation partialled out. This is a
descriptive analysis designed after the confirmatory one and outside the declared families.

    python scripts/22_mechanism.py --runs runs_final --out results
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

SETS = ("ambig_eval", "r1_test", "r2_test", "sst_dev_validated")
MODELS = ("lfm-1.2b", "qwen-2b", "smollm3-3b")
BASELINE = "sft"
#: The thirteen arms of the landscape table.
ARMS = ("sft", "sft_soft", "dpo", "rdpo", "ipo", "grdpo", "mixdpo", "simpo", "alphapo",
        "kto", "kto-dreg1.0", "rdpo-dreg1.0", "ipo-dreg1.0")
LABELS = {"sft": "SFT", "sft_soft": "SFT-soft", "dpo": "DPO", "rdpo": "R-DPO", "ipo": "IPO",
          "grdpo": "GR-DPO", "mixdpo": "MixDPO", "simpo": "SimPO", "alphapo": "AlphaPO",
          "kto": "KTO", "kto-dreg1.0": "KTO+reg", "rdpo-dreg1.0": "R-DPO+reg",
          "ipo-dreg1.0": "IPO+reg"}
#: Colour follows the role an arm plays in the paper, the same hues as the near-certain
#: figure: the two objectives that lose ranking, the anchored arms, everything else.
GROUPS = (("KTO", ("kto",), "#c0392b"),
          ("R-DPO", ("rdpo",), "#b8860b"),
          ("anchored (+reg)", ("kto-dreg1.0", "rdpo-dreg1.0", "ipo-dreg1.0"), "#1f77b4"),
          ("other objectives", ("sft_soft", "dpo", "ipo", "grdpo", "mixdpo", "simpo",
                                "alphapo"), "#8c8c8c"))
MARKERS = {"lfm-1.2b": "o", "qwen-2b": "s", "smollm3-3b": "^"}
#: The few cells the text argues from carry a direct label; the rest are read by group.
DIRECT_LABELS = {"d_log_odds": (("qwen-2b", "dpo", "DPO, GR-DPO", (0, 9)),
                                ("lfm-1.2b", "ipo", "IPO", (6, -3)),
                                ("qwen-2b", "mixdpo", "MixDPO", (6, -3)),
                                ("smollm3-3b", "mixdpo", "MixDPO", (6, -3))),
                 "d_sensitivity": (("lfm-1.2b", "ipo", "IPO", (6, -3)),
                                   ("qwen-2b", "mixdpo", "MixDPO", (6, -3)),
                                   ("smollm3-3b", "mixdpo", "MixDPO", (6, -3)))}


def _stats():
    spec = importlib.util.spec_from_file_location("regulariser_stats",
                                                  ROOT / "scripts" / "06_regulariser_stats.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["regulariser_stats"] = module
    spec.loader.exec_module(module)
    return module


def ranks(values) -> np.ndarray:
    """Average ranks, ties sharing the mean of the ranks they span."""
    values = np.asarray(values, dtype=float)
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    return (np.cumsum(counts) - (counts - 1) / 2.0)[inverse]


def pearson(x, y) -> float:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y) -> float:
    return pearson(ranks(x), ranks(y))


def partial(x, y, z) -> float:
    """Correlation of x and y with z partialled out of both."""
    rxy, rxz, ryz = pearson(x, y), pearson(x, z), pearson(y, z)
    return float((rxy - rxz * ryz) / np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2)))


def set_metrics(rows, *, auroc) -> dict:
    """AUROC, agreement sensitivity and median log-odds of c on one evaluation set."""
    score = np.array([order_score(r) for r in rows])
    agreement = np.array([max(r["p_human"]) for r in rows])
    correct = np.array([bool(r["correct"]) for r in rows])
    return {"auroc": auroc(score, correct),
            "sensitivity": spearman(score[correct], agreement[correct]),
            "log_odds": float(np.median(score))}


def _rows(path: Path):
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("correct") is not None:
                yield row


def collect(runs: Path, *, models=MODELS, arms=ARMS, seeds=SEEDS, stats=None) -> dict:
    """Seed means of the per-seed, four-set averages, keyed ``model|arm``."""
    stats = stats or _stats()
    out, missing = {}, []
    for model in models:
        for arm in arms:
            per_seed = []
            for seed in seeds:
                run = runs / f"main__{model}__{arm}__eps0.0__tau0.2__n8000__s{seed}" / "eval"
                paths = [run / f"predictions_{name}.jsonl" for name in SETS]
                if not all(p.exists() for p in paths):
                    missing.append(run.parent.name)
                    continue
                cells = [set_metrics(list(_rows(p)), auroc=stats.auroc) for p in paths]
                per_seed.append({k: float(np.mean([c[k] for c in cells])) for k in cells[0]})
            if per_seed:
                out[f"{model}|{arm}"] = {k: float(np.mean([s[k] for s in per_seed]))
                                         for k in per_seed[0]} | {"seeds": len(per_seed)}
    if missing:
        raise SystemExit("runs not scored on every set:\n  " + "\n  ".join(missing))
    return out


def summarise(means, *, models=MODELS) -> dict:
    """Each arm's change against its model's baseline, and how the changes co-vary."""
    cells = []
    for key, v in means.items():
        model, arm = key.split("|")
        if arm == BASELINE:
            continue
        base = means[f"{model}|{BASELINE}"]
        cells.append({"model": model, "arm": arm,
                      **{f"d_{k}": v[k] - base[k] for k in ("auroc", "sensitivity",
                                                            "log_odds")}})
    col = {k: [c[k] for c in cells] for k in ("d_auroc", "d_sensitivity", "d_log_odds")}
    within = {}
    for model in models:
        mine = [c for c in cells if c["model"] == model]
        within[model] = {
            "sensitivity": pearson([c["d_auroc"] for c in mine],
                                   [c["d_sensitivity"] for c in mine]),
            "inflation": pearson([c["d_auroc"] for c in mine], [c["d_log_odds"] for c in mine])}
    return {"cells": cells, "n_cells": len(cells),
            "sensitivity": {"pearson": pearson(col["d_auroc"], col["d_sensitivity"]),
                            "spearman": spearman(col["d_auroc"], col["d_sensitivity"]),
                            "partial_given_inflation": partial(
                                col["d_auroc"], col["d_sensitivity"], col["d_log_odds"])},
            "inflation": {"pearson": pearson(col["d_auroc"], col["d_log_odds"]),
                          "spearman": spearman(col["d_auroc"], col["d_log_odds"]),
                          "partial_given_sensitivity": partial(
                              col["d_auroc"], col["d_log_odds"], col["d_sensitivity"])},
            "within_model": within}


def figure(result, out_path: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # The journal asks for Helvetica or Arial lettering in figures; fonts are embedded.
    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                         "mathtext.fontset": "custom", "mathtext.rm": "Arial",
                         "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
                         "pdf.fonttype": 42})

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.9), sharey=True)
    panels = (("d_log_odds", "inflation",
               "change in median log-odds of $c$\n(inflation)"),
              ("d_sensitivity", "sensitivity",
               "change in Spearman $\\rho$($c$, annotator agreement),\ncorrect answers only"))
    for ax, (key, which, xlabel) in zip(axes, panels):
        ax.axhline(0, color="#bbbbbb", linewidth=0.6, zorder=0)
        ax.axvline(0, color="#bbbbbb", linewidth=0.6, zorder=0)
        for label, arms, colour in GROUPS:
            for model, marker in MARKERS.items():
                pts = [c for c in result["cells"] if c["arm"] in arms and c["model"] == model]
                ax.scatter([c[key] for c in pts], [c["d_auroc"] for c in pts], s=26,
                           marker=marker, color=colour, edgecolor="white", linewidth=0.8,
                           zorder=3)
        cells = {(c["model"], c["arm"]): c for c in result["cells"]}
        for model, arm, text, offset in DIRECT_LABELS[key]:
            c = cells[(model, arm)]
            ax.annotate(text, (c[key], c["d_auroc"]), xytext=offset,
                        textcoords="offset points", fontsize=6.5, color="#333333",
                        ha={1: "left", 0: "center", -1: "right"}[int(np.sign(offset[0]))],
                        va="center")
        r = result[which]["pearson"]
        ax.text(0.97 if which == "inflation" else 0.03, 0.95, f"Pearson $r$ = {r:.2f}",
                transform=ax.transAxes, ha="right" if which == "inflation" else "left",
                va="top", fontsize=8, color="#333333")
        ax.set_xlabel(xlabel, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25, linewidth=0.5)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("change in AUROC against SFT\n(higher is better)", fontsize=8)
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=colour, markersize=5,
                          label=label) for label, _, colour in GROUPS]
    handles += [plt.Line2D([], [], marker=marker, linestyle="", color="#555555",
                           markerfacecolor="none", markersize=5, label=display_name(model))
                for model, marker in MARKERS.items()]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False,
               fontsize=7, bbox_to_anchor=(0.5, 1.07), handletextpad=0.2, columnspacing=0.9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", type=Path, default=Path("runs_final"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args(argv)
    means = collect(args.runs)
    result = {"means": means} | summarise(means)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "mechanism.json").write_text(json.dumps(result, indent=2) + "\n")
    figure(result, args.out / "fig_mechanism.pdf")
    for key, v in means.items():
        print(f"  {key:26} AUROC={v['auroc']:.3f} rho_correct={v['sensitivity']:.3f} "
              f"median log-odds={v['log_odds']:5.1f}")
    for which in ("sensitivity", "inflation"):
        print(f"  {which:12} " + "  ".join(f"{k}={v:+.3f}" for k, v in result[which].items()))
    for model, v in result["within_model"].items():
        print(f"  within {model:12} sensitivity={v['sensitivity']:+.3f} "
              f"inflation={v['inflation']:+.3f}")
    print(f"  cells: {result['n_cells']}; wrote {args.out / 'fig_mechanism.pdf'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
