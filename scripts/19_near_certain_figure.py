#!/usr/bin/env python3
"""Errors shown as near-certain, per 1,000 items, on both sentiment corpora.

The figure the Discussion argues from: how many of the answers a person would read as
certain are wrong. Confidence above 0.9 is the threshold a reviewer might learn to accept
unread, the same one the agreement-band table and Table 10 use. A threshold count does not
depend on how ties are ordered, so it reads the stored probabilities exactly as a user
would see them.

Items are pooled over each corpus's evaluation sets within a seed (the four DynaSent sets,
the two GoEmotions sets), the share is averaged over the five declared seeds, and the whisker
is one seed standard deviation. Descriptive, outside the declared families.

    python scripts/19_near_certain_figure.py --runs runs_final --out results
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sentalign.config import SEEDS                                    # noqa: E402
from sentalign.confidence import top_probability                     # noqa: E402
from sentalign.modeling import display_name                          # noqa: E402

CONFIDENT = 0.9
MODELS = ("lfm-1.2b", "qwen-2b", "smollm3-3b")
CORPORA = (("DynaSent", "main", ("ambig_eval", "r1_test", "r2_test", "sst_dev_validated")),
           ("GoEmotions", "goemo", ("ambig_eval", "go_test")))
#: SFT is the neutral reference, as in every figure of the paper; the anchored arms share
#: the blue family and a hatch, so identity never rests on colour alone. The four coloured
#: slots pass the palette validator (adjacent CVD separation >= 20.9).
ARMS = (("sft", "SFT", "#444444", None),
        ("kto", "KTO", "#c0392b", None),
        ("kto-dreg1.0", "KTO+reg", "#1f77b4", "////"),
        ("rdpo", "R-DPO", "#b8860b", None),
        ("rdpo-dreg1.0", "R-DPO+reg", "#4a9bd1", "////"))


def near_certain(run_dir: Path, sets) -> dict | None:
    """Shares of items above the threshold, and of items both wrong and above it."""
    conf, wrong = [], []
    for name in sets:
        path = run_dir / "eval" / f"predictions_{name}.jsonl"
        if not path.exists():
            return None
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("correct") is None:
                continue
            conf.append(top_probability(row))
            wrong.append(0.0 if row["correct"] else 1.0)
    conf, wrong = np.array(conf), np.array(wrong)
    above = conf > CONFIDENT
    return {"confident": float(above.mean()), "confident_error": float((above * wrong).mean()),
            "n": int(len(conf))}


def collect(runs: Path, models=MODELS, seeds=SEEDS) -> dict:
    out, missing = {}, []
    for corpus, study, sets in CORPORA:
        for model in models:
            for arm, *_ in ARMS:
                per_seed = []
                for seed in seeds:
                    run = runs / f"{study}__{model}__{arm}__eps0.0__tau0.2__n8000__s{seed}"
                    v = near_certain(run, sets)
                    if v is None:
                        missing.append(run.name)
                    else:
                        per_seed.append(v)
                if per_seed:
                    out[f"{corpus}|{model}|{arm}"] = {
                        k: float(np.mean([v[k] for v in per_seed]))
                        for k in ("confident", "confident_error")} | {
                        "confident_error_sd": float(np.std([v["confident_error"]
                                                            for v in per_seed], ddof=1)),
                        "seeds": len(per_seed)}
    if missing:
        raise SystemExit("runs not scored on every set:\n  " + "\n  ".join(missing))
    return out


def figure(result, out_path: Path, models=MODELS) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # The journal asks for Helvetica or Arial lettering in figures; fonts are embedded.
    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                         "mathtext.fontset": "custom", "mathtext.rm": "Arial",
                         "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
                         "pdf.fonttype": 42})

    plt.rcParams["hatch.linewidth"] = 0.6
    fig, axes = plt.subplots(1, len(CORPORA), figsize=(3.4 * len(CORPORA), 2.7), sharey=True)
    width = 0.16
    x = np.arange(len(models))
    for ax, (corpus, _, _) in zip(axes, CORPORA):
        for j, (arm, label, colour, hatch) in enumerate(ARMS):
            cells = [result[f"{corpus}|{m}|{arm}"] for m in models]
            ys = [1000 * c["confident_error"] for c in cells]
            err = [1000 * c["confident_error_sd"] for c in cells]
            ax.bar(x + (j - 2) * width, ys, width * 0.92, color=colour, hatch=hatch,
                   edgecolor="white", linewidth=0.8, label=label,
                   yerr=err, error_kw={"elinewidth": 0.6, "ecolor": "#555555", "capsize": 0})
            if hatch:
                # The anchored bars are the result and nearly invisible at this scale, so
                # they alone carry their value; the other bars are read off the axis.
                for xi, y, e in zip(x + (j - 2) * width, ys, err):
                    ax.text(xi, y + e + 3, f"{y:.0f}", ha="center", va="bottom", fontsize=6,
                            color="#333333")
        ax.set_xticks(x)
        ax.set_xticklabels([display_name(m) for m in models], fontsize=8)
        ax.set_title(corpus, fontsize=9)
        ax.grid(axis="y", alpha=0.25, linewidth=0.5)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel(f"wrong and shown above {CONFIDENT},\nper 1,000 items (lower is better)",
                       fontsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(ARMS), frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def table(result, models=MODELS) -> str:
    """The figure's numbers, as the table view the figure needs for its lighter bars."""
    body = []
    for i, (corpus, _, _) in enumerate(CORPORA):
        if i:
            body.append(r"\addlinespace")
        for j, model in enumerate(models):
            cells = [corpus if j == 0 else "", display_name(model)]
            for arm, *_ in ARMS:
                v = result[f"{corpus}|{model}|{arm}"]
                cells.append(f"{1000 * v['confident_error']:.0f} ({100 * v['confident']:.0f}\\%)")
            body.append(" & ".join(cells) + r" \\")
    head = " & ".join(["Corpus", "Model"] + [label for _, label, _, _ in ARMS])
    return "\n".join([
        r"\begin{table}[t]", r"\centering",
        f"\\caption{{Errors shown with confidence above {CONFIDENT}, per 1{{,}}000 items, with "
        f"the share of all items shown above {CONFIDENT} in brackets; the numbers behind "
        "the near-certain-error figure. Items pooled over each corpus's evaluation sets within "
        "a seed, averaged over five seeds. $\\downarrow$: lower is better for the count; the "
        "share in brackets has no better direction. Descriptive, outside the declared "
        "families.}",
        r"\label{tab:near-certain}", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{ll" + "r" * len(ARMS) + "}", r"\toprule",
        f"& & \\multicolumn{{{len(ARMS)}}}{{c}}{{Wrong and $>{CONFIDENT}$ per 1{{,}}000 "
        f"$\\downarrow$ (share $>{CONFIDENT}$)}} \\\\ \\cmidrule(lr){{3-{2 + len(ARMS)}}}",
        head + r" \\", r"\midrule", *body, r"\bottomrule", r"\end{tabular}", r"\end{table}"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", type=Path, default=Path("runs_final"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args(argv)
    result = collect(args.runs)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "near_certain.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.out / "table_near_certain.tex").write_text(table(result) + "\n")
    figure(result, args.out / "fig_near_certain.pdf")
    for key, v in result.items():
        print(f"  {key:36} wrong&>{CONFIDENT} per 1000 = {1000 * v['confident_error']:6.1f}  "
              f">{CONFIDENT} = {100 * v['confident']:5.1f}%")
    print(f"wrote {args.out / 'fig_near_certain.pdf'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
