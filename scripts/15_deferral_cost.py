#!/usr/bin/env python3
"""What a deferral policy costs: the paper's human-centred consequence, priced.

The ranking results say how well confidence orders a model's errors. A deployment asks a
narrower question: if a person reviews whatever the model is least sure about, and a wrong
answer that is not reviewed costs more than a review, what does each objective cost per
item? This script answers it from the predictions already on disk.

The cost model is deliberately plain. A review costs 1 and resolves the item; an item the
model keeps costs nothing if it is right and ``r`` if it is wrong, ``r`` being how many
reviews an undetected error is worth. The policy defers every item whose confidence is
below a threshold. Two reference policies bound it: deferring everything costs 1 per
item, deferring nothing costs ``r`` times the error rate.

The threshold is chosen out of sample. Each seed's items, pooled over the four evaluation
sets, are split in two by a hash of their identity; the threshold that minimises cost on
one half is paid for on the other, and the two directions are averaged. Choosing and
scoring on the same items would reward a model for the noise in its own confidences. The
split depends only on the item, so every arm of a model is split identically and the
arms are compared on the same items.

Reviewers are assumed to resolve every item they see. That favours deferral equally for
every arm, so it moves the level of every curve and not the comparison between them; the
manuscript states it as an assumption.

    python scripts/15_deferral_cost.py --runs runs_final --out results
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sentalign.config import SEEDS                                  # noqa: E402
from sentalign.modeling import display_name                        # noqa: E402
from sentalign.confidence import order_score, top_probability   # noqa: E402

SETS = ("ambig_eval", "r1_test", "r2_test", "sst_dev_validated")
MODELS = ("lfm-1.2b", "qwen-2b", "smollm3-3b")
ARMS = (("sft", "SFT"), ("kto", "KTO"), ("kto-dreg1.0", "KTO+reg"),
        ("rdpo", "R-DPO"), ("rdpo-dreg1.0", "R-DPO+reg"))
TABLE_RATIOS = (3.0, 10.0, 30.0)
FIGURE_RATIOS = tuple(float(r) for r in np.geomspace(1.0, 50.0, 40))
CURVES = (("sft", "SFT", "#444444", "-"),
          ("kto", "KTO", "#c0392b", "--"),
          ("kto-dreg1.0", "KTO+reg", "#1f77b4", "-"))
#: Paired comparisons reported beside the table, as (first, second).
CONTRASTS = (("kto-dreg1.0", "kto"), ("kto-dreg1.0", "sft"), ("rdpo-dreg1.0", "rdpo"))
T975_DF4 = 2.776


def load_items(run_dir: Path, sets=SETS):
    """``(item key, order score, correct)`` over the scored sets, or None if one is missing.

    The threshold is on the exact order of the confidence (``sentalign.confidence``): a
    threshold on ``c`` and one on its order score defer the same items, except where a double
    has rounded ``c`` to 1.0 and could not defer any of them.
    """
    out = []
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
            out.append((f"{name}/{row['text_id']}", float(order_score(row)),
                        1.0 if row["correct"] else 0.0))
    return out


def half(key: str) -> int:
    """0 or 1, from the item alone, so every arm splits the same items the same way."""
    return hashlib.sha256(key.encode()).digest()[0] & 1


def choose_threshold(conf, correct, ratio: float) -> float:
    """The threshold minimising ``deferred + ratio * wrong_and_kept`` per item.

    An item is deferred when its confidence is strictly below the threshold, so tied
    confidences are always deferred or kept together and the result does not depend on
    the order the items were written in. The candidates are -inf (defer nothing), the
    midpoints between consecutive distinct confidences, and +inf (defer everything).
    Midpoints rather than the observed values themselves, because the threshold is applied
    to other items: "defer nothing" chosen on one half must defer nothing on the other,
    not everything below the lowest confidence the first half happened to contain. Ties in
    cost go to the lower threshold, the policy that defers less.
    """
    conf = np.asarray(conf, dtype=float)
    wrong = 1.0 - np.asarray(correct, dtype=float)
    n = len(conf)
    values, inverse = np.unique(conf, return_inverse=True)
    count_at = np.bincount(inverse, minlength=len(values)).astype(float)
    wrong_at = np.bincount(inverse, weights=wrong, minlength=len(values))
    deferred_below = np.concatenate([[0.0], np.cumsum(count_at)])      # below values[j]
    wrong_below = np.concatenate([[0.0], np.cumsum(wrong_at)])
    total_wrong = wrong.sum()
    cost = (deferred_below + ratio * (total_wrong - wrong_below)) / n  # j = 0..len(values)
    j = int(np.argmin(cost))
    if j == 0:
        return float("-inf")
    if j == len(values):
        return float("inf")
    return float((values[j - 1] + values[j]) / 2.0)


def policy_cost(conf, correct, threshold: float, ratio: float) -> dict:
    conf = np.asarray(conf, dtype=float)
    wrong = 1.0 - np.asarray(correct, dtype=float)
    deferred = conf < threshold
    wrong_kept = wrong[~deferred].sum() / len(conf)
    return {"cost": float(deferred.mean() + ratio * wrong_kept),
            "deferred": float(deferred.mean()),
            "wrong_kept": float(wrong_kept)}


def cross_fitted(items, ratio: float) -> dict:
    """Choose on one half, pay on the other, both ways, and average."""
    halves = [[(c, k) for key, c, k in items if half(key) == h] for h in (0, 1)]
    results = []
    for fit, score in ((0, 1), (1, 0)):
        c_fit, k_fit = zip(*halves[fit])
        c_score, k_score = zip(*halves[score])
        t = choose_threshold(c_fit, k_fit, ratio)
        results.append(policy_cost(c_score, k_score, t, ratio))
    return {key: float(np.mean([r[key] for r in results])) for key in results[0]}


def collect(runs: Path, models=MODELS, arms=ARMS, seeds=SEEDS) -> dict:
    out: dict = {}
    for model in models:
        for arm, _ in arms:
            for seed in seeds:
                run = runs / f"main__{model}__{arm}__eps0.0__tau0.2__n8000__s{seed}"
                items = load_items(run)
                if items is None:
                    raise SystemExit(f"{run.name} is not scored on every set; the costs "
                                     "are compared on the declared seeds or not at all")
                out[(model, arm, seed)] = items
    return out


def summarise(collected, models=MODELS, arms=ARMS, seeds=SEEDS, ratios=TABLE_RATIOS):
    per_seed: dict = {}
    for model in models:
        for arm, _ in arms:
            for seed in seeds:
                items = collected[(model, arm, seed)]
                err = 1.0 - float(np.mean([k for _, _, k in items]))
                for r in ratios:
                    per_seed[(model, arm, seed, r)] = {**cross_fitted(items, r),
                                                       "defer_none": r * err}
    means = {}
    for model in models:
        for arm, _ in arms:
            for r in ratios:
                cells = [per_seed[(model, arm, s, r)] for s in seeds]
                means[(model, arm, r)] = {k: float(np.mean([c[k] for c in cells]))
                                          for k in cells[0]}
    contrasts = {}
    for model in models:
        for first, second in CONTRASTS:
            for r in ratios:
                d = np.array([per_seed[(model, first, s, r)]["cost"]
                              - per_seed[(model, second, s, r)]["cost"] for s in seeds])
                half_width = T975_DF4 * d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0.0
                base = np.mean([per_seed[(model, second, s, r)]["cost"] for s in seeds])
                contrasts[(model, first, second, r)] = {
                    "diff": float(d.mean()), "ci95": [float(d.mean() - half_width),
                                                      float(d.mean() + half_width)],
                    "relative": float(d.mean() / base), "favours_first_on": int((d < 0).sum()),
                    "n": len(d)}
    return means, contrasts


def table(means, models=MODELS, arms=ARMS, ratios=TABLE_RATIOS) -> str:
    body = []
    for model in models:
        first = True
        for arm, label in arms:
            cells = [display_name(model) if first else "", label]
            for r in ratios:
                m = means[(model, arm, r)]
                cells.append(f"{m['cost']:.3f} ({100 * m['deferred']:.0f}\\%)")
            body.append(" & ".join(cells) + r" \\")
            first = False
        body.append(r"\addlinespace")
    # Every cell is a cost, so lower is better throughout; the arrow sits on the spanning
    # header. The share deferred beside it has no better direction of its own.
    span = (f"& & \\multicolumn{{{len(ratios)}}}{{c}}{{Cost per item $\\downarrow$ "
            f"(share deferred)}} \\\\ \\cmidrule(lr){{3-{2 + len(ratios)}}}")
    head = " & ".join(["Model", "Objective"] + [f"$r = {r:g}$" for r in ratios])
    return "\n".join([
        r"\begin{table}[t]", r"\centering",
        "\\caption{Expected cost per item of a deferral policy, in units of one human "
        "review, with the share of items deferred in brackets. An undetected error costs "
        "$r$ reviews. The confidence threshold is chosen on one half of the items and paid "
        "for on the other, both ways, pooled over the four evaluation sets and averaged "
        "over five seeds. Deferring every item costs $1$. $\\downarrow$: lower is better; "
        "the share deferred has no better direction.}",
        r"\label{tab:deferral}", r"\footnotesize",
        f"\\begin{{tabular}}{{ll{'r' * len(ratios)}}}", r"\toprule", span, head + r" \\",
        r"\midrule", *body[:-1], r"\bottomrule", r"\end{tabular}", r"\end{table}"])


def figure(collected, out_path: Path, models=MODELS, seeds=SEEDS, ratios=FIGURE_RATIOS):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # The journal asks for Helvetica or Arial lettering in figures; fonts are embedded.
    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                         "mathtext.fontset": "custom", "mathtext.rm": "Arial",
                         "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
                         "pdf.fonttype": 42})

    fig, axes = plt.subplots(1, len(models), figsize=(3.4 * len(models), 2.8), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, model in zip(axes, models):
        for arm, label, colour, style in CURVES:
            ys = [np.mean([cross_fitted(collected[(model, arm, s)], r)["cost"]
                           for s in seeds]) for r in ratios]
            ax.plot(ratios, ys, label=label, color=colour, linestyle=style, linewidth=1.6)
        ax.axhline(1.0, color="#999999", linewidth=0.8, linestyle=":")
        ax.set_xscale("log")
        ax.set_xticks([1, 3, 10, 30])
        ax.set_xticklabels(["1", "3", "10", "30"])
        ax.minorticks_off()
        ax.set_title(display_name(model), fontsize=9)
        ax.set_xlabel("cost of an undetected error, $r$ (reviews)")
        ax.grid(alpha=0.25, linewidth=0.5)
    axes[0].set_ylabel("expected cost per item")
    axes[0].text(1.05, 0.965, "defer everything", fontsize=7, color="#777777", va="top")
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", type=Path, default=Path("runs_final"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args(argv)

    collected = collect(args.runs)
    means, contrasts = summarise(collected)
    for (model, arm, r), m in means.items():
        print(f"  {model:11s} {arm:13s} r={r:4g} cost={m['cost']:.3f} "
              f"deferred={m['deferred']:.2f} wrong_kept={m['wrong_kept']:.3f} "
              f"defer_none={m['defer_none']:.3f}")
    for (model, a, b, r), c in contrasts.items():
        print(f"  {model:11s} {a} - {b} r={r:4g} diff={c['diff']:+.3f} "
              f"[{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}] rel={100 * c['relative']:+.0f}% "
              f"lower on {c['favours_first_on']}/{c['n']}")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "table_deferral.tex").write_text(table(means) + "\n")
    (args.out / "deferral_cost.json").write_text(json.dumps({
        "ratios": list(TABLE_RATIOS), "seeds": list(SEEDS), "sets": list(SETS),
        "means": {f"{m}|{a}|{r:g}": v for (m, a, r), v in means.items()},
        "contrasts": {f"{m}|{a}-{b}|{r:g}": v for (m, a, b, r), v in contrasts.items()},
    }, indent=2) + "\n")
    for model in MODELS:
        for arm, label, _, _ in CURVES:
            ceiling = next((r for r in FIGURE_RATIOS
                            if np.mean([cross_fitted(collected[(model, arm, s)], r)["cost"]
                                        for s in SEEDS]) >= 1.0 - 1e-9), None)
            print(f"  {model:11s} {label:8s} reaches defer-everything at r = "
                  + (f"{ceiling:.1f}" if ceiling else "never (up to 50)"))
    if not args.no_figure:
        figure(collected, args.out / "fig_deferral_cost.pdf")
    print(f"wrote {args.out / 'table_deferral.tex'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
