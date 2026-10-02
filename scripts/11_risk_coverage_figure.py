#!/usr/bin/env python3
"""The risk-coverage curve: the finding in the form a deployment meets it.

One panel per model. Each curve is the error rate among the items a model keeps as it
abstains on the least confident ones, averaged over seeds at each coverage. A model whose
confidence carries information bends down to the left; one whose confidence does not stays
flat, and the gap between those two shapes is what AUROC summarises in a single number.

Curves are averaged over seeds at fixed coverage, never pooled across seeds, for the same
reason the primary endpoint averages seeds: pooling would treat five runs as one long run.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sentalign.modeling import display_name                        # noqa: E402

CURVES = (("sft", "SFT", "#444444", "-"),
          ("kto", "KTO", "#c0392b", "--"),
          ("kto-dreg1.0", "KTO+reg", "#1f77b4", "-"))
GRID = np.linspace(0.05, 1.0, 40)


def coverage_module():
    path = ROOT / "scripts" / "10_coverage_table.py"
    spec = importlib.util.spec_from_file_location("coverage_table", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["coverage_table"] = module
    spec.loader.exec_module(module)
    return module


def curve(seed_items, grid=GRID, *, risk_at_coverage) -> np.ndarray:
    """Mean over seeds of the risk at each coverage, seed by seed."""
    per_seed = np.array([[risk_at_coverage(items, c) for c in grid]
                         for items in seed_items.values()])
    return per_seed.mean(axis=0)


def figure(collected, models, out_path: Path, *, risk_at_coverage, grid=GRID):
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
            seeds = collected.get((model, arm))
            if not seeds:
                continue
            ax.plot(grid, curve(seeds, grid, risk_at_coverage=risk_at_coverage),
                    label=label, color=colour, linestyle=style, linewidth=1.6)
        ax.set_title(display_name(model), fontsize=9)
        ax.set_xlabel("coverage")
        ax.grid(alpha=0.25, linewidth=0.5)
    axes[0].set_ylabel("error rate among kept items")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", type=Path, default=Path("runs_final"))
    ap.add_argument("--models", nargs="+", default=["lfm-1.2b", "qwen-2b"])
    ap.add_argument("--out", type=Path, default=Path("results/fig_risk_coverage.pdf"))
    args = ap.parse_args(argv)

    cov = coverage_module()
    collected = cov.collect(args.runs, set(args.models),
                            arms=tuple((a, l) for a, l, _, _ in CURVES))
    missing = [(m, a) for m in args.models for a, _, _, _ in CURVES
               if (m, a) not in collected]
    if missing:
        raise SystemExit(f"no scored runs for {missing}")
    path = figure(collected, args.models, args.out,
                  risk_at_coverage=cov.risk_at_coverage)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
