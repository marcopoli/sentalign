#!/usr/bin/env python3
"""How many runs the paper reports, and what they cost in GPU-hours.

The manuscript quotes one run count and one GPU-hour total. Both are computed here from
the manifests of the runs each analysis reads, so adding a study moves them with it rather
than leaving a number typed once to stop tracking the runs. A run without a manifest is
still training or evaluating: it is listed as in progress and counted in neither total. The
natural language inference runs are not reported in the paper, so they are not counted.

    python scripts/17_compute_budget.py --runs runs_final --out results
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sentalign.config import SEEDS                                    # noqa: E402

RUN_ID = re.compile(r"^(?P<name>[^_]+)__(?P<model>[^_]+)__(?P<objective>.+?)__eps"
                    r"(?P<eps>[\d.]+)__tau(?P<tau>[\d.]+)__n(?P<n>\w+?)__s(?P<seed>\d+)$")
MODELS = ("lfm-1.2b", "qwen-2b", "smollm3-3b")
#: The thirteen arms of the landscape table.
LANDSCAPE = ("sft", "sft_soft", "dpo", "rdpo", "ipo", "grdpo", "mixdpo", "simpo", "alphapo",
             "kto", "kto-dreg1.0", "rdpo-dreg1.0", "ipo-dreg1.0")
LAMBDA = ("kto-dreg0.1", "kto-dreg0.3", "kto-dreg3.0")
GOEMO = ("sft", "kto", "rdpo", "kto-dreg1.0", "rdpo-dreg1.0")
#: The two studies declared on 28 September 2026 (plan.beta_sensitivity, plan.rpo_comparison).
BETA = tuple(f"{objective}-beta{beta}{suffix}" for objective in ("kto", "rdpo")
             for beta in (0.03, 0.3) for suffix in ("", "-dreg1.0"))
RPO = ("rdpo-rpo1.0",)


def study(name: str, model: str, objective: str) -> str | None:
    """The analysis a run belongs to, or None when no reported analysis reads it."""
    if name == "main" and model in MODELS and objective in LANDSCAPE:
        return "sentiment landscape"
    if name == "main" and model == "lfm-1.2b" and objective in LAMBDA:
        return "anchor-weight check"
    if name == "main" and model in ("lfm-1.2b", "qwen-2b") and objective == "kto-dreg-hard":
        return "hard-target ablation"
    if name == "main" and model == "lfm-1.2b" and objective in BETA:
        return "beta sensitivity"
    if name == "main" and model in ("lfm-1.2b", "qwen-2b") and objective in RPO:
        return "RPO comparison"
    if name == "goemo" and model in MODELS and objective in GOEMO:
        return "GoEmotions replication"
    return None


def budget(runs: Path, *, seeds=SEEDS) -> dict:
    groups: dict = {}
    in_progress: list[str] = []
    for run_dir in sorted(runs.iterdir()):
        m = RUN_ID.match(run_dir.name)
        if not m or int(m["seed"]) not in seeds:
            continue
        if m["n"] != "8000" or m["eps"] != "0.0" or m["tau"] != "0.2":
            continue
        group = study(m["name"], m["model"], m["objective"])
        if group is None:
            continue
        manifest = run_dir / "manifest.json"
        if not manifest.exists():
            in_progress.append(run_dir.name)
            continue
        hours = float(json.loads(manifest.read_text()).get("gpu_hours") or 0.0)
        g = groups.setdefault(group, {"runs": 0, "gpu_hours": 0.0})
        g["runs"] += 1
        g["gpu_hours"] += hours
    total = sum(g["gpu_hours"] for g in groups.values())
    for g in groups.values():
        g["gpu_hours"] = round(g["gpu_hours"], 1)
    return {"groups": groups, "runs": sum(g["runs"] for g in groups.values()),
            "gpu_hours": round(total, 1), "in_progress": in_progress}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", type=Path, default=Path("runs_final"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args(argv)
    result = budget(args.runs)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "compute_budget.json").write_text(json.dumps(result, indent=2) + "\n")
    for name, g in sorted(result["groups"].items()):
        print(f"  {name:28} {g['runs']:4d} runs {g['gpu_hours']:7.1f} GPU-h")
    print(f"  {'total':28} {result['runs']:4d} runs {result['gpu_hours']:7.1f} GPU-h")
    if result["in_progress"]:
        print(f"  in progress, not counted: {len(result['in_progress'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
