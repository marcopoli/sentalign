"""Does an arm know *which items* are ambiguous?

A global temperature is a one-parameter family that rescales sharpness uniformly, so any
method whose output is a fixed monotone rescaling of the reference logits is
indistinguishable from calibrated SFT. The only way a distributional objective can beat
calibration is by learning item-conditional entropy.

This measures exactly that, and it is deliberately chosen to be un-gameable by
calibration: the rank correlation between predicted entropy and annotator entropy is
close to invariant under a common temperature, so an arm cannot buy a better score by
being better calibrated. The script reports that invariance rather than assuming it.

    python scripts/04_entropy_diagnostic.py --runs runs_final --eval-set ambig_eval
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

RUN_ID = re.compile(
    r"^(?P<name>[^_]+)__(?P<model>[^_]+)__(?P<objective>.+?)__eps(?P<eps>[\d.]+)"
    r"__tau(?P<tau>[\d.]+)__n(?P<n>\d+|all)__s(?P<seed>\d+)$")
EPS = 1e-12


def entropy(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1.0)
    p = p / p.sum(-1, keepdims=True)
    return -(p * np.log(p)).sum(-1)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / denom) if denom else float("nan")


def temper(probs: np.ndarray, t: float) -> np.ndarray:
    z = np.log(np.clip(probs, EPS, 1.0)) / t
    z -= z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=Path("runs_final"))
    ap.add_argument("--eval-set", default="ambig_eval")
    ap.add_argument("--model", default=None, help="restrict to one model")
    ap.add_argument("--name", default=None, help="restrict to one study prefix")
    args = ap.parse_args()

    per_arm: dict[tuple[str, str], list[float]] = defaultdict(list)
    conc: dict[tuple[str, str], list[float]] = defaultdict(list)
    invariance: list[tuple[float, float]] = []

    for run_dir in sorted(args.runs.iterdir()):
        m = RUN_ID.match(run_dir.name)
        if not m:
            continue
        if args.model and m["model"] != args.model:
            continue
        if args.name and m["name"] != args.name:
            continue
        path = run_dir / "eval" / f"predictions_{args.eval_set}.jsonl"
        if not path.exists():
            continue
        probs, human, logits = [], [], []
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("p_human") is None:
                continue
            probs.append(row["probs"])
            human.append(row["p_human"])
            logits.append(row.get("logits"))
        if len(probs) < 30:
            continue
        q, p = np.array(probs, float), np.array(human, float)
        hq, hp = entropy(q), entropy(p)
        rho = spearman(hq, hp)
        per_arm[(m["model"], m["objective"])].append(rho)

        # The concentration channel: sum_k exp(l_k), the logit magnitude softmax
        # discards. Polya supervises it directly; for every other arm it is whatever
        # happened to end up there, which is exactly the baseline it must beat.
        if logits[0] is not None:
            z = np.array(logits, float)
            a0 = np.exp(z - z.max(axis=-1, keepdims=True)).sum(axis=-1)
            conc[(m["model"], m["objective"])].append(spearman(-a0, hp))
        # Is the statistic actually calibration-proof? Recompute under a strong
        # temperature and record the drift.
        invariance.append((rho, spearman(entropy(temper(q, 2.5)), hp)))

    if not per_arm:
        print(f"no runs with predictions_{args.eval_set}.jsonl under {args.runs}")
        return 1

    drift = np.array([b - a for a, b in invariance])
    print(f"eval set: {args.eval_set}   runs: {sum(len(v) for v in per_arm.values())}")
    print(f"temperature invariance of the statistic: mean |drift| under T=2.5 "
          f"is {np.abs(drift).mean():.4f} (0 = calibration cannot change the ranking)\n")
    print(f"{'model':10s} {'arm':14s} {'rho(H_model, H_human)':>22s} {'sd':>7s} {'n':>3s}")
    for (model, arm) in sorted(per_arm):
        v = np.array(per_arm[(model, arm)])
        print(f"{model:10s} {arm:14s} {v.mean():>22.4f} {v.std():>7.4f} {len(v):>3d}")

    if conc:
        print(f"\n{'model':10s} {'arm':14s} {'rho(-concentration, H_human)':>30s} {'n':>4s}")
        for key in sorted(conc):
            v = np.array(conc[key])
            print(f"{key[0]:10s} {key[1]:14s} {v.mean():>30.4f} {len(v):>4d}")
        print("  (the degree of freedom softmax discards; Polya supervises it directly)")

    print("\nRead: an arm beats calibrated SFT only if it ranks item ambiguity better.")
    print("A tie with sft here means the objective adds nothing calibration cannot.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
