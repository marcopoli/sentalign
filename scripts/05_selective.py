"""Selective prediction: does an arm know when it is wrong?

Every distributional objective in this study collapses to temperature-calibrated SFT when
scored by JSD, because JSD is sensitive to the *scale* of a distribution and one scalar
fixes the scale. Risk-coverage is sensitive to the *ordering* of confidence instead, which
a temperature largely preserves (measured drift 0.0054 across a 32x temperature range), so
it can see a difference that JSD structurally cannot.

Every arm is scored at its own best temperature, so the comparison is between methods
after the cheap fix rather than before it, and nothing here can be won by calibration
alone. Tests are seed-paired, because the unit of replication is the seed and pooling
items across seeds overstates power by treating correlated draws as independent.

    python scripts/05_selective.py --runs runs --eval-set ambig_eval --model lfm-1.2b
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


def aurc(confidence: np.ndarray, correct: np.ndarray) -> float:
    """Area under the risk-coverage curve. Lower is better."""
    order = np.argsort(-confidence)
    c = np.asarray(correct, dtype=float)[order]
    coverage = np.arange(1, len(c) + 1)
    return float((np.cumsum(1.0 - c) / coverage).mean())


def best_aurc(probs: np.ndarray, correct: np.ndarray, grid: int = 40) -> float:
    """AURC at the temperature that minimises it, i.e. the arm's own best case."""
    log_p = np.log(np.clip(probs, EPS, 1.0))
    out = []
    for t in np.logspace(-0.8, 1.0, grid):
        z = log_p / t
        z -= z.max(axis=1, keepdims=True)
        e = np.exp(z)
        out.append(aurc((e / e.sum(axis=1, keepdims=True)).max(axis=1), correct))
    return min(out)


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues.items(), key=lambda kv: kv[1])
    m, adjusted, running = len(ordered), {}, 0.0
    for i, (key, p) in enumerate(ordered):
        running = max(running, min(1.0, (m - i) * p))
        adjusted[key] = running
    return adjusted


def paired_t(diffs: list[float]) -> tuple[float, float]:
    """Two-sided p from a paired t on the seed differences, via a normal-ish table."""
    from statistics import mean, stdev

    n = len(diffs)
    if n < 2:
        return float("nan"), float("nan")
    m, sd = mean(diffs), stdev(diffs)
    if sd == 0:
        # Every seed moved by the same amount. A nonzero shift is as decisive as the data
        # can be; a zero shift is no evidence of a difference, and returning p = 0 for it
        # would report the absence of an effect as the strongest effect in the table.
        return (float("inf") if m > 0 else float("-inf"), 0.0) if m else (0.0, 1.0)
    t = m / (sd / n ** 0.5)
    # Student t survival via the incomplete beta, without SciPy's stats module.
    from math import lgamma, log

    df = n - 1
    x = df / (df + t * t)

    def betainc(a, b, x, terms=2000):
        if x <= 0:
            return 0.0
        if x >= 1:
            return 1.0
        lbeta = lgamma(a) + lgamma(b) - lgamma(a + b)
        front = np.exp(a * log(x) + b * log(1 - x) - lbeta) / a
        f, c, d = 1.0, 1.0, 0.0
        for i in range(terms):
            m_ = i // 2
            if i == 0:
                num = 1.0
            elif i % 2 == 0:
                num = (m_ * (b - m_) * x) / ((a + 2 * m_ - 1) * (a + 2 * m_))
            else:
                num = -((a + m_) * (a + b + m_) * x) / ((a + 2 * m_) * (a + 2 * m_ + 1))
            d = 1.0 + num * d
            d = 1e-30 if abs(d) < 1e-30 else d
            d = 1.0 / d
            c = 1.0 + num / c
            c = 1e-30 if abs(c) < 1e-30 else c
            f *= c * d
            if abs(1.0 - c * d) < 1e-10:
                break
        return front * (f - 1.0)

    return t, float(betainc(df / 2.0, 0.5, x))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=Path("runs_final"))
    ap.add_argument("--eval-set", default="ambig_eval")
    ap.add_argument("--model", default="lfm-1.2b")
    ap.add_argument("--name", default="main")
    ap.add_argument("--baseline", default="sft")
    ap.add_argument("--n", default="8000")
    ap.add_argument("--arms", nargs="+", default=None,
                    help="restrict the comparison family. Holm over every arm in the "
                         "directory is the wrong correction for a pre-registered "
                         "hypothesis about a few of them, and with 5 seeds it erases "
                         "effects at t=-3.4. Name the family in advance and report it.")
    args = ap.parse_args()

    scores: dict[str, dict[int, float]] = defaultdict(dict)
    for run_dir in sorted(args.runs.iterdir()):
        m = RUN_ID.match(run_dir.name)
        if not m or m["model"] != args.model or m["name"] != args.name:
            continue
        if m["n"] != args.n or m["eps"] != "0.0" or m["tau"] != "0.2":
            continue
        path = run_dir / "eval" / f"predictions_{args.eval_set}.jsonl"
        if not path.exists():
            continue
        probs, correct = [], []
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("correct") is None:
                continue
            probs.append(row["probs"])
            correct.append(1.0 if row["correct"] else 0.0)
        if len(probs) < 50:
            continue
        scores[m["objective"]][int(m["seed"])] = best_aurc(
            np.array(probs, dtype=float), np.array(correct, dtype=float))

    base = scores.get(args.baseline)
    if not base:
        print(f"no {args.baseline} runs under {args.runs} for {args.model}/{args.name}")
        return 1

    raw, rows = {}, {}
    for objective, by_seed in scores.items():
        if objective == args.baseline:
            continue
        if args.arms and objective not in args.arms:
            continue
        shared = sorted(set(by_seed) & set(base))
        if len(shared) < 3:
            continue
        diffs = [by_seed[s] - base[s] for s in shared]
        t, p = paired_t(diffs)
        rows[objective] = (float(np.mean([by_seed[s] for s in shared])),
                           float(np.mean(diffs)), t, len(shared))
        raw[objective] = p
    adjusted = holm(raw)

    b = [base[s] for s in sorted(base)]
    print(f"{args.model} | {args.eval_set} | AURC at each arm's own best temperature")
    print(f"  seeds: {len(b)}   baseline {args.baseline} = {np.mean(b):.4f} "
          f"(sd {np.std(b):.4f})   family of {len(rows)}\n")
    print(f"{'arm':14s} {'AURC':>8s} {'Δ vs base':>10s} {'t':>7s} {'p (Holm)':>10s} {'n':>3s}")
    for objective, (mean, delta, t, n) in sorted(rows.items(), key=lambda kv: kv[1][1]):
        star = "***" if adjusted[objective] < .001 else \
               "**" if adjusted[objective] < .01 else \
               "*" if adjusted[objective] < .05 else ""
        print(f"{objective:14s} {mean:8.4f} {delta:+10.4f} {t:+7.2f} "
              f"{adjusted[objective]:10.4f}{star} {n:3d}")
    print("\nLower AURC is better. Scored after each arm's own optimal temperature, so "
          "nothing here is won by calibration.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
