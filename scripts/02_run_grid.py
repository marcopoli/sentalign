"""Run the experiment programme, resumably, on one GPU.

    python scripts/02_run_grid.py --dry-run             inspect before committing GPU time
    python scripts/02_run_grid.py                       run everything, skipping what is done
    python scripts/02_run_grid.py --studies main noise  run selected studies
    python scripts/02_run_grid.py --retry-failed        re-attempt runs that failed
    python scripts/02_run_grid.py --max-hours 20        stop cleanly after a time budget

Written in Python rather than shell because the resume logic needs the same definition of
"complete" that the rest of the package uses, and because a failure in one run must not
take down the programme: each run is executed in a subprocess, so an out-of-memory error
or a segfault in run 90 of 262 costs that run and nothing else.

Preference arms need their model's own SFT checkpoint as the reference policy, so the
ordering below always trains SFT for a (model, seed) before anything that anchors on it.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import Counter
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sentalign.plan import STUDIES, full_plan                     # noqa: E402
from sentalign.runlog import is_complete, read_status             # noqa: E402

#: Arms that must exist before the arms that anchor on them.
REFERENCE_STAGE = ("sft",)


def order_runs(runs):
    """SFT first within each (model, seed), then everything that depends on it."""
    return sorted(runs, key=lambda r: (r.model, r.seed,
                                       0 if r.objective in REFERENCE_STAGE else 1,
                                       r.objective))


def run_dir_for(run, output_root: Path, data_dir: Path):
    cfg = run.to_config()
    cfg.output_root = output_root
    # Only the default task takes the caller's build directory. A task bundle carries its
    # own, and overwriting it here would evaluate the NLI runs against the sentiment sets.
    if run.task == "sentiment":
        cfg.data.build_dir = data_dir
    return cfg.run_dir, cfg


def build_command(run, cfg, data_dir: Path, output_root: Path) -> list[str]:
    cmd = [sys.executable, "-m", "sentalign.cli", "train",
           "--task", run.task, "--output-root", str(output_root),
           "--model", run.model, "--seed", str(run.seed),
           "--objective", cfg.train.objective, "--tau", str(run.tau),
           "--noise", str(run.noise_epsilon), "--name", run.name]
    # The task bundle carries its own build directory, so --data is passed only for the
    # default task, where the caller's value is the meaningful one.
    if run.task == "sentiment":
        cmd += ["--data", str(data_dir)]
    if cfg.variant:
        # Without this the subprocess recomputes the run_id minus the variant and every
        # cspo-ablation run lands in the main-grid cspo directory, reads as complete,
        # and the whole study silently produces nothing.
        cmd += ["--variant", cfg.variant]
    if run.train_subsample is not None:
        cmd += ["--subsample", str(run.train_subsample)]
    for key, value in run.overrides.items():
        cmd += ["--set", f"{key}={value}"]
    return cmd


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data/build"))
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--studies", nargs="+", choices=sorted(STUDIES))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--retry-failed", action="store_true",
                    help="re-attempt runs whose status is failed or interrupted")
    ap.add_argument("--max-hours", type=float, default=None,
                    help="stop launching new runs after this much wall-clock")
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--skip-baselines", action="store_true")
    ap.add_argument("--stop-on-error", action="store_true",
                    help="halt the programme on the first failure instead of continuing")
    args = ap.parse_args()

    runs = order_runs(full_plan(tuple(args.studies) if args.studies else None))
    args.runs.mkdir(parents=True, exist_ok=True)
    ledger = args.runs / "programme.jsonl"

    todo, done, retry, baselines = [], 0, 0, []
    for run in runs:
        if run.objective.startswith(("prompt", "encoder")):
            baselines.append(run)
            continue        # dispatched separately, below
        run_dir, cfg = run_dir_for(run, args.runs, args.data)
        if is_complete(run_dir):
            done += 1
            continue
        status = (read_status(run_dir) or {}).get("status")
        if status in ("failed", "interrupted") and not args.retry_failed:
            continue
        if status in ("failed", "interrupted"):
            retry += 1
        todo.append((run, cfg, run_dir))

    print(f"plan: {len(runs)} runs, {done} already complete, {len(todo)} to run "
          f"({retry} retries)")
    est = sum(r.hours for r, _, _ in todo)
    print(f"estimated {est:.1f} GPU-h, about {timedelta(hours=int(est))}")
    if args.dry_run:
        for run, cfg, run_dir in todo:
            print(f"  {run.study:<16} {run_dir.name}")
        return 0

    started = time.time()
    outcomes: Counter = Counter()
    for index, (run, cfg, run_dir) in enumerate(todo, 1):
        elapsed_h = (time.time() - started) / 3600
        if args.max_hours is not None and elapsed_h >= args.max_hours:
            print(f"\nreached the {args.max_hours} h budget after {index - 1} runs; "
                  f"rerun to continue")
            break

        cmd = build_command(run, cfg, args.data, args.runs)
        print(f"\n[{index}/{len(todo)}] {run.study}: {run_dir.name} "
              f"(est {run.hours:.2f} h, elapsed {elapsed_h:.1f} h)")
        print("  " + " ".join(cmd))

        run_started = time.time()
        # Each run is a subprocess: a CUDA fault or an OOM that escapes recovery kills
        # the child, not the programme.
        proc = subprocess.run(cmd)
        duration = time.time() - run_started
        ok = proc.returncode == 0

        if ok and not args.skip_eval:
            eval_cmd = [sys.executable, "-m", "sentalign.cli", "eval",
                        "--run", str(run_dir), "--data", str(cfg.data.build_dir)]
            ok = subprocess.run(eval_cmd).returncode == 0

        outcomes["ok" if ok else "failed"] += 1
        with ledger.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "index": index, "study": run.study, "run_id": run_dir.name,
                "returncode": proc.returncode, "ok": ok,
                "wallclock_s": round(duration, 1),
                "estimated_hours": run.hours,
                "measured_hours": round(duration / 3600, 4),
            }) + "\n")

        if not ok:
            print(f"  FAILED (exit {proc.returncode}); see {run_dir}/console.log")
            if args.stop_on_error:
                return 1

    if baselines and not args.skip_baselines:
        print(f"\n== baselines ({len(baselines)} runs) ==")
        kinds = sorted({"encoder" if b.objective.startswith("encoder") else "prompt"
                        for b in baselines})
        cmd = [sys.executable, "-m", "sentalign.cli", "baselines",
               "--data", str(args.data), "--output-root", str(args.runs),
               "--kinds", *kinds]
        print("  " + " ".join(cmd))
        outcomes["ok" if subprocess.run(cmd).returncode == 0 else "failed"] += 1

    total_h = (time.time() - started) / 3600
    print(f"\nfinished: {dict(outcomes)} in {total_h:.1f} GPU-h")
    print(f"ledger: {ledger}")

    if not args.dry_run and outcomes["ok"]:
        print("\naggregating")
        subprocess.run([sys.executable,
                        str(Path(__file__).resolve().parent / "03_make_tables.py"),
                        "--runs", str(args.runs), "--out", str(args.results)])
    return 0 if not outcomes["failed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
