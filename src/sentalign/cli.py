"""Command-line entry points.

Importing this module imports Unsloth first, before TRL or transformers, which is
required for its trainer patches to apply. Every command therefore goes through here
rather than importing the training modules directly.

    sentalign plan                                  print the run plan and its cost
    sentalign build   --out data/build              build every dataset
    sentalign train   --model lfm-1.2b --objective cspo --seed 13
    sentalign eval    --run runs/<run_id>
    sentalign tables  --runs runs --out results
    sentalign status  --runs runs                   what has finished, failed, or stalled
    sentalign doctor                                environment and hardware check
"""

from __future__ import annotations

# Import order matters: Unsloth patches TRL at import time.
from . import modeling  # noqa: F401  (isort: skip)

import argparse
import json
import sys
from pathlib import Path

from .config import SEEDS, ExperimentConfig, apply_task
from .plan import STUDIES, full_plan, summarise
from .runlog import is_complete, read_status
from .train.objectives import (ALL_OBJECTIVES, DISTRIBUTIONAL_OBJECTIVES,
                                FROM_BASE)


def _load_jsonl(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _apply_overrides(cfg: ExperimentConfig, overrides: list[str] | None) -> None:
    for item in overrides or []:
        if "=" not in item:
            raise SystemExit(f"--set expects key=value, got {item!r}")
        key, raw = item.split("=", 1)
        for section in (cfg.train, cfg.data, cfg.eval):
            if hasattr(section, key):
                current = getattr(section, key)
                value: object = raw
                if isinstance(current, bool):
                    value = raw.lower() in ("1", "true", "yes")
                elif isinstance(current, int) and not isinstance(current, bool):
                    value = int(raw)
                elif isinstance(current, float):
                    value = float(raw)
                elif current is None:
                    value = float(raw) if raw.replace(".", "", 1).isdigit() else raw
                setattr(section, key, value)
                break
        else:
            raise SystemExit(f"unknown setting {key!r}")


def _build_config(args) -> ExperimentConfig:
    cfg = ExperimentConfig(name=args.name, model=args.model,
                           variant=getattr(args, "variant", ""))
    cfg.train.objective = args.objective
    cfg.train.seed = args.seed
    cfg.data.tau = args.tau
    cfg.data.noise_epsilon = args.noise
    if args.subsample is not None:
        cfg.data.train_subsample = args.subsample
        cfg.data.max_train_pairs = args.subsample
    # The task bundle first, so an explicit --data or --set still overrides it.
    apply_task(cfg, getattr(args, "task", "sentiment") or "sentiment")
    if args.data is not None:
        cfg.data.build_dir = Path(args.data)
    cfg.output_root = Path(args.output_root)
    _apply_overrides(cfg, args.set)
    return cfg


def _annotator_targets(build: Path) -> dict:
    """The annotator distribution for every item in the build, keyed two ways.

    ``p_human`` is a property of the item, so it is identical wherever the item appears,
    and the union over the distributional splits is the only map that covers an arbitrary
    split: the preference pairs at n8000 are drawn from the full pool rather than from the
    8k distributional subsample, so joining against ``train_n8000`` matched 1,988 of 8,000
    pairs. Under the KTO path, which dropped the unmatched, that trained the regularised
    arm on a quarter of the corpus every other KTO arm receives, which is a protocol
    difference wearing the label of an objective difference.

    Two failure modes look alike in the data and are not alike at all, so they are
    separated here. One item appearing twice with the same prompt and two distributions
    means two builds are mixed in one directory, which would make the join depend on file
    order, and raises. One *identifier* carrying two different prompts means the corpus
    reuses an identifier for two different items: the NLI build draws ids from SNLI
    (``89202781.jpg#0r1e``) and from MNLI (``4667c``), and four of the short ones collide
    among 8,000 training items. That is resolvable, since every record carries its prompt,
    so the pair map resolves it and the identifier map drops the ambiguous ids rather than
    guessing between two items.
    """
    by_pair: dict = {}
    by_id: dict = {}
    ambiguous: set = set()
    for path in sorted(build.glob("cspo/*.jsonl")):
        for record in _load_jsonl(path):
            key, prompt, value = record["text_id"], record.get("prompt"), record["p_human"]
            pair = (key, prompt)
            if by_pair.get(pair, value) != value:
                raise SystemExit(
                    f"{key} has two different annotator distributions under the same "
                    f"prompt inside {build / 'cspo'}; the directory mixes builds, so "
                    "rebuild it")
            by_pair[pair] = value
            if by_id.get(key, value) != value:
                ambiguous.add(key)
            by_id[key] = value
    if not by_pair:
        raise SystemExit(f"no distributional records under {build / 'cspo'}; the "
                         f"regulariser has no target to score against")
    for key in ambiguous:
        by_id.pop(key, None)
    return {"by_pair": by_pair, "by_id": by_id, "ambiguous": ambiguous}


def _target_for(record, targets):
    """The distribution for one record: by item, then by identifier alone.

    The prompt disambiguates a reused identifier. Falling back to the identifier keeps the
    join working where two splits phrase the prompt differently, and that map has the
    ambiguous identifiers removed, so the fallback can never return another item's target.
    """
    pair = (record.get("text_id"), record.get("prompt"))
    if pair in targets["by_pair"]:
        return targets["by_pair"][pair]
    return targets["by_id"].get(record.get("text_id"))


def _attach_targets(records, targets, *, split: str, drop_unmatched: bool):
    """Join ``p_human`` onto records for the distributional regulariser.

    The training split raises on a miss, because an arm that quietly trains on fewer items
    than its label claims is not the arm being reported. The evaluation split drops
    instead: it drives the checkpoint-selection cadence rather than any reported number,
    and dropping mirrors how GR-DPO handles a dev group absent from training. An empty
    result raises in both cases, since a regularised arm with nothing to regularise is the
    silent failure this join exists to prevent.

    The one exception to raising is an identifier the corpus reuses for two different items
    whose prompt does not resolve it. Those are named and dropped, because attaching either
    candidate would be attaching the wrong item's distribution, and because refusing a run
    over a handful of collisions in a public corpus stops work those collisions do not
    threaten. Anything else missing still raises.
    """
    resolved = [(r, _target_for(r, targets)) for r in records]
    missing = [r for r, value in resolved if value is None]
    unresolvable = [r for r in missing if r.get("text_id") in targets["ambiguous"]]
    if missing and not drop_unmatched:
        if len(unresolvable) != len(missing):
            raise SystemExit(
                f"{len(missing)} of {len(records)} {split} records have no annotator "
                f"distribution (first {missing[0].get('text_id')}); the regulariser cannot "
                f"be applied. Build the preference and distributional splits from the same n.")
        print(f"{split}: dropped {len(unresolvable)} records whose identifier is reused by "
              f"two different items and whose prompt did not resolve it "
              f"({sorted({r.get('text_id') for r in unresolvable})})")
    kept = [(r, value) for r, value in resolved if value is not None]
    if not kept:
        raise SystemExit(f"no {split} record has an annotator distribution; build the "
                         f"preference and distributional splits from the same corpus")
    for record, value in kept:
        record["p_human"] = value
    return [record for record, _ in kept]


def _load_training_data(cfg: ExperimentConfig) -> dict:
    """Assemble exactly the inputs this objective needs."""
    from .data.preferences import apply_flips, load_pairs

    build = Path(cfg.data.build_dir)
    n = cfg.data.train_subsample
    objective = cfg.train.objective
    data: dict = {"eval_records": _load_jsonl(build / "sft" / "dev.jsonl")}

    if objective == "sft":
        data["sft_records"] = _load_jsonl(build / "sft" / f"train_n{n}.jsonl")
    elif objective in DISTRIBUTIONAL_OBJECTIVES:
        records = _load_jsonl(build / "cspo" / f"train_n{n}.jsonl")
        if cfg.data.target_temperature != 1.0 or cfg.data.target_smoothing > 0:
            from .train.objectives import soft_target
            import numpy as np

            votes = np.array([[round(p * 5) for p in r["p_human"]] for r in records])
            shaped = soft_target(votes, temperature=cfg.data.target_temperature,
                                 smoothing=cfg.data.target_smoothing)
            for record, row in zip(records, shaped, strict=True):
                record["p_human"] = [float(v) for v in row]
        data["cspo_records"] = records
    elif objective == "kto":
        examples = _load_jsonl(build / "kto" / "train.jsonl")
        # KTO is unpaired, so `n` cannot mean pairs here. It means the same number of
        # *scored completions*: a DPO pair carries two, so 8000 pairs and 16000 KTO
        # examples put the same volume of sequence through the model each epoch.
        #
        # Without this cap KTO read the whole 115,053-example corpus at every n, 14x
        # what the arms it is compared against receive, which cost 6.5 GPU-h against
        # their 0.36 and made the arm a different experiment under the same label.
        budget = 2 * cfg.data.max_train_pairs if cfg.data.max_train_pairs else None
        if budget and len(examples) > budget:
            import random

            random.Random(cfg.train.seed).shuffle(examples)
            examples = examples[:budget]
        # Capped like the pairwise dev split: this drives the evaluation cadence that
        # checkpoint selection hangs off, not a reported number.
        dev_examples = _load_jsonl(build / "kto" / "dev.jsonl")[:2000]
        if cfg.train.pref_distributional_lambda > 0:
            # KTO's desirability bands drop items, so a training example without a
            # distribution is a build mismatch; the dev split is only a cadence.
            targets = _annotator_targets(build)
            examples = _attach_targets(examples, targets, split="KTO training",
                                       drop_unmatched=False)
            dev_examples = _attach_targets(dev_examples, targets, split="KTO dev",
                                           drop_unmatched=True)
        data["kto_examples"] = examples
        data["kto_dev_examples"] = dev_examples
    else:
        pref_dir = build / "pref" / f"tau{cfg.data.tau}"
        pairs = load_pairs(pref_dir / "train.jsonl")
        if cfg.data.noise_epsilon > 0:
            ladder = json.loads(
                (pref_dir / "noise" / f"eps{cfg.data.noise_epsilon}.json").read_text())
            pairs = apply_flips(pairs, ladder["flip_indices"])
            print(f"applied noise eps={ladder['epsilon']}: "
                  f"{ladder['n_flipped']:,}/{ladder['n_pairs']:,} pairs flipped "
                  f"(realised {ladder['realised_epsilon']})")
        if cfg.data.max_train_pairs and len(pairs) > cfg.data.max_train_pairs:
            import random

            random.Random(cfg.train.seed).shuffle(pairs)
            pairs = pairs[:cfg.data.max_train_pairs]
        data["pref_pairs"] = [p.as_trl() | {"group": p.group, "margin": p.margin,
                                            "text_id": p.text_id}
                              for p in pairs]
        # `margin`, `group` and `text_id` for the same reason the training split carries
        # them: MixDPO routes on the margin, GR-DPO groups by band, the distributional
        # regulariser joins on the id, and each refuses a batch without its column. The
        # dev split reaches the same collators through the periodic evaluation, so
        # omitting them here failed every run of those arms at the first eval step.
        data["pref_dev_pairs"] = [p.as_trl() | {"group": p.group, "margin": p.margin,
                                                "text_id": p.text_id}
                                  for p in load_pairs(pref_dir / "dev.jsonl")[:2000]]
        if cfg.train.pref_distributional_lambda > 0:
            # Join the annotator distribution onto each pair by text_id, rather than
            # rebuilding the preference corpus to carry it. The distributional records
            # already hold p_human for every item, including the no-majority ones a
            # preference pair is drawn from.
            targets = _annotator_targets(build)
            data["pref_pairs"] = _attach_targets(
                data["pref_pairs"], targets, split="preference training",
                drop_unmatched=False)
            data["pref_dev_pairs"] = _attach_targets(
                data["pref_dev_pairs"], targets, split="preference dev",
                drop_unmatched=True)
    return data


# --------------------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------------------

def cmd_plan(args) -> int:
    runs = full_plan(tuple(args.studies) if args.studies else None)
    if args.verbose:
        for run in runs:
            print(f"  {run.study:<16} {run.command()}")
        print()
    # summarise re-costs the plan from the programme ledger when it can find one. Calling
    # it without the directory left that path unreachable from the command line, so the
    # estimate stayed the a-priori one however many runs the machine had already measured.
    print(summarise(runs, args.runs))
    return 0


def cmd_build(args) -> int:
    from subprocess import call

    script = Path(__file__).resolve().parents[2] / "scripts" / "01_build_data.py"
    return call([sys.executable, str(script), "--out", str(args.out)])


def cmd_train(args) -> int:
    from .recovery import AlreadyComplete
    from .train.driver import run_training

    cfg = _build_config(args)
    if is_complete(cfg.run_dir) and not args.force:
        print(f"{cfg.run_id}: already complete, skipping "
              f"(pass --force to retrain)")
        return 0

    reference = None
    if cfg.train.objective not in FROM_BASE:
        reference = Path(args.reference) if args.reference else _default_reference(cfg)
        if not reference.exists():
            print(f"reference policy missing: {reference}\n"
                  f"run `sentalign train --objective sft --model {cfg.model} "
                  f"--seed {cfg.train.seed}` first", file=sys.stderr)
            return 2

    data = _load_training_data(cfg)
    try:
        outcome = run_training(cfg, data, reference_adapter=reference,
                               skip_if_complete=not args.force)
    except AlreadyComplete as exc:
        # A run trained to its step budget but not yet evaluated fails is_complete, so
        # rerunning a block to fill in missing seeds reaches the trainer, resumes, and
        # steps nowhere. Reported as a skip so the rest of the block continues, and the
        # finished run's record is left as it is.
        print(f"{cfg.run_id}: {exc}", file=sys.stderr)
        return 0
    if outcome is None:
        print(f"{cfg.run_id}: already complete")
        return 0
    print(json.dumps(outcome.as_dict(), indent=2))
    return 0


def _default_reference(cfg: ExperimentConfig) -> Path:
    # The SFT anchor is tau-independent: tau shapes only the preference pairs
    # (data/build/pref/tau*), never the SFT corpus (data/build/sft/). Copying
    # cfg.data.tau here made every tau != 0.2 run look for an SFT checkpoint
    # the plan never trains, which failed the whole margin ablation with
    # "reference policy missing". The anchor is pinned to the default tau.
    sft = ExperimentConfig(name=cfg.name, model=cfg.model)
    sft.train.objective = "sft"
    sft.train.seed = cfg.train.seed
    sft.data.train_subsample = cfg.data.train_subsample
    sft.data.max_train_pairs = cfg.data.max_train_pairs
    sft.output_root = cfg.output_root
    return sft.run_dir


def cmd_baselines(args) -> int:
    """Run the encoder and prompting reference points."""
    from .baselines.runner import run_encoder_baseline, run_prompt_baseline

    failures = 0
    for kind in args.kinds:
        if kind == "encoder":
            for soft in (False, True):
                for seed in (SEEDS if args.seed is None else (args.seed,)):
                    cfg = ExperimentConfig(name=args.name, model="deberta")
                    cfg.train.objective = "encoder_soft" if soft else "encoder_hard"
                    cfg.train.seed = seed
                    cfg.data.build_dir = Path(args.data)
                    cfg.output_root = Path(args.output_root)
                    print(f"\n{cfg.run_id}")
                    failures += run_encoder_baseline(
                        cfg, soft_labels=soft, skip_if_complete=not args.force)
        elif kind == "prompt":
            models = args.models or ["lfm-350m", "lfm-1.2b", "qwen-2b"]
            for model in models:
                for shots in (0, 5):
                    cfg = ExperimentConfig(name=args.name, model=model)
                    cfg.train.objective = f"prompt{shots}"
                    cfg.train.seed = 0
                    cfg.data.build_dir = Path(args.data)
                    cfg.output_root = Path(args.output_root)
                    print(f"\n{cfg.run_id}")
                    failures += run_prompt_baseline(
                        cfg, n_shots=shots, skip_if_complete=not args.force)
    return 1 if failures else 0


def cmd_eval(args) -> int:
    from .evaluate.run_eval import evaluate_run

    return evaluate_run(Path(args.run), Path(args.data), force=args.force)


def cmd_tables(args) -> int:
    from subprocess import call

    script = Path(__file__).resolve().parents[2] / "scripts" / "03_make_tables.py"
    return call([sys.executable, str(script), "--runs", str(args.runs),
                 "--out", str(args.out)])


def cmd_status(args) -> int:
    """What has finished, what failed, and what is still marked running."""
    root = Path(args.runs)
    if not root.exists():
        print(f"no runs directory at {root}")
        return 1
    rows = []
    for run_dir in sorted(root.iterdir()):
        if not run_dir.is_dir():
            continue
        if not (run_dir / "status.json").exists():
            # Not a run. Unsloth writes `unsloth_compiled_cache/` into the working
            # directory at import, so running this command from inside runs/ leaves a
            # library cache sitting among the runs, counted as an unknown row forever.
            continue
        status = read_status(run_dir) or {"status": "unknown"}
        # GPU-hours come from the manifest, not from status.json: evaluation opens a
        # second RunLogger on the same directory and its final set_status overwrites
        # elapsed_s with the eval phase's duration, which under-reported the programme
        # by about 3x. The manifest's gpu_hours is written once, at training save.
        hours = None
        manifest = run_dir / "manifest.json"
        if manifest.exists():
            try:
                hours = json.loads(manifest.read_text()).get("gpu_hours")
            except (json.JSONDecodeError, OSError):
                hours = None
        if hours is None:
            hours = status.get("elapsed_s", 0) / 3600
        rows.append((run_dir.name, status.get("status"), round(hours, 2),
                     status.get("error", "")[:60]))
    counts: dict[str, int] = {}
    for _, state, _, _ in rows:
        counts[state] = counts.get(state, 0) + 1

    print(f"{'run':<62}{'status':<14}{'GPU-h':>7}  error")
    print("-" * 110)
    for name, state, hours, error in rows:
        if args.only and state != args.only:
            continue
        print(f"{name:<62}{state:<14}{hours:>7.2f}  {error}")
    print("-" * 110)
    print("  " + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    # Completed runs only. An in-progress run has no manifest, so its cell falls back to
    # status.json's elapsed_s, and adding that partial figure to a total labelled
    # "measured" overstates it by however far the current run has got.
    total = sum(h for _, state, h, _ in rows if state == "completed")
    print(f"  total measured GPU-h: {total:.1f} (completed runs only)")
    return 0


def cmd_doctor(args) -> int:
    """Check the environment before committing days of GPU time to it."""
    from .recovery import preflight
    from .runlog import collect_environment

    env = collect_environment()
    print("environment:")
    for key in ("hostname", "python", "torch", "transformers", "trl", "peft", "unsloth",
                "gpu_name", "gpu_total_memory_gb", "bf16_supported", "nvidia_driver"):
        print(f"  {key:<22} {env.get(key)}")
    print(f"  {'code_fingerprint':<22} {env.get('code_fingerprint')} "
          f"({env.get('code_files')} files)")
    if env.get("git_sha"):
        print(f"  {'git_sha':<22} {env['git_sha']}")
    print()
    print("preflight:")
    report = preflight(output_dir=Path(args.output_root))
    print(report.summary())
    return 0 if report.ok else 1


def build_parser() -> argparse.ArgumentParser:
    """The CLI parser, exposed so tests can drive the exact subprocess boundary.

    The grid runner talks to this module only through argv. Anything the plan encodes
    in a run (the cspo-ablation ``variant`` above all) must survive that round trip, or
    the subprocess recomputes a different run_id and writes into another run's
    directory. A test in test_infra parses every planned command through this parser
    and asserts the run_id comes back identical.
    """
    ap = argparse.ArgumentParser(prog="sentalign", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data", type=Path, default=None,
                        help="build directory; defaults to the task's own")
    common.add_argument("--name", default="main")
    common.add_argument("--model", default="lfm-1.2b")
    common.add_argument("--seed", type=int, default=13)
    common.add_argument("--tau", type=float, default=0.2)
    common.add_argument("--noise", type=float, default=0.0)
    common.add_argument("--subsample", type=int, default=None)
    common.add_argument("--output-root", type=Path, default=Path("runs"))
    common.add_argument("--task", default="sentiment",
                        help="field bundle selecting the dataset and its eval sets")
    common.add_argument("--set", action="append", metavar="KEY=VALUE",
                        help="override any config field, repeatable")
    common.add_argument("--variant", default="",
                        help="run_id variant suffix, e.g. 'hard' for cspo-hard; keeps "
                             "ablation runs out of the base objective's directory")

    p = sub.add_parser("plan", help="print the run plan and its estimated cost")
    p.add_argument("--studies", nargs="+", choices=sorted(STUDIES))
    p.add_argument("--runs", type=Path, default=Path("runs"),
                   help="directory whose programme ledger re-costs the estimate from "
                        "measured throughput; without one the estimate is a priori")
    p.add_argument("--verbose", action="store_true")
    p.set_defaults(func=cmd_plan)

    p = sub.add_parser("build", help="build every dataset")
    p.add_argument("--out", type=Path, default=Path("data/build"))
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("train", parents=[common], help="train one arm")
    p.add_argument("--objective", default="sft", choices=list(ALL_OBJECTIVES))
    p.add_argument("--reference", type=Path, default=None,
                   help="SFT adapter to start from and anchor on")
    p.add_argument("--force", action="store_true", help="retrain a completed run")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("baselines", help="run the encoder and prompting reference points")
    p.add_argument("--kinds", nargs="+", default=["encoder", "prompt"],
                   choices=["encoder", "prompt"])
    p.add_argument("--models", nargs="+", help="models for the prompting baseline")
    p.add_argument("--data", type=Path, default=Path("data/build"))
    p.add_argument("--name", default="main")
    p.add_argument("--seed", type=int, default=None,
                   help="single seed for the encoder; default runs all five")
    p.add_argument("--output-root", type=Path, default=Path("runs"))
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_baselines)

    p = sub.add_parser("eval", help="evaluate a finished run")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--data", type=Path, default=Path("data/build"))
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("tables", help="aggregate runs into paper tables")
    p.add_argument("--runs", type=Path, default=Path("runs"))
    p.add_argument("--out", type=Path, default=Path("results"))
    p.set_defaults(func=cmd_tables)

    p = sub.add_parser("status", help="report on every run under a directory")
    p.add_argument("--runs", type=Path, default=Path("runs"))
    p.add_argument("--only", help="show only runs in this state")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("doctor", help="environment and hardware check")
    p.add_argument("--output-root", type=Path, default=Path("runs"))
    p.set_defaults(func=cmd_doctor)

    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
