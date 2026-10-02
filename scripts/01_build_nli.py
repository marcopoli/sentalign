"""Build the NLI study: coarse five-annotator training, dense 100-annotator evaluation.

    python scripts/01_build_nli.py --out data/build_nli

The sentiment study cannot separate "the model learned the annotator distribution" from
"the model has the right overall confidence", because a five-annotator target is quantised
to multiples of 0.2 and one temperature can set an entropy level. ChaosNLI re-annotates
SNLI/MNLI items with 100 annotators, so here the target has shape a scalar cannot fake.

Everything downstream of the items is the sentiment pipeline unchanged: the same
subsampler, the same preference builder, the same SFT and distributional record builders.
Only the source and the label space differ, which is the point: a difference in results
cannot then be attributed to a difference in machinery.

Writes, under ``--out``:

    sft/{train_n*,dev}.jsonl      instruction-tuning examples
    cspo/{train_n*,dev}.jsonl     annotator-distribution records
    pref/tau{T}/*.jsonl           preference pairs and the noise ladder
    kto/{train,dev}.jsonl         unpaired completions
    eval/ambig_eval.jsonl         ChaosNLI, both sources, the dense target
    eval/chaos_{snli,mnli}.jsonl  the same split by source
    eval/nli_dev.jsonl            held-out coarse items, for temperature fitting
    stats.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sentalign.data.chaosnli import load_nli                          # noqa: E402
from sentalign.data.preferences import (build_kto_examples, build_pairs,  # noqa: E402
                                        build_sft_examples, flip_indices,
                                        write_jsonl)
from sentalign.data.splits import stratified_subsample                # noqa: E402
from sentalign.labels import LABEL_SPACES, NO_MAJORITY                # noqa: E402
from sentalign.train.cspo import build_cspo_records                   # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=Path, default=Path(".cache/nli"))
    ap.add_argument("--out", type=Path, default=Path("data/build_nli"))
    ap.add_argument("--label-space", default="nli3", choices=sorted(LABEL_SPACES))
    ap.add_argument("--tau", type=float, nargs="+", default=[0.2])
    ap.add_argument("--noise", type=float, nargs="+", default=[0.0])
    ap.add_argument("--max-pairs-per-item", type=int, default=3)
    ap.add_argument("--dev-size", type=int, default=2000)
    ap.add_argument("--subsample", type=int, nargs="+", default=[8000])
    ap.add_argument("--subsample-seed", type=int, default=101)
    ap.add_argument("--split-seed", type=int, default=20260819)
    ap.add_argument("--train-target", choices=("dense", "coarse"), default="dense",
                    help="with --chaos-only, whether training targets use the 100 "
                         "annotators or the original 5 for the SAME items")
    ap.add_argument("--chaos-only", action="store_true",
                    help="skip SNLI/MNLI and split ChaosNLI itself, so training targets "
                         "are dense too; use when only the ChaosNLI archive is available")
    args = ap.parse_args()

    space = LABEL_SPACES[args.label_space]
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    stats: dict = {"label_space": args.label_space, "tau_values": args.tau,
                   "noise_values": args.noise}

    print("[1/5] loading NLI sources")
    splits = load_nli(args.cache, chaos_only=args.chaos_only,
                      train_target=args.train_target)
    pool = splits["train"]
    dense, coarse = splits["chaos_dense"], splits["chaos_coarse"]
    stats["regime"] = splits["regime"]
    print(f"      regime: {splits['regime']}")
    print(f"      train pool {len(pool):,}")
    print(f"      paired eval {len(dense):,} items, scored against a dense "
          f"(100-annotator) and a coarse (5-annotator) target")

    # The paired sets must cover the same items in the same order, or the contrast is
    # between two different samples rather than between two targets.
    if [i.text_id for i in dense] != [i.text_id for i in coarse]:
        raise SystemExit("dense and coarse evaluation sets are not aligned by id")
    overlap = {i.text_id for i in pool} & {i.text_id for i in dense}
    if overlap:
        raise SystemExit(f"{len(overlap)} evaluation items are in the training pool")

    for name, items in (("dense", dense), ("coarse", coarse)):
        n_nm = sum(1 for i in items if i.gold == NO_MAJORITY)
        print(f"      {name:6s} target: {n_nm:,} no-majority "
              f"({100 * n_nm / max(len(items), 1):.1f}%)")
        stats[f"no_majority_{name}"] = n_nm

    print("[2/5] train/dev split and subsamples")
    shuffled = list(pool)
    random.Random(args.split_seed).shuffle(shuffled)
    # A fixed dev size silently emptied the training split when the pool was small:
    # --chaos-only leaves about 800 items and the default 2000 consumed all of them,
    # after which the trainer died on next(iter(train_dataset)) with StopIteration.
    dev_size = min(args.dev_size, max(1, len(shuffled) // 5))
    if dev_size != args.dev_size:
        print(f"      dev size reduced {args.dev_size} -> {dev_size} "
              f"(pool is only {len(shuffled):,} items)")
    dev, train = shuffled[:dev_size], shuffled[dev_size:]
    if not train:
        raise SystemExit(f"training split is empty: pool {len(shuffled)}, dev {dev_size}")
    subsamples = {n: stratified_subsample(train, n, seed=args.subsample_seed)
                  for n in sorted(args.subsample)}
    for n, items in subsamples.items():
        if not items:
            raise SystemExit(f"subsample n={n} is empty from a train split of "
                             f"{len(train)}; lower --subsample or --dev-size")
    print(f"      dev {len(dev):,}, train {len(train):,}, "
          f"subsamples {{{', '.join(f'{n}: {len(v):,}' for n, v in subsamples.items())}}}")
    stats["dev"] = len(dev)
    stats["subsamples"] = {str(n): len(v) for n, v in subsamples.items()}

    print("[3/5] SFT and distributional records")
    write_jsonl(build_sft_examples(dev, space), out / "sft" / "dev.jsonl")
    write_jsonl(build_cspo_records(dev, space), out / "cspo" / "dev.jsonl")
    stats["sft"], stats["cspo"] = {"by_size": {}}, {"by_size": {}}
    for n, items in subsamples.items():
        sft = build_sft_examples(items, space)
        write_jsonl(sft, out / "sft" / f"train_n{n}.jsonl")
        cspo = build_cspo_records(items, space)
        write_jsonl(cspo, out / "cspo" / f"train_n{n}.jsonl")
        stats["sft"]["by_size"][str(n)] = len(sft)
        stats["cspo"]["by_size"][str(n)] = len(cspo)
        print(f"      n={n:>6}: SFT {len(sft):>6} "
              f"(drops {len(items) - len(sft)} no-majority), distributional {len(cspo):>6}")

    print("[4/5] preference pairs and KTO")
    pair_source = subsamples[max(subsamples)]
    stats["preferences"] = {}
    for tau in args.tau:
        pairs, pair_stats = build_pairs(pair_source, space, tau=tau,
                                        max_pairs_per_item=args.max_pairs_per_item, seed=0)
        dev_pairs, _ = build_pairs(dev, space, tau=tau,
                                   max_pairs_per_item=args.max_pairs_per_item, seed=0)
        directory = out / "pref" / f"tau{tau}"
        write_jsonl((p.as_record() for p in pairs), directory / "train.jsonl")
        write_jsonl((p.as_record() for p in dev_pairs), directory / "dev.jsonl")
        entry = {"n_pairs": pair_stats.n_pairs, "n_items_used": pair_stats.n_items_used,
                 "noise": {}}
        for eps in args.noise:
            idx = flip_indices(pairs, eps, seed=7)
            (directory / "noise").mkdir(parents=True, exist_ok=True)
            (directory / "noise" / f"eps{eps}.json").write_text(json.dumps(
                {"epsilon": eps, "seed": 7, "n_pairs": len(pairs), "n_flipped": len(idx),
                 "realised_epsilon": round(len(idx) / max(len(pairs), 1), 5),
                 "flip_indices": idx}))
            entry["noise"][str(eps)] = {"n_flipped": len(idx)}
        stats["preferences"][str(tau)] = entry
        print(f"      tau={tau}: {pair_stats.n_pairs:,} pairs "
              f"from {pair_stats.n_items_used:,} items")

    for name, items in (("train", pair_source), ("dev", dev)):
        examples = build_kto_examples(items, space)
        write_jsonl(({"prompt": e.prompt, "completion": e.completion,
                      "label": e.desirable, "text_id": e.text_id, "group": e.group}
                     for e in examples), out / "kto" / f"{name}.jsonl")
        stats.setdefault("kto", {})[name] = len(examples)

    print("[5/5] evaluation sets")
    written = {}
    for name, items in (("ambig_eval", dense), ("chaos_coarse", coarse),
                        ("nli_dev", dev)):
        written[name] = write_jsonl((i.to_dict() for i in items),
                                    out / "eval" / f"{name}.jsonl")
    print(f"      {written}")
    stats["eval_sets"] = written

    (out / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(f"\nwrote {out}/stats.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
