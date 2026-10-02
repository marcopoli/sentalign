"""Build every dataset the experiments need, and refuse to emit a leaking one.

    python scripts/01_build_data.py --out data/build --tau 0.2 --noise 0 0.1 0.2 0.4

Writes, under ``--out``:

    splits/manifest.json          item ids and the integrity report
    sft/{train,dev}.jsonl         instruction-tuning examples
    pref/tau{T}/eps{E}/*.jsonl    HDP-Sent preference pairs, per noise level
    kto/{train,dev}.jsonl         unpaired desirable/undesirable completions
    eval/*.jsonl                  every evaluation set, incl. the ambiguity strata
    stats.json                    the numbers that go into Table 1 of the paper
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sentalign.data.dynasent import counterfactual_pairs           # noqa: E402
from sentalign.data.preferences import (build_kto_examples, build_pairs,   # noqa: E402
                                        build_sft_examples, flip_indices,
                                        write_jsonl)
from sentalign.data.splits import (build_splits, class_weights,     # noqa: E402
                                   drop_leaking_items, stratified_subsample,
                                   stratify_report, subsample_report)
from sentalign.train.cspo import build_cspo_records                   # noqa: E402
from sentalign.labels import LABEL_SPACES                            # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=Path, default=Path(".cache/dynasent"))
    ap.add_argument("--out", type=Path, default=Path("data/build"))
    ap.add_argument("--label-space", default="ternary+mixed", choices=sorted(LABEL_SPACES))
    ap.add_argument("--tau", type=float, nargs="+", default=[0.2, 0.4, 0.6],
                    help="minimum human preference margins to build")
    ap.add_argument("--noise", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.4],
                    help="preference-noise ladder (epsilon)")
    ap.add_argument("--max-pairs-per-item", type=int, default=3)
    ap.add_argument("--ambig-fraction", type=float, default=0.06)
    ap.add_argument("--holdout-seed", type=int, default=20260819)
    ap.add_argument("--skip-near-duplicates", action="store_true",
                    help="exact-match auditing only (much faster; near-dup pass is O(n))")
    ap.add_argument("--subsample", type=int, nargs="+", default=[3000, 8000, 16000, 32000],
                    help="training-set sizes to emit; nested, so the smallest is a "
                         "subset of the next")
    ap.add_argument("--subsample-seed", type=int, default=101)
    args = ap.parse_args()

    space = LABEL_SPACES[args.label_space]
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    stats: dict = {"label_space": args.label_space, "tau_values": args.tau,
                   "noise_values": args.noise}

    # ---------------------------------------------------------------- splits
    print("[1/5] building splits (split -> audit -> balance, never the reverse)")
    bundle = build_splits(args.cache, ambig_fraction=args.ambig_fraction,
                          seed=args.holdout_seed,
                          check_near_duplicates=not args.skip_near_duplicates)
    if not bundle.integrity.clean:
        n_before = sum(bundle.sizes().values())
        bundle = drop_leaking_items(bundle)
        print(f"      removed {bundle.provenance['dropped_contaminated_eval_items']} "
              f"contaminated evaluation items "
              f"({n_before - sum(bundle.sizes().values())} rows)")
    bundle.save_manifest(out / "splits" / "manifest.json")
    print(f"      sizes: {bundle.sizes()}")
    print(f"      integrity: {'clean' if bundle.integrity.clean else 'LEAKING'}")
    stats["splits"] = bundle.sizes()
    stats["ambig_bands"] = dict(Counter(i.agreement_band for i in bundle.ambig_eval))

    # --------------------------------------------------------- subsampling
    print("[2/6] stratified training subsamples")
    subsamples = {}
    for n in sorted(args.subsample):
        subsamples[n] = stratified_subsample(bundle.train, n, seed=args.subsample_seed)
    largest = subsamples[max(subsamples)]
    print(subsample_report(bundle.train, largest))
    nested = all(
        {i.text_id for i in subsamples[a]} <= {i.text_id for i in subsamples[b]}
        for a, b in zip(sorted(subsamples), sorted(subsamples)[1:]))
    print(f"      sizes {sorted(subsamples)} nested: {nested}")
    stats["subsamples"] = {str(n): len(v) for n, v in subsamples.items()}
    stats["subsample_nested"] = nested
    stats["subsample_seed"] = args.subsample_seed

    # ------------------------------------------------------------------ SFT
    print("[3/6] SFT and distributional examples")
    sft_dev = build_sft_examples(bundle.official["r1_dev"], space)
    write_jsonl(sft_dev, out / "sft" / "dev.jsonl")
    cspo_dev = build_cspo_records(bundle.official["r1_dev"], space)
    write_jsonl(cspo_dev, out / "cspo" / "dev.jsonl")

    stats["sft"] = {"dev": len(sft_dev), "by_size": {}}
    stats["cspo"] = {"dev": len(cspo_dev), "by_size": {}}
    for n, items in subsamples.items():
        sft_train = build_sft_examples(items, space)
        write_jsonl(sft_train, out / "sft" / f"train_n{n}.jsonl")
        # The distributional arms keep every item, including those with no majority
        # label, which is the supervision cross-entropy structurally cannot use.
        cspo_train = build_cspo_records(items, space)
        write_jsonl(cspo_train, out / "cspo" / f"train_n{n}.jsonl")
        dropped = len(items) - len(sft_train)
        stats["sft"]["by_size"][str(n)] = {"train": len(sft_train),
                                           "no_majority_dropped": dropped}
        stats["cspo"]["by_size"][str(n)] = {"train": len(cspo_train)}
        print(f"      n={n:>6}: SFT {len(sft_train):>6} "
              f"(drops {dropped:>5} no-majority = {100*dropped/len(items):.1f}%), "
              f"distributional {len(cspo_train):>6} (drops none)")
    stats["class_weights"] = class_weights(bundle.train, space)

    # ---------------------------------------------------------- preferences
    print("[4/6] HDP-Sent preference pairs")
    stats["preferences"] = {}
    pair_source = subsamples[max(subsamples)]
    for tau in args.tau:
        pairs, pair_stats = build_pairs(pair_source, space, tau=tau,
                                        max_pairs_per_item=args.max_pairs_per_item, seed=0)
        dev_pairs, _ = build_pairs(bundle.official["r1_dev"], space, tau=tau,
                                   max_pairs_per_item=args.max_pairs_per_item, seed=0)
        print(f"  tau={tau}")
        print(pair_stats.summary())
        entry = {
            "n_pairs": pair_stats.n_pairs,
            "n_items_used": pair_stats.n_items_used,
            "margin_hist": {str(k): v for k, v in sorted(pair_stats.margin_hist.items())},
            "band_hist": dict(pair_stats.band_hist),
            "n_groups": len(pair_stats.group_hist),
            "noise": {},
        }
        directory = out / "pref" / f"tau{tau}"
        write_jsonl((p.as_record() for p in pairs), directory / "train.jsonl")
        write_jsonl((p.as_record() for p in dev_pairs), directory / "dev.jsonl")
        # The ladder is stored as flip-index sets over the base file, not as copies.
        for eps in args.noise:
            idx = flip_indices(pairs, eps, seed=7)
            (directory / "noise").mkdir(parents=True, exist_ok=True)
            (directory / "noise" / f"eps{eps}.json").write_text(
                json.dumps({"epsilon": eps, "seed": 7, "n_pairs": len(pairs),
                            "n_flipped": len(idx),
                            "realised_epsilon": round(len(idx) / max(len(pairs), 1), 5),
                            "flip_indices": idx}))
            entry["noise"][str(eps)] = {
                "n_flipped": len(idx),
                "realised_epsilon": round(len(idx) / max(len(pairs), 1), 5)}
        stats["preferences"][str(tau)] = entry

    # ------------------------------------------------------------------ KTO
    print("[5/6] KTO unpaired examples")
    kto_train = build_kto_examples(pair_source, space)
    kto_dev = build_kto_examples(bundle.official["r1_dev"], space)
    write_jsonl(({"prompt": e.prompt, "completion": e.completion, "label": e.desirable,
                  "text_id": e.text_id, "group": e.group} for e in kto_train),
                out / "kto" / "train.jsonl")
    write_jsonl(({"prompt": e.prompt, "completion": e.completion, "label": e.desirable,
                  "text_id": e.text_id, "group": e.group} for e in kto_dev),
                out / "kto" / "dev.jsonl")
    n_des = sum(1 for e in kto_train if e.desirable)
    print(f"      {len(kto_train):,} completions: {n_des:,} desirable / "
          f"{len(kto_train) - n_des:,} undesirable")
    stats["kto"] = {"n": len(kto_train), "desirable": n_des,
                    "undesirable": len(kto_train) - n_des}

    # ------------------------------------------------------------ eval sets
    print("[6/6] evaluation sets")
    eval_dir = out / "eval"
    written = {}
    for name, items in [("ambig_eval", bundle.ambig_eval), *bundle.official.items()]:
        written[name] = write_jsonl((i.to_dict() for i in items), eval_dir / f"{name}.jsonl")
    written["dynasent_rewrites"] = write_jsonl(
        iter(counterfactual_pairs(args.cache)), eval_dir / "dynasent_rewrites.jsonl")
    print(f"      {written}")
    stats["eval_sets"] = written

    print("\nambiguity-stratified held-out set:")
    print(stratify_report(bundle.ambig_eval))

    (out / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(f"\nwrote {out}/stats.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
