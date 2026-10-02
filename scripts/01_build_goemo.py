"""Build the GoEmotions replication: a second sentiment corpus, same pipeline.

    python scripts/01_build_goemo.py --out data/build_goemo

GoEmotions releases every rater's emotion labels; ``sentalign.data.goemotions`` maps each
rating onto the main study's four sentiment labels, so each item carries a distribution
over raters as a DynaSent item does. Everything downstream of the items is the sentiment
pipeline unchanged: the same prompt and verbalizers, the same stratified subsampler, the
same preference, KTO, SFT and distributional builders, and the same leakage audit. Only
the corpus differs, which is what a replication should vary.

The layout mirrors the main build, including where each arm's data comes from: SFT and
the soft-target records at every subsample size, preference pairs and KTO completions
from the largest subsample, capped per run at training time exactly as on DynaSent.

Writes, under ``--out``:

    splits/manifest.json          held-out ids, sizes, bands, the integrity report
    sft/{train_n*,dev}.jsonl      instruction-tuning examples (majority label)
    cspo/{train_n*,dev}.jsonl     annotator-distribution records, every item
    pref/tau{T}/*.jsonl           preference pairs, margins over each item's own raters
    kto/{train,dev}.jsonl         unpaired completions
    eval/{ambig_eval,go_test,go_dev}.jsonl
    stats.json                    sizes, bands, the derivation rule and the source digests
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sentalign.data.goemotions import (AMBIG_FRACTION, AMBIG_SEED,     # noqa: E402
                                       AMBIGUOUS_AS, MIN_RATERS, build_splits,
                                       download, load_goemotions)
from sentalign.data.preferences import (build_kto_examples, build_pairs,  # noqa: E402
                                        build_sft_examples, flip_indices,
                                        write_jsonl)
from sentalign.data.splits import (stratified_subsample, stratify_report,  # noqa: E402
                                   subsample_report)
from sentalign.labels import LABEL_SPACES, NO_MAJORITY                # noqa: E402
from sentalign.train.cspo import build_cspo_records                   # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=Path, default=Path(".cache/goemotions"))
    ap.add_argument("--out", type=Path, default=Path("data/build_goemo"))
    ap.add_argument("--ambiguous-as", choices=AMBIGUOUS_AS, default="neutral",
                    help="a rating that marks only surprise, curiosity, confusion or "
                         "realization counts as neutral, or is dropped")
    ap.add_argument("--min-raters", type=int, default=MIN_RATERS)
    ap.add_argument("--tau", type=float, nargs="+", default=[0.2])
    ap.add_argument("--noise", type=float, nargs="+", default=[0.0])
    ap.add_argument("--max-pairs-per-item", type=int, default=3)
    ap.add_argument("--ambig-fraction", type=float, default=AMBIG_FRACTION)
    ap.add_argument("--holdout-seed", type=int, default=AMBIG_SEED)
    ap.add_argument("--skip-near-duplicates", action="store_true")
    ap.add_argument("--subsample", type=int, nargs="+", default=[8000, 32000],
                    help="nested training sizes; pairs and KTO come from the largest, as "
                         "in the main build")
    ap.add_argument("--subsample-seed", type=int, default=101)
    args = ap.parse_args(argv)

    space = LABEL_SPACES["ternary+mixed"]
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    stats: dict = {"corpus": "goemotions", "label_space": space.name,
                   "tau_values": args.tau, "noise_values": args.noise,
                   "ambiguous_as": args.ambiguous_as, "min_raters": args.min_raters}

    print("[1/6] GoEmotions release")
    stats["source_sha256"] = download(args.cache)
    by_split, report = load_goemotions(args.cache, ambiguous_as=args.ambiguous_as,
                                       min_raters=args.min_raters)
    stats["load"] = report.as_dict()
    print(f"      {report.ratings:,} ratings -> {report.items:,} items "
          f"({report.items_too_few_raters:,} with fewer than {args.min_raters} usable "
          f"raters, {report.items_duplicate_text:,} duplicate texts removed)")
    print(f"      raters per item {dict(report.raters_per_item)}; "
          f"{report.ratings_ambiguous_only:,} ratings marked only ambiguous emotions "
          f"({args.ambiguous_as})")

    print("[2/6] splits and leakage audit")
    bundle = build_splits(by_split, ambig_fraction=args.ambig_fraction,
                          seed=args.holdout_seed,
                          check_near=not args.skip_near_duplicates)
    if not bundle.integrity.clean:
        raise SystemExit("the build still leaks after removing contaminated items")
    bundle.save_manifest(out / "splits" / "manifest.json")
    stats["splits"] = bundle.sizes()
    stats["dropped_contaminated_eval_items"] = bundle.provenance.get(
        "dropped_by_split", {})
    stats["bands"] = {name: dict(Counter(i.agreement_band for i in items))
                      for name, items in [("train", bundle.train),
                                          ("ambig_eval", bundle.ambig_eval),
                                          *bundle.official.items()]}
    print(f"      sizes {bundle.sizes()}, contaminated eval items removed "
          f"{stats['dropped_contaminated_eval_items'] or 0}")

    print("[3/6] stratified subsamples")
    subsamples = {n: stratified_subsample(bundle.train, n, seed=args.subsample_seed)
                  for n in sorted(args.subsample)}
    sizes = sorted(subsamples)
    nested = all({i.text_id for i in subsamples[a]} <= {i.text_id for i in subsamples[b]}
                 for a, b in zip(sizes, sizes[1:]))
    if not nested:
        raise SystemExit("subsamples are not nested")
    for n, items in subsamples.items():
        if len(items) < n:
            print(f"      n={n}: the pool has only {len(items):,} items; using all of them")
    largest = subsamples[sizes[-1]]
    print(subsample_report(bundle.train, largest))
    stats["subsamples"] = {str(n): len(v) for n, v in subsamples.items()}

    print("[4/6] SFT and distributional records")
    dev = bundle.official["go_dev"]
    write_jsonl(build_sft_examples(dev, space), out / "sft" / "dev.jsonl")
    write_jsonl(build_cspo_records(dev, space), out / "cspo" / "dev.jsonl")
    stats["sft"], stats["cspo"] = {"by_size": {}}, {"by_size": {}}
    for n, items in subsamples.items():
        sft = build_sft_examples(items, space)
        write_jsonl(sft, out / "sft" / f"train_n{n}.jsonl")
        cspo = build_cspo_records(items, space)
        write_jsonl(cspo, out / "cspo" / f"train_n{n}.jsonl")
        stats["sft"]["by_size"][str(n)] = {"train": len(sft),
                                           "no_majority_dropped": len(items) - len(sft)}
        stats["cspo"]["by_size"][str(n)] = {"train": len(cspo)}
        print(f"      n={n:>6}: SFT {len(sft):>6} (drops {len(items) - len(sft)} "
              f"no-majority), distributional {len(cspo):>6}")

    print("[5/6] preference pairs and KTO")
    stats["preferences"] = {}
    for tau in args.tau:
        # Per-item denominators: items here have three, four or five raters.
        pairs, pair_stats = build_pairs(largest, space, tau=tau,
                                        max_pairs_per_item=args.max_pairs_per_item,
                                        seed=0, n_annotators=None)
        dev_pairs, _ = build_pairs(dev, space, tau=tau,
                                   max_pairs_per_item=args.max_pairs_per_item, seed=0,
                                   n_annotators=None)
        directory = out / "pref" / f"tau{tau}"
        write_jsonl((p.as_record() for p in pairs), directory / "train.jsonl")
        write_jsonl((p.as_record() for p in dev_pairs), directory / "dev.jsonl")
        entry = {"n_pairs": pair_stats.n_pairs, "n_items_used": pair_stats.n_items_used,
                 "margin_hist": {str(k): v for k, v in sorted(pair_stats.margin_hist.items())},
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
        print(f"      tau={tau}: {pair_stats.n_pairs:,} pairs from "
              f"{pair_stats.n_items_used:,} items")

    for name, items in (("train", largest), ("dev", dev)):
        examples = build_kto_examples(items, space)
        write_jsonl(({"prompt": e.prompt, "completion": e.completion,
                      "label": e.desirable, "text_id": e.text_id, "group": e.group}
                     for e in examples), out / "kto" / f"{name}.jsonl")
        stats.setdefault("kto", {})[name] = {
            "n": len(examples), "desirable": sum(1 for e in examples if e.desirable)}
    print(f"      KTO {stats['kto']['train']['n']:,} completions, "
          f"{stats['kto']['train']['desirable']:,} desirable")

    print("[6/6] evaluation sets")
    written = {}
    for name, items in (("ambig_eval", bundle.ambig_eval),
                        ("go_test", bundle.official["go_test"]), ("go_dev", dev)):
        written[name] = write_jsonl((i.to_dict() for i in items),
                                    out / "eval" / f"{name}.jsonl")
    stats["eval_sets"] = written
    stats["no_majority"] = {name: sum(1 for i in items if i.gold == NO_MAJORITY)
                            for name, items in (("ambig_eval", bundle.ambig_eval),
                                                ("go_test", bundle.official["go_test"]))}
    print(f"      {written}; no-majority {stats['no_majority']}")
    print("\nambiguity-stratified held-out set:")
    print(stratify_report(bundle.ambig_eval))

    (out / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(f"\nwrote {out}/stats.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
