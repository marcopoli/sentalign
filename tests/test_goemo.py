"""The GoEmotions replication: the derivation, the build, the plan and the analysis.

A second corpus is only a replication if the one thing that changes is the corpus. These
tests pin the places where GoEmotions differs from DynaSent, and where a silent difference
would change what the arms learn: how a rater's emotions become a sentiment label, the
varying number of raters behind each target, duplicates across the official splits, and
the analysis families declared before the runs.
"""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest

from sentalign.data import goemotions as go
from sentalign.data.dynasent import Item
from sentalign.data.preferences import build_pairs
from sentalign.labels import MIXED, NEGATIVE, NEUTRAL, NO_MAJORITY, POSITIVE, TERNARY_PLUS_MIXED

ROOT = Path(__file__).resolve().parents[1]


def _script(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------------- the derivation

@pytest.mark.parametrize("marked, expected", [
    (["joy"], POSITIVE),
    (["joy", "curiosity"], POSITIVE),
    (["anger", "confusion"], NEGATIVE),
    (["joy", "anger"], MIXED),
    (["joy", "anger", "surprise"], MIXED),
    (["neutral"], NEUTRAL),
    (["surprise"], NEUTRAL),
    ([], None),
])
def test_each_rating_maps_to_one_sentiment_label(marked, expected):
    assert go.rater_sentiment(marked) == expected


def test_a_rating_of_only_ambiguous_emotions_can_be_dropped_instead():
    """The one derivation choice a reader could make differently is a parameter."""
    assert go.rater_sentiment(["curiosity", "realization"], ambiguous_as="drop") is None
    assert go.rater_sentiment(["curiosity", "joy"], ambiguous_as="drop") == POSITIVE


def test_an_unknown_emotion_is_refused_not_mapped_to_neutral():
    with pytest.raises(ValueError, match="unknown emotion"):
        go.rater_sentiment(["happiness"])


def test_the_coded_grouping_is_checked_against_the_released_one(tmp_path):
    released = {g: list(v) for g, v in go.SENTIMENT_GROUPS.items()}
    (tmp_path / "m.json").write_text(json.dumps(released))
    go.check_mapping(tmp_path / "m.json")
    released["positive"].remove("relief")
    released["ambiguous"].append("relief")
    (tmp_path / "m.json").write_text(json.dumps(released))
    with pytest.raises(ValueError, match="does not match"):
        go.check_mapping(tmp_path / "m.json")


def test_majority_is_more_than_half_at_three_four_and_five_raters():
    assert go.gold_from_votes({POSITIVE: 2, NEGATIVE: 1}) == POSITIVE
    assert go.gold_from_votes({POSITIVE: 1, NEGATIVE: 1, NEUTRAL: 1}) == NO_MAJORITY
    assert go.gold_from_votes({POSITIVE: 2, NEGATIVE: 2}) == NO_MAJORITY
    assert go.gold_from_votes({POSITIVE: 3, NEGATIVE: 2}) == POSITIVE
    assert go.band_of({POSITIVE: 3, NEGATIVE: 0}, POSITIVE) == "unanimous"
    assert go.band_of({POSITIVE: 2, NEGATIVE: 1}, POSITIVE) == "majority"


def _row(item_id, text, *emotions, unclear=False):
    row = {"id": item_id, "text": text, "example_very_unclear": str(unclear)}
    row.update({e: "1" if e in emotions else "0" for e in go.EMOTIONS})
    return row


def test_items_count_raters_not_emotions_and_drop_unusable_ratings():
    rows = [_row("a", "t", "joy", "love"), _row("a", "t", "anger"), _row("a", "t", "joy"),
            _row("a", "t", "anger", unclear=True),
            _row("b", "u", "joy"), _row("b", "u", "joy")]
    report = go.LoadReport()
    items = go.items_from_ratings(rows, {"a": "test"}, report=report)
    assert [i.text_id for i in items] == ["a"], "b has two usable raters, below the minimum"
    item = items[0]
    assert item.votes == {POSITIVE: 2, NEGATIVE: 1, NEUTRAL: 0, MIXED: 0}
    assert item.gold == POSITIVE and item.meta["n_annotators"] == 3
    assert item.round == "go_test"
    assert report.ratings_unusable == 1 and report.items_too_few_raters == 1, \
        "a rating flagged very unclear is dropped even when it marks an emotion"


def test_one_item_with_two_texts_is_an_error():
    with pytest.raises(ValueError, match="two different texts"):
        go.items_from_ratings([_row("a", "t", "joy"), _row("a", "u", "joy")], {})


def _item(item_id, text, split, **votes):
    counts = {POSITIVE: 0, NEGATIVE: 0, NEUTRAL: 0, MIXED: 0} | votes
    gold = go.gold_from_votes(counts)
    return Item(text_id=item_id, text=text, votes=counts, gold=gold, round=split,
                source="goemotions", meta={"band": go.band_of(counts, gold)})


def test_a_duplicated_text_survives_in_the_test_split_not_in_training():
    """Keeping the training copy would delete a test item; keeping both would score a text
    the model was trained on."""
    items = [_item("x1", "Thank you!", "go_train", positive=3),
             _item("x2", "thank you", "go_test", positive=3),
             _item("x3", "Other text", "go_train", positive=3)]
    kept = go.dedupe_by_text(items)
    assert {i.text_id for i in kept} == {"x2", "x3"}


# ------------------------------------------------------------------ the pair margins

def test_a_one_vote_gap_among_three_raters_is_a_margin_of_one_third():
    """Dividing by five, as the DynaSent default does, would record 0.2."""
    item = _item("a", "t", "go_train", positive=2, negative=1)
    pairs, _ = build_pairs([item], TERNARY_PLUS_MIXED, tau=0.2, n_annotators=None)
    by_labels = {(p.chosen_label, p.rejected_label): p.margin for p in pairs}
    assert by_labels[(POSITIVE, NEGATIVE)] == pytest.approx(1 / 3)
    assert by_labels[(POSITIVE, NEUTRAL)] == pytest.approx(2 / 3)
    assert (NEUTRAL, MIXED) not in by_labels, "no vote gap, no preference"


def test_per_item_margins_leave_a_five_rater_corpus_unchanged():
    """The DynaSent builds must be byte-identical whichever denominator rule is used."""
    items = [_item(str(i), f"t{i}", "go_train", **v) for i, v in enumerate([
        {"positive": 3, "negative": 2}, {"neutral": 5}, {"mixed": 2, "positive": 2,
                                                          "negative": 1}])]
    fixed, _ = build_pairs(items, TERNARY_PLUS_MIXED, tau=0.2)
    per_item, _ = build_pairs(items, TERNARY_PLUS_MIXED, tau=0.2, n_annotators=None)
    assert [p.as_record() for p in fixed] == [p.as_record() for p in per_item]


# ------------------------------------------------------------------ splits and audit

def test_an_eval_set_outside_the_priority_is_refused_rather_than_unaudited():
    from sentalign.data.integrity import IntegrityReport
    from sentalign.data.splits import SplitBundle, drop_leaking_items

    report = IntegrityReport(overlaps=[], duplicate_rates={}, split_sizes={})
    bundle = SplitBundle([], [], {"go_test": []}, report, {})
    with pytest.raises(ValueError, match="escape the audit"):
        drop_leaking_items(bundle, check_near=False)


def test_the_held_out_set_is_disjoint_and_a_leaking_eval_item_leaves_eval_not_train():
    pool = [_item(f"t{i}", f"training sentence number {i} about food", "go_train",
                  positive=3 - i % 2, negative=i % 2) for i in range(300)]
    test = [_item("leak", "training sentence number 7 about food", "go_test", positive=3),
            _item("clean", "a sentence nobody trained on", "go_test", positive=3)]
    bundle = go.build_splits({"go_train": pool, "go_unsplit": [], "go_test": test,
                              "go_dev": []}, check_near=False)
    train_ids = {i.text_id for i in bundle.train}
    assert not train_ids & {i.text_id for i in bundle.ambig_eval}
    assert {i.text_id for i in bundle.official["go_test"]} == {"clean"}
    assert "t7" in train_ids or "t7" in {i.text_id for i in bundle.ambig_eval}
    assert len(bundle.train) + len(bundle.ambig_eval) == len(pool)


# --------------------------------------------------------------- the build end to end

def _fake_release(cache: Path, n: int = 400) -> None:
    cache.mkdir(parents=True)
    header = ["text", "id", "author", "subreddit", "link_id", "parent_id", "created_utc",
              "rater_id", "example_very_unclear", *go.EMOTIONS]
    choices = [["joy"], ["anger"], ["neutral"], ["joy", "anger"], ["surprise"]]
    splits = {s: [] for s in go.SPLIT_FILES}
    rows = []
    for i in range(n):
        item_id = f"e{i:04d}"
        raters = 5 if i % 4 == 0 else 3
        for r in range(raters):
            marked = choices[(i + (r if i % 3 else 0)) % len(choices)]
            rows.append({"text": f"comment {i} says something {i % 17}", "id": item_id,
                         "author": "a", "subreddit": "s", "link_id": "l",
                         "parent_id": "p", "created_utc": "0", "rater_id": str(r),
                         "example_very_unclear": "False",
                         **{e: int(e in marked) for e in go.EMOTIONS}})
        split = ("test", "dev", "train", "train", "train", None)[i % 6]
        if split:
            splits[split].append(item_id)
    for k, name in enumerate(go.RAW_FILES):
        with (cache / name).open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=header)
            writer.writeheader()
            writer.writerows(rows[k::3])
    for split, name in go.SPLIT_FILES.items():
        (cache / name).write_text("".join(f"text\t0\t{i}\n" for i in splits[split]))
    (cache / go.MAPPING_FILE).write_text(json.dumps(
        {g: list(v) for g, v in go.SENTIMENT_GROUPS.items()}))


def test_the_build_writes_what_the_training_path_reads(tmp_path):
    """The layout the driver reads, and the regulariser join, on a small fake release."""
    import sentalign.cli as cli
    from sentalign.config import ExperimentConfig, apply_task

    cache, out = tmp_path / "cache", tmp_path / "build"
    _fake_release(cache)
    build = _script("build_goemo", "01_build_goemo.py")
    assert build.main(["--cache", str(cache), "--out", str(out), "--subsample", "80", "160",
                       "--skip-near-duplicates"]) == 0
    for rel in ("sft/train_n80.jsonl", "sft/dev.jsonl", "cspo/train_n160.jsonl",
                "pref/tau0.2/train.jsonl", "pref/tau0.2/noise/eps0.0.json",
                "kto/train.jsonl", "eval/ambig_eval.jsonl", "eval/go_test.jsonl",
                "eval/go_dev.jsonl", "splits/manifest.json"):
        assert (out / rel).exists(), rel
    stats = json.loads((out / "stats.json").read_text())
    assert stats["ambiguous_as"] == "neutral" and "source_sha256" in stats

    cfg = ExperimentConfig()
    apply_task(cfg, "goemo")
    cfg.data.build_dir, cfg.data.train_subsample, cfg.data.max_train_pairs = out, 80, 80
    for objective in ("kto", "rdpo"):
        cfg.train.objective, cfg.train.pref_distributional_lambda = objective, 1.0
        data = cli._load_training_data(cfg)
        records = data["kto_examples" if objective == "kto" else "pref_pairs"]
        assert records and all("p_human" in r for r in records)


# ------------------------------------------------------------------- task and plan

def test_the_goemo_task_reads_its_own_build_and_sets():
    from sentalign.config import ExperimentConfig, apply_task

    cfg = ExperimentConfig()
    apply_task(cfg, "goemo")
    assert cfg.data.build_dir == Path("data/build_goemo")
    assert cfg.data.label_space == "ternary+mixed", "the same labels as DynaSent"
    assert cfg.eval.eval_sets == ("ambig_eval", "go_test")
    assert cfg.eval.temperature_fit_split == "go_dev"


def test_the_plan_declares_five_arms_per_seed_and_anchors_each_on_its_own_sft():
    from sentalign.cli import _default_reference
    from sentalign.config import SEEDS
    from sentalign.plan import full_plan

    runs = full_plan(("goemo",))
    assert len(runs) == 2 * len(SEEDS) * 5
    ids = [r.to_config().run_id for r in runs]
    assert all(i.startswith("goemo__") for i in ids)
    regularised = [r.to_config() for r in runs if r.variant == "dreg1.0"]
    assert regularised and all(c.train.pref_distributional_lambda == 1.0 for c in regularised)
    for run in runs:
        cfg = run.to_config()
        if cfg.train.objective != "sft":
            assert _default_reference(cfg).name in ids, "anchors on a goemo SFT run"
    assert not set(ids) & {r.to_config().run_id for r in full_plan(("main",))}


def test_the_grid_runner_leaves_the_goemo_build_directory_alone(tmp_path):
    grid = _script("run_grid", "02_run_grid.py")
    from sentalign.plan import full_plan

    run = full_plan(("goemo",))[0]
    _, cfg = grid.run_dir_for(run, tmp_path, Path("data/build"))
    assert cfg.data.build_dir == Path("data/build_goemo")
    assert "--data" not in grid.build_command(run, cfg, Path("data/build"), tmp_path)


# ------------------------------------------------------------------------ analysis

def _write_run(runs: Path, arm: str, seed: int, model="lfm-1.2b", study="goemo",
               acc=0.6, spread=0.5):
    spread += (seed % 7) * 0.01            # seeds differ, so paired tests have variance
    run = runs / f"{study}__{model}__{arm}__eps0.0__tau0.2__n8000__s{seed}"
    (run / "eval").mkdir(parents=True)
    rows = []
    for i in range(100):
        correct = i < acc * 100
        conf = 0.5 + spread * (0.5 if correct else 0.1) + 0.001 * i
        rows.append(json.dumps({"probs": [min(conf, 0.99), 1 - min(conf, 0.99)],
                                "correct": correct}))
    for name in ("ambig_eval", "go_test"):
        (run / "eval" / f"predictions_{name}.jsonl").write_text("\n".join(rows))


def test_every_run_of_the_study_is_read_not_only_the_first(tmp_path):
    """The loader once reused its study-name argument as the loop variable over sets, so
    every directory after the first was compared against a set name and skipped."""
    nli = _script("nli_robustness", "12_nli_robustness.py")
    for seed in (13, 21, 34):
        _write_run(tmp_path, "sft", seed)
    _write_run(tmp_path, "kto", 13)
    scores = nli.load(tmp_path, "lfm-1.2b", sets=("ambig_eval", "go_test"), study="goemo")
    assert set(scores) == {"sft", "kto"} and set(scores["sft"]) == {13, 21, 34}


def test_the_declared_families_span_both_models_and_refuse_to_shrink(tmp_path):
    rep = _script("goemo_replication", "16_goemo_replication.py")
    for model in rep.MODELS:
        for arm in rep.ARMS:
            for seed in (13, 21, 34, 55, 89):
                _write_run(tmp_path, arm, seed, model=model,
                           spread=0.1 if arm in ("kto", "rdpo") else 0.5)
    scores = rep.collect(tmp_path)
    assert rep.shortfall(scores) == []
    result = rep.panels(scores)
    n = len(rep.MODELS) * 2 * len(rep.METRICS)
    assert len(result["A"]) == n and len(result["B"]) == n, "one family per panel"
    assert all(r["p_holm"] >= r["p"] for r in result["A"] + result["B"])
    for panel in result.values():
        smallest = min(panel, key=lambda r: r["p"])
        assert smallest["p_holm"] == pytest.approx(min(1.0, n * smallest["p"])), \
            "the first Holm step multiplies by the size of the whole family"
    assert f"Holm over {n}" in rep.table(result)

    import shutil
    shutil.rmtree(tmp_path / "goemo__qwen-2b__rdpo-dreg1.0__eps0.0__tau0.2__n8000__s89")
    with pytest.raises(SystemExit, match="incomplete"):
        rep.main(["--runs", str(tmp_path), "--out", str(tmp_path / "out")])
    assert rep.main(["--runs", str(tmp_path), "--out", str(tmp_path / "out"),
                     "--interim"]) == 0
    assert "INTERIM" in (tmp_path / "out" / "table_goemo.tex").read_text()


def test_a_difference_of_exactly_zero_on_every_seed_is_not_significant():
    """The paired t returned p = 0 when all five seed differences were zero, so two arms
    that never differ would be starred as the strongest result in their panel."""
    selective = _script("selective", "05_selective.py")
    assert selective.paired_t([0.0] * 5)[1] == 1.0
    assert selective.paired_t([0.01] * 5)[1] == 0.0
    assert selective.paired_t([-0.01] * 5)[0] < 0


def test_the_family_takes_in_every_model_that_ran(tmp_path):
    """The declaration corrects over every model that ran. The optional third family's
    runs must therefore enter the family by default, and while they are incomplete the
    declared analysis must refuse rather than quietly correct over two models."""
    rep = _script("goemo_replication", "16_goemo_replication.py")
    for model in rep.MODELS:
        for arm in rep.ARMS:
            for seed in (13, 21, 34, 55, 89):
                _write_run(tmp_path, arm, seed, model=model)
    assert rep.models_that_ran(tmp_path) == list(rep.MODELS)
    _write_run(tmp_path, "sft", 13, model="smollm3-3b")
    assert rep.models_that_ran(tmp_path) == [*rep.MODELS, "smollm3-3b"]
    with pytest.raises(SystemExit, match="smollm3-3b"):
        rep.main(["--runs", str(tmp_path), "--out", str(tmp_path / "out")])
