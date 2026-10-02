"""The beta and RPO analysis computes the families that were declared, and only those."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _script(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mods():
    return (_script("beta_rpo", "23_beta_rpo.py"),
            _script("regulariser_stats", "06_regulariser_stats.py"),
            _script("selective", "05_selective.py"))


def _cell(rng, centre):
    return {"auroc": centre + rng.normal(scale=0.01), "eaurc": 0.1 + rng.normal(scale=0.01),
            "acc": 0.72 + rng.normal(scale=0.01), "ece": 0.1}


def _mean(arms, models, seeds, centre=None, rng=None):
    rng = rng or np.random.default_rng(0)
    centre = centre or {}
    return {(m, "mean4", a): {s: _cell(rng, centre.get(a, 0.75)) for s in seeds}
            for m in models for a in arms}


RPO_ARMS = ("sft", "rdpo", "rdpo-dreg1.0", "rdpo-rpo1.0")
SEEDS = (13, 21, 34, 55, 89)


def test_the_rpo_family_is_holm_over_its_four_declared_tests(mods):
    mod, stats, selective = mods
    mean = _mean(RPO_ARMS, ("lfm-1.2b", "qwen-2b"), SEEDS,
                 centre={"rdpo": 0.70, "rdpo-rpo1.0": 0.74, "rdpo-dreg1.0": 0.78})
    rows, missing = mod.rpo_rows(mean, stats=stats, selective=selective)
    assert not missing
    r1 = rows["R1"]
    assert len(r1) == 4 and {r["metric"] for r in r1} == {"auroc"}
    assert {(r["model"], r["first"], r["second"]) for r in r1} == {
        (m, f, s) for m in ("lfm-1.2b", "qwen-2b")
        for f, s in (("rdpo-rpo1.0", "rdpo"), ("rdpo-dreg1.0", "rdpo-rpo1.0"))}
    want = selective.holm({i: r["p"] for i, r in enumerate(r1)})
    assert [r["p_holm"] for r in r1] == [want[i] for i in range(4)]
    assert all("p_holm" not in r for r in rows["R2"]), "R2 is descriptive"
    assert {(r["first"], r["second"], r["metric"]) for r in rows["R2"]} >= {
        ("rdpo-rpo1.0", "sft", m) for m in ("auroc", "eaurc", "acc")}


def test_beta_rows_use_the_declared_seeds_and_the_twin_at_the_same_beta(mods):
    mod, stats, _ = mods
    arms = ["sft"] + [mod.beta_arm(o, b, r) for o in ("kto", "rdpo") for b in mod.BETAS
                      for r in (False, True)]
    mean = _mean(arms, ("lfm-1.2b",), SEEDS, centre={"kto-beta0.03": 0.60})
    # The paper's beta = 0.1 arms and SFT have five seeds; seeds 55 and 89 are made
    # extreme so a row that used them would show it.
    for arm in ("sft", "kto", "kto-dreg1.0"):
        for s in (55, 89):
            mean[("lfm-1.2b", "mean4", arm)][s]["auroc"] = 5.0
    rows, missing = mod.beta_rows(mean, stats=stats)
    assert not missing and all(r["n"] == 3 and r["seeds"] == [13, 21, 34] for r in rows)
    repair = next(r for r in rows if r["beta"] == 0.03 and r["metric"] == "auroc"
                  and r["first"] == "kto-beta0.03-dreg1.0")
    assert repair["second"] == "kto-beta0.03"
    assert repair["mean_diff"] > 0.1, "compared with the damaged twin at the same beta"
    assert {r["beta"] for r in rows} == {0.03, 0.1, 0.3}


def test_a_declared_comparison_without_its_runs_stops_the_analysis(mods):
    mod, stats, selective = mods
    mean = _mean(("sft", "rdpo", "rdpo-dreg1.0"), ("lfm-1.2b", "qwen-2b"), SEEDS)
    with pytest.raises(SystemExit, match="without their runs"):
        mod.rpo_rows(mean, stats=stats, selective=selective)
    rows, missing = mod.rpo_rows(mean, stats=stats, selective=selective, interim=True)
    assert missing and not rows["R1"]
    partial = _mean(RPO_ARMS, ("lfm-1.2b", "qwen-2b"), SEEDS)
    del partial[("qwen-2b", "mean4", "rdpo-rpo1.0")][89]
    with pytest.raises(SystemExit):
        mod.rpo_rows(partial, stats=stats, selective=selective)


def _run(root, model, arm, seed, manifest=None, scored=True):
    run = root / f"main__{model}__{arm}__eps0.0__tau0.2__n8000__s{seed}"
    (run / "eval").mkdir(parents=True, exist_ok=True) if scored else run.mkdir(parents=True)
    if manifest is not None:
        (run / "manifest.json").write_text(json.dumps(manifest))
    return run


def test_an_rpo_run_whose_manifest_does_not_show_the_term_is_refused(mods, tmp_path):
    mod, _, _ = mods
    good = {"rpo_alpha": 1.0, "rpo_nll_logs": 20}
    _run(tmp_path, "lfm-1.2b", "rdpo-rpo1.0", 13, good)
    _run(tmp_path, "lfm-1.2b", "rdpo-rpo1.0", 21, {"rpo_alpha": 1.0})
    _run(tmp_path, "qwen-2b", "rdpo-rpo1.0", 13, {"rpo_alpha": 1.0, "rpo_nll_logs": 0})
    _run(tmp_path, "qwen-2b", "rdpo-rpo1.0", 21, None, scored=False)   # still training
    problems = mod.rpo_manifest_problems(tmp_path)
    assert len(problems) == 2
    assert all("s13" not in p or "qwen" in p for p in problems)


def test_a_batch_plan_that_differs_from_its_comparator_is_reported(mods, tmp_path):
    mod, _, _ = mods
    same = {"batch_plan": {"per_device_train_batch_size": 8, "gradient_accumulation_steps": 4}}
    halved = {"batch_plan": {"per_device_train_batch_size": 4, "gradient_accumulation_steps": 8}}
    _run(tmp_path, "qwen-2b", "rdpo", 13, same)
    _run(tmp_path, "qwen-2b", "rdpo-rpo1.0", 13, halved)
    _run(tmp_path, "qwen-2b", "rdpo", 21, same)
    _run(tmp_path, "qwen-2b", "rdpo-rpo1.0", 21, same)
    got = mod.batch_plan_differences(tmp_path, [("qwen-2b", "rdpo-rpo1.0", "rdpo", (13, 21))])
    assert len(got) == 1 and got[0].startswith("qwen-2b s13")


def _predictions(rng, n=60):
    rows = []
    for i in range(n):
        z = rng.normal(scale=2.0, size=4)
        p = np.exp(z - z.max()) / np.exp(z - z.max()).sum()
        rows.append({"text_id": f"t{i}", "logits": z.tolist(), "probs": p.tolist(),
                     "correct": bool(rng.random() < p.max()), "p_human": [1, 0, 0, 0]})
    return rows


def test_the_analysis_runs_end_to_end_on_the_real_layout(mods, tmp_path):
    mod, stats, _ = mods
    runs, out = tmp_path / "runs", tmp_path / "out"
    rng = np.random.default_rng(5)
    for model in mod.RPO_MODELS:
        for arm in RPO_ARMS:
            for seed in SEEDS:
                manifest = {"selection_history": [{"step": 150}]}
                if arm == "rdpo-rpo1.0":
                    manifest |= {"rpo_alpha": 1.0, "rpo_nll_logs": 20}
                run = _run(runs, model, arm, seed, manifest)
                for name in stats.SETS:
                    (run / "eval" / f"predictions_{name}.jsonl").write_text(
                        "\n".join(json.dumps(r) for r in _predictions(rng)))
    assert mod.main(["--runs", str(runs), "--out", str(out), "--interim"]) == 0
    result = json.loads((out / "beta_rpo.json").read_text())
    assert len(result["rpo"]["R1"]) == 4 and result["interim"] is True
    table = (out / "table_rpo.tex").read_text()
    assert "better" in table and r"$\uparrow$" in table and r"$\downarrow$" in table
    with pytest.raises(SystemExit):
        mod.main(["--runs", str(runs), "--out", str(out)])     # beta runs are missing


def test_the_confidence_readout_covers_every_beta_arm_on_the_declared_seeds(mods, tmp_path):
    mod, _, _ = mods
    seen = {}

    class FakeMechanism:
        @staticmethod
        def collect(runs, *, models, arms, seeds):
            seen.update(models=models, arms=arms, seeds=seeds)
            return {f"{models[0]}|{a}": {"auroc": 0.7, "log_odds": 3.0, "sensitivity": 0.3,
                                        "seeds": len(seeds)} for a in arms}

    out = mod.beta_confidence(tmp_path, mech=FakeMechanism)
    assert seen["seeds"] == (13, 21, 34) and seen["models"] == ("lfm-1.2b",)
    want = {"sft"} | {mod.beta_arm(o, b, r) for o in ("kto", "rdpo") for b in mod.BETAS
                      for r in (False, True)}
    assert set(seen["arms"]) == want and set(out) == want
