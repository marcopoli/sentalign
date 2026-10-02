"""The beta-sensitivity and RPO studies train what they were declared to train.

Both studies change one setting of an arm that already exists, so the ways they can go
wrong are quiet: a beta that never reaches the trainer reproduces the beta = 0.1 run under
another name, an anchored arm paired with the twin at the wrong beta compares two changes
at once, and an ``rpo_alpha`` that a trainer ignores trains plain R-DPO as the RPO arm.
"""

from __future__ import annotations

import importlib.util
from dataclasses import asdict
from importlib.util import find_spec
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
needs_trl = pytest.mark.skipif(find_spec("trl") is None, reason="trl not installed")


def _script(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _study(name):
    from sentalign.plan import full_plan

    return [r for r in full_plan((name,))]


def test_the_beta_study_is_the_declared_grid():
    runs = _study("beta")
    got = {(r.to_config().train.objective, r.to_config().train.beta,
            r.to_config().train.pref_distributional_lambda, r.seed) for r in runs}
    want = {(objective, beta, lam, seed) for objective in ("kto", "rdpo")
            for beta in (0.03, 0.3) for lam in (0.0, 1.0) for seed in (13, 21, 34)}
    assert got == want and len(runs) == len(want)
    assert {r.model for r in runs} == {"lfm-1.2b"}
    assert all(r.to_config().train.rpo_alpha is None for r in runs)


def test_each_anchored_beta_arm_is_compared_with_its_twin_at_the_same_beta():
    stats = _script("regulariser_stats_beta", "06_regulariser_stats.py")
    arms = {}
    for run in _study("beta"):
        cfg = run.to_config()
        arms[cfg.run_id.split("__")[2]] = cfg.train
    for arm, train in arms.items():
        if train.pref_distributional_lambda:
            twin = arms[stats.twin(arm)]
            assert twin.pref_distributional_lambda == 0.0
            assert (twin.objective, twin.beta) == (train.objective, train.beta)


def test_the_rpo_study_is_r_dpo_with_the_term_on_two_models_and_five_seeds():
    runs = _study("rpo")
    ids = {r.to_config().run_id for r in runs}
    assert ids == {f"main__{m}__rdpo-rpo1.0__eps0.0__tau0.2__n8000__s{s}"
                   for m in ("lfm-1.2b", "qwen-2b") for s in (13, 21, 34, 55, 89)}
    for run in runs:
        train = run.to_config().train
        assert (train.objective, train.rpo_alpha, train.beta, train.label_smoothing,
                train.pref_distributional_lambda) == ("rdpo", 1.0, 0.1, 0.1, 0.0)


def test_the_trainer_rebuilds_every_declared_setting_from_the_command():
    """The runner reaches the trainer only through argv. The run directory surviving that
    trip is tested elsewhere; here the whole training section has to."""
    from sentalign.cli import _build_config, build_parser

    grid = _script("run_grid_beta", "02_run_grid.py")
    parser = build_parser()
    for run in _study("beta") + _study("rpo"):
        planned = run.to_config()
        cmd = grid.build_command(run, planned, Path("data/build"), Path("runs"))
        rebuilt = _build_config(parser.parse_args([str(part) for part in cmd[3:]]))
        assert rebuilt.run_id == planned.run_id
        assert asdict(rebuilt.train) == asdict(planned.train), planned.run_id


def _args(objective, **train):
    from sentalign.config import ExperimentConfig
    from sentalign.recovery import BatchPlan
    from sentalign.train.po import build_dpo_config, build_kto_config

    cfg = ExperimentConfig(name="test", model="lfm-350m")
    cfg.train.objective = objective
    for key, value in train.items():
        setattr(cfg.train, key, value)
    if objective == "kto":
        return build_kto_config(cfg, Path("out"), BatchPlan(8, 4), desirable_weight=1.0,
                                undesirable_weight=1.3)
    return build_dpo_config(cfg, Path("out"), BatchPlan(8, 4))


@needs_trl
def test_the_rpo_arm_differs_from_r_dpo_in_the_term_alone():
    plain, rpo = _args("rdpo").to_dict(), _args("rdpo", rpo_alpha=1.0).to_dict()
    assert plain["rpo_alpha"] is None, "an arm that did not ask for the term must not get it"
    assert rpo["rpo_alpha"] == 1.0
    differing = {k for k in plain if plain[k] != rpo[k]}
    assert differing == {"rpo_alpha"}, differing


@needs_trl
@pytest.mark.parametrize("objective", ("mixdpo", "grdpo", "simpo", "alphapo", "kto"))
def test_rpo_alpha_is_refused_where_the_trainer_would_drop_it(objective):
    with pytest.raises(ValueError, match="rpo_alpha"):
        _args(objective, rpo_alpha=1.0)


def test_the_guard_stops_a_run_whose_loss_lacks_the_term():
    from sentalign.train.po import RPOTermGuard

    guard = RPOTermGuard()
    guard.check({"eval_loss": 0.4, "eval_macro_f1": 0.6})     # evaluation: not checked
    guard.check({"train_runtime": 10.0, "train_loss": 0.3})    # end of training summary
    assert guard.seen == 0
    guard.check({"loss": 0.5, "nll_loss": 0.2, "learning_rate": 1e-5})
    assert guard.seen == 1
    with pytest.raises(RuntimeError, match="nll_loss"):
        guard.check({"loss": 0.5, "learning_rate": 1e-5})
