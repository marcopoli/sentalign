"""Ranking metrics must read the order of c(x), not its double-precision rounding.

Under margin-inflating objectives most stored confidences are exactly 1.0 while the logits
still order the items. Every loader that feeds a ranking metric is checked on items whose
stored confidence is 1.0 throughout and whose logits put every error below every correct
answer: a loader reading the stored value sees ties (AUROC 0.5, no gain from abstaining), one
reading the exact order sees a perfect ranking.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sentalign.confidence import order_score, top_probability        # noqa: E402

N = 60                                   # the loaders ignore sets with fewer than 50 items


def _script(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _rows():
    rows = []
    for i in range(N):
        margin = 40.0 + i * 0.5              # all saturate: exp(-40) is below double epsilon
        z = np.array([0.0, -margin, -margin - 3.0, -margin - 6.0])
        p = np.exp(z - z.max()) / np.exp(z - z.max()).sum()
        rows.append({"text_id": f"t{i}", "logits": z.tolist(), "probs": p.tolist(),
                     "correct": i >= N // 2, "group": "5of5", "p_human": [1, 0, 0, 0]})
    assert all(max(r["probs"]) == 1.0 for r in rows), "premise: every stored c is 1.0"
    return rows


def _write(runs: Path, study: str, sets, arm="kto", model="lfm-1.2b", seed=13):
    run = runs / f"{study}__{model}__{arm}__eps0.0__tau0.2__n8000__s{seed}" / "eval"
    run.mkdir(parents=True)
    for name in sets:
        (run / f"predictions_{name}.jsonl").write_text(
            "\n".join(json.dumps(r) for r in _rows()))
    return run.parent


def test_order_score_is_the_exact_order_of_c():
    rng = np.random.default_rng(1)
    for _ in range(100):
        z = rng.normal(scale=3.0, size=4)
        p = np.exp(z - z.max()) / np.exp(z - z.max()).sum()
        row = {"logits": z.tolist(), "probs": p.tolist()}
        assert math.isclose(order_score(row), math.log(p.max() / (1 - p.max())), rel_tol=1e-9)
        assert top_probability(row) == p.max()
    scores = [order_score(r) for r in _rows()]
    assert scores == sorted(scores) and len(set(scores)) == N


def test_the_regulariser_statistics_rank_saturated_items(tmp_path):
    stats = _script("regulariser_stats", "06_regulariser_stats.py")
    _write(tmp_path, "main", stats.SETS)
    scores, _, _ = stats.load(tmp_path, {"lfm-1.2b"})
    cell = scores[("lfm-1.2b", "ambig_eval", "kto")][13]
    assert cell["auroc"] == 1.0 and cell["eaurc"] == pytest.approx(0.0)


def test_the_second_task_and_corpus_loader_ranks_saturated_items(tmp_path):
    nli = _script("nli_robustness", "12_nli_robustness.py")
    _write(tmp_path, "goemo", ("ambig_eval", "go_test"))
    scores = nli.load(tmp_path, "lfm-1.2b", sets=("ambig_eval", "go_test"), study="goemo")
    assert scores["kto"][13]["auroc"] == 1.0


def test_coverage_and_deferral_rank_saturated_items(tmp_path):
    coverage = _script("coverage_table", "10_coverage_table.py")
    deferral = _script("deferral_cost", "15_deferral_cost.py")
    run = _write(tmp_path, "main", coverage.SETS)
    pairs = coverage.pooled_items(run)
    assert coverage.risk_at_coverage(pairs, 0.5) == 0.0, "abstaining must buy accuracy"
    items = deferral.load_items(run)
    assert len({c for _, c, _ in items}) == N, "the threshold must be able to split them"


def test_within_band_auroc_ranks_saturated_items():
    bands = _script("agreement_bands", "13_agreement_bands.py")
    stats = _script("regulariser_stats", "06_regulariser_stats.py")
    result = bands.band_metrics(_rows(), auroc=stats.auroc)
    assert result["5of5"]["auroc"] == 1.0
    assert result["5of5"]["above_threshold"] == 1.0, "threshold counts read c itself"
