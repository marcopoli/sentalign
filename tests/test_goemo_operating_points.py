"""GoEmotions operating points: kept-item error, near-certain errors, review cost."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (13, 21, 34, 55, 89)


def _module():
    spec = importlib.util.spec_from_file_location(
        "goemo_operating", ROOT / "scripts" / "18_goemo_operating_points.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(runs: Path, arm: str, seed: int, confidences, correct) -> None:
    run = runs / f"goemo__lfm-1.2b__{arm}__eps0.0__tau0.2__n8000__s{seed}"
    (run / "eval").mkdir(parents=True)
    for name in ("ambig_eval", "go_test"):
        rows = [json.dumps({"text_id": f"{name}-{i}", "probs": [c, 1 - c], "correct": k})
                for i, (c, k) in enumerate(zip(confidences, correct))]
        (run / "eval" / f"predictions_{name}.jsonl").write_text("\n".join(rows))


def test_ranking_and_near_certain_errors_are_measured_on_the_right_items(tmp_path):
    op = _module()
    correct = [True] * 30 + [False] * 10
    ranked = [0.95] * 30 + [0.6] * 10             # every error below every correct answer
    flat = [0.97] * 40                            # the same confidence on everything
    arms = (("sft", "SFT"), ("kto", "KTO"))
    for seed in SEEDS:
        _write(tmp_path, "sft", seed, ranked, correct)
        _write(tmp_path, "kto", seed, flat, correct)
    result = op.summarise(op.collect(tmp_path, ("lfm-1.2b",), arms, SEEDS),
                          ("lfm-1.2b",), arms, SEEDS)
    sft, kto = result["means"]["lfm-1.2b|sft"], result["means"]["lfm-1.2b|kto"]
    assert sft["error_full"] == kto["error_full"] == pytest.approx(0.25)
    assert sft["error_half"] == 0.0, "a perfect ranking keeps no error at half coverage"
    assert kto["error_half"] == pytest.approx(0.25), "a flat score cannot buy accuracy"
    assert sft["confident_error"] == 0.0 and kto["confident_error"] == pytest.approx(0.25)
    assert kto["confident"] == 1.0
    assert "tab:goemo-operating" in op.table(result, ("lfm-1.2b",), arms)


def test_a_missing_run_is_an_error(tmp_path):
    op = _module()
    for seed in SEEDS[:-1]:
        _write(tmp_path, "sft", seed, [0.9, 0.8], [True, False])
    with pytest.raises(SystemExit, match="s89"):
        op.collect(tmp_path, ("lfm-1.2b",), (("sft", "SFT"),), SEEDS)
