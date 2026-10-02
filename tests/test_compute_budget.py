"""The run count and GPU-hour total the manuscript quotes."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _module():
    spec = importlib.util.spec_from_file_location("compute_budget",
                                                  ROOT / "scripts" / "17_compute_budget.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(root: Path, name: str, hours: float | None) -> None:
    d = root / name
    d.mkdir(parents=True)
    if hours is not None:
        (d / "manifest.json").write_text(json.dumps({"gpu_hours": hours}))


def test_counts_reported_runs_and_leaves_out_the_rest(tmp_path):
    budget = _module().budget
    _run(tmp_path, "main__lfm-1.2b__kto__eps0.0__tau0.2__n8000__s13", 1.3)
    _run(tmp_path, "goemo__qwen-2b__kto-dreg1.0__eps0.0__tau0.2__n8000__s21", 2.0)
    _run(tmp_path, "main__lfm-1.2b__rdpo-beta0.3-dreg1.0__eps0.0__tau0.2__n8000__s13", 0.5)
    _run(tmp_path, "main__qwen-2b__rdpo-rpo1.0__eps0.0__tau0.2__n8000__s89", 0.6)
    # A beta variant of an arm outside the study is not counted either.
    _run(tmp_path, "main__lfm-1.2b__cspo-beta0.5__eps0.0__tau0.2__n8000__s13", 9.0)
    # Read by no reported analysis: an exploratory arm, another budget, an undeclared seed,
    # and the natural language inference study the paper does not report.
    _run(tmp_path, "main__lfm-1.2b__cspo__eps0.0__tau0.2__n8000__s13", 9.0)
    _run(tmp_path, "main__lfm-1.2b__kto__eps0.0__tau0.2__n16000__s13", 9.0)
    _run(tmp_path, "main__lfm-1.2b__kto__eps0.0__tau0.2__n8000__s7", 9.0)
    _run(tmp_path, "nli__lfm-1.2b__kto__eps0.0__tau0.2__n8000__s13", 9.0)
    # Still running: no manifest, so in neither total.
    _run(tmp_path, "goemo__smollm3-3b__kto__eps0.0__tau0.2__n8000__s89", None)
    result = budget(tmp_path)
    assert result["runs"] == 4
    assert result["gpu_hours"] == 4.4
    assert result["groups"]["beta sensitivity"] == {"runs": 1, "gpu_hours": 0.5}
    assert result["groups"]["RPO comparison"] == {"runs": 1, "gpu_hours": 0.6}
    assert result["groups"]["GoEmotions replication"] == {"runs": 1, "gpu_hours": 2.0}
    assert result["in_progress"] == ["goemo__smollm3-3b__kto__eps0.0__tau0.2__n8000__s89"]
