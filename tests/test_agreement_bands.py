"""The agreement-band analysis the discussion quotes."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "13_agreement_bands.py"


@pytest.fixture(scope="module")
def bands():
    spec = importlib.util.spec_from_file_location("agreement_bands", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _auroc(conf, correct):
    pos, neg = conf[correct == 1], conf[correct == 0]
    return float(((pos[:, None] > neg[None, :]).sum()
                  + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))


def row(group, correct, conf):
    return {"group": group, "correct": correct, "probs": [conf, 1.0 - conf]}


def test_each_band_is_measured_on_its_own_items(bands):
    """A band's numbers must not leak items from another band: the claim is exactly about
    how the bands differ."""
    rows = ([row("5of5", True, 0.95)] * 8 + [row("5of5", False, 0.6)] * 2
            + [row("3of5", True, 0.7)] * 5 + [row("3of5", False, 0.97)] * 5)
    out = bands.band_metrics(rows, auroc=_auroc)
    assert out["5of5"]["error"] == pytest.approx(0.2)
    assert out["5of5"]["confident_error"] == pytest.approx(0.0), "its errors are unconfident"
    assert out["3of5"]["error"] == pytest.approx(0.5)
    assert out["3of5"]["confident_error"] == pytest.approx(0.5)
    assert out["3of5"]["auroc"] == pytest.approx(0.0), "every error ranked above every hit"
    assert "4of5" not in out


def test_the_share_above_threshold_is_reported_beside_the_rate(bands):
    """The confident-error rate of an arm that rarely exceeds the threshold is near zero
    whatever its ranking, so the share it sits beside is what lets a reader discount it."""
    rows = [row("4of5", False, 0.6)] * 10
    out = bands.band_metrics(rows, auroc=_auroc)
    assert out["4of5"]["confident_error"] == 0.0
    assert out["4of5"]["above_threshold"] == 0.0


def _result(bands):
    """Every arm, band and model gets numbers that identify it, so a value printed in the
    wrong row cannot pass for the right one."""
    out = {}
    for m, model in enumerate(bands.MODELS):
        for a, arm in enumerate(bands.ARMS):
            out[f"{model}|{arm}"] = {
                band: {"n": 1000.0 + b, "error": 0.1 * (m + 1) + 0.01 * a + 0.001 * b,
                       "above_threshold": 0.1 * a + 0.01 * b + 0.3 * m,
                       "confident_error": 0.2 + 0.01 * m + 0.001 * a + 0.0001 * b,
                       "auroc": 0.5 + 0.1 * m + 0.01 * a + 0.001 * b}
                for b, band in enumerate(bands.BANDS)}
    return out


def test_every_table_row_carries_its_own_arm_and_band(bands):
    """The discussion reads KTO's 3/5 row against SFT's; a row that printed another arm's
    numbers under its label would reverse the argument without any other symptom."""
    result = _result(bands)
    lines = bands.table(result).splitlines()
    rows = [l for l in lines if l.endswith(r"\\") and "&" in l and not l.startswith("Model")]
    assert len(rows) == len(bands.MODELS) * len(bands.BANDS) * len(bands.ARMS)
    model = band = None
    for line in rows:
        cells = [c.strip() for c in line[:-2].split("&")]
        model = next((k for k, v in bands.MODEL_NAMES.items() if v == cells[0]), model)
        band = next((k for k, v in bands.BAND_NAMES.items() if v == cells[1]), band)
        arm = next(k for k, v in bands.ARM_NAMES.items() if v == cells[2])
        v = result[f"{model}|{arm}"][band]
        assert cells[3:] == [f"{v['error']:.3f}", f"{v['above_threshold']:.2f}",
                             f"{v['confident_error']:.3f}", f"{v['auroc']:.3f}"], line


def test_the_caption_states_the_band_sizes_it_was_computed_on(bands):
    caption = bands.table(_result(bands))
    assert "5/5: 1{,}000" in caption and "3/5: 1{,}002" in caption
