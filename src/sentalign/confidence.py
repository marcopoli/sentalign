"""The confidence the paper defines, computed so that it cannot saturate.

The confidence shown to a user is ``c(x) = max_y softmax(l)_y`` (Eq. 3 of the paper). Every
ranking metric in the analysis (AUROC, E-AURC, the risk-coverage curve, the deferral
threshold) depends on ``c`` only through the order it puts items in. Stored as a double, ``c``
stops ordering items once the runner-up probability falls below the resolution of the type:
under the objectives that inflate the decision margin, logit gaps of 40 to 60 put ``c`` at
exactly 1.0 for most items, and the ranking metrics then see ties that the model does not
have. ``order_score`` is ``-log((1 - c) / c)`` computed from the logits, a strictly increasing
function of the exact ``c``: it orders items exactly as ``c`` does in exact arithmetic and does
not saturate. Calibration and threshold counts read ``c`` itself, where the difference between
1.0 and 1 - 1e-17 is immaterial.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


def top_probability(row: Mapping) -> float:
    """``c(x)``, the largest label probability, as the user would see it."""
    return float(max(row["probs"]))


def order_score(row: Mapping) -> float:
    """A strictly increasing function of the exact ``c(x)``, for every ranking metric.

    From the label logits when the row stores them (every reported run does); otherwise from
    the stored probability, which is exact below saturation and ties above it.
    """
    logits: Sequence[float] | None = row.get("logits")
    if logits:
        top = max(range(len(logits)), key=lambda i: logits[i])
        rest = [logits[i] - logits[top] for i in range(len(logits)) if i != top]
        m = max(rest)
        return -(m + math.log(sum(math.exp(r - m) for r in rest)))
    c = top_probability(row)
    return math.inf if c >= 1.0 else math.log(c) - math.log1p(-c)
