"""Version constants and parsing, with no sentalign imports.

This module is deliberately dependency-free. The supported-TRL range lives here rather
than in ``train.po`` because ``preflight`` needs it, and reaching into ``train.po`` for a
constant drags in ``config`` and then ``modeling``, which imports Unsloth and patches the
whole process. Checking a version string should not cost twenty seconds and a global
side effect.
"""

from __future__ import annotations

#: Supported TRL range, verified against the TRL source at both ends.
#:
#: The floor is 0.24.0: below it ``loss_type`` is a bare string and the ``robust`` loss
#: used for rDPO is absent. The ceiling is the 1.x line, which is a genuine API break:
#: 1.x removed ``CPOTrainer`` (which provides SimPO and AlphaPO), replaced
#: ``get_batch_loss_metrics`` with ``_compute_loss``, and changed the preference batch
#: from separate chosen and rejected fields to a single concatenated tensor.
#:
#: The 0.2x line is also what Unsloth targets: Unsloth 2026.x gates its TRL compatibility
#: between 0.23.0 and 0.28.0 and has no branch for 1.x.
MIN_TRL = (0, 24, 0)
MAX_TRL_EXCLUSIVE = (1, 0, 0)

#: Loss types selected by name, checked against the installed TRL's documentation.
REQUIRED_LOSS_TYPES = frozenset({"sigmoid", "robust", "ipo"})

#: Packages Unsloth patches at import time. Importing any of them *before* Unsloth means
#: the patches never apply. Unsloth imports them itself once loaded, which is expected and
#: correct; the invariant is ordering, not absence.
UNSLOTH_PATCHED = ("trl", "transformers", "peft")


def parse_version(text: str) -> tuple[int, int, int]:
    """Parse a version string tolerantly, so a dev or rc suffix does not read as zero."""
    parts: list[int] = []
    for chunk in str(text).split(".")[:3]:
        digits = "".join(c for c in chunk if c.isdigit())
        parts.append(int(digits) if digits else 0)
    while len(parts) < 3:
        parts.append(0)
    return parts[0], parts[1], parts[2]


def trl_supported(version: str) -> bool:
    return MIN_TRL <= parse_version(version) < MAX_TRL_EXCLUSIVE


def trl_range_message(version: str) -> str:
    """Explain why a TRL version is rejected, and how to fix it."""
    parsed = parse_version(version)
    if parsed < MIN_TRL:
        return (f"trl {version} is too old (need >= {'.'.join(map(str, MIN_TRL))}). "
                "Below 0.24.0 `loss_type` is a bare string and the 'robust' loss used "
                "for rDPO is absent, so an arm would train something other than its "
                "name. Install with: pip install 'trl>=0.24,<1.0'")
    return (f"trl {version} is on the 1.x line, which this package does not target. "
            "1.x removed CPOTrainer (which provides SimPO and AlphaPO), renamed "
            "get_batch_loss_metrics to _compute_loss, and changed the preference batch "
            "layout; it is also outside the range Unsloth 2026.x supports. "
            "Install with: pip install 'trl>=0.24,<1.0'")
