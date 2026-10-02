"""Test-session setup.

Establishes the same import order the production entry points use. ``sentalign.modeling``
calls ``ensure_unsloth_first()`` at module scope, so importing it here guarantees Unsloth
loads before any test module reaches for transformers or peft.

Without this, ``tests/test_lora_targets.py`` imports peft directly to attach adapters to
its mock module trees, and on a machine with the training stack installed that happens
before any test touches ``sentalign.modeling``. Unsloth then warns, correctly, that its
patches did not apply. The warning is real but test-only: ``sentalign.cli`` imports
``modeling`` as its first statement, and the subprocess probes in ``test_infra.py`` verify
that every production entry point loads Unsloth first. Importing it here removes the noise
without papering over anything, because the probes still run in fresh interpreters where
this file has no effect.
"""

from __future__ import annotations

import warnings

# Import for the side effect: Unsloth before transformers, trl, and peft.
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    try:
        import sentalign.modeling  # noqa: F401
    except Exception:
        # A machine without the training stack is fine: the import-order probes and the
        # static checks in test_infra.py cover the invariant on their own.
        pass
