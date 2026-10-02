"""Failure recovery for long unattended runs on a single GPU.

The experiment programme is roughly 130 hours of continuous single-GPU time. Over that
span the failures that actually happen are mundane and repetitive: an out-of-memory error
on the one run whose batch happened to contain long inputs, a Hub timeout, a loss that
goes to NaN in bf16, a machine that reboots. None of them should cost more than the run
in flight, and none should require a human to notice.

What this module provides:

``preflight``            check VRAM, disk, tokenizer, and dependencies before any weights
                         are loaded, so a misconfiguration fails in seconds not hours
``retry``                bounded retry with exponential backoff for transient failures
``oom_backoff``          halve the micro-batch and double gradient accumulation, keeping
                         the effective batch fixed, then retry; the arithmetic of the
                         experiment is preserved while the memory footprint shrinks
``NanGuard``             detect non-finite loss and stop before the checkpoint is poisoned
``install_signal_handlers``  turn SIGINT and SIGTERM into a clean, recorded shutdown
``resume_point``         find the checkpoint to resume from, ignoring partial writes
``assert_resume_made_progress``  refuse to rewrite the record of an already finished run
``checkpoint_is_valid``  reject a checkpoint that was being written when the job died

The design rule is that recovery never silently changes what the experiment measures. An
OOM backoff preserves the effective batch size; a resume restores optimizer state rather
than restarting the schedule; anything that cannot be preserved raises instead.
"""

from __future__ import annotations

import functools
import json
import os
import re
import shutil
import signal
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence, TypeVar

T = TypeVar("T")

#: Substrings that identify an out-of-memory condition across torch and CUDA versions.
OOM_MARKERS = ("out of memory", "cuda oom", "cublas_status_alloc_failed",
               "hip out of memory", "alloc failed")

#: Failures worth retrying: network, filesystem contention, transient Hub errors.
TRANSIENT_MARKERS = ("connection reset", "timed out", "timeout", "temporary failure",
                     "503", "504", "429", "connectionerror", "readtimeout",
                     "incomplete read", "resource temporarily unavailable")


class PreflightError(RuntimeError):
    """A run cannot start. Raised before any expensive work."""


class NonFiniteLossError(RuntimeError):
    """Training loss became NaN or infinite."""


class Interrupted(RuntimeError):
    """A termination signal was received and handled."""


class AlreadyComplete(RuntimeError):
    """A resumed run had nothing left to train, so its record must not be rewritten."""


def is_oom(exc: BaseException) -> bool:
    if type(exc).__name__ in ("OutOfMemoryError", "CudaOutOfMemoryError"):
        return True
    text = str(exc).lower()
    return any(marker in text for marker in OOM_MARKERS)


def is_transient(exc: BaseException) -> bool:
    text = f"{type(exc).__name__} {exc}".lower()
    return any(marker in text for marker in TRANSIENT_MARKERS)


def free_cuda_memory() -> None:
    """Release cached blocks so a retry starts from a clean allocator."""
    try:
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
    except Exception:
        pass


# --------------------------------------------------------------------------------------
# Preflight
# --------------------------------------------------------------------------------------

@dataclass
class PreflightReport:
    checks: dict[str, tuple[bool, str]] = field(default_factory=dict)

    def add(self, name: str, ok: bool, detail: str) -> None:
        self.checks[name] = (ok, detail)

    @property
    def ok(self) -> bool:
        return all(ok for ok, _ in self.checks.values())

    def summary(self) -> str:
        lines = []
        for name, (ok, detail) in self.checks.items():
            lines.append(f"  [{'ok  ' if ok else 'FAIL'}] {name}: {detail}")
        return "\n".join(lines)

    def raise_if_failed(self) -> None:
        if not self.ok:
            failed = [f"{n}: {d}" for n, (ok, d) in self.checks.items() if not ok]
            raise PreflightError("preflight failed:\n  " + "\n  ".join(failed))


def preflight(
    *,
    min_vram_gb: float = 22.0,
    min_disk_gb: float = 20.0,
    output_dir: Path | None = None,
    tokenizer=None,
    label_space=None,
    require_bf16: bool = True,
) -> PreflightReport:
    """Validate the environment before loading any weights.

    Defaults target the RTX 3090 this programme is designed for: 24 GB of VRAM, of which
    roughly 22 GB is usable once the driver and display take their share.
    """
    report = PreflightReport()

    try:
        import torch

        if not torch.cuda.is_available():
            report.add("cuda", False, "no CUDA device visible")
        else:
            props = torch.cuda.get_device_properties(0)
            total = props.total_memory / 1e9
            report.add("cuda", total >= min_vram_gb,
                       f"{props.name}, {total:.1f} GB "
                       f"(need {min_vram_gb:.0f} GB)")
            if require_bf16:
                ok = torch.cuda.is_bf16_supported()
                report.add("bf16", ok,
                           "supported" if ok else
                           "not supported; set fp16 explicitly and expect instability")
            free = torch.cuda.mem_get_info()[0] / 1e9
            enough = free >= min_vram_gb * 0.8
            report.add("vram_free", enough,
                       f"{free:.1f} GB free" if enough else
                       f"only {free:.1f} GB free of {total:.1f} GB; another process is "
                       "probably holding memory (check nvidia-smi)")
    except ImportError:
        report.add("cuda", False, "torch is not installed")

    target = Path(output_dir or ".")
    target.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(target).free / 1e9
    report.add("disk", free_gb >= min_disk_gb,
               f"{free_gb:.1f} GB free at {target} (need {min_disk_gb:.0f} GB)")

    # Read versions without importing: see runlog.package_version. Doing this with
    # __import__ here was the second instance of the same defect, and it fired on every
    # preflight, which is the first thing every run does.
    from .runlog import is_installed, package_version

    for name in ("transformers", "trl", "peft", "datasets"):
        version = package_version(name)
        report.add(name, version is not None or is_installed(name),
                   version or "not installed")

    # Imports the dependency-free `versions` module, not `train.po`: reaching into the
    # training package for a constant would pull in `config` and then `modeling`, which
    # loads Unsloth and patches the process. A version comparison should not do that.
    try:
        from .runlog import package_version
        from .versions import trl_range_message, trl_supported

        trl_version = package_version("trl")
        if trl_version is None:
            report.add("trl_version", False, "trl is not installed")
        else:
            ok = trl_supported(trl_version)
            report.add("trl_version", ok,
                       trl_version if ok else trl_range_message(trl_version))
    except Exception as exc:
        report.add("trl_version", False, str(exc))

    if tokenizer is not None and label_space is not None:
        from .labels import VerbalizerError, resolve_verbalizers

        try:
            table = resolve_verbalizers(tokenizer, label_space)
            report.add("verbalizers", True,
                       f"single-token: {list(zip(table.surfaces, table.token_ids))}")
        except VerbalizerError as exc:
            report.add("verbalizers", False, str(exc))

    return report


# --------------------------------------------------------------------------------------
# Retry and OOM backoff
# --------------------------------------------------------------------------------------

def retry(
    attempts: int = 4,
    base_delay: float = 5.0,
    max_delay: float = 300.0,
    predicate: Callable[[BaseException], bool] = is_transient,
    on_retry: Callable[[int, BaseException, float], None] | None = None,
):
    """Retry a callable with exponential backoff while ``predicate`` holds.

    Only transient failures are retried by default. Retrying a genuine bug wastes hours
    and buries the traceback, so anything the predicate does not recognise propagates
    immediately.
    """

    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs) -> T:
            last: BaseException | None = None
            for attempt in range(1, attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except BaseException as exc:      # noqa: BLE001
                    if not predicate(exc) or attempt == attempts:
                        raise
                    last = exc
                    delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
                    if on_retry is not None:
                        on_retry(attempt, exc, delay)
                    free_cuda_memory()
                    time.sleep(delay)
            raise last                            # pragma: no cover
        return wrapper
    return decorator


@dataclass
class BatchPlan:
    """A micro-batch and accumulation pair with a fixed effective batch size."""

    per_device_batch_size: int
    gradient_accumulation_steps: int

    @property
    def effective(self) -> int:
        return self.per_device_batch_size * self.gradient_accumulation_steps

    def halve(self) -> "BatchPlan":
        """Halve the micro-batch and double accumulation, preserving the effective batch.

        Preserving the effective batch is the point. Simply lowering the batch size after
        an OOM would change the optimisation problem, so the run that eventually succeeds
        would not be comparable to the runs that did not need the backoff.
        """
        if self.per_device_batch_size <= 1:
            raise PreflightError(
                "cannot reduce the micro-batch below 1; the model or sequence length "
                "does not fit on this GPU even at batch size 1")
        return BatchPlan(self.per_device_batch_size // 2,
                         self.gradient_accumulation_steps * 2)

    def as_dict(self) -> dict[str, int]:
        return {"per_device_train_batch_size": self.per_device_batch_size,
                "gradient_accumulation_steps": self.gradient_accumulation_steps}


def oom_backoff(
    fn: Callable[[BatchPlan], T],
    plan: BatchPlan,
    *,
    max_reductions: int = 3,
    on_backoff: Callable[[BatchPlan, BatchPlan, BaseException], None] | None = None,
) -> tuple[T, BatchPlan]:
    """Call ``fn(plan)``, halving the micro-batch on OOM and retrying.

    Returns the result and the plan that succeeded, so the plan actually used is recorded
    with the run rather than inferred from the config that was requested.
    """
    current = plan
    for reduction in range(max_reductions + 1):
        try:
            return fn(current), current
        except BaseException as exc:              # noqa: BLE001
            if not is_oom(exc) or reduction == max_reductions:
                raise
            free_cuda_memory()
            reduced = current.halve()
            if on_backoff is not None:
                on_backoff(current, reduced, exc)
            current = reduced
    raise PreflightError("unreachable")           # pragma: no cover


# --------------------------------------------------------------------------------------
# Loss guarding
# --------------------------------------------------------------------------------------

@dataclass
class NanGuard:
    """Stop training when the loss stops being finite.

    A NaN loss in bf16 is usually unrecoverable and silently poisons every checkpoint
    saved afterwards. Detecting it at the step it appears means the last good checkpoint
    is still on disk, so the run can be restarted at a lower learning rate from a known
    state instead of being discovered as garbage during evaluation days later.
    """

    patience: int = 1
    _strikes: int = field(default=0, init=False)
    history: list[dict] = field(default_factory=list, init=False)

    def check(self, loss: float, step: int) -> None:
        import math

        finite = loss is not None and math.isfinite(float(loss))
        if finite:
            self._strikes = 0
            return
        self._strikes += 1
        self.history.append({"step": step, "loss": str(loss), "strikes": self._strikes})
        if self._strikes >= self.patience:
            raise NonFiniteLossError(
                f"loss was non-finite at step {step} ({loss}) on "
                f"{self._strikes} consecutive check(s). The last good checkpoint is "
                "intact; restart from it with a lower learning rate or a smaller beta.")

    def as_transformers_callback(self):
        from transformers import TrainerCallback

        guard = self

        class _Callback(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):
                if logs and "loss" in logs:
                    guard.check(logs["loss"], state.global_step)
                return control

        return _Callback()


# --------------------------------------------------------------------------------------
# Signals
# --------------------------------------------------------------------------------------

def install_signal_handlers(on_signal: Callable[[int], None] | None = None) -> None:
    """Convert SIGINT and SIGTERM into a KeyboardInterrupt the trainer can unwind.

    A hard kill loses the in-flight checkpoint and leaves the status file saying
    ``running`` forever, which makes an interrupted programme indistinguishable from a
    hung one when it is resumed.
    """

    def handler(signum, _frame):
        if on_signal is not None:
            try:
                on_signal(signum)
            except Exception:
                pass
        raise KeyboardInterrupt(f"received signal {signum}")

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, handler)
        except (ValueError, OSError):
            pass          # not on the main thread, or unsupported on this platform


# --------------------------------------------------------------------------------------
# Checkpoints and resumption
# --------------------------------------------------------------------------------------

CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")

#: Files a usable transformers checkpoint must contain. A directory that was being
#: written when the process died will be missing at least one of them.
REQUIRED_CHECKPOINT_FILES = ("trainer_state.json",)
WEIGHT_FILES = ("adapter_model.safetensors", "adapter_model.bin",
                "model.safetensors", "pytorch_model.bin")


def checkpoint_is_valid(path: Path) -> bool:
    path = Path(path)
    if not path.is_dir():
        return False
    if any(p.name.endswith(".tmp") or p.name.startswith(".") for p in path.iterdir()):
        return False
    if not all((path / name).exists() for name in REQUIRED_CHECKPOINT_FILES):
        return False
    if not any((path / name).exists() for name in WEIGHT_FILES):
        return False
    try:
        json.loads((path / "trainer_state.json").read_text())
    except Exception:
        return False
    return True


def checkpoint_step(path: Path | str | None) -> int:
    """The optimizer step a ``checkpoint-N`` directory holds, or 0 if it is not one."""
    if path is None:
        return 0
    match = CHECKPOINT_RE.match(Path(path).name)
    return int(match.group(1)) if match else 0


def assert_resume_made_progress(steps: int | None, resume: Path | str | None,
                                run_dir: Path | str) -> None:
    """Refuse to record a resumed run that executed no optimizer step.

    ``Trainer.train(resume_from_checkpoint=...)`` on a checkpoint that already sits at
    ``max_steps`` returns without stepping: ``global_step`` is the checkpoint's own step
    and ``training_loss`` is 0.0. A manifest built from that return value looks complete,
    so re-running a finished run overwrites its real provenance with a null record: loss
    0.0, an empty selection history, seconds of wallclock. That is what happened to
    ipo-dreg1.0 seed 13, which lost the checkpoint-selection record the arm is compared
    on. A finished run is left as it is; a genuine rerun starts from an empty directory.
    """
    resumed_at = checkpoint_step(resume)
    if resumed_at and int(steps or 0) <= resumed_at:
        raise AlreadyComplete(
            f"{Path(run_dir).name} resumed from {Path(resume).name} and trained no "
            f"further step, so its manifest would be replaced by a record with no loss "
            f"and no selection history. The run is already complete. Delete "
            f"{run_dir} and rerun if you want it trained again.")


def list_checkpoints(run_dir: Path) -> list[Path]:
    run_dir = Path(run_dir)
    found = []
    for base in (run_dir, run_dir / "checkpoints"):
        if base.is_dir():
            found += [p for p in base.iterdir() if CHECKPOINT_RE.match(p.name)]
    return sorted(found, key=lambda p: int(CHECKPOINT_RE.match(p.name).group(1)))


def config_changed(run_dir: Path, current_hash: str | None) -> str | None:
    """The stored config hash when it differs from ``current_hash``, else ``None``.

    Resuming across a configuration change is the quiet way to produce a run that is
    neither the old experiment nor the new one. The case that motivated this: a smoke test
    trained with an adapter configuration that reached a fraction of the decoder, then the
    configuration was corrected, and the checkpoint from the broken run was still on disk
    and still valid by every structural test. Resuming it would have carried the defect
    forward under the corrected config's name.
    """
    path = Path(run_dir) / "config.json"
    if not path.exists() or current_hash is None:
        return None
    try:
        stored = json.loads(path.read_text()).get("config_hash")
    except json.JSONDecodeError:
        return None
    return stored if stored and stored != current_hash else None


def resume_point(run_dir: Path, current_hash: str | None = None) -> Path | None:
    """The newest valid checkpoint, or ``None`` to start fresh.

    Returns ``None`` when the run's stored configuration differs from the current one, so
    a changed config always starts a clean run rather than inheriting state from a
    different experiment.

    Invalid trailing checkpoints are skipped rather than deleted: an interrupted write is
    evidence about what happened, and removing it during recovery destroys that evidence.
    """
    if config_changed(run_dir, current_hash) is not None:
        return None
    for path in reversed(list_checkpoints(run_dir)):
        if checkpoint_is_valid(path):
            return path
    return None


#: Signatures of a checkpoint whose optimizer state no longer matches the model. This
#: happens whenever the adapter configuration changes: the parameter groups differ in
#: size, and the optimizer refuses the state dict.
INCOMPATIBLE_STATE_MARKERS = (
    "doesn't match the size of optimizer's group",
    "parameter group that doesn't match",
    "loaded state dict has a different number of parameter groups",
    "size mismatch for",
)


def is_incompatible_checkpoint(exc: BaseException) -> bool:
    text = str(exc).lower()
    return any(marker.lower() in text for marker in INCOMPATIBLE_STATE_MARKERS)


def train_with_resume_fallback(train_fn, resume: Path | None, *, on_fallback=None):
    """Run ``train_fn(resume)``, restarting cleanly if the checkpoint is incompatible.

    The staleness check catches a changed configuration before training starts, but it
    depends on a recorded hash being present and correct. This is the backstop for every
    other route to the same state: a checkpoint written by a different adapter layout, an
    optimizer whose parameter groups no longer line up, a partially upgraded dependency.
    Restarting loses the partial run and nothing else; crashing loses the slot in a
    multi-day programme.
    """
    if resume is None:
        return train_fn(None), None
    try:
        return train_fn(resume), resume
    except (ValueError, RuntimeError) as exc:
        if not is_incompatible_checkpoint(exc):
            raise
        if on_fallback is not None:
            on_fallback(resume, exc)
        free_cuda_memory()
        return train_fn(None), None


def prune_checkpoints(run_dir: Path, keep: int = 2) -> list[Path]:
    """Delete all but the newest ``keep`` valid checkpoints.

    Called between runs, not during one. Each checkpoint of a 2B QLoRA model is small,
    but 313 runs of them is not, and filling the disk at hour 90 fails every run after it.
    """
    valid = [p for p in list_checkpoints(run_dir) if checkpoint_is_valid(p)]
    removed = []
    for path in valid[:-keep] if keep > 0 else valid:
        shutil.rmtree(path, ignore_errors=True)
        removed.append(path)
    return removed


def cleanup_partial(run_dir: Path) -> list[Path]:
    """Remove checkpoint directories that failed validation, after a resume point is chosen."""
    removed = []
    for path in list_checkpoints(run_dir):
        if not checkpoint_is_valid(path):
            shutil.rmtree(path, ignore_errors=True)
            removed.append(path)
    return removed
