"""Structured on-disk logging. Everything a run produces is written to files.

The rule is that a completed run must be fully reconstructable from its directory with no
reference to the console, the scheduler, or the shell history that launched it. A run that
crashed must be diagnosable from the same place. This matters more than usual here because
the programme is roughly 130 hours of unattended single-GPU time, and a failure at hour 90
that leaves no trace costs days.

Layout written under ``runs/<run_id>/``::

    config.json          the full resolved configuration plus its hash
    environment.json     package versions, GPU, driver, git commit, host
    status.json          lifecycle state, updated atomically at each transition
    events.jsonl         append-only structured event stream, one JSON object per line
    console.log          captured stdout and stderr
    metrics.jsonl        every training and evaluation metric, with step and wall-clock
    telemetry.jsonl      periodic VRAM, utilisation, throughput samples
    eval/metrics.json    final aggregated metrics per evaluation set
    eval/predictions_*.jsonl   per-item predictions, probabilities, and logits
    checkpoints/         trainer checkpoints for resumption

Every writer appends and flushes, so a killed process leaves a readable partial log rather
than a truncated buffer.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import threading
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterator


class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"
    OOM_RETRY = "oom_retry"


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


#: Packages whose import order matters. Importing any of these before ``unsloth``
#: permanently disables its trainer patches for the process, and the only symptom is a run
#: slower and hungrier than its own manifest claims.
UNSLOTH_PATCHED = ("transformers", "trl", "peft")


def package_version(name: str) -> str | None:
    """Version of an installed package, without importing it.

    Reading a version must never be the thing that loads a package. Two call sites in
    this codebase used ``__import__`` for it, and because both iterated a list that put
    ``trl`` ahead of ``unsloth``, merely recording the environment was enough to switch
    off Unsloth's optimisations for the rest of the process.

    Resolution order:

    1.  An already-imported module, which carries the richest string. ``torch`` reports
        ``2.11.0+cu130`` this way against ``2.11.0`` from distribution metadata, and the
        CUDA build is worth keeping in a run manifest.
    2.  Distribution metadata, which has no import side effects.

    Returns ``None`` when the package is not installed.
    """
    module = sys.modules.get(name)
    version = getattr(module, "__version__", None) if module is not None else None
    if version:
        return str(version)

    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _dist_version

    try:
        return _dist_version(name)
    except PackageNotFoundError:
        return None
    except Exception:
        return None


def is_installed(name: str) -> bool:
    """Whether a package is installed, without importing it."""
    if name in sys.modules:
        return True
    from importlib.util import find_spec

    try:
        return find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (set, frozenset, tuple)):
        return list(value)
    if hasattr(value, "item"):          # numpy scalars
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, "tolist"):        # numpy arrays
        try:
            return value.tolist()
        except Exception:
            pass
    return str(value)


def write_json_atomic(path: Path, payload: Any) -> None:
    """Write via a temporary file and rename, so a reader never sees a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n",
                   encoding="utf-8")
    os.replace(tmp, path)


class JsonlWriter:
    """Append-only JSONL sink that flushes and fsyncs on every record."""

    def __init__(self, path: Path, fsync: bool = False) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")
        self._lock = threading.Lock()
        self._fsync = fsync

    def write(self, record: dict) -> None:
        line = json.dumps(record, default=_json_default)
        with self._lock:
            self._fh.write(line + "\n")
            self._fh.flush()
            if self._fsync:
                os.fsync(self._fh.fileno())

    def close(self) -> None:
        with self._lock:
            if not self._fh.closed:
                self._fh.close()


class Tee:
    """Duplicate a stream to a file, so console output survives the process."""

    def __init__(self, stream, path: Path) -> None:
        self.stream = stream
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = path.open("a", encoding="utf-8")

    def write(self, data: str) -> int:
        self.stream.write(data)
        self.file.write(data)
        self.file.flush()
        return len(data)

    def flush(self) -> None:
        self.stream.flush()
        self.file.flush()

    def isatty(self) -> bool:
        return getattr(self.stream, "isatty", lambda: False)()

    def close(self) -> None:
        if not self.file.closed:
            self.file.close()


#: Directories whose contents define "the code that produced this run".
FINGERPRINT_ROOTS = ("src", "scripts")
FINGERPRINT_SUFFIXES = (".py", ".sh", ".toml", ".txt")

#: Build and install metadata that lives under ``src`` but is not source. An editable
#: install writes ``src/<pkg>.egg-info/*.txt``, and hashing those made the digest answer
#: a different question than the one it is asked. ``SOURCES.txt`` enumerates every file
#: in the distribution, so adding a *test* moved it; re-running ``pip install -e .``
#: rewrote it with no code change at all; and a machine with no editable install
#: computed a different digest for byte-identical source, which defeats comparing two
#: hosts. Found when the GPU host and the laptop disagreed while all 35 source files
#: matched exactly.
FINGERPRINT_EXCLUDED_DIR_SUFFIXES = (".egg-info", ".dist-info", ".eggs")


def code_fingerprint(root: Path | None = None) -> dict[str, Any]:
    """A content hash of the source tree, as version control without version control.

    A run needs to record which code produced it. Where the project is under version
    control the commit answers that, but a commit is a proxy: it identifies a tree, and
    only if the working copy is clean. Hashing the files directly answers the question
    without depending on any tooling being set up, and it cannot be defeated by
    uncommitted edits.

    The value that matters is stability across a programme. If a fix lands at run 90, the
    fingerprint changes, and the manifests of runs 1 to 89 differ from those of 91 to 277.
    That difference is visible in aggregation rather than silently averaged away, which is
    the whole point of recording provenance in the first place.

    Returns the digest, the file count, and the newest modification time, so a changed
    fingerprint can be localised in time without a diff.
    """
    root = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    digest = hashlib.blake2b(digest_size=16)
    count = 0
    newest = 0.0
    for directory in FINGERPRINT_ROOTS:
        base = root / directory
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file() or path.suffix not in FINGERPRINT_SUFFIXES:
                continue
            if "__pycache__" in path.parts:
                continue
            if any(part.endswith(FINGERPRINT_EXCLUDED_DIR_SUFFIXES)
                   for part in path.parts):
                continue
            # Path first, so a rename changes the digest even when contents do not.
            digest.update(str(path.relative_to(root)).encode())
            digest.update(path.read_bytes())
            count += 1
            newest = max(newest, path.stat().st_mtime)
    return {
        "digest": digest.hexdigest(),
        "n_files": count,
        "newest_mtime": (datetime.fromtimestamp(newest, timezone.utc)
                         .isoformat(timespec="seconds") if newest else None),
        "roots": list(FINGERPRINT_ROOTS),
    }


def collect_environment() -> dict[str, Any]:
    """Everything needed to reproduce or explain a run, recorded once at start."""
    env: dict[str, Any] = {
        "timestamp": utcnow(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "cpu_count": os.cpu_count(),
        "argv": sys.argv,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    # The kernel packages belong here with the rest: whether they are installed decides
    # which implementation of LFM2's short convolution and Qwen3.5's linear attention runs,
    # and the two are the same function only up to floating-point association. A run
    # trained with them is not bit-comparable to one trained without.
    for name in ("torch", "transformers", "trl", "peft", "datasets", "accelerate",
                 "bitsandbytes", "unsloth", "numpy",
                 "causal-conv1d", "flash-linear-attention"):
        env[name] = package_version(name)
    # torch is safe to import: unsloth does not patch it, and it is already loaded in any
    # process that is about to train. Guarded so a metadata-only call stays side-effect free.
    try:
        import torch

        env["torch_cuda"] = torch.version.cuda
        env["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            env["gpu_name"] = props.name
            env["gpu_total_memory_gb"] = round(props.total_memory / 1e9, 2)
            env["gpu_capability"] = f"{props.major}.{props.minor}"
            env["gpu_count"] = torch.cuda.device_count()
            env["bf16_supported"] = torch.cuda.is_bf16_supported()
    except Exception:
        pass
    fingerprint = code_fingerprint()
    env["code_fingerprint"] = fingerprint["digest"]
    env["code_files"] = fingerprint["n_files"]
    env["code_newest_mtime"] = fingerprint["newest_mtime"]

    # Version control is optional. When present it is recorded as well, because a commit
    # is easier to communicate than a digest; when absent the fingerprint above stands
    # alone and answers the same question.
    for cmd, key in ((["git", "rev-parse", "HEAD"], "git_sha"),
                     (["git", "status", "--porcelain"], "git_dirty"),
                     (["nvidia-smi", "--query-gpu=driver_version",
                       "--format=csv,noheader"], "nvidia_driver")):
        try:
            out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL,
                                          timeout=10).strip()
            env[key] = bool(out) if key == "git_dirty" else out
        except Exception:
            env[key] = None
    return env


def gpu_snapshot() -> dict[str, Any]:
    """Current GPU memory and utilisation, cheap enough to sample often."""
    snap: dict[str, Any] = {"t": time.time()}
    try:
        import torch

        if torch.cuda.is_available():
            snap["allocated_gb"] = round(torch.cuda.memory_allocated() / 1e9, 3)
            snap["reserved_gb"] = round(torch.cuda.memory_reserved() / 1e9, 3)
            snap["max_allocated_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 3)
    except Exception:
        pass
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,temperature.gpu,"
             "power.draw", "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL, timeout=5).strip().splitlines()[0]
        util, mem, temp, power = (x.strip() for x in out.split(","))
        snap.update({"gpu_util_pct": float(util), "gpu_mem_used_mb": float(mem),
                     "gpu_temp_c": float(temp), "gpu_power_w": float(power)})
    except Exception:
        pass
    return snap


@dataclass
class RunLogger:
    """One run's on-disk record.

    Use as a context manager so that status transitions and tracebacks are written even
    when the process is killed or raises.
    """

    run_dir: Path
    run_id: str
    _events: JsonlWriter | None = field(default=None, init=False, repr=False)
    _metrics: JsonlWriter | None = field(default=None, init=False, repr=False)
    _telemetry: JsonlWriter | None = field(default=None, init=False, repr=False)
    _tees: list[Tee] = field(default_factory=list, init=False, repr=False)
    _started: float = field(default=0.0, init=False, repr=False)
    _stop_telemetry: threading.Event | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.run_dir = Path(self.run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "eval").mkdir(exist_ok=True)
        self._events = JsonlWriter(self.run_dir / "events.jsonl", fsync=True)
        self._metrics = JsonlWriter(self.run_dir / "metrics.jsonl")
        self._telemetry = JsonlWriter(self.run_dir / "telemetry.jsonl")
        self._started = time.time()

    # -- lifecycle ---------------------------------------------------------------------

    def __enter__(self) -> "RunLogger":
        sys.stdout = Tee(sys.stdout, self.run_dir / "console.log")
        sys.stderr = Tee(sys.stderr, self.run_dir / "console.log")
        self._tees = [sys.stdout, sys.stderr]
        write_json_atomic(self.run_dir / "environment.json", collect_environment())
        self.set_status(RunStatus.RUNNING)
        self.event("run_start", run_id=self.run_id)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.stop_telemetry()
        if exc_type is None:
            self.set_status(RunStatus.COMPLETED)
            self.event("run_end", wallclock_s=round(time.time() - self._started, 2))
        elif exc_type is KeyboardInterrupt:
            self.set_status(RunStatus.INTERRUPTED)
            self.event("run_interrupted",
                       wallclock_s=round(time.time() - self._started, 2))
        else:
            self.set_status(RunStatus.FAILED, error=f"{exc_type.__name__}: {exc}")
            self.event("run_failed", error_type=exc_type.__name__, error=str(exc),
                       traceback="".join(traceback.format_exception(exc_type, exc, tb)))
        self.close()
        return False        # never swallow the exception

    def close(self) -> None:
        for writer in (self._events, self._metrics, self._telemetry):
            if writer is not None:
                writer.close()
        for tee in self._tees:
            if isinstance(tee, Tee):
                tee.close()
        sys.stdout = getattr(sys.stdout, "stream", sys.stdout)
        sys.stderr = getattr(sys.stderr, "stream", sys.stderr)

    # -- writers -----------------------------------------------------------------------

    def event(self, kind: str, **payload: Any) -> None:
        self._events.write({"t": utcnow(), "elapsed_s": round(time.time() - self._started, 2),
                            "kind": kind, **payload})

    def metric(self, stage: str, step: int | None = None, **values: Any) -> None:
        self._metrics.write({"t": utcnow(),
                             "elapsed_s": round(time.time() - self._started, 2),
                             "stage": stage, "step": step, **values})

    def set_status(self, status: RunStatus, **extra: Any) -> None:
        write_json_atomic(self.run_dir / "status.json", {
            "run_id": self.run_id, "status": status.value, "updated": utcnow(),
            "elapsed_s": round(time.time() - self._started, 2), **extra})

    def save_config(self, config: dict) -> None:
        write_json_atomic(self.run_dir / "config.json", config)

    def save(self, name: str, payload: Any) -> None:
        write_json_atomic(self.run_dir / name, payload)

    # -- telemetry ---------------------------------------------------------------------

    def start_telemetry(self, interval_s: float = 30.0) -> None:
        """Sample GPU state on a background thread for the cost table."""
        self._stop_telemetry = threading.Event()

        def loop() -> None:
            while not self._stop_telemetry.wait(interval_s):
                try:
                    self._telemetry.write(gpu_snapshot())
                except Exception:
                    return

        threading.Thread(target=loop, daemon=True, name="telemetry").start()

    def stop_telemetry(self) -> None:
        if self._stop_telemetry is not None:
            self._stop_telemetry.set()

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Bracket a phase of the run so its duration and outcome are both recorded."""
        started = time.time()
        self.event("stage_start", stage=name)
        try:
            yield
        except Exception as exc:
            self.event("stage_failed", stage=name, error_type=type(exc).__name__,
                       error=str(exc), duration_s=round(time.time() - started, 2))
            raise
        else:
            self.event("stage_end", stage=name,
                       duration_s=round(time.time() - started, 2))


def read_status(run_dir: Path) -> dict | None:
    path = Path(run_dir) / "status.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def is_complete(run_dir: Path) -> bool:
    """A run counts as done only when its status says so and its evaluation exists."""
    status = read_status(run_dir)
    return (status is not None
            and status.get("status") == RunStatus.COMPLETED.value
            and (Path(run_dir) / "eval" / "metrics.json").exists())


def read_events(run_dir: Path) -> list[dict]:
    path = Path(run_dir) / "events.jsonl"
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue        # tolerate a truncated final line from a killed process
    return out
