"""Tests for logging and recovery.

These exist because the programme is roughly 150 hours of unattended single-GPU time.
Every path here is one that only executes when something has gone wrong, which is exactly
when it is least convenient to discover it does not work.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from sentalign.recovery import (BatchPlan, NanGuard, NonFiniteLossError, PreflightError,
                                PreflightReport, checkpoint_is_valid, cleanup_partial,
                                is_oom, is_transient, list_checkpoints, oom_backoff,
                                preflight, prune_checkpoints, resume_point, retry)
from sentalign.runlog import (JsonlWriter, RunLogger, RunStatus, is_complete,
                              read_events, read_status, write_json_atomic)


# --------------------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------------------

def test_run_logger_writes_every_expected_file(tmp_path):
    run_dir = tmp_path / "run"
    with RunLogger(run_dir, "demo") as log:
        log.save_config({"model": "lfm-1.2b"})
        log.metric("train", step=1, loss=0.5)
        log.event("checkpoint_saved", step=1)

    names = {p.name for p in run_dir.rglob("*") if p.is_file()}
    assert {"config.json", "environment.json", "status.json", "events.jsonl",
            "metrics.jsonl", "console.log"} <= names
    assert read_status(run_dir)["status"] == RunStatus.COMPLETED.value


def test_failure_is_recorded_with_its_traceback(tmp_path):
    run_dir = tmp_path / "run"
    with pytest.raises(ValueError):
        with RunLogger(run_dir, "boom"):
            raise ValueError("simulated CUDA failure")

    status = read_status(run_dir)
    assert status["status"] == RunStatus.FAILED.value
    assert "simulated CUDA failure" in status["error"]
    failed = [e for e in read_events(run_dir) if e["kind"] == "run_failed"]
    assert failed and "Traceback" in failed[0]["traceback"]


def test_interruption_is_distinguished_from_failure(tmp_path):
    run_dir = tmp_path / "run"
    with pytest.raises(KeyboardInterrupt):
        with RunLogger(run_dir, "interrupted"):
            raise KeyboardInterrupt("signal 15")
    assert read_status(run_dir)["status"] == RunStatus.INTERRUPTED.value


def test_stage_records_duration_and_failure(tmp_path):
    run_dir = tmp_path / "run"
    with RunLogger(run_dir, "staged") as log:
        with log.stage("load_model"):
            pass
        with pytest.raises(RuntimeError):
            with log.stage("train"):
                raise RuntimeError("nope")
        log.set_status(RunStatus.COMPLETED)

    kinds = [e["kind"] for e in read_events(run_dir)]
    assert "stage_end" in kinds and "stage_failed" in kinds
    ended = next(e for e in read_events(run_dir) if e["kind"] == "stage_end")
    assert ended["stage"] == "load_model" and "duration_s" in ended


def test_is_complete_requires_both_status_and_evaluation(tmp_path):
    run_dir = tmp_path / "run"
    with RunLogger(run_dir, "x"):
        pass
    assert not is_complete(run_dir)               # status says completed, no eval yet
    (run_dir / "eval").mkdir(exist_ok=True)
    (run_dir / "eval" / "metrics.json").write_text("{}")
    assert is_complete(run_dir)


def test_atomic_write_leaves_no_partial_file(tmp_path):
    target = tmp_path / "status.json"
    write_json_atomic(target, {"a": 1})
    assert json.loads(target.read_text()) == {"a": 1}
    assert not list(tmp_path.glob("*.tmp"))


def test_jsonl_survives_a_truncated_final_line(tmp_path):
    path = tmp_path / "events.jsonl"
    writer = JsonlWriter(path)
    writer.write({"kind": "a"})
    writer.write({"kind": "b"})
    writer.close()
    with path.open("a") as fh:
        fh.write('{"kind": "trunca')      # what a killed process leaves behind
    assert [e["kind"] for e in read_events(tmp_path)] == ["a", "b"]


def test_json_encoder_handles_numpy_and_paths(tmp_path):
    import numpy as np

    target = tmp_path / "x.json"
    write_json_atomic(target, {"p": Path("/tmp/x"), "arr": np.array([1, 2]),
                               "scalar": np.float64(1.5)})
    payload = json.loads(target.read_text())
    assert payload["arr"] == [1, 2] and payload["scalar"] == 1.5


# --------------------------------------------------------------------------------------
# Recovery: OOM
# --------------------------------------------------------------------------------------

def test_oom_backoff_preserves_the_effective_batch_size():
    """This is the property that keeps a recovered run comparable to the others.

    Simply lowering the batch size after an OOM would change the optimisation problem,
    so the run that eventually succeeded would not be measuring the same thing.
    """
    seen = []

    def flaky(plan: BatchPlan) -> str:
        seen.append(plan.as_dict())
        if plan.per_device_batch_size > 2:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        return "ok"

    result, used = oom_backoff(flaky, BatchPlan(8, 4))
    assert result == "ok"
    assert used.per_device_batch_size == 2 and used.gradient_accumulation_steps == 16
    assert all(p["per_device_train_batch_size"] * p["gradient_accumulation_steps"] == 32
               for p in seen)


def test_oom_backoff_gives_up_below_batch_size_one():
    def always_oom(plan):
        raise RuntimeError("CUDA out of memory")

    with pytest.raises((PreflightError, RuntimeError)):
        oom_backoff(always_oom, BatchPlan(2, 1), max_reductions=5)


def test_non_oom_errors_are_not_retried():
    calls = []

    def buggy(plan):
        calls.append(1)
        raise ValueError("a real bug")

    with pytest.raises(ValueError):
        oom_backoff(buggy, BatchPlan(8, 4))
    assert len(calls) == 1, "a genuine bug must surface immediately, not after retries"


def test_oom_detection_covers_the_common_phrasings():
    assert is_oom(RuntimeError("CUDA out of memory. Tried to allocate"))
    assert is_oom(RuntimeError("CUBLAS_STATUS_ALLOC_FAILED"))
    assert not is_oom(ValueError("shape mismatch"))


# --------------------------------------------------------------------------------------
# Recovery: retries, NaN, checkpoints
# --------------------------------------------------------------------------------------

def test_retry_backs_off_only_on_transient_failures():
    attempts = {"n": 0}

    @retry(attempts=3, base_delay=0.0)
    def flaky():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise OSError("Connection reset by peer")
        return "ok"

    assert flaky() == "ok" and attempts["n"] == 3


def test_retry_reraises_a_genuine_bug_immediately():
    attempts = {"n": 0}

    @retry(attempts=4, base_delay=0.0)
    def buggy():
        attempts["n"] += 1
        raise KeyError("missing column")

    with pytest.raises(KeyError):
        buggy()
    assert attempts["n"] == 1


def test_transient_classification():
    assert is_transient(OSError("Read timed out"))
    assert is_transient(RuntimeError("HTTP 503 Service Unavailable"))
    assert not is_transient(ValueError("bad config"))


def test_nan_guard_stops_before_the_checkpoint_is_poisoned():
    guard = NanGuard(patience=1)
    guard.check(1.0, 1)
    with pytest.raises(NonFiniteLossError, match="last good checkpoint is"):
        guard.check(float("nan"), 2)


def test_nan_guard_tolerates_a_single_spike_when_patient():
    guard = NanGuard(patience=2)
    guard.check(float("inf"), 1)
    guard.check(0.5, 2)                    # recovered: the strike counter resets
    guard.check(float("nan"), 3)           # one strike again, below patience


def _make_checkpoint(base: Path, step: int, *, valid: bool = True) -> Path:
    path = base / f"checkpoint-{step}"
    path.mkdir(parents=True)
    if valid:
        (path / "trainer_state.json").write_text(json.dumps({"global_step": step}))
        (path / "adapter_model.safetensors").write_bytes(b"weights")
    else:
        (path / "adapter_model.safetensors.tmp").write_bytes(b"partial")
    return path


def test_resume_skips_a_checkpoint_that_was_being_written(tmp_path):
    _make_checkpoint(tmp_path, 100)
    _make_checkpoint(tmp_path, 200)
    _make_checkpoint(tmp_path, 300, valid=False)      # killed mid-write

    assert len(list_checkpoints(tmp_path)) == 3
    assert resume_point(tmp_path).name == "checkpoint-200"


def test_checkpoint_validation_requires_state_and_weights(tmp_path):
    incomplete = tmp_path / "checkpoint-10"
    incomplete.mkdir()
    (incomplete / "trainer_state.json").write_text("{}")
    assert not checkpoint_is_valid(incomplete)        # no weights
    (incomplete / "adapter_model.safetensors").write_bytes(b"w")
    assert checkpoint_is_valid(incomplete)
    (incomplete / "trainer_state.json").write_text("not json")
    assert not checkpoint_is_valid(incomplete)


def test_resume_returns_none_when_there_is_nothing_to_resume(tmp_path):
    assert resume_point(tmp_path) is None


def test_prune_keeps_the_newest_checkpoints(tmp_path):
    for step in (100, 200, 300):
        _make_checkpoint(tmp_path, step)
    removed = prune_checkpoints(tmp_path, keep=1)
    assert [p.name for p in removed] == ["checkpoint-100", "checkpoint-200"]
    assert [p.name for p in list_checkpoints(tmp_path)] == ["checkpoint-300"]


def test_cleanup_removes_only_invalid_checkpoints(tmp_path):
    _make_checkpoint(tmp_path, 100)
    _make_checkpoint(tmp_path, 200, valid=False)
    removed = cleanup_partial(tmp_path)
    assert [p.name for p in removed] == ["checkpoint-200"]
    assert [p.name for p in list_checkpoints(tmp_path)] == ["checkpoint-100"]


# --------------------------------------------------------------------------------------
# Preflight
# --------------------------------------------------------------------------------------

def test_preflight_reports_rather_than_raising_by_default(tmp_path):
    report = preflight(output_dir=tmp_path, min_vram_gb=1e9, min_disk_gb=0.0)
    assert not report.ok
    assert "cuda" in report.checks
    with pytest.raises(PreflightError, match="preflight failed"):
        report.raise_if_failed()


def test_preflight_checks_the_verbalizers(tmp_path):
    from sentalign.labels import TERNARY_PLUS_MIXED

    class Tok:
        def __init__(self, table):
            self.table = table

        def __call__(self, text, add_special_tokens=False):
            # Named `text` as transformers names it: callers pass it by keyword, because
            # a multimodal processor's first positional parameter is `images`.
            return {"input_ids": self.table[text]}

    good = Tok({" negative": [1], " neutral": [2], " positive": [3], " mixed": [4]})
    report = preflight(output_dir=tmp_path, min_disk_gb=0.0, tokenizer=good,
                       label_space=TERNARY_PLUS_MIXED)
    assert report.checks["verbalizers"][0]

    bad = Tok({" negative": [1], " neutral": [2, 9], " positive": [3], " mixed": [4]})
    report = preflight(output_dir=tmp_path, min_disk_gb=0.0, tokenizer=bad,
                       label_space=TERNARY_PLUS_MIXED)
    assert not report.checks["verbalizers"][0]


def test_preflight_summary_is_human_readable(tmp_path):
    report = PreflightReport()
    report.add("disk", True, "500 GB free")
    report.add("cuda", False, "no device")
    text = report.summary()
    assert "[ok  ] disk" in text and "[FAIL] cuda" in text


# --------------------------------------------------------------------------------------
# Import order
# --------------------------------------------------------------------------------------

def _probe_in_subprocess(body: str, preamble: str = "") -> str:
    """Run a snippet in a fresh interpreter and return its stdout.

    A subprocess is the only honest way to test import order: once pytest has loaded a
    package, an in-process assertion about what loads it cannot fail.
    """
    import subprocess
    import sys

    script = (preamble + "\nimport sys, warnings\nsys.path.insert(0, 'src')\n"
              "warnings.simplefilter('ignore')\n" + body)
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                         cwd=str(Path(__file__).resolve().parents[1]), timeout=300)
    assert out.returncode == 0, f"probe failed:\n{out.stdout}\n{out.stderr}"
    return out.stdout


#: Import-order recorder, installed before anything else in the probe subprocess.
#:
#: The invariant is **ordering, not absence**. Unsloth imports trl, transformers, and peft
#: itself, because patching them requires importing them, so asserting that trl is absent
#: after Unsloth has loaded is simply wrong. What breaks the optimisation is trl arriving
#: *first*. An earlier version of this test asserted absence and failed on a machine where
#: Unsloth was installed, which is precisely the machine it was meant to protect.
_ORDER_RECORDER = """
import sys
_order = []


class _Recorder:
    def find_spec(self, fullname, path=None, target=None):
        top = fullname.split('.')[0]
        if top not in _order:
            _order.append(top)
        return None          # record only; let the real finders do the work


sys.meta_path.insert(0, _Recorder())
"""

# The recorder sees every *lookup*, including ones that resolve to nothing: probing
# whether a package exists (importlib.util.find_spec, importlib.metadata.version) goes
# through the meta path without importing anything. Intersecting with sys.modules keeps
# only modules that actually loaded, while _order supplies the sequence.
_ORDER_VERDICT = """
loaded = [m for m in _order if m in sys.modules]
patched = [m for m in ('trl', 'transformers', 'peft') if m in loaded]
if not patched:
    print('CLEAN nothing-patched-was-imported')
elif 'unsloth' not in loaded:
    print('BAD imported-without-unsloth', patched)
elif all(loaded.index('unsloth') < loaded.index(m) for m in patched):
    print('CLEAN unsloth-first', patched)
else:
    early = [m for m in patched if loaded.index(m) < loaded.index('unsloth')]
    print('BAD imported-before-unsloth', early)
"""


#: Every entry point that reports package versions or runs before training. Each is a
#: candidate for the defect that reading a version is what loads a package, in the wrong
#: order. Adding a new one here is cheaper than rediscovering the bug from a warning.
VERSION_REPORTERS = [
    ("collect_environment",
     "from sentalign.runlog import collect_environment; collect_environment()"),
    ("preflight",
     "from sentalign.recovery import preflight; "
     "preflight(output_dir='/tmp', min_disk_gb=0.0, min_vram_gb=0.0)"),
    ("package_version",
     "from sentalign.runlog import package_version; "
     "[package_version(n) for n in ('trl', 'transformers', 'peft', 'torch')]"),
    ("environment_manifest",
     "from sentalign.config import environment_manifest; environment_manifest()"),
    ("cli_import", "import sentalign.cli"),
]


@pytest.mark.parametrize("name,body", VERSION_REPORTERS,
                         ids=[n for n, _ in VERSION_REPORTERS])
def test_patched_packages_are_never_imported_before_unsloth(name, body):
    """Unsloth must reach the interpreter before trl, transformers, or peft.

    Importing any of those first means Unsloth's trainer patches never apply, and the only
    symptom is a run slower and hungrier than its own manifest claims, with the manifest
    still recording the Unsloth version.

    This defect was found twice by inspection (``collect_environment``, then ``preflight``)
    and the fix is verified here by recording the actual order of first import in a fresh
    interpreter. Passing vacuously when the packages are absent is expected on a machine
    without the training stack; the static check below covers that case.
    """
    out = _probe_in_subprocess(body + _ORDER_VERDICT, preamble=_ORDER_RECORDER)
    assert "CLEAN" in out, f"{name}: {out.strip().splitlines()[-1]}"


def test_package_version_prefers_the_richer_loaded_string():
    """An already-imported module carries more than distribution metadata does.

    torch reports 2.11.0+cu130 through its module and 2.11.0 through metadata, and the
    CUDA build is worth keeping in a run manifest, so a loaded module wins.
    """
    import sys
    import types

    from sentalign.runlog import package_version

    fake = types.ModuleType("sentalign_fake_pkg")
    fake.__version__ = "1.2.3+localbuild"
    sys.modules["sentalign_fake_pkg"] = fake
    try:
        assert package_version("sentalign_fake_pkg") == "1.2.3+localbuild"
    finally:
        del sys.modules["sentalign_fake_pkg"]

    assert package_version("sentalign_definitely_absent") is None


def test_no_module_scope_imports_of_patched_packages():
    """Only modeling.py may touch the patched packages, and it imports unsloth first.

    A module-scope `import trl` anywhere else would fire the moment the package is
    imported, before modeling.py has a chance to load unsloth.
    """
    import re

    src = Path(__file__).resolve().parents[1] / "src" / "sentalign"
    pattern = re.compile(r"^(?:import|from)\s+(trl|transformers|peft)\b", re.MULTILINE)
    offenders = []
    for path in src.rglob("*.py"):
        text = path.read_text()
        for match in pattern.finditer(text):
            line = text[:match.start()].count("\n") + 1
            offenders.append(f"{path.relative_to(src)}:{line} {match.group(0)}")
    assert not offenders, (
        "module-scope imports of unsloth-patched packages:\n  " + "\n  ".join(offenders))


def test_no_dunder_import_of_patched_packages_anywhere():
    """``__import__(name)`` in a loop over package names is how this bug arises twice.

    Both instances looked innocent: a list of packages, a loop, a version read. The
    pattern itself is the hazard, so it is banned outright rather than reviewed
    case by case.
    """
    import re

    src = Path(__file__).resolve().parents[1] / "src" / "sentalign"
    offenders = []
    for path in src.rglob("*.py"):
        text = path.read_text()
        for match in re.finditer(r"__import__\s*\(", text):
            line = text[:match.start()].count("\n") + 1
            offenders.append(f"{path.relative_to(src)}:{line}")
    assert not offenders, (
        "__import__ used in:\n  " + "\n  ".join(offenders)
        + "\nUse sentalign.runlog.package_version, which has no import side effects.")


def test_cli_import_does_not_load_patched_packages():
    """`sentalign <anything>` must not pull in trl before unsloth."""
    import subprocess
    import sys

    probe = (
        "import sys; sys.path.insert(0, 'src'); "
        "import warnings; warnings.simplefilter('ignore'); "
        "import sentalign.cli; "
        "early = [m for m in ('trl', 'transformers', 'peft') if m in sys.modules]; "
        "unsloth_first = 'unsloth' in sys.modules or not early; "
        "print('OK' if unsloth_first else f'BAD {early}')"
    )
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True,
                         cwd=str(Path(__file__).resolve().parents[1]))
    assert "OK" in out.stdout, f"{out.stdout}\n{out.stderr}"


# --------------------------------------------------------------------------------------
# Code provenance without version control
# --------------------------------------------------------------------------------------

def test_code_fingerprint_is_stable_and_change_sensitive(tmp_path):
    """Provenance must not depend on version control being set up.

    A run has to record which code produced it. A commit answers that only when the
    project is under git and the working copy is clean; hashing the source tree answers
    it unconditionally and cannot be defeated by uncommitted edits.
    """
    from sentalign.runlog import code_fingerprint

    (tmp_path / "src").mkdir()
    (tmp_path / "scripts").mkdir()
    module = tmp_path / "src" / "thing.py"
    module.write_text("x = 1\n")
    (tmp_path / "scripts" / "run.sh").write_text("echo hi\n")

    first = code_fingerprint(tmp_path)
    assert first["n_files"] == 2
    assert code_fingerprint(tmp_path)["digest"] == first["digest"], "must be stable"

    module.write_text("x = 2\n")
    assert code_fingerprint(tmp_path)["digest"] != first["digest"], "must see edits"

    module.write_text("x = 1\n")
    assert code_fingerprint(tmp_path)["digest"] == first["digest"], "must be reversible"


def test_code_fingerprint_notices_a_rename(tmp_path):
    """Path is hashed before content, so moving a file changes the digest."""
    from sentalign.runlog import code_fingerprint

    (tmp_path / "src").mkdir()
    original = tmp_path / "src" / "a.py"
    original.write_text("payload\n")
    before = code_fingerprint(tmp_path)["digest"]

    original.rename(tmp_path / "src" / "b.py")
    assert code_fingerprint(tmp_path)["digest"] != before


def test_code_fingerprint_ignores_caches_and_data(tmp_path):
    """Only source counts. Build artefacts and caches must not perturb the digest."""
    from sentalign.runlog import code_fingerprint

    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("payload\n")
    before = code_fingerprint(tmp_path)["digest"]

    (tmp_path / "src" / "__pycache__").mkdir()
    (tmp_path / "src" / "__pycache__" / "a.cpython-312.pyc").write_bytes(b"\x00compiled")
    (tmp_path / "src" / "notes.md").write_text("prose\n")
    assert code_fingerprint(tmp_path)["digest"] == before


def test_environment_records_provenance_without_git():
    """collect_environment must carry a code version even where git_sha is None."""
    from sentalign.runlog import collect_environment

    env = collect_environment()
    assert env.get("code_fingerprint"), "no provenance recorded"
    assert isinstance(env.get("code_files"), int) and env["code_files"] > 0


def test_resume_refuses_to_cross_a_configuration_change(tmp_path):
    """A checkpoint from a different config must not be inherited.

    The motivating case: a smoke test trained with an adapter configuration that reached
    a fraction of the decoder. The configuration was then corrected, but the checkpoint
    was still structurally valid, so resume would have carried the defect forward under
    the corrected config's name.
    """
    from sentalign.recovery import config_changed, resume_point

    (tmp_path / "config.json").write_text(json.dumps({"config_hash": "aaaa1111"}))
    _make_checkpoint(tmp_path, 500)

    assert resume_point(tmp_path, "aaaa1111").name == "checkpoint-500"   # same config
    assert resume_point(tmp_path, "bbbb2222") is None                    # changed
    assert config_changed(tmp_path, "bbbb2222") == "aaaa1111"
    assert config_changed(tmp_path, "aaaa1111") is None

    # No hash to compare against: fall back to the structural check rather than refusing.
    assert resume_point(tmp_path, None).name == "checkpoint-500"


def test_config_changed_tolerates_a_missing_or_corrupt_config(tmp_path):
    from sentalign.recovery import config_changed

    assert config_changed(tmp_path, "abc") is None          # no config.json
    (tmp_path / "config.json").write_text("{ not json")
    assert config_changed(tmp_path, "abc") is None


def test_staleness_check_must_precede_writing_the_new_config():
    """The check has to read the previous attempt's hash, not the one just written.

    Saving the config first makes ``config_changed`` compare the new hash against itself,
    so it never fires and a checkpoint from a different configuration is resumed. That is
    a dead check that looks alive, which is worse than no check.
    """
    import re

    source = (Path(__file__).resolve().parents[1]
              / "src" / "sentalign" / "train" / "driver.py").read_text()
    body = source[source.index("def run_training("):]
    save_at = body.index("logger.save_config(")
    check_at = body.index("config_changed(")
    assert check_at < save_at, (
        "config_changed() runs after save_config(), so it compares the new hash against "
        "itself and can never detect a change")


def test_incompatible_checkpoint_falls_back_to_a_clean_start():
    """A checkpoint whose optimizer state no longer fits must cost the run, not the slot."""
    from sentalign.recovery import is_incompatible_checkpoint, train_with_resume_fallback

    observed = ("loaded state dict contains a parameter group that doesn't match "
                "the size of optimizer's group")
    assert is_incompatible_checkpoint(ValueError(observed))
    assert not is_incompatible_checkpoint(ValueError("some other problem"))

    attempts, notified = [], []

    def train(ckpt):
        attempts.append(ckpt)
        if ckpt is not None:
            raise ValueError(observed)
        return "trained"

    result, used = train_with_resume_fallback(
        train, Path("/tmp/checkpoint-500"),
        on_fallback=lambda c, e: notified.append(c))
    assert result == "trained" and used is None
    assert attempts == [Path("/tmp/checkpoint-500"), None]
    assert notified == [Path("/tmp/checkpoint-500")]


def test_unrelated_training_errors_still_propagate():
    from sentalign.recovery import train_with_resume_fallback

    def train(ckpt):
        raise ValueError("a genuine bug in the loss")

    with pytest.raises(ValueError, match="genuine bug"):
        train_with_resume_fallback(train, Path("/tmp/checkpoint-1"))


# --------------------------------------------------------------------------------------
# Log noise
# --------------------------------------------------------------------------------------

def test_only_the_lazy_alias_message_is_treated_as_noise():
    """The filter matches one sentence. Everything else transformers says still lands."""
    from sentalign.modeling import is_lazy_alias_noise

    assert is_lazy_alias_noise(
        "Accessing `causal_conv1d_fn` from `.models.vitpose.image_processing_vitpose`. "
        "Returning `causal_conv1d_fn` instead. Behavior may be different and this alias "
        "will be removed in future versions.")
    assert not is_lazy_alias_noise(
        "Both `max_new_tokens` (=8) and `max_length`(=128000) seem to have been set.")
    assert not is_lazy_alias_noise("Some weights were not initialized")
    assert not is_lazy_alias_noise("")


@pytest.mark.skipif("transformers" not in __import__("sys").modules
                    and __import__("importlib.util", fromlist=["util"])
                    .find_spec("transformers") is None,
                    reason="transformers not installed")
def test_the_filter_drops_child_logger_records_and_keeps_the_rest():
    """The records come from child loggers, which bypass the parent's own filters."""
    import logging

    import transformers  # noqa: F401  (the filter needs the library loaded)

    from sentalign.modeling import silence_lazy_alias_warnings

    silence_lazy_alias_warnings()
    root = logging.getLogger("transformers")
    assert root.handlers, "nothing to attach a filter to"

    seen: list[str] = []

    class _Capture(logging.Handler):
        def emit(self, record):
            seen.append(record.getMessage())

    child = logging.getLogger("transformers.models.vitpose.image_processing_vitpose")
    capture = _Capture()
    for existing in root.handlers[0].filters:
        capture.addFilter(existing)
    root.addHandler(capture)
    try:
        child.warning("Accessing `x`. Returning `x` instead. Behavior may be different "
                      "and this alias will be removed in future versions.")
        child.warning("a real warning about your model")
    finally:
        root.removeHandler(capture)      # the logger is shared process state

    assert seen == ["a real warning about your model"]


# --------------------------------------------------------------------------------------
# Multimodal processors arriving where a tokenizer is expected
# --------------------------------------------------------------------------------------

class _FakeTokenizer:
    def encode(self, text):
        return [1, 2, 3]

    def __call__(self, text=None, **kwargs):
        return {"input_ids": [[1, 2, 3]]}


class _FakeProcessor:
    """Shaped like Qwen3VLProcessor: images first, text second."""

    def __init__(self):
        self.tokenizer = _FakeTokenizer()
        self.image_processor = object()

    def __call__(self, images=None, text=None, videos=None, **kwargs):
        if images is not None:
            raise ValueError(f"Incorrect image source. Got {images}")
        return {"input_ids": [[1, 2, 3]]}


def test_a_multimodal_processor_is_unwrapped_to_its_tokenizer():
    """Qwen3.5 ships a processor, and a positional text argument lands in `images`.

    The symptom is `PIL.UnidentifiedImageError: cannot identify image file` raised from
    inside the verbalizer resolution, which names neither the tokenizer nor the model.
    """
    from sentalign.modeling import text_tokenizer

    processor = _FakeProcessor()
    info: dict = {}
    unwrapped = text_tokenizer(processor, info)

    assert unwrapped is processor.tokenizer
    assert info["tokenizer_unwrapped_from"] == "_FakeProcessor"
    # A plain tokenizer is returned untouched, and nothing is recorded.
    plain_info: dict = {}
    plain = _FakeTokenizer()
    assert text_tokenizer(plain, plain_info) is plain
    assert plain_info == {}


def test_verbalizers_resolve_through_a_processor_even_unwrapped():
    """Belt and braces: `text=` keeps a processor working if one ever reaches a call site."""
    from sentalign.labels import TERNARY, resolve_verbalizers

    class _SingleTokenProcessor(_FakeProcessor):
        # Fixed ids, not hash(text): str hashing is seed-randomized per process, and a
        # collision between two surfaces trips the duplicate-id check about one run in
        # three hundred, an unreproducible red build.
        TABLE = {" negative": 11, " neutral": 12, " positive": 13}

        def __call__(self, images=None, text=None, videos=None, **kwargs):
            if images is not None:
                raise ValueError(f"Incorrect image source. Got {images}")
            return {"input_ids": [self.TABLE[text]]}

    table = resolve_verbalizers(_SingleTokenProcessor(), TERNARY)
    assert table.single_token


def test_kernel_fast_paths_reads_the_loaded_modules_not_the_package_list():
    """An installed kernel package is not the same as a kernel the GPU can run.

    causal-conv1d installs on this stack and Unsloth then disables it as incompatible
    with an RTX 3090, so package metadata says one thing and the forward pass does
    another. The module flag is what the forward pass branches on.
    """
    import sys
    import types

    from sentalign.modeling import kernel_fast_paths

    fake_slow = types.ModuleType("transformers.models.fakearch.modeling_fakearch")
    fake_slow.is_fast_path_available = False
    fake_fast = types.ModuleType("transformers.models.otherarch.modeling_otherarch")
    fake_fast.is_fast_path_available = True
    unrelated = types.ModuleType("transformers.models.thirdarch.configuration_thirdarch")
    unrelated.is_fast_path_available = True          # not a modeling module

    for name, module in ((fake_slow.__name__, fake_slow), (fake_fast.__name__, fake_fast),
                         (unrelated.__name__, unrelated)):
        sys.modules[name] = module
    try:
        paths = kernel_fast_paths()
        assert paths["fakearch"] is False
        assert paths["otherarch"] is True
        assert "thirdarch" not in paths
    finally:
        for name in (fake_slow.__name__, fake_fast.__name__, unrelated.__name__):
            sys.modules.pop(name, None)


# --------------------------------------------------------------------------------------
# The subprocess boundary between the grid runner and the CLI
# --------------------------------------------------------------------------------------

def _run_grid_module():
    import importlib.util

    path = Path(__file__).resolve().parent.parent / "scripts" / "02_run_grid.py"
    spec = importlib.util.spec_from_file_location("run_grid_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_planned_command_reproduces_its_run_id_through_the_cli():
    """The runner talks to the trainer only through argv, so anything the plan encodes
    must survive that round trip. The cspo ablation did not: ``variant`` never entered
    the command line, the subprocess recomputed the run_id without it, all five variants
    collapsed onto the main-grid cspo directory, and the study reported success while
    producing nothing. This drives every planned run through the real parser and the
    real config builder and requires the identical run_id back.
    """
    from sentalign.cli import _build_config, build_parser
    from sentalign.plan import full_plan

    grid = _run_grid_module()
    parser = build_parser()
    mismatches = []
    for run in full_plan():
        if run.study == "baselines":
            continue          # routed through `sentalign baselines`, not `train`
        planned = run.to_config()
        cmd = grid.build_command(run, planned, Path("data/build"), Path("runs"))
        args = parser.parse_args([str(part) for part in cmd[3:]])
        rebuilt = _build_config(args)
        if rebuilt.run_id != planned.run_id:
            mismatches.append((planned.run_id, rebuilt.run_id))
    assert not mismatches, (
        f"{len(mismatches)} planned runs write to a different directory than the plan "
        f"believes, first: {mismatches[0]}")


def test_status_reports_training_hours_from_the_manifest(tmp_path, capsys):
    """Evaluation opens a second RunLogger on the run directory and its final status
    write overwrites elapsed_s with the eval phase's duration, so a status table built
    from elapsed_s under-reported the programme by roughly 3x."""
    import argparse
    import json as _json

    from sentalign.cli import cmd_status

    run = tmp_path / "main__x__sft__eps0.0__tau0.2__n8000__s13"
    run.mkdir()
    (run / "status.json").write_text(_json.dumps(
        {"status": "completed", "elapsed_s": 400.0}))          # the eval overwrite
    (run / "manifest.json").write_text(_json.dumps({"gpu_hours": 0.83}))

    bare = tmp_path / "main__x__sft__eps0.0__tau0.2__n8000__s21"
    bare.mkdir()
    (bare / "status.json").write_text(_json.dumps(
        {"status": "failed", "elapsed_s": 360.0}))             # no manifest: fallback

    live = tmp_path / "main__x__dpo__eps0.0__tau0.2__n8000__s34"
    live.mkdir()
    (live / "status.json").write_text(_json.dumps(
        {"status": "running", "elapsed_s": 1080.0}))           # 0.3 h and counting

    (tmp_path / "unsloth_compiled_cache").mkdir()      # a library cache, not a run

    cmd_status(argparse.Namespace(runs=tmp_path, only=None))
    out = capsys.readouterr().out
    assert "unsloth_compiled_cache" not in out and "unknown" not in out, out
    assert "0.83" in out, out
    total = [l for l in out.splitlines() if "total measured" in l][0]
    # 0.83 from the one completed run. Counting the others too would read 1.2: the
    # failed run produced nothing, and the running one has not finished producing it.
    assert "0.8" in total and "1.2" not in total, total
    assert "0.11" not in out, "the eval-phase elapsed_s must not masquerade as GPU-h"
    assert "0.1" in out, "runs without a manifest fall back to status elapsed_s"


def test_the_code_fingerprint_ignores_packaging_metadata(tmp_path):
    """The digest must answer "which code produced this run", nothing else.

    An editable install writes `src/<pkg>.egg-info/*.txt`, which lived under a
    fingerprinted root with a fingerprinted suffix. The consequences were all wrong in
    the same direction: `SOURCES.txt` lists every distribution file, so adding a test
    moved the digest; `pip install -e .` rewrote it with no code change; and a checkout
    without an editable install hashed differently from byte-identical source on another
    machine, which is precisely the comparison the digest exists to support.
    """
    from sentalign.runlog import code_fingerprint

    (tmp_path / "src" / "pkg").mkdir(parents=True)
    (tmp_path / "scripts").mkdir()
    source = tmp_path / "src" / "pkg" / "mod.py"
    source.write_text("VALUE = 1\n")

    baseline = code_fingerprint(tmp_path)
    assert baseline["n_files"] == 1

    egg = tmp_path / "src" / "pkg.egg-info"
    egg.mkdir()
    (egg / "SOURCES.txt").write_text("src/pkg/mod.py\ntests/test_mod.py\n")
    (egg / "top_level.txt").write_text("pkg\n")
    after_install = code_fingerprint(tmp_path)

    assert after_install["digest"] == baseline["digest"], (
        "an editable install must not move the digest")
    assert after_install["n_files"] == 1

    (egg / "SOURCES.txt").write_text("src/pkg/mod.py\n")      # a test file was removed
    assert code_fingerprint(tmp_path)["digest"] == baseline["digest"]

    source.write_text("VALUE = 2\n")                          # real source change
    assert code_fingerprint(tmp_path)["digest"] != baseline["digest"], (
        "a source change must still move the digest")


def test_text_only_unregistration_is_confined_to_training(monkeypatch):
    """Unsloth loads Qwen3.5 *through* AutoModelForImageTextToText, so unregistering the
    model type for an inference load breaks the next load in the same process. The
    prompting baselines load every model in one interpreter, and qwen prompt5 died with
    "Unrecognized configuration class" right after qwen prompt0 succeeded.
    """
    import sentalign.modeling as modeling
    from transformers.models.auto import modeling_auto

    mapping = modeling_auto.MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES
    mapping["sentalign_probe_vl"] = "ProbeForConditionalGeneration"
    calls: list[bool] = []

    class _Config:
        model_type = "sentalign_probe_vl"

    class _Model:
        config = _Config()

    monkeypatch.setattr(modeling, "treat_as_text_only",
                        lambda model, info=None: calls.append(True))
    try:
        # The guard under test is the call site, so assert on the source of the branch.
        import inspect

        source = inspect.getsource(modeling.load_causal_lm)
        train_branch = source.index("if for_training:")
        assert source.index("treat_as_text_only(model, info)") > train_branch, (
            "unregistering before the for_training guard breaks inference loads")
        assert "sentalign_probe_vl" in mapping, "an inference load must leave it alone"
    finally:
        mapping.pop("sentalign_probe_vl", None)


def test_every_planned_run_id_parses_for_aggregation():
    """The table builder matched no run id at all, so the whole programme aggregated to
    zero rows and printed "no completed runs with evaluations" after 103 GPU-hours.

    Two causes, both silent: the pattern had no `n{subsample}` segment, and `[^_]+` for
    the objective cannot match `sft_soft`, `encoder_hard`, or `encoder_soft`. A parser
    that matches nothing looks exactly like an empty results directory.
    """
    import importlib.util

    from sentalign.plan import full_plan

    path = Path(__file__).resolve().parent.parent / "scripts" / "03_make_tables.py"
    spec = importlib.util.spec_from_file_location("make_tables_run_ids", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    planned = [run.to_config().run_id for run in full_plan()]
    unparsed = [rid for rid in planned if module.parse_run_id(rid) is None]
    assert not unparsed, f"{len(unparsed)} of {len(planned)} unparsed, e.g. {unparsed[:3]}"

    meta = module.parse_run_id("main__lfm-1.2b__sft_soft__eps0.0__tau0.4__n32000__s21")
    assert meta["objective"] == "sft_soft"      # underscores inside the objective
    assert meta["n"] == "32000" and meta["seed"] == "21" and meta["tau"] == "0.4"

    # `n` must reach the index key, or the four scaling sizes share one slot per seed.
    keys = {(meta["model"], meta["objective"], meta["eps"], meta["tau"], meta["n"])}
    other = module.parse_run_id("main__lfm-1.2b__sft_soft__eps0.0__tau0.4__n3000__s21")
    keys.add((other["model"], other["objective"], other["eps"], other["tau"], other["n"]))
    assert len(keys) == 2, "two sizes must not collapse onto one index entry"


def test_the_nli_study_is_isolated_from_the_sentiment_runs():
    """The task is deliberately not part of the run_id, so the study name is the only
    thing keeping 50 NLI runs out of the sentiment directories. build_command forgot to
    emit --name at first, which pointed every NLI run at an existing sentiment run: it
    would have read as already complete and the study would have produced nothing, the
    same failure the --variant flag had."""
    from sentalign.plan import full_plan, nli_study

    nli = list(nli_study())
    assert nli, "the NLI study is registered but emits no runs"
    for run in nli:
        assert run.task == "nli" and run.name == "nli"
        cfg = run.to_config()
        assert cfg.run_id.startswith("nli__")
        assert cfg.data.label_space == "nli3"
        assert cfg.data.build_dir.name == "build_nli"
        assert "sst_dev_validated" not in cfg.eval.eval_sets
        assert cfg.eval.transfer_sets == ()

    sentiment_ids = {r.to_config().run_id for r in full_plan() if r.task == "sentiment"}
    nli_ids = {r.to_config().run_id for r in nli}
    assert not (sentiment_ids & nli_ids), "an NLI run would overwrite a sentiment run"


def test_a_task_bundle_never_half_applies():
    """A task is a bundle precisely because applying half of it (an NLI label space
    against the sentiment build directory, say) would train on the wrong data while
    looking fine."""
    import pytest

    from sentalign.config import TASKS, ExperimentConfig, apply_task

    for task in TASKS:
        cfg = ExperimentConfig()
        apply_task(cfg, task)
    with pytest.raises(SystemExit, match="unknown task"):
        apply_task(ExperimentConfig(), "not-a-task")


def test_the_grid_evaluates_each_run_against_its_own_build_directory():
    """The runner passed its --data to every run and to every eval. For a task that
    carries its own build directory that silently pointed the NLI runs at the sentiment
    sets, which have neither the NLI label space nor the ChaosNLI eval files."""
    from pathlib import Path

    from sentalign.plan import main_grid, nli_study
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "grid", Path(__file__).resolve().parents[1] / "scripts" / "02_run_grid.py")
    grid = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(grid)

    nli = next(iter(nli_study()))
    _, cfg = grid.run_dir_for(nli, Path("runs"), Path("data/build"))
    assert cfg.data.build_dir == Path("data/build_nli"), (
        "the caller's --data overwrote the NLI task's own build directory")

    sentiment = next(iter(main_grid()))
    _, cfg = grid.run_dir_for(sentiment, Path("runs"), Path("data/build"))
    assert cfg.data.build_dir == Path("data/build"), "sentiment must still honour --data"


def test_the_confirmation_set_is_in_the_plan_and_names_the_directories_that_exist():
    """The manuscript quotes ``sentalign plan``, so a confirmation set launched from an
    ad-hoc chain is a run programme the plan does not contain: the run table and the runs
    can then disagree with nothing to catch it. The arm names are part of the claim, not
    cosmetic. ``pref_distributional_lambda=1.0`` renders as ``lambda1.0`` under the
    derived-variant rule, which would point the plan at directories that do not exist and
    that the analysis does not read.
    """
    from sentalign.plan import full_plan

    runs = [r for r in full_plan() if r.study == "regulariser confirmation"]
    configs = {r.to_config().run_id: r.to_config() for r in runs}
    assert len(configs) == 25, "five arms by five seeds"
    for arm in ("sft", "kto", "kto-dreg1.0", "rdpo", "rdpo-dreg1.0"):
        rid = f"main__smollm3-3b__{arm}__eps0.0__tau0.2__n8000__s13"
        assert rid in configs, f"the plan does not emit {rid}"

    dreg = configs["main__smollm3-3b__kto-dreg1.0__eps0.0__tau0.2__n8000__s13"].train
    twin = configs["main__smollm3-3b__kto__eps0.0__tau0.2__n8000__s13"].train
    assert (dreg.pref_distributional_lambda, dreg.pref_regularizer) == (1.0, "soft")
    assert twin.pref_distributional_lambda == 0.0, "the twin must be the plain objective"
    assert dreg.objective == twin.objective == "kto"


def test_the_analysis_and_the_plan_declare_the_same_confirmation_arms():
    """The statistics refuse to shrink a family with missing data, which only helps if the
    runs behind it were planned. An arm added to the analysis and not to the plan stops
    the whole confirmation at the last step, after the GPU time is spent."""
    import importlib.util

    from sentalign.plan import full_plan

    path = Path(__file__).resolve().parent.parent / "scripts" / "06_regulariser_stats.py"
    spec = importlib.util.spec_from_file_location("regulariser_stats_plan", path)
    stats = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stats)

    planned = {rid.split("__")[2] for rid in
               (r.to_config().run_id for r in full_plan() if r.model == "smollm3-3b")}
    wanted = set(stats.PLANNED["smollm3-3b"])
    wanted |= {stats.twin(arm) for arm in stats.PLANNED["smollm3-3b"]}
    wanted.add(stats.BASELINE)
    assert wanted <= planned, f"the analysis reads arms the plan never runs: {wanted - planned}"


def test_the_nli_replication_arms_are_in_the_plan_and_do_not_shadow_the_nli_study():
    """The NLI degradation arms were launched outside the plan, like the third model family
    was. Declaring them is what keeps the run table and the runs together; declaring them
    twice would point two studies at one directory, which full_plan refuses."""
    from sentalign.plan import NLI_REG_ARMS, full_plan, nli_study

    runs = [r for r in full_plan() if r.study == "nli regulariser"]
    ids = {r.to_config().run_id for r in runs}
    assert len(ids) == 5 * len(NLI_REG_ARMS)
    for arm in ("kto", "kto-dreg1.0", "rdpo", "rdpo-dreg1.0"):
        assert f"nli__lfm-1.2b__{arm}__eps0.0__tau0.2__n8000__s13" in ids

    assert not (ids & {r.to_config().run_id for r in nli_study()}), "two studies, one directory"
    for run in runs:
        cfg = run.to_config()
        assert cfg.data.label_space == "nli3" and cfg.run_id.startswith("nli__")
    regularised = next(r.to_config() for r in runs
                       if "kto-dreg1.0" in r.to_config().run_id)
    assert regularised.train.pref_distributional_lambda == 1.0
    assert regularised.train.pref_regularizer == "soft"


def test_the_landscape_completion_cannot_move_a_confirmatory_p_value():
    """Arms run after the endpoint was declared are safe only if they enter no confirmatory
    family. Every family in the analysis compares a regularised arm against its twin or
    against SFT, so an arm with no regulariser adds no test; a regularised one would
    enlarge P1, P2 and P3 and re-correct every comparison in the paper.

    The guard is the property, not the arm list: adding ``ipo`` with the regulariser to
    ``LANDSCAPE_ARMS`` has to fail here.
    """
    import importlib.util

    from sentalign.plan import LANDSCAPE_MODEL, full_plan

    runs = [r for r in full_plan() if r.study == "landscape completion"]
    assert runs, "the study is not in the plan"
    for run in runs:
        train = run.to_config().train
        assert not train.pref_distributional_lambda, (
            f"{run.to_config().run_id} carries the regulariser, so it joins the "
            "confirmatory families and re-corrects the whole paper")

    path = Path(__file__).resolve().parent.parent / "scripts" / "06_regulariser_stats.py"
    spec = importlib.util.spec_from_file_location("regulariser_stats_landscape", path)
    stats = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stats)

    # The analysis builds its families from PLANNED, so the family sizes are a function of
    # that mapping alone. The new arms must not appear in it.
    planned_regularised = set(stats.PLANNED[LANDSCAPE_MODEL])
    emitted = {r.to_config().run_id.split("__")[2] for r in runs}
    assert not (emitted & planned_regularised), (
        f"{emitted & planned_regularised} would be counted twice")
    assert all("dreg" not in arm for arm in emitted)

    repair = stats.repair_plan(list(stats.MODELS))
    assert len(repair[LANDSCAPE_MODEL]) == 2, (
        "the third family declares two regularised comparisons; the landscape arms must "
        "leave that number alone")


def test_the_two_smollm3_studies_do_not_claim_one_directory():
    """full_plan refuses duplicate run ids, but the refusal is only exercised if both
    studies are in it, and the confirmation set and the landscape completion share a model,
    a task and a seed list."""
    from sentalign.plan import full_plan

    confirm = {r.to_config().run_id for r in full_plan() if r.study == "regulariser confirmation"}
    landscape = {r.to_config().run_id for r in full_plan() if r.study == "landscape completion"}
    assert confirm and landscape
    assert not (confirm & landscape)
    assert len(landscape) == 35, "seven arms by five seeds"


def test_the_plan_re_costs_itself_from_the_programme_ledger(tmp_path):
    """``measured_throughput`` existed and nothing called it with a directory, so the
    estimate stayed the a-priori one however many hours the machine had already measured.
    The property is that a ledger showing the machine is slower than assumed must raise the
    estimate, and by the ratio, not by a guess.
    """
    from sentalign.config import THROUGHPUT_EXAMPLES_PER_S
    from sentalign.plan import full_plan, summarise

    runs = [r for r in full_plan() if r.study == "landscape completion"]
    model = runs[0].model
    planned_rate = THROUGHPUT_EXAMPLES_PER_S[model]

    ledger = tmp_path / "programme.jsonl"
    rows = []
    for i, run in enumerate(runs[:4]):
        rid = run.to_config().run_id
        seen, wall = 16_000, 16_000 / (planned_rate / 2.0)      # half the assumed rate
        (tmp_path / rid).mkdir(parents=True, exist_ok=True)
        (tmp_path / rid / "manifest.json").write_text(json.dumps(
            {"model": model, "n_train_records": seen / 2.0, "wallclock_s": wall}))
        rows.append(json.dumps({"run_id": rid, "ok": True, "measured_hours": wall / 3600}))
    ledger.write_text("\n".join(rows) + "\n")

    plain = summarise(runs)
    recosted = summarise(runs, tmp_path)
    assert "Re-costed from measured throughput" not in plain
    assert "Re-costed from measured throughput" in recosted

    import re as _re

    def total(text, marker):
        line = next(l for l in text.splitlines() if marker in l)
        return float(_re.search(r"(\d+\.\d+) GPU-h", line).group(1))

    doubled = total(recosted, "Re-costed")
    a_priori = float(next(l for l in plain.splitlines()
                          if l.startswith("total")).split()[-1])
    assert doubled == pytest.approx(a_priori * 2.0, rel=0.02), (
        "a machine running at half the assumed rate must double the estimate")


def test_the_plan_command_passes_the_runs_directory_through(tmp_path, capsys):
    """The re-costing is only reachable if the command hands the directory over, which is
    the part that was missing."""
    from sentalign.cli import build_parser

    args = build_parser().parse_args(["plan", "--studies", "landscape"])
    assert hasattr(args, "runs"), "the plan command cannot see a runs directory"
    assert args.runs == Path("runs")


def test_the_ipo_reg_completion_is_regularised_and_outside_the_declared_families():
    """The run has to be the regularised twin of the IPO arm that exists, land in the
    directory the landscape table reads, and leave the analysis's declared families alone:
    it was chosen after the endpoint, so it is reported beside them, not inside them."""
    import importlib.util

    from sentalign.plan import full_plan

    runs = [r for r in full_plan() if r.study == "ipo-reg completion"]
    ids = {r.to_config().run_id for r in runs}
    assert ids == {f"main__smollm3-3b__ipo-dreg1.0__eps0.0__tau0.2__n8000__s{s}"
                   for s in (13, 21, 34, 55, 89)}
    for run in runs:
        train = run.to_config().train
        assert (train.objective, train.pref_distributional_lambda,
                train.pref_regularizer) == ("ipo", 1.0, "soft")

    path = Path(__file__).resolve().parent.parent / "scripts" / "06_regulariser_stats.py"
    spec = importlib.util.spec_from_file_location("regulariser_stats_iporeg", path)
    stats = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stats)
    assert "ipo-dreg1.0" not in stats.PLANNED["smollm3-3b"]
    assert len(stats.repair_plan(list(stats.MODELS))["smollm3-3b"]) == 2
