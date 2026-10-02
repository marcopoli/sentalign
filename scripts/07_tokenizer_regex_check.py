#!/usr/bin/env python3
"""Does transformers' Mistral-regex fix change how this protocol tokenizes anything?

transformers 5.5 warns, on every SmolLM3 run directory, that the saved tokenizer carries
the old Mistral pretokenizer regex and that loading it without ``fix_mistral_regex=True``
"will lead to incorrect tokenization". Two things follow, and only one of them is a
question about our results.

Turning the flag on is not a repair: the models were trained under the tokenizer as
saved, so a flag that changed the token stream at evaluation would introduce exactly the
train/eval mismatch it appears to prevent. What the paper needs is a statement of whether
the flag makes any difference to the strings this protocol actually encodes, which are
the evaluation prompts and the four verbalizer surfaces, not of whether the warning
exists.

So both are encoded under a tokenizer loaded each way and the ids are compared exactly.
Identical ids mean the warning is inert here and every number already computed stands.
Different ids mean the affected runs were trained on a token stream the fixed tokenizer
would not reproduce, which is a retraining decision and not a footnote, so the script
exits non-zero rather than printing a note nobody reads. A transformers that does not
know the flag at all is reported as unsupported: it is not evidence of agreement, and
the one failure worth guarding against here is a check that passes without checking.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sentalign.labels import TERNARY_PLUS_MIXED, build_prompt, resolve_verbalizers  # noqa: E402

FLAG = "fix_mistral_regex"
#: The sentence transformers logs for a tokenizer it considers affected.
WARNING_MARK = "incorrect regex pattern"


def _default_loader(path: str, **kwargs):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(path, trust_remote_code=True, **kwargs)


class _WarningCapture(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record) -> None:
        self.messages.append(record.getMessage())


def load_plain(path, loader):
    """Load the tokenizer as the harness does, and note whether transformers flagged it.

    The flag is not asked for here. Whether the fix applies to a tokenizer is
    transformers' judgement, not ours, and it announces that judgement by warning during
    an ordinary load. Reading it from the warning keeps this check from inventing its own
    rule about which tokenizers are affected, and from forcing Mistral's pretokenizer
    onto a tokenizer the library would never touch.
    """
    capture = _WarningCapture()
    root = logging.getLogger("transformers")
    root.addHandler(capture)
    try:
        tok = loader(str(path))
    finally:
        root.removeHandler(capture)
    return tok, any(WARNING_MARK in m for m in capture.messages)


def backend_state(tok) -> str | None:
    """The serialized backend tokenizer, which is what actually decides tokenization."""
    backend = getattr(tok, "backend_tokenizer", None)
    to_str = getattr(backend, "to_str", None)
    return to_str() if callable(to_str) else None


def installed_split_pattern(source: str | None = None) -> str | None:
    """The fixed pretokenizer regex, read out of the installed transformers.

    Copied into this file it would be a second source of truth that no upgrade updates.
    Read from the module that does the patching, a version whose pattern changed is
    compared against its own pattern, and a version that no longer has one is reported as
    having nothing to apply instead of silently applying the old one.
    """
    if source is None:
        try:
            import transformers.tokenization_utils_tokenizers as module
        except ImportError:
            return None
        source = Path(module.__file__).read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not (isinstance(node, ast.FunctionDef) and node.name == "_patch_mistral_regex"):
            continue
        for call in ast.walk(node):
            if (isinstance(call, ast.Call)
                    and getattr(call.func, "attr", None) == "Regex"
                    and call.args and isinstance(call.args[0], ast.Constant)
                    and isinstance(call.args[0].value, str)):
                return call.args[0].value
    return None


def installed_mistral_model_types(source: str | None = None) -> tuple[str, ...]:
    """The model types transformers' patch considers Mistral, read from the patch itself.

    The same reason the regex is read rather than copied: a list that lives in two files
    is a list that disagrees with itself after an upgrade.
    """
    if source is None:
        try:
            import transformers.tokenization_utils_tokenizers as module
        except ImportError:
            return ()
        source = Path(module.__file__).read_text()
    for node in ast.walk(ast.parse(source)):
        if not (isinstance(node, ast.FunctionDef) and node.name == "_patch_mistral_regex"):
            continue
        for item in ast.walk(node):
            if (isinstance(item, ast.List) and item.elts
                    and all(isinstance(e, ast.Constant) and isinstance(e.value, str)
                            for e in item.elts)
                    and any(e.value == "mistral" for e in item.elts)):
                return tuple(e.value for e in item.elts)
    return ()


def why_flagged(run_dir) -> dict:
    """What in the run directory made transformers suspect a Mistral tokenizer.

    The patch skips a local tokenizer whose config declares a ``transformers_version``
    outside the affected band, and otherwise consults ``model_type``. The ``config.json``
    in a run directory is this harness's experiment config, not a model config, so it has
    neither key and neither skip can fire. That is the whole mechanism of the warning,
    and it is checkable rather than assertable: a config with no ``model_type`` at all is
    not a config that says anything about Mistral.
    """
    path = Path(run_dir) / "config.json"
    config = json.loads(path.read_text()) if path.exists() else {}
    model_type = config.get("model_type")
    return {"model_type": model_type,
            "transformers_version": config.get("transformers_version"),
            "is_model_config": model_type is not None}


def published_tokenizer(model_key, *, loader=None):
    """The base model's own tokenizer, from the local hub cache, never downloaded."""
    from sentalign.modeling import MODEL_REGISTRY

    hf_id = MODEL_REGISTRY.get(model_key, {}).get("hf_id")
    if hf_id is None:
        return None, "unknown model key"
    if loader is not None:
        return loader(hf_id), "loader"
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(hf_id, local_files_only=True,
                                             trust_remote_code=True), "cache"
    except Exception as exc:
        return None, f"not loadable from the cache: {type(exc).__name__}: {exc}"


def published_model_type(model_key) -> str | None:
    """``model_type`` from the base model's published config, which is the file that does
    say whether this is a Mistral model."""
    from huggingface_hub import try_to_load_from_cache

    from sentalign.modeling import MODEL_REGISTRY

    hf_id = MODEL_REGISTRY.get(model_key, {}).get("hf_id")
    if hf_id is None:
        return None
    cached = try_to_load_from_cache(hf_id, "config.json")
    if not isinstance(cached, str):
        return None
    return json.loads(Path(cached).read_text()).get("model_type")


def matches_published_tokenizer(run_dir, model_key) -> dict:
    """Byte comparison of the saved tokenizer files against the published ones.

    Agreement settles the question outright. Disagreement settles nothing: saving an
    adapter rewrites tokenizer_config.json with the pad token the harness sets and
    re-serializes tokenizer.json, neither of which need move a single token. That is what
    the behavioural comparison is for.
    """
    from huggingface_hub import try_to_load_from_cache

    from sentalign.modeling import MODEL_REGISTRY

    hf_id = MODEL_REGISTRY.get(model_key, {}).get("hf_id")
    out: dict = {"hf_id": hf_id, "files": {}, "identical": None}
    if hf_id is None:
        return out
    for name in ("tokenizer.json", "tokenizer_config.json"):
        local = Path(run_dir) / name
        cached = try_to_load_from_cache(hf_id, name)
        if not local.exists() or not isinstance(cached, str):
            out["files"][name] = "not comparable"
            continue
        same = (hashlib.sha256(local.read_bytes()).hexdigest()
                == hashlib.sha256(Path(cached).read_bytes()).hexdigest())
        out["files"][name] = "identical" if same else "differs"
    states = list(out["files"].values())
    out["identical"] = (all(s == "identical" for s in states)
                        if states and "not comparable" not in states else None)
    return out


def reproduces_published_tokenizer(run_tok, model_key, texts, space, *, loader=None) -> dict:
    """Does the saved tokenizer encode exactly as the published one does?

    This is the question the warning raises for a model the fix was never meant for: not
    whether Mistral's regex would move our tokens, but whether the tokens we score are
    the ones anybody loading this model gets. Byte-level differences in the saved files
    are routine; a different encoding of a prompt is not.
    """
    published, note = published_tokenizer(model_key, loader=loader)
    out: dict = {"note": note, "identical": None}
    if published is None:
        return out
    prompts = [build_prompt(t, space) for t in texts]
    surfaces = list(resolve_verbalizers(run_tok, space, require_single_token=False).surfaces)
    out["prompts"] = compare_encodings(encode_all(run_tok, prompts),
                                       encode_all(published, prompts),
                                       labels=[f"prompt[{i}]" for i in range(len(prompts))])
    out["verbalizers"] = compare_encodings(
        encode_all(run_tok, surfaces, add_special_tokens=False),
        encode_all(published, surfaces, add_special_tokens=False), labels=surfaces)
    out["identical"] = bool(out["prompts"]["identical"] and out["verbalizers"]["identical"])
    return out


def apply_mistral_regex_fix(tok, *, pattern=None, tokenizers_module=None) -> str:
    """Do to a loaded tokenizer what transformers' own patch does, on the object that
    has the attribute the patch reaches for.

    transformers 5.5.0 cannot do this through the constructor: ``__init__`` hands
    ``_patch_mistral_regex`` the raw ``tokenizers.Tokenizer`` and the patch immediately
    asks it for ``.backend_tokenizer``, which only the wrapper has, so every affected
    tokenizer raises AttributeError before the fix is applied. Applying the same
    transformation afterwards, to the wrapper, is what lets the comparison happen at all.

    Returns "applied", "unchanged" (the fix leaves this tokenizer's backend byte for
    byte as it was, so it cannot change any token), or "unavailable".
    """
    if tokenizers_module is None:
        try:
            import tokenizers as tokenizers_module
        except ImportError:
            return "unavailable"
    pattern = pattern or installed_split_pattern()
    backend = getattr(tok, "backend_tokenizer", None)
    if pattern is None or backend is None:
        return "unavailable"

    before = backend_state(tok)
    pre = tokenizers_module.pre_tokenizers
    try:
        split = pre.Split(pattern=tokenizers_module.Regex(pattern), behavior="isolated")
        current = backend.pre_tokenizer
        if isinstance(current, pre.Sequence):
            backend.pre_tokenizer[0] = split
        else:
            if isinstance(current, pre.Metaspace):
                # Metaspace(split=False) loses spaces next to the Split pretokenizer, so
                # the library swaps it for ByteLevel here. Mirrored, not improved on.
                current = pre.ByteLevel(add_prefix_space=False, use_regex=False)
            backend.pre_tokenizer = pre.Sequence([split, current])
    except AttributeError as exc:
        # A tokenizers build without the pretokenizers this patch composes. Nothing was
        # applied, and a half-applied tokenizer is not something to compare against.
        return f"unavailable: {exc}"
    return "applied" if backend_state(tok) != before else "unchanged"


def fixed_tokenizer(path, *, loader, plain_state, tokenizers_module=None):
    """A tokenizer with the fix applied, and how it was obtained.

    The constructor keyword is tried first and then verified, because on this
    transformers it is accepted and ignored for tokenizers under the vocabulary
    threshold: a keyword that raises is obvious, one that is quietly dropped would have
    this check compare a tokenizer against itself and report agreement.
    """
    how = []
    try:
        tok = loader(str(path), **{FLAG: True})
        if backend_state(tok) != plain_state:
            return tok, "constructor"
        how.append("constructor accepted the flag and changed nothing")
    except TypeError:
        how.append("constructor does not take the flag")
    except Exception as exc:
        how.append(f"constructor raised {type(exc).__name__}: {exc}")

    tok = loader(str(path))
    status = apply_mistral_regex_fix(tok, tokenizers_module=tokenizers_module)
    how.append(f"applied from the installed source: {status}")
    if status == "applied":
        return tok, "; ".join(how)
    if status == "unchanged":
        return tok, "; ".join(how)          # a no-op fix cannot move a token
    return None, "; ".join(how)


def encode_all(tokenizer, texts, *, add_special_tokens: bool = True) -> list[list[int]]:
    # `text=` rather than positional, for the reason documented in labels.resolve_verbalizers.
    return [list(tokenizer(text=t, add_special_tokens=add_special_tokens)["input_ids"])
            for t in texts]


def compare_encodings(left, right, labels=None) -> dict:
    """Compare two id sequences element by element, over their whole length.

    Reports every differing position, not the first one: a pretokenizer change that hit
    one item in a thousand and was reported as "a difference" would read like a rounding
    detail, while the count says how much of the corpus it touches.
    """
    if len(left) != len(right):
        raise ValueError(f"comparing {len(left)} encodings against {len(right)}")
    labels = list(labels) if labels is not None else [str(i) for i in range(len(left))]
    differing = [i for i, (a, b) in enumerate(zip(left, right)) if a != b]
    examples = [{"label": labels[i], "plain": left[i], "fixed": right[i]}
                for i in differing[:5]]
    return {"n": len(left), "n_differing": len(differing),
            "identical": not differing, "examples": examples}


def check_tokenizer(path, texts, *, space=TERNARY_PLUS_MIXED, loader=None,
                    tokenizers_module=None, model_key=None,
                    published_loader=None) -> dict:
    """The full comparison for one tokenizer directory."""
    loader = loader or _default_loader
    plain, affected = load_plain(path, loader)
    report: dict = {"path": str(path), "n_texts": len(texts), "affected": affected}
    if model_key is not None:
        report["config"] = why_flagged(path)
        published = matches_published_tokenizer(path, model_key)
        published["model_type"] = published_model_type(model_key)
        published["is_mistral_type"] = (published["model_type"]
                                        in installed_mistral_model_types()
                                        if published["model_type"] else None)
        if published["identical"] is not True:
            # Different bytes are not different tokens. Ask the tokenizers, not the files.
            published["behaviour"] = reproduces_published_tokenizer(
                plain, model_key, texts, space, loader=published_loader)
        report["published"] = published
    if not affected:
        # No warning means transformers does not consider this tokenizer one the fix
        # applies to, and forcing it on anyway would answer a question nobody asked.
        report["flag"] = "not applicable, transformers did not flag this tokenizer"
        report["identical"] = True
        return report

    fixed, how = fixed_tokenizer(path, loader=loader, plain_state=backend_state(plain),
                                 tokenizers_module=tokenizers_module)
    report["flag"] = how
    if fixed is None:
        report["identical"] = None
        return report

    prompts = [build_prompt(t, space) for t in texts]
    report["prompts"] = compare_encodings(encode_all(plain, prompts),
                                          encode_all(fixed, prompts),
                                          labels=[f"prompt[{i}]" for i in range(len(prompts))])
    # The verbalizer ids are scored directly, so a change in them is a change in every
    # logit this protocol reads, whatever the prompts do. The surfaces come from
    # resolve_verbalizers so that the leading space is attached exactly once and in the
    # same place as in the scorer, but the comparison is on the full encoding: the
    # table's token_ids collapse to a sentinel when a surface is not single-token, and
    # comparing two sentinels would agree no matter what the tokenizer did.
    surfaces = list(resolve_verbalizers(plain, space, require_single_token=False).surfaces)
    plain_ids = encode_all(plain, surfaces, add_special_tokens=False)
    fixed_ids = encode_all(fixed, surfaces, add_special_tokens=False)
    report["verbalizers"] = {
        "surfaces": surfaces, "plain": plain_ids, "fixed": fixed_ids,
        "identical": plain_ids == fixed_ids,
        "single_token": all(len(ids) == 1 for ids in plain_ids + fixed_ids)}
    report["identical"] = bool(report["prompts"]["identical"]
                               and report["verbalizers"]["identical"])
    return report


def verdict(report) -> str:
    """What this tokenizer's result means for the runs that used it.

    Three ways to be in the clear, and they are not the same statement:

    ``not flagged``          transformers does not consider the fix to apply here at all.
    ``fix changes nothing``  it applies, it was applied, and no scored token moved.
    ``fix does not apply``   it moves tokens, but the base model is not one of the types
                             the patch is for, and the saved tokenizer encodes every
                             prompt and verbalizer exactly as the published one does. The
                             difference is then what the fix would break, not what the
                             runs got wrong.

    Anything else is ``unresolved``: a real difference that has not been shown to be a
    false positive is a retraining decision, not a footnote, and the two must not be
    allowed to look alike.
    """
    if report.get("affected") is False:
        return "not flagged"
    if report.get("identical") is True:
        return "fix changes nothing"
    if report.get("identical") is None:
        return "unresolved"
    published = report.get("published") or {}
    faithful = (published.get("identical") is True
                or (published.get("behaviour") or {}).get("identical") is True)
    if faithful and published.get("is_mistral_type") is False:
        return "fix does not apply"
    return "unresolved"


def exit_code(reports) -> int:
    """Non-zero unless every tokenizer checked came out of ``verdict`` in the clear.

    Nothing compared counts as not shown, so neither a stale transformers nor a library
    bug in the fix itself can turn this check into a pass.
    """
    return 0 if reports and all(verdict(r) != "unresolved" for r in reports) else 1


def scored_sets() -> tuple[str, ...]:
    """The sets the protocol actually encodes, read from the evaluation config.

    Not a glob over the build directory: that also picks up files this protocol never
    scores, such as the contrast set, whose rows carry ``original`` and ``rewritten``
    instead of ``text``. The question here is whether the flag moves a token the models
    are scored on, so the answer has to be computed over the sets they are scored on.
    """
    from sentalign.config import EvalConfig

    cfg = EvalConfig()
    return tuple(cfg.eval_sets) + (cfg.temperature_fit_split,)


def sample_texts(build_dir: Path, limit_per_set: int, sets=None) -> list[str]:
    """Evaluation texts, deterministically thinned so a rerun compares the same strings."""
    texts: list[str] = []
    for name in (sets if sets is not None else scored_sets()):
        path = build_dir / "eval" / f"{name}.jsonl"
        if not path.exists():
            raise SystemExit(f"{path} is missing; the check needs the sets that are scored")
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        stride = max(1, len(rows) // limit_per_set) if limit_per_set else 1
        texts.extend(r["text"] for r in rows[::stride][:limit_per_set or None])
    return texts


def model_of(run_dir: Path) -> str:
    parts = run_dir.name.split("__")
    return parts[1] if len(parts) > 2 else run_dir.name


def tokenizer_dirs(runs_root: Path, *, every_run: bool) -> list[Path]:
    """One run directory per model by default: runs of a model share a tokenizer, and
    checking 300 copies of the same files says nothing the first one did not."""
    dirs, seen = [], set()
    for run_dir in sorted(p for p in runs_root.glob("*") if (p / "config.json").exists()):
        if not (run_dir / "tokenizer_config.json").exists():
            continue
        key = model_of(run_dir)
        if not every_run and key in seen:
            continue
        seen.add(key)
        dirs.append(run_dir)
    return dirs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--data", type=Path, default=Path("data/build"))
    ap.add_argument("--out", type=Path, default=Path("results/tokenizer_regex_check.json"))
    ap.add_argument("--tokenizer", type=Path, action="append",
                    help="check this directory instead of one run per model")
    ap.add_argument("--every-run", action="store_true")
    ap.add_argument("--limit-per-set", type=int, default=400,
                    help="0 checks every evaluation item")
    ap.add_argument("--sets", nargs="+", default=None,
                    help=f"defaults to the scored sets: {' '.join(scored_sets())}")
    args = ap.parse_args(argv)

    texts = sample_texts(args.data, args.limit_per_set, args.sets)
    if not texts:
        raise SystemExit(f"no evaluation texts under {args.data / 'eval'}")
    paths = args.tokenizer or tokenizer_dirs(args.runs, every_run=args.every_run)
    if not paths:
        raise SystemExit(f"no run directory with a saved tokenizer under {args.runs}")

    reports = []
    for path in paths:
        key = model_of(Path(path))
        report = check_tokenizer(path, texts, model_key=key)
        report["model"] = key
        reports.append(report)
        if report["identical"] is None:
            print(f"  {report['model']:<12} NOT COMPARED: {report['flag']}")
            continue
        if "prompts" not in report:
            print(f"  {report['model']:<12} {verdict(report)}: {report['flag']}")
            continue
        published = report.get("published", {})
        behaviour = (published.get("behaviour") or {}).get("identical")
        reproduces = ("bytes identical" if published.get("identical") is True
                      else {True: "encodes identically", False: "ENCODES DIFFERENTLY",
                           None: "not compared"}[behaviour])
        print(f"  {report['model']:<12} {verdict(report)}: "
              f"{report['prompts']['n_differing']}/{report['prompts']['n']} prompts move "
              f"under the fix, verbalizer ids "
              f"{'unchanged' if report['verbalizers']['identical'] else 'CHANGED'}; "
              f"base model_type={published.get('model_type')!r}, "
              f"run config is a model config: {report['config']['is_model_config']}, "
              f"vs published tokenizer: {reproduces}")

    code = exit_code(reports)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"flag": FLAG, "n_texts": len(texts), "exit_code": code,
         "verdicts": {r["model"]: verdict(r) for r in reports}, "reports": reports},
        indent=2) + "\n")
    print(f"wrote {args.out}")
    if code:
        print("tokenization was not shown to be unaffected; see the report before "
              "using these runs")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
