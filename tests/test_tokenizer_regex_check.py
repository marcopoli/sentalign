"""The check that says whether transformers' Mistral-regex fix moves any token we score.

The dangerous failure for a check like this is not a wrong answer but a cheerful one:
transformers 5.5.0 accepts ``fix_mistral_regex=True`` and ignores it below a vocabulary
threshold, and raises AttributeError from inside its own patch above one, so both "the
flag was set" and "no exception" are worthless as evidence. Each test breaks the
corresponding property in ``scripts/07_tokenizer_regex_check.py`` and watches it fail.
"""

from __future__ import annotations

import importlib.util
import json
import logging
from pathlib import Path

import pytest

from sentalign.labels import TERNARY_PLUS_MIXED

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "07_tokenizer_regex_check.py"
WARNING = ("The tokenizer you are loading from 'x' with an incorrect regex pattern: "
           "you should set the `fix_mistral_regex=True` flag")


@pytest.fixture(scope="module")
def check():
    spec = importlib.util.spec_from_file_location("tokenizer_regex_check", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# -- fakes: a tokenizer whose encoding follows its pretokenizer, as a real one's does ----


class FakeSplit:
    def __init__(self, pattern=None, behavior=None):
        self.pattern, self.behavior = pattern, behavior

    def describe(self):
        return f"Split({self.pattern},{self.behavior})"


class FakeSequence(list):
    def describe(self):
        return "Sequence[" + ",".join(p.describe() for p in self) + "]"


class FakeMetaspace:
    def describe(self):
        return "Metaspace"


class FakeByteLevel:
    def __init__(self, add_prefix_space=None, use_regex=None):
        self.add_prefix_space, self.use_regex = add_prefix_space, use_regex

    def describe(self):
        return "ByteLevel"


class FakePreTokenizers:
    Split, Sequence, Metaspace, ByteLevel = (FakeSplit, FakeSequence, FakeMetaspace,
                                             FakeByteLevel)


class FakeTokenizersModule:
    pre_tokenizers = FakePreTokenizers

    @staticmethod
    def Regex(pattern):
        return f"Regex({pattern})"


class FakeBackend:
    def __init__(self, pre_tokenizer):
        self.pre_tokenizer = pre_tokenizer

    def to_str(self):
        return self.pre_tokenizer.describe()


class FakeTokenizer:
    """Encodes by character, and one token longer once a Split pretokenizer is in place,
    which is how a real tokenizer answers a changed pretokenizer."""

    def __init__(self, pre_tokenizer=None):
        self.backend_tokenizer = FakeBackend(pre_tokenizer or FakeMetaspace())

    def __call__(self, text=None, add_special_tokens=True, **kwargs):
        ids = [ord(c) % 97 for c in text]
        if "Split" in self.backend_tokenizer.to_str():
            ids = ids + [999]
        return {"input_ids": ids}


def loader_for(factory, *, raises=None, affected=True, ignores_flag=False):
    """Every call builds a new tokenizer, as ``from_pretrained`` does. Returning one
    shared object would let the fix applied to the second load reach the first, and the
    comparison would then be a tokenizer against itself."""

    def loader(path, **kwargs):
        if kwargs:
            if raises is not None:
                raise raises
            return factory()               # a load that ignores the flag: same as plain
        if affected:
            logging.getLogger("transformers.fake").warning(WARNING)
        return factory()
    return loader


# -- the comparison itself ---------------------------------------------------------------


def test_a_difference_anywhere_in_the_corpus_is_counted(check):
    left = [[1, 2], [3], [4, 5], [6], [7, 8]]
    right = [[1, 2], [3], [4, 5], [6], [7, 9]]
    report = check.compare_encodings(left, right, labels=[f"p{i}" for i in range(5)])
    assert report["identical"] is False
    assert report["n"] == 5 and report["n_differing"] == 1
    assert report["examples"][0]["label"] == "p4"


def test_a_difference_that_keeps_the_token_count_is_caught(check):
    """A pretokenizer change can move a merge boundary without changing how many tokens
    come out, so comparing lengths is not comparing tokenization."""
    report = check.compare_encodings([[1, 2, 3]], [[1, 9, 3]])
    assert report["n_differing"] == 1


# -- what counts as evidence -------------------------------------------------------------


def test_a_tokenizer_transformers_does_not_flag_is_left_alone(check):
    """LFM2.5 sits under the vocabulary threshold, so the library never applies the fix
    to it and never warns. Forcing Mistral's pretokenizer on anyway would measure a
    tokenizer that this protocol never loads."""
    report = check.check_tokenizer("tok", ["a good film"],
                                   loader=loader_for(FakeTokenizer, affected=False),
                                   tokenizers_module=FakeTokenizersModule)
    assert report["affected"] is False
    assert report["identical"] is True
    assert "prompts" not in report, "nothing was forced onto an unaffected tokenizer"


def test_a_flag_the_constructor_accepts_and_ignores_is_not_evidence(check):
    """transformers 5.5.0 takes fix_mistral_regex=True and drops it silently below its
    vocabulary threshold. Trusting the keyword would compare a tokenizer against itself
    and call the agreement a result."""
    report = check.check_tokenizer("tok", [f"review {i}" for i in range(4)],
                                   loader=loader_for(FakeTokenizer, ignores_flag=True),
                                   tokenizers_module=FakeTokenizersModule)
    assert "changed nothing" in report["flag"]
    assert report["prompts"]["n_differing"] == 4, "compared against a genuinely fixed one"
    assert report["identical"] is False


def test_a_constructor_that_raises_falls_back_to_the_installed_patch(check):
    """The path this environment actually takes: __init__ hands the patch a raw
    tokenizers.Tokenizer, which has no backend_tokenizer, so the flag never applies."""
    report = check.check_tokenizer(
        "tok", ["a good film"],
        loader=loader_for(FakeTokenizer,
                          raises=AttributeError("no attribute 'backend_tokenizer'")),
        tokenizers_module=FakeTokenizersModule)
    assert "AttributeError" in report["flag"] and "applied" in report["flag"]
    assert report["identical"] is False, "the fix changed this fake's tokenization"
    assert report["prompts"]["n_differing"] == 1


def test_a_fix_that_cannot_be_applied_is_not_agreement(check):
    """No pattern in the installed source means nothing was compared, which is not the
    same as nothing changed."""
    report = check.check_tokenizer(
        "tok", ["a good film"],
        loader=loader_for(FakeTokenizer, raises=AttributeError("boom")),
        tokenizers_module=_NoTokenizers)
    assert report["identical"] is None
    assert check.exit_code([report]) == 1


class _NoTokenizers:
    """A tokenizers module without the pretokenizers the patch needs."""

    class pre_tokenizers:
        pass

    @staticmethod
    def Regex(pattern):
        return pattern


def test_a_fix_that_changes_nothing_is_reported_as_such(check):
    """Applying the patch to a tokenizer that already carries it must leave the backend
    byte for byte as it was. That is agreement, and it has to be distinguishable from a
    patch that never ran."""
    pattern = "PATTERN"
    already = FakeSequence([FakeSplit(pattern=f"Regex({pattern})", behavior="isolated"),
                            FakeByteLevel()])
    tok = FakeTokenizer(already)
    assert check.apply_mistral_regex_fix(tok, pattern=pattern,
                                         tokenizers_module=FakeTokenizersModule) == "unchanged"
    fresh = FakeTokenizer()
    assert check.apply_mistral_regex_fix(fresh, pattern=pattern,
                                         tokenizers_module=FakeTokenizersModule) == "applied"
    # The library swaps Metaspace for ByteLevel rather than keeping it next to the Split.
    assert "Metaspace" not in fresh.backend_tokenizer.to_str()


def test_a_verbalizer_id_change_fails_even_when_every_prompt_agrees(check):
    """The label ids are read straight out of the logits, so they are the one thing a
    prompt-only comparison would miss entirely."""

    class OnlyVerbalizersMove(FakeTokenizer):
        def __call__(self, text=None, add_special_tokens=True, **kwargs):
            ids = [ord(c) % 97 for c in text]
            if text.startswith(" ") and "Split" in self.backend_tokenizer.to_str():
                ids = ids + [999]
            return {"input_ids": ids}

    report = check.check_tokenizer(
        "tok", ["a good film"],
        loader=loader_for(OnlyVerbalizersMove,
                          raises=AttributeError("no attribute 'backend_tokenizer'")),
        tokenizers_module=FakeTokenizersModule)
    assert report["prompts"]["identical"] is True
    assert report["verbalizers"]["identical"] is False
    assert report["identical"] is False


def test_nothing_checked_is_not_success(check):
    assert check.exit_code([]) == 1
    assert check.exit_code([{"identical": True}, {"identical": None}]) == 1
    assert check.exit_code([{"identical": True}, {"identical": True}]) == 0


# -- the pattern comes from the installed library, not from this file --------------------


def test_the_split_pattern_is_read_from_the_patching_function(check):
    source = '''
class T:
    def _patch_mistral_regex(cls, tokenizer):
        split = tokenizers.pre_tokenizers.Split(
            pattern=tokenizers.Regex(r"THE-PATTERN"), behavior="isolated")
        return tokenizer

def unrelated():
    x = tokenizers.Regex(r"NOT-THIS-ONE")
'''
    assert check.installed_split_pattern(source) == "THE-PATTERN"


def test_a_library_without_a_pattern_yields_none_rather_than_a_stale_one(check):
    source = '''
class T:
    def _patch_mistral_regex(cls, tokenizer):
        return tokenizer
'''
    assert check.installed_split_pattern(source) is None
    assert check.installed_split_pattern("def other(): pass") is None


def test_the_installed_transformers_still_carries_the_pattern(check):
    pytest.importorskip("transformers")
    pattern = check.installed_split_pattern()
    assert pattern and "\\p{L}" in pattern


# -- which texts and which tokenizers ----------------------------------------------------


def test_the_sample_is_the_same_strings_on_every_run(check, tmp_path):
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    (eval_dir / "r1_test.jsonl").write_text(
        "\n".join(json.dumps({"text": f"t{i}"}) for i in range(100)) + "\n")
    (eval_dir / "r2_test.jsonl").write_text(
        "\n".join(json.dumps({"text": f"u{i}"}) for i in range(7)) + "\n")

    sets = ("r1_test", "r2_test")
    first = check.sample_texts(tmp_path, 10, sets)
    assert first == check.sample_texts(tmp_path, 10, sets)
    assert len(first) == 17, "10 thinned from the large set, all 7 of the small one"
    # Spread across the file, not its first ten lines: the evaluation sets are written in
    # source and round order, so a head sample would check one corner of the corpus.
    assert "t90" in first and "t9" not in first
    assert len(check.sample_texts(tmp_path, 0, sets)) == 107


def test_only_the_scored_sets_are_read(check, tmp_path):
    """The build directory also holds files this protocol never scores. The contrast set
    has no ``text`` field at all, so globbing the directory crashed the check on data it
    had no reason to encode."""
    from sentalign.config import EvalConfig

    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    for name in check.scored_sets():
        (eval_dir / f"{name}.jsonl").write_text(json.dumps({"text": f"in {name}"}) + "\n")
    (eval_dir / "dynasent_rewrites.jsonl").write_text(
        json.dumps({"original": "o", "rewritten": "r", "text_id": "x"}) + "\n")

    texts = check.sample_texts(tmp_path, 400)
    assert len(texts) == len(check.scored_sets())
    assert all(t.startswith("in ") for t in texts)
    cfg = EvalConfig()
    assert set(check.scored_sets()) == set(cfg.eval_sets) | {cfg.temperature_fit_split}


def test_a_missing_scored_set_is_an_error(check, tmp_path):
    (tmp_path / "eval").mkdir()
    with pytest.raises(SystemExit):
        check.sample_texts(tmp_path, 400)


def test_one_run_per_model_unless_asked_for_all(check, tmp_path):
    for name in ("main__lfm-1.2b__sft__s13", "main__lfm-1.2b__kto__s13",
                 "main__smollm3-3b__sft__s13"):
        run = tmp_path / name
        run.mkdir()
        (run / "config.json").write_text("{}")
        (run / "tokenizer_config.json").write_text("{}")
    (tmp_path / "no_tokenizer").mkdir()
    (tmp_path / "no_tokenizer" / "config.json").write_text("{}")

    assert [check.model_of(p) for p in check.tokenizer_dirs(tmp_path, every_run=False)] \
        == ["lfm-1.2b", "smollm3-3b"]
    assert len(check.tokenizer_dirs(tmp_path, every_run=True)) == 3


# -- what a difference means -------------------------------------------------------------


def test_the_mistral_model_types_come_from_the_patch(check):
    source = '''
class T:
    def _patch_mistral_regex(cls, tokenizer):
        if model_type not in ["mistral", "mistral3", "voxtral"]:
            return tokenizer
'''
    assert check.installed_mistral_model_types(source) == ("mistral", "mistral3", "voxtral")
    assert check.installed_mistral_model_types("def f(): pass") == ()
    pytest.importorskip("transformers")
    assert "mistral" in check.installed_mistral_model_types()


def test_the_run_config_is_not_a_model_config(check, tmp_path):
    """The config.json transformers reads in a run directory is this harness's experiment
    config. It has neither transformers_version nor model_type, so neither of the patch's
    two skip rules can fire, and that is the entire reason a Qwen and a SmolLM3 tokenizer
    are treated as possibly Mistral."""
    run = tmp_path / "main__smollm3-3b__sft__s13"
    run.mkdir()
    (run / "config.json").write_text(json.dumps({"model": "smollm3-3b", "train": {}}))
    facts = check.why_flagged(run)
    assert facts["model_type"] is None
    assert facts["transformers_version"] is None
    assert facts["is_model_config"] is False

    other = tmp_path / "real_model"
    other.mkdir()
    (other / "config.json").write_text(
        json.dumps({"model_type": "smollm3", "transformers_version": "4.56.0"}))
    assert check.why_flagged(other)["is_model_config"] is True


def test_the_saved_tokenizer_is_compared_to_the_published_one_by_behaviour(check):
    """Saving an adapter rewrites tokenizer_config.json with the harness's pad token and
    re-serializes tokenizer.json, so the files differ from the published ones on every
    run. Different bytes are not different tokens, and only the tokens are scored."""
    same = check.reproduces_published_tokenizer(
        FakeTokenizer(), "smollm3-3b", ["a good film", "a bad film"], TERNARY_PLUS_MIXED,
        loader=lambda hf_id: FakeTokenizer())
    assert same["identical"] is True
    assert same["prompts"]["n_differing"] == 0 and same["verbalizers"]["n_differing"] == 0

    moved = check.reproduces_published_tokenizer(
        FakeTokenizer(), "smollm3-3b", ["a good film"], TERNARY_PLUS_MIXED,
        loader=lambda hf_id: FakeTokenizer(FakeSequence([FakeSplit(), FakeByteLevel()])))
    assert moved["identical"] is False

    # The label surfaces are encoded separately from the prompts and read straight out of
    # the logits, so a saved tokenizer can agree on every prompt and still score a
    # different set of ids.
    class VerbalizersOnly(FakeTokenizer):
        def __call__(self, text=None, add_special_tokens=True, **kwargs):
            ids = [ord(c) % 97 for c in text]
            if text.startswith(" ") and "Split" in self.backend_tokenizer.to_str():
                ids = ids + [999]
            return {"input_ids": ids}

    labels_moved = check.reproduces_published_tokenizer(
        VerbalizersOnly(), "smollm3-3b", ["a good film"], TERNARY_PLUS_MIXED,
        loader=lambda hf_id: VerbalizersOnly(FakeSequence([FakeSplit(), FakeByteLevel()])))
    assert labels_moved["prompts"]["identical"] is True
    assert labels_moved["verbalizers"]["identical"] is False
    assert labels_moved["identical"] is False


def test_an_unavailable_published_tokenizer_settles_nothing(check):
    out = check.reproduces_published_tokenizer(
        FakeTokenizer(), "smollm3-3b", ["a good film"], TERNARY_PLUS_MIXED,
        loader=lambda hf_id: None)
    assert out["identical"] is None
    assert "prompts" not in out


def test_the_run_tokenizer_is_compared_against_the_published_one(check, tmp_path, monkeypatch):
    """A non-Mistral model's real question is whether we score with the tokenizer the
    model was published with, which is a byte comparison and not a matter of opinion."""
    import huggingface_hub

    from sentalign.modeling import MODEL_REGISTRY

    published = tmp_path / "cache"
    published.mkdir()
    (published / "tokenizer.json").write_text('{"a": 1}')
    (published / "tokenizer_config.json").write_text('{"b": 2}')
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache",
                        lambda repo_id, filename: str(published / filename))

    run = tmp_path / "run"
    run.mkdir()
    (run / "tokenizer.json").write_text('{"a": 1}')
    (run / "tokenizer_config.json").write_text('{"b": 2}')
    same = check.matches_published_tokenizer(run, "smollm3-3b")
    assert same["identical"] is True
    assert same["hf_id"] == MODEL_REGISTRY["smollm3-3b"]["hf_id"]

    (run / "tokenizer.json").write_text('{"a": 2}')
    assert check.matches_published_tokenizer(run, "smollm3-3b")["identical"] is False


def test_an_uncached_base_model_is_not_a_match(check, tmp_path, monkeypatch):
    """try_to_load_from_cache returns a sentinel object, not a path, when the file is not
    there. Treating that as a match would certify a comparison that never happened."""
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache",
                        lambda repo_id, filename: None)
    run = tmp_path / "run"
    run.mkdir()
    (run / "tokenizer.json").write_text("{}")
    (run / "tokenizer_config.json").write_text("{}")
    assert check.matches_published_tokenizer(run, "smollm3-3b")["identical"] is None


def test_a_difference_is_dismissed_only_on_evidence(check):
    by_bytes = {"affected": True, "identical": False,
                "published": {"identical": True, "is_mistral_type": False}}
    assert check.verdict(by_bytes) == "fix does not apply"
    assert check.exit_code([by_bytes]) == 0

    by_behaviour = {"affected": True, "identical": False,
                    "published": {"identical": False, "is_mistral_type": False,
                                  "behaviour": {"identical": True}}}
    assert check.verdict(by_behaviour) == "fix does not apply"

    unverified = {"affected": True, "identical": False,
                  "published": {"identical": None, "is_mistral_type": False,
                                "behaviour": {"identical": None}}}
    assert check.verdict(unverified) == "unresolved", "nothing compared, nothing dismissed"

    moved = {"affected": True, "identical": False,
             "published": {"identical": False, "is_mistral_type": False,
                           "behaviour": {"identical": False}}}
    assert check.verdict(moved) == "unresolved", "the saved tokenizer scores differently"

    mistral = {"affected": True, "identical": False,
               "published": {"identical": True, "is_mistral_type": True}}
    assert check.verdict(mistral) == "unresolved", "the fix is for exactly this model type"

    unknown_type = {"affected": True, "identical": False,
                    "published": {"identical": True, "is_mistral_type": None}}
    assert check.verdict(unknown_type) == "unresolved", "an unread base config is not a no"

    assert check.verdict({"affected": False}) == "not flagged"
    assert check.verdict({"affected": True, "identical": True}) == "fix changes nothing"
    assert check.verdict({"affected": True, "identical": None}) == "unresolved"
    assert check.exit_code([by_bytes, mistral]) == 1
