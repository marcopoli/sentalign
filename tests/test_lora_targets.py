"""LoRA target discovery, tested against the real architectures of both model families.

The bug this guards against is silent by nature. A target list that matches nothing in a
layer yields a model that loads, trains, and converges, on a fraction of the parameters
the configuration implies, with a plausible number in the manifest. It was found only by
noticing that a 350M model reported 491,520 trainable parameters and working out by hand
which projections that could correspond to.

The module inventories below were read from the published checkpoints.
"""

from __future__ import annotations

import re

import pytest

from sentalign.lora_targets import (LAYER_PATTERN, CoverageReport, LoraCoverageError,
                                    count_layers, discover_targets, is_excluded,
                                    leaf_targets, resolve_and_verify, verify_coverage)

from importlib.util import find_spec

#: Only the module-building tests need torch and peft. The tests that document the defect
#: work from module inventories read out of the published checkpoints, and must run
#: everywhere, including on a machine with no training stack installed.
needs_torch = pytest.mark.skipif(find_spec("torch") is None, reason="torch not installed")
needs_peft = pytest.mark.skipif(find_spec("peft") is None, reason="peft not installed")


def build_stack(layer_spec):
    """A module tree shaped like a decoder stack, from {name: (in, out)} per layer."""
    import torch.nn as nn

    model = nn.Module()
    model.layers = nn.ModuleList()
    for spec in layer_spec:
        layer = nn.Module()
        for dotted, shape in spec.items():
            target, parent = layer, None
            parts = dotted.split(".")
            for part in parts[:-1]:
                if not hasattr(target, part):
                    setattr(target, part, nn.Module())
                parent, target = target, getattr(target, part)
            if shape is None:
                setattr(target, parts[-1], nn.LayerNorm(8))
            else:
                setattr(target, parts[-1], nn.Linear(*shape, bias=False))
        model.layers.append(layer)
    model.embed_tokens = nn.Embedding(32, 8)
    model.lm_head = nn.Linear(8, 32, bias=False)
    return model


#: LFM2.5: 16 layers, 6 attention and 10 convolutional, feed-forward in every layer.
LFM2_LAYERS = (
    [{"self_attn.q_proj": (8, 8), "self_attn.k_proj": (8, 4),
      "self_attn.v_proj": (8, 4), "self_attn.out_proj": (8, 8),
      "self_attn.q_layernorm": None, "self_attn.k_layernorm": None,
      "feed_forward.w1": (8, 16), "feed_forward.w2": (16, 8),
      "feed_forward.w3": (8, 16), "ffn_norm": None, "operator_norm": None}] * 6
    + [{"conv.in_proj": (8, 16), "conv.out_proj": (8, 8),
        "feed_forward.w1": (8, 16), "feed_forward.w2": (16, 8),
        "feed_forward.w3": (8, 16), "ffn_norm": None, "operator_norm": None}] * 10)

#: Qwen3.5-2B: 24 layers, 6 full attention and 18 linear attention, MLP in every layer.
#: Counts read from the loaded checkpoint on 2026-08-20, not from the model card.
QWEN_LAYERS = (
    [{"self_attn.q_proj": (8, 8), "self_attn.k_proj": (8, 4),
      "self_attn.v_proj": (8, 4), "self_attn.o_proj": (8, 8),
      "self_attn.q_norm": None, "self_attn.k_norm": None,
      "mlp.gate_proj": (8, 16), "mlp.up_proj": (8, 16), "mlp.down_proj": (16, 8),
      "input_layernorm": None, "post_attention_layernorm": None}] * 6
    + [{"linear_attn.in_proj_qkv": (8, 16), "linear_attn.in_proj_a": (8, 4),
        "linear_attn.in_proj_b": (8, 4), "linear_attn.in_proj_z": (8, 8),
        "linear_attn.out_proj": (8, 8), "linear_attn.norm": None,
        "mlp.gate_proj": (8, 16), "mlp.up_proj": (8, 16), "mlp.down_proj": (16, 8),
        "input_layernorm": None, "post_attention_layernorm": None}] * 18)

#: SmolLM3-3B: 36 dense layers, attention and MLP in every one, no other token mixer.
#: Names and depth read from the published safetensors index on 2026-09-15.
SMOLLM3_LAYERS = (
    [{"self_attn.q_proj": (8, 8), "self_attn.k_proj": (8, 4),
      "self_attn.v_proj": (8, 4), "self_attn.o_proj": (8, 8),
      "mlp.gate_proj": (8, 16), "mlp.up_proj": (8, 16), "mlp.down_proj": (16, 8),
      "input_layernorm": None, "post_attention_layernorm": None}] * 36)

#: The near-universal Llama-style list, which is what this codebase used to hard-code.
LLAMA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj",
                 "gate_proj", "up_proj", "down_proj")


def adapted_layers(layer_spec, targets) -> set[int]:
    """Which layers a target list would reach, matching as PEFT does (endswith)."""
    covered = set()
    for index, spec in enumerate(layer_spec):
        for dotted, shape in spec.items():
            if shape is None:
                continue
            if any(dotted == t or dotted.endswith("." + t) or dotted.split(".")[-1] == t
                   for t in targets):
                covered.add(index)
    return covered


# --------------------------------------------------------------------------------------
# The defect, reproduced
# --------------------------------------------------------------------------------------

def test_llama_targets_miss_most_of_lfm2():
    """Reproduces the observed failure: q, k, v in 6 of 16 layers and no feed-forward."""
    covered = adapted_layers(LFM2_LAYERS, LLAMA_TARGETS)
    assert len(covered) == 6, f"expected only the attention layers, got {sorted(covered)}"

    reached = {d.split(".")[-1] for spec in LFM2_LAYERS for d, s in spec.items()
               if s is not None and d.split(".")[-1] in LLAMA_TARGETS}
    assert reached == {"q_proj", "k_proj", "v_proj"}
    assert "out_proj" not in reached, "LFM2 calls it out_proj, not o_proj"
    assert not {"w1", "w2", "w3"} & reached, "the feed-forward stack is never reached"


def test_llama_targets_miss_qwen_linear_attention():
    """Every MLP is reached, but 18 of 24 layers get no token-mixing adapter."""
    mixing = [{k: v for k, v in spec.items() if not k.startswith("mlp.")}
              for spec in QWEN_LAYERS]
    assert len(adapted_layers(mixing, LLAMA_TARGETS)) == 6
    assert len(adapted_layers(QWEN_LAYERS, LLAMA_TARGETS)) == 24   # via mlp only


def test_the_two_families_are_handicapped_differently():
    """This is why the defect confounds the family comparison rather than just weakening it."""
    lfm = len(adapted_layers(LFM2_LAYERS, LLAMA_TARGETS)) / len(LFM2_LAYERS)
    qwen = len(adapted_layers(QWEN_LAYERS, LLAMA_TARGETS)) / len(QWEN_LAYERS)
    assert lfm < 0.5 and qwen == 1.0


# --------------------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("name,spec,expected_layers", [
    ("lfm2", LFM2_LAYERS, 16),
    ("qwen", QWEN_LAYERS, 24),
    ("smollm3", SMOLLM3_LAYERS, 36),
])
@needs_torch
@needs_torch
def test_discovery_reaches_every_layer(name, spec, expected_layers):
    model = build_stack(spec)
    assert count_layers(model) == expected_layers
    targets = discover_targets(model)
    covered = adapted_layers(spec, targets)
    assert len(covered) == expected_layers, (
        f"{name}: uncovered layers {sorted(set(range(expected_layers)) - covered)}")


@needs_torch
@needs_torch
def test_discovery_finds_the_real_lfm2_names():
    targets = set(discover_targets(build_stack(LFM2_LAYERS)))
    assert {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
            "self_attn.out_proj"} <= targets
    assert {"feed_forward.w1", "feed_forward.w2",
            "feed_forward.w3"} <= targets, "the feed-forward stack must be adapted"


@needs_torch
def test_targets_are_dotted_so_a_shared_leaf_name_is_unambiguous():
    """LFM2 has both self_attn.out_proj and conv.out_proj.

    A bare ``out_proj`` requests both, but policy adapts only the attention one, so
    coverage reads 6 of 16 and fails. Qualifying the target with its parent block makes
    the request match exactly what policy selects.
    """
    targets = discover_targets(build_stack(LFM2_LAYERS))
    assert "self_attn.out_proj" in targets
    assert "out_proj" not in targets
    assert not any(t.startswith("conv.") for t in targets), "conv is frozen by policy"


@needs_torch
def test_policy_freezes_the_alternative_token_mixer_in_both_families():
    """The two hybrids must be treated alike, or the family comparison is confounded."""
    lfm = discover_targets(build_stack(LFM2_LAYERS))
    qwen = discover_targets(build_stack(QWEN_LAYERS))
    assert not any("conv" in t for t in lfm)
    assert not any("linear_attn" in t for t in qwen)
    # Each family keeps its ordinary attention and its full feed-forward stack.
    assert any(t.startswith("self_attn.") for t in lfm)
    assert any(t.startswith("self_attn.") for t in qwen)
    assert sum(t.startswith("feed_forward.") for t in lfm) == 3
    assert sum(t.startswith("mlp.") for t in qwen) == 3


@needs_torch
def test_survey_reports_the_blocks_policy_excludes():
    from sentalign.lora_targets import survey_blocks

    survey = survey_blocks(build_stack(LFM2_LAYERS))
    assert survey["token_mixer"]["conv.in_proj"] == 10
    assert survey["token_mixer"]["conv.out_proj"] == 10
    assert survey["mlp"]["feed_forward.w1"] == 16
    assert survey["attention"]["self_attn.q_proj"] == 6


@needs_torch
@needs_torch
def test_discovery_finds_the_real_qwen_names():
    targets = set(discover_targets(build_stack(QWEN_LAYERS)))
    assert {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
            "self_attn.o_proj"} <= targets
    assert {"mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"} <= targets


@needs_torch
@needs_torch
def test_discovery_excludes_norms_head_and_embeddings():
    targets = discover_targets(build_stack(LFM2_LAYERS))
    assert not any(is_excluded(t) for t in targets)
    assert "lm_head" not in targets and "embed_tokens" not in targets
    assert not any("norm" in t for t in targets)


@needs_torch
@needs_torch
def test_discovery_is_deterministic():
    model = build_stack(QWEN_LAYERS)
    assert discover_targets(model) == discover_targets(model) == sorted(discover_targets(model))


# --------------------------------------------------------------------------------------
# The two forms a target list has to take
# --------------------------------------------------------------------------------------

#: The block tags in ``unsloth_zoo.peft_utils.get_peft_regex``, with
#: finetune_attention_modules and finetune_mlp_modules both on.
UNSLOTH_TAGS = ("self_attn", "attention", "attn", "mixer",
                "mlp", "feed_forward", "ffn", "dense")


def unsloth_regex(targets) -> str:
    """The matcher Unsloth compiles from a target list, and hands to peft as a regex.

    For a text model its vision and language branches drop out and this is what is left:
    a block tag, then a name from the list. peft applies it with ``re.fullmatch``.
    """
    return (r".*?(?:" + "|".join(UNSLOTH_TAGS) + r").*?(?:"
            + "|".join(re.escape(t) for t in targets) + r").*?")


def linear_names(model) -> list[str]:
    import torch.nn as nn

    return [name for name, module in model.named_modules() if isinstance(module, nn.Linear)]


@needs_torch
def test_dotted_targets_match_nothing_under_unsloth():
    """Reproduces "Unsloth: No layers to finetune?".

    Its regex puts a block tag before the requested name, so a dotted target asks for
    ``self_attn`` twice in one module path. Nothing matches and it raises.
    """
    model = build_stack(LFM2_LAYERS)
    pattern = unsloth_regex(discover_targets(model))
    assert not any(re.fullmatch(pattern, name) for name in linear_names(model))


@needs_torch
@pytest.mark.parametrize("name,spec,frozen", [
    ("lfm2", LFM2_LAYERS, "conv"),
    ("qwen", QWEN_LAYERS, "linear_attn"),
    ("smollm3", SMOLLM3_LAYERS, "linear_attn"),     # no mixer to freeze; must stay empty
])
def test_leaf_targets_select_exactly_the_policy_set_under_unsloth(name, spec, frozen):
    """And the leaf form Unsloth needs happens to select what policy selects.

    Its tags do the block selection, not the target list, so this has to be checked
    rather than assumed: ``attn`` is one of its tags and matches Qwen's ``linear_attn``,
    which policy freezes. It is the leaf names that differ there (``in_proj_qkv``, not
    ``q_proj``), and that is a property of these two architectures, not a guarantee.
    """
    model = build_stack(spec)
    dotted = discover_targets(model)
    pattern = unsloth_regex(leaf_targets(dotted))
    matched = {n for n in linear_names(model) if re.fullmatch(pattern, n)}

    blocks = {t.split(".")[0] for t in dotted}
    expected = {n for n in linear_names(model)
                if any(f".{block}." in n for block in blocks)}
    assert matched == expected
    assert not any(f".{frozen}." in n for n in matched), "the token mixer stays frozen"
    assert "lm_head" not in matched


@needs_torch
def test_leaf_targets_are_the_dotted_ones_with_the_block_dropped():
    assert leaf_targets(discover_targets(build_stack(LFM2_LAYERS))) == [
        "k_proj", "out_proj", "q_proj", "v_proj", "w1", "w2", "w3"]
    assert leaf_targets(discover_targets(build_stack(QWEN_LAYERS))) == [
        "down_proj", "gate_proj", "k_proj", "o_proj", "q_proj", "up_proj", "v_proj"]


# --------------------------------------------------------------------------------------
# Coverage verification
# --------------------------------------------------------------------------------------

def _attach(model, targets):
    import peft
    config = peft.LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, bias="none",
                             target_modules=list(targets), task_type=None)
    return peft.get_peft_model(model, config)


@needs_peft
@needs_peft
def test_verify_coverage_flags_the_llama_list_on_lfm2():
    wrapped = _attach(build_stack(LFM2_LAYERS), LLAMA_TARGETS)
    report = verify_coverage(wrapped, LLAMA_TARGETS)
    assert not report.complete
    assert len(report.uncovered) == 10, report.summary()


@needs_peft
def test_verify_coverage_passes_with_discovered_targets():
    model = build_stack(LFM2_LAYERS)
    targets = discover_targets(model)
    report = verify_coverage(_attach(model, targets), targets)
    assert report.complete, report.summary()
    assert report.trainable_params > 0
    # One entry per adapted module: peft exposes both `lora_A` and `lora_A.default`, and
    # counting each owner twice would put a doubled module count in the manifest.
    assert len(report.adapted_modules) == len(set(report.adapted_modules))
    assert len(report.adapted_modules) == 6 * 4 + 16 * 3     # attention x6, ffn x16


@needs_peft
@needs_peft
def test_discovered_targets_train_far_more_parameters():
    """The practical consequence: the corrected configuration adapts the feed-forward
    stack, which is where most of the parameters live."""
    llama = _attach(build_stack(LFM2_LAYERS), LLAMA_TARGETS)
    llama_params = sum(p.numel() for p in llama.parameters() if p.requires_grad)

    model = build_stack(LFM2_LAYERS)
    discovered = _attach(model, discover_targets(model))
    discovered_params = sum(p.numel() for p in discovered.parameters() if p.requires_grad)
    assert discovered_params > 3 * llama_params, (llama_params, discovered_params)


@needs_torch
def test_resolve_falls_back_to_a_configured_list_only_when_discovery_is_empty():
    import torch

    bare = torch.nn.Module()
    with pytest.raises(LoraCoverageError, match="no linear projections"):
        resolve_and_verify(bare, configured=None)
    assert resolve_and_verify(bare, configured=("q_proj",)) == ["q_proj"]

    # On a model discovery can read, the configured Llama list is ignored entirely: the
    # result is the dotted names policy selects, with the token mixer still frozen.
    model = build_stack(QWEN_LAYERS)
    resolved = resolve_and_verify(model, configured=LLAMA_TARGETS)
    assert resolved == discover_targets(model)
    assert "self_attn.o_proj" in resolved and "o_proj" not in resolved
    assert not any("linear_attn" in t for t in resolved)
    # Only with discovery switched off does the configured list win.
    assert resolve_and_verify(model, configured=LLAMA_TARGETS,
                              auto=False) == list(LLAMA_TARGETS)


@needs_peft
@needs_peft
def test_coverage_report_summary_names_the_gap():
    wrapped = _attach(build_stack(LFM2_LAYERS), LLAMA_TARGETS)
    text = verify_coverage(wrapped, LLAMA_TARGETS).summary()
    assert "UNCOVERED" in text and "6/16 layers" in text


# --------------------------------------------------------------------------------------
# Per-target attachment
# --------------------------------------------------------------------------------------

@needs_peft
def test_an_absent_target_is_not_a_coverage_failure():
    """Per-layer coverage alone is too weak.

    An adapter library can drop a requested target entirely, and if another target in the
    same layer succeeded, every layer still reports as covered. Unsloth does exactly this:
    it intersects an explicit target list with its own finetune_* filters and attaches
    only where both select.
    """
    model = build_stack(LFM2_LAYERS)
    real = discover_targets(model)
    # Request a target the model does not expose at all alongside the real ones.
    wrapped = _attach(model, real)
    report = verify_coverage(wrapped, real + ["mlp.gate_proj"])

    assert report.available_per_target.get("mlp.gate_proj", 0) == 0, "not in this model"
    assert not report.unattached_targets, "absent modules are not a coverage failure"
    assert report.complete


@needs_peft
def test_partial_attachment_is_reported_per_target():
    """A target present in 16 layers but adapted in none must not read as complete."""
    model = build_stack(LFM2_LAYERS)
    wrapped = _attach(model, ["self_attn.q_proj"])
    report = verify_coverage(wrapped, ["self_attn.q_proj", "feed_forward.w1"])

    assert report.per_target.get("self_attn.q_proj") == 6
    assert report.available_per_target.get("feed_forward.w1") == 16
    assert "feed_forward.w1" in report.unattached_targets, report.summary()
    assert not report.complete


@needs_peft
def test_a_bare_leaf_name_reproduces_the_observed_6_of_16():
    """The exact reading from the failing run, and why dotted targets fix it.

    The request and the attachment come apart because they are made by different things.
    The run asked for a bare ``out_proj``; Unsloth intersects that request with its own
    finetune_* filters and attaches inside attention only, so ``conv.out_proj`` in the
    other ten layers was requested and never adapted. Plain peft has no such filter, so
    the policy attachment is written out here rather than left to the library.
    """
    model = build_stack(LFM2_LAYERS)
    wrapped = _attach(model, discover_targets(model))   # attention and feed-forward only
    bare = verify_coverage(wrapped, ["out_proj"])
    assert bare.available_per_target["out_proj"] == 16   # self_attn and conv both match
    assert bare.per_target["out_proj"] == 6              # only attention attached
    assert bare.partially_attached["out_proj"] == (6, 16)
    assert not bare.complete, bare.summary()

    dotted = verify_coverage(wrapped, ["self_attn.out_proj"])
    assert dotted.available_per_target["self_attn.out_proj"] == 6
    assert dotted.per_target["self_attn.out_proj"] == 6
    assert dotted.complete


@needs_peft
def test_an_adapter_in_a_frozen_block_fails_coverage():
    """The other direction of the same defect, and the risk the leaf form carries.

    A bare ``out_proj`` reaches ``conv.out_proj`` as well. Adapting the convolution gives
    LFM2 a token mixer Qwen does not get, which confounds the family comparison exactly
    as under-adapting does, so coverage must refuse it rather than read as complete.
    """
    model = build_stack(LFM2_LAYERS)
    wrapped = _attach(model, ["out_proj", "q_proj", "k_proj", "v_proj", "w1", "w2", "w3"])
    report = verify_coverage(wrapped, discover_targets(build_stack(LFM2_LAYERS)))

    assert report.off_policy_modules == {"conv.out_proj": 10}, report.summary()
    assert not report.uncovered and not report.unattached_targets
    assert not report.complete, "every layer is covered, and it is still wrong"
    assert "OFF-POLICY" in report.summary()


@needs_peft
def test_blocks_argument_decides_what_counts_as_off_policy():
    """Freezing the mixer is a policy, not a law: widen the policy and it is legitimate."""
    model = build_stack(LFM2_LAYERS)
    wrapped = _attach(model, ["out_proj", "q_proj", "k_proj", "v_proj", "w1", "w2", "w3"])
    targets = discover_targets(build_stack(LFM2_LAYERS))
    report = verify_coverage(wrapped, targets,
                             blocks=("attention", "mlp", "token_mixer"))
    assert not report.off_policy_modules
    assert report.complete, report.summary()


@needs_peft
def test_summary_names_the_unattached_target():
    model = build_stack(LFM2_LAYERS)
    wrapped = _attach(model, ["self_attn.q_proj"])
    text = verify_coverage(
        wrapped, ["self_attn.q_proj", "feed_forward.w1"]).summary()
    assert "REQUESTED BUT NEVER ATTACHED" in text
    assert "feed_forward.w1" in text and "16" in text


@needs_peft
def test_full_discovery_reports_complete():
    model = build_stack(QWEN_LAYERS)
    targets = discover_targets(model)
    report = verify_coverage(_attach(model, targets), targets)
    assert report.complete, report.summary()
    assert not report.unattached_targets and not report.partially_attached
    for target in targets:
        got = report.per_target.get(target, 0)
        available = report.available_per_target.get(target, 0)
        assert got == available, f"{target}: {got}/{available}"


@needs_peft
def test_an_adapter_outside_the_decoder_stack_fails_coverage():
    """The gate promises adapters landed where, and only where, policy selects.

    An adapter on the output head sits outside every ``layers.N.`` path, so it appears in
    ``trainable_params`` while evading ``per_target``, ``layers_covered``, and the
    off-policy check. A configured list naming ``lm_head`` with discovery off is enough
    to produce one, and the report must refuse it rather than read complete.
    """
    model = build_stack(LFM2_LAYERS)
    targets = discover_targets(model)
    wrapped = _attach(model, targets + ["lm_head"])
    report = verify_coverage(wrapped, targets)

    assert not report.uncovered and not report.unattached_targets
    assert any(key.endswith("lm_head") for key in report.off_policy_modules), (
        report.summary())
    assert not report.complete, "an off-stack adapter must not read as complete"


@needs_torch
def test_a_dense_transformer_gets_the_same_intervention_as_the_hybrids():
    """SmolLM3 has nothing to freeze, so policy adapts attention and MLP in all 36 layers.

    For this architecture the Llama list is exactly right, which is the point: the three
    families receive one intervention, and only the hybrids have a block it leaves frozen.
    """
    from sentalign.lora_targets import survey_blocks

    model = build_stack(SMOLLM3_LAYERS)
    targets = discover_targets(model)
    assert leaf_targets(targets) == sorted(LLAMA_TARGETS)
    assert len(adapted_layers(SMOLLM3_LAYERS, targets)) == 36
    survey = survey_blocks(model)
    assert not survey.get("token_mixer")
    assert survey["attention"]["self_attn.q_proj"] == 36
    assert survey["mlp"]["mlp.down_proj"] == 36

