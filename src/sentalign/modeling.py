"""Model loading, Unsloth first, calibrated for a single RTX 3090.

Unsloth is the primary path. Two details of the installed version (2026.6.1) are
load-bearing and are enforced rather than hoped for:

*   ``import unsloth`` must happen before ``transformers`` and ``trl``. Unsloth patches
    the TRL trainers at import time (``_patch_trl_trainer`` is invoked from its GPU
    init), and importing TRL first means the patches never apply. Nothing raises when
    this happens; the run is simply slower and uses more memory, which is the kind of
    silent difference that makes a cost table wrong.
*   Unsloth's fused LoRA path requires ``lora_dropout == 0.0`` and ``bias == "none"``.
    With any other value it prints a notice and falls back to the unfused implementation.
    Dropout is therefore 0.0 by default here, which is also standard for LoRA at these
    ranks.

A plain transformers plus peft path is kept as a fallback so the pipeline is not hostage
to one accelerator library, and both paths produce the same adapter layout.

Hardware target: 24 GB, Ampere, bf16 available, no FP8, FlashAttention 2 available.
Memory for a 2B model at 4-bit with rank-16 adapters and 256-token sequences is roughly
1.2 GB of weights plus 1 to 2 GB of activations, so VRAM is not the binding constraint;
throughput is. That is why sequence length, dataset size, and the number of forward
passes per example are treated as first-class design variables throughout.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: Base models. Scale varies within the LFM2.5 family with architecture held fixed; the
#: Qwen3.5 entry varies family and tokenizer at roughly the top of the scale range that
#: fits the compute budget.
#:
#: ``measured_peak_gb`` is the largest peak this programme has recorded for that model,
#: over 295 lfm-1.2b runs, 110 qwen-2b, 60 lfm-350m and 21 smollm3-3b as of 2026-09-18, and
#: ``min_vram_gb`` is the preflight floor derived from it. The floors were round guesses
#: before that (20 GB for a model that never exceeded 4.7), which is nearly the whole card:
#: it refused any run that would have shared the GPU with another, including the ones this
#: programme deliberately overlaps, while telling the user only that memory was short. The
#: gate admits at 0.8 of the floor, so each floor is set to keep that admission threshold
#: above the model's own recorded peak with room to spare.
MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "lfm-350m": {"hf_id": "LiquidAI/LFM2.5-350M-Base", "params": 0.35e9,
                 "family": "lfm2.5", "role": "scale-ladder",
                 "measured_peak_gb": 3.6, "min_vram_gb": 6.0},
    "lfm-1.2b": {"hf_id": "LiquidAI/LFM2.5-1.2B-Base", "params": 1.2e9,
                 "family": "lfm2.5", "role": "scale-ladder,ablation-host",
                 "measured_peak_gb": 4.7, "min_vram_gb": 8.0},
    "qwen-2b": {"hf_id": "Qwen/Qwen3.5-2B-Base", "params": 2.0e9,
                "family": "qwen3.5", "role": "family-comparison,scale-ladder",
                "measured_peak_gb": 14.5, "min_vram_gb": 20.0},
    # A dense transformer, added because both families above are hybrids and the
    # regulariser result could otherwise be specific to hybrids. Base checkpoint, as for
    # the others: SmolLM3-3B without the suffix is the instruct, dual-mode reasoning model.
    # 36 layers of self_attn.{q,k,v,o}_proj and mlp.{gate,up,down}_proj, no other token
    # mixer, tied embeddings; 3,075,098,624 parameters per the published safetensors index.
    # Runs the five-arm regulariser confirmation set, not the full grid.
    "smollm3-3b": {"hf_id": "HuggingFaceTB/SmolLM3-3B-Base", "params": 3.08e9,
                   "family": "smollm3", "role": "family-comparison,dense-transformer",
                   "measured_peak_gb": 9.9, "min_vram_gb": 15.0,
                   # This config loads with flex_attention, which cannot build a block
                   # mask for a batched generate call. See
                   # apply_inference_attn_implementation for what that costs.
                   "inference_attn_implementation": "sdpa"},
    "deberta": {"hf_id": "microsoft/deberta-v3-base", "params": 0.18e9,
                "family": "encoder", "role": "baseline", "encoder": True},
}

#: Excluded, with the reason recorded so the choice is reviewable rather than implicit.
EXCLUDED_MODELS: dict[str, str] = {
    "google/gemma-4-E2B":
        "any-to-any multimodal checkpoint. Its vision and audio towers occupy VRAM that "
        "this task never uses, and its per-layer-embedding architecture does not expose "
        "the same projection names as the other models, so a LoRA configuration matched "
        "across families is not available. Excluded to keep the family comparison "
        "controlled rather than to avoid difficulty.",
    "LiquidAI/LFM2.5-2.6B-Base":
        "fits in 24 GB but roughly doubles the wall-clock of every arm, which does not "
        "fit the compute budget alongside five seeds. The scale axis is covered by "
        "0.35B, 1.2B, and 2.0B. SmolLM3-3B enters later as a confirmation family on five "
        "arms rather than the full grid, which is what makes a 3B model affordable there.",
}


def ensure_unsloth_first() -> bool:
    """Import Unsloth before TRL and transformers, or explain why it is unavailable.

    Returns True when Unsloth is active. Warns loudly if TRL was already imported, since
    at that point the patches cannot be applied and the run will silently differ from the
    configuration recorded in its manifest.
    """
    silence_lazy_alias_warnings()
    already = [name for name in ("transformers", "trl", "peft") if name in sys.modules]
    try:
        importlib.import_module("unsloth")
    except Exception as exc:
        warnings.warn(f"Unsloth unavailable ({type(exc).__name__}: {exc}); "
                      "falling back to transformers and peft.", RuntimeWarning)
        return False
    if "trl" in already:
        warnings.warn(
            "TRL was imported before Unsloth, so Unsloth's trainer patches did not "
            "apply. Import sentalign.modeling (or unsloth) first. This run will be "
            "slower and use more memory than its manifest implies.", RuntimeWarning)
    return True


#: The one transformers message this codebase suppresses, matched in full.
LAZY_ALIAS_NOISE = "alias will be removed in future versions"


def is_lazy_alias_noise(message: str) -> bool:
    """Whether a log line is transformers' lazy-module alias notice.

    Installing causal-conv1d makes something probe ``causal_conv1d_fn`` and its siblings
    through transformers' lazy module. The lookup succeeds, as the message says by
    returning ``causal_conv1d_update``, but it is announced once per candidate
    module, the ViTMatte and ViTPose image processors among them, so a single process
    emits hundreds of lines about a symbol nothing asked for by that path.
    """
    return LAZY_ALIAS_NOISE in message


def silence_lazy_alias_warnings() -> bool:
    """Filter that one message out of the transformers logger, and nothing else.

    Suppressing a library's warnings is usually how a real problem goes unseen, so this
    matches the exact sentence and leaves every other transformers warning in the log.

    Called twice, for two reasons. The alias notice is logged by ``transformers/__init__``
    on the ``transformers`` logger itself, during the import that ``ensure_unsloth_first``
    performs, so the filter has to be attached to that logger *before* the import.
    ``logging.getLogger`` allows that without importing transformers at all, and the filter
    survives the configuration transformers does later. Records from child loggers, though, reach
    the parent's *handler* without consulting the parent's filters, and that handler does
    not exist until transformers configures itself. So later calls top up whatever
    handlers have appeared since.

    Returns whether this call attached anything.
    """
    global _alias_filter

    import logging

    root = logging.getLogger("transformers")
    first = _alias_filter is None
    if first:
        class _AliasFilter(logging.Filter):
            def filter(self, record: logging.LogRecord) -> bool:
                return not is_lazy_alias_noise(record.getMessage())

        _alias_filter = _AliasFilter()
        root.addFilter(_alias_filter)

    attached = False
    for handler in root.handlers:
        if _alias_filter not in handler.filters:
            handler.addFilter(_alias_filter)
            attached = True
    return first or attached


_alias_filter = None

UNSLOTH_AVAILABLE = ensure_unsloth_first()
silence_lazy_alias_warnings()


@dataclass
class LoraConfigSpec:
    r: int = 16
    alpha: int = 32
    #: Discover the projections to adapt by inspecting the model, rather than trusting a
    #: fixed name list. Both model families here are hybrids whose module names differ
    #: from the Llama convention, and a mismatched list adapts a fraction of the model
    #: without raising. See ``lora_targets`` for the inventories and the consequence.
    auto_discover_targets: bool = True
    #: Refuse to train when a selected target would receive no adapter.
    require_full_coverage: bool = True
    #: Which blocks of a decoder layer receive adapters. Attention and MLP is the standard
    #: LoRA recipe and the only choice that treats the two hybrid families alike; the
    #: alternative token mixer (LFM2's convolution, Qwen3.5's linear attention) is frozen
    #: in both. See ``lora_targets.DEFAULT_ADAPTED_BLOCKS``.
    adapted_blocks: tuple[str, ...] = ("attention", "mlp")
    #: Unsloth's fused path requires exactly 0.0. Any other value silently disables it.
    dropout: float = 0.0
    target_modules: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    bias: str = "none"          # likewise required by the fused path
    use_rslora: bool = False

    def validate_for_unsloth(self) -> list[str]:
        problems = []
        if self.dropout != 0.0:
            problems.append(
                f"lora_dropout={self.dropout} disables Unsloth's fused LoRA kernels "
                "(it requires 0.0)")
        if self.bias != "none":
            problems.append(
                f"bias={self.bias!r} disables Unsloth's fused LoRA kernels "
                "(it requires 'none')")
        return problems


@dataclass
class ModelSpec:
    key: str
    load_in_4bit: bool = True
    #: 256 covers over 99% of DynaSent items. Sequence length is quadratic in attention
    #: and linear in everything else, so this is one of the largest single levers on the
    #: total programme cost.
    max_seq_length: int = 256
    dtype: str | None = None
    lora: LoraConfigSpec = field(default_factory=LoraConfigSpec)
    attn_implementation: str | None = None
    trust_remote_code: bool = True
    random_state: int = 3407

    @property
    def hf_id(self) -> str:
        if self.key not in MODEL_REGISTRY:
            raise KeyError(f"unknown model {self.key!r}; known: {sorted(MODEL_REGISTRY)}")
        return MODEL_REGISTRY[self.key]["hf_id"]

    @property
    def params(self) -> float:
        return MODEL_REGISTRY[self.key]["params"]

    @property
    def is_encoder(self) -> bool:
        return bool(MODEL_REGISTRY[self.key].get("encoder"))


def _torch_dtype(name: str | None):
    import torch

    if name is None:
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return getattr(torch, name)


def load_causal_lm(spec: ModelSpec, *, for_training: bool = True):
    """Load a base model and tokenizer, attaching LoRA when training.

    Returns ``(model, tokenizer, info)`` where ``info`` records which backend was used
    and the resolved LoRA configuration, for the run manifest.
    """
    silence_lazy_alias_warnings()      # a no-op unless transformers arrived after import
    info: dict[str, Any] = {"backend": None, "hf_id": spec.hf_id,
                            "max_seq_length": spec.max_seq_length,
                            "load_in_4bit": spec.load_in_4bit}

    if UNSLOTH_AVAILABLE:
        from unsloth import FastLanguageModel

        problems = spec.lora.validate_for_unsloth()
        if problems:
            warnings.warn("Unsloth fast path disabled: " + "; ".join(problems),
                          RuntimeWarning)
        info["unsloth_fast_path"] = not problems

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=spec.hf_id,
            max_seq_length=spec.max_seq_length,
            dtype=_torch_dtype(spec.dtype),
            load_in_4bit=spec.load_in_4bit,
            full_finetuning=False,
            trust_remote_code=spec.trust_remote_code,
            random_state=spec.random_state,
        )
        info["backend"] = "unsloth"
        tokenizer = text_tokenizer(tokenizer, info)
        if for_training:
            treat_as_text_only(model, info)
            from .lora_targets import leaf_targets

            targets = _resolve_targets(model, spec, info)
            # Unsloth cannot take the dotted names peft matches. It compiles its own regex
            # from its finetune_(vision|language|attention|mlp) tags and the *leaf* names,
            # so `self_attn.q_proj` matches nothing and it raises "No layers to finetune?".
            # Leaves are what it gets; its tags then do the block selection, and the
            # ambiguity that reintroduces (LFM2 has both self_attn.out_proj and
            # conv.out_proj) is caught by the coverage check below, which counts against
            # the dotted names and rejects any adapter in a frozen block.
            #
            # The filters are pinned rather than left default for the same reason: a
            # default that can narrow the target set is the silent mis-targeting this
            # discovery step exists to prevent.
            requested = leaf_targets(targets)
            info["lora_target_leaves"] = requested
            model = FastLanguageModel.get_peft_model(
                model,
                r=spec.lora.r,
                target_modules=requested,
                lora_alpha=spec.lora.alpha,
                lora_dropout=spec.lora.dropout,
                bias=spec.lora.bias,
                use_gradient_checkpointing="unsloth",
                random_state=spec.random_state,
                use_rslora=spec.lora.use_rslora,
                finetune_vision_layers=False,
                finetune_language_layers=True,
                finetune_attention_modules=True,
                finetune_mlp_modules=True,
            )
            _check_coverage(model, targets, spec, info)
    else:
        from transformers import AutoTokenizer

        tokenizer = text_tokenizer(AutoTokenizer.from_pretrained(
            spec.hf_id, trust_remote_code=spec.trust_remote_code), info)
        model = _load_backbone_hf(spec)
        info["backend"] = "transformers+peft"
        if for_training:
            targets = _resolve_targets(model, spec, info)
            treat_as_text_only(model, info)
            model = _attach_lora_hf(model, spec, targets)
            _check_coverage(model, targets, spec, info)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    trainable, total = count_trainable(model)
    info.update({"trainable_params": trainable, "total_params": total,
                 "trainable_fraction": trainable / max(total, 1),
                 "kernel_fast_path": kernel_fast_paths()})
    return model, tokenizer, info


def treat_as_text_only(model, info: dict | None = None) -> str | None:
    """Stop TRL routing a text-only checkpoint down its image-text path.

    ``DPOTrainer`` decides with ``model.config.model_type in
    MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES``. Qwen3.5's ``model_type`` is registered
    there because the family has a vision-language variant, so the text-only base
    checkpoint used here is classified as multimodal and its rows are prepared by
    ``process_row``, which expects a processor and image columns this dataset does not
    have. LFM2 is absent from that mapping, which is why only Qwen failed, on every
    pairwise arm, after roughly twenty seconds.

    This study is text-only for every model, so the registration is simply wrong here.
    Removing the entry is narrow: the flag is read once when a trainer is built, and the
    key is recorded in the manifest so the intervention is visible rather than implicit.

    Call this only on the training path. The removal outlives the load, and Unsloth also
    *loads* Qwen3.5 through ``AutoModelForImageTextToText``, so a process that loads the
    model a second time finds the class unregistered and dies with "Unrecognized
    configuration class". Training processes load once and are unaffected; the prompting
    baselines load every model in one interpreter, and the second qwen load is exactly
    where this surfaced.
    """
    from transformers.models.auto import modeling_auto

    mapping = modeling_auto.MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES
    key = getattr(getattr(model, "config", None), "model_type", None)
    if key is None or key not in mapping:
        return None
    mapping.pop(key)
    if info is not None:
        info["model_type_unregistered_from_image_text"] = key
    return key


def kernel_fast_paths() -> dict[str, bool]:
    """Which forward implementation each loaded architecture will actually take.

    Package presence is not the answer, and recording it alone would be misleading.
    causal-conv1d can be installed and still unusable: on an RTX 3090 Unsloth reports
    "causal_conv1d CUDA kernels not compatible with this GPU" and forces the torch path,
    after which ``is_causal_conv1d_available()`` answers one way outside the pipeline and
    the model branches the other way inside it. What decides the arithmetic is the
    module-level ``is_fast_path_available`` in the loaded modeling module, so that is what
    goes in the manifest.

    Reads ``sys.modules`` only. It never imports anything, which keeps it safe to call
    before the Unsloth import order is established.
    """
    paths: dict[str, bool] = {}
    for name, module in list(sys.modules.items()):
        leaf = name.rsplit(".", 1)[-1]
        if ".models." not in name or not leaf.startswith("modeling_"):
            continue
        flag = getattr(module, "is_fast_path_available", None)
        if isinstance(flag, bool):
            paths[leaf.removeprefix("modeling_")] = flag
    return dict(sorted(paths.items()))


def text_tokenizer(tokenizer, info: dict | None = None):
    """The text tokenizer, unwrapped from a multimodal processor if that is what arrived.

    Qwen3.5's checkpoints ship a processor, not a bare tokenizer. The family shares its
    tokenizer with the VL variants, vision tokens and all, which is also why Unsloth
    reports picking ``<|vision_pad|>`` as the pad token. ``FastLanguageModel`` hands that
    processor back, and ``Qwen3VLProcessor.__call__`` takes ``images`` as its first
    positional parameter. Every ``tokenizer(text, ...)`` call in this codebase then passes
    a label surface form such as " negative" as an image, and it fails inside PIL with
    ``cannot identify image file``, several frames away from anything recognisable.

    This study is text-only, so the inner tokenizer is what every path here wants. The
    unwrapping is recorded in the manifest, because "which tokenizer produced these token
    ids" is exactly the kind of question a verbalizer table makes load-bearing.
    """
    inner = getattr(tokenizer, "tokenizer", None)
    if inner is None or not hasattr(inner, "encode"):
        return tokenizer
    if info is not None:
        info["tokenizer_unwrapped_from"] = type(tokenizer).__name__
    return inner


def _load_backbone_hf(spec: ModelSpec):
    from transformers import AutoModelForCausalLM

    kwargs: dict[str, Any] = {"dtype": _torch_dtype(spec.dtype),
                              "trust_remote_code": spec.trust_remote_code}
    if spec.attn_implementation:
        kwargs["attn_implementation"] = spec.attn_implementation
    if spec.load_in_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=_torch_dtype(spec.dtype),
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True)
    return AutoModelForCausalLM.from_pretrained(spec.hf_id, **kwargs)


def _attach_lora_hf(model, spec: ModelSpec, targets: list[str]):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    if spec.load_in_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    return get_peft_model(model, LoraConfig(
        r=spec.lora.r, lora_alpha=spec.lora.alpha, lora_dropout=spec.lora.dropout,
        target_modules=targets, bias=spec.lora.bias,
        use_rslora=spec.lora.use_rslora, task_type="CAUSAL_LM"))


def _resolve_targets(model, spec: ModelSpec, info: dict) -> list[str]:
    """Which projections to adapt, discovered from the model unless told otherwise."""
    from .lora_targets import resolve_and_verify, survey_blocks

    targets = resolve_and_verify(model, spec.lora.target_modules,
                                 auto=spec.lora.auto_discover_targets,
                                 blocks=spec.lora.adapted_blocks)
    survey = survey_blocks(model)
    info["lora_targets"] = targets
    info["lora_targets_discovered"] = spec.lora.auto_discover_targets
    info["lora_adapted_blocks"] = list(spec.lora.adapted_blocks)
    info["lora_block_survey"] = survey
    # Record what policy excludes, so a frozen block is a documented decision rather than
    # an absence a reader has to notice.
    info["lora_frozen_blocks"] = {block: mods for block, mods in survey.items()
                                  if block not in spec.lora.adapted_blocks}
    return targets


def _check_coverage(model, targets: list[str], spec: ModelSpec, info: dict) -> None:
    """Refuse to train a model whose decoder is only partly adapted.

    This is the gate that would have caught a 350M model reporting 491,520 trainable
    parameters: the adapters reached six of sixteen layers and no feed-forward block at
    all, and the run trained to convergence regardless.

    It also gates the opposite error. The backends take the target list in different
    forms, and the leaf names Unsloth requires are ambiguous where the dotted names peft
    takes are not, so an adapter reaching a frozen block is a real possibility rather than
    a hypothetical one, and is refused here.
    """
    from .lora_targets import LoraCoverageError, verify_coverage

    report = verify_coverage(model, targets, blocks=spec.lora.adapted_blocks)
    info["lora_coverage"] = {
        "n_layers": report.n_layers,
        "layers_covered": len(report.layers_covered),
        "adapted_modules": len(report.adapted_modules),
        "trainable_params": report.trainable_params,
        "uncovered_layers": report.uncovered,
        "per_target": report.per_target,
        "available_per_target": report.available_per_target,
        "unattached_targets": report.unattached_targets,
        "partially_attached": {k: list(v) for k, v in report.partially_attached.items()},
        "off_policy_modules": report.off_policy_modules,
    }
    # Always visible, not only on failure: the parameter count alone does not say which
    # projections were reached, and that is the number that was wrong for a whole run.
    print("LoRA coverage:")
    print(report.summary())
    frozen = info.get("lora_frozen_blocks") or {}
    if frozen:
        print("  frozen by policy (adapted_blocks="
              f"{list(spec.lora.adapted_blocks)}):")
        for block, mods in frozen.items():
            detail = ", ".join(f"{k} x{v}" for k, v in mods.items())
            print(f"    {block:<12} {detail}")
    if report.complete:
        return
    problems = []
    if report.uncovered:
        problems.append(f"{len(report.layers_covered)}/{report.n_layers} layers covered, "
                        f"uncovered: {report.uncovered}")
    if report.unattached_targets:
        problems.append(f"requested but never attached: {report.unattached_targets}")
    if report.partially_attached:
        problems.append("attached in fewer layers than available: "
                        + ", ".join(f"{k} ({got}/{avail})"
                                    for k, (got, avail) in report.partially_attached.items()))
    if report.off_policy_modules:
        problems.append("adapters attached in blocks frozen by policy "
                        f"(adapted_blocks={list(spec.lora.adapted_blocks)}): "
                        + ", ".join(f"{k} (x{n})"
                                    for k, n in report.off_policy_modules.items()))
    message = "LoRA coverage is incomplete: " + "; ".join(problems) + "\n" + report.summary()
    if spec.lora.require_full_coverage:
        raise LoraCoverageError(
            message + "\nThis trains a fraction of the model the configuration implies "
            "and would not raise on its own. Set require_full_coverage=False only with a "
            "recorded reason.")
    warnings.warn(message, RuntimeWarning)


def load_adapter(spec: ModelSpec, adapter_dir: Path, *, merge: bool = False,
                 for_inference: bool = True):
    """Reload a trained adapter for evaluation or as a starting policy."""
    adapter_dir = Path(adapter_dir)
    if UNSLOTH_AVAILABLE:
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(adapter_dir),
            max_seq_length=spec.max_seq_length,
            dtype=_torch_dtype(spec.dtype),
            load_in_4bit=spec.load_in_4bit,
            trust_remote_code=spec.trust_remote_code,
        )
        if for_inference:
            FastLanguageModel.for_inference(model)
        else:
            treat_as_text_only(model)
        return model, text_tokenizer(tokenizer)

    from peft import PeftModel
    from transformers import AutoTokenizer

    tokenizer = text_tokenizer(AutoTokenizer.from_pretrained(adapter_dir))
    model = PeftModel.from_pretrained(_load_backbone_hf(spec), adapter_dir)
    treat_as_text_only(model)      # same reason as the Unsloth branch above
    if merge:
        model = model.merge_and_unload()
    model.eval()
    return model, tokenizer


def display_name(key: str) -> str:
    """The checkpoint name a table should print.

    Derived from the registry rather than typed into each script, so a table cannot name
    a checkpoint the runs did not use. The ``-Base`` suffix is dropped because every model
    in this programme is a base checkpoint, so it is column width spent on a constant.
    """
    entry = MODEL_REGISTRY.get(key)
    if not entry:
        return key
    name = entry["hf_id"].split("/")[-1]
    return name[:-len("-Base")] if name.endswith("-Base") else name


def inference_attn_implementation(spec: ModelSpec) -> str | None:
    """The attention kernel to score with, read from the registry.

    Pinned per family because the kernel is a property of the architecture. A pin that
    lived in the evaluation path instead would apply to whichever model was loaded
    there, which is the shape of every arm-comparability defect in this harness so far.
    """
    if spec.attn_implementation:
        return spec.attn_implementation
    return MODEL_REGISTRY.get(spec.key, {}).get("inference_attn_implementation")


def _attn_configs(model) -> list:
    """The model config plus any sub-config that carries its own attention setting."""
    config = getattr(model, "config", None)
    if config is None:
        return []
    configs = [config]
    for name in ("text_config", "decoder", "encoder"):
        sub = getattr(config, name, None)
        if sub is not None and hasattr(sub, "_attn_implementation"):
            configs.append(sub)
    return configs


def current_attn_implementation(model) -> str | None:
    """Whatever kernel the loaded model will actually use, or None if it does not say."""
    for config in _attn_configs(model):
        name = getattr(config, "_attn_implementation", None)
        if name:
            return str(name)
    return None


def apply_inference_attn_implementation(spec: ModelSpec, model) -> str | None:
    """Pin the attention kernel before scoring, and refuse a pin that did not take.

    SmolLM3 loads with flex_attention. Its forward pass is fine, which is why training
    and the verbalizer scorer never complained, but a batched left-padded ``generate``
    routes through the per-layer-pattern mask builder, which hands torch a mask with
    more dimensions than ``create_block_mask`` unpacks, and the free-generation arm dies
    with "too many values to unpack (expected 4)" after the model has fully loaded.

    Writing the config attribute is not enough to trust on its own: a transformers
    version that ignored it would leave the crashing kernel in place with nothing to
    show that the pin had been tried, so the effective value is read back and a
    disagreement is raised rather than carried into scoring.
    """
    wanted = inference_attn_implementation(spec)
    if not wanted:
        return current_attn_implementation(model)
    setter = getattr(model, "set_attn_implementation", None)
    if callable(setter):
        try:
            setter(wanted)
        except Exception as exc:                     # older transformers, or a refusal
            warnings.warn(f"set_attn_implementation({wanted!r}) failed: {exc}",
                          RuntimeWarning)
    for config in _attn_configs(model):
        try:
            config._attn_implementation = wanted
        except (AttributeError, ValueError):         # read-only on some config classes
            pass
    effective = [getattr(c, "_attn_implementation", None) for c in _attn_configs(model)]
    if not effective or any(name != wanted for name in effective):
        raise RuntimeError(
            f"{spec.key}: asked to score with attn_implementation={wanted!r} but the "
            f"loaded model reports {effective}; scoring would run on the kernel the "
            "registry pinned it away from")
    return wanted


def set_inference_mode(model) -> None:
    """Enable the backend's fast inference path, if it has one."""
    if UNSLOTH_AVAILABLE:
        try:
            from unsloth import FastLanguageModel

            FastLanguageModel.for_inference(model)
            return
        except Exception:
            pass
    model.eval()


def set_training_mode(model) -> None:
    if UNSLOTH_AVAILABLE:
        try:
            from unsloth import FastLanguageModel

            FastLanguageModel.for_training(model)
            return
        except Exception:
            pass
    model.train()


def save_run(model, tokenizer, out_dir: Path, manifest: dict) -> None:
    """Persist adapter, tokenizer, and the manifest that makes the run reproducible."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str)
                                           + "\n", encoding="utf-8")


def count_trainable(model) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def adapter_size_mb(adapter_dir: Path) -> float:
    """On-disk adapter size, for the deployment cost table."""
    adapter_dir = Path(adapter_dir)
    total = sum(p.stat().st_size for p in adapter_dir.glob("adapter_model.*"))
    return round(total / 1e6, 2)
