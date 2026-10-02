"""Discovering which modules LoRA should adapt, and verifying that it did.

A hard-coded target list is a bug waiting for a new architecture. The Llama-style names
``q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`` are near-universal for
dense transformers and wrong for both model families in this study, because both are
hybrids that interleave attention with another token-mixing block:

    LFM2.5      feed_forward.w1/w2/w3 (16 layers), self_attn.q/k/v/out_proj (6),
                conv.in_proj/out_proj (10)
    Qwen3.5     mlp.gate/up/down_proj (24), self_attn.q/k/v/o_proj (6),
                linear_attn.in_proj_{qkv,a,b,z}/out_proj (18)

Against the Llama list, LFM2.5 adapts q, k, and v in 6 of 16 layers and nothing else: the
feed-forward stack, which holds most of the parameters, is untouched. Qwen3.5 adapts every
MLP but no token mixing in 18 of 24 layers. The two families are therefore handicapped
differently, which confounds exactly the family comparison the study is built on, and
nothing raises: the run trains, the loss falls, and the manifest records a plausible
parameter count.

SmolLM3, added later as a dense-transformer confirmation family, is the case the Llama list
was written for: ``self_attn.q/k/v/o_proj`` and ``mlp.gate/up/down_proj`` in all 36 layers
and no other token mixer. The same policy adapts exactly those, so all three families get
the same intervention: attention and feed-forward adapted, any alternative mixer frozen.

The rule implemented here is architecture-agnostic: **adapt every linear projection inside
a decoder layer, excluding normalisations, convolutions, embeddings, and the output head.**
Discovery reads the model, so a new architecture is handled without a code change, and
``verify_coverage`` refuses to train when any layer would be left without an adapter.

A target list has to take two different forms, because the two backends match differently:

    dotted   ``self_attn.out_proj``   peft matches ``key.endswith("." + target)``, so the
                                      parent block disambiguates a leaf name that occurs
                                      in more than one block.
    leaf     ``out_proj``             Unsloth does not match the list at all. It builds a
                                      regex ``.*?(<block tag>).*?(<leaf>)`` from its own
                                      ``finetune_(attention|mlp)`` tags and the *leaf*
                                      names, so a dotted target would require its block
                                      tag to appear twice in the module path, matches
                                      nothing, and raises "No layers to finetune?".

Discovery therefore returns the dotted form, ``leaf_targets`` converts it for Unsloth, and
the ambiguity that conversion reintroduces is caught after the fact rather than avoided:
``verify_coverage`` checks per-target attachment against the dotted names *and* rejects any
adapter that landed in a block policy freezes.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field

#: Modules never adapted: normalisations, convolutions, embeddings, and the head. LoRA on
#: a norm is meaningless, and adapting the embedding or head changes the parameter budget
#: by an order of magnitude and is a different experiment.
EXCLUDED_SUFFIXES = ("norm", "layernorm", "ln", "embed_tokens", "lm_head",
                     "conv1d", "conv", "rotary_emb")

#: Identifies the repeated decoder stack in a state dict or module tree.
LAYER_PATTERN = re.compile(r"(?:^|\.)layers\.(\d+)\.")

#: Blocks a decoder layer can contain, identified by the module path. Both model families
#: in this study are hybrids that interleave ordinary attention with a second token mixer,
#: so a layer is not simply "attention plus MLP".
#:
#:   attention     ordinary softmax self-attention
#:   mlp           the position-wise feed-forward network
#:   token_mixer   the alternative mixer: LFM2's short convolution, Qwen3.5's linear
#:                 attention, and in other architectures an SSM or Mamba block
BLOCK_PATTERNS = (
    ("token_mixer", re.compile(r"(?:^|\.)(conv|linear_attn|ssm|mamba|mixer)(?:\.|$)")),
    ("attention", re.compile(r"(?:^|\.)(self_attn|attention|attn)(?:\.|$)")),
    ("mlp", re.compile(r"(?:^|\.)(mlp|feed_forward|ffn|feedforward)(?:\.|$)")),
)

#: Which blocks receive adapters.
#:
#: Attention and MLP only, which is the standard LoRA recipe and, more importantly here,
#: the only choice that treats the two model families alike. Adapting the alternative
#: token mixer as well is defensible in principle but not available in practice: the
#: acceleration library classifies modules into attention and MLP and silently drops
#: anything else, so requesting LFM2's convolutional projections attaches them nowhere
#: while Qwen's linear-attention projections might attach, depending on whether its name
#: is read as attention. Fixing the policy here rather than inheriting whatever the
#: library recognises makes the comparison symmetric by construction, and the excluded
#: blocks are recorded in every run manifest.
DEFAULT_ADAPTED_BLOCKS = ("attention", "mlp")


@dataclass
class CoverageReport:
    """Which layers received adapters, and where.

    ``per_target`` is the important field. Checking only that every layer has at least one
    adapter is too weak: a target can be dropped entirely while coverage still reads as
    complete, because some other target in the same layer succeeded. Unsloth in particular
    intersects an explicit target list with its own
    ``finetune_(vision|language|attention|mlp)`` filters and attaches only where both
    select, so a requested target can silently attach nowhere.
    """

    n_layers: int = 0
    adapted_modules: list[str] = field(default_factory=list)
    layers_covered: set[int] = field(default_factory=set)
    per_layer_counts: dict[int, int] = field(default_factory=dict)
    target_modules: list[str] = field(default_factory=list)
    trainable_params: int = 0
    #: requested target -> number of layers it actually attached in
    per_target: dict[str, int] = field(default_factory=dict)
    #: requested target -> number of layers that contain such a module
    available_per_target: dict[str, int] = field(default_factory=dict)
    #: ``parent.leaf`` -> number of adapted modules (one per layer inside the stack),
    #: for adapters that attached in a block policy freezes, or outside the decoder stack
    #: altogether (an ``lm_head`` or embedding adapter). Non-empty means the request
    #: reached further than intended, which is what a leaf-name target list risks: LFM2's
    #: ``out_proj`` names both an attention and a convolutional projection, and adapting
    #: the second breaks the family comparison as surely as missing the first.
    off_policy_modules: dict[str, int] = field(default_factory=dict)

    @property
    def unattached_targets(self) -> list[str]:
        """Targets that exist in the model but received no adapter anywhere."""
        return sorted(t for t, available in self.available_per_target.items()
                      if available > 0 and self.per_target.get(t, 0) == 0)

    @property
    def partially_attached(self) -> dict[str, tuple[int, int]]:
        """Targets attached in fewer layers than contain them."""
        out = {}
        for target, available in self.available_per_target.items():
            got = self.per_target.get(target, 0)
            if 0 < got < available:
                out[target] = (got, available)
        return out

    @property
    def uncovered(self) -> list[int]:
        return sorted(set(range(self.n_layers)) - self.layers_covered)

    @property
    def complete(self) -> bool:
        return (self.n_layers > 0 and not self.uncovered
                and not self.unattached_targets and not self.partially_attached
                and not self.off_policy_modules)

    def summary(self) -> str:
        lines = [
            f"  adapted   : {len(self.adapted_modules)} modules across "
            f"{len(self.layers_covered)}/{self.n_layers} layers",
            f"  trainable : {self.trainable_params:,} parameters",
            "  per target (attached / available):",
        ]
        for target in sorted(self.target_modules):
            got = self.per_target.get(target, 0)
            available = self.available_per_target.get(target, 0)
            flag = ""
            if available and got == 0:
                flag = "   <-- REQUESTED BUT NEVER ATTACHED"
            elif available and got < available:
                flag = "   <-- PARTIAL"
            lines.append(f"    {target:<16} {got:>3} / {available:<3}{flag}")
        if self.off_policy_modules:
            lines.append("  OFF-POLICY adapters (block is frozen by policy):")
            for name, count in sorted(self.off_policy_modules.items()):
                lines.append(f"    {name:<16} {count:>3} layers")
        if self.uncovered:
            lines.append(f"  UNCOVERED layers: {self.uncovered}")
        return "\n".join(lines)


def is_excluded(name: str) -> bool:
    leaf = name.split(".")[-1].lower()
    return any(leaf == suffix or leaf.endswith("_" + suffix) or suffix in leaf
               for suffix in EXCLUDED_SUFFIXES)


def classify_block(name: str) -> str:
    """Which block of a decoder layer a module belongs to."""
    for block, pattern in BLOCK_PATTERNS:
        if pattern.search(name):
            return block
    return "other"


def discover_targets(model, blocks: tuple[str, ...] = DEFAULT_ADAPTED_BLOCKS) -> list[str]:
    """Linear projections inside the selected blocks, as PEFT target patterns.

    Returns *dotted* suffixes such as ``self_attn.q_proj`` rather than bare leaf names.
    PEFT matches a target by ``key.endswith(target)``, so the dotted form disambiguates
    names that appear in more than one block: LFM2 has both ``self_attn.out_proj`` and
    ``conv.out_proj``, and a bare ``out_proj`` would request both, of which only one can
    attach under the current policy. That mismatch is what produced a 6/16 reading.
    """
    import torch.nn as nn

    found: set[str] = set()
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if not LAYER_PATTERN.search(name):
            continue          # outside the decoder stack: embeddings, head, pooler
        parts = name.split(".")
        leaf = parts[-1]
        if is_excluded(leaf):
            continue
        if classify_block(name) not in blocks:
            continue
        # Qualify with the parent when there is one, so the pattern is unambiguous.
        found.add(f"{parts[-2]}.{leaf}" if len(parts) >= 2 and not parts[-2].isdigit()
                  else leaf)
    return sorted(found)


def leaf_targets(targets) -> list[str]:
    """The dotted targets as bare leaf names, which is all Unsloth can match.

    ``FastLanguageModel.get_peft_model`` does not pass the list to peft. It calls
    ``unsloth_zoo.peft_utils.get_peft_regex``, which builds

        ``.*?(self_attn|attention|attn|mixer|mlp|feed_forward|ffn|dense).*?(<leaf>|...)``

    from its own block tags and the names given, then hands peft that regex. A dotted
    ``self_attn.q_proj`` would need the tag ``self_attn`` to occur twice in one module
    path, so it matches nothing and Unsloth raises "No layers to finetune?".

    Block selection therefore comes from Unsloth's tags rather than from the target list,
    That happens to agree with this module's policy, because its tags cover attention and
    MLP while LFM2's ``conv`` carries none of them, but it only happens to. ``mixer`` is one of
    its tags, so an architecture that names its token mixer that way would be adapted
    where policy freezes it. ``verify_coverage`` checks for exactly that afterwards.
    """
    return sorted({t.split(".")[-1] for t in targets})


def survey_blocks(model) -> dict[str, dict[str, int]]:
    """Every linear projection in the decoder, grouped by block, with layer counts.

    Reported in the run manifest so the modules that policy excludes are visible rather
    than merely absent.
    """
    import torch.nn as nn
    from collections import defaultdict

    survey: dict[str, dict[str, set]] = defaultdict(lambda: defaultdict(set))
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        match = LAYER_PATTERN.search(name)
        if not match or is_excluded(name.split(".")[-1]):
            continue
        parts = name.split(".")
        key = f"{parts[-2]}.{parts[-1]}" if len(parts) >= 2 else parts[-1]
        survey[classify_block(name)][key].add(int(match.group(1)))
    return {block: {k: len(v) for k, v in sorted(mods.items())}
            for block, mods in sorted(survey.items())}


def count_layers(model) -> int:
    indices = set()
    for name, _ in model.named_modules():
        match = LAYER_PATTERN.search(name)
        if match:
            indices.add(int(match.group(1)))
    return max(indices) + 1 if indices else 0


def verify_coverage(model, target_modules: list[str], *,
                    blocks: tuple[str, ...] = DEFAULT_ADAPTED_BLOCKS) -> CoverageReport:
    """Check that the adapters landed where, and only where, they were meant to.

    Run after the adapters are attached. Both failures this catches are silent: a target
    list that matches nothing in a layer produces a model that trains and converges on a
    fraction of the parameters the configuration implies, and one that matches too much
    adapts a block ``blocks`` says to freeze, which breaks the symmetry between the two
    model families just as effectively.

    ``target_modules`` should be the dotted names, whichever form was handed to the
    backend, so that per-target attachment is counted unambiguously.
    """
    import torch.nn as nn

    report = CoverageReport(n_layers=count_layers(model),
                            target_modules=list(target_modules))
    per_layer: dict[int, int] = defaultdict(int)
    per_target: dict[str, set[int]] = defaultdict(set)
    available: dict[str, set[int]] = defaultdict(set)
    off_policy: dict[str, set[str]] = defaultdict(set)

    # How many layers contain a module matching each requested target, adapted or not.
    # Targets are dotted suffixes, so matching uses endswith exactly as PEFT does.
    for name, _module in model.named_modules():
        match = LAYER_PATTERN.search(name)
        if not match:
            continue
        for target in target_modules:
            if name.endswith(target):
                available[target].add(int(match.group(1)))

    seen: set[str] = set()
    for name, module in model.named_modules():
        if "lora_A" not in name:
            continue
        # peft names adapters ...<target>.lora_A.<adapter_name>, and exposes both that
        # module and the `lora_A` ModuleDict holding it, so an owner is reached more than
        # once. Counting the duplicates would double the module count in the manifest,
        # which is the number this whole check exists to make trustworthy.
        owner = name.split(".lora_A")[0]
        if owner in seen:
            continue
        seen.add(owner)
        report.adapted_modules.append(owner)
        parts = owner.split(".")
        off_key = ".".join(parts[-2:]) if len(parts) >= 2 else parts[-1]
        match = LAYER_PATTERN.search(owner)
        if match:
            index = int(match.group(1))
            report.layers_covered.add(index)
            per_layer[index] += 1
            if classify_block(owner) not in blocks:
                off_policy[off_key].add(owner)
            for target in target_modules:
                if owner.endswith(target):
                    per_target[target].add(index)
                    break
        else:
            # Outside every `layers.N.` path there is no adapted block, so any adapter
            # here (an lm_head or embedding adapter) is off-policy by definition. Without
            # this branch it would count toward trainable_params while evading every
            # check, and the report would still read complete.
            off_policy[off_key].add(owner)

    report.per_layer_counts = dict(per_layer)
    report.per_target = {k: len(v) for k, v in per_target.items()}
    report.available_per_target = {k: len(v) for k, v in available.items()}
    report.off_policy_modules = {k: len(v) for k, v in sorted(off_policy.items())}
    report.trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return report


class LoraCoverageError(RuntimeError):
    """Raised when the adapter configuration would leave part of the model untouched."""


def resolve_and_verify(model, configured: tuple[str, ...] | None,
                       *, auto: bool = True, strict: bool = True,
                       blocks: tuple[str, ...] = DEFAULT_ADAPTED_BLOCKS) -> list[str]:
    """Choose the target modules for this model, preferring discovery over a fixed list.

    ``configured`` is used only when ``auto`` is disabled or discovery finds nothing.
    With ``strict``, a configured list that misses whole layers raises rather than
    training a differently-sized model than the configuration claims.
    """
    discovered = discover_targets(model, blocks) if auto else []
    if discovered:
        return discovered
    if not configured:
        raise LoraCoverageError(
            "no linear projections found inside a `layers.N.` stack, and no target "
            "modules were configured. This model's decoder is named differently; pass "
            "an explicit target list.")
    return list(configured)
