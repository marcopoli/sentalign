"""The label space, verbalizers, and prompt construction.

Two decisions here are load-bearing and are the reason this lives in its own module:

1.  The instruction does **not** enumerate the label words. The v1 prompt read
    "classify its sentiment as negative, neutral, or positive", which makes any
    first-match parser return "negative" for every example whenever the prompt is
    included in the decoded string (AUDIT S2-6). The label set is communicated by
    the answer format instead, and the parser is anchored on ``Sentiment:``.

2.  ``mixed`` is a first-class label, not a nuisance class. DynaSent annotators use it
    for 3,900 / 94,459 round-1 and 3,334 / 18,535 round-2 training items. Collapsing it
    into ``neutral`` is the modelling choice that erases exactly the phenomenon this
    study is about, so it is a configurable ``LabelSpace`` rather than a hard-coded list.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Mapping, Sequence

# DynaSent's four annotator options. ``no-majority`` is a derived state (no option
# reaches 3/5), not something an annotator can select.
POSITIVE = "positive"
NEGATIVE = "negative"
NEUTRAL = "neutral"
MIXED = "mixed"
NO_MAJORITY = "no-majority"

ANNOTATOR_OPTIONS: tuple[str, ...] = (POSITIVE, NEGATIVE, NEUTRAL, MIXED)


# --------------------------------------------------------------------------------------
# Prompt surfaces, one set per task
# --------------------------------------------------------------------------------------
#
# These sit above ``LabelSpace`` because a label space carries its own prompting: the
# instruction, the template, and the anchor word the parser keys on are properties of the
# task, not globals. A second task (NLI) can then be added as another ``LabelSpace``
# rather than by threading a parallel "task" argument through every call site.

INSTRUCTION = (
    "Read the text and judge the sentiment the author expresses.\n"
    "Answer on one line in the form `Sentiment: <label>`."
)

#: ``{anchor}`` formats to "Sentiment" here, so this is byte-identical to the v1 template
#: and every prompt built for the completed sentiment runs is unchanged.
PROMPT_TEMPLATE = "{instruction}\n\nText: {text}\n{anchor}:"

NLI_INSTRUCTION = (
    "Read the premise and decide whether the hypothesis follows from it.\n"
    "Answer on one line in the form `Answer: <label>`."
)

#: NLI carries two fields, so the premise/hypothesis block is built into ``Item.text``
#: and the template stays single-slot.
NLI_TEMPLATE = "{instruction}\n\n{text}\n{anchor}:"


@dataclass(frozen=True)
class LabelSpace:
    """A closed set of labels plus the surface strings the model is scored against.

    ``verbalizers`` maps each label to the exact continuation used for constrained
    scoring. Keep them short and mutually non-prefixing: if one verbalizer is a token
    prefix of another, length-normalised scoring becomes ill-defined.
    """

    name: str
    labels: tuple[str, ...]
    verbalizers: Mapping[str, str]
    collapse: Mapping[str, str] = field(default_factory=dict)
    #: Prompting. Defaults are the sentiment surfaces, so existing spaces are unchanged.
    instruction: str = INSTRUCTION
    template: str = PROMPT_TEMPLATE
    anchor: str = "Sentiment"

    def __post_init__(self) -> None:
        missing = [y for y in self.labels if y not in self.verbalizers]
        if missing:
            raise ValueError(f"{self.name}: no verbalizer for {missing}")
        surfaces = [self.verbalizers[y].lower() for y in self.labels]
        if len(set(surfaces)) != len(surfaces):
            raise ValueError(f"{self.name}: duplicate verbalizers {surfaces}")
        for i, a in enumerate(surfaces):
            for j, b in enumerate(surfaces):
                if i != j and b.startswith(a):
                    raise ValueError(
                        f"{self.name}: verbalizer {a!r} is a prefix of {b!r}; "
                        "length-normalised scoring would be ill-defined"
                    )

    @property
    def size(self) -> int:
        return len(self.labels)

    def index(self, label: str) -> int:
        return self.labels.index(self.normalise(label))

    def normalise(self, label: str) -> str:
        """Map an annotator option into this space, applying any collapse rules."""
        label = label.strip().lower()
        label = self.collapse.get(label, label)
        if label not in self.labels:
            raise KeyError(f"{label!r} is not in label space {self.name}: {self.labels}")
        return label

    def contains(self, label: str | None) -> bool:
        if label is None:
            return False
        try:
            self.normalise(label)
        except KeyError:
            return False
        return True


#: Four-way space. The default: it keeps ``mixed`` distinct, which is what H3 needs.
TERNARY_PLUS_MIXED = LabelSpace(
    name="ternary+mixed",
    labels=(NEGATIVE, NEUTRAL, POSITIVE, MIXED),
    verbalizers={NEGATIVE: "negative", NEUTRAL: "neutral",
                 POSITIVE: "positive", MIXED: "mixed"},
)

#: Three-way space matching DynaSent's official dev/test protocol, where ``mixed`` items
#: are filtered out rather than relabelled. Used for comparability with published numbers.
TERNARY = LabelSpace(
    name="ternary",
    labels=(NEGATIVE, NEUTRAL, POSITIVE),
    verbalizers={NEGATIVE: "negative", NEUTRAL: "neutral", POSITIVE: "positive"},
)

#: Three-way space that folds ``mixed`` into ``neutral``. Reported as an ablation only , 
#: it is the conventional choice and it destroys the signal H3 tests, so we quantify the cost.
TERNARY_COLLAPSED = LabelSpace(
    name="ternary-collapsed",
    labels=(NEGATIVE, NEUTRAL, POSITIVE),
    verbalizers={NEGATIVE: "negative", NEUTRAL: "neutral", POSITIVE: "positive"},
    collapse={MIXED: NEUTRAL},
)

#: Natural language inference. A second task, added to test whether the sentiment
#: result is dataset-specific and, more importantly, to get a target distribution
#: estimated from 100 annotators rather than 5. Verbalised as yes/maybe/no because
#: "entailment" and "contradiction" are multi-token under both tokenizers, and P1 needs
#: every verbalizer to be a single token.
ENTAILMENT = "entailment"
CONTRADICTION = "contradiction"

NLI3 = LabelSpace(
    name="nli3",
    labels=(ENTAILMENT, NEUTRAL, CONTRADICTION),
    verbalizers={ENTAILMENT: "yes", NEUTRAL: "maybe", CONTRADICTION: "no"},
    instruction=NLI_INSTRUCTION,
    template=NLI_TEMPLATE,
    anchor="Answer",
)

LABEL_SPACES: Mapping[str, LabelSpace] = {
    ls.name: ls for ls in (TERNARY_PLUS_MIXED, TERNARY, TERNARY_COLLAPSED, NLI3)
}

# --------------------------------------------------------------------------------------
# Prompting
# --------------------------------------------------------------------------------------

#: Parsers are derived from the space rather than hard-coded, so a space whose
#: verbalizers are "yes|maybe|no" gets a parser for those and not for the sentiment words.
#: Cached by space name: ``parse_completion`` runs once per generated item.
_PATTERNS: dict[str, tuple[re.Pattern, re.Pattern]] = {}


def answer_patterns(space: "LabelSpace") -> tuple[re.Pattern, re.Pattern]:
    """The anchored parser and the bare-continuation fallback for this space.

    The alternation is sorted longest-first so that a verbalizer which shares a prefix
    with another cannot shadow it. ``LabelSpace`` already forbids one verbalizer being a
    prefix of another, so this is belt and braces rather than load-bearing.
    """
    cached = _PATTERNS.get(space.name)
    if cached is not None:
        return cached
    surfaces = sorted((re.escape(space.verbalizers[y]) for y in space.labels),
                      key=len, reverse=True)
    alt = "|".join(surfaces)
    anchored = re.compile(rf"{re.escape(space.anchor.lower())}\s*:\s*\**\s*({alt})\b",
                          re.IGNORECASE)
    bare = re.compile(rf"^\W{{0,4}}({alt})\b", re.IGNORECASE)
    _PATTERNS[space.name] = (anchored, bare)
    return anchored, bare


#: Kept as module names because callers and tests import them directly.
ANSWER_PATTERN, BARE_PATTERN = answer_patterns(TERNARY_PLUS_MIXED)

_SURFACE_TO_LABEL: dict[str, dict[str, str]] = {}


def label_for_surface(space: "LabelSpace", surface: str) -> str | None:
    """Map a matched verbalizer back to its label.

    For the sentiment spaces the two coincide, which is why v1 could get away with
    calling ``normalise`` on the match. They do not coincide in general: NLI verbalises
    ``entailment`` as "yes", so parsing has to invert the verbalizer map, not the labels.
    """
    table = _SURFACE_TO_LABEL.get(space.name)
    if table is None:
        table = {space.verbalizers[y].lower(): y for y in space.labels}
        _SURFACE_TO_LABEL[space.name] = table
    return table.get(surface.strip().lower())


def build_prompt(text: str, space: "LabelSpace" = None, *,
                 instruction: str | None = None) -> str:
    """The prompt fed to the model. Ends at ``Sentiment:`` with no trailing space.

    The trailing space belongs to the verbalizer, not the prompt: most BPE tokenizers
    encode " positive" as a single token but "positive" as two, so putting the space on
    the prompt side silently changes the number of scored tokens per label.
    """
    space = TERNARY_PLUS_MIXED if space is None else space
    return space.template.format(
        instruction=space.instruction if instruction is None else instruction,
        text=text.strip(), anchor=space.anchor)


def build_target(label: str, space: LabelSpace = TERNARY_PLUS_MIXED) -> str:
    """The SFT target continuation for ``label``.

    Deliberately minimal. v1 appended a fixed rationale ("This text expresses X
    sentiment.") to every target, which meant the token-level loss was dominated by
    boilerplate and the sampled preference candidates differed in a single token
    (AUDIT S1-3). The label *is* the answer; anything else has to earn its place.
    """
    return " " + space.verbalizers[space.normalise(label)]


def parse_completion(completion: str, space: LabelSpace = TERNARY_PLUS_MIXED) -> str | None:
    """Extract a label from a generated continuation, or ``None`` if it does not parse.

    ``completion`` must be the newly generated text only. Passing prompt+completion is
    the bug this whole module exists to prevent, so we defend against the obvious case.
    """
    if space.instruction.splitlines()[0] in completion:
        raise ValueError(
            "parse_completion received text containing the instruction; pass the "
            "generated continuation only (see AUDIT.md S2-6)"
        )
    for pattern in answer_patterns(space):
        match = pattern.search(completion.strip())
        if match:
            label = label_for_surface(space, match.group(1))
            if label is not None:
                return label
    return None


def verbalizer_sequences(space: LabelSpace = TERNARY_PLUS_MIXED) -> Sequence[str]:
    """Continuations scored during constrained decoding, in label order."""
    return [" " + space.verbalizers[y] for y in space.labels]


# --------------------------------------------------------------------------------------
# Single-token verbalizers
# --------------------------------------------------------------------------------------
#
# When every verbalizer is a single token, the whole label set can be scored from one
# forward pass over the prompt: the logits at the final position already contain a score
# for each label. This is what makes both the CSPO objective and the evaluation scorer
# cost O(1) forward passes instead of O(K), and it is the property that makes the whole
# programme fit on one 24 GB GPU.
#
# It is a property of the tokenizer, not an assumption, so it is resolved and checked at
# run time. Verified for the models used here: " negative", " neutral", " positive", and
# " mixed" are each a single token under the LFM2.5 (64,400 vocab) and Qwen3.5 (248,066
# vocab) tokenizers. Capitalised forms are not always single tokens (" Neutral" is two
# under LFM2.5), which is why the verbalizers are lowercase.


class VerbalizerError(RuntimeError):
    """Raised when a label space cannot be scored in a single forward pass."""


@dataclass(frozen=True)
class VerbalizerTable:
    """Resolved token ids for each label, in label order."""

    space: LabelSpace
    token_ids: tuple[int, ...]
    surfaces: tuple[str, ...]
    single_token: bool

    def __len__(self) -> int:
        return len(self.token_ids)


def resolve_verbalizers(tokenizer, space: LabelSpace = TERNARY_PLUS_MIXED,
                        *, require_single_token: bool = True) -> VerbalizerTable:
    """Map each label to its token id under ``tokenizer``.

    Raises ``VerbalizerError`` when a verbalizer is not a single token and
    ``require_single_token`` is set, rather than silently falling back to a slower and
    differently-normalised scoring path. Callers that can tolerate the fallback pass
    ``require_single_token=False`` and check ``table.single_token``.
    """
    surfaces = tuple(" " + space.verbalizers[y] for y in space.labels)
    # `text=` rather than positional: a multimodal processor takes `images` first, and
    # reads a bare " negative" as an image path (see modeling.text_tokenizer).
    encodings = [tokenizer(text=sfc, add_special_tokens=False)["input_ids"]
                 for sfc in surfaces]

    lengths = [len(ids) for ids in encodings]
    single = all(n == 1 for n in lengths)
    if not single and require_single_token:
        detail = ", ".join(f"{sfc!r}={n} tokens" for sfc, n in zip(surfaces, lengths))
        raise VerbalizerError(
            f"label space {space.name!r} is not single-token under this tokenizer "
            f"({detail}). Single-token verbalizers are required for single-pass "
            "scoring; choose different surface forms or pass "
            "require_single_token=False to use the slower multi-token path.")

    ids = tuple(enc[0] for enc in encodings) if single else tuple(-1 for _ in encodings)
    if single and len(set(ids)) != len(ids):
        raise VerbalizerError(
            f"two labels of {space.name!r} map to the same token id: {ids}")
    return VerbalizerTable(space=space, token_ids=ids, surfaces=surfaces,
                           single_token=single)
