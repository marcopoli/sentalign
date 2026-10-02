"""Zero-shot and few-shot prompting of the base models.

The no-training floor. It answers "how much did the whole pipeline actually buy?" and it
is scored by the same constrained scorer as every trained arm, so the comparison is not
confounded by a different decision rule.

Few-shot demonstrations are drawn from the training split, stratified across labels, with
a fixed seed. Prompt-induced label bias is a known and large effect at this scale
\\cite{zhao2021calibrate}, so we report both the raw and the temperature-scaled numbers
and note that the ordering can change between them.
"""

from __future__ import annotations

import random
from typing import Sequence

import numpy as np

from ..labels import INSTRUCTION, LabelSpace, build_prompt


def build_fewshot_instruction(
    demos: Sequence[tuple[str, str]], space: LabelSpace,
    instruction: str = INSTRUCTION,
) -> str:
    """Prepend labelled demonstrations to the instruction, in the answer format."""
    lines = [instruction, ""]
    for text, label in demos:
        lines.append(f"Text: {text.strip()}")
        lines.append(f"Sentiment: {space.verbalizers[space.normalise(label)]}")
        lines.append("")
    return "\n".join(lines).rstrip()


def sample_demonstrations(
    records: Sequence[dict], space: LabelSpace, n_shots: int, seed: int = 0,
) -> list[tuple[str, str]]:
    """Label-stratified demonstrations, shuffled so label order does not encode a prior."""
    rng = random.Random(seed)
    by_label: dict[str, list[dict]] = {y: [] for y in space.labels}
    for r in records:
        if r.get("label") in by_label:
            by_label[r["label"]].append(r)

    demos: list[tuple[str, str]] = []
    labels = list(space.labels)
    for i in range(n_shots):
        label = labels[i % len(labels)]
        pool = by_label.get(label) or [r for rs in by_label.values() for r in rs]
        if not pool:
            continue
        chosen = rng.choice(pool)
        demos.append((chosen.get("text") or chosen["prompt"], label))
    rng.shuffle(demos)
    return demos


def score_prompted(
    model, tokenizer, space: LabelSpace, texts: Sequence[str],
    *, demos: Sequence[tuple[str, str]] = (), variant: str = "mean",
    batch_size: int = 32, max_length: int = 512,
) -> np.ndarray:
    """Constrained-score an untrained model, optionally with demonstrations."""
    from ..evaluate.scoring import ConstrainedScorer

    instruction = (build_fewshot_instruction(demos, space) if demos else space.instruction)
    scorer = ConstrainedScorer(model=model, tokenizer=tokenizer, space=space,
                               variant=variant, batch_size=batch_size,
                               max_length=max_length)
    prompts = [build_prompt(t, space, instruction=instruction) for t in texts]
    return scorer.score(prompts).logits
