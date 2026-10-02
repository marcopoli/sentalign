"""Turning a language model into a classifier, once and consistently.

The protocol has exactly one primary decision rule: **single-pass verbalizer scoring**.
The prompt ends at ``Sentiment:``, so the logits at its final position already contain a
score for every label. One forward pass yields the complete label distribution.

Why this matters beyond tidiness:

*   It is the same computation the CSPO objective optimises, so training and evaluation
    agree by construction rather than by convention.
*   It costs one forward pass instead of K. On the 24 GB single-GPU budget this
    programme runs on, that is the difference between evaluation being a rounding error
    and being a third of the compute.
*   Predictions and probabilities come from one computation, so accuracy and ECE describe
    the same system. Taking predictions from free generation while computing calibration
    from a separate constrained pass, which is common, yields an ECE over a distribution
    that never produced the reported predictions.

It requires single-token verbalizers, which is a property of the tokenizer and is checked
at construction rather than assumed. Verified for the models used here. A multi-token
fallback is provided for tokenizers where it does not hold, and the scorer records which
path it took so the two are never silently mixed.

Free generation is measured as a second, clearly labelled arm, since it is what a
deployment runs. Its parser reads only the newly generated tokens, located by token count.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

from ..labels import LabelSpace, VerbalizerTable, build_prompt, parse_completion, resolve_verbalizers

NULL_PROMPT_TEXT = "N/A"


@dataclass
class ScoredBatch:
    """Verbalizer scores for a batch of prompts.

    ``logits`` are unnormalised label scores in label order. Keeping them unnormalised is
    what makes post-hoc temperature scaling and the CSPO reward difference possible.
    """

    logits: np.ndarray
    text_ids: list[str] = field(default_factory=list)
    scoring_path: str = "single_pass"

    @property
    def probs(self) -> np.ndarray:
        z = self.logits - self.logits.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    @property
    def predictions(self) -> np.ndarray:
        return self.logits.argmax(axis=1)


def _import_torch():
    try:
        import torch
    except ImportError as exc:                     # pragma: no cover
        raise ImportError("scoring needs PyTorch; install the train extras") from exc
    return torch


@dataclass
class VerbalizerScorer:
    """Scores the whole label set from the logits at the final prompt position.

    ``variant`` selects the score transform:

    ``raw``   the label logit itself. This is the quantity CSPO operates on.
    ``norm``  log-softmax over the full vocabulary. Differs from ``raw`` by a per-example
              constant, so it changes nothing about predictions or the softmax over
              labels; reported because it is the conventional definition.
    ``pmi``   domain-conditional pointwise mutual information: subtract the label's score
              under a content-free prompt, removing the surface-form frequency advantage
              of common label words (Holtzman et al., 2021).
    """

    model: object
    tokenizer: object
    space: LabelSpace
    variant: str = "norm"
    batch_size: int = 32
    max_length: int = 256
    device: str | None = None
    require_single_token: bool = True
    table: VerbalizerTable | None = None
    #: When set, the scored distribution is a mixture over perspectives read from the
    #: final hidden state, not a softmax of the verbalizer logits. Loaded from the run
    #: directory by the evaluator; None for every other objective.
    mixture_head: object | None = None

    def __post_init__(self) -> None:
        if self.variant not in {"raw", "norm", "pmi"}:
            raise ValueError(f"unknown scoring variant {self.variant!r}")
        torch = _import_torch()
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.table is None:
            self.table = resolve_verbalizers(
                self.tokenizer, self.space,
                require_single_token=self.require_single_token)
        self._null_bias: np.ndarray | None = None

    @property
    def scoring_path(self) -> str:
        return "single_pass" if self.table.single_token else "multi_token"

    # -- core --------------------------------------------------------------------------

    def _final_position_logits(self, prompts: Sequence[str]) -> np.ndarray:
        """Vocabulary logits at the last real token of each prompt, shape (B, V).

        Left padding puts every prompt's final token at the same index, which removes the
        per-row index arithmetic that is the usual source of off-by-one errors here.
        """
        torch = _import_torch()
        self.tokenizer.padding_side = "left"
        enc = self.tokenizer(text=list(prompts), return_tensors="pt", padding=True,
                             truncation=True, max_length=self.max_length,
                             add_special_tokens=True)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            out = self.model(**enc, output_hidden_states=self.mixture_head is not None)
        if self.mixture_head is not None:
            # The trained distribution is a mixture over perspectives, not a softmax of
            # the verbalizer logits. Returning its log-probabilities keeps every caller
            # unchanged: softmax of an already-normalised log q is q.
            import torch

            hidden = out.hidden_states[-1][:, -1, :].float()
            with torch.no_grad():
                log_q = self.mixture_head(hidden)
            return log_q.detach().cpu().numpy().astype(np.float64)
        return out.logits[:, -1, :].float().detach().cpu().numpy().astype(np.float64)

    def _score_single_pass(self, prompts: Sequence[str]) -> np.ndarray:
        vocab_logits = self._final_position_logits(prompts)
        if self.mixture_head is not None:
            # Already the K normalised log-probabilities of the trained mixture. There is
            # no full-vocabulary normaliser to subtract and no column to select.
            return vocab_logits
        label_logits = vocab_logits[:, list(self.table.token_ids)]
        if self.variant == "raw":
            return label_logits
        # Full-vocabulary log-softmax. This subtracts a per-row constant, so it leaves the
        # softmax over labels, and therefore every reported metric, unchanged; it is
        # computed so the numbers match the conventional definition of log pi(y|x).
        m = vocab_logits.max(axis=1, keepdims=True)
        log_z = m + np.log(np.exp(vocab_logits - m).sum(axis=1, keepdims=True))
        return label_logits - log_z

    def _score_multi_token(self, prompts: Sequence[str]) -> np.ndarray:
        """Fallback for tokenizers where a verbalizer spans several tokens.

        Scores the length-normalised log-likelihood of each verbalizer continuation, at a
        cost of K forward passes. Used only when ``require_single_token=False``.
        """
        torch = _import_torch()
        self.tokenizer.padding_side = "right"
        out = np.zeros((len(prompts), self.space.size), dtype=np.float64)
        for k, surface in enumerate(self.table.surfaces):
            cont = self.tokenizer(text=surface, add_special_tokens=False)["input_ids"]
            n_cont = len(cont)
            encoded = [self.tokenizer(text=p, add_special_tokens=True, truncation=True,
                                      max_length=self.max_length - n_cont)["input_ids"]
                       for p in prompts]
            sequences = [ids + cont for ids in encoded]
            max_len = max(len(s) for s in sequences)
            pad = self.tokenizer.pad_token_id
            input_ids = torch.full((len(sequences), max_len), pad, dtype=torch.long)
            attention = torch.zeros((len(sequences), max_len), dtype=torch.long)
            for i, seq in enumerate(sequences):
                input_ids[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
                attention[i, :len(seq)] = 1
            input_ids, attention = input_ids.to(self.device), attention.to(self.device)
            with torch.no_grad():
                logits = self.model(input_ids=input_ids, attention_mask=attention).logits
            log_probs = torch.log_softmax(logits.float(), dim=-1)
            for i, seq in enumerate(sequences):
                start = len(seq) - n_cont
                total = sum(float(log_probs[i, start + j - 1, tok])
                            for j, tok in enumerate(cont))
                out[i, k] = total / n_cont
        return out

    def _null_prompt_bias(self) -> np.ndarray:
        if self._null_bias is None:
            null = build_prompt(NULL_PROMPT_TEXT, self.space)
            scores = (self._score_single_pass([null]) if self.table.single_token
                      else self._score_multi_token([null]))
            self._null_bias = scores[0]
        return self._null_bias

    # -- API ---------------------------------------------------------------------------

    def score(self, prompts: Sequence[str],
              text_ids: Sequence[str] | None = None) -> ScoredBatch:
        out = np.zeros((len(prompts), self.space.size), dtype=np.float64)
        scorer = (self._score_single_pass if self.table.single_token
                  else self._score_multi_token)
        for start in range(0, len(prompts), self.batch_size):
            chunk = list(prompts[start:start + self.batch_size])
            out[start:start + len(chunk)] = scorer(chunk)
        if self.variant == "pmi":
            out = out - self._null_prompt_bias()[None, :]
        return ScoredBatch(logits=out, text_ids=list(text_ids or []),
                           scoring_path=self.scoring_path)

    def score_texts(self, texts: Sequence[str],
                    text_ids: Sequence[str] | None = None) -> ScoredBatch:
        return self.score([build_prompt(t, self.space) for t in texts], text_ids)

    def label_logits(self, prompts: Sequence[str]) -> np.ndarray:
        """Raw label logits, which is what CSPO consumes. Never PMI-adjusted."""
        saved, self.variant = self.variant, "raw"
        try:
            return self.score(prompts).logits
        finally:
            self.variant = saved


#: Backwards-compatible alias. The earlier name described a K-pass implementation that no
#: longer exists; the behaviour is the same but the cost is not.
ConstrainedScorer = VerbalizerScorer


def clear_generation_max_length(model) -> bool:
    """Drop a checkpoint's ``generation_config.max_length`` so it stops arguing with us.

    LFM2 ships ``max_length=128000``. Passing ``max_new_tokens`` alongside it makes
    transformers log "Both `max_new_tokens` and `max_length` seem to have been set" on
    every batch (a few hundred lines per evaluation, per run) and then use
    ``max_new_tokens`` regardless: it recomputes ``max_length`` as
    ``max_new_tokens + input_ids_length`` whichever way the config was set. Clearing it
    changes no generated token; it silences a message that would otherwise bury the ones
    worth reading in a grid of a hundred runs.
    """
    config = getattr(model, "generation_config", None)
    if config is None or getattr(config, "max_length", None) is None:
        return False
    config.max_length = None
    return True


@dataclass
class GenerationScorer:
    """Greedy free generation with an anchored parser, the deployment-realism arm."""

    model: object
    tokenizer: object
    space: LabelSpace
    batch_size: int = 16
    max_new_tokens: int = 8
    max_length: int = 256
    device: str | None = None

    def __post_init__(self) -> None:
        torch = _import_torch()
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        clear_generation_max_length(self.model)

    def generate(self, prompts: Sequence[str]) -> list[str]:
        """Return generated continuations only, never prompt plus continuation."""
        torch = _import_torch()
        completions: list[str] = []
        for start in range(0, len(prompts), self.batch_size):
            chunk = list(prompts[start:start + self.batch_size])
            enc = self.tokenizer(text=chunk, return_tensors="pt", padding=True,
                                 truncation=True, max_length=self.max_length)
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                out = self.model.generate(
                    **enc, max_new_tokens=self.max_new_tokens, do_sample=False,
                    num_beams=1, pad_token_id=self.tokenizer.pad_token_id)
            prompt_len = enc["input_ids"].shape[1]
            for row in out:
                completions.append(
                    self.tokenizer.decode(row[prompt_len:], skip_special_tokens=True))
        return completions

    def predict(self, texts: Sequence[str]) -> tuple[list[str | None], list[str]]:
        completions = self.generate([build_prompt(t, self.space) for t in texts])
        return [parse_completion(c, self.space) for c in completions], completions


def to_label_indices(labels: Iterable[str | None], space: LabelSpace) -> tuple[np.ndarray, int]:
    """Map label strings to indices, counting unparseable entries separately."""
    idx, failures = [], 0
    for label in labels:
        if label is None or not space.contains(label):
            failures += 1
        else:
            idx.append(space.index(label))
    return np.asarray(idx, dtype=np.int64), failures
