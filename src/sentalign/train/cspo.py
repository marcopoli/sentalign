"""CSPO and its control, implemented directly on the transformers Trainer.

These two objectives do not go through TRL. They need one forward pass over a prompt and
a read of K logits at its final position, which is simpler than anything TRL's preference
trainers are built around, and subclassing them would mean inheriting a pairwise data
pipeline in order to discard it. Writing the loop directly is both shorter and far less
exposed to the internal-API drift documented in ``train.po``.

Two objectives share this file because they differ in exactly one term, and the whole
point of the control is that the difference is isolated:

    CSPO      L = CE( p_human, softmax_k[ beta * (l_theta[v_k] - l_ref[v_k]) ] )
    SFT-soft  L = CE( p_human, softmax_k[        l_theta[v_k]               ] )

CSPO anchors on the reference policy, so it constrains the *relative* reward among the K
labels and leaves total likelihood mass untouched. SFT-soft has no anchor and moves the
distribution directly. If SFT-soft matches CSPO, the reference anchoring is not earning
its place and the paper should say so; that is why the control is in the main grid rather
than in an appendix.

The reference logits are precomputed once before training and carried in the dataset, so
no reference model is resident during optimisation. On a 24 GB card that is the
difference between one model in memory and two.
"""

from __future__ import annotations

import math

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..labels import LabelSpace, VerbalizerTable, build_prompt, resolve_verbalizers


@dataclass
class CSPOCollator:
    """Batch prompts with their targets and precomputed reference logits.

    Left padding puts every prompt's final token at the same index, so the label logits
    are read from position -1 for every row with no per-row index arithmetic.
    """

    tokenizer: Any
    max_length: int = 256

    def __call__(self, features: Sequence[dict]) -> dict:
        import torch

        self.tokenizer.padding_side = "left"
        enc = self.tokenizer(text=[f["prompt"] for f in features], return_tensors="pt",
                             padding=True, truncation=True, max_length=self.max_length,
                             add_special_tokens=True)
        batch = {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"],
                 "p_human": torch.tensor([f["p_human"] for f in features],
                                         dtype=torch.float32)}
        if "votes" in features[0]:
            batch["votes"] = torch.tensor([f["votes"] for f in features],
                                          dtype=torch.float32)
        if "ref_logits" in features[0] and features[0]["ref_logits"] is not None:
            batch["ref_logits"] = torch.tensor([f["ref_logits"] for f in features],
                                               dtype=torch.float32)
        return batch


def make_cspo_trainer_class():
    """A ``Trainer`` whose loss is the CSPO cross-entropy."""
    import torch
    from transformers import Trainer

    class CSPOTrainer(Trainer):
        def __init__(self, *args, verbalizer_ids: Sequence[int], beta: float = 1.0,
                     use_reference: bool = True, kl_lambda: float | None = None,
                     ada_lambda0: float | None = None,
                     eb_alpha: float = 0.0,
                     polya: bool = False,
                     mixture_head=None,
                     polya_scale: float = 1.0,
                     polya_max_concentration: float = 1e4,
                     **kwargs):
            super().__init__(*args, **kwargs)
            self.verbalizer_ids = list(verbalizer_ids)
            self.beta = beta
            self.use_reference = use_reference
            #: When set, fit the policy's own distribution and keep the reference as a
            #: KL penalty beside the loss instead of inside the softmax. See
            #: objectives.cspo_kl_loss for why the placement changes what is learned.
            self.kl_lambda = kl_lambda
            self.ada_lambda0 = ada_lambda0
            self.eb_alpha = float(eb_alpha)
            self.polya = bool(polya)
            self.mixture_head = mixture_head
            self.polya_scale = float(polya_scale)
            self.polya_max_concentration = float(polya_max_concentration)
            #: Annotators per item; only used by the empirical-Bayes shrinkage.
            self._votes_per_item = 5.0
            self._reward_stats: list[dict] = []

        def compute_loss(self, model, inputs, return_outputs=False,
                         num_items_in_batch=None):
            p_human = inputs.pop("p_human")
            ref_logits = inputs.pop("ref_logits", None)
            if self.use_reference and ref_logits is None:
                raise RuntimeError(
                    "CSPO requires precomputed reference logits in the batch. They are "
                    "produced by precompute_reference_logits() before training; their "
                    "absence means the dataset was rebuilt without them, and training "
                    "would silently become SFT-soft.")

            outputs = model(input_ids=inputs["input_ids"],
                            attention_mask=inputs["attention_mask"],
                            use_cache=False,
                            output_hidden_states=self.mixture_head is not None)
            # Logits at the final prompt position, restricted to the label verbalizers.
            label_logits = outputs.logits[:, -1, self.verbalizer_ids].float()

            if self.mixture_head is not None:
                # A mixture of softmaxes over latent perspectives. Not a rescaling of
                # label_logits, so no temperature reproduces it, which is the whole point.
                hidden = outputs.hidden_states[-1][:, -1, :].float()
                log_q = self.mixture_head(hidden)
                target = p_human.to(log_q.device)
                loss = -(target * log_q).sum(dim=-1).mean()
                reward = log_q
                if self.state.global_step % max(self.args.logging_steps, 1) == 0:
                    with torch.no_grad():
                        gate = torch.log_softmax(
                            hidden @ self.mixture_head.gate.T
                            + self.mixture_head.gate_bias[None], dim=-1).exp()
                        self._reward_stats.append(
                            {"step": int(self.state.global_step),
                             "gate_entropy": float(-(gate * gate.clamp_min(1e-12).log())
                                                   .sum(-1).mean()),
                             "gate_max": float(gate.max(dim=-1).values.mean())})
                return (loss, outputs) if return_outputs else loss

            target = p_human.to(label_logits.device)

            if self.polya:
                # Model the counts, do not match their frequencies. alpha_k = exp(l_k)
                # makes the mean softmax(l) and the concentration sum_k exp(l_k), the
                # logit magnitude softmax discards. See objectives.polya_loss.
                votes = inputs.pop("votes", None)
                if votes is None:
                    raise RuntimeError(
                        "Polya alignment needs raw annotator counts in the batch. They "
                        "are written by build_cspo_records as `votes`; their absence "
                        "means the data was built by an older version, and training "
                        "would silently fall back to matching frequencies.")
                n = votes.to(label_logits.device).float()
                shifted = label_logits - label_logits.max(dim=-1, keepdim=True).values
                alpha = shifted.exp() * self.polya_scale
                a0 = alpha.sum(dim=-1, keepdim=True)
                over = (a0 / self.polya_max_concentration).clamp_min(1.0)
                alpha = alpha / over
                a0 = alpha.sum(dim=-1)
                m = n.sum(dim=-1)
                loss = -(torch.lgamma(a0) - torch.lgamma(a0 + m)
                         + (torch.lgamma(alpha + n) - torch.lgamma(alpha)).sum(dim=-1)
                         ).mean()
                log_q = torch.log_softmax(label_logits, dim=-1)
                reward = label_logits
            if self.eb_alpha > 0 and not self.polya:
                # Shrink the M-annotator frequencies toward a uniform prior. At M=5 the
                # target carries ~0.074 JSD of pure sampling noise, so the raw
                # frequencies spend capacity on noise; alpha=0 leaves them untouched.
                k = target.shape[-1]
                target = ((target * self._votes_per_item + self.eb_alpha / k)
                          / (self._votes_per_item + self.eb_alpha))
                target = target / target.sum(dim=-1, keepdim=True)

            if self.polya:
                pass                      # loss already computed above
            elif self.ada_lambda0 is not None:
                # Anchor per item, decaying with that item's ambiguity: the hard-label
                # reference is confidently wrong exactly where annotators disagree.
                reference = ref_logits.to(label_logits.device)
                log_q = torch.log_softmax(label_logits, dim=-1)
                log_ref = torch.log_softmax(reference, dim=-1)
                kl = (log_q.exp() * (log_q - log_ref)).sum(dim=-1)
                h = -(target * torch.log(target.clamp_min(1e-12))).sum(dim=-1)
                lam = self.ada_lambda0 * (1.0 - h / math.log(target.shape[-1]))
                loss = (-(target * log_q).sum(dim=-1) + lam.clamp_min(0.0) * kl).mean()
                reward = label_logits
            elif self.kl_lambda is not None:
                # The scored distribution is the trained one: cross-entropy on
                # softmax(l_theta), with the reference as a penalty beside it.
                reference = ref_logits.to(label_logits.device)
                log_q = torch.log_softmax(label_logits, dim=-1)
                log_ref = torch.log_softmax(reference, dim=-1)
                kl = (log_q.exp() * (log_q - log_ref)).sum(dim=-1)
                loss = (-(target * log_q).sum(dim=-1) + self.kl_lambda * kl).mean()
                reward = label_logits
            else:
                if self.use_reference:
                    reward = self.beta * (label_logits
                                          - ref_logits.to(label_logits.device))
                else:
                    reward = label_logits

                log_q = torch.log_softmax(reward, dim=-1)
                loss = -(target * log_q).sum(dim=-1).mean()

            if self.state.global_step % max(self.args.logging_steps, 1) == 0:
                with torch.no_grad():
                    q = log_q.exp()
                    self._reward_stats.append({
                        "step": int(self.state.global_step),
                        "reward_mean": float(reward.mean()),
                        "reward_spread": float((reward.max(-1).values
                                                - reward.min(-1).values).mean()),
                        "q_entropy": float(-(q * log_q).sum(-1).mean()),
                        "target_entropy": float(
                            -(target * torch.log(target.clamp(min=1e-12))).sum(-1).mean()),
                        "top1_agreement": float(
                            (q.argmax(-1) == target.argmax(-1)).float().mean()),
                    })
            return (loss, outputs) if return_outputs else loss

    return CSPOTrainer


def precompute_reference_logits(
    model,
    tokenizer,
    records: Sequence[dict],
    table: VerbalizerTable,
    *,
    batch_size: int = 32,
    max_length: int = 256,
    progress_every: int = 2000,
    logger=None,
) -> list[list[float]]:
    """Score every prompt once under the frozen reference policy.

    Returns K reference logits per record. This is the entire memory cost of CSPO's
    reference: K floats per training example rather than a second set of model weights.
    For 8,000 examples and K = 4 that is 128 KB.
    """
    import torch

    if not table.single_token:
        raise RuntimeError(
            "CSPO needs single-token verbalizers; this tokenizer does not provide them")

    ids = list(table.token_ids)
    out: list[list[float]] = []
    model.eval()
    device = next(model.parameters()).device
    tokenizer.padding_side = "left"
    started = time.time()

    with torch.no_grad():
        for start in range(0, len(records), batch_size):
            chunk = records[start:start + batch_size]
            enc = tokenizer(text=[r["prompt"] for r in chunk], return_tensors="pt",
                            padding=True, truncation=True, max_length=max_length,
                            add_special_tokens=True)
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc, use_cache=False).logits[:, -1, ids].float().cpu()
            out.extend(logits.tolist())
            if logger is not None and progress_every and start % progress_every == 0:
                logger.event("reference_logits_progress", done=start + len(chunk),
                             total=len(records),
                             elapsed_s=round(time.time() - started, 1))
    return out


def build_cspo_records(
    items: Sequence,
    space: LabelSpace,
    *,
    temperature: float = 1.0,
    smoothing: float = 0.0,
) -> list[dict]:
    """Prompt plus annotator distribution for every item, including no-majority ones.

    Unlike the SFT path, nothing is dropped: an item on which annotators split two to two
    has no point label but a perfectly well defined target distribution, and those items
    are 10.8 percent of the training pool.
    """
    from .objectives import soft_target

    records = []
    for item in items:
        votes = np.array([[item.votes.get(y, 0) for y in space.labels]], dtype=np.float64)
        if votes.sum() == 0:
            continue
        target = soft_target(votes, temperature=temperature, smoothing=smoothing)[0]
        records.append({
            "text_id": item.text_id,
            "prompt": build_prompt(item.text, space),
            "p_human": [float(v) for v in target],
            # Raw counts, not just the normalised target. Polya alignment models the
            # counts as a multinomial sample rather than matching their frequencies, and
            # the empirical-Bayes shrinkage needs the real annotator count: DynaSent has
            # five per item, ChaosNLI a hundred, and treating them alike is wrong.
            "votes": [float(v) for v in votes[0]],
            "n_annotators": float(votes.sum()),
            "gold": item.gold,
            "agreement_band": item.agreement_band,
            "group": f"{item.round}:{item.agreement_band}",
        })
    return records
