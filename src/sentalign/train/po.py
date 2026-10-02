"""Preference-optimization training.

TRL 1.10 supplies DPO, cDPO, rDPO, IPO, DiscoPOP (all through ``DPOTrainer``) and KTO.
It does **not** supply SimPO or ORPO: the ``CPOTrainer`` that hosted them was removed , 
and it has never supplied Dr. DPO or group-robust DPO. Those three come from
``sentalign.train.objectives`` via the trainer subclasses below.

Note on the acronym: TRL's ``GRPOTrainer`` is DeepSeekMath's *Group Relative* Policy
Optimization, an on-policy RL algorithm. What this study runs is Ramesh et al.'s
*Group Robust* preference optimization, written ``GR-DPO`` throughout to keep the two
apart (AUDIT.md S2-1).

The version of TRL is checked at import time. A silent API change here would not crash , 
it would train a different objective than the one named in the results table, which is
precisely the failure mode this package exists to prevent.
"""

from __future__ import annotations

def _warmup_steps(cfg, tc) -> int:
    """Absolute warmup steps: transformers 5.x deprecates ``warmup_ratio``."""
    from ..config import warmup_steps_for

    return warmup_steps_for(cfg)


import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from ..config import ExperimentConfig
from ..labels import LabelSpace
from .objectives import TRL_OBJECTIVES, GroupRobustState, group_robust_dpo_loss

from ..versions import (MAX_TRL_EXCLUSIVE, MIN_TRL, REQUIRED_LOSS_TYPES,
                        parse_version, trl_range_message, trl_supported)

#: Kept as a module-level alias so existing call sites and tests keep working.
_parse_version = parse_version


def check_trl() -> str:
    """Assert the installed TRL is in the supported range, and explain it when it is not."""
    import trl

    version = getattr(trl, "__version__", "0.0.0")
    if not trl_supported(version):
        raise RuntimeError(trl_range_message(version))

    from trl import DPOConfig

    doc = DPOConfig.__doc__ or ""
    missing = {name for name in REQUIRED_LOSS_TYPES if f"'{name}'" not in doc}
    if missing:
        warnings.warn(
            f"TRL {version} does not advertise loss types {sorted(missing)}; verify "
            "before reporting results for the corresponding arms.", RuntimeWarning)
    return version


def optional_early_stopping(cfg: ExperimentConfig) -> list:
    """``[EarlyStoppingCallback]`` when patience is set, otherwise nothing.

    Only the trainer-driven arms can carry this callback at all; the distributional arms
    select through ``BestAdapterTracker``, which has no equivalent. Attaching it to some
    arms and not others gives them different realised budgets *and* different numbers of
    checkpoint draws at the same ``n``, which is a protocol difference masquerading as an
    algorithmic one. Defaulting patience to 0 keeps every arm on one fixed budget.
    """
    if not cfg.train.early_stopping_patience:
        return []
    from transformers import EarlyStoppingCallback

    return [EarlyStoppingCallback(cfg.train.early_stopping_patience)]


def kto_weights(n_desirable: int, n_undesirable: int) -> tuple[float, float]:
    """Desirable/undesirable weights satisfying KTO's recommended balance.

    Ethayarajh et al. recommend keeping ``(λ_D·n_D)/(λ_U·n_U)`` in roughly [1, 4/3]. On
    this task the unpaired set is ~1 desirable to ~3 undesirable completions per item
    (one correct label, three wrong), so leaving both weights at 1.0: the default, and
    what a naive port would do: trains KTO at a 3:1 imbalance it was never meant to see.
    """
    if n_desirable == 0 or n_undesirable == 0:
        return 1.0, 1.0
    target = 7 / 6                       # midpoint of the recommended band
    return float(target * n_undesirable / n_desirable), 1.0


def shared_training_fields(cfg: ExperimentConfig, plan) -> dict:
    """The recipe every arm trains under, whatever its objective.

    The arms are supposed to differ in their loss and in nothing else. They did differ in
    more, because SimPO, AlphaPO, and KTO had their configurations written out separately
    and silently inherited TRL's defaults: a linear schedule instead of cosine, no warmup,
    no weight decay, and, most consequentially, no periodic evaluation, hence no
    checkpoint selection, while the DPO family selected its best checkpoint on dev
    macro-F1. Best-of-three against a single final reading is a protocol advantage, not an
    algorithmic one, in exactly the comparison this study reports.

    It was also a hard failure: ``train_preference`` attaches an ``EarlyStoppingCallback``
    to every arm, and that callback asserts on ``eval_strategy="no"`` and on a missing
    ``metric_for_best_model``, so those three arms raised before their first step.

    Anything shared belongs here rather than in the per-objective builders, so the next
    arm cannot drift the same way.
    """
    import torch

    tc = cfg.train
    return dict(
        max_length=cfg.max_seq_length,
        # TRL defaults this to 512 and CPOTrainer refuses a value that is not strictly
        # below max_length, which killed every simpo and alphapo run. The completions here
        # are a single label token, so giving the prompt all but eight of the budget
        # changes nothing about what is truncated and makes the split explicit for the
        # pairwise arms as well.
        max_prompt_length=cfg.max_seq_length - 8,
        num_train_epochs=tc.num_train_epochs,
        learning_rate=tc.po_learning_rate,
        warmup_steps=_warmup_steps(cfg, tc),
        weight_decay=tc.weight_decay,
        lr_scheduler_type=tc.lr_scheduler_type,
        max_grad_norm=tc.max_grad_norm,
        logging_steps=tc.logging_steps,
        eval_strategy="steps",
        eval_steps=tc.eval_steps,
        save_strategy="steps",
        save_steps=tc.save_steps,
        save_total_limit=tc.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model=tc.metric_for_best_model,
        greater_is_better=tc.greater_is_better,
        optim=tc.optim,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        gradient_checkpointing=tc.gradient_checkpointing,
        seed=tc.seed,
        data_seed=tc.seed,
        report_to=[],
        **plan.as_dict(),
    )


#: The objectives whose loss TRL's ``DPOTrainer.get_batch_loss_metrics`` computes, and so
#: the only ones where ``rpo_alpha`` reaches the loss. GR-DPO and MixDPO replace that method
#: and KTO and the CPO arms use other trainers: each would accept the setting and train
#: without the term.
RPO_OBJECTIVES = ("dpo", "rdpo", "ipo")


def _check_rpo(tc) -> None:
    if tc.rpo_alpha is not None and tc.objective not in RPO_OBJECTIVES:
        raise ValueError(
            f"rpo_alpha={tc.rpo_alpha} is set for {tc.objective!r}, whose trainer never "
            f"adds the RPO term; it applies only to {RPO_OBJECTIVES}")


class RPOTermGuard:
    """Stop a run whose RPO term never reaches the loss.

    TRL adds the term, and logs ``nll_loss`` beside the loss, inside
    ``DPOTrainer.get_batch_loss_metrics``. A patch that replaced that method would train
    plain R-DPO under the RPO arm's name with nothing in the loss curve to show it, so a
    training log without the metric fails the run. ``seen`` is recorded in the manifest,
    which is how the term's presence is checked from the analysis side.
    """

    def __init__(self):
        self.seen = 0

    def check(self, logs: dict) -> None:
        if "loss" not in logs:
            return          # evaluation and end-of-training summaries
        if "nll_loss" not in logs:
            raise RuntimeError(
                "rpo_alpha is set but the training log carries no nll_loss: the RPO term "
                f"is not in the loss (log keys: {sorted(logs)[:12]})")
        self.seen += 1

    def as_transformers_callback(self):
        from transformers import TrainerCallback

        outer = self

        class _Callback(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):
                outer.check(logs or {})
                return control

        return _Callback()


def build_dpo_config(cfg: ExperimentConfig, output_dir: Path, plan):
    from trl import DPOConfig

    tc = cfg.train
    _check_rpo(tc)
    spec = TRL_OBJECTIVES.get(tc.objective, {})
    kwargs = dict(
        output_dir=str(output_dir),
        beta=tc.beta,
        **shared_training_fields(cfg, plan),
    )
    if tc.objective == "mixdpo":
        # The routing rule reads the human margin, which is an extra column.
        kwargs["remove_unused_columns"] = False
    if tc.objective == "grdpo":
        # The group label is an extra column; TRL would otherwise strip it before the
        # collator ever sees it (see GroupPreservingCollator).
        kwargs["remove_unused_columns"] = False
    if tc.objective in {"grdpo", "mixdpo"}:
        # These override _compute_loss and deliberately do not reimplement TRL's
        # reference-model path, which has PEFT adapter-swapping subtleties. Scoring the
        # reference once up front is both safer and faster, and on a 24 GB card it also
        # keeps a second set of weights out of memory.
        kwargs["precompute_ref_log_probs"] = True
    if "loss_type" in spec:
        kwargs["loss_type"] = [spec["loss_type"]]
    if "label_smoothing" in spec:
        kwargs["label_smoothing"] = tc.label_smoothing if spec["label_smoothing"] else 0.0
    if "discopop_tau" in spec:
        kwargs["discopop_tau"] = spec["discopop_tau"]
    if tc.rpo_alpha is not None:
        kwargs["rpo_alpha"] = tc.rpo_alpha
    return DPOConfig(**kwargs)


# --------------------------------------------------------------------------------------
# Custom trainers
# --------------------------------------------------------------------------------------
#
# GR-DPO and MixDPO override ``DPOTrainer.get_batch_loss_metrics``, which in the 0.2x line
# is the documented seam between "score the batch" and "turn scores into a loss". It
# receives the batch, calls ``concatenated_forward`` for the policy log-probabilities, and
# returns ``(loss, metrics)``. Overriding it means we reuse TRL's tokenisation, padding,
# reference handling, and metric plumbing, and replace only the aggregation.
#
# SimPO and AlphaPO are **not** overridden: TRL 0.2x provides them natively through
# ``CPOTrainer`` (``loss_type="simpo"`` with ``cpo_alpha=0``). Using the library's
# implementation rather than our own removes a whole class of transcription risk, and the
# formula in ``objectives.simpo_loss`` remains under test as the specification that
# implementation is checked against.
#
# The batch layout in 0.2x keeps chosen and rejected in separate fields
# (``chosen_input_ids``, ``rejected_input_ids``, and their attention masks), unlike the
# 1.x concatenated layout. ``_assert_batch_contract`` pins that, because an override that
# silently reads the wrong field would train a different objective without raising.

REQUIRED_BATCH_KEYS = ("prompt_input_ids", "chosen_input_ids", "rejected_input_ids")


def _assert_batch_contract(inputs: dict, *, needs_reference: bool) -> None:
    """Fail loudly if TRL's batch layout is not what these overrides assume."""
    missing = [k for k in REQUIRED_BATCH_KEYS if k not in inputs]
    if missing:
        raise RuntimeError(
            f"TRL batch is missing {missing}. sentalign's custom objectives target the "
            f"TRL 0.2x layout, which keeps chosen and rejected in separate fields "
            f"{list(REQUIRED_BATCH_KEYS)}. Present keys: {sorted(inputs)[:12]}. "
            "Re-verify against the installed TRL before trusting these arms.")
    if needs_reference and "ref_chosen_logps" not in inputs:
        raise RuntimeError(
            "reference log-probabilities are not in the batch. GR-DPO and MixDPO set "
            "`precompute_ref_log_probs=True`; their absence means the config was "
            "overridden or TRL changed the key name.")


def _completion_lengths(inputs: dict):
    """Token counts of the chosen and rejected completions, for length normalisation."""
    import torch

    def count(key: str):
        mask = inputs.get(f"{key}_attention_mask")
        if mask is not None:
            return mask.sum(dim=1).clamp(min=1).float()
        return torch.ones(inputs[f"{key}_input_ids"].shape[0],
                          device=inputs[f"{key}_input_ids"].device)

    return count("chosen"), count("rejected")


def _policy_and_reference_logps(trainer, model, inputs):
    """Policy log-probabilities from ``concatenated_forward``, plus the reference pair."""
    output = trainer.concatenated_forward(model, inputs)
    if "ref_chosen_logps" in inputs and "ref_rejected_logps" in inputs:
        ref_chosen = inputs["ref_chosen_logps"]
        ref_rejected = inputs["ref_rejected_logps"]
    else:
        ref_chosen, ref_rejected = trainer.compute_ref_log_probs(inputs)
    return output, ref_chosen, ref_rejected


class TargetPreservingCollator:
    """Keep the annotator distribution in the batch, for the distributional regulariser.

    Same reason as ``MarginPreservingCollator``: TRL builds its output dictionary from
    scratch, so an extra column is dropped silently. A missing target raises rather than
    defaulting to the unregularised loss, because that would report a regularised arm
    under its own name while training plain preference optimisation.
    """

    def __init__(self, inner):
        self.inner = inner

    def __call__(self, examples):
        import torch

        if "p_human" not in examples[0]:
            raise RuntimeError(
                "`p_human` is missing from the collated examples, so the distributional "
                "regulariser would silently vanish and the arm would train as plain "
                "preference optimisation under a different name.")
        target = torch.tensor([e["p_human"] for e in examples], dtype=torch.float32)
        prompts = [e["prompt"] for e in examples]
        batch = self.inner(examples)
        batch["p_human"] = target
        batch["_prompts"] = prompts
        return batch


def distributional_penalty(model, tokenizer, prompts, target, verbalizer_ids,
                           max_length: int, regularizer: str = "soft"):
    """``CE(target, softmax(l_theta))`` over the closed label set at the prompt's end.

    The term is computed on the *policy's own* label distribution, which is the quantity a
    proper scoring rule constrains and the quantity evaluation reads. One extra forward
    pass over the prompts only, which is cheaper than the pair the preference loss already
    scores.

    ``regularizer='hard'`` replaces the annotator distribution with its majority label and
    is the ablation that separates this from ordinary SFT regularisation: if the hard
    variant works equally well, the claim is not about annotator distributions.
    """
    import torch

    tokenizer.padding_side = "left"
    enc = tokenizer(text=list(prompts), return_tensors="pt", padding=True,
                    truncation=True, max_length=max_length, add_special_tokens=True)
    device = next(model.parameters()).device
    enc = {k: v.to(device) for k, v in enc.items()}
    logits = model(**enc, use_cache=False).logits[:, -1, list(verbalizer_ids)].float()
    log_q = torch.log_softmax(logits, dim=-1)

    t = target.to(log_q.device)
    if regularizer == "hard":
        t = torch.zeros_like(t).scatter_(1, t.argmax(dim=1, keepdim=True), 1.0)
    return -(t * log_q).sum(dim=-1).mean()


def add_distributional_regulariser(trainer_class, lam: float, regularizer: str,
                                   verbalizer_ids, max_length: int):
    """Wrap any TRL preference trainer with the properness term.

    Returns the class unchanged when ``lam`` is zero, so the unregularised objective is
    reproduced exactly rather than approximately.
    """
    if not lam:
        return trainer_class

    class Regularised(trainer_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.data_collator = TargetPreservingCollator(self.data_collator)
            # TRL prunes every column outside a fixed whitelist of tokenised fields,
            # so `p_human` and `prompt` never reach the collator. DPOConfig leaves
            # `remove_unused_columns` at True, while KTOConfig forces it False, which is
            # why the unpaired arm trained and the pairwise ones died at step zero. The
            # flag is cleared here rather than in each config builder because the
            # regulariser is orthogonal to the objective: a per-objective branch is the
            # asymmetry that has now cost this project eight runs.
            self.args.remove_unused_columns = False
            self._reg_lambda = float(lam)
            self._reg_kind = regularizer

        def get_batch_loss_metrics(self, model, batch, *args, **kwargs):
            # TRL's trainers do not share this signature: DPOTrainer takes a fourth
            # `train_eval` argument and KTOTrainer takes none, so the wrapper forwards
            # whatever the parent expects rather than assuming one of them.
            prompts = batch.pop("_prompts", None)
            target = batch.pop("p_human", None)
            loss, metrics = super().get_batch_loss_metrics(model, batch, *args, **kwargs)
            if prompts is None or target is None:
                raise RuntimeError("the distributional regulariser lost its batch fields")
            penalty = distributional_penalty(model, self.processing_class, prompts,
                                             target, verbalizer_ids, max_length,
                                             self._reg_kind)
            train_eval = kwargs.get("train_eval", args[0] if args else "train")
            prefix = "eval_" if train_eval == "eval" else ""
            metrics[f"{prefix}loss/pref"] = float(loss)
            metrics[f"{prefix}loss/distributional"] = float(penalty)
            return loss + self._reg_lambda * penalty, metrics

    Regularised.__name__ = f"Regularised{trainer_class.__name__}"
    return Regularised


def make_mixdpo_trainer_class():
    """MixDPO (Pang et al., 2026): route hard pairs to the supervised loss.

    The published method defines pair difficulty by an implicit reward margin estimated
    from the model. Here the margin is the observed annotator gap, carried in the batch as
    ``margin``, so the routing rule is applied to ground truth rather than to an estimate
    of itself. That is the reason for including it as an arm rather than only citing it.
    """
    import torch
    import torch.nn.functional as F
    from trl import DPOTrainer

    class MixDPOTrainer(DPOTrainer):
        def __init__(self, *args, threshold: float = 0.5, sft_weight: float = 1.0,
                     **kwargs):
            super().__init__(*args, **kwargs)
            self.threshold, self.sft_weight = threshold, sft_weight
            self.routing_stats = {"easy": 0, "hard": 0}
            self._contract_checked = False
            self.data_collator = MarginPreservingCollator(self.data_collator)

        def get_batch_loss_metrics(self, model, batch, train_eval="train"):
            if not self._contract_checked:
                _assert_batch_contract(batch, needs_reference=False)
                self._contract_checked = True
            if "margin" not in batch:
                raise RuntimeError(
                    "MixDPO batch has no `margin`; without it every pair would be routed "
                    "to the preference loss and the arm would be plain DPO under another "
                    "name.")

            output, ref_chosen, ref_rejected = _policy_and_reference_logps(
                self, model, batch)
            chosen_logps = output["chosen_logps"]
            rejected_logps = output["rejected_logps"]
            delta = (chosen_logps - ref_chosen) - (rejected_logps - ref_rejected)
            pref = -F.logsigmoid(self.beta * delta)

            chosen_len, _ = _completion_lengths(batch)
            sft = -chosen_logps / chosen_len.to(chosen_logps.device)

            easy = batch["margin"].abs().to(delta.device) >= self.threshold
            losses = torch.where(easy, pref, self.sft_weight * sft)
            loss = losses.mean()

            self.routing_stats["easy"] += int(easy.sum())
            self.routing_stats["hard"] += int((~easy).sum())
            prefix = "eval_" if train_eval == "eval" else ""
            metrics = {
                f"{prefix}rewards/accuracies": (delta > 0).float().mean().item(),
                f"{prefix}rewards/margins": delta.mean().item(),
                f"{prefix}routing/easy_fraction": float(easy.float().mean()),
                f"{prefix}loss/pref_mean": float(pref.mean()),
                f"{prefix}loss/sft_mean": float(sft.mean()),
            }
            return loss, metrics

    return MixDPOTrainer


class MarginPreservingCollator:
    """Keep the human margin in the batch, for MixDPO's routing rule.

    TRL's preference collator builds its output dictionary from scratch and returns only
    the tokenised fields and the reference log-probabilities, so any extra column is
    discarded. Without this wrapper MixDPO would see no margins, route every pair to the
    preference loss, and report as plain DPO under its own name. A missing margin
    therefore raises rather than defaulting.
    """

    def __init__(self, inner):
        self.inner = inner

    def __call__(self, examples):
        if "margin" not in examples[0]:
            raise RuntimeError(
                "`margin` is missing from the collated examples, so MixDPO would route "
                "every pair to the preference loss and become plain DPO. Ensure the "
                "preference dataset carries `margin` and `remove_unused_columns=False`.")
        values = [float(e["margin"]) for e in examples]

        import torch

        batch = self.inner(examples)
        batch["margin"] = torch.tensor(values, dtype=torch.float32)
        return batch


class GroupPreservingCollator:
    """Keep group ids in the batch, for GR-DPO. Same failure mode as the margin above."""

    def __init__(self, inner, n_groups: int):
        self.inner = inner
        self.n_groups = n_groups

    def __call__(self, examples):
        if "group_id" not in examples[0]:
            raise RuntimeError(
                "`group_id` is missing from the collated examples, so GR-DPO would "
                "silently train as mean DPO. Ensure the preference dataset carries an "
                "integer `group_id` column and that `remove_unused_columns=False`.")
        raw = [int(e["group_id"]) for e in examples]
        if max(raw) >= self.n_groups or min(raw) < 0:
            raise RuntimeError(
                f"group_id out of range: saw [{min(raw)}, {max(raw)}] for "
                f"{self.n_groups} groups")

        import torch

        batch = self.inner(examples)
        batch["group_id"] = torch.tensor(raw, dtype=torch.long)
        return batch


def make_grdpo_trainer_class():
    """``DPOTrainer`` with Ramesh et al.'s exponentiated-gradient group reweighting.

    Group weights are logged so the paper can show which groups the objective chose to
    protect, not merely that the worst-group score moved.
    """
    import torch
    import torch.nn.functional as F
    from trl import DPOTrainer

    class GRDPOTrainer(DPOTrainer):
        def __init__(self, *args, groups: Sequence[str] = (), step_size: float = 0.01,
                     **kwargs):
            super().__init__(*args, **kwargs)
            self.group_state = GroupRobustState(groups=tuple(groups), step_size=step_size)
            self.weight_history: list[dict] = []
            self._contract_checked = False
            self.data_collator = GroupPreservingCollator(self.data_collator, len(groups))

        def get_batch_loss_metrics(self, model, batch, train_eval="train"):
            if not self._contract_checked:
                _assert_batch_contract(batch, needs_reference=False)
                self._contract_checked = True
            if "group_id" not in batch:
                raise RuntimeError(
                    "GR-DPO batch has no `group_id`; refusing to fall back to mean DPO, "
                    "which would report an untrained arm under the GR-DPO name.")

            output, ref_chosen, ref_rejected = _policy_and_reference_logps(
                self, model, batch)
            delta = ((output["chosen_logps"] - ref_chosen)
                     - (output["rejected_logps"] - ref_rejected))
            per_example = -F.logsigmoid(self.beta * delta)

            ids = batch["group_id"].detach().cpu().numpy().astype(np.int64)
            # The weight update runs on detached values; the loss keeps its graph.
            _, weights = group_robust_dpo_loss(
                per_example.detach().float().cpu().numpy(), ids, self.group_state,
                update=(train_eval == "train"))
            w = torch.as_tensor(weights[ids], dtype=per_example.dtype,
                                device=per_example.device)
            loss = (per_example * w).sum() / w.sum().clamp(min=1e-8)

            if train_eval == "train":
                self.weight_history.append(
                    {"step": int(self.state.global_step),
                     **{g: float(weights[i])
                        for i, g in enumerate(self.group_state.groups)}})
            prefix = "eval_" if train_eval == "eval" else ""
            metrics = {
                f"{prefix}rewards/accuracies": (delta > 0).float().mean().item(),
                f"{prefix}rewards/margins": delta.mean().item(),
                f"{prefix}group/max_weight": float(weights.max()),
                f"{prefix}group/min_weight": float(weights.min()),
            }
            return loss, metrics

    return GRDPOTrainer


def build_kto_config(cfg: ExperimentConfig, output_dir, plan,
                     desirable_weight: float, undesirable_weight: float):
    """KTO, whose weights depend on the desirable/undesirable balance of its own data.

    A builder rather than an inline config, so this arm's settings can be compared with
    the others in a test instead of only at runtime.
    """
    from trl import KTOConfig

    tc = cfg.train
    _check_rpo(tc)
    return KTOConfig(
        output_dir=str(output_dir),
        beta=tc.beta,
        desirable_weight=desirable_weight,
        undesirable_weight=undesirable_weight,
        **shared_training_fields(cfg, plan))


def build_cpo_config(cfg: ExperimentConfig, output_dir, plan):
    """SimPO and AlphaPO, both native in TRL 0.2x through ``CPOTrainer``.

    ``cpo_alpha=0`` removes the behaviour-cloning regulariser, which is what turns CPO
    into SimPO proper: a reference-free, length-normalised objective with a target reward
    margin. AlphaPO adds a reward-shape parameter on top of the same loss.
    """
    from trl import CPOConfig

    tc = cfg.train
    return CPOConfig(
        output_dir=str(output_dir),
        loss_type="alphapo" if tc.objective == "alphapo" else "simpo",
        cpo_alpha=0.0,
        simpo_gamma=tc.simpo_gamma,
        alpha=tc.alphapo_alpha if tc.objective == "alphapo" else 0.0,
        beta=tc.simpo_beta,
        **shared_training_fields(cfg, plan))


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------

def with_group_ids(pairs, index: dict[str, int]) -> list[dict]:
    """Attach the integer ``group_id`` GR-DPO's collator requires, dropping unknowns.

    Both the training and the evaluation split go through ``GroupPreservingCollator``,
    which refuses a batch without the column rather than silently training as mean DPO.
    Only the training split ever carried it, so every GR-DPO run failed the moment it
    built its first evaluation batch.
    """
    return [dict(p, group_id=index[p["group"]]) for p in pairs if p["group"] in index]


def train_preference(cfg, data, model, tokenizer, space, plan, selector, guard, logger,
                     resume) -> dict:
    """Train one pairwise or unpaired preference arm.

    Called by ``train.driver.run_training``, which owns preflight, logging, OOM backoff,
    and checkpoint handling. This function is responsible only for choosing the trainer
    and its data.
    """
    from datasets import Dataset

    from .driver import MetricLogCallback

    trl_version = check_trl()
    objective = cfg.train.objective
    callbacks = [selector.as_transformers_callback(),
                 MetricLogCallback(logger, guard).as_transformers_callback(),
                 *optional_early_stopping(cfg)]
    extra: dict = {"trl_version": trl_version}
    rpo_guard = None
    if cfg.train.rpo_alpha is not None:
        _check_rpo(cfg.train)
        rpo_guard = RPOTermGuard()
        callbacks.append(rpo_guard.as_transformers_callback())

    # Distributional regularisation applies to every preference trainer identically, so it
    # is a class wrapper rather than a branch repeated at each construction site.
    _lam = cfg.train.pref_distributional_lambda
    _ids = None
    if _lam:
        from ..labels import resolve_verbalizers

        _ids = resolve_verbalizers(tokenizer, space).token_ids
        extra["pref_distributional_lambda"] = _lam
        extra["pref_regularizer"] = cfg.train.pref_regularizer

    def _wrap(cls):
        if not _lam:
            return cls
        return add_distributional_regulariser(cls, _lam, cfg.train.pref_regularizer,
                                              _ids, cfg.max_seq_length)


    if objective == "kto":
        from trl import KTOTrainer

        examples = list(data["kto_examples"])
        n_des = sum(1 for e in examples if e["label"])
        des_w, und_w = kto_weights(n_des, len(examples) - n_des)
        # `is not None`, not `or`: an explicit 0.0 override is a request to zero that
        # side of the loss, and `or` would silently replace it with the computed weight.
        des_cfg = cfg.train.kto_desirable_weight
        und_cfg = cfg.train.kto_undesirable_weight
        args = build_kto_config(
            cfg, cfg.run_dir, plan,
            desirable_weight=des_cfg if des_cfg is not None else des_w,
            undesirable_weight=und_cfg if und_cfg is not None else und_w)
        # Periodic evaluation needs something to evaluate; KTO is unpaired, so its dev
        # split is unpaired too. The selection metric is the callback's dev macro-F1
        # either way: this dataset is what lets the trainer reach the callback.
        trainer = _wrap(KTOTrainer)(model=model, args=args, processing_class=tokenizer,
                             train_dataset=Dataset.from_list(examples),
                             eval_dataset=Dataset.from_list(data["kto_dev_examples"]),
                             callbacks=callbacks)
        extra["kto_weights"] = {"desirable": args.desirable_weight,
                                "undesirable": args.undesirable_weight,
                                "n_desirable": n_des,
                                "n_undesirable": len(examples) - n_des}
        logger.event("kto_balance", **extra["kto_weights"])
    else:
        pairs = list(data["pref_pairs"])
        dev_pairs = list(data["pref_dev_pairs"])
        args = build_dpo_config(cfg, cfg.run_dir, plan)
        common = dict(model=model, args=args,
                      train_dataset=Dataset.from_list(pairs),
                      eval_dataset=Dataset.from_list(dev_pairs),
                      processing_class=tokenizer, callbacks=callbacks)
        extra["n_train_pairs"] = len(pairs)

        if objective in ("simpo", "alphapo"):
            # Native in TRL 0.2x through CPOTrainer, so no override and no transcription
            # risk. It takes the same prompt/chosen/rejected dataset as DPO.
            from trl import CPOTrainer

            trainer = _wrap(CPOTrainer)(
                model=model, args=build_cpo_config(cfg, cfg.run_dir, plan),
                train_dataset=Dataset.from_list(pairs),
                eval_dataset=Dataset.from_list(dev_pairs),
                processing_class=tokenizer, callbacks=callbacks)
            extra["cpo_loss_type"] = trainer.args.loss_type
            extra["cpo_alpha"] = trainer.args.cpo_alpha
            extra["simpo_gamma"] = trainer.args.simpo_gamma
        elif objective == "mixdpo":
            trainer = _wrap(make_mixdpo_trainer_class())(
                **common, threshold=cfg.train.mixdpo_threshold,
                sft_weight=cfg.train.mixdpo_sft_weight)
            extra["mixdpo_threshold"] = cfg.train.mixdpo_threshold
        elif objective == "grdpo":
            groups = sorted({p["group"] for p in pairs})
            index = {g: i for i, g in enumerate(groups)}
            common["train_dataset"] = Dataset.from_list(with_group_ids(pairs, index))
            # The evaluation split runs through the same collator, so it needs the same
            # column. A dev group absent from training has no weight to update and is
            # dropped rather than folded into an arbitrary id.
            dev_with_groups = with_group_ids(dev_pairs, index)
            common["eval_dataset"] = Dataset.from_list(dev_with_groups)
            extra["n_dev_pairs_dropped_unknown_group"] = len(dev_pairs) - len(dev_with_groups)
            trainer = _wrap(make_grdpo_trainer_class())(
                **common, groups=groups, step_size=cfg.train.grdpo_step_size)
            extra["groups"] = groups
            extra["group_sizes"] = {g: sum(1 for p in pairs if p["group"] == g)
                                    for g in groups}
        elif objective in TRL_OBJECTIVES:
            from trl import DPOTrainer

            trainer = _wrap(DPOTrainer)(**common)
        else:
            raise ValueError(f"unknown preference objective {objective!r}")

    from ..recovery import train_with_resume_fallback

    result, used = train_with_resume_fallback(
        lambda ckpt: trainer.train(resume_from_checkpoint=str(ckpt) if ckpt else None),
        resume,
        on_fallback=lambda ckpt, exc: logger.event(
            "incompatible_checkpoint_discarded", checkpoint=str(ckpt),
            error=str(exc)[:300]))
    extra["resumed_from"] = str(used) if used else None
    if rpo_guard is not None:
        if not rpo_guard.seen:
            raise RuntimeError("rpo_alpha is set but no training log was checked for the "
                               "RPO term, so the run cannot show the term was applied")
        extra["rpo_alpha"] = trainer.args.rpo_alpha
        extra["rpo_nll_logs"] = rpo_guard.seen
    if hasattr(trainer, "weight_history") and trainer.weight_history:
        extra["group_weight_history"] = trainer.weight_history[::50]
    if hasattr(trainer, "routing_stats"):
        extra["routing_stats"] = trainer.routing_stats
    return {"steps": int(result.global_step),
            "train_loss": float(result.training_loss), **extra}
