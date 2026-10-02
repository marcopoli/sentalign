"""One entry point for every training arm, with logging and recovery around it.

Every objective goes through ``run_training``. That is deliberate: the arms differ in
their loss and their data, and in nothing else that could affect a comparison. Sharing
the checkpoint selection, the seeding, the evaluation callback, the OOM handling, and the
manifest means a difference in the results table is a difference in the objective.

Recovery behaviour, in the order it applies:

1.  ``preflight`` checks VRAM, disk, dependency versions, and that the tokenizer gives
    single-token verbalizers, before any weights load.
2.  A prior run directory is inspected. A completed run is skipped; an interrupted one
    resumes from its newest valid checkpoint, and partial checkpoint directories left by
    a killed process are removed only after the resume point has been chosen.
3.  Transient failures such as Hub timeouts retry with exponential backoff.
4.  An out-of-memory error halves the micro-batch and doubles gradient accumulation, so
    the effective batch size, and therefore the optimisation problem, is unchanged.
5.  A non-finite loss stops the run at the step it appears, leaving the last good
    checkpoint intact rather than overwriting it.
6.  SIGINT and SIGTERM unwind cleanly and record an ``interrupted`` status, so a resumed
    programme can tell an interrupted run from a hung one.
"""

from __future__ import annotations

def _warmup_steps(cfg, tc) -> int:
    """Absolute warmup steps: transformers 5.x deprecates ``warmup_ratio``."""
    from ..config import warmup_steps_for

    return warmup_steps_for(cfg)


import json
import random
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..config import ExperimentConfig
from ..labels import LabelSpace, resolve_verbalizers
from ..modeling import (MODEL_REGISTRY, adapter_size_mb, count_trainable, load_adapter, load_causal_lm,
                        save_run, set_training_mode)
from ..recovery import (BatchPlan, Interrupted, NanGuard, assert_resume_made_progress,
                        config_changed, install_signal_handlers, list_checkpoints,
                        oom_backoff, preflight, prune_checkpoints, resume_point, retry)
from ..runlog import RunLogger, RunStatus, is_complete
from .objectives import (ALL_OBJECTIVES, DISTRIBUTIONAL_OBJECTIVES, FROM_BASE,
                         REFERENCE_FREE)


@dataclass
class TrainingOutcome:
    run_dir: Path
    objective: str
    best_metric: float
    steps: int
    wallclock_s: float
    peak_vram_gb: float
    trainable_params: int
    batch_plan: dict[str, int]
    history: list[dict] = field(default_factory=list)
    resumed_from: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"run_dir": str(self.run_dir), "objective": self.objective,
                "best_metric": self.best_metric, "steps": self.steps,
                "wallclock_s": round(self.wallclock_s, 1),
                "gpu_hours": round(self.wallclock_s / 3600, 4),
                "peak_vram_gb": self.peak_vram_gb,
                "trainable_params": self.trainable_params,
                "batch_plan": self.batch_plan, "resumed_from": self.resumed_from}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def peak_vram_gb() -> float:
    try:
        import torch

        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / 1e9, 3)
    except ImportError:
        pass
    return 0.0


class ConstrainedF1Callback:
    """Evaluate with the decision rule the paper reports, not with the training loss.

    Selecting a checkpoint on token-level evaluation loss while reporting macro-F1 is a
    silent mismatch: on a one-token target the two are related only loosely, and the
    objectives differ in how much of their loss is about the label at all.
    """

    def __init__(self, model, tokenizer, space: LabelSpace, texts: Sequence[str],
                 labels: np.ndarray, batch_size: int = 32, max_length: int = 256,
                 logger: RunLogger | None = None):
        self.model, self.tokenizer, self.space = model, tokenizer, space
        self.texts = list(texts)
        self.labels = np.asarray(labels)
        self.batch_size, self.max_length = batch_size, max_length
        self.logger = logger
        self.history: list[dict] = []

    def evaluate(self, step: int = 0) -> dict[str, float]:
        from ..evaluate.metrics import expected_calibration_error, macro_f1
        from ..evaluate.scoring import VerbalizerScorer

        was_training = self.model.training
        self.model.eval()
        try:
            scorer = VerbalizerScorer(model=self.model, tokenizer=self.tokenizer,
                                      mixture_head=getattr(self.model, "mixture_head",
                                                           None),
                                      space=self.space, batch_size=self.batch_size,
                                      max_length=self.max_length)
            batch = scorer.score_texts(self.texts)
        finally:
            if was_training:
                self.model.train()

        scores = {
            "eval_macro_f1": macro_f1(self.labels, batch.predictions, self.space.size),
            "eval_accuracy": float((batch.predictions == self.labels).mean()),
            "eval_ece": expected_calibration_error(batch.probs, self.labels),
        }
        self.history.append({"step": step, **scores})
        if self.logger is not None:
            self.logger.metric("eval", step=step, **scores)
        return scores

    def as_transformers_callback(self):
        from transformers import TrainerCallback

        outer = self

        class _Callback(TrainerCallback):
            def on_evaluate(self, args, state, control, metrics=None, **kwargs):
                scores = outer.evaluate(state.global_step)
                if metrics is not None:
                    metrics.update(scores)
                return control

        return _Callback()


class BestAdapterTracker:
    """Checkpoint selection for the arms that cannot use ``load_best_model_at_end``.

    Every other arm gets selection from the HF trainer, which needs a periodic evaluation
    over an ``eval_dataset``. The distributional arms have none: CSPO's loss consumes the
    annotator distribution and, when anchored, one precomputed reference logit vector per
    example, so giving the trainer something to score would mean building a dev split in
    that format and running a second reference pass over it, every run.

    Selecting here keeps the *rule* identical across arms, the best dev macro-F1 measured
    every ``eval_steps`` by the same callback the others select on, without inventing an
    eval set for the trainer. The adapter is a few million parameters, so the
    snapshot is held in memory rather than written out.
    """

    def __init__(self, selector: "ConstrainedF1Callback", every: int,
                 logger: RunLogger | None = None):
        self.selector, self.every, self.logger = selector, every, logger
        self.best_score = float("-inf")
        self.best_step = -1
        self._state: dict | None = None
        self._last_step = -1

    def consider(self, model, step: int) -> dict[str, float]:
        """Evaluate at ``step``, keeping the adapter if it is the best seen."""
        if step == self._last_step:
            return {}
        self._last_step = step
        scores = self.selector.evaluate(step)
        if scores["eval_macro_f1"] > self.best_score:
            from peft import get_peft_model_state_dict

            self.best_score, self.best_step = scores["eval_macro_f1"], step
            self._state = {k: v.detach().to("cpu").clone()
                           for k, v in get_peft_model_state_dict(model).items()}
        return scores

    def restore(self, model) -> bool:
        """Put the best adapter back, unless it is already the one loaded."""
        if self._state is None or self.best_step == self._last_step:
            return False
        from peft import set_peft_model_state_dict

        set_peft_model_state_dict(model, self._state)
        if self.logger is not None:
            self.logger.event("best_checkpoint_restored", step=self.best_step,
                              eval_macro_f1=self.best_score,
                              discarded_step=self._last_step)
        return True

    def as_transformers_callback(self, model):
        from transformers import TrainerCallback

        outer = self

        class _Callback(TrainerCallback):
            def on_step_end(self, args, state, control, **kwargs):
                if outer.every and state.global_step % outer.every == 0:
                    outer.consider(model, state.global_step)
                return control

        return _Callback()


class MetricLogCallback:
    """Mirror every trainer log line into the run's metrics file."""

    def __init__(self, logger: RunLogger, guard: NanGuard):
        self.logger, self.guard = logger, guard

    def as_transformers_callback(self):
        from transformers import TrainerCallback

        outer = self

        class _Callback(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):
                if not logs:
                    return control
                outer.logger.metric("train", step=state.global_step, **logs)
                if "loss" in logs:
                    outer.guard.check(logs["loss"], state.global_step)
                return control

            def on_save(self, args, state, control, **kwargs):
                outer.logger.event("checkpoint_saved", step=state.global_step)
                return control

        return _Callback()


def run_training(
    cfg: ExperimentConfig,
    data: dict[str, Any],
    *,
    skip_if_complete: bool = True,
    reference_adapter: Path | None = None,
) -> TrainingOutcome | None:
    """Train one arm end to end.

    ``data`` carries whatever the objective needs: ``sft_records``, ``pref_pairs`` with
    ``pref_dev_pairs``, ``kto_examples`` with ``kto_dev_examples``, ``cspo_records``,
    plus ``eval_records`` used for checkpoint selection. The dev splits exist so each
    trainer evaluates on a schedule; what selects the checkpoint is always the dev
    macro-F1 measured by ``ConstrainedF1Callback``. Returns ``None`` when the run was
    already complete and skipped.
    """
    if cfg.train.objective not in ALL_OBJECTIVES:
        raise ValueError(f"unknown objective {cfg.train.objective!r}; "
                         f"known: {ALL_OBJECTIVES}")

    run_dir = cfg.run_dir
    if skip_if_complete and is_complete(run_dir):
        return None

    install_signal_handlers()
    seed_everything(cfg.train.seed)
    space = cfg.data.space

    # Read the previous attempt's configuration *before* the logger overwrites it.
    # Doing this after save_config compares the new hash against itself, which can never
    # differ, so the staleness check silently never fires and a checkpoint from a
    # different configuration is resumed. That is how a crippled adapter run survived a
    # corrected adapter configuration.
    stale = config_changed(run_dir, cfg.config_hash)

    with RunLogger(run_dir, cfg.run_id) as logger:
        logger.save_config(cfg.to_dict() | {"config_hash": cfg.config_hash})
        logger.start_telemetry(cfg.eval.telemetry_interval_s)

        with logger.stage("preflight"):
            # Read from the registry, not branched on a model name: the branch knew only
            # qwen-2b, so every model added after it would silently get the smaller floor.
            report = preflight(output_dir=run_dir,
                               min_vram_gb=MODEL_REGISTRY[cfg.model]["min_vram_gb"])
            logger.save("preflight.json",
                        {k: {"ok": ok, "detail": d} for k, (ok, d) in report.checks.items()})
            print(report.summary())
            report.raise_if_failed()

        if stale is not None:
            logger.event("config_changed_discarding_checkpoints",
                         stored_hash=stale, current_hash=cfg.config_hash,
                         removed=[str(p) for p in list_checkpoints(run_dir)])
            print(f"configuration changed since the last attempt "
                  f"({stale} -> {cfg.config_hash}); starting fresh rather than resuming")
            for path in list_checkpoints(run_dir):
                shutil.rmtree(path, ignore_errors=True)

        # After discarding, there is nothing to resume from; pass no hash so the
        # structural check alone decides.
        resume = resume_point(run_dir) if (cfg.train.resume and stale is None) else None
        if resume is not None:
            logger.event("resuming", checkpoint=str(resume))
            print(f"resuming from {resume}")

        with logger.stage("load_model"):
            if reference_adapter is not None and cfg.train.objective not in FROM_BASE:
                model, tokenizer = load_adapter(cfg.model_spec, reference_adapter,
                                                for_inference=False)
                set_training_mode(model)
                load_info = {"backend": "adapter", "from": str(reference_adapter)}
            else:
                model, tokenizer, load_info = load_causal_lm(cfg.model_spec,
                                                             for_training=True)
            logger.event("model_loaded", **load_info)

            table = resolve_verbalizers(tokenizer, space)
            logger.event("verbalizers_resolved", token_ids=list(table.token_ids),
                         surfaces=list(table.surfaces))

        eval_records = data["eval_records"]
        limit = cfg.eval.selection_subset
        if limit and len(eval_records) > limit:
            # Deterministic subset: the same items for every arm, so checkpoint selection
            # is comparable across the grid rather than varying with the seed.
            step = len(eval_records) / limit
            eval_records = [eval_records[int(i * step)] for i in range(limit)]
            logger.event("selection_subset", n=len(eval_records),
                         of=len(data["eval_records"]))
        selector = ConstrainedF1Callback(
            model, tokenizer, space,
            texts=[r["prompt"] for r in eval_records],
            labels=np.array([space.index(r["label"]) for r in eval_records]),
            batch_size=cfg.eval.batch_size, max_length=cfg.max_seq_length, logger=logger)
        guard = NanGuard(patience=cfg.train.nan_patience)

        plan = BatchPlan(cfg.train.per_device_batch_size,
                         cfg.train.gradient_accumulation_steps)
        started = time.time()

        def attempt(current: BatchPlan):
            logger.event("training_attempt", **current.as_dict(),
                         effective_batch=current.effective)
            return _dispatch(cfg, data, model, tokenizer, space, table, current,
                             selector, guard, logger, resume)

        with logger.stage("train"):
            result, used_plan = oom_backoff(
                attempt, plan, max_reductions=cfg.train.max_oom_reductions,
                on_backoff=lambda old, new, exc: (
                    logger.set_status(RunStatus.OOM_RETRY, from_plan=old.as_dict(),
                                      to_plan=new.as_dict()),
                    logger.event("oom_backoff", from_plan=old.as_dict(),
                                 to_plan=new.as_dict(), error=str(exc)[:400])))

        # Before anything is written: a resume that stepped nowhere must not replace the
        # record of the finished run it resumed from. Checked here rather than in each
        # objective because every arm reaches this one save stage.
        assert_resume_made_progress(result.get("steps"),
                                    result.get("resumed_from") or resume, run_dir)

        wallclock = time.time() - started
        trainable, total = count_trainable(model)

        with logger.stage("save"):
            manifest = {
                "run_id": cfg.run_id, "config_hash": cfg.config_hash,
                "objective": cfg.train.objective, "model": cfg.model,
                "seed": cfg.train.seed, "reference_free": cfg.train.objective in REFERENCE_FREE,
                "distributional": cfg.train.objective in DISTRIBUTIONAL_OBJECTIVES,
                "batch_plan": used_plan.as_dict(),
                "effective_batch": used_plan.effective,
                "wallclock_s": round(wallclock, 1),
                "gpu_hours": round(wallclock / 3600, 4),
                "estimated_gpu_hours": cfg.estimated_hours(),
                "peak_vram_gb": peak_vram_gb(),
                "trainable_params": trainable, "total_params": total,
                "steps": result.get("steps"), "train_loss": result.get("train_loss"),
                "resumed_from": str(resume) if resume else None,
                "nan_events": guard.history,
                "selection_history": selector.history,
                "verbalizer_token_ids": list(table.token_ids),
                **{k: v for k, v in result.items() if k not in ("steps", "train_loss")},
            }
            save_run(model, tokenizer, run_dir, manifest)
            # The mixture head is not part of the adapter, so save_run would not persist
            # it and evaluation would silently score the plain verbalizer softmax instead
            # of the distribution that was trained.
            head = getattr(model, "mixture_head", None)
            if head is not None:
                import torch

                torch.save({"state_dict": head.state_dict(),
                            "n_perspectives": int(head.n_perspectives),
                            "verbalizer_token_ids": list(table.token_ids)},
                           run_dir / "mixture_head.pt")
                manifest["mixture_perspectives"] = int(head.n_perspectives)
            manifest["adapter_mb"] = adapter_size_mb(run_dir)
            logger.save("manifest.json", manifest)
            removed = prune_checkpoints(run_dir, keep=1)
            logger.event("checkpoints_pruned", removed=[str(p) for p in removed])

        best = max((h["eval_macro_f1"] for h in selector.history), default=float("nan"))
        logger.event("training_complete", best_eval_macro_f1=best,
                     wallclock_s=round(wallclock, 1))

        return TrainingOutcome(
            run_dir=run_dir, objective=cfg.train.objective, best_metric=best,
            steps=int(result.get("steps") or 0), wallclock_s=wallclock,
            peak_vram_gb=peak_vram_gb(), trainable_params=trainable,
            batch_plan=used_plan.as_dict(), history=selector.history,
            resumed_from=str(resume) if resume else None)


def _dispatch(cfg, data, model, tokenizer, space, table, plan, selector, guard, logger,
              resume) -> dict:
    """Route to the trainer for this objective. All arms share everything else."""
    objective = cfg.train.objective
    if objective in DISTRIBUTIONAL_OBJECTIVES:
        return _train_cspo_family(cfg, data, model, tokenizer, space, table, plan,
                                  selector, guard, logger, resume)
    if objective == "sft":
        from .sft import train_sft

        return train_sft(cfg, data, model, tokenizer, plan, selector, guard, logger, resume)
    from .po import train_preference

    return train_preference(cfg, data, model, tokenizer, space, plan, selector, guard,
                            logger, resume)


def _build_mixture_head(cfg, model, table):
    """The mixture head for `mopa`, initialised from the model's own verbalizer readout.

    Returning None for every other objective keeps the trainer's fast path untouched. The
    head is initialised from the unembedding rows for the label tokens, so at A=1 and
    before any step the mixture reproduces the base model's label distribution exactly.
    """
    if cfg.train.objective != "mopa":
        return None
    import torch

    from .mixture import make_mixture_head

    embed = model.get_output_embeddings()
    if embed is None:
        raise RuntimeError("mopa needs the output embedding to initialise its heads")
    rows = embed.weight.detach()[list(table.token_ids)].float().cpu()
    head = make_mixture_head(rows.shape[1], rows.numpy(),
                             n_perspectives=cfg.train.mopa_perspectives,
                             init_scale=cfg.train.mopa_init_scale,
                             seed=cfg.train.seed)
    device = next(model.parameters()).device
    head = head.to(device=device, dtype=torch.float32)
    # Attached to the model so the Trainer's optimiser picks the parameters up.
    model.mixture_head = head
    return head


def _train_cspo_family(cfg, data, model, tokenizer, space, table, plan, selector, guard,
                       logger, resume) -> dict:
    """CSPO and its unanchored control."""
    import torch
    from datasets import Dataset
    from transformers import TrainingArguments

    from .cspo import CSPOCollator, make_cspo_trainer_class, precompute_reference_logits

    records = list(data["cspo_records"])
    # cspo_kl also needs the reference logits, it just uses them outside the softmax.
    use_reference = cfg.train.objective in ("cspo", "cspo_kl", "cspo_ada")

    if use_reference:
        with logger.stage("precompute_reference_logits"):
            ref = precompute_reference_logits(
                model, tokenizer, records, table, batch_size=cfg.eval.batch_size,
                max_length=cfg.max_seq_length, logger=logger)
            for record, values in zip(records, ref, strict=True):
                record["ref_logits"] = values
            logger.event("reference_logits_done", n=len(ref), k=len(ref[0]),
                         bytes=len(ref) * len(ref[0]) * 4)
        set_training_mode(model)

    # The column whitelist is what reaches the collator, so anything an objective needs
    # has to be named here. Polya models the raw annotator counts rather than matching
    # their frequencies, and dropping `votes` silently starved it: the loss guard fired
    # only because it refuses to fall back to frequency matching.
    columns = ["prompt", "p_human"]
    if use_reference:
        columns.append("ref_logits")
    if cfg.train.objective == "polya" or cfg.train.cspo_eb_alpha > 0:
        missing = [r for r in records if "votes" not in r]
        if missing:
            raise RuntimeError(
                f"{cfg.train.objective} needs raw annotator counts, but "
                f"{len(missing)} of {len(records)} records have no `votes`. Rebuild the "
                f"data with scripts/01_build_nli.py (or 01_build_data.py) from a version "
                f"that writes them.")
        columns += ["votes", "n_annotators"]
    train_ds = Dataset.from_list([{k: r[k] for k in columns} for r in records])

    args = TrainingArguments(
        output_dir=str(cfg.run_dir),
        num_train_epochs=cfg.train.num_train_epochs,
        learning_rate=cfg.train.po_learning_rate if use_reference else cfg.train.learning_rate,
        warmup_steps=_warmup_steps(cfg, cfg.train),
        weight_decay=cfg.train.weight_decay,
        lr_scheduler_type=cfg.train.lr_scheduler_type,
        max_grad_norm=cfg.train.max_grad_norm,
        logging_steps=cfg.train.logging_steps,
        # No eval_dataset exists in this format (see BestAdapterTracker), so the trainer
        # cannot run the evaluation that HF checkpoint selection hangs off. The tracker
        # below applies the same selection rule on the same schedule instead.
        eval_strategy="no",
        save_strategy="steps",
        save_steps=cfg.train.save_steps,
        save_total_limit=cfg.train.save_total_limit,
        optim=cfg.train.optim,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        gradient_checkpointing=cfg.train.gradient_checkpointing,
        seed=cfg.train.seed, data_seed=cfg.train.seed,
        report_to=[], remove_unused_columns=False,
        **plan.as_dict())

    tracker = BestAdapterTracker(selector, cfg.train.eval_steps, logger)
    trainer_class = make_cspo_trainer_class()
    trainer = trainer_class(
        model=model, args=args, train_dataset=train_ds,
        data_collator=CSPOCollator(tokenizer, max_length=cfg.max_seq_length),
        callbacks=[tracker.as_transformers_callback(model),
                   MetricLogCallback(logger, guard).as_transformers_callback()],
        verbalizer_ids=list(table.token_ids),
        beta=cfg.train.cspo_beta, use_reference=use_reference,
        kl_lambda=(cfg.train.cspo_kl_lambda
                   if cfg.train.objective == "cspo_kl" else None),
        ada_lambda0=(cfg.train.cspo_ada_lambda0
                     if cfg.train.objective == "cspo_ada" else None),
        eb_alpha=cfg.train.cspo_eb_alpha,
        mixture_head=_build_mixture_head(cfg, model, table) ,
        polya=(cfg.train.objective == "polya"),
        polya_scale=cfg.train.polya_scale,
        polya_max_concentration=cfg.train.polya_max_concentration)

    from ..recovery import train_with_resume_fallback

    result, used = train_with_resume_fallback(
        lambda ckpt: trainer.train(resume_from_checkpoint=str(ckpt) if ckpt else None),
        resume,
        on_fallback=lambda ckpt, exc: logger.event(
            "incompatible_checkpoint_discarded", checkpoint=str(ckpt),
            error=str(exc)[:300]))
    tracker.consider(model, int(result.global_step))
    restored = tracker.restore(model)
    return {"steps": int(result.global_step),
            "selected_step": tracker.best_step,
            "restored_best_checkpoint": restored,
            "train_loss": float(result.training_loss),
            "reward_stats": trainer._reward_stats[-20:],
            "n_train_records": len(records),
            "used_reference": use_reference}
