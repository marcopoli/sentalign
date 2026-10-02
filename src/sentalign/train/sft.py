"""Supervised fine-tuning, the reference policy for every preference arm.

The SFT baseline has to be strong, because every other arm is measured against it and an
undertrained baseline manufactures improvements that are really just more optimisation.
Three things that commonly weaken it are handled explicitly:

*   The learning rate is a LoRA rate. A value near 2e-5 is a full fine-tuning rate, and
    applying it to adapters whose B matrix is zero-initialised leaves them barely moved.
*   The checkpoint is selected on macro-F1 from the same verbalizer scorer the paper
    reports with, not on token-level evaluation loss.
*   The loss is computed on the completion only, so the gradient is not spent on the
    fixed instruction.

Items without a majority label are dropped here, because cross-entropy needs a point
target and inventing one for a genuinely split item is the modelling error this study is
about. Those items remain available to the preference and distributional arms, and that
asymmetry is the mechanism the paper tests.
"""

from __future__ import annotations

def _warmup_steps(cfg, tc) -> int:
    """Absolute warmup steps: transformers 5.x deprecates ``warmup_ratio``."""
    from ..config import warmup_steps_for

    return warmup_steps_for(cfg)


from typing import Any, Sequence

from ..config import ExperimentConfig
from ..recovery import BatchPlan


def build_sft_args(cfg: ExperimentConfig, plan: BatchPlan):
    import torch
    from trl import SFTConfig

    tc = cfg.train
    return SFTConfig(
        output_dir=str(cfg.run_dir),
        max_length=cfg.max_seq_length,
        num_train_epochs=tc.num_train_epochs,
        learning_rate=tc.learning_rate,
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
        # Spend the gradient on the answer, not on the instruction that precedes it.
        completion_only_loss=True,
        **plan.as_dict())


def train_sft(cfg: ExperimentConfig, data: dict[str, Any], model, tokenizer,
              plan: BatchPlan, selector, guard, logger, resume) -> dict:
    from datasets import Dataset
    from .po import optional_early_stopping
    from trl import SFTTrainer

    from .driver import MetricLogCallback

    records = list(data["sft_records"])
    eval_records = list(data["eval_records"])

    def to_dataset(rows: Sequence[dict]) -> "Dataset":
        return Dataset.from_list(
            [{"prompt": r["prompt"], "completion": r["completion"]} for r in rows])

    trainer = SFTTrainer(
        model=model,
        args=build_sft_args(cfg, plan),
        train_dataset=to_dataset(records),
        eval_dataset=to_dataset(eval_records),
        processing_class=tokenizer,
        callbacks=[selector.as_transformers_callback(),
                   MetricLogCallback(logger, guard).as_transformers_callback(),
                   *optional_early_stopping(cfg)])

    from ..recovery import train_with_resume_fallback

    result, used = train_with_resume_fallback(
        lambda ckpt: trainer.train(resume_from_checkpoint=str(ckpt) if ckpt else None),
        resume,
        on_fallback=lambda ckpt, exc: logger.event(
            "incompatible_checkpoint_discarded", checkpoint=str(ckpt),
            error=str(exc)[:300]))
    return {"steps": int(result.global_step),
            "train_loss": float(result.training_loss),
            "n_train_records": len(records),
            "resumed_from": str(used) if used else None}
