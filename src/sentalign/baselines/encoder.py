"""Fine-tuned encoder baseline.

A 184M DeBERTa-v3 \\cite{he2023debertav3} costs a fraction of any arm in the main grid.
If it matches the aligned SLMs in-domain, that is the single most useful sentence in the
paper for a practitioner, and the first thing a reviewer will ask about. Reporting it is
not a courtesy to the baseline; it is what makes the SLM numbers interpretable.

The encoder is scored on the same items with the same metrics, so its probabilities drop
straight into the same calibration, selective-prediction, and distributional analyses.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from ..labels import LabelSpace


@dataclass
class EncoderConfig:
    hf_id: str = "microsoft/deberta-v3-base"
    learning_rate: float = 2e-5          # a full fine-tuning rate, correct here
    batch_size: int = 32
    epochs: float = 3.0
    max_length: int = 256
    warmup_ratio: float = 0.06
    weight_decay: float = 0.01
    seed: int = 13


def train_encoder(
    cfg: EncoderConfig,
    train_records: Sequence[dict],
    dev_records: Sequence[dict],
    space: LabelSpace,
    out_dir: Path,
    *,
    soft_labels: bool = False,
) -> dict:
    """Fine-tune an encoder classifier.

    ``soft_labels`` trains against the full annotator distribution rather than the
    majority label. It is included because it is the natural non-preference way to use
    human label variation, and a preference method that cannot beat it has not earned its
    complexity: the comparison a paper about label variation owes the reader.
    """
    import torch
    from datasets import Dataset
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              Trainer, TrainingArguments)

    from ..evaluate.metrics import macro_f1

    tokenizer = AutoTokenizer.from_pretrained(cfg.hf_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.hf_id, num_labels=space.size,
        problem_type="multi_label_classification" if soft_labels else "single_label_classification")

    def encode(records: Sequence[dict]) -> "Dataset":
        rows = []
        for r in records:
            item = tokenizer(r["text"] if "text" in r else r["prompt"],
                             truncation=True, max_length=cfg.max_length)
            if soft_labels:
                item["labels"] = [float(p) for p in r["p_human"]]
            else:
                item["labels"] = space.index(r["label"])
            rows.append(item)
        return Dataset.from_list(rows)

    dev_labels = np.array([space.index(r["label"]) for r in dev_records])

    def compute_metrics(pred):
        logits = pred.predictions
        preds = logits.argmax(axis=-1)
        return {"macro_f1": macro_f1(dev_labels[:len(preds)], preds, space.size)}

    args = TrainingArguments(
        output_dir=str(out_dir),
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size * 2,
        num_train_epochs=cfg.epochs,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        seed=cfg.seed,
        bf16=torch.cuda.is_bf16_supported(),
        report_to=[],
    )
    started = time.time()
    trainer = Trainer(model=model, args=args, train_dataset=encode(train_records),
                      eval_dataset=encode(dev_records), processing_class=tokenizer,
                      compute_metrics=compute_metrics)
    trainer.train()
    out_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(out_dir)
    return {"wallclock_s": time.time() - started,
            "soft_labels": soft_labels,
            "params": sum(p.numel() for p in model.parameters())}


def score_encoder(model_dir: Path, texts: Sequence[str], space: LabelSpace,
                  batch_size: int = 64, max_length: int = 256) -> np.ndarray:
    """Return unnormalised logits, matching the interface of ``ConstrainedScorer``."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    out = []
    for start in range(0, len(texts), batch_size):
        chunk = list(texts[start:start + batch_size])
        enc = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True,
                        max_length=max_length).to(device)
        with torch.no_grad():
            out.append(model(**enc).logits.float().cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float64)
