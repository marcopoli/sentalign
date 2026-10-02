"""Run the encoder and prompting baselines, producing the same artifacts as a training arm.

The reference points for RQ6 and RQ7 have to be scored exactly like everything else, or
the comparison is confounded by the decision rule rather than by the model. So both
baselines go through the same evaluation sets, the same metric suite, and the same output
layout as a fine-tuned arm: a run directory with ``config.json``, ``eval/metrics.json``,
and per-item predictions that the aggregation script consumes without special-casing.

Two baselines:

``encoder``   DeBERTa-v3-base fine-tuned as a classifier, with hard majority labels and
              with the annotator distribution as a soft target. The soft variant is the
              natural non-preference way to use human label variation, and an objective
              that cannot beat it has not earned its complexity.
``prompt``    the base SLM with no training, zero-shot and five-shot, scored with the same
              single-pass verbalizer scorer. This is the floor that says what the whole
              pipeline actually bought.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from ..config import ExperimentConfig
from ..labels import LABEL_SPACES, LabelSpace
from ..runlog import RunLogger, is_complete


def _load_items(build_dir: Path, name: str):
    from ..data.dynasent import Item

    path = Path(build_dir) / "eval" / f"{name}.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [Item(text_id=r["text_id"], text=r["text"], votes=r["votes"], gold=r["gold"],
                 round=r["round"], source=r["source"], meta=r.get("meta", {}))
            for r in rows]


def _build_eval_sets(cfg: ExperimentConfig, space: LabelSpace):
    from ..evaluate.run_eval import (concat_eval_sets, eval_set_from_items,
                                     eval_set_no_majority)

    eval_sets, pooled = [], []
    for name in cfg.eval.eval_sets:
        items = _load_items(cfg.data.build_dir, name)
        built = eval_set_from_items(items, space, name)
        eval_sets.append(built)
        if name in cfg.eval.pooled_sets:
            pooled.append(built)
        if name == "ambig_eval":
            eval_sets.append(eval_set_no_majority(items, space, "ambig_no_majority"))
    if pooled:
        eval_sets.append(concat_eval_sets(pooled, "pooled"))
    fit = eval_set_from_items(
        _load_items(cfg.data.build_dir, cfg.eval.temperature_fit_split), space, "fit")
    return eval_sets, fit


def _score_and_save(cfg, space, eval_sets, fit_set, score_fn, logger, extra: dict) -> None:
    """Apply ``score_fn(texts) -> logits`` to every set and write the standard artifacts."""
    from ..evaluate.metrics import evaluate_predictions
    from ..evaluate.run_eval import EvaluationOutput

    fit_mask = fit_set.labels >= 0
    fit_logits = score_fn(list(np.asarray(fit_set.texts)[fit_mask]))
    fit_labels = fit_set.labels[fit_mask]

    results, per_item, total, started = {}, {}, 0, time.time()
    for eval_set in eval_sets:
        logits = score_fn(eval_set.texts)
        probs = np.exp(logits - logits.max(1, keepdims=True))
        probs /= probs.sum(1, keepdims=True)
        preds = logits.argmax(1)
        total += len(eval_set.texts)

        labelled = eval_set.labels >= 0
        if labelled.any():
            results[eval_set.name] = evaluate_predictions(
                logits[labelled], eval_set.labels[labelled], space.size,
                p_human=(eval_set.p_human[labelled] if eval_set.p_human is not None
                         else None),
                groups=[g for g, keep in zip(eval_set.groups, labelled) if keep],
                temperature_fit_logits=fit_logits, temperature_fit_labels=fit_labels)
        elif eval_set.p_human is not None:
            from ..evaluate.metrics import EvalResult, human_distribution_metrics

            results[eval_set.name] = EvalResult(
                n=len(eval_set.texts),
                metrics={"n": float(len(eval_set.texts)),
                         **human_distribution_metrics(probs, eval_set.p_human)})

        per_item[eval_set.name] = [
            {"text_id": tid, "gold": int(gold), "pred": int(pred),
             "probs": [float(x) for x in prob], "logits": [float(x) for x in logit],
             "group": group, "correct": bool(gold == pred) if gold >= 0 else None,
             "p_human": [float(x) for x in ph] if ph is not None else None,
             "gen_label": None, "gen_text": ""}
            for tid, gold, pred, prob, logit, group, ph in zip(
                eval_set.text_ids, eval_set.labels, preds, probs, logits,
                eval_set.groups,
                eval_set.p_human if eval_set.p_human is not None else [None] * len(preds),
                strict=True)]
        logger.event("eval_set_scored", name=eval_set.name, n=len(eval_set.texts))

    wallclock = time.time() - started
    output = EvaluationOutput(run_id=cfg.run_id, results=results, per_item=per_item,
                              wallclock_s=wallclock,
                              throughput_items_per_s=total / max(wallclock, 1e-9),
                              scoring_path=extra.get("scoring_path", "baseline"))
    output.save(cfg.run_dir / "eval")
    logger.save("manifest.json", {"run_id": cfg.run_id, "baseline": True, **extra})

    for name, result in results.items():
        m = result.metrics
        print(f"  {name:<20} n={int(m.get('n', 0)):>6}  "
              f"macro-F1={m.get('macro_f1', float('nan')):.4f}  "
              f"ECE={m.get('ece', float('nan')):.4f}  "
              f"JSD={m.get('jsd_human', float('nan')):.4f}")


def run_encoder_baseline(cfg: ExperimentConfig, *, soft_labels: bool,
                         skip_if_complete: bool = True) -> int:
    """Fine-tune and evaluate the encoder baseline."""
    from .encoder import EncoderConfig, score_encoder, train_encoder

    if skip_if_complete and is_complete(cfg.run_dir):
        print(f"{cfg.run_id}: already complete")
        return 0
    space = cfg.data.space

    with RunLogger(cfg.run_dir, cfg.run_id) as logger:
        logger.save_config(cfg.to_dict() | {"config_hash": cfg.config_hash})
        logger.start_telemetry(cfg.eval.telemetry_interval_s)

        build = Path(cfg.data.build_dir)
        n = cfg.data.train_subsample
        # The soft variant needs the annotator distribution, which only the
        # distributional records carry; the hard variant uses the majority label.
        source = "cspo" if soft_labels else "sft"
        train = [json.loads(l) for l in
                 (build / source / f"train_n{n}.jsonl").read_text().splitlines() if l.strip()]
        dev = [json.loads(l) for l in
               (build / "sft" / "dev.jsonl").read_text().splitlines() if l.strip()]
        for record in train:
            record.setdefault("text", record["prompt"])
            if soft_labels and "label" not in record:
                record["label"] = space.labels[int(np.argmax(record["p_human"]))]

        with logger.stage("train_encoder"):
            info = train_encoder(EncoderConfig(seed=cfg.train.seed), train, dev, space,
                                 cfg.run_dir / "encoder", soft_labels=soft_labels)
            logger.event("encoder_trained", **info)

        eval_sets, fit_set = _build_eval_sets(cfg, space)
        with logger.stage("score"):
            _score_and_save(
                cfg, space, eval_sets, fit_set,
                lambda texts: score_encoder(cfg.run_dir / "encoder", texts, space,
                                            batch_size=cfg.eval.batch_size,
                                            max_length=cfg.max_seq_length),
                logger, {"baseline": "encoder", "soft_labels": soft_labels, **info})
    return 0


def run_prompt_baseline(cfg: ExperimentConfig, *, n_shots: int,
                        skip_if_complete: bool = True) -> int:
    """Score an untrained model, zero-shot or few-shot."""
    from ..modeling import (apply_inference_attn_implementation, load_causal_lm,
                            set_inference_mode)
    from .prompting import sample_demonstrations, score_prompted

    if skip_if_complete and is_complete(cfg.run_dir):
        print(f"{cfg.run_id}: already complete")
        return 0
    space = cfg.data.space

    with RunLogger(cfg.run_dir, cfg.run_id) as logger:
        logger.save_config(cfg.to_dict() | {"config_hash": cfg.config_hash})
        logger.start_telemetry(cfg.eval.telemetry_interval_s)

        with logger.stage("load_model"):
            model, tokenizer, info = load_causal_lm(cfg.model_spec, for_training=False)
            set_inference_mode(model)
            info["attn_implementation"] = apply_inference_attn_implementation(
                cfg.model_spec, model)
            logger.event("model_loaded", **info)

        demos = []
        if n_shots:
            build = Path(cfg.data.build_dir)
            pool = [json.loads(l) for l in
                    (build / "sft" / "train_n3000.jsonl").read_text().splitlines()
                    if l.strip()]
            for record in pool:
                record.setdefault("text", record["prompt"])
            demos = sample_demonstrations(pool, space, n_shots, seed=cfg.train.seed)
            logger.event("demonstrations", n=len(demos),
                         labels=[label for _, label in demos])

        eval_sets, fit_set = _build_eval_sets(cfg, space)
        # Few-shot prompts carry the demonstrations, so they need more room.
        max_length = cfg.max_seq_length * (1 + n_shots // 2) if n_shots else cfg.max_seq_length
        with logger.stage("score"):
            _score_and_save(
                cfg, space, eval_sets, fit_set,
                lambda texts: score_prompted(model, tokenizer, space, texts, demos=demos,
                                             variant=cfg.eval.scoring_variant,
                                             batch_size=cfg.eval.batch_size,
                                             max_length=max_length),
                logger, {"baseline": "prompt", "n_shots": n_shots,
                         "scoring_path": "single_pass", **info})
    return 0
