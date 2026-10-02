"""Evaluation orchestration: one model, many evaluation sets, one scoring rule.

Produces a per-item prediction record for every evaluation set. Per-item records: not
just aggregate scores: are what the paired bootstrap in ``stats`` consumes, and they are
released with the paper so a reader can recompute any number or apply a metric we did not
think of.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from math import isfinite

from ..labels import LabelSpace
from .metrics import EvalResult, counterfactual_flip_rate, evaluate_predictions
from .scoring import GenerationScorer, VerbalizerScorer


@dataclass
class EvalSet:
    """One evaluation set, already mapped into the model's label space."""

    name: str
    texts: list[str]
    labels: np.ndarray
    text_ids: list[str]
    groups: list[str] = field(default_factory=list)
    p_human: np.ndarray | None = None

    def __post_init__(self) -> None:
        n = len(self.texts)
        if not (len(self.labels) == len(self.text_ids) == n):
            raise ValueError(f"{self.name}: ragged evaluation set")
        if not self.groups:
            self.groups = ["all"] * n


def eval_set_from_items(items: Sequence, space: LabelSpace, name: str,
                        *, drop_no_majority: bool = True) -> EvalSet:
    """Build an ``EvalSet`` from DynaSent ``Item`` objects.

    Items with no annotator majority have no point label, so they are excluded from
    accuracy-style metrics by default. They are *not* discarded: ``eval_set_no_majority``
    keeps them for the distributional metrics, where "no majority" is a meaningful target
    rather than a missing one.
    """
    from ..labels import NO_MAJORITY

    keep = [i for i in items
            if not (drop_no_majority and i.gold == NO_MAJORITY) and
            (i.gold == NO_MAJORITY or space.contains(i.gold))]
    labels = np.array([space.index(i.gold) if i.gold != NO_MAJORITY else -1 for i in keep])
    return EvalSet(
        name=name,
        texts=[i.text for i in keep],
        labels=labels,
        text_ids=[i.text_id for i in keep],
        groups=[i.agreement_band for i in keep],
        p_human=np.array([i.distribution_vector(space) for i in keep]),
    )


def eval_set_no_majority(items: Sequence, space: LabelSpace, name: str) -> EvalSet:
    """The complement: only items where annotators did not reach a majority.

    Accuracy is undefined here; JSD to the human distribution is not. This slice is where
    H2 is most directly visible, because it is exactly the data SFT cannot train on.
    """
    from ..labels import NO_MAJORITY

    keep = [i for i in items if i.gold == NO_MAJORITY]
    return EvalSet(
        name=name, texts=[i.text for i in keep],
        labels=np.full(len(keep), -1),
        text_ids=[i.text_id for i in keep],
        groups=["no-majority"] * len(keep),
        p_human=np.array([i.distribution_vector(space) for i in keep]) if keep else None,
    )


@dataclass
class EvaluationOutput:
    run_id: str
    results: dict[str, EvalResult]
    per_item: dict[str, list[dict]]
    wallclock_s: float
    throughput_items_per_s: float
    scoring_path: str = "single_pass"
    #: The attention kernel the scoring actually ran on. Recorded because it is not the
    #: same for every family: SmolLM3 is pinned away from flex_attention, and a silent
    #: change here would be a difference between runs that nothing else would show.
    attn_implementation: str | None = None

    @staticmethod
    def _json_safe(node):
        """Map non-finite floats to null, recursively.

        ``json.dumps`` writes bare ``NaN`` and ``Infinity`` by default, which no strict
        JSON parser accepts: jq, R's jsonlite, and JavaScript all reject the file, and
        the only reader that tolerates it is the Python that wrote it. Undefined metrics
        do occur here by construction. ``worst_group_f1`` and ``group_gap`` are undefined
        when no group reaches ``min_size``, and the label-free slice has no point metrics
        at all. ``null`` says "undefined" in a way every consumer understands.
        """
        import math

        if isinstance(node, dict):
            return {k: EvaluationOutput._json_safe(v) for k, v in node.items()}
        if isinstance(node, (list, tuple)):
            return [EvaluationOutput._json_safe(v) for v in node]
        if isinstance(node, float) and not math.isfinite(node):
            return None
        return node

    def save(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "run_id": self.run_id,
            "wallclock_s": self.wallclock_s,
            "throughput_items_per_s": self.throughput_items_per_s,
            "scoring_path": self.scoring_path,
            "attn_implementation": self.attn_implementation,
            "sets": {name: {"metrics": r.metrics, "groups": r.groups}
                     for name, r in self.results.items()},
        }
        (out_dir / "metrics.json").write_text(
            json.dumps(self._json_safe(summary), indent=2, allow_nan=False) + "\n")
        for name, rows in self.per_item.items():
            with (out_dir / f"predictions_{name}.jsonl").open("w", encoding="utf-8") as fh:
                for row in rows:
                    fh.write(json.dumps(row) + "\n")


def evaluate_model(
    model,
    tokenizer,
    space: LabelSpace,
    eval_sets: Sequence[EvalSet],
    *,
    run_id: str = "run",
    scoring_variant: str = "mean",
    batch_size: int = 32,
    max_length: int = 256,
    also_free_generation: bool = True,
    temperature_fit_set: EvalSet | None = None,
    mixture_head=None,
    logger=None,
) -> EvaluationOutput:
    """Score every evaluation set with the constrained scorer, plus optional generation."""
    started = time.time()
    scorer = VerbalizerScorer(model=model, tokenizer=tokenizer, space=space,
                              mixture_head=mixture_head,
                              variant=scoring_variant, batch_size=batch_size,
                              max_length=max_length)
    if logger is not None:
        logger.event("scorer_ready", path=scorer.scoring_path, variant=scoring_variant,
                     verbalizer_token_ids=list(scorer.table.token_ids))

    fit_logits = fit_labels = None
    dist_temperature: float | None = None
    if temperature_fit_set is not None:
        mask = temperature_fit_set.labels >= 0
        batch = scorer.score_texts(temperature_fit_set.texts)
        fit_logits = batch.logits[mask]
        fit_labels = temperature_fit_set.labels[mask]
        if temperature_fit_set.p_human is not None:
            # A second temperature, fitted to the annotator distribution rather than to
            # hard labels, on the same held-out split. This is the control that matters
            # for every distributional claim: one scalar on a hard-label model reproduced
            # the distributional arms on the sentiment data, so each arm has to be shown
            # after the cheap fix, not only before it.
            from .metrics import fit_temperature_to_distribution

            dist_temperature = fit_temperature_to_distribution(
                batch.logits, temperature_fit_set.p_human)
            if logger is not None:
                logger.event("distribution_temperature_fitted",
                             temperature=round(dist_temperature, 5),
                             n=int(len(batch.logits)))

    generator = None
    if also_free_generation:
        generator = GenerationScorer(model=model, tokenizer=tokenizer, space=space,
                                     batch_size=max(1, batch_size // 2),
                                     max_length=max_length)

    results: dict[str, EvalResult] = {}
    per_item: dict[str, list[dict]] = {}
    total_items = 0

    for eval_set in eval_sets:
        batch = scorer.score_texts(eval_set.texts, eval_set.text_ids)
        probs, preds = batch.probs, batch.predictions
        total_items += len(eval_set.texts)

        gen_labels: list[str | None] = [None] * len(eval_set.texts)
        gen_texts: list[str] = [""] * len(eval_set.texts)
        if generator is not None:
            gen_labels, gen_texts = generator.predict(eval_set.texts)
        parse_failures = sum(1 for g in gen_labels if g is None) if generator else 0

        # Corrected with the held-out distributional temperature, so the reported
        # comparison is between methods *after* the cheap fix rather than before it.
        temp_probs = None
        if dist_temperature is not None and eval_set.p_human is not None:
            from .metrics import apply_temperature

            temp_probs = apply_temperature(batch.logits, dist_temperature)

        labelled = eval_set.labels >= 0
        if labelled.any():
            results[eval_set.name] = evaluate_predictions(
                batch.logits[labelled], eval_set.labels[labelled], space.size,
                p_human=(eval_set.p_human[labelled] if eval_set.p_human is not None else None),
                groups=[g for g, keep in zip(eval_set.groups, labelled) if keep],
                parse_failures=parse_failures,
                temperature_fit_logits=fit_logits, temperature_fit_labels=fit_labels)
        elif eval_set.p_human is not None:
            # No point labels (the no-majority slice): distributional metrics only.
            from .metrics import human_distribution_metrics

            results[eval_set.name] = EvalResult(
                n=len(eval_set.texts),
                metrics={"n": float(len(eval_set.texts)),
                         **human_distribution_metrics(probs, eval_set.p_human)})

        if temp_probs is not None and eval_set.name in results:
            from .metrics import human_distribution_metrics as _hdm

            keep = eval_set.labels >= 0 if labelled.any() else slice(None)
            results[eval_set.name].metrics.update({
                f"temp/{k}": v for k, v in
                _hdm(temp_probs[keep], eval_set.p_human[keep]).items()})
            results[eval_set.name].metrics["temp/temperature"] = float(dist_temperature)

        if logger is not None:
            logger.event("eval_set_scored", name=eval_set.name, n=len(eval_set.texts),
                         parse_failures=parse_failures)

        per_item[eval_set.name] = [
            {
                "text_id": tid,
                "gold": int(gold),
                "pred": int(pred),
                "probs": [float(p) for p in prob],
                "logits": [float(l) for l in logit],
                "group": group,
                "correct": bool(gold == pred) if gold >= 0 else None,
                "p_human": [float(p) for p in ph] if ph is not None else None,
                "temp_probs": ([float(p) for p in tp] if tp is not None else None),
                "gen_label": gen,
                "gen_text": text,
            }
            for tid, gold, pred, prob, logit, group, ph, tp, gen, text in zip(
                eval_set.text_ids, eval_set.labels, preds, probs, batch.logits,
                eval_set.groups,
                eval_set.p_human if eval_set.p_human is not None else [None] * len(preds),
                temp_probs if temp_probs is not None else [None] * len(preds),
                gen_labels, gen_texts, strict=True)
        ]

    wallclock = time.time() - started
    return EvaluationOutput(
        run_id=run_id, results=results, per_item=per_item, wallclock_s=wallclock,
        throughput_items_per_s=total_items / wallclock if wallclock else 0.0,
        scoring_path=scorer.scoring_path)


def evaluate_counterfactuals(
    model, tokenizer, space: LabelSpace, pairs: Sequence[dict],
    *, batch_size: int = 32, max_length: int = 256,
) -> dict[str, float]:
    """Sensitivity to sentiment-flipping edits and stability to non-flipping ones."""
    if not pairs:
        return {}
    scorer = ConstrainedScorer(model=model, tokenizer=tokenizer, space=space,
                               batch_size=batch_size, max_length=max_length)
    original = scorer.score_texts([p["original_text"] for p in pairs]).predictions
    edited = scorer.score_texts([p["edited_text"] for p in pairs]).predictions
    should_flip = np.array([bool(p["should_flip"]) for p in pairs])
    out = counterfactual_flip_rate(original, edited, should_flip)
    out["n_pairs"] = float(len(pairs))
    return out


def unselected_run_reason(run_dir: Path) -> str | None:
    """Why ``run_dir`` holds no adapter the protocol selected, or ``None`` if it does.

    Evaluation loads whatever adapter sits at the run root. Two states leave the wrong one
    there, or none. Training that died before its save stage leaves no manifest and no
    adapter, and Unsloth then fails on a missing ``model_type`` that names neither cause.
    A relaunch that resumed a finished run and trained no step rewrote the root adapter
    with no checkpoint selection behind it; ipo-dreg1.0 seed 21 was scored in that state.
    Every trained arm evaluates on dev at least once (435 of 435 manifests on record), so
    an empty selection history always means the adapter was not chosen on dev macro-F1.
    """
    manifest = Path(run_dir) / "manifest.json"
    if not manifest.exists():
        return ("training never reached its save stage (no manifest.json), so there is no "
                "adapter to score; the cause is the run_failed event in events.jsonl")
    record = json.loads(manifest.read_text())
    if record.get("baseline"):
        return None
    if not record.get("selection_history"):
        return ("the manifest records no checkpoint selection, so the adapter at the run "
                "root was not chosen on dev macro-F1; delete the run directory and retrain")
    return None


def evaluate_run(run_dir: Path, build_dir: Path, *, force: bool = False) -> int:
    """Evaluate a finished training run and write everything it produces to disk.

    Reads the run's own ``config.json`` rather than accepting evaluation settings on the
    command line, so a run is always scored with the configuration it was trained under.
    Writes per-item predictions for every evaluation set: the aggregate metrics are a
    summary, but the paired bootstrap and any metric a reader wants to add later both
    need the per-item records, so those are the primary artifact.
    """
    import json

    from ..config import ExperimentConfig
    from ..data.dynasent import Item
    from ..labels import LABEL_SPACES
    from ..modeling import (apply_inference_attn_implementation, load_adapter,
                            set_inference_mode)
    from ..runlog import RunLogger

    run_dir = Path(run_dir)
    if (run_dir / "eval" / "metrics.json").exists() and not force:
        print(f"{run_dir.name}: already evaluated (pass --force to redo)")
        return 0
    if not (run_dir / "config.json").exists():
        print(f"{run_dir}: no config.json; not a completed run", file=sys.stderr)
        return 1
    refusal = unselected_run_reason(run_dir)
    if refusal is not None:
        print(f"{run_dir.name}: {refusal}", file=sys.stderr)
        return 1

    cfg = ExperimentConfig.load(run_dir / "config.json")
    space = LABEL_SPACES[cfg.data.label_space]
    build_dir = Path(build_dir)

    def load_items(name: str) -> list[Item]:
        rows = [json.loads(line) for line
                in (build_dir / "eval" / f"{name}.jsonl").read_text().splitlines()
                if line.strip()]
        return [Item(text_id=r["text_id"], text=r["text"], votes=r["votes"],
                     gold=r["gold"], round=r["round"], source=r["source"],
                     meta=r.get("meta", {})) for r in rows]

    with RunLogger(run_dir, cfg.run_id + "::eval") as logger:
        logger.start_telemetry(cfg.eval.telemetry_interval_s)

        mixture_head = None
        with logger.stage("load_model"):
            model, tokenizer = load_adapter(cfg.model_spec, run_dir, for_inference=True)
            set_inference_mode(model)
            # After set_inference_mode, not before: the backend's inference patch can
            # rewrite the config it is given, and the kernel that scores has to be the
            # one that survives every step of loading.
            attn = apply_inference_attn_implementation(cfg.model_spec, model)
            logger.event("attn_implementation", name=attn)
            head_file = run_dir / "mixture_head.pt"
            if head_file.exists():
                # The trained distribution is the mixture, not the verbalizer softmax.
                # Scoring without the head would silently evaluate a different model.
                import torch

                from ..train.mixture import make_mixture_head

                blob = torch.load(head_file, map_location="cpu", weights_only=False)
                embed = model.get_output_embeddings()
                rows = embed.weight.detach()[list(blob["verbalizer_token_ids"])] \
                    .float().cpu().numpy()
                mixture_head = make_mixture_head(rows.shape[1], rows,
                                                 n_perspectives=blob["n_perspectives"])
                mixture_head.load_state_dict(blob["state_dict"])
                device = next(model.parameters()).device
                mixture_head = mixture_head.to(device=device, dtype=torch.float32).eval()
                logger.event("mixture_head_loaded",
                             perspectives=int(blob["n_perspectives"]))

        eval_sets: list[EvalSet] = []
        pooled_parts: list[EvalSet] = []
        with logger.stage("prepare_eval_sets"):
            for name in cfg.eval.eval_sets:
                items = load_items(name)
                built = eval_set_from_items(items, space, name)
                eval_sets.append(built)
                if name in cfg.eval.pooled_sets:
                    pooled_parts.append(built)
                if name == "ambig_eval":
                    eval_sets.append(eval_set_no_majority(items, space,
                                                          "ambig_no_majority"))
            if pooled_parts:
                eval_sets.append(concat_eval_sets(pooled_parts, "pooled"))
            fit_set = eval_set_from_items(
                load_items(cfg.eval.temperature_fit_split), space, "fit")
            logger.event("eval_sets_ready",
                         sizes={e.name: len(e.texts) for e in eval_sets})

        with logger.stage("score"):
            output = evaluate_model(
                model, tokenizer, space, eval_sets, run_id=run_dir.name,
                mixture_head=mixture_head,
                scoring_variant=cfg.eval.scoring_variant,
                batch_size=cfg.eval.batch_size,
                max_length=cfg.max_seq_length,
                also_free_generation=cfg.eval.also_free_generation,
                temperature_fit_set=fit_set, logger=logger)
            output.attn_implementation = attn

        # A reference-anchored objective is scored on two distributions: the policy's
        # own, and the reward-induced one it was trained to shape. The reference has
        # already been evaluated on these items, so this costs no forward passes.
        if cfg.train.objective in REFERENCE_READOUT_OBJECTIVES:
            reference_dir = _reference_run_dir(cfg, run_dir)
            if reference_dir is not None and reference_dir.exists():
                with logger.stage("reward_readout"):
                    added = attach_reward_readout(output, reference_dir,
                                                  cfg.train.cspo_beta)
                    logger.event("reward_readout", reference=str(reference_dir),
                                 sets=sorted(added), beta=cfg.train.cspo_beta)
            else:
                logger.event("reward_readout_skipped",
                             reason="reference run not evaluated",
                             expected=str(reference_dir))
        output.save(run_dir / "eval")

        for name, result in output.results.items():
            m = result.metrics
            logger.metric("final", stage_name=name, **{k: v for k, v in m.items()
                                                       if isinstance(v, (int, float))})
            def cell(key: str) -> str:
                # A label-free slice has no macro-F1 to report, so say so instead of
                # printing nan, which reads as a failed computation rather than a
                # metric that is undefined for this set by construction.
                value = m.get(key)
                return f"{value:.4f}" if isinstance(value, float) and isfinite(value) \
                    else "   n/a"

            print(f"  {name:<20} n={int(m.get('n', 0)):>6}  "
                  f"macro-F1={cell('macro_f1')}  ECE={cell('ece')}  "
                  f"JSD={cell('jsd_human')}  AURC={cell('aurc')}")
        print(f"  scoring path: {output.scoring_path}, "
              f"{output.throughput_items_per_s:.1f} items/s")
    return 0


def concat_eval_sets(parts: Sequence[EvalSet], name: str) -> EvalSet:
    """Pool several evaluation sets into one.

    Pooling roughly triples the item count relative to any single set, which is the
    cheapest available reduction in the item-variance term of the power calculation and
    is why the pooled set is the one the headline comparison uses.
    """
    texts, labels, ids, groups, humans = [], [], [], [], []
    for part in parts:
        texts.extend(part.texts)
        labels.append(part.labels)
        ids.extend(f"{part.name}:{t}" for t in part.text_ids)
        groups.extend(f"{part.name}|{g}" for g in part.groups)
        if part.p_human is not None:
            humans.append(part.p_human)
    return EvalSet(name=name, texts=texts, labels=np.concatenate(labels),
                   text_ids=ids, groups=groups,
                   p_human=np.concatenate(humans) if len(humans) == len(parts) else None)


def load_per_item(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


#: Objectives whose loss is defined on a reward built from a reference, so the policy's
#: own softmax is not the distribution they fit. `cspo_kl` is deliberately absent: it
#: fits the policy's own distribution, which is what makes it comparable to sft_soft
#: without a second read-out.
REFERENCE_READOUT_OBJECTIVES = ("cspo",)


def _reference_run_dir(cfg, run_dir: Path) -> Path | None:
    """The SFT anchor this run was trained against, by the same rule training used.

    Mirrors ``cli._default_reference``: same model, seed, and subsample, at the default
    tau, because the supervised corpus has no tau or eps partition.
    """
    from ..config import ExperimentConfig

    sft = ExperimentConfig(name=cfg.name, model=cfg.model)
    sft.train.objective = "sft"
    sft.train.seed = cfg.train.seed
    sft.data.train_subsample = cfg.data.train_subsample
    sft.data.max_train_pairs = cfg.data.max_train_pairs
    sft.output_root = Path(run_dir).parent
    return sft.run_dir


def attach_reward_readout(output, reference_dir: Path, beta: float) -> dict[str, dict]:
    """Score a CSPO run through the distribution it actually parameterises.

    CSPO fits ``softmax(beta * (l_theta - l_ref))`` to the annotators, but every read-out
    here scores ``softmax(l_theta)``. Those are different distributions: at the optimum
    ``l_theta = l_ref + (1/beta) log p_human + c``, so the policy's own softmax is
    ``pi_ref * p_human**(1/beta)``, a product of experts with the reference. Measured on
    the grid, the policy read-out put CSPO at 0.3086 JSD on the no-majority items while
    the reward read-out reaches 0.1412, against 0.1403 for its control. Reporting only
    the first understates the objective by a factor of two.

    The reference logits are not recomputed. The reference is the SFT run this policy was
    anchored to, and it has already been evaluated on the same items, so its per-item
    logits are read from its predictions files and aligned by ``text_id``. Items the
    reference did not score are dropped rather than guessed.

    Mutates ``output`` in place: each per-item row gains ``reward_probs``, and each set's
    metrics gain a ``reward/`` prefixed block. Returns the per-set metrics it added.
    """
    added: dict[str, dict] = {}
    for name, rows in output.per_item.items():
        path = Path(reference_dir) / "eval" / f"predictions_{name}.jsonl"
        if not path.exists():
            continue
        reference = {}
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    row = json.loads(line)
                    reference[row["text_id"]] = row["logits"]

        probs, human, matched = [], [], 0
        for row in rows:
            ref = reference.get(row["text_id"])
            if ref is None or row.get("logits") is None:
                row["reward_probs"] = None
                continue
            z = beta * (np.asarray(row["logits"], dtype=np.float64)
                        - np.asarray(ref, dtype=np.float64))
            z = z - z.max()
            q = np.exp(z)
            q /= q.sum()
            row["reward_probs"] = [float(v) for v in q]
            matched += 1
            if row.get("p_human") is not None:
                probs.append(q)
                human.append(row["p_human"])

        if not probs:
            continue
        from .metrics import human_distribution_metrics

        block = {f"reward/{k}": v for k, v in human_distribution_metrics(
            np.array(probs), np.array(human)).items()}
        block["reward/n_matched"] = float(matched)
        block["reward/beta"] = float(beta)
        if name in output.results:
            output.results[name].metrics.update(block)
        added[name] = block
    return added


def per_item_scores(rows: Sequence[dict], metric: str = "correct") -> np.ndarray:
    """Extract the per-item score vector the paired bootstrap consumes."""
    if metric == "correct":
        return np.array([float(bool(r["correct"])) for r in rows])
    if metric == "nll":
        return np.array([-np.log(max(r["probs"][r["gold"]], 1e-12)) for r in rows])
    if metric == "jsd":
        from .metrics import jensen_shannon

        probs = np.array([r["probs"] for r in rows])
        human = np.array([r["p_human"] for r in rows])
        return jensen_shannon(probs, human)
    if metric == "jsd_temp":
        # The same policy distribution after the held-out distributional temperature.
        # This is the comparison that survives the obvious reviewer control: a single
        # scalar on a hard-label model matched every distributional arm on the sentiment
        # data, so every claim has to be shown after the cheap fix as well as before it.
        from .metrics import jensen_shannon

        missing = [r["text_id"] for r in rows if r.get("temp_probs") is None]
        if missing:
            raise ValueError(
                f"{len(missing)} rows have no temp_probs (e.g. {missing[0]}); "
                "re-evaluate with a fit split that carries p_human")
        return jensen_shannon(np.array([r["temp_probs"] for r in rows], dtype=float),
                              np.array([r["p_human"] for r in rows], dtype=float))
    if metric == "jsd_reward":
        # The CSPO reward distribution, attached by attach_reward_readout. Absent for
        # arms that have no reference, which is why this is opt-in rather than default.
        from .metrics import jensen_shannon

        missing = [r["text_id"] for r in rows if r.get("reward_probs") is None]
        if missing:
            raise ValueError(
                f"{len(missing)} rows have no reward_probs (e.g. {missing[0]}); this "
                "metric is only defined for runs evaluated against a reference")
        probs = np.array([r["reward_probs"] for r in rows])
        human = np.array([r["p_human"] for r in rows])
        return jensen_shannon(probs, human)
    if metric == "brier":
        out = []
        for r in rows:
            onehot = np.zeros(len(r["probs"]))
            onehot[r["gold"]] = 1.0
            out.append(float(((np.array(r["probs"]) - onehot) ** 2).sum()))
        return np.array(out)
    raise ValueError(f"unknown per-item metric {metric!r}")
