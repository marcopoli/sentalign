"""Experiment configuration, sized for a single RTX 3090.

Everything that varies across runs lives in one dataclass, so a run is fully described by
its configuration hash. The defaults are not generic: they are chosen against a stated
hardware budget and a stated throughput model, both recorded here so a reader can check
the arithmetic rather than take the run counts on trust.

Hardware assumption: one RTX 3090, 24 GB, Ampere, bf16 available, no FP8. VRAM is not the
binding constraint for models up to about 2B at 4-bit with rank-16 adapters and 256-token
sequences; throughput is. The design therefore treats three quantities as the levers:
sequence length, number of training examples seen, and forward passes per example.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .labels import LABEL_SPACES, LabelSpace
from .modeling import MODEL_REGISTRY, LoraConfigSpec, ModelSpec

#: Five seeds, fixed in advance by the power analysis in ``evaluate.stats``. Under the
#: paired design used here (all arms share their model's SFT reference and are scored on
#: identical items, pooled to roughly 11,000) the minimum detectable effect at 80 percent
#: power is 0.56 macro-F1 points; an unpaired comparison on a single 3,600-item test set
#: at the same budget would resolve only 1.53 points. Differences below 0.56 points are
#: reported as inconclusive rather than as findings.
SEEDS = (13, 21, 34, 55, 89)

#: Throughput model for the compute plan, in training examples per second at sequence
#: length 256 with 4-bit weights, rank-16 adapters and gradient checkpointing.
#:
#: These began as an analytic estimate (roughly 8 * N_params * n_tokens FLOPs per example
#: at an assumed 30 effective TFLOPS) and were about 1.8x optimistic against the first
#: measured run: an RTX 3090 delivered 24.9 examples per second on the 0.35B model where
#: the model predicted 45. The values below are anchored on that measurement and scaled by
#: parameter count, and they remain provisional: ``plan.measured_throughput`` overrides
#: them from ``runs/programme.jsonl`` as real runs land, so the plan sharpens rather than
#: staying wrong. Every run records its own wall-clock, and the paper reports measured
#: numbers, never these.
#:
#: Note that the first measurement was taken while the adapter configuration was reaching
#: only a fraction of the decoder (see ``lora_targets``); with full coverage these are
#: upper bounds and will be revised downward.
THROUGHPUT_EXAMPLES_PER_S: dict[str, float] = {
    # Measured on an RTX 3090 with full adapter coverage: 19.1 examples/s excluding
    # evaluation, 17.1 including it. The earlier 24.89 was taken while the adapter
    # configuration reached a tenth of the parameters it should have, so it was never a
    # valid baseline. Bounding the selection subset recovers part of the evaluation cost.
    "lfm-350m": 18.0,
    "lfm-1.2b": 6.0,
    # Measured on 2026-08-20 from a 21-step smoke run at n3000: 5.42 examples/s excluding
    # evaluation, with each dev evaluation costing 88.5s. Projected to a full run (446
    # steps, three evaluations) that is 4.9 all-in, so the 3.8 this entry carried was
    # conservative. It stays a projection from a short run, not a measurement of a full
    # one, and `plan.measured_throughput` replaces it once the ledger has real entries.
    "qwen-2b": 4.9,
    # A projection, not a measurement: qwen-2b's 4.9 scaled by parameter count (2.0B to
    # 3.08B). Replaced by `plan.measured_throughput` once the ledger has SmolLM3 entries.
    "smollm3-3b": 3.2,
    "deberta": 60.0,
}

#: Relative cost of one training example under each objective, in units of the SFT cost.
#: Pairwise objectives score two sequences per example. CSPO scores the prompt once and
#: reads the label logits from a single position, so it is cheaper than DPO by about the
#: factor that DPO spends on its second sequence.
OBJECTIVE_COST_FACTOR: dict[str, float] = {
    "sft": 1.0, "sft_soft": 1.0,
    "dpo": 2.0, "rdpo": 2.0, "ipo": 2.0, "mixdpo": 2.0, "grdpo": 2.0,
    "kto": 1.6, "simpo": 1.8, "alphapo": 1.8,
    "cspo": 1.0, "cspo_kl": 1.0, "cspo_ada": 1.0, "polya": 1.0, "mopa": 1.0,
}


@dataclass
class DataConfig:
    cache_dir: Path = Path(".cache/dynasent")
    build_dir: Path = Path("data/build")
    label_space: str = "ternary+mixed"
    tau: float = 0.2
    include_no_majority: bool = True
    max_pairs_per_item: int = 2
    noise_epsilon: float = 0.0
    noise_seed: int = 7
    ambig_fraction: float = 0.06
    holdout_seed: int = 20260819
    check_near_duplicates: bool = True
    near_threshold: float = 0.85
    kto_desirable_at: float = 0.6
    kto_undesirable_at: float = 0.2
    #: Training items retained after stratified subsampling. The full pool is 106,215
    #: items; at 14 examples per second a two-epoch pass over all of them is roughly four
    #: hours for the 1.2B model alone, which does not fit 130 GPU-hours across the grid.
    #: Subsampling is stratified by agreement band and gold label so the composition that
    #: the hypotheses depend on is preserved. A data-scaling ablation over
    #: {3k, 6k, 12k, 24k} tests whether the conclusions depend on this choice.
    train_subsample: int | None = 8_000
    subsample_seed: int = 101
    #: Cap on preference pairs, applied after subsampling and stratified by human margin.
    max_train_pairs: int | None = 8_000
    #: Target distribution shaping for the distributional objectives.
    target_temperature: float = 1.0
    target_smoothing: float = 0.0

    @property
    def space(self) -> LabelSpace:
        return LABEL_SPACES[self.label_space]


@dataclass
class TrainConfig:
    objective: str = "sft"
    seed: int = 13
    learning_rate: float = 2e-4          # LoRA rate; 2e-5 is a full fine-tuning rate
    po_learning_rate: float = 5e-5
    #: Micro-batch and accumulation. On 24 GB a 2B model at 256 tokens fits comfortably at
    #: 8; the OOM backoff in ``recovery`` halves the micro-batch and doubles accumulation,
    #: so the effective batch of 32 is preserved if a reduction is needed.
    per_device_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    num_train_epochs: float = 2.0
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"
    max_grad_norm: float = 1.0
    logging_steps: int = 25
    eval_steps: int = 150
    save_steps: int = 150
    save_total_limit: int = 2
    metric_for_best_model: str = "eval_macro_f1"
    greater_is_better: bool = True
    #: 0 disables early stopping, which is the default because it is the only setting
    #: that makes the arms comparable. With patience 3 at 150-step evaluations, nothing
    #: fires below ~450 steps, so every n8000 arm trains its full budget while the larger
    #: data-scaling points do not: DPO at n32000 stopped at 600 of 2000 steps with 4
    #: evaluation draws, against CSPO's 2000 steps and 14 draws, because the
    #: distributional arms select through BestAdapterTracker and never stop early. Best
    #: of 14 beats best of 4 from the same distribution, so the protocol was quietly
    #: favouring the proposed method in the one study built to compare them.
    early_stopping_patience: int = 0
    optim: str = "adamw_8bit"
    gradient_checkpointing: bool = True
    # Objective hyperparameters
    beta: float = 0.1                    # DPO, rDPO, IPO, MixDPO, GR-DPO
    label_smoothing: float = 0.1         # rDPO noise rate
    #: RPO (Pang et al. 2024): TRL adds rpo_alpha times the NLL of the preferred completion
    #: to every pair's loss. None leaves the DPO-family loss exactly as TRL defines it, so
    #: every arm that does not ask for the term trains as before.
    rpo_alpha: float | None = None
    simpo_beta: float = 2.0
    simpo_gamma: float = 0.5
    alphapo_alpha: float = -0.25   # reward-shape parameter; 0 recovers SimPO
    grdpo_step_size: float = 0.01
    mixdpo_threshold: float = 0.5        # human margin above which the preference loss applies
    mixdpo_sft_weight: float = 1.0
    cspo_beta: float = 1.0               # CSPO reward temperature
    #: Weight on the KL pull toward the reference in `cspo_kl`. 0 recovers sft_soft.
    cspo_kl_lambda: float = 1.0
    #: Base anchor strength for `cspo_ada`. The realised weight is
    #: lam0 * (1 - H(p_human)/log K), so 0 recovers sft_soft and a flat target is
    #: unanchored. See objectives.adaptive_lambda for why a global anchor cannot work.
    cspo_ada_lambda0: float = 1.0
    #: Dirichlet concentration for empirical-Bayes target shrinkage; 0 disables it.
    cspo_eb_alpha: float = 0.0
    #: Pólya alignment. `polya_scale` multiplies alpha_k = exp(l_k); the cap keeps
    #: lgamma finite for confidently unanimous items. Larger scale means a sharper
    #: Dirichlet, and the limit recovers sft_soft exactly.
    #: Mixture-of-perspectives head. A=1 reproduces sft_soft exactly, so the control is
    #: nested rather than adjacent. The head reads the final hidden state, which LoRA can
    #: move, unlike the logit magnitude that polya depended on.
    #: Distributional regularisation of a preference objective:
    #:     L = L_pref + lambda * CE(target, softmax(l_theta))
    #: Cross-entropy is a proper scoring rule, so its optimum is the true conditional and
    #: confidence tracks error probability. Margin-maximising preference losses are not
    #: proper: they tilt the distribution uniformly regardless of how reliable each item's
    #: label is, which manufactures confident errors on ambiguous items. Measured: KTO
    #: loses 0.11 to 0.18 AUROC of confidence against correctness in 8 of 8 dataset-model
    #: cells, while SimPO and AlphaPO, whose margins are normalised, are neutral.
    #: 0 reproduces the unregularised objective exactly.
    pref_distributional_lambda: float = 0.0
    #: `soft` regularises toward the annotator distribution, which is the claim. `hard`
    #: regularises toward the majority label and is the ablation that separates this from
    #: ordinary SFT regularisation.
    pref_regularizer: str = "soft"
    mopa_perspectives: int = 4
    mopa_init_scale: float = 0.02
    polya_scale: float = 1.0
    polya_max_concentration: float = 1e4
    kto_desirable_weight: float | None = None
    kto_undesirable_weight: float = 1.0
    # Recovery
    resume: bool = True
    max_oom_reductions: int = 3
    max_transient_retries: int = 4
    nan_patience: int = 1


#: Field bundles that switch the whole pipeline to a different task.
#:
#: A task is not one setting. It is a label space, a build directory, the evaluation sets
#: that exist in that directory, and the transfer sets that do not. Several of those are
#: tuples, which ``--set`` cannot express, so they travel together as a named bundle
#: rather than as six flags a caller could get half-right.
#:
#: The task is deliberately absent from ``run_id``: the study ``name`` separates NLI runs
#: from sentiment runs, so adding it would rewrite the identity of 283 completed runs to
#: no purpose.
TASKS: dict[str, dict[str, dict]] = {
    "sentiment": {},                      # the defaults on the dataclasses themselves
    "nli": {
        "data": {
            "label_space": "nli3",
            "build_dir": Path("data/build_nli"),
            "cache_dir": Path(".cache/nli"),
        },
        "eval": {
            # `ambig_eval` is the combined ChaosNLI set. Keeping the name means the
            # evaluator still derives `ambig_no_majority` from it and the aggregation
            # already understands both, so the NLI slices line up with the sentiment ones.
            # `ambig_eval` is the dense 100-annotator target, named so the evaluator
            # still derives `ambig_no_majority` from it. `chaos_coarse` is the *same
            # items* with their original five annotations: scoring one model against both
            # is the paired contrast this study is built on.
            "eval_sets": ("ambig_eval", "chaos_coarse"),
            "pooled_sets": ("ambig_eval",),
            "temperature_fit_split": "nli_dev",
            "transfer_sets": (),          # the sentiment transfer sets are not NLI
        },
    },
    #: GoEmotions, a second sentiment corpus whose rater-level emotion labels are mapped
    #: to the main study's four labels (``sentalign.data.goemotions``). Same label space,
    #: prompt and verbalizers as DynaSent, so the only thing that changes is the corpus.
    #: ``ambig_eval`` is held out of the training-side pool by agreement band, as on
    #: DynaSent; ``go_test`` is the official test split; ``go_dev`` selects checkpoints
    #: and fits temperatures.
    "goemo": {
        "data": {
            "build_dir": Path("data/build_goemo"),
            "cache_dir": Path(".cache/goemotions"),
        },
        "eval": {
            "eval_sets": ("ambig_eval", "go_test"),
            "pooled_sets": ("ambig_eval", "go_test"),
            "temperature_fit_split": "go_dev",
            "transfer_sets": (),
        },
    },
}


def apply_task(cfg: "ExperimentConfig", task: str) -> None:
    """Apply a task bundle in place. Unknown task names are an error, not a default."""
    if task not in TASKS:
        raise SystemExit(f"unknown task {task!r}; choose from {sorted(TASKS)}")
    for section_name, fields in TASKS[task].items():
        section = getattr(cfg, section_name)
        for key, value in fields.items():
            if not hasattr(section, key):
                raise SystemExit(f"task {task!r} sets unknown field {key!r}")
            setattr(section, key, value)


@dataclass
class EvalConfig:
    scoring_variant: str = "norm"        # raw | norm | pmi
    batch_size: int = 32
    also_free_generation: bool = True
    max_new_tokens: int = 8
    bootstrap_resamples: int = 10_000
    alpha: float = 0.05
    min_group_size: int = 30
    eval_sets: tuple[str, ...] = ("r1_test", "r2_test", "ambig_eval", "sst_dev_validated")
    #: Metrics are also computed on the pooled set. Pooling roughly triples the item count
    #: relative to round-1 test alone, which is the cheapest available reduction in the
    #: item-variance term of the power calculation.
    pooled_sets: tuple[str, ...] = ("r1_test", "r2_test", "ambig_eval")
    transfer_sets: tuple[str, ...] = ("tweeteval", "financial_phrasebank")
    temperature_fit_split: str = "r1_dev"
    telemetry_interval_s: float = 30.0
    #: Items used for checkpoint selection during training. The full development set is
    #: 3,554 items and was costing about 13 percent of a short run's wall-clock, scored
    #: twice per run for a decision that only needs to rank checkpoints. Final metrics
    #: always use the complete evaluation sets; this bound applies to selection only.
    selection_subset: int | None = 1_000


@dataclass
class ExperimentConfig:
    name: str = "main"
    model: str = "lfm-1.2b"
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    lora: LoraConfigSpec = field(default_factory=LoraConfigSpec)
    max_seq_length: int = 256
    load_in_4bit: bool = True
    output_root: Path = Path("runs")
    #: Distinguishes runs that share an objective but differ in overrides
    #: (cspo_hard / cspo_sharp / cspo_smooth, the cspo_beta sweep). Without it
    #: all five cspo-ablation variants collapsed onto the main-grid cspo
    #: run directory and were skipped as "already complete". Hyphen-joined,
    #: never underscores: double underscores delimit run_id fields.
    variant: str = ""

    @property
    def model_spec(self) -> ModelSpec:
        return ModelSpec(key=self.model, load_in_4bit=self.load_in_4bit,
                         max_seq_length=self.max_seq_length, lora=self.lora,
                         random_state=self.train.seed)

    @property
    def run_id(self) -> str:
        objective = self.train.objective
        if self.variant:
            objective = f"{objective}-{self.variant}"
        parts = [self.name, self.model, objective,
                 f"eps{self.data.noise_epsilon}", f"tau{self.data.tau}",
                 f"n{self.data.train_subsample or 'all'}", f"s{self.train.seed}"]
        return "__".join(parts)

    @property
    def run_dir(self) -> Path:
        return self.output_root / self.run_id

    def estimated_hours(self) -> float:
        """Planning estimate for this run, from the throughput model above."""
        rate = THROUGHPUT_EXAMPLES_PER_S.get(self.model, 10.0)
        factor = OBJECTIVE_COST_FACTOR.get(self.train.objective, 2.0)
        n = self.data.train_subsample or 106_215
        if self.train.objective not in ("sft", "sft_soft", "cspo", "cspo_kl",
                                        "cspo_ada", "polya", "mopa"):
            n = min(self.data.max_train_pairs or n, n * self.data.max_pairs_per_item)
        seen = n * self.train.num_train_epochs
        return round(seen * factor / rate / 3600.0, 3)

    def to_dict(self) -> dict[str, Any]:
        def walk(node):
            if isinstance(node, dict):
                return {k: walk(v) for k, v in node.items()}
            if isinstance(node, (list, tuple)):
                return [walk(v) for v in node]
            if isinstance(node, Path):
                return str(node)
            return node

        return walk(asdict(self))

    @property
    def config_hash(self) -> str:
        return hashlib.blake2b(
            json.dumps(self.to_dict(), sort_keys=True).encode(), digest_size=8).hexdigest()

    def save(self, path: Path | None = None) -> Path:
        path = path or (self.run_dir / "config.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict()
        payload["config_hash"] = self.config_hash
        payload["run_id"] = self.run_id
        payload["estimated_hours"] = self.estimated_hours()
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> "ExperimentConfig":
        payload = json.loads(Path(path).read_text())
        for key in ("config_hash", "run_id", "estimated_hours"):
            payload.pop(key, None)
        cfg = cls(name=payload["name"], model=payload["model"],
                  max_seq_length=payload["max_seq_length"],
                  load_in_4bit=payload["load_in_4bit"],
                  output_root=Path(payload["output_root"]),
                  variant=payload.get("variant", ""))
        for section, klass in (("data", DataConfig), ("train", TrainConfig),
                               ("eval", EvalConfig), ("lora", LoraConfigSpec)):
            values = payload.get(section, {})
            fields = {f for f in klass.__dataclass_fields__}
            kwargs = {}
            for k, v in values.items():
                if k not in fields:
                    continue
                if k in ("cache_dir", "build_dir"):
                    v = Path(v)
                if isinstance(v, list):
                    v = tuple(v)
                kwargs[k] = v
            setattr(cfg, section, klass(**kwargs))
        return cfg


def warmup_steps_for(cfg: "ExperimentConfig", n_examples: int | None = None) -> int:
    """Absolute warmup steps, since transformers 5.x deprecates ``warmup_ratio``.

    Converting here rather than passing a ratio keeps the schedule identical across arms
    that see different numbers of examples, which matters because the comparison is
    between objectives and not between learning-rate schedules.
    """
    n = n_examples or cfg.data.train_subsample or 8_000
    effective_batch = max(1, cfg.train.per_device_batch_size
                          * cfg.train.gradient_accumulation_steps)
    total = max(1, int(n * cfg.train.num_train_epochs / effective_batch))
    return max(1, int(round(total * cfg.train.warmup_ratio)))


def environment_manifest() -> dict[str, Any]:
    """Package versions and hardware, recorded with every run."""
    from .runlog import collect_environment

    return collect_environment()
