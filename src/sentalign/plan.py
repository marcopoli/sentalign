"""The experiment plan: one definition of what runs, used by the runner and the paper.

Keeping the plan in code rather than in prose means the run table in the manuscript, the
shell script that executes it, and the cost estimate cannot drift apart. ``sentalign plan``
prints it; ``scripts/02_run_grid.sh`` consumes it; the paper quotes its output.

Costs are planning estimates from the throughput model in ``config``. Every run records
measured wall-clock, and the paper reports measured numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from .config import SEEDS, ExperimentConfig, apply_task
from .train.objectives import ALL_OBJECTIVES

MAIN_MODELS = ("lfm-350m", "lfm-1.2b", "qwen-2b")
HOST = "lfm-1.2b"                     # ablation host: mid-scale, cheapest useful point
SEEDS3 = SEEDS[:3]

#: Objectives whose behaviour under preference noise the noise ladder tests. DPO is the
#: reference point, rDPO is the objective with a theoretical guarantee under exactly this
#: noise model, MixDPO is the 2026 routing approach, and CSPO is ours.
# cspo is deliberately absent for the same reason it is absent from
# MARGIN_ARMS below: the noise ladder flips preference pairs
# (pref/tau*/noise/eps*.json), and cspo consumes no pairs, so its
# eps>0 runs would silently train on the identical eps=0 data.
NOISE_ARMS = ("dpo", "rdpo", "mixdpo")
NOVEL_EPS = (0.1, 0.2, 0.4)

# cspo is deliberately absent: tau enters the pipeline only through preference-
# pair construction (data/build/pref/tau*), and cspo trains on graded soft
# targets (data/build/cspo/) that no tau touches. Including it would train the
# identical model once per tau under three names and report the copies as an
# ablation.
MARGIN_ARMS = ("dpo", "mixdpo")
NOVEL_TAU = (0.4, 0.6)

#: CSPO ablations. Each isolates one claim: that the distribution matters beyond its
#: argmax, that the reward temperature is not doing the work, and that smoothing the
#: target is not a substitute for the annotator signal.
CSPO_VARIANTS = (
    ("cspo_hard", {"target_temperature": 0.01}),
    ("cspo_sharp", {"target_temperature": 0.5}),
    ("cspo_smooth", {"target_smoothing": 0.1}),
)
CSPO_BETAS = (0.5, 2.0)

#: The KL weight in `cspo_kl`. lam=0 is exactly sft_soft, which the main grid already
#: trains, so the sweep brackets the default on both sides: a weak anchor and a strong
#: one. The beta sweep on cspo was what revealed that anchoring inside the softmax only
#: helps by pulling the policy back to the reference, so the analogous sweep here is the
#: evidence that the penalty formulation behaves differently.
CSPO_KL_LAMBDAS = (0.1, 10.0)

DATA_SCALING = (3_000, 16_000, 32_000)
SCALING_ARMS = ("sft", "dpo", "cspo")


@dataclass
class PlannedRun:
    study: str
    model: str
    objective: str
    seed: int
    noise_epsilon: float = 0.0
    tau: float = 0.2
    train_subsample: int | None = None
    overrides: dict = field(default_factory=dict)
    #: Dataset bundle (label space, build dir, eval sets). See config.TASKS.
    task: str = "sentiment"
    #: First segment of the run_id. Two tasks can otherwise produce identical ids, since
    #: the task is deliberately not part of the id itself.
    name: str = "main"
    #: An explicit directory name for arms whose name is not derivable from the override
    #: keys. ``pref_distributional_lambda=1.0`` renders as ``lambda1.0`` under the rule
    #: below, while the runs that exist, and the analysis that reads them, say
    #: ``dreg1.0``. Naming them apart would point the plan at directories that are not
    #: the ones on disk.
    variant: str = ""

    def to_config(self, base: ExperimentConfig | None = None) -> ExperimentConfig:
        cfg = base or ExperimentConfig()
        cfg.name = self.name
        apply_task(cfg, self.task)
        cfg.model = self.model
        # Membership, not a `cspo_` prefix. The prefix rule rewrote the *objective*
        # `cspo_kl` into `cspo` with variant `kl`, so its main-grid runs collided with
        # the ablation's and full_plan refused the plan. Only the names actually declared
        # as variants are variants; anything else is an objective in its own right.
        variant_names = {name for name, _ in CSPO_VARIANTS}
        if self.objective in variant_names:
            cfg.train.objective = self.objective.split("_", 1)[0]
            cfg.variant = self.objective.split("_", 1)[1]
        else:
            cfg.train.objective = self.objective
        if self.variant:
            cfg.variant = self.variant
        elif not cfg.variant and self.overrides:
            # Parameter sweeps on an unrenamed objective (the cspo_beta and
            # cspo_kl_lambda runs): encode the overriding values so each point owns
            # its directory.
            cfg.variant = "-".join(
                f"{key.rsplit('_', 1)[-1]}{value}"
                for key, value in sorted(self.overrides.items()))
        cfg.train.seed = self.seed
        cfg.data.noise_epsilon = self.noise_epsilon
        cfg.data.tau = self.tau
        if self.train_subsample is not None:
            cfg.data.train_subsample = self.train_subsample
            cfg.data.max_train_pairs = self.train_subsample
        for key, value in self.overrides.items():
            target = cfg.data if hasattr(cfg.data, key) else cfg.train
            setattr(target, key, value)
        return cfg

    @property
    def hours(self) -> float:
        return self.to_config().estimated_hours()

    def command(self) -> str:
        cfg = self.to_config()
        stage = "sft" if self.objective in ("sft", "sft_soft") else "po"
        parts = [f"sentalign {stage}", f"--model {self.model}", f"--seed {self.seed}"]
        if stage == "po":
            parts.append(f"--objective {cfg.train.objective}")
        if self.noise_epsilon:
            parts.append(f"--noise {self.noise_epsilon}")
        if self.tau != 0.2:
            parts.append(f"--tau {self.tau}")
        if self.train_subsample is not None:
            parts.append(f"--subsample {self.train_subsample}")
        for key, value in self.overrides.items():
            parts.append(f"--set {key}={value}")
        return " ".join(parts)


def main_grid() -> Iterator[PlannedRun]:
    for model in MAIN_MODELS:
        for seed in SEEDS:
            for objective in ALL_OBJECTIVES:
                yield PlannedRun("main grid", model, objective, seed)


def noise_ladder() -> Iterator[PlannedRun]:
    for eps in NOVEL_EPS:
        for objective in NOISE_ARMS:
            for seed in SEEDS3:
                yield PlannedRun("noise ladder", HOST, objective, seed, noise_epsilon=eps)


def margin_ablation() -> Iterator[PlannedRun]:
    for tau in NOVEL_TAU:
        for objective in MARGIN_ARMS:
            for seed in SEEDS3:
                yield PlannedRun("margin ablation", HOST, objective, seed, tau=tau)


def cspo_ablation() -> Iterator[PlannedRun]:
    for name, overrides in CSPO_VARIANTS:
        for seed in SEEDS3:
            yield PlannedRun("cspo ablation", HOST, name, seed, overrides=dict(overrides))
    for beta in CSPO_BETAS:
        for seed in SEEDS3:
            yield PlannedRun("cspo ablation", HOST, "cspo", seed,
                             overrides={"cspo_beta": beta})
    for lam in CSPO_KL_LAMBDAS:
        for seed in SEEDS3:
            yield PlannedRun("cspo ablation", HOST, "cspo_kl", seed,
                             overrides={"cspo_kl_lambda": lam})


def data_scaling() -> Iterator[PlannedRun]:
    for size in DATA_SCALING:
        for objective in SCALING_ARMS:
            for seed in SEEDS3:
                yield PlannedRun("data scaling", HOST, objective, seed,
                                 train_subsample=size)


def baselines() -> Iterator[PlannedRun]:
    for seed in SEEDS:
        for objective in ("encoder_hard", "encoder_soft"):
            yield PlannedRun("baselines", "deberta", objective, seed)
    for model in MAIN_MODELS:
        for shots in (0, 5):
            yield PlannedRun("baselines", model, f"prompt{shots}", 0)


#: Arms for the NLI study. Narrower than the main grid on purpose: the question here is
#: whether a *dense* target changes the distributional result, so it needs the two
#: distributional arms, the control that could refute them, and two reference points.
NLI_ARMS = ("sft", "sft_soft", "dpo", "cspo", "cspo_kl", "cspo_ada", "polya")
NLI_MODELS = (HOST, "qwen-2b")


def nli_study() -> Iterator[PlannedRun]:
    """SNLI/MNLI training against a 100-annotator ChaosNLI target.

    The sentiment study cannot tell distribution learning from confidence calibration,
    because a five-annotator target is quantised to multiples of 0.2 and one temperature
    sets the entropy level. This study puts the same arms against a target estimated from
    100 annotators, where per-item shape is real and a scalar cannot reproduce it.
    """
    for model in NLI_MODELS:
        for seed in SEEDS:
            for objective in NLI_ARMS:
                yield PlannedRun("nli", model, objective, seed,
                                 task="nli", name="nli")


#: Sweeps for the adaptive anchor, on the NLI task because only a dense target can
#: resolve differences of this size: the 100-annotator floor is ~0.004 JSD against
#: ~0.074 at five annotators.
ADA_LAMBDA0 = (0.5, 2.0)
ADA_EB_ALPHA = (1.0, 2.0)
#: Concentration scale for Pólya alignment. The limit of large scale is sft_soft, so a
#: sweep that brackets 1.0 also traces the path back to the control it generalises.
POLYA_SCALES = (0.25, 4.0)


def ada_ablation() -> Iterator[PlannedRun]:
    """What the two new components each contribute, separately."""
    for lam0 in ADA_LAMBDA0:
        for seed in SEEDS3:
            yield PlannedRun("ada ablation", HOST, "cspo_ada", seed, task="nli",
                             name="nli", overrides={"cspo_ada_lambda0": lam0})
    for alpha in ADA_EB_ALPHA:
        for seed in SEEDS3:
            yield PlannedRun("ada ablation", HOST, "cspo_ada", seed, task="nli",
                             name="nli", overrides={"cspo_eb_alpha": alpha})
    for scale in POLYA_SCALES:
        for seed in SEEDS3:
            yield PlannedRun("ada ablation", HOST, "polya", seed, task="nli",
                             name="nli", overrides={"polya_scale": scale})
    # Empirical-Bayes shrinkage applied to the reference-free control, which isolates the
    # target-denoising effect from the anchoring one.
    for alpha in ADA_EB_ALPHA:
        for seed in SEEDS3:
            yield PlannedRun("ada ablation", HOST, "sft_soft", seed, task="nli",
                             name="nli", overrides={"cspo_eb_alpha": alpha})


#: The distributional regulariser, as it is applied to a preference objective.
DREG = {"pref_distributional_lambda": 1.0, "pref_regularizer": "soft"}

#: A third model family for the confirmation set. The claim is about preference losses
#: that are not proper scoring rules, not about one architecture, so the two objectives
#: the regulariser repairs on lfm-1.2b and qwen-2b are run again here with their twins
#: and the SFT reference they anchor on. IPO is deliberately absent: it is the boundary
#: of the method rather than the claim, and it is established on the two models where the
#: full arm set ran. Five arms by five seeds, since a seed-paired comparison needs at
#: least three seeds in common and the pre-registered design declares five.
CONFIRM_MODEL = "smollm3-3b"
CONFIRM_ARMS = (("sft", {}, ""),
                ("kto", {}, ""), ("kto", DREG, "dreg1.0"),
                ("rdpo", {}, ""), ("rdpo", DREG, "dreg1.0"))


def regulariser_confirmation() -> Iterator[PlannedRun]:
    """The five-arm confirmation set on a third family."""
    for seed in SEEDS:
        for objective, overrides, variant in CONFIRM_ARMS:
            yield PlannedRun("regulariser confirmation", CONFIRM_MODEL, objective, seed,
                             overrides=dict(overrides), variant=variant)


#: The second task. Its degradation arms were launched outside the plan and are declared
#: here so the run table and the runs cannot drift apart; ``nli_study`` already emits sft,
#: sft_soft and dpo, so this study must not repeat them or two studies would claim one
#: directory. The regularised arms are what make NLI a replication of the whole claim
#: rather than of its first half.
NLI_REG_MODEL = "lfm-1.2b"
NLI_REG_ARMS = (("kto", {}, ""), ("kto", DREG, "dreg1.0"),
                ("rdpo", {}, ""), ("rdpo", DREG, "dreg1.0"),
                ("ipo", {}, ""), ("mixdpo", {}, ""))


def nli_regulariser() -> Iterator[PlannedRun]:
    """The regulariser on a second task, with the unregularised arms it is compared to."""
    for seed in SEEDS:
        for objective, overrides, variant in NLI_REG_ARMS:
            yield PlannedRun("nli regulariser", NLI_REG_MODEL, objective, seed,
                             overrides=dict(overrides), variant=variant,
                             task="nli", name="nli")


#: The objectives the third family never ran, which is why its column of the landscape
#: table is half empty. Every one of them is descriptive: none enters a confirmatory
#: family, because those compare a regularised arm against its own twin and against SFT,
#: and no arm here is regularised. Filling the column therefore cannot move a single
#: p-value in the confirmatory analysis, which is the property that makes it safe to run
#: after the endpoint was declared.
#:
#: ``ipo-dreg1.0`` is deliberately not here. It would be a regularised arm on a model
#: whose plan declares two, so it would enlarge P1 from eight tests to ten, P2 likewise
#: and P3 from sixteen to twenty, re-correcting every comparison in the paper. At the
#: current p-values two E-AURC cells fall out of significance if that arm's own repair
#: lands weak, so running it is a decision about the confirmatory claim and not about the
#: table, and it belongs to a separate declaration made before the run rather than to
#: this one.
LANDSCAPE_MODEL = "smollm3-3b"
LANDSCAPE_ARMS = ("sft_soft", "dpo", "ipo", "simpo", "alphapo", "grdpo", "mixdpo")


def landscape_completion() -> Iterator[PlannedRun]:
    """The descriptive arms that fill the third family's column of the landscape table."""
    for seed in SEEDS:
        for objective in LANDSCAPE_ARMS:
            yield PlannedRun("landscape completion", LANDSCAPE_MODEL, objective, seed)


#: The one regularised cell the third family lacks, declared after the confirmatory
#: families were fixed and therefore reported descriptively: it fills Table 1 and is compared
#: with its twin in the text, and the analysis's PLANNED mapping is left as it was, so the
#: declared families keep the size they had when the endpoint was chosen. Counting it would
#: re-correct every confirmatory p-value on the strength of a run chosen afterwards.
IPO_REG_MODEL = "smollm3-3b"


def ipo_reg_completion() -> Iterator[PlannedRun]:
    """IPO with the anchor on the third family, five seeds, outside the declared families."""
    for seed in SEEDS:
        yield PlannedRun("ipo-reg completion", IPO_REG_MODEL, "ipo", seed,
                         overrides=dict(DREG), variant="dreg1.0")


#: The second sentiment corpus. GoEmotions' rater-level labels mapped onto the same four
#: labels (``sentalign.data.goemotions``), the same prompt and verbalizers, and the five
#: arms that carry the claim: SFT, the two objectives whose ranking degrades, and each
#: with the anchor. Declared on 23 September 2026, before any GoEmotions run, together with
#: the analysis that reads it (``scripts/16_goemo_replication.py``): panel A compares KTO
#: and R-DPO with SFT, panel B each regularised arm with its twin, on AUROC and accuracy,
#: Holm-corrected within each panel over every model that ran. These families are separate
#: from the DynaSent ones, so nothing here moves a confirmatory p-value on DynaSent.
GOEMO_ARMS = CONFIRM_ARMS
GOEMO_MODELS = ("lfm-1.2b", "qwen-2b")
#: The third family, as a separate study because it costs about three times the other two
#: together and can be dropped without leaving a declared cell empty on them.
GOEMO_LARGE_MODEL = "smollm3-3b"


def _goemo(study: str, models) -> Iterator[PlannedRun]:
    for model in models:
        for seed in SEEDS:
            for objective, overrides, variant in GOEMO_ARMS:
                yield PlannedRun(study, model, objective, seed, overrides=dict(overrides),
                                 variant=variant, task="goemo", name="goemo")


def goemo_replication() -> Iterator[PlannedRun]:
    """Damage and repair on a second sentiment corpus, LFM2.5 and Qwen3.5."""
    return _goemo("goemo replication", GOEMO_MODELS)


def goemo_large() -> Iterator[PlannedRun]:
    """The same five arms on SmolLM3-3B."""
    return _goemo("goemo smollm3", (GOEMO_LARGE_MODEL,))


#: Sensitivity of the damage and of the repair to the preference temperature. Declared on
#: 28 September 2026, before any run, as a check requested of the paper rather than part of
#: its confirmatory design. Every arm in the paper trains at beta = 0.1; this study reruns
#: the two objectives that lose ranking, each with and without the anchor, at a third and
#: at three times that value, on the ablation host with three seeds, as the anchor-weight
#: check did. The anchor keeps lambda = 1, so the study also asks whether the weight fixed
#: at beta = 0.1 still repairs at another beta. Comparisons, descriptive like that check's
#: (seed-paired mean difference with a 95 percent interval, uncorrected p, and the number
#: of seeds that agree in sign), on the four-set mean AUROC, E-AURC and accuracy, at each
#: beta: KTO and R-DPO against SFT, and each anchored arm against its unanchored twin at
#: the same beta. SFT and the beta = 0.1 arms are the runs the paper already reports, so
#: nothing here can move a declared DynaSent p-value.
BETA_MODEL = HOST
BETA_VALUES = (0.03, 0.3)
BETA_ARMS = ("kto", "rdpo")


def beta_sensitivity() -> Iterator[PlannedRun]:
    """KTO and R-DPO, with and without the anchor, at beta 0.03 and 0.3."""
    for beta in BETA_VALUES:
        for seed in SEEDS3:
            for objective in BETA_ARMS:
                yield PlannedRun("beta sensitivity", BETA_MODEL, objective, seed,
                                 overrides={"beta": beta})
                # beta before dreg, so the analysis's twin rule (strip "-dreg...") maps
                # the anchored arm onto the unanchored one at the same beta.
                yield PlannedRun("beta sensitivity", BETA_MODEL, objective, seed,
                                 overrides={"beta": beta, **DREG},
                                 variant=f"beta{beta}-dreg1.0")


#: TRL's own likelihood term against the anchor. Declared on 28 September 2026, before any
#: run. RPO (Pang et al. 2024) adds to every pair's DPO-family loss the negative
#: log-likelihood of the preferred completion; TRL implements it as ``rpo_alpha`` and
#: recommends 1.0. It is the closest off-the-shelf alternative to the anchor: a likelihood
#: term on whichever label won the pair, including the pairs between two minority labels,
#: over the whole completion rather than over the closed label set at the decision
#: position. R-DPO is rerun with it on the two families where the full arm set ran, five
#: seeds, against the existing R-DPO, R-DPO+reg and SFT runs of the same seeds.
#: Declared family, Holm over its four tests, seed-paired t on the four-set mean AUROC:
#: R-DPO+RPO against R-DPO, and R-DPO+reg against R-DPO+RPO, on each model. The same pairs
#: on E-AURC and accuracy, and R-DPO+RPO against SFT, are descriptive. KTO is absent
#: because TRL's KTO has no such term; IPO is absent because it is the paper's boundary
#: case rather than its claim.
RPO_MODELS = ("lfm-1.2b", "qwen-2b")
RPO = {"rpo_alpha": 1.0}


def rpo_comparison() -> Iterator[PlannedRun]:
    """R-DPO with TRL's RPO term, the off-the-shelf rival to the anchor."""
    for model in RPO_MODELS:
        for seed in SEEDS:
            yield PlannedRun("rpo comparison", model, "rdpo", seed, overrides=dict(RPO),
                             variant="rpo1.0")


STUDIES = {
    "main": main_grid,
    "beta": beta_sensitivity,
    "rpo": rpo_comparison,
    "goemo": goemo_replication,
    "goemo-smollm3": goemo_large,
    "confirm": regulariser_confirmation,
    "landscape": landscape_completion,
    "ipo-reg": ipo_reg_completion,
    "nli-reg": nli_regulariser,
    "noise": noise_ladder,
    "margin": margin_ablation,
    "cspo": cspo_ablation,
    "scaling": data_scaling,
    "baselines": baselines,
    "nli": nli_study,
    "ada": ada_ablation,
}


def full_plan(studies: tuple[str, ...] | None = None) -> list[PlannedRun]:
    names = studies or tuple(STUDIES)
    runs = [run for name in names for run in STUDIES[name]()]
    seen: dict[str, str] = {}
    for run in runs:
        rid = run.to_config().run_id
        if rid in seen:
            raise ValueError(
                f"plan emits run_id {rid!r} twice (studies {seen[rid]!r} and "
                f"{run.study!r}); distinct runs must map to distinct "
                f"directories or one silently shadows the other")
        seen[rid] = run.study
    return runs


def measured_throughput(runs_dir: Path | str = "runs") -> dict[str, float]:
    """Examples per second per model, from completed runs.

    A planning estimate that never learns from its own programme is a guess repeated. The
    ledger written by the grid runner carries measured wall-clock per run, and the run
    manifests carry the example counts, so after a handful of runs the plan can be
    re-costed from data rather than from a FLOPs model.
    """
    import json

    runs_dir = Path(runs_dir)
    ledger = runs_dir / "programme.jsonl"
    if not ledger.exists():
        return {}

    totals: dict[str, list[float]] = {}
    for line in ledger.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not row.get("ok") or not row.get("measured_hours"):
            continue
        manifest = runs_dir / row["run_id"] / "manifest.json"
        if not manifest.exists():
            continue
        try:
            data = json.loads(manifest.read_text())
        except json.JSONDecodeError:
            continue
        model = data.get("model")
        seen = data.get("n_train_records") or data.get("n_train_pairs")
        if not (model and seen and data.get("wallclock_s")):
            continue
        epochs = 2.0
        totals.setdefault(model, []).append(seen * epochs / data["wallclock_s"])

    return {model: sum(values) / len(values) for model, values in totals.items()}


def summarise(runs: list[PlannedRun], runs_dir: Path | str | None = None) -> str:
    measured = measured_throughput(runs_dir) if runs_dir else {}

    by_study: dict[str, list[PlannedRun]] = {}
    for run in runs:
        by_study.setdefault(run.study, []).append(run)

    lines = []
    if measured:
        from .config import THROUGHPUT_EXAMPLES_PER_S

        lines.append("throughput (examples/s), planned vs measured:")
        for model, rate in sorted(measured.items()):
            planned = THROUGHPUT_EXAMPLES_PER_S.get(model)
            ratio = f"{rate / planned:.2f}x" if planned else "n/a"
            lines.append(f"  {model:<12} planned {planned or 0:>6.1f}  "
                         f"measured {rate:>6.1f}  ({ratio})")
        lines.append("")
    lines.append(f"{'study':<18}{'runs':>6}{'est. GPU-h':>12}")
    lines.append("-" * 36)
    total_runs = total_hours = corrected_hours = 0.0
    for study, group in by_study.items():
        hours = sum(_a_priori_hours(r) for r in group)
        lines.append(f"{study:<18}{len(group):>6}{hours:>12.1f}")
        total_runs += len(group)
        total_hours += hours
        corrected_hours += sum(_a_priori_hours(r) * _measured_correction(r, measured)
                               for r in group)
    lines.append("-" * 36)
    lines.append(f"{'total':<18}{total_runs:>6.0f}{total_hours:>12.1f}")
    lines.append("")
    lines.append(f"Evaluation adds roughly 10 percent. Single RTX 3090 at "
                 f"{total_hours * 1.1:.0f} GPU-h is about "
                 f"{total_hours * 1.1 / 24:.1f} days of continuous running.")
    if measured and abs(corrected_hours - total_hours) > 0.05 * max(total_hours, 1e-9):
        # The a-priori figure is a throughput constant times an objective cost factor, and
        # the programme has since measured the throughput. Reporting only the constant
        # would be planning from an assumption the machine has already contradicted, so the
        # corrected total is printed beside it and the reader can see the size of the gap.
        lines.append("")
        lines.append(f"Re-costed from measured throughput: {corrected_hours:.1f} GPU-h "
                     f"({corrected_hours * 1.1:.0f} with evaluation, "
                     f"{corrected_hours * 1.1 / 24:.1f} days). The a-priori figure above is "
                     f"{total_hours / corrected_hours:.1f}x that.")
    return "\n".join(lines)


def _a_priori_hours(run: PlannedRun) -> float:
    if run.objective.startswith("prompt"):
        return 0.05
    if run.objective.startswith("encoder"):
        return 0.2
    return run.hours


def _measured_correction(run: PlannedRun, measured: dict[str, float]) -> float:
    """How much the measured throughput moves this run's estimate, as a multiplier.

    Proportional and nothing more: the planned hours assume a rate, the ledger has
    observed one, so the estimate is scaled by their ratio. No new cost factor is invented
    here, because the per-objective factors have not been re-measured and a correction
    that mixed a measured rate with a guessed factor would report a precision it does not
    have.
    """
    from .config import THROUGHPUT_EXAMPLES_PER_S

    planned = THROUGHPUT_EXAMPLES_PER_S.get(run.model)
    rate = measured.get(run.model)
    if not planned or not rate:
        return 1.0
    return planned / rate
