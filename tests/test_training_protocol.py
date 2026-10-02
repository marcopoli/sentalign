"""The recipe every training arm shares, and the checkpoint each arm ends up with.

The arms are meant to differ in their objective and in nothing else. Three of them
differed in more: SimPO, AlphaPO, and KTO had their trainer configurations written out
separately and inherited TRL's defaults, so they ran with a linear schedule, no warmup,
and, most consequentially, no periodic evaluation, hence no checkpoint selection,
while the DPO family selected its best checkpoint on dev macro-F1. Best of three readings
against a single final one is a protocol advantage rather than an algorithmic one, in
exactly the comparison this study reports.

The same omission was also fatal: ``train_preference`` attaches an ``EarlyStoppingCallback``
to every arm, and that callback asserts on ``eval_strategy="no"`` and on a missing
``metric_for_best_model``, so those three arms raised before their first training step.
"""

from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path

import pytest

needs_trl = pytest.mark.skipif(find_spec("trl") is None, reason="trl not installed")
needs_torch = pytest.mark.skipif(find_spec("torch") is None, reason="torch not installed")
needs_peft = pytest.mark.skipif(find_spec("peft") is None, reason="peft not installed")

PAIRWISE = ("dpo", "rdpo", "ipo", "mixdpo", "grdpo")
REFERENCE_FREE_PAIRWISE = ("simpo", "alphapo")

#: Settings that define the shared recipe. An arm that differs in any of these is not
#: being compared on its objective alone.
SHARED = ("num_train_epochs", "learning_rate", "lr_scheduler_type", "warmup_steps",
          "weight_decay", "max_grad_norm", "logging_steps", "eval_strategy", "eval_steps",
          "save_strategy", "save_steps", "load_best_model_at_end",
          "metric_for_best_model", "greater_is_better", "optim", "seed", "data_seed",
          "gradient_checkpointing", "per_device_train_batch_size",
          "gradient_accumulation_steps")


def build(objective: str):
    """The trainer arguments this objective would actually be trained with."""
    from sentalign.config import ExperimentConfig
    from sentalign.recovery import BatchPlan
    from sentalign.train.po import build_cpo_config, build_dpo_config, build_kto_config

    cfg = ExperimentConfig(name="test", model="lfm-350m")
    cfg.train.objective = objective
    plan = BatchPlan(8, 4)
    if objective == "kto":
        return build_kto_config(cfg, Path("out"), plan, desirable_weight=1.0,
                                undesirable_weight=1.3)
    if objective in REFERENCE_FREE_PAIRWISE:
        return build_cpo_config(cfg, Path("out"), plan)
    return build_dpo_config(cfg, Path("out"), plan)


@needs_trl
@pytest.mark.parametrize("objective", (*PAIRWISE, *REFERENCE_FREE_PAIRWISE, "kto"))
def test_every_arm_selects_its_checkpoint_on_dev_macro_f1(objective):
    args = build(objective)
    assert args.eval_strategy == "steps", (
        f"{objective} never evaluates, so it keeps its last checkpoint while the DPO "
        "family keeps its best")
    assert args.metric_for_best_model == "eval_macro_f1"
    assert args.load_best_model_at_end is True
    assert args.greater_is_better is True
    assert args.eval_steps == args.save_steps, (
        "load_best_model_at_end needs a checkpoint saved at every evaluation")


@needs_trl
@pytest.mark.parametrize("objective", (*PAIRWISE, *REFERENCE_FREE_PAIRWISE, "kto"))
def test_early_stopping_callback_accepts_every_arm(objective):
    """The exact precondition that failed: this raised AssertionError before step 1."""
    from transformers import EarlyStoppingCallback

    EarlyStoppingCallback(3).on_train_begin(build(objective), None, None)


@needs_trl
@pytest.mark.parametrize("objective", (*REFERENCE_FREE_PAIRWISE, "kto"))
def test_the_arms_train_under_one_recipe(objective):
    """Everything but the loss is held fixed, so the comparison is of objectives."""
    reference = build("dpo")
    args = build(objective)
    differing = {field: (getattr(reference, field), getattr(args, field))
                 for field in SHARED
                 if getattr(reference, field) != getattr(args, field)}
    assert not differing, f"{objective} differs from dpo in {differing}"


@needs_trl
def test_the_shared_recipe_comes_from_the_configuration_not_the_library():
    """Pins the values to the config, so a TRL default change cannot move a run."""
    from sentalign.config import ExperimentConfig
    from sentalign.recovery import BatchPlan
    from sentalign.train.po import shared_training_fields

    cfg = ExperimentConfig(name="test", model="lfm-350m")
    fields = shared_training_fields(cfg, BatchPlan(8, 4))
    assert fields["lr_scheduler_type"] == cfg.train.lr_scheduler_type
    assert fields["weight_decay"] == cfg.train.weight_decay
    assert fields["eval_steps"] == cfg.train.eval_steps
    assert fields["warmup_steps"] > 0, "TRL's default is no warmup at all"
    assert fields["metric_for_best_model"] == "eval_macro_f1"


# --------------------------------------------------------------------------------------
# The distributional arms, which select without the trainer's help
# --------------------------------------------------------------------------------------

class StubSelector:
    """Returns a scripted macro-F1 for each evaluation."""

    def __init__(self, scores):
        self.scores = list(scores)
        self.steps: list[int] = []

    def evaluate(self, step: int) -> dict[str, float]:
        self.steps.append(step)
        return {"eval_macro_f1": self.scores[len(self.steps) - 1]}


def tiny_peft_model():
    import peft
    import torch.nn as nn

    model = nn.Module()
    model.layers = nn.ModuleList()
    layer = nn.Module()
    layer.self_attn = nn.Module()
    layer.self_attn.q_proj = nn.Linear(8, 8, bias=False)
    model.layers.append(layer)
    return peft.get_peft_model(model, peft.LoraConfig(
        r=4, lora_alpha=8, lora_dropout=0.0, bias="none",
        target_modules=["self_attn.q_proj"], task_type=None))


def bump(model, value: float):
    """Move the adapter, so a restored snapshot is distinguishable from the live one."""
    import torch

    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_B" in name:
                param.add_(value)


def lora_b(model):
    import torch

    return torch.cat([p.flatten() for n, p in model.named_parameters() if "lora_B" in n])


@needs_peft
def test_the_tracker_restores_the_best_adapter_not_the_last():
    from sentalign.train.driver import BestAdapterTracker

    model = tiny_peft_model()
    tracker = BestAdapterTracker(StubSelector([0.40, 0.61, 0.55]), every=150)

    bump(model, 1.0)
    tracker.consider(model, 150)
    bump(model, 1.0)
    tracker.consider(model, 300)          # the best
    best = lora_b(model).clone()
    bump(model, 1.0)
    tracker.consider(model, 446)          # worse, and it is what training ended on

    assert tracker.best_step == 300
    assert tracker.restore(model) is True
    assert lora_b(model).equal(best), "the saved adapter must be the selected one"


@needs_peft
def test_the_tracker_leaves_the_model_alone_when_the_last_step_is_best():
    from sentalign.train.driver import BestAdapterTracker

    model = tiny_peft_model()
    tracker = BestAdapterTracker(StubSelector([0.40, 0.52, 0.61]), every=150)
    for step in (150, 300):
        bump(model, 1.0)
        tracker.consider(model, step)
    bump(model, 1.0)
    tracker.consider(model, 446)
    final = lora_b(model).clone()

    assert tracker.best_step == 446
    assert tracker.restore(model) is False
    assert lora_b(model).equal(final)


@needs_peft
def test_the_tracker_evaluates_on_the_same_cadence_as_the_other_arms():
    """Same rule, same schedule: the selection is comparable across arms."""
    from sentalign.train.driver import BestAdapterTracker

    model = tiny_peft_model()
    selector = StubSelector([0.4, 0.5, 0.6, 0.7])
    tracker = BestAdapterTracker(selector, every=150)
    callback = tracker.as_transformers_callback(model)

    class State:
        global_step = 0

    state = State()
    for step in range(1, 451):
        state.global_step = step
        callback.on_step_end(None, state, None)
    assert selector.steps == [150, 300, 450]

    tracker.consider(model, 450)          # the end-of-training call must not double up
    assert selector.steps == [150, 300, 450]


# --------------------------------------------------------------------------------------
# The four ways the first grid attempt died
# --------------------------------------------------------------------------------------

@needs_trl
@pytest.mark.parametrize("objective", (*PAIRWISE, *REFERENCE_FREE_PAIRWISE, "kto"))
def test_the_prompt_budget_fits_inside_the_sequence_budget(objective):
    """CPOTrainer refuses max_prompt_length >= max_length, and TRL defaults it to 512.

    Every simpo and alphapo run died on this, 30 in all, after the EarlyStopping fix let
    them reach the config validation that had been hidden behind the earlier crash.
    """
    args = build(objective)
    assert args.max_prompt_length < args.max_length, (
        f"{objective}: prompt budget {args.max_prompt_length} does not fit inside "
        f"max_length {args.max_length}")


def test_group_ids_are_attached_to_both_splits_and_unknowns_are_dropped():
    """GR-DPO's collator refuses a batch with no group_id, on eval as well as on train."""
    from sentalign.train.po import with_group_ids

    index = {"r1_high": 0, "r1_low": 1}
    pairs = [{"prompt": "a", "group": "r1_high"}, {"prompt": "b", "group": "r1_low"},
             {"prompt": "c", "group": "r2_unseen"}]
    out = with_group_ids(pairs, index)

    assert [row["group_id"] for row in out] == [0, 1]
    assert len(out) == 2, "a dev group absent from training carries no weight to update"
    assert all(row["group_id"] < len(index) for row in out), "the collator range-checks"


def test_a_text_only_checkpoint_is_not_registered_as_image_text():
    """Qwen3.5 shares its model_type with a vision variant, so TRL prepared its rows with
    process_row and asked a tokenizer for `.tokenizer`. LFM2 is not in that mapping, which
    is why only Qwen failed, on all five pairwise arms."""
    from transformers.models.auto import modeling_auto

    from sentalign.modeling import treat_as_text_only

    mapping = modeling_auto.MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES
    assert "lfm2" not in mapping, "LFM2 never took this path"

    class _Config:
        model_type = "sentalign_fake_vl"

    class _Model:
        config = _Config()

    mapping["sentalign_fake_vl"] = "FakeForConditionalGeneration"
    try:
        info: dict = {}
        assert treat_as_text_only(_Model(), info) == "sentalign_fake_vl"
        assert "sentalign_fake_vl" not in mapping
        assert info["model_type_unregistered_from_image_text"] == "sentalign_fake_vl"
        # Idempotent, and silent about a model that was never registered.
        assert treat_as_text_only(_Model(), info) is None
    finally:
        mapping.pop("sentalign_fake_vl", None)


def test_the_dev_split_carries_what_the_custom_collators_need():
    """The training split had margin and group; the evaluation split did not."""
    import json
    from pathlib import Path

    build_dir = Path("data/build/pref/tau0.2/dev.jsonl")
    if not build_dir.exists():
        pytest.skip("built preference data not present")

    from sentalign.config import ExperimentConfig

    cfg = ExperimentConfig(name="test", model="lfm-350m")
    cfg.train.objective = "mixdpo"
    from sentalign.cli import _load_training_data

    data = _load_training_data(cfg)
    for split in ("pref_pairs", "pref_dev_pairs"):
        row = data[split][0]
        assert "margin" in row, f"{split} has no margin, so MixDPO cannot route"
        assert "group" in row, f"{split} has no group, so GR-DPO cannot weight"
        json.dumps(row)          # the row must survive the arrow conversion


def test_every_arm_trains_one_fixed_budget_by_default():
    """Early stopping was attached to the trainer-driven arms and to nothing else.

    The distributional arms select through BestAdapterTracker, which cannot stop early,
    so at n32000 DPO trained 600 of 2000 steps with 4 evaluation draws while CSPO trained
    all 2000 with 14. Best of 14 exceeds best of 4 drawn from the same distribution even
    when neither arm is better, which is a protocol advantage in the study built to
    compare them. The default is now one budget for everyone.
    """
    from sentalign.config import ExperimentConfig
    from sentalign.train.po import optional_early_stopping

    cfg = ExperimentConfig(name="test", model="lfm-350m")
    assert cfg.train.early_stopping_patience == 0
    assert optional_early_stopping(cfg) == [], (
        "the default must add no callback, or the arms diverge again")

    cfg.train.early_stopping_patience = 3        # still available when asked for
    callbacks = optional_early_stopping(cfg)
    assert len(callbacks) == 1
    assert type(callbacks[0]).__name__ == "EarlyStoppingCallback"


def test_the_main_grid_budget_is_too_short_for_early_stopping_to_have_fired():
    """Why the completed n8000 runs need no re-running: with patience 3 and 150-step
    evaluations, nothing can stop before 450 steps plus the step the best landed on, and
    an n8000 arm runs 446 to 500 steps in total."""
    from sentalign.config import ExperimentConfig

    cfg = ExperimentConfig(name="test", model="lfm-1.2b")
    patience, every = 3, cfg.train.eval_steps
    sft_steps = 7136 * 2 // 32           # the real n8000 supervised budget
    pref_steps = 8000 * 2 // 32
    assert every == 150
    for steps in (sft_steps, pref_steps):
        assert steps < every * (patience + 1), (
            f"{steps} steps leaves room for a patience-{patience} stop; the completed "
            "n8000 runs would then need re-running too")


def test_kto_scores_the_same_number_of_completions_as_the_pairwise_arms():
    """KTO was the one arm `n` never reached.

    The pairwise branch truncates to `max_train_pairs` and SFT reads a prebuilt
    train_n{n} file, but the KTO branch loaded kto/train.jsonl whole: 115,053 examples
    against 8,000 pairs, at every n. It cost 6.51 GPU-h where every other lfm-1.2b arm
    cost 0.36, and no comparison against it would have survived review.

    A DPO pair carries two completions, so the matched budget is 2n.
    """
    from pathlib import Path

    import pytest as _pytest

    if not Path("data/build/kto/train.jsonl").exists():
        _pytest.skip("built KTO data not present")

    from sentalign.cli import _load_training_data
    from sentalign.config import ExperimentConfig

    cfg = ExperimentConfig(name="test", model="lfm-350m")
    cfg.train.objective = "kto"
    cfg.data.train_subsample = 8000
    cfg.data.max_train_pairs = 8000

    data = _load_training_data(cfg)
    assert len(data["kto_examples"]) == 16_000, (
        f"KTO got {len(data['kto_examples'])} completions where the pairwise arms score "
        f"{2 * cfg.data.max_train_pairs}")

    cfg.data.max_train_pairs = 3000                  # the cap tracks n, it is not fixed
    assert len(_load_training_data(cfg)["kto_examples"]) == 6_000

    # Seeded, so the subset is reproducible from the manifest rather than incidental.
    cfg.train.seed = 13
    first = [e["prompt"] for e in _load_training_data(cfg)["kto_examples"][:20]]
    cfg.train.seed = 21
    other = [e["prompt"] for e in _load_training_data(cfg)["kto_examples"][:20]]
    cfg.train.seed = 13
    again = [e["prompt"] for e in _load_training_data(cfg)["kto_examples"][:20]]
    assert first == again and first != other


def test_the_loader_hands_every_distributional_arm_its_records(tmp_path):
    """The trainer reads ``data["cspo_records"]`` for the whole distributional family.
    The loader built that key from a hand-maintained ``("cspo", "sft_soft")`` tuple that
    forgot cspo_kl, so the arm crashed with ``KeyError: 'cspo_records'`` one layer past
    the dispatch fix. The loader and the trainer must agree on the set; this pins it."""
    from sentalign.cli import _load_training_data
    from sentalign.config import ExperimentConfig
    from sentalign.train.objectives import DISTRIBUTIONAL_OBJECTIVES

    build = tmp_path / "build"
    (build / "sft").mkdir(parents=True)
    (build / "cspo").mkdir(parents=True)
    n = 8_000
    dev = [{"text_id": "d0", "prompt": "x", "p_human": [0.2, 0.3, 0.5]}]
    train = [{"text_id": "t0", "prompt": "p", "p_human": [0.2, 0.3, 0.5]}]
    (build / "sft" / "dev.jsonl").write_text(
        "".join(__import__("json").dumps(r) + "\n" for r in dev))
    (build / "cspo" / f"train_n{n}.jsonl").write_text(
        "".join(__import__("json").dumps(r) + "\n" for r in train))

    for objective in DISTRIBUTIONAL_OBJECTIVES:
        cfg = ExperimentConfig(name="test", model="lfm-350m")
        cfg.data.build_dir = build
        cfg.data.train_subsample = n
        cfg.train.objective = objective
        data = _load_training_data(cfg)
        assert "cspo_records" in data, (
            f"{objective} is distributional but the loader gave it no cspo_records")
        assert data["cspo_records"] == train


def test_every_field_an_objective_needs_survives_the_column_whitelist():
    """The dataset is built from an explicit column list, so a field an objective needs
    is silently dropped unless that list names it. Polya trained on nothing at all for
    this reason: `votes` was written to disk, loaded into the records, and then discarded
    one line before the collator. Only its refusal to fall back to frequency matching
    made the failure visible rather than silent."""
    import inspect

    from sentalign.train import driver

    source = inspect.getsource(driver._train_cspo_family)
    whitelist = source.split("train_ds = Dataset.from_list")[0]
    for field in ("prompt", "p_human", "ref_logits", "votes"):
        assert f'"{field}"' in whitelist, (
            f"{field} is never added to the column whitelist, so it cannot reach the "
            f"collator no matter what the data builder writes")


def test_polya_records_carry_the_counts_and_not_only_the_frequencies():
    """p_human is a normalised frequency and loses M. Polya needs the counts, and the
    empirical-Bayes shrinkage needs the real annotator number: DynaSent has five per
    item and ChaosNLI a hundred."""
    from types import SimpleNamespace

    from sentalign.labels import TERNARY_PLUS_MIXED as SPACE
    from sentalign.train.cspo import build_cspo_records

    item = SimpleNamespace(
        text_id="t1", text="fine.", gold="positive", round="r1",
        agreement_band="4of5",
        votes={"negative": 0, "neutral": 1, "positive": 4, "mixed": 0})
    rec = build_cspo_records([item], SPACE)[0]
    assert rec["votes"] == [0.0, 1.0, 4.0, 0.0]
    assert rec["n_annotators"] == 5.0
    assert sum(rec["votes"]) == rec["n_annotators"]


def test_a_supervised_arm_is_never_warm_started_from_its_own_control():
    """polya is supervised and has no reference, yet the hard-coded gate gave it the SFT
    adapter as an initialisation while sft_soft, the arm it is compared against, trained
    from base. A whole extra stage of training is not an objective difference. Both the
    reference lookup and the model-init gate must route off the same registry."""
    import inspect

    from sentalign import cli
    from sentalign.train import driver
    from sentalign.train.objectives import DISTRIBUTIONAL_OBJECTIVES, FROM_BASE

    assert "polya" in FROM_BASE and "sft_soft" in FROM_BASE and "sft" in FROM_BASE
    # Reference-anchored arms must keep their warm start.
    for objective in ("cspo", "cspo_kl", "cspo_ada", "dpo"):
        assert objective not in FROM_BASE, objective

    for source in (inspect.getsource(cli.cmd_train),
                   inspect.getsource(driver.run_training)):
        assert 'not in ("sft", "sft_soft")' not in source, (
            "a hard-coded supervised tuple has come back; route off FROM_BASE")

    # Every purely supervised distributional arm trains from base.
    for objective in DISTRIBUTIONAL_OBJECTIVES:
        if objective in ("cspo", "cspo_kl", "cspo_ada"):
            continue                      # these consume a reference by design
        assert objective in FROM_BASE, objective


def test_zero_lambda_returns_the_unregularised_trainer_unchanged():
    """The regularised arm must nest the arm it repairs, or the comparison is between
    two different trainers rather than between lambda values."""
    from sentalign.train.po import add_distributional_regulariser

    class Fake:
        pass

    assert add_distributional_regulariser(Fake, 0.0, "soft", (1, 2), 256) is Fake
    wrapped = add_distributional_regulariser(Fake, 0.5, "soft", (1, 2), 256)
    assert wrapped is not Fake and issubclass(wrapped, Fake)


def test_the_target_collator_refuses_to_drop_the_annotator_distribution():
    """TRL rebuilds its batch dict, so an extra column vanishes silently. A regularised
    arm that lost its target would train as plain preference optimisation while being
    reported under the regularised name, which is the failure mode this project has hit
    six times."""
    import pytest

    from sentalign.train.po import TargetPreservingCollator

    collator = TargetPreservingCollator(lambda ex: {"input_ids": object()})
    with pytest.raises(RuntimeError, match="p_human"):
        collator([{"prompt": "x"}, {"prompt": "y"}])

    pytest.importorskip("torch")
    batch = collator([{"prompt": "a", "p_human": [0.5, 0.3, 0.2]},
                      {"prompt": "b", "p_human": [0.1, 0.1, 0.8]}])
    assert batch["p_human"].shape == (2, 3)
    assert batch["_prompts"] == ["a", "b"]


@needs_torch
def test_the_hard_ablation_differs_from_the_soft_regulariser():
    """The claim is that regularising toward the *annotator distribution* is what repairs
    confidence discrimination. If the majority-label variant were identical, the result
    would be ordinary SFT regularisation under a new name, so the two must differ."""
    import torch

    from sentalign.train.po import distributional_penalty

    class Out:
        def __init__(self, logits): self.logits = logits

    class Model(torch.nn.Module):
        def __init__(self): super().__init__(); self.p = torch.nn.Parameter(torch.zeros(1))
        def forward(self, **kw):
            b = kw["input_ids"].shape[0]
            base = torch.tensor([[0.2, 1.4, -0.3, 0.0]]).repeat(b, 1)
            return Out(base.unsqueeze(1).repeat(1, 3, 1) + self.p)

    class Tok:
        padding_side = "right"
        def __call__(self, text=None, **kw):
            n = len(text)
            return {"input_ids": torch.ones(n, 3, dtype=torch.long),
                    "attention_mask": torch.ones(n, 3, dtype=torch.long)}

    target = torch.tensor([[0.5, 0.3, 0.2, 0.0], [0.4, 0.4, 0.1, 0.1]])
    soft = distributional_penalty(Model(), Tok(), ["a", "b"], target, (0, 1, 2, 3), 32,
                                  "soft")
    hard = distributional_penalty(Model(), Tok(), ["a", "b"], target, (0, 1, 2, 3), 32,
                                  "hard")
    soft, hard = soft.detach().item(), hard.detach().item()
    assert abs(soft - hard) > 0.05, (soft, hard)


def test_the_regulariser_wrapper_matches_each_trl_trainer_signature():
    """DPOTrainer.get_batch_loss_metrics takes a fourth `train_eval` argument and
    KTOTrainer takes none. The wrapper hardcoded the DPO shape and KTO died with
    'takes 3 positional arguments but 4 were given' on the first step."""
    from sentalign.train.po import add_distributional_regulariser

    class KTOLike:
        processing_class = None
        def get_batch_loss_metrics(self, model, batch):
            return 1.0, {}

    class DPOLike:
        processing_class = None
        def get_batch_loss_metrics(self, model, batch, train_eval="train"):
            return 1.0, {"seen": train_eval}

    import sentalign.train.po as po
    calls = {}
    po_penalty = po.distributional_penalty
    po.distributional_penalty = lambda *a, **k: 0.0
    try:
        for cls, args in ((KTOLike, ()), (DPOLike, ("eval",))):
            wrapped = add_distributional_regulariser(cls, 0.5, "soft", (1, 2), 128)
            obj = wrapped.__new__(wrapped)
            obj._reg_lambda, obj._reg_kind = 0.5, "soft"
            batch = {"_prompts": ["a"], "p_human": [[0.5, 0.5]]}
            loss, metrics = wrapped.get_batch_loss_metrics(obj, None, batch, *args)
            calls[cls.__name__] = float(loss)
    finally:
        po.distributional_penalty = po_penalty
    assert calls == {"KTOLike": 1.0, "DPOLike": 1.0}, calls


def test_the_dev_split_carries_the_annotator_distribution_for_every_regularised_arm():
    """The regularised arms joined ``p_human`` onto the training split and nothing else.

    ``TargetPreservingCollator`` raises rather than defaulting to the unregularised loss,
    so the omission did not train a wrong objective: it killed the run at the first
    periodic evaluation, 150 steps in, after the GPU time was already spent. This is the
    same asymmetry as the margin and group columns pinned above, on a third column, so
    the check covers the unpaired arm and the pairwise arms together.
    """
    from pathlib import Path

    if not Path("data/build/cspo/dev.jsonl").exists():
        pytest.skip("built distributional data not present")

    from sentalign.cli import _load_training_data
    from sentalign.config import ExperimentConfig

    for objective, splits in (("kto", ("kto_examples", "kto_dev_examples")),
                              ("dpo", ("pref_pairs", "pref_dev_pairs"))):
        cfg = ExperimentConfig(name="test", model="lfm-350m")
        cfg.train.objective = objective
        cfg.train.pref_distributional_lambda = 0.5
        cfg.data.train_subsample = 8000
        cfg.data.max_train_pairs = 8000

        data = _load_training_data(cfg)
        widths = set()
        for split in splits:
            rows = data[split]
            assert rows, f"{objective}: {split} is empty under the regulariser"
            assert all("p_human" in r for r in rows), (
                f"{objective}: {split} lost p_human, so the collator kills the run at "
                f"the first evaluation step")
            widths |= {len(r["p_human"]) for r in rows}
        # One width across both splits: the penalty indexes a fixed verbaliser row set,
        # so a dev target of a different arity would be a silently wrong cross-entropy.
        assert len(widths) == 1, f"{objective}: mismatched target widths {widths}"


def test_the_unregularised_loader_leaves_the_dev_split_untouched():
    """lambda 0 must reproduce the plain arm exactly, including its columns, or the
    regularised and unregularised runs are not comparable."""
    from pathlib import Path

    if not Path("data/build/cspo/dev.jsonl").exists():
        pytest.skip("built distributional data not present")

    from sentalign.cli import _load_training_data
    from sentalign.config import ExperimentConfig

    cfg = ExperimentConfig(name="test", model="lfm-350m")
    cfg.train.objective = "dpo"
    cfg.data.train_subsample = 8000
    cfg.data.max_train_pairs = 8000
    data = _load_training_data(cfg)
    assert not any("p_human" in r for r in data["pref_dev_pairs"])


def test_the_regulariser_does_not_shrink_the_training_corpus():
    """The join dropped every example the size-n distributional file did not contain.

    ``p_human`` is a property of the item, so it is the same wherever the item appears,
    but the preference and KTO splits are drawn from the full pool while ``train_n8000``
    holds one 8k subsample of it. Joining against that file matched 1,988 of 8,000
    preference pairs and about a quarter of the KTO completions, so the regularised arm
    trained on a quarter of the data its unregularised twin receives: a protocol
    difference reported as an objective difference. The corpora must match exactly.
    """
    from pathlib import Path

    if not Path("data/build/cspo/dev.jsonl").exists():
        pytest.skip("built distributional data not present")

    from sentalign.cli import _load_training_data
    from sentalign.config import ExperimentConfig

    def sizes(objective, lam, splits):
        cfg = ExperimentConfig(name="test", model="lfm-350m")
        cfg.train.objective = objective
        cfg.train.pref_distributional_lambda = lam
        cfg.data.train_subsample = 8000
        cfg.data.max_train_pairs = 8000
        data = _load_training_data(cfg)
        return tuple(len(data[s]) for s in splits)

    for objective, splits in (("kto", ("kto_examples", "kto_dev_examples")),
                              ("dpo", ("pref_pairs", "pref_dev_pairs"))):
        plain = sizes(objective, 0.0, splits)
        regularised = sizes(objective, 0.5, splits)
        assert plain == regularised, (
            f"{objective}: the regulariser changed the corpus from {plain} to "
            f"{regularised}, so the two arms differ by more than their loss")


def test_trl_prunes_the_columns_the_regulariser_needs():
    """Verified against the installed TRL, not assumed: ``DPOTrainer`` replaces the
    signature columns with a fixed whitelist of tokenised fields, so an extra column
    survives only when ``remove_unused_columns`` is off. This is the mechanism the next
    test defends against, pinned here so a TRL upgrade that changes it is visible."""
    try:
        from trl.trainer.dpo_trainer import DPOTrainer
    except Exception as exc:                     # optional heavy deps, absent on the Mac
        pytest.skip(f"DPOTrainer is not importable here: {type(exc).__name__}")

    trainer = object.__new__(DPOTrainer)
    trainer._signature_columns = None
    DPOTrainer._set_signature_columns_if_needed(trainer)
    assert "p_human" not in trainer._signature_columns
    assert "prompt" not in trainer._signature_columns


def test_the_regulariser_keeps_its_columns_past_that_pruning():
    """``remove_unused_columns`` was cleared per objective, for mixdpo and grdpo only.

    KTOConfig forces the flag off inside TRL, so the unpaired arm trained while every
    pairwise regularised arm died at step zero with the collator's own error. The flag
    belongs to the regulariser, which is orthogonal to the objective, so the wrapper
    clears it for whatever trainer it wraps.
    """
    from types import SimpleNamespace

    from sentalign.train.po import add_distributional_regulariser

    class _Stub:
        def __init__(self, args, data_collator):
            self.args, self.data_collator = args, data_collator

    wrapped = add_distributional_regulariser(_Stub, 1.0, "soft", (0, 1, 2), 32)
    trainer = wrapped(args=SimpleNamespace(remove_unused_columns=True),
                      data_collator=lambda examples: {})
    assert trainer.args.remove_unused_columns is False, (
        "TRL will prune p_human and prompt before the collator runs")

    # lambda 0 must leave the trainer untouched, flag included, or the regularised and
    # unregularised arms differ by more than their loss.
    plain = add_distributional_regulariser(_Stub, 0.0, "soft", (0, 1, 2), 32)
    assert plain is _Stub


def test_a_finished_run_is_not_rerecorded_when_it_is_launched_again(tmp_path):
    """Relaunching a complete run must leave its manifest alone.

    ``Trainer.train(resume_from_checkpoint=...)`` on a checkpoint already at ``max_steps``
    returns without stepping, reporting the checkpoint's own ``global_step`` and a
    ``training_loss`` of 0.0. The manifest assembled from that looks finished, so a rerun
    of an already finished arm replaced its provenance with a record carrying no loss, no
    selection history, and seconds of wallclock. ipo-dreg1.0 seed 13 lost the
    checkpoint-selection record it is compared on that way.
    """
    from sentalign.recovery import AlreadyComplete, assert_resume_made_progress

    run_dir = tmp_path / "main__lfm-1.2b__ipo-dreg1.0__eps0.0__tau0.2__n8000__s13"
    run_dir.mkdir()
    manifest = run_dir / "manifest.json"
    real = '{"steps": 500, "train_loss": 12.3728, "selection_history": [1, 2, 3, 4]}'
    manifest.write_text(real)

    # The observed failure: resumed at step 500, returned step 500, zero steps trained.
    with pytest.raises(AlreadyComplete):
        assert_resume_made_progress(500, run_dir / "checkpoint-500", run_dir)
    assert manifest.read_text() == real, "the finished run's record must survive"

    # A genuine continuation is not blocked, and neither is a fresh run.
    assert_resume_made_progress(1000, run_dir / "checkpoint-500", run_dir) is None
    assert_resume_made_progress(500, None, run_dir) is None


def test_the_save_path_checks_that_before_it_writes_anything():
    """The guard must run before the manifest is built, and once for every objective.

    Placing it per objective is the asymmetry that has cost this project nine runs, so it
    sits in the one save stage every arm reaches, ahead of the first write.
    """
    from pathlib import Path as _Path

    source = _Path("src/sentalign/train/driver.py").read_text()
    guard = source.index("assert_resume_made_progress(result.get(\"steps\")")
    assert guard < source.index('logger.stage("save")'), (
        "the guard must run before the save stage opens")
    assert guard < source.index("save_run(model, tokenizer, run_dir, manifest)")
    assert source.count("assert_resume_made_progress(result.get") == 1, (
        "one call on the shared path, not one per objective")


@pytest.mark.parametrize("state", ["training_died", "no_op_resume"])
def test_evaluation_refuses_an_adapter_the_protocol_did_not_select(tmp_path, monkeypatch,
                                                                   state):
    """Scoring must stop before any configuration or weights are loaded.

    ipo-dreg1.0 seed 13 died before its save stage, and evaluation reached Unsloth, which
    failed on a missing ``model_type`` and buried the training failure. Seed 21 had been
    relaunched after finishing, resumed, trained no step, and was then scored on an
    adapter no checkpoint selection had chosen.
    """
    import json

    import sentalign.config as config
    from sentalign.evaluate.run_eval import evaluate_run

    def loaded(*args, **kwargs):
        raise AssertionError("evaluation went past the guard and loaded the run")

    monkeypatch.setattr(config.ExperimentConfig, "load", staticmethod(loaded))
    run_dir = tmp_path / "run"
    (run_dir / "eval").mkdir(parents=True)
    (run_dir / "config.json").write_text("{}")
    if state == "no_op_resume":
        # The record the relaunch actually wrote, fields as found on disk.
        (run_dir / "manifest.json").write_text(json.dumps({
            "steps": 500, "train_loss": 0.0, "wallclock_s": 15.8, "selection_history": [],
            "resumed_from": "runs/x/checkpoint-500"}))

    assert evaluate_run(run_dir, tmp_path / "build") == 1
    assert not (run_dir / "eval" / "metrics.json").exists()


def test_evaluation_still_proceeds_for_a_run_that_selected_its_checkpoint(tmp_path,
                                                                           monkeypatch):
    """The guard must not block a normal run, or every arm silently stops being scored."""
    import json

    import sentalign.config as config
    from sentalign.evaluate.run_eval import evaluate_run

    class Reached(Exception):
        pass

    def loaded(*args, **kwargs):
        raise Reached

    monkeypatch.setattr(config.ExperimentConfig, "load", staticmethod(loaded))
    run_dir = tmp_path / "run"
    (run_dir / "eval").mkdir(parents=True)
    (run_dir / "config.json").write_text("{}")
    (run_dir / "manifest.json").write_text(json.dumps({
        "steps": 1000, "train_loss": 1.6606, "resumed_from": None,
        "selection_history": [{"step": 150, "eval_macro_f1": 0.56}]}))

    with pytest.raises(Reached):
        evaluate_run(run_dir, tmp_path / "build")


def test_every_language_model_declares_its_vram_floor_and_preflight_reads_it():
    """The floor was ``20.0 if model != "qwen-2b" else 22.0``, so a newly registered model
    inherited 20 GB whatever its size. It now lives on the registry entry, and indexing
    rather than ``.get`` makes a model registered without one fail at preflight."""
    from pathlib import Path as _Path

    from sentalign.modeling import MODEL_REGISTRY

    for key, entry in MODEL_REGISTRY.items():
        if entry.get("encoder"):
            continue
        floor, peak = entry.get("min_vram_gb"), entry.get("measured_peak_gb")
        assert isinstance(floor, float), f"{key} has no VRAM floor"
        assert isinstance(peak, float), f"{key} has no recorded peak to justify its floor"
        # The gate admits at 0.8 of the floor, so that threshold has to clear the peak the
        # model actually reached, by a gigabyte. The margin is absolute rather than a
        # percentage because the largest model peaks at 14.5 GB on a 24 GB card, where a
        # percentage margin would demand a floor the card cannot express.
        assert 0.8 * floor >= peak + 1.0, f"{key}: floor {floor} too close to peak {peak}"
        # And not far above it either. A floor of 20 GB for a model that never exceeded
        # 4.7 demanded nearly the whole card, so it refused any run that would have shared
        # the GPU, reporting only that memory was short.
        assert 0.8 * floor <= peak + 4.0, \
            f"{key}: floor {floor} demands {0.8 * floor - peak:.1f} GB beyond its own peak"
        assert floor <= 20.0, f"{key}: a floor of {floor} leaves nothing for a companion run"
    # Parameter count does not order VRAM here: qwen-2b peaks above smollm3-3b, so the
    # floors follow the measurements rather than the model sizes.
    assert MODEL_REGISTRY["qwen-2b"]["min_vram_gb"] > MODEL_REGISTRY["smollm3-3b"]["min_vram_gb"]

    source = _Path("src/sentalign/train/driver.py").read_text()
    assert 'MODEL_REGISTRY[cfg.model]["min_vram_gb"]' in source
    assert '!= "qwen-2b"' not in source, "no per-model branch for the VRAM floor"

