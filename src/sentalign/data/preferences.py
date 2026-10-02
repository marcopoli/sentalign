"""HDP-Sent: preference pairs whose margin is measured from human annotators.

This is contribution C1 (RESEARCH_DESIGN.md §3) and the direct repair of AUDIT.md S1-3.

The v1 pipeline sampled K completions from a template-tuned SFT policy and scored them
with ``s = s_label + s_format + s_length``. Because the policy always emitted the
template, ``s_format`` was always 1 and ``s_length`` always 0, so ``s in {1, 2}`` and the
score gap ``Δs`` was **1 for every pair in the dataset**. The RDPO weight
``w = min(1, max(0, Δs/2))`` therefore evaluated to the constant 0.5, making "robust DPO"
identical to DPO with the loss halved. A preference set with no margin variation cannot
distinguish margin-sensitive objectives, which was most of what v1 set out to compare.

Here the margin comes from the annotators. For an item with vote counts ``n(y)`` over the
four DynaSent options and ``N = 5``:

    p(y) = n(y) / N
    emit  y_w > y_l   iff   p(y_w) - p(y_l) >= tau
    margin  Δ = p(y_w) - p(y_l)  in  {0.2, 0.4, 0.6, 0.8, 1.0}

Δ is a real quantity with real spread, and it is what the margin- and weight-consuming
objectives (SimPO's γ, rDPO's δ, cDPO's ε, Dr. DPO's β') are supposed to consume.

Two further products come out of the same annotator counts:

*   **The noise ladder** (C2). Flip a pair's direction independently with known
    probability ε. Robust-PO papers derive guarantees under exactly this Bernoulli model
    and then validate on chat data where ε is unknown; here it is set by us.
*   **Unpaired KTO data.** KTO (Ethayarajh et al., 2024) consumes independently labelled
    desirable/undesirable completions, not pairs. v1 described KTO as a pairwise
    odds-ratio method (that is ORPO) and wrote a pairwise loss for it (AUDIT S1-4).
"""

from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from ..labels import NO_MAJORITY, LabelSpace, build_prompt, build_target
from .dynasent import Item

#: Only these margins are representable with 5 annotators, which is why the threshold is
#: expressed in votes internally: floating-point comparison against 0.2 is not exact.
REPRESENTABLE_MARGINS = (0.2, 0.4, 0.6, 0.8, 1.0)


@dataclass(frozen=True)
class PreferencePair:
    """One preference triple with its provenance.

    ``margin`` is the human quantity. ``flipped`` records whether the noise ladder
    reversed this pair, so a corrupted set still knows its own ground truth: which is
    what makes the robustness comparison in H4 a measurement rather than an assumption.
    """

    text_id: str
    prompt: str
    chosen: str
    rejected: str
    chosen_label: str
    rejected_label: str
    margin: float
    n_chosen: int
    n_rejected: int
    agreement_band: str
    group: str
    source: str
    flipped: bool = False
    true_margin: float | None = None

    def as_trl(self) -> dict:
        """The record shape TRL's ``DPOTrainer`` family expects."""
        return {"prompt": self.prompt, "chosen": self.chosen, "rejected": self.rejected}

    def as_record(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class KTOExample:
    """One unpaired completion with a desirability flag, as KTO actually consumes."""

    text_id: str
    prompt: str
    completion: str
    label: str
    desirable: bool
    p_human: float
    agreement_band: str
    group: str

    def as_trl(self) -> dict:
        return {"prompt": self.prompt, "completion": self.completion,
                "label": self.desirable}


@dataclass
class PreferenceStats:
    n_items: int = 0
    n_items_used: int = 0
    n_pairs: int = 0
    margin_hist: Counter = field(default_factory=Counter)
    band_hist: Counter = field(default_factory=Counter)
    group_hist: Counter = field(default_factory=Counter)
    n_flipped: int = 0

    def summary(self) -> str:
        lines = [
            f"items seen {self.n_items:,}, contributing {self.n_items_used:,} "
            f"({100 * self.n_items_used / max(self.n_items, 1):.1f}%)",
            f"pairs {self.n_pairs:,}"
            + (f", flipped {self.n_flipped:,} ({100 * self.n_flipped / max(self.n_pairs,1):.1f}%)"
               if self.n_flipped else ""),
            "margin  " + "  ".join(f"Δ={m}: {c:,} ({100*c/max(self.n_pairs,1):.1f}%)"
                                   for m, c in sorted(self.margin_hist.items())),
            "band    " + "  ".join(f"{b}: {c:,}" for b, c in sorted(self.band_hist.items())),
        ]
        return "\n".join("  " + line for line in lines)


def assign_group(item: Item) -> str:
    """Group label for GR-DPO.

    Groups are ``round × agreement band``. v1 grouped by input length, which is a proxy
    for nothing in particular; the agreement band is the axis the paper's worst-group
    claim is actually about, and the round separates naturally occurring text from
    adversarially written text.
    """
    return f"{item.round}:{item.agreement_band}"


def build_pairs(
    items: Iterable[Item],
    space: LabelSpace,
    *,
    tau: float = 0.2,
    max_pairs_per_item: int | None = None,
    include_no_majority: bool = True,
    seed: int = 0,
    n_annotators: int | None = 5,
) -> tuple[list[PreferencePair], PreferenceStats]:
    """Construct HDP pairs from annotator vote counts.

    ``n_annotators`` fixes the denominator of the margin. DynaSent has exactly five
    judgements per item, so the default keeps its builds byte-identical and keeps the
    threshold in whole votes. A corpus whose items carry a varying number of raters
    (GoEmotions has three, four or five) passes ``None``, and each item's margin is then
    its vote gap over its own rater count: dividing a one-vote gap among three raters by
    five would record 0.2 for what is a margin of one third.

    ``tau`` is the minimum human margin. τ=0.2 (a one-vote gap) keeps the ambiguous
    items that carry the H3 signal; τ=0.6 keeps only near-consensus pairs and is the
    ablation that shows what a conventional "clean preferences" filter throws away.

    ``include_no_majority`` controls whether items with no 3/5 majority contribute. They
    have no gold label at all, so SFT cannot use them: but they still express a
    *relative* preference (3 votes for positive beats 1 for negative), which is precisely
    the supervision preference optimization can consume and cross-entropy cannot. That
    asymmetry is the mechanism behind H2, so these items are kept by default.
    """
    rng = random.Random(seed)
    if n_annotators is not None:
        min_votes = round(tau * n_annotators)
        if abs(min_votes - tau * n_annotators) > 1e-9:
            raise ValueError(
                f"tau={tau} is not representable with {n_annotators} annotators; "
                f"use one of {REPRESENTABLE_MARGINS}"
            )

    pairs: list[PreferencePair] = []
    stats = PreferenceStats()

    for item in items:
        stats.n_items += 1
        if item.gold == NO_MAJORITY and not include_no_majority:
            continue

        votes = {y: item.votes.get(y, 0) for y in space.labels}
        if sum(votes.values()) == 0:
            continue

        n_ann = n_annotators if n_annotators is not None else sum(votes.values())
        prompt = build_prompt(item.text, space)
        group = assign_group(item)
        candidates: list[PreferencePair] = []
        ordered = sorted(space.labels, key=lambda y: -votes[y])
        for i, win in enumerate(ordered):
            for lose in ordered[i + 1:]:
                gap = votes[win] - votes[lose]
                if n_annotators is not None:
                    if gap < min_votes:
                        continue
                # Per-item denominators: at least one vote, and a margin of at least tau,
                # compared with a tolerance because one third is not exact in floating point.
                elif gap < 1 or gap / n_ann < tau - 1e-9:
                    continue
                candidates.append(PreferencePair(
                    text_id=item.text_id,
                    prompt=prompt,
                    chosen=build_target(win, space),
                    rejected=build_target(lose, space),
                    chosen_label=win,
                    rejected_label=lose,
                    margin=gap / n_ann,
                    n_chosen=votes[win],
                    n_rejected=votes[lose],
                    agreement_band=item.agreement_band,
                    group=group,
                    source=item.source,
                ))

        if not candidates:
            continue
        if max_pairs_per_item is not None and len(candidates) > max_pairs_per_item:
            candidates = rng.sample(candidates, max_pairs_per_item)

        stats.n_items_used += 1
        for pair in candidates:
            stats.n_pairs += 1
            stats.margin_hist[round(pair.margin, 1)] += 1
            stats.band_hist[pair.agreement_band] += 1
            stats.group_hist[pair.group] += 1
        pairs.extend(candidates)

    return pairs, stats


def inject_noise(
    pairs: Sequence[PreferencePair],
    epsilon: float,
    *,
    seed: int = 0,
    stratify_by_margin: bool = True,
) -> tuple[list[PreferencePair], PreferenceStats]:
    """Flip each pair's direction with probability ``epsilon`` (contribution C2).

    With ``stratify_by_margin`` the flip count is fixed exactly within each margin
    stratum rather than being Binomial. That removes the sampling variance in ε itself,
    so a difference between two objectives at the same nominal ε is attributable to the
    objective rather than to how many flips each happened to draw: which matters
    because the effects H4 looks for are small.

    A flipped pair keeps ``true_margin``, so an oracle-margin ablation stays available.
    """
    if not 0.0 <= epsilon <= 0.5:
        raise ValueError("epsilon must lie in [0, 0.5]; above 0.5 the labels are inverted")
    rng = random.Random(seed)
    out: list[PreferencePair] = []
    stats = PreferenceStats(n_pairs=len(pairs))

    if stratify_by_margin:
        strata: dict[float, list[int]] = {}
        for idx, pair in enumerate(pairs):
            strata.setdefault(round(pair.margin, 1), []).append(idx)
        flip_idx: set[int] = set()
        for margin, members in sorted(strata.items()):
            k = int(round(epsilon * len(members)))
            flip_idx.update(rng.sample(members, k))
    else:
        flip_idx = {i for i in range(len(pairs)) if rng.random() < epsilon}

    for idx, pair in enumerate(pairs):
        if idx in flip_idx:
            out.append(PreferencePair(
                text_id=pair.text_id, prompt=pair.prompt,
                chosen=pair.rejected, rejected=pair.chosen,
                chosen_label=pair.rejected_label, rejected_label=pair.chosen_label,
                margin=-pair.margin,
                n_chosen=pair.n_rejected, n_rejected=pair.n_chosen,
                agreement_band=pair.agreement_band, group=pair.group,
                source=pair.source, flipped=True, true_margin=pair.margin,
            ))
            stats.n_flipped += 1
        else:
            out.append(pair)
        stats.margin_hist[round(out[-1].margin, 1)] += 1
        stats.band_hist[pair.agreement_band] += 1
        stats.group_hist[pair.group] += 1
    stats.n_items = len({p.text_id for p in pairs})
    stats.n_items_used = stats.n_items
    return out, stats


def build_kto_examples(
    items: Iterable[Item],
    space: LabelSpace,
    *,
    desirable_at: float = 0.6,
    undesirable_at: float = 0.2,
) -> list[KTOExample]:
    """Unpaired desirable/undesirable completions for KTO.

    A completion is desirable when at least ``desirable_at`` of annotators chose that
    label and undesirable when at most ``undesirable_at`` did; the band between is
    dropped rather than forced to a side. This mirrors KTO's own framing: independent
    judgements of individual outputs: instead of decomposing pairs, which would
    reintroduce the paired assumption KTO exists to avoid.
    """
    if not 0.0 <= undesirable_at < desirable_at <= 1.0:
        raise ValueError("require 0 <= undesirable_at < desirable_at <= 1")
    out: list[KTOExample] = []
    for item in items:
        total = sum(item.votes.get(y, 0) for y in space.labels)
        if total == 0:
            continue
        prompt = build_prompt(item.text, space)
        group = assign_group(item)
        for label in space.labels:
            p = item.votes.get(label, 0) / total
            if p >= desirable_at:
                desirable = True
            elif p <= undesirable_at:
                desirable = False
            else:
                continue
            out.append(KTOExample(
                text_id=item.text_id, prompt=prompt,
                completion=build_target(label, space), label=label,
                desirable=desirable, p_human=p,
                agreement_band=item.agreement_band, group=group,
            ))
    return out


def build_sft_examples(
    items: Iterable[Item],
    space: LabelSpace,
    *,
    require_majority: bool = True,
) -> list[dict]:
    """Instruction-tuning examples.

    Items without a majority label are skipped by default: cross-entropy needs a point
    target and inventing one for a genuinely ambiguous item is the modelling error the
    paper is arguing against. Those items reappear in the preference set, which is the
    contrast H2 tests.
    """
    out = []
    for item in items:
        if item.gold == NO_MAJORITY:
            if require_majority:
                continue
            gold = max(space.labels, key=lambda y: item.votes.get(y, 0))
        else:
            if not space.contains(item.gold):
                continue
            gold = space.normalise(item.gold)
        prompt = build_prompt(item.text, space)
        out.append({
            "text_id": item.text_id,
            "prompt": prompt,
            "completion": build_target(gold, space),
            "text": prompt + build_target(gold, space),
            "label": gold,
            "agreement_band": item.agreement_band,
            "group": assign_group(item),
            "p_human": item.distribution_vector(space),
        })
    return out


def flip_indices(pairs: Sequence[PreferencePair], epsilon: float, *, seed: int = 0,
                 stratify_by_margin: bool = True) -> list[int]:
    """Which pair indices the noise ladder flips at level ``epsilon``.

    Storing the ladder as index sets rather than as full copies of the corpus keeps the
    released artifact small and makes the corruption auditable: a reader can see exactly
    which preferences were inverted, and at which human margin, without diffing files.
    """
    noisy, _ = inject_noise(pairs, epsilon, seed=seed,
                           stratify_by_margin=stratify_by_margin)
    return [i for i, p in enumerate(noisy) if p.flipped]


def apply_flips(pairs: Sequence[PreferencePair],
                indices: Iterable[int]) -> list[PreferencePair]:
    """Reconstruct a noisy preference set from the base set and a flip-index list."""
    flip = set(indices)
    out: list[PreferencePair] = []
    for i, pair in enumerate(pairs):
        if i in flip:
            out.append(PreferencePair(
                text_id=pair.text_id, prompt=pair.prompt,
                chosen=pair.rejected, rejected=pair.chosen,
                chosen_label=pair.rejected_label, rejected_label=pair.chosen_label,
                margin=-pair.margin, n_chosen=pair.n_rejected, n_rejected=pair.n_chosen,
                agreement_band=pair.agreement_band, group=pair.group, source=pair.source,
                flipped=True, true_margin=pair.margin))
        else:
            out.append(pair)
    return out


def load_pairs(path: Path) -> list[PreferencePair]:
    """Read a preference file back into ``PreferencePair`` objects."""
    out = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                out.append(PreferencePair(**json.loads(line)))
    return out


def write_jsonl(records: Iterable[dict], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n
