"""Tests for the data pipeline, anchored on the defects in AUDIT.md.

Each test names the failure it prevents. The three S1 defects are all data-construction
bugs that produced plausible-looking numbers, so a test that only checks happy paths
would not have caught any of them.
"""

from __future__ import annotations

import pytest

from sentalign.data.dynasent import Item
from sentalign.data.integrity import (LeakageError, MinHashIndex, assert_no_resampling,
                                      audit_splits, normalise_text, stratified_holdout)
from sentalign.data.preferences import (apply_flips, build_kto_examples, build_pairs,
                                        build_sft_examples, flip_indices, inject_noise)
from sentalign.labels import NO_MAJORITY, TERNARY, TERNARY_PLUS_MIXED


def make_item(text_id: str, votes: dict[str, int], text: str = "some text") -> Item:
    full = {"positive": 0, "negative": 0, "neutral": 0, "mixed": 0, **votes}
    top = max(full.values())
    gold = (next(k for k, v in full.items() if v == top) if top >= 3 else NO_MAJORITY)
    return Item(text_id=text_id, text=text, votes=full, gold=gold,
                round="r1", source="r1_train")


# --------------------------------------------------------------------------------------
# S1-1: resampling with replacement before splitting
# --------------------------------------------------------------------------------------

def test_resampling_is_rejected():
    """The exact v1 failure: 139 unique texts inflated into a 120,000-row class."""
    rows = [(f"id{i}", f"tweet {i % 139}") for i in range(120_000)]
    with pytest.raises(LeakageError, match="S1-1"):
        assert_no_resampling(rows, "neutral")


def test_natural_duplicate_rate_is_tolerated():
    """Real corpora repeat the odd sentence; the guard must not fire on 0.5%."""
    rows = [(f"id{i}", f"unique sentence number {i}") for i in range(1000)]
    rows += [(f"dup{i}", "great food") for i in range(5)]
    assert_no_resampling(rows, "train")     # must not raise


def test_audit_refuses_leaking_splits():
    with pytest.raises(LeakageError):
        audit_splits({"train": [("a", "the food was great")],
                      "test": [("b", "The  FOOD was great!!")]})


def test_audit_direction_is_meaningful():
    """train -> test asks how much of test was seen in training. The reverse differs."""
    report = audit_splits(
        {"train": [(f"t{i}", f"sentence {i}") for i in range(100)],
         "test": [("x", "sentence 3"), ("y", "brand new sentence")]},
        raise_on_leak=False, check_near=False)
    forward = next(o for o in report.overlaps if (o.split_a, o.split_b) == ("train", "test"))
    backward = next(o for o in report.overlaps if (o.split_a, o.split_b) == ("test", "train"))
    assert forward.exact_rate == pytest.approx(0.5)      # 1 of 2 test items
    assert backward.exact_rate == pytest.approx(0.01)    # 1 of 100 train items


def test_near_duplicate_detection_catches_minimal_edits():
    idx = MinHashIndex()
    idx.add("orig", "The service here was slow but the food more than made up for it")
    hits = idx.query("The service here was slow, but the food more than made up for it!",
                     threshold=0.8)
    assert hits and hits[0][0] == "orig"


def test_normalisation_ignores_case_and_punctuation_only():
    assert normalise_text("Great Food!!!") == normalise_text("great  food")
    assert normalise_text("great food") != normalise_text("bad food")


def test_stratified_holdout_is_deterministic_and_balanced():
    keys = [f"i{n}" for n in range(600)]
    strata = ["a"] * 400 + ["b"] * 200
    first = stratified_holdout(keys, strata, 0.1, seed=3)
    assert first == stratified_holdout(keys, strata, 0.1, seed=3)
    assert first != stratified_holdout(keys, strata, 0.1, seed=4)
    by_stratum = {s: sum(1 for k in first if strata[keys.index(k)] == s) for s in "ab"}
    assert by_stratum == {"a": 40, "b": 20}


# --------------------------------------------------------------------------------------
# S1-3: the preference margin must vary
# --------------------------------------------------------------------------------------

def test_margin_varies_across_items():
    """v1's score gap was constant at 1, which collapsed rDPO onto DPO."""
    items = [
        make_item("a", {"positive": 5}),                              # Delta = 1.0
        make_item("b", {"positive": 4, "negative": 1}),               # Delta = 0.6
        make_item("c", {"positive": 3, "negative": 2}),               # Delta = 0.2
        make_item("d", {"positive": 2, "negative": 2, "mixed": 1}),   # no majority
    ]
    pairs, stats = build_pairs(items, TERNARY_PLUS_MIXED, tau=0.2)
    margins = {round(p.margin, 1) for p in pairs}
    assert len(margins) >= 3, f"margin must vary, got {margins}"
    assert 1.0 in margins and 0.2 in margins


def test_margin_equals_the_human_vote_gap():
    item = make_item("x", {"positive": 4, "negative": 1})
    pairs, _ = build_pairs([item], TERNARY, tau=0.2)
    winner = next(p for p in pairs
                  if (p.chosen_label, p.rejected_label) == ("positive", "negative"))
    assert winner.margin == pytest.approx(0.6)        # (4 - 1) / 5
    assert (winner.n_chosen, winner.n_rejected) == (4, 1)


def test_tau_filters_low_margin_pairs():
    """Each item yields one pair per ordered label pair, so tau is tested per pair.

    For votes {positive: 3, negative: 2, neutral: 0} the three pairs have margins
    positive>negative = 0.2, negative>neutral = 0.4, positive>neutral = 0.6.
    """
    items = [make_item("c", {"positive": 3, "negative": 2})]

    def margins(tau):
        pairs, _ = build_pairs(items, TERNARY, tau=tau)
        return {(p.chosen_label, p.rejected_label): round(p.margin, 1) for p in pairs}

    assert margins(0.2) == {("positive", "negative"): 0.2,
                            ("positive", "neutral"): 0.6,
                            ("negative", "neutral"): 0.4}
    assert margins(0.4) == {("positive", "neutral"): 0.6, ("negative", "neutral"): 0.4}
    assert margins(0.6) == {("positive", "neutral"): 0.6}


def test_unrepresentable_tau_is_rejected():
    with pytest.raises(ValueError, match="not representable"):
        build_pairs([make_item("a", {"positive": 5})], TERNARY, tau=0.3)


def test_no_majority_items_reach_preferences_but_not_sft():
    """The asymmetry that is the mechanism behind H2."""
    items = [make_item("d", {"positive": 2, "negative": 2, "mixed": 1})]
    assert items[0].gold == NO_MAJORITY
    assert build_sft_examples(items, TERNARY_PLUS_MIXED) == []
    pairs, _ = build_pairs(items, TERNARY_PLUS_MIXED, tau=0.2)
    assert pairs, "a no-majority item still expresses relative preferences"


def test_pairs_are_ordered_by_votes():
    items = [make_item("a", {"negative": 4, "positive": 1})]
    pairs, _ = build_pairs(items, TERNARY, tau=0.2)
    for p in pairs:
        assert p.n_chosen > p.n_rejected


# --------------------------------------------------------------------------------------
# Noise ladder
# --------------------------------------------------------------------------------------

def test_noise_rate_is_exact_within_each_margin_stratum():
    items = [make_item(f"i{n}", {"positive": 4, "negative": 1}) for n in range(50)]
    items += [make_item(f"j{n}", {"positive": 5}) for n in range(50)]
    pairs, _ = build_pairs(items, TERNARY, tau=0.2)
    noisy, stats = inject_noise(pairs, 0.2, seed=1)
    assert stats.n_flipped == pytest.approx(0.2 * len(pairs), abs=2)

    by_margin: dict[float, list[bool]] = {}
    for p in noisy:
        by_margin.setdefault(round(abs(p.margin), 1), []).append(p.flipped)
    for margin, flags in by_margin.items():
        assert sum(flags) / len(flags) == pytest.approx(0.2, abs=0.02), margin


def test_flipped_pairs_retain_their_true_margin():
    """A corrupted set knows its own ground truth: that is what makes H4 measurable."""
    items = [make_item(f"i{n}", {"positive": 4, "negative": 1}) for n in range(20)]
    pairs, _ = build_pairs(items, TERNARY, tau=0.2)
    original = {(p.text_id, p.chosen_label, p.rejected_label): p.margin for p in pairs}
    noisy, _ = inject_noise(pairs, 0.5, seed=2)

    n_flipped = 0
    for p in noisy:
        if not p.flipped:
            continue
        n_flipped += 1
        assert p.margin < 0, "a flipped pair carries a negative margin"
        assert p.true_margin == pytest.approx(-p.margin)
        # The reversed pair must correspond to a real pair of the source set.
        assert original[(p.text_id, p.rejected_label, p.chosen_label)] == \
            pytest.approx(p.true_margin)
    assert n_flipped > 0


def test_flip_indices_round_trip():
    items = [make_item(f"i{n}", {"positive": 4, "negative": 1}) for n in range(40)]
    pairs, _ = build_pairs(items, TERNARY, tau=0.2)
    idx = flip_indices(pairs, 0.25, seed=5)
    rebuilt = apply_flips(pairs, idx)
    direct, _ = inject_noise(pairs, 0.25, seed=5)
    assert [p.chosen_label for p in rebuilt] == [p.chosen_label for p in direct]
    assert sum(p.flipped for p in rebuilt) == len(idx)


def test_epsilon_above_half_is_rejected():
    pairs, _ = build_pairs([make_item("a", {"positive": 5})], TERNARY, tau=0.2)
    with pytest.raises(ValueError):
        inject_noise(pairs, 0.6)


# --------------------------------------------------------------------------------------
# KTO
# --------------------------------------------------------------------------------------

def test_kto_is_unpaired_with_a_discarded_middle_band():
    items = [make_item("a", {"positive": 3, "neutral": 2})]
    examples = build_kto_examples(items, TERNARY, desirable_at=0.6, undesirable_at=0.2)
    by_label = {e.label: e for e in examples}
    assert by_label["positive"].desirable is True         # 3/5 = 0.6
    assert by_label["negative"].desirable is False        # 0/5 = 0.0
    assert "neutral" not in by_label                      # 2/5 = 0.4, in the dead band


def test_kto_bands_must_be_ordered():
    with pytest.raises(ValueError):
        build_kto_examples([], TERNARY, desirable_at=0.2, undesirable_at=0.6)


# --------------------------------------------------------------------------------------
# Item mechanics
# --------------------------------------------------------------------------------------

def test_distribution_vector_renormalises_within_the_label_space():
    item = make_item("a", {"positive": 3, "mixed": 2})
    four = item.distribution_vector(TERNARY_PLUS_MIXED)
    three = item.distribution_vector(TERNARY)
    assert four[TERNARY_PLUS_MIXED.index("positive")] == pytest.approx(0.6)
    assert four[TERNARY_PLUS_MIXED.index("mixed")] == pytest.approx(0.4)
    # The three-way space has no 'mixed', so the mass renormalises over what remains.
    assert three[TERNARY.index("positive")] == pytest.approx(1.0)
    assert sum(three) == pytest.approx(1.0)


def test_agreement_band_and_entropy():
    unanimous = make_item("a", {"positive": 5})
    split = make_item("b", {"positive": 3, "negative": 2})
    assert unanimous.agreement_band == "5of5" and unanimous.entropy == pytest.approx(0.0)
    assert split.agreement_band == "3of5" and split.entropy > 0.6
