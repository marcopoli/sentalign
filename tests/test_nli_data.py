"""The NLI source contract.

The point of the NLI study is a target estimated from 100 annotators instead of 5, so the
two things that must not break silently are the vote arithmetic and the guarantee that
the dense evaluation items never appear in training.
"""

from __future__ import annotations

import pytest

from sentalign.data.chaosnli import (CHAOS_ABBREV, _band, _gold_from_votes, chaos_items,
                                     format_text, source_items)
from sentalign.labels import NLI3, NO_MAJORITY, build_prompt


def test_the_majority_rule_matches_dynasent_at_five_votes_and_generalises_to_a_hundred():
    """DynaSent's rule is three of five. Expressed as "more than half" it is unchanged
    there and still meaningful at 100 annotators, where 45/35/20 is genuinely ambiguous
    rather than a 45-vote 'winner'."""
    assert _gold_from_votes({"entailment": 3, "neutral": 2, "contradiction": 0}) == "entailment"
    assert _gold_from_votes({"entailment": 2, "neutral": 2, "contradiction": 1}) == NO_MAJORITY
    assert _gold_from_votes({"entailment": 60, "neutral": 40, "contradiction": 0}) == "entailment"
    assert _gold_from_votes({"entailment": 45, "neutral": 35, "contradiction": 20}) == NO_MAJORITY
    assert _gold_from_votes({}) == NO_MAJORITY


def test_dense_items_get_a_coarse_band_not_one_stratum_each():
    """"43of100" would put almost every item in its own group, which breaks the
    group-robust arms and the agreement-band analysis."""
    assert _band({"entailment": 100, "neutral": 0, "contradiction": 0}, "entailment") == "strong"
    assert _band({"entailment": 70, "neutral": 30, "contradiction": 0}, "entailment") == "moderate"
    assert _band({"entailment": 55, "neutral": 45, "contradiction": 0}, "entailment") == "weak"
    assert _band({"entailment": 45, "neutral": 35, "contradiction": 20}, NO_MAJORITY) == "no-majority"


def test_the_nli_prompt_carries_both_fields_and_ends_on_its_own_anchor():
    text = format_text("A man naps on a bench.", "A man is asleep.")
    prompt = build_prompt(text, NLI3)
    assert "Premise: A man naps on a bench." in prompt
    assert "Hypothesis: A man is asleep." in prompt
    assert prompt.endswith("Answer:"), "the verbalizer supplies the leading space"
    assert "Sentiment" not in prompt


class _Zip:
    """Stands in for a release archive so the parsers can be tested without a download."""

    def __init__(self, rows): self.rows = rows


def _patch(monkeypatch, rows):
    monkeypatch.setattr("sentalign.data.chaosnli.iter_jsonl",
                        lambda archive, member: iter(rows))


def test_source_items_count_annotator_labels_and_reject_a_simplified_release(monkeypatch):
    _patch(monkeypatch, [{
        "pairID": "p1", "sentence1": "A man naps.", "sentence2": "A man is asleep.",
        "annotator_labels": ["entailment", "entailment", "entailment", "neutral", "-"],
    }])
    items = source_items(_Zip([]), "m", "snli")
    assert items[0].votes == {"entailment": 3, "neutral": 1, "contradiction": 0}
    assert items[0].gold == "entailment"          # 3 of 4 in-space votes is a majority
    assert items[0].text_id == "p1"

    _patch(monkeypatch, [{"pairID": "p1", "sentence1": "a", "sentence2": "b",
                          "gold_label": "entailment"}])
    with pytest.raises(KeyError, match="annotator_labels"):
        source_items(_Zip([]), "m", "snli")


def test_chaos_items_require_the_full_hundred_annotations(monkeypatch):
    good = {"uid": "u1", "label_counter": {"e": 45, "n": 35, "c": 20},
            "example": {"premise": "A man naps.", "hypothesis": "A man is asleep."}}
    _patch(monkeypatch, [good])
    it = chaos_items(_Zip([]), "m", "chaos_snli")[0]
    assert it.votes == {"entailment": 45, "neutral": 35, "contradiction": 20}
    assert it.gold == NO_MAJORITY
    assert it.distribution_vector(NLI3) == pytest.approx([0.45, 0.35, 0.20])

    short = dict(good, label_counter={"e": 5, "n": 3, "c": 2})
    _patch(monkeypatch, [short])
    with pytest.raises(ValueError, match="expected 100"):
        chaos_items(_Zip([]), "m", "chaos_snli")


def test_the_abbreviations_cover_the_whole_label_space():
    assert set(CHAOS_ABBREV.values()) == set(NLI3.labels)


def _fake_pair(n=8):
    """n dense items and their coarse counterparts, sharing ids."""
    from sentalign.data.dynasent import Item
    dense, coarse = [], []
    for k in range(n):
        e = 40 + k
        dv = {"entailment": e, "neutral": 100 - e - 10, "contradiction": 10}
        cv = {"entailment": 3, "neutral": 1, "contradiction": 1}
        for votes, bucket in ((dv, dense), (cv, coarse)):
            bucket.append(Item(text_id=f"u{k}", text=f"Premise: p{k}\nHypothesis: h{k}",
                               votes=votes, gold="entailment", round="r", source="s",
                               meta={"band": "weak"}))
    return dense, coarse


def test_the_paired_evaluation_sets_cover_the_same_items_and_never_touch_training(monkeypatch):
    """The whole point is scoring one model against a dense and a coarse target over the
    *same* items. If the two sets drift apart, the contrast silently becomes a comparison
    of two different samples, which would look like a result."""
    from pathlib import Path

    import sentalign.data.chaosnli as mod

    monkeypatch.setattr(mod, "chaos_items_from_hf", lambda *a, **k: _fake_pair(8))
    out = mod.load_nli(Path("/nonexistent"), chaos_only=True)

    dense, coarse, train = out["chaos_dense"], out["chaos_coarse"], out["train"]
    assert [i.text_id for i in dense] == [i.text_id for i in coarse] != []
    assert out["regime"] == "dense_train_dense_eval"
    assert not ({i.text_id for i in train} & {i.text_id for i in dense})
    # The two targets differ; that is the variable under study.
    assert dense[0].votes != coarse[0].votes
    assert sum(dense[0].votes.values()) == 100
    assert sum(coarse[0].votes.values()) == 5


def test_an_unreachable_archive_falls_back_instead_of_failing_the_build(monkeypatch):
    """The upstream ChaosNLI archive is behind a share that answers scripts with HTML.
    A build must still be possible, and must say which regime it produced."""
    from pathlib import Path

    import sentalign.data.chaosnli as mod

    monkeypatch.setattr(mod, "chaos_items_from_hf", lambda *a, **k: _fake_pair(6))

    def boom(*a, **k):
        raise ValueError("not a zip")

    monkeypatch.setattr(mod, "download", boom)
    out = mod.load_nli(Path("/nonexistent"))
    assert out["regime"] == "dense_train_dense_eval"
    assert out["train"] and out["chaos_dense"]


def test_a_label_nobody_chose_reads_as_zero_not_as_a_crash(monkeypatch):
    """The mirror stores label_counter as a struct, so a label absent from the original
    JSON arrives as None. Treating that as zero is required; dropping the row or the key
    would renormalise the target over the wrong support."""
    import sentalign.data.chaosnli as mod

    rows = [{"uid": "u1", "label_counter": {"e": 100, "n": None, "c": None},
             "premise": "A man naps.", "hypothesis": "A man is asleep.",
             "old_labels": ["entailment", "entailment", "entailment",
                            "entailment", "neutral"]}]
    monkeypatch.setattr(mod, "load_dataset", lambda *a, **k: rows, raising=False)
    monkeypatch.setitem(__import__("sys").modules, "datasets",
                        type("m", (), {"load_dataset": staticmethod(lambda *a, **k: rows)}))

    dense, coarse = mod.chaos_items_from_hf()
    assert dense[0].votes == {"entailment": 100, "neutral": 0, "contradiction": 0}
    assert sum(dense[0].votes.values()) == 100
    assert coarse[0].votes == {"entailment": 4, "neutral": 1, "contradiction": 0}


def test_chaos_only_can_supervise_the_same_items_at_either_density(monkeypatch):
    """The dense-training experiment is only interpretable against a coarse build over
    the SAME items: otherwise density is confounded with item count. Both builds must
    therefore carry identical training ids and differ only in the vote counts."""
    from pathlib import Path

    import sentalign.data.chaosnli as mod

    monkeypatch.setattr(mod, "chaos_items_from_hf", lambda *a, **k: _fake_pair(10))
    d = mod.load_nli(Path("/nonexistent"), chaos_only=True, train_target="dense")
    c = mod.load_nli(Path("/nonexistent"), chaos_only=True, train_target="coarse")

    assert [i.text_id for i in d["train"]] == [i.text_id for i in c["train"]]
    assert sum(d["train"][0].votes.values()) == 100
    assert sum(c["train"][0].votes.values()) == 5
    assert d["regime"] == "dense_train_dense_eval"
    assert c["regime"] == "coarse_train_dense_eval"
    # The evaluation half is dense in both, and never overlaps training.
    for out in (d, c):
        assert not ({i.text_id for i in out["train"]}
                    & {i.text_id for i in out["chaos_dense"]})
