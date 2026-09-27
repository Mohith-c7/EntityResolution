"""Independent regression review; Cursor snapshot supplied via environment.

The known-bug tests intentionally demonstrate deficiencies in commit 608749a.
They are marked xfail until Cursor supplies a replacement satisfying them.
"""
import importlib.util
import os
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("cursor_review", ROOT / "research/final_2h/cursor_evaluation.py")
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)
CONFIG = {"threshold": .83, "t_first": .70, "t_rest": .83}


def frame(rows):
    return pd.DataFrame(rows, columns=review.PAIR_KEYS + ["probability"])


@pytest.fixture
def cursor():
    path = os.environ.get("CURSOR_POLICY_PATH")
    if not path:
        pytest.skip("Set CURSOR_POLICY_PATH to immutable Cursor ownership snapshot")
    return review.load_policy(path)


def test_tie_is_global_and_deterministic(cursor):
    rows = frame([("B", "X", .9), ("A", "X", .9), ("C", "X", .9)])
    config = {**CONFIG, "tie_break": True, "cover_rank_one": True}
    for order in ([0, 1, 2], [2, 0, 1], [1, 2, 0]):
        shuffled = rows.iloc[order].reset_index(drop=True)
        chosen, _ = cursor.decide_v2(shuffled, shuffled, config)
        assert list(shuffled.loc[chosen].source1_entity_id) == ["A"]


def test_rescue_target_has_one_owner(cursor):
    rows = frame([("A", "X", .95), ("B", "X", .72)])
    chosen, _ = cursor.decide_v2(rows, rows, {**CONFIG, "tie_break": True, "cover_rank_one": True})
    assert chosen.tolist() == [True, False]


@pytest.mark.xfail(reason="608749a allows non-proposed lower-band rival to suppress viable owner", strict=True)
def test_discarded_rival_does_not_suppress_viable_owner(cursor):
    rows = frame([("A", "Y", .80), ("A", "X", .75), ("B", "X", .72)])
    chosen, _ = cursor.decide_v2(rows, rows, {**CONFIG, "tie_break": True, "cover_rank_one": True})
    assert chosen.tolist() == [True, False, True]


@pytest.mark.xfail(reason="608749a indexes empty order at position zero", strict=True)
def test_empty_graph(cursor):
    rows = frame([])
    chosen, lost = cursor.decide_v2(rows, rows, {**CONFIG, "tie_break": True, "cover_rank_one": True})
    assert len(chosen) == len(lost) == 0


def test_admissible_comparator_keeps_viable_owner():
    rows = frame([("A", "Y", .80), ("A", "X", .75), ("B", "X", .72)])
    chosen, _ = review.admissible_lex(rows, CONFIG)
    assert chosen.tolist() == [True, False, True]


def test_admissible_comparator_does_not_retry_lost_original_rescue():
    rows = frame([("A", "X", .80), ("A", "Y", .75), ("B", "X", .95)])
    chosen, _ = review.admissible_lex(rows, CONFIG)
    assert chosen.tolist() == [False, False, True]


def test_admissible_empty_graph():
    chosen, lost = review.admissible_lex(frame([]), CONFIG)
    assert len(chosen) == len(lost) == 0
