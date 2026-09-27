import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code/business_entity_resolution"))
from src.features.competition import COMPETITION_NAMES, KEY_KINDS, competition_features, count_universe, record_keys


def features(reference, target, universe):
    wanted = {kind: {k[kind] for k in (reference, target) if k[kind] is not None} for kind in KEY_KINDS}
    return dict(zip(COMPETITION_NAMES, competition_features(reference, target, count_universe(universe, wanted))))


def test_reference_is_excluded_from_its_own_counts():
    ours = record_keys("south minerals", "south minerals", "38 schreiner way sand lake ny")
    target = record_keys("south minerals", "south minerals", "")
    result = features(ours, target, [ours])
    assert result["cmp_target_core_others"] == 0
    assert result["cmp_reference_core_others"] == 0
    assert result["cmp_reference_address_others"] == 0


def test_another_owner_is_counted_and_missing_address_is_not_uniqueness():
    ours = record_keys("south minerals", "south minerals", "38 schreiner way sand lake ny")
    other = record_keys("south minerals", "south minerals", "551 sunset road russell county va")
    target = record_keys("south mínerals", "south mínerals", "")
    result = features(ours, target, [ours, other])
    assert result["cmp_target_core_others"] == 1
    assert result["cmp_target_address_others"] == -1
    assert result["cmp_target_core_number_others"] == -1


def test_keys_ignore_token_order_and_zero_padding_and_repeat_exactly():
    left = record_keys("tech data integrated", "tech data integrated", "12 main street")
    right = record_keys("integrated data tech", "integrated data tech", "street main 012")
    assert left["core_set"] == right["core_set"]
    assert left["address_set"] == right["address_set"]
    assert left["core_number"] == right["core_number"]
    assert left == record_keys("tech data integrated", "tech data integrated", "12 main street")


def test_reference_outside_the_universe_is_rejected():
    ours = record_keys("alpha", "alpha", "1 road")
    with pytest.raises(ValueError):
        features(ours, ours, [])
