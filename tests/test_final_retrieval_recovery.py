"""Bounded retrieval invariants; no labels or real model/index assets."""
import importlib.util
from pathlib import Path

import pandas as pd
import pytest


PATH = Path(__file__).resolve().parents[1] / 'research/final_2h/retrieval_recovery.py'
spec = importlib.util.spec_from_file_location('retrieval_recovery', PATH)
recovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recovery)


def pairs(targets):
    return pd.DataFrame([('S1-a', target) for target in targets], columns=recovery.KEYS)


def test_original_candidates_preserved_and_additions_identified():
    added = recovery.validate_extension(pairs(['S2-a', 'S3-a']), pairs(['S2-a', 'S3-a', 'S2-b']),
                                        {'S1-a': ['S2-a', 'S3-a']})
    assert added.to_dict('records') == [{'source1_entity_id': 'S1-a', 'candidate_entity_id': 'S2-b'}]


def test_removed_or_duplicate_candidate_rejected():
    with pytest.raises(ValueError, match='removed'):
        recovery.validate_extension(pairs(['S2-a', 'S3-a']), pairs(['S2-a']), {'S1-a': ['S2-a', 'S3-a']})
    with pytest.raises(ValueError, match='duplicate'):
        recovery.pair_set(pairs(['S2-a', 'S2-a']))


def test_more_than_four_additions_rejected():
    with pytest.raises(ValueError, match='four'):
        recovery.validate_extension(pairs(['S2-a']), pairs(['S2-a', 'S2-b', 'S2-c', 'S2-d', 'S2-e', 'S2-f']),
                                    {'S1-a': ['S2-a']})


def test_sealed_baseline_mismatch_rejected():
    with pytest.raises(ValueError, match='sealed'):
        recovery.validate_extension(pairs(['S2-a']), pairs(['S2-a', 'S3-a']), {'S1-a': ['S2-other']})


def test_features_route_low_probability_addition_without_seed():
    frame = pairs(['S2-a', 'S3-a'])
    frame['first_stage'] = [.5, .2]
    frame['probability'] = [.7, .03]
    tasks = recovery.feature_jobs(frame, {'S1-a': 'US'})
    assert {row[1] for row in tasks} == {'S2-a', 'S3-a'}
    assert all(row[-2:] == ([], []) for row in tasks)
    assert tasks[1][3:7] == (.2, .03, 2, 2)
