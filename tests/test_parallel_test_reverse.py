"""Owner partitioning must preserve frozen scores and global candidate ranks."""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'research/sprint_6h/sibling'))
import train_reverse_adapter as adapter


def test_partitioned_joins_equal_global_join_with_ties_and_unrouted_peers():
    frame = pd.DataFrame({
        'source1_entity_id': ['A']*4 + ['B']*4 + ['C']*4,
        'candidate_entity_id': ['Z', 'X', 'Y', 'W']*3,
        'probability': [.8, .8, .6, .9, .5, .4, .3, .6, .1, .2, .3, .4],
        'first_stage': [.95, .7, .8, .99]*3,
    })
    selected = [0, 2, 5]
    raw = frame.iloc[selected][adapter.sibling.KEYS].copy()
    for field in adapter.REVERSE_FEATURES:
        raw[field] = [0., .2, .4]
    expected = adapter.join_reverse(frame, raw, {i: [] for i in selected})
    pieces = []
    for owner in ['A', 'B']:
        complete = frame[frame.source1_entity_id == owner].reset_index(drop=True)
        features = raw[raw.source1_entity_id == owner].reset_index(drop=True)
        indices = complete.index[complete.candidate_entity_id.isin(features.candidate_entity_id)]
        pieces.append(adapter.join_reverse(complete, features, {i: [] for i in indices}))
    actual = pd.concat(pieces).sort_values(adapter.sibling.KEYS).reset_index(drop=True)
    pd.testing.assert_frame_equal(actual, expected)
    assert actual.candidate_rank.tolist() == [4, 3, 3]
