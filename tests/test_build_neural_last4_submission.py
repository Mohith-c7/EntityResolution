"""The faster exporter must preserve the reviewed heldout scatter semantics."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


exporter = load('last4_exporter', 'scripts/build_neural_last4_submission.py')
heldout = load('last4_heldout', 'research/final_2h/score_neural_last4_heldout.py')


def graph():
    # A's tied contenders span scatter chunks; lexical tie ranking matters.
    return pd.DataFrame({
        'source1_entity_id': ['A', 'A', 'A', 'B', 'B', 'C'],
        'candidate_entity_id': ['Z', 'X', 'Y', 'W', 'V', 'U'],
        'candidate_order': [0, 1, 2, 0, 1, 0],
        'probability': [.8, .8, .6, 0., 1., .3],
        'first_stage': [.9, .7, .5, .1, .95, .4],
    })


def routed(base):
    result = base.iloc[[3, 2, 0, 4]][[*exporter.KEYS, 'probability', 'first_stage']].copy()
    result['candidate_rank'] = exporter.nn.reverse.frozen_p2_rank(base)[[3, 2, 0, 4]]
    return result.reset_index(drop=True)


def test_chunk_scatter_matches_reviewed_scatter_with_full_owner_ranks():
    base = graph(); original = base.copy(deep=True); features = routed(base)
    correction = np.array([0., -.25, .4, 0.])
    expected = heldout.scatter_corrections(base, features, correction)
    actual = exporter.scatter_corrections(base, features, correction, chunk_rows=2)
    assert actual[expected.columns.drop('fine_adapter_correction')].equals(
        expected.drop(columns='fine_adapter_correction'))
    pd.testing.assert_frame_equal(base, original)
    assert actual.probability.iloc[[1, 5]].tolist() == [.8, .3]
    # A zero correction must preserve even exactly-zero/one frozen probabilities.
    assert actual.probability.iloc[[3, 4]].tolist() == [0., 1.]


def test_scatter_does_not_rank_only_routed_candidates():
    base = graph(); features = routed(base)
    features.loc[features.candidate_entity_id == 'Z', 'candidate_rank'] = 1
    with pytest.raises(ValueError, match='rank'):
        exporter.scatter_corrections(base, features, np.zeros(len(features)), chunk_rows=1)


@pytest.mark.parametrize('column', ['probability', 'first_stage'])
def test_scatter_rejects_changed_frozen_anchor(column):
    base = graph(); features = routed(base); features.loc[0, column] += .01
    with pytest.raises(ValueError, match='anchor'):
        exporter.scatter_corrections(base, features, np.zeros(len(features)))


def test_scatter_rejects_foreign_and_duplicate_keys():
    base = graph(); features = routed(base)
    features.loc[0, 'candidate_entity_id'] = 'foreign'
    with pytest.raises(ValueError, match='Foreign'):
        exporter.scatter_corrections(base, features, np.zeros(len(features)))
    features = pd.concat([routed(base), routed(base).iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match='Duplicate'):
        exporter.scatter_corrections(base, features, np.zeros(len(features)))


def test_fine_join_is_exact_keyed_and_preserves_old37():
    features = routed(graph())
    for name in exporter.nn.FEATURES:
        if name not in features:
            features[name] = .4
    old = features.copy(deep=True)
    values = features.iloc[::-1][exporter.KEYS].reset_index(drop=True)
    values['neural_probability'] = [.11, .22, .33, .44]
    result = exporter.assemble_fine(features, values)
    pd.testing.assert_frame_equal(result[old.columns], old)
    assert result.neural_fine_probability.tolist() == [.44, .33, .22, .11]
    with pytest.raises(ValueError, match='exactly cover'):
        exporter.assemble_fine(features, values.iloc[:-1])
    values.loc[0, 'candidate_entity_id'] = 'foreign'
    with pytest.raises(ValueError, match='exactly cover'):
        exporter.assemble_fine(features, values)


def test_native_adapter_applies_base_logit_once():
    base = graph(); features = routed(base)
    for name in exporter.nn.FEATURES:
        if name not in features:
            features[name] = .4
    features['neural_fine_probability'] = .9

    class ZeroResidual:
        def feature_name(self): return exporter.FEATURES
        def num_feature(self): return len(exporter.FEATURES)
        def predict(self, inputs, **kwargs):
            assert kwargs['raw_score'] is True
            assert inputs.shape == (len(features), 38)
            return np.zeros(len(inputs))

    result = exporter.apply_adapter(base, features, ZeroResidual(), chunk_rows=2)
    np.testing.assert_array_equal(result.probability, base.probability)


def test_two_peer_edges_cover_one_scored_pair():
    edges = pd.DataFrame({'source1_entity_id': ['S1-A', 'S1-A'],
        'candidate_entity_id': ['S2-X', 'S2-X'], 'peer_entity_id': ['S2-Y', 'S3-Z']})
    features = edges.iloc[[0]][exporter.KEYS].copy()
    pairs = exporter.distinct_route_pairs(edges, features, expected_rows=1)
    pd.testing.assert_frame_equal(pairs, features.reset_index(drop=True))
    with pytest.raises(ValueError, match='triplets'):
        exporter.distinct_route_pairs(pd.concat([edges, edges.iloc[[0]]]), features, expected_rows=1)


def test_distinct_route_still_rejects_foreign_feature_pair():
    edges = pd.DataFrame({'source1_entity_id': ['S1-A', 'S1-A'],
        'candidate_entity_id': ['S2-X', 'S2-X'], 'peer_entity_id': ['S2-Y', 'S3-Z']})
    features = pd.DataFrame({'source1_entity_id': ['S1-A'], 'candidate_entity_id': ['S2-foreign']})
    with pytest.raises(ValueError, match='exactly cover'):
        exporter.distinct_route_pairs(edges, features, expected_rows=1)
