import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SPEC = importlib.util.spec_from_file_location('residual_strength', Path(__file__).resolve().parents[1] / 'research/final_2h/residual_strength.py')
experiment = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(experiment)


def fixtures():
    original = pd.DataFrame({'source1_entity_id': ['A', 'A', 'B'],
        'candidate_entity_id': ['X', 'Y', 'Z'], 'candidate_order': [0, 1, 0],
        'probability': [.8, 1., .3]})
    selected = original.copy(); selected['probability_original'] = original.probability
    selected.loc[0, 'probability'] = .9
    features = original.iloc[[0, 2]].copy()
    return original, selected, features


def test_strength_one_exact_and_half_uses_residual_once():
    original, selected, features = fixtures()
    positions, residual = experiment.route_residuals(original, selected, features)
    same = experiment.apply_strength(original, selected, positions, residual, 1.)
    np.testing.assert_array_equal(same.probability, selected.probability)
    half = experiment.apply_strength(original, selected, positions, residual, .5)
    expected = 1 / (1 + np.exp(-(.5 * np.log(.8 / .2) + .5 * np.log(.9 / .1))))
    assert half.probability.iloc[0] == pytest.approx(expected)
    assert half.probability.iloc[1] == 1.
    assert half.probability.iloc[2] == .3
    assert original.probability.tolist() == [.8, 1., .3]


def test_scatter_rejects_changed_outside_route():
    original, selected, features = fixtures(); selected.loc[1, 'probability'] = .99
    with pytest.raises(ValueError, match='outside-route'):
        experiment.route_residuals(original, selected, features)


def test_saturated_changed_probability_cannot_be_inverted():
    original, selected, features = fixtures(); selected.loc[0, 'probability'] = 1.
    with pytest.raises(ValueError, match='Saturated'):
        experiment.route_residuals(original, selected, features)
