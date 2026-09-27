import copy
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def module(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / file)
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
    return result


reserve = module('reserve_last8_extension', 'research/final_2h/reserve_last8_extension_audit.py')
evaluate = module('evaluate_last8_development', 'research/final_2h/evaluate_last8_development.py')


def test_hash_reservation_independent_of_record_order_and_excludes_exposure():
    ids = [f'S1-{i}' for i in range(100)]
    excluded = ids[:30]
    selected, eligible = reserve.sample_ids(ids, excluded, count=10)
    reversed_selected, _ = reserve.sample_ids(ids[::-1], excluded[::-1], count=10)
    assert selected == reversed_selected and eligible == 70
    assert len(set(selected)) == 10 and not set(selected) & set(excluded)
    with pytest.raises(ValueError, match='Insufficient'):
        reserve.sample_ids(ids, excluded, count=71)


def passing_result():
    return {'paired_macro_f05_delta': .0003, 'paired_delta_95pct_ci': [.00001, .0007],
        'candidate': {'micro_precision': .997, 'singleton_false_positives': 12},
        'baseline': {'micro_precision': .996, 'singleton_false_positives': 12},
        'country_deltas': {'US': .0004, 'India': -.0005}}


def test_acceptance_requires_predeclared_point_and_ci_gain():
    result = passing_result(); assert evaluate.accepted(result)
    result['paired_macro_f05_delta'] = .0002999
    assert not evaluate.accepted(result)
    result = passing_result(); result['paired_delta_95pct_ci'][0] = 0
    assert not evaluate.accepted(result)


@pytest.mark.parametrize('harm', ['precision', 'singletons', 'country'])
def test_acceptance_rejects_each_predeclared_quality_harm(harm):
    result = copy.deepcopy(passing_result())
    if harm == 'precision': result['candidate']['micro_precision'] = .9959
    elif harm == 'singletons': result['candidate']['singleton_false_positives'] = 13
    else: result['country_deltas']['India'] = -.0005001
    assert not evaluate.accepted(result)
