import pytest
from research.final_2h.current4_remaining_loss import allocation, text_classes


def test_loss_interactions_reconcile_and_both_fn_categories_receive_credit():
    values, missing, rejected, false = allocation({'a','b','c'}, {'a','x'}, {'a','b','x'})
    assert missing == {'c'} and rejected == {'b'} and false == {'x'}
    assert all(v > 0 for v in values[:3])
    assert sum(values) == pytest.approx(1 - 1.25 / (.75 + 2))


def test_empty_non_singletons_are_separate_and_singleton_false_costs_one():
    values, *_ = allocation({'a','b'}, set(), {'a'})
    assert values == [0,0,0,1]
    values, *_ = allocation(set(), {'x','y'}, {'x','y'})
    assert values == pytest.approx([0,0,1,0])
    assert allocation(set(),set(),set())[0] == [0,0,0,0]


def test_script_name_and_address_are_record_classes_only():
    assert text_classes({'business_name':'अमर स्टोर','business_address':'Plot 12'}) == {
        'script':'non_latin','name':'2_to_4_words','address':'has_digits'}
