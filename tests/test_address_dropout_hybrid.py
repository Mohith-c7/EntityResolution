import importlib.util
from pathlib import Path

spec=importlib.util.spec_from_file_location('dropout_hybrid',Path(__file__).parents[1]/'research/final_2h/evaluate_address_dropout_hybrid.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_prospective_macro_rule_keeps_precision_diagnostic_but_country_binding():
    r={'paired_macro_f05_delta':.0005,'paired_delta_95pct_ci':[.0001,.0009],'country_deltas':{'US':.001,'India':-.0004},'links':{'candidate_precision':.99,'baseline_precision':.999}}
    assert m.passed(r)
    r['country_deltas']['India']=-.0006
    assert not m.passed(r)

def test_zero_lower_bound_or_subminimum_point_gain_is_not_promotable():
    r={'paired_macro_f05_delta':.0004,'paired_delta_95pct_ci':[0.,.001],'country_deltas':{'US':.001,'India':.001}}
    assert not m.passed(r)
    r['paired_delta_95pct_ci'][0]=.0001;r['paired_macro_f05_delta']=.00029
    assert not m.passed(r)
