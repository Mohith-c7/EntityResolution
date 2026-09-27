import pandas as pd
import pytest
from research.final_2h.evaluate_fixed_hybrid import combine,prospective_pass


def frames():
    f=pd.DataFrame({'source1_entity_id':['S1-a']*3,'candidate_entity_id':['S2-a','S2-b','S2-c'],
        'probability':[.2,.3,.1]})
    blue=f.copy();blue.loc[0,'probability']=.8
    eight=f.copy();eight.loc[0,'probability']=.9
    rescue=blue.copy();rescue.loc[1,'probability']=.7
    return f,blue,eight,rescue


def test_saved_corrected_probability_overwrite_is_exact_no_double_anchor():
    f,b,e,r=frames();out,_,_=combine(f,b,e,r,f.iloc[[0]],f.iloc[[1]])
    assert out.probability.tolist()==[.9,.7,.1]


def test_overlap_rejected_and_component_scope_must_be_exact():
    f,b,e,r=frames()
    with pytest.raises(ValueError,match='disjoint'):combine(f,b,e,r,f.iloc[[0]],f.iloc[[0]])
    e.loc[2,'probability']=.5
    with pytest.raises(ValueError,match='outside its original route'):combine(f,b,e,r,f.iloc[[0]],f.iloc[[1]])


def test_new_metric_rule_keeps_precision_and_singletons_diagnostic_only():
    r={'paired_macro_f05_delta':.0004,'paired_delta_95pct_ci':[.0001,.0008],
        'country_deltas':{'us':.0001,'india':.0007},'legacy_guard_diagnostics':{'quality_improvement_pass':False}}
    assert prospective_pass(r)
    r['paired_macro_f05_delta']=.00029
    assert not prospective_pass(r)


def test_new_rule_requires_positive_ci_and_country_floor():
    r={'paired_macro_f05_delta':.0004,'paired_delta_95pct_ci':[0.,.0008],'country_deltas':{'us':.0004}}
    assert not prospective_pass(r)
    r['paired_delta_95pct_ci'][0]=.0001;r['country_deltas']['india']=-.00051
    assert not prospective_pass(r)
