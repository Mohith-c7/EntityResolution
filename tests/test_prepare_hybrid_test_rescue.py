import importlib.util
from pathlib import Path
import pandas as pd
import pytest

P=Path(__file__).resolve().parents[1]/'research/final_2h/prepare_hybrid_test_rescue.py'
spec=importlib.util.spec_from_file_location('hybrid_test_prepare',P)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def graph():
    return pd.DataFrame([{'source1_entity_id':'S1-a','candidate_entity_id':f'S2-{i:02d}','probability':1-i/100,'first_stage':.5,'candidate_order':i} for i in range(40)])

def test_exclude_after_top4_never_refill_fifth():
    f=graph();out=m.select_extra(f,{('S1-a','S2-00'),('S1-a','S2-02')})
    assert list(out.candidate_entity_id)==['S2-01','S2-03']
    assert list(out.probability)==[.99,.97]

def test_ties_target_id_and_complete_context():
    f=graph();f['probability']=.8
    out=m.select_extra(f.sample(frac=1,random_state=3),set())
    assert list(out.candidate_entity_id)==['S2-00','S2-01','S2-02','S2-03']
    with pytest.raises(ValueError,match='forty'):
        m.select_extra(f.iloc[:39],set())
    with pytest.raises(ValueError,match='forty'):
        m.select_extra(pd.concat([f.iloc[:39],f.iloc[:1]]),set())
