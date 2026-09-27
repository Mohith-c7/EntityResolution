import numpy as np
import pandas as pd
import pytest
from research.final_2h.prepare_empty_rescue import select_empty_top4


def graph():
    return pd.DataFrame([{'source1_entity_id':e,'candidate_entity_id':f'S2-{i:02}',
        'probability':.2,'first_stage':.1} for e in ['S1-empty','S1-kept'] for i in reversed(range(40))])


def test_empty_route_deterministic_ties_needs_no_confident_seed():
    f=graph();chosen=np.zeros(len(f),dtype=bool);chosen[40]=True
    result=select_empty_top4(f,chosen,['S1-empty','S1-kept'])
    assert list(result.candidate_entity_id)==['S2-00','S2-01','S2-02','S2-03']
    assert set(result.source1_entity_id)=={'S1-empty'}
    assert result.first_stage.max()<.9


def test_truncated_owner_context_rejected():
    f=graph().iloc[1:]
    with pytest.raises(ValueError,match='Complete40'):
        select_empty_top4(f,np.zeros(len(f),dtype=bool),['S1-empty','S1-kept'])
