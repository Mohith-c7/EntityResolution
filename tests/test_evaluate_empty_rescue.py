import numpy as np
import pandas as pd
import pytest
from research.final_2h.evaluate_empty_rescue import extra_only_scatter,adapter


class Model:
    def predict(self,values,**kwargs):return np.full(len(values),.5)


def frames():
    original=pd.DataFrame({'source1_entity_id':['S1-a']*3,'candidate_entity_id':['S2-a','S2-b','S2-c'],
        'first_stage':[.2,.1,.1],'probability':[.8,.4,.2]})
    baseline=original.copy();baseline.loc[0,'probability']=.95
    f=original.copy();f['candidate_rank']=[1,2,3]
    return original,baseline,f


def test_preserve_existing_correction_and_apply_original_anchor_once():
    original,baseline,f=frames()
    result,extra,positions=extra_only_scatter(original,baseline,f,{('S1-a','S2-a')},Model(),['probability'])
    assert result.probability.iat[0]==.95
    assert np.array_equal(result.probability.iloc[1:],adapter.corrected_probability([.4,.2],[.5,.5]))
    assert len(extra)==2 and list(positions)==[1,2]


def test_already_corrected_extra_anchor_rejected():
    original,baseline,f=frames()
    with pytest.raises(ValueError,match='already corrected'):
        extra_only_scatter(original,baseline,f,set(),Model(),['probability'])


def test_rank_from_truncated_context_rejected():
    original,baseline,f=frames();f.loc[1,'candidate_rank']=1
    with pytest.raises(ValueError,match='full40rank'):
        extra_only_scatter(original,baseline,f,{('S1-a','S2-a')},Model(),['probability'])
