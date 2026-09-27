"""Guard against sparse auxiliary scores changing the wrong business pair."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
spec=importlib.util.spec_from_file_location('neural_export',ROOT/'scripts/build_neural_submission.py')
export=importlib.util.module_from_spec(spec);spec.loader.exec_module(export)


class Native:
    def __init__(self, value=0.): self.value=value
    def feature_name(self): return export.nn.FEATURES
    def predict(self, values, **kwargs):
        assert kwargs['raw_score'] is True
        return np.full(len(values),self.value)


def frames():
    base=pd.DataFrame({'source1_entity_id':['S1-a','S1-a','S1-b','S1-b'],
        'candidate_entity_id':['S2-x','S3-y','S2-z','S3-w'],
        'candidate_order':[0,1,0,1],'first_stage':[.9,.7,.8,.6],
        'second_stage':[.4,.6,.3,.7],'probability':[.4,.6,.3,.7]})
    features=pd.DataFrame(0.,index=range(2),columns=export.nn.FEATURES)
    routed=base.iloc[[3,0]].reset_index(drop=True)
    for key in export.reuse.PAIR_KEYS:features[key]=routed[key]
    features['probability']=routed.probability
    features['first_stage']=routed.first_stage
    features['candidate_rank']=[1,2]
    return base,features


def test_keyed_scatter_preserves_unrouted_and_candidate_order():
    base,features=frames()
    after=export.apply_adapter(base,features,Native(np.log(9)))
    assert after.loc[[1,2],'probability'].equals(base.loc[[1,2],'probability'])
    assert after.loc[0,'probability'] == pytest.approx(6/7)
    assert after.loc[3,'probability'] == pytest.approx(21/22)
    assert after[export.reuse.PAIR_KEYS+['candidate_order']].equals(base[export.reuse.PAIR_KEYS+['candidate_order']])
    assert after.probability_original.equals(base.probability)
    assert base.probability.tolist()==[.4,.6,.3,.7]


def test_zero_correction_is_bit_identical():
    base,features=frames()
    assert np.array_equal(export.apply_adapter(base,features,Native()).probability,base.probability)


@pytest.mark.parametrize('damage',['foreign','anchor','rank','duplicate'])
def test_incompatible_sparse_inputs_are_rejected(damage):
    base,features=frames()
    if damage=='foreign':features.loc[0,'candidate_entity_id']='S3-unknown'
    if damage=='anchor':features.loc[0,'probability']=.71
    if damage=='rank':features.loc[0,'candidate_rank']=2
    if damage=='duplicate':features=pd.concat([features,features.iloc[[0]]],ignore_index=True)
    with pytest.raises(ValueError):export.apply_adapter(base,features,Native())
