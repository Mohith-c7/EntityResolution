import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd

SPEC=importlib.util.spec_from_file_location('early_macro_metric',Path(__file__).resolve().parents[1]/'research/final_2h/early_macro_metric.py')
MODULE=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(MODULE)


def test_complete_truth_includes_unretrieved_matches_and_singletons():
    full=pd.DataFrame({'source1_entity_id':['A','A','B','C','C'],'candidate_entity_id':['X','Y','Z','U','V'],'probability':[.99,.8,.1,.75,.75]})
    truth={'A':['X','Y','missing'],'B':[],'C':['U']}
    routed=full.iloc[[1,2,3,4]]
    metric=MODULE.EarlyMacroMetric(full,routed,truth)
    _,score,_=metric(np.array([.84,.2,.75,.75]))
    assert np.isclose(score,(2.5/2.75+1+1)/3)


def test_missing_routed_ids_rejected():
    full=pd.DataFrame({'source1_entity_id':['A'],'candidate_entity_id':['X'],'probability':[.9]})
    routed=full.assign(candidate_entity_id='absent')
    import pytest
    with pytest.raises(ValueError,match='Routed keys'):
        MODULE.EarlyMacroMetric(full,routed,{'A':['X']})
