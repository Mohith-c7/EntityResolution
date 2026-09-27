import importlib.util
import json
from pathlib import Path
import pandas as pd
import pytest

SPEC=importlib.util.spec_from_file_location('neural_peer',Path(__file__).resolve().parents[1]/'research/final_2h/neural_peer.py')
MODULE=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(MODULE)


def fixture(tmp_path,owner='outer'):
    scores=tmp_path/'scores';scores.mkdir()
    values=[{'source1_entity_id':'B','candidate_entity_id':'Y','neural_probability':.9},
        {'source1_entity_id':'A','candidate_entity_id':'X','neural_probability':.3}]
    (scores/'scores.jsonl').write_text('\n'.join(json.dumps(r) for r in values)+'\n')
    head=tmp_path/'head.json';head.write_text(json.dumps({'training_owners':[owner]}))
    im=tmp_path/'input.json';im.write_text(json.dumps({'labels_read':False,'input_sha256':'input','feature_manifest_sha256':'reverse'}))
    (scores/'manifest.json').write_text(json.dumps({'labels_read':False,'scores_sha256':MODULE.sha(scores/'scores.jsonl'),
        'input_sha256':'input','checkpoint_manifest_sha256':MODULE.sha(head)}))
    frame=pd.DataFrame({'source1_entity_id':['A','B'],'candidate_entity_id':['X','Y'],'neural_probability':[.1,.2]})
    return frame,{'reverse_feature_manifest_sha256':'reverse'},scores,im,head


def test_keyed_fine_scores_keep_original_route_order(tmp_path):
    args=fixture(tmp_path)
    result=MODULE.add_fine(*args)
    assert result.neural_fine_probability.tolist()==pytest.approx([.3,.9])
    assert result.neural_probability.tolist()==[.1,.2]


def test_fine_fitted_owner_cannot_enter_calibration(tmp_path):
    with pytest.raises(ValueError,match='fitted owner'):
        MODULE.add_fine(*fixture(tmp_path,owner='A'))


def test_changed_score_file_is_rejected(tmp_path):
    args=fixture(tmp_path);(args[2]/'scores.jsonl').write_text('{}\n')
    with pytest.raises(ValueError,match='score seal'):
        MODULE.add_fine(*args)
