import importlib.util
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
P=Path(__file__).resolve().parents[1]/'scripts/build_hybrid_submission.py'
spec=importlib.util.spec_from_file_location('hybrid_submission',P)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class Model:
    def __init__(self,margin):self.margin=margin
    def feature_name(self):return m.FEATURES
    def num_feature(self):return 38
    def predict(self,x,**kwargs):return np.full(len(x),self.margin)

def inputs():
    base=pd.DataFrame([{'source1_entity_id':'S1-a','candidate_entity_id':f'S2-{i:02d}','probability':.9-i*.01,'first_stage':.5,'candidate_order':i} for i in range(40)])
    def feature(i):
        f=base.iloc[[i]].copy()
        for name in m.FEATURES:
            if name not in f:f[name]=0.
        f['candidate_rank']=i+1
        return f
    return base,feature(0),feature(3)

def test_disjoint_two_models_one_original_anchor_and_outside_exact():
    base,route,extra=inputs()
    result=m.apply_hybrid(base,route,extra,Model(.3),Model(-.5))
    expected=base.probability.to_numpy(copy=True)
    expected[[0,3]]=m.nn.reverse.corrected_probability(expected[[0,3]],np.array([.3,-.5]))
    assert np.array_equal(result.probability,expected)
    assert np.array_equal(base.probability,result.probability_original)
    assert result[m.KEYS+['candidate_order']].equals(base[m.KEYS+['candidate_order']])
    with pytest.raises(ValueError,match='overlap'):
        m.apply_hybrid(base,route,route,Model(.3),Model(-.5))

def test_changed_anchor_or_rank_rejected():
    base,route,extra=inputs();extra['probability']=.3
    with pytest.raises(ValueError,match='anchor'):
        m.apply_hybrid(base,route,extra,Model(.3),Model(-.5))
    base,route,extra=inputs();extra['candidate_rank']=5
    with pytest.raises(ValueError,match='rank'):
        m.apply_hybrid(base,route,extra,Model(.3),Model(-.5))

def test_exact_key_old_join_preserves_order():
    _,route,extra=inputs();frame=pd.concat([route,extra]).drop(columns=['neural_probability','neural_fine_probability'])
    values=frame[m.KEYS].iloc[::-1].copy();values['neural_probability']=[.7,.2]
    result=m.add_old(frame,values)
    assert result[m.KEYS].equals(frame[m.KEYS]);assert list(result.neural_probability)==[.2,.7]
    with pytest.raises(ValueError,match='universe'):
        m.add_old(frame,values.iloc[:1])

def test_audit_cannot_reuse_wrong_freeze_or_legacy_acceptance(tmp_path,monkeypatch):
    path=tmp_path/'gate.json';freeze=tmp_path/'freeze.json';freeze.write_text('{}')
    frozen={'baseline_frozen_sha256':'baseline','reservation_plan_sha256':'reserve','audit_freeze_sha256':'audit','extension_acceptance_criteria':m.CRITERIA}
    gate={'status':'paired_extension_audit_complete','candidate_frozen_sha256':'audit','baseline_frozen_sha256':'baseline','reservation_plan_sha256':'reserve','hybrid_prospective_acceptance':{'passed':True,'criteria':m.CRITERIA}}
    gate.update(runtime={'passed':True,'evidence':{'candidate_frozen_sha256':'audit'}},full_claim_graph={'references':331012,'pairs':13240480},claimant_universe={'kind':'complete_heldout','complete':True,'scoring_complete':True,'supervised_training_excluded':True,'declared_references':331012,'scored_references':331012,'pairs':13240480})
    monkeypatch.setattr(m,'pinned',lambda path,frozen,group:json.loads(path.read_text()))
    path.write_text(json.dumps(gate));assert m.verify_audit(path,frozen,freeze)==gate
    gate['candidate_frozen_sha256']='old';path.write_text(json.dumps(gate))
    with pytest.raises(ValueError,match='another'):m.verify_audit(path,frozen,freeze)
    gate['candidate_frozen_sha256']='audit';gate.pop('hybrid_prospective_acceptance');gate['acceptance']={'promotion_pass':True};path.write_text(json.dumps(gate))
    with pytest.raises(ValueError,match='failed'):m.verify_audit(path,frozen,freeze)


def test_extra_route_uses_blue_global_empty_and_top4_not_fifth(tmp_path):
    base,route,extra=inputs();blue=tmp_path/'blue';blue.mkdir()
    pd.DataFrame({'source1_entity_id':['S1-a'],'matched_entity_ids':['']}).to_csv(blue/'matching_results.tsv',sep='\t',index=False)
    protocol={'blue_matching_sha256':m.reuse.digest(blue/'matching_results.tsv')}
    old=base.iloc[[0]][m.KEYS]
    expected=base.iloc[[1,2,3]].copy()
    assert m.validate_extra_route(base,old,expected,blue,protocol)==1
    bad=base.iloc[[1,2,4]].copy()
    with pytest.raises(ValueError,match='top4'):
        m.validate_extra_route(base,old,bad,blue,protocol)


def test_actual_eight_layer_scores_use_generic_unchanged_inference_code():
    # Training used neural_layers8; inference intentionally used the generic
    # verified neural_layers scorer. Keep that code binding strict.
    source=m.ROOT/'models/sprint_6h/neural/last8_full_test06/manifest.json'
    if not source.exists():pytest.skip('Actual ignored score cache unavailable in this checkout')
    marker=json.loads(source.read_text())
    assert marker['status']=='complete'
    assert marker['checkpoint_manifest_sha256']==m.HEAD8
    assert marker['code_sha256']==m.reuse.digest(m.FINE_SCORE_CODE)
    assert m.FINE_SCORE_CODE.name=='neural_layers.py'
