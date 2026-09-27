import copy
import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('address_dropout',Path(__file__).parents[1]/'research/final_2h/prepare_address_dropout.py')
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

def row(i,label,left=None,right=None):
    return {'source1_entity_id':str(i),'candidate_entity_id':'t'+str(i),'label':label,'text_left':left or 'name: left'+str(i)+' country: US address: Road','text_right':right or 'name: target'+str(i)+' country: US address: Road'}

def test_balanced_suffix_only_and_modified_decoy_exclusion():
    original=[row(i,i%2) for i in range(8)]
    street=copy.deepcopy(original); street[0]['text_left']='name: decoy country: US address: Road'
    chosen,decoys,skipped=m.selection(original,street,2,42,'a','b')
    derived=m.derive(original,street,chosen)
    assert decoys=={0} and 0 not in chosen
    assert sum(street[i]['label']==0 for i in chosen)==2
    assert sum(street[i]['label']==1 for i in chosen)==2
    for i,r in enumerate(derived):
        assert r['text_left']==street[i]['text_left'] and r['label']==street[i]['label']
        assert r['text_right']==m.masked(street[i]['text_right'])[0] if i in chosen else r==street[i]
    assert m.selection(original,street,2,42,'a','b')[0]==chosen

def test_skip_opposite_label_text_collision_and_fill():
    data=[row(0,0,left='shared',right='name: same country: US address: Road'),row(1,1,left='shared',right='name: same country: US address: '),row(2,0),row(3,1)]
    chosen,_,skipped=m.selection(data,data,1,42,'a','b')
    assert 0 not in chosen and 2 in chosen and 3 in chosen and skipped==1

def test_parent_key_label_reorder_and_insufficient_eligible_rejected():
    data=[row(0,0),row(1,1)]
    changed=copy.deepcopy(data); changed[0]['label']=1
    with pytest.raises(ValueError,match='keys/order/labels'):m.selection(data,changed,1,42,'a','b')
    with pytest.raises(ValueError,match='Insufficient'):m.selection(data,data,2,42,'a','b')
