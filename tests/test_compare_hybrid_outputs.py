import importlib.util
from pathlib import Path
import pytest
P=Path(__file__).resolve().parents[1]/'research/final_2h/compare_hybrid_outputs.py'
s=importlib.util.spec_from_file_location('hybrid_diff',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

def test_order_change_is_not_link_change_and_empty_transitions():
    _,_,delta=m.changes('S2-a,S3-b','S3-b,S2-a');assert delta['changed_owners']==0;assert delta['added_links']==delta['removed_links']==0
    _,_,delta=m.changes('','S2-a');assert delta['empty_to_nonempty']==1;assert delta['added_links']==1
    _,_,delta=m.changes('S2-a','');assert delta['nonempty_to_empty']==1;assert delta['removed_links']==1
    with pytest.raises(ValueError,match='Repeated'):m.changes('S2-a,S2-a','')
