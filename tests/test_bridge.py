from types import SimpleNamespace
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"code/business_entity_resolution"))
from src.blocking import bridge
from src.blocking.disk_index import normalize_record
from src.blocking.contracts import Candidate


def test_bridge_uses_original_reference_and_preserves_final_budget(monkeypatch):
    def rec(eid,name):return normalize_record(dict(entity_id=eid,business_name=name,business_address="12 road",country="New Country"))
    root=rec("S1-1","Root");seed=rec("S2-1","Seed");new=rec("S2-2","New");decoy=rec("S2-3","Decoy")
    class Index:
        source="S2";config=SimpleNamespace(top_k=2)
        def query(self,reference):
            values=[seed,decoy] if reference.name=="root" else [seed,new]
            return [(Candidate(root.entity_id,t.entity_id,"S2",.8,("exact_core",)),t) for t in values]
        def rerank(self,reference,target,paths):
            assert reference==root and paths==("bridge",)
            return .4
        def prefetch_frequencies(self,records):pass
    class Ranker:
        def predict(self,matrix,**kwargs):return matrix[:,0]
    def features(reference,target,index):
        assert reference==root
        return [{"seed":.999,"new":.998,"decoy":.1}[target.name]]
    monkeypatch.setattr(bridge,"rank_features",features)
    rows=bridge.retrieve_with_bridges(root,[Index()],Ranker())["S2"]
    assert [c.candidate_entity_id for c,_ in rows]==["S2-1","S2-2"]
    assert rows[1][0].blocking_paths==("bridge",)
    assert rows[1][0].blocking_score==.4
    assert [c.rank_within_source for c,_ in rows]==[1,2]
