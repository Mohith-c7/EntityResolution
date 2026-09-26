import importlib.util
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
spec = importlib.util.spec_from_file_location("context", Path(__file__).resolve().parents[1]/"scripts/build_context_features.py")
context = importlib.util.module_from_spec(spec)
spec.loader.exec_module(context)


def test_candidate_cannot_be_its_own_supporting_peer():
    record=context.clean_record("alpha paris", "12 rue paris")
    values=context.pair_context(record,{"S2-1":record},["S2-1"],np.array([.999]))[0]
    result=dict(zip(context.CONTEXT_NAMES,values))
    assert result["ctx_strong_peers"] == 0
    assert result["ctx_peer_name_set"] == 0
    assert result["ctx_peer_address_set"] == 0
    assert result["ctx_best_other_probability"] == 0
    assert record[2] == "alpha"


def test_peer_evidence_is_order_independent_and_excludes_low_confidence_records():
    reference=context.clean_record("alpha", "12 road")
    targets={"S2-1":reference,"S3-2":reference,"S3-3":context.clean_record("different","99 avenue")}
    ids=["S2-1","S3-2","S3-3"]; probabilities=np.array([.99,.96,.2])
    first=context.pair_context(reference,targets,ids,probabilities)
    second=context.pair_context(reference,targets,ids[::-1],probabilities[::-1])
    np.testing.assert_array_equal(first,second[::-1])
    result=dict(zip(context.CONTEXT_NAMES,first[0]))
    assert result["ctx_strong_peers"] == 1
    assert result["ctx_opposite_source_strong_peers"] == 1
    assert result["ctx_peer_name_set"] == 1
    assert result["ctx_peer_address_set"] == 1
