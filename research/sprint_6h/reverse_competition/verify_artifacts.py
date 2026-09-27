"""Independent keyed, label-free checks of a completed reverse evidence export."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[3]/"scripts"))
import reverse_competition as reverse


def verify(directory):
    root=Path(directory)
    marker=json.loads((root/"manifest.json").read_text())
    frame=pd.read_parquet(root/"features.parquet")
    assert marker["status"] == "complete"
    assert list(frame) == reverse.KEYS+reverse.FEATURE_NAMES
    assert not frame.duplicated(reverse.KEYS).any()
    assert np.isfinite(frame[reverse.FEATURE_NAMES].to_numpy()).all()
    assert reverse.digest(root/"features.parquet") == marker["features_sha256"]
    assert reverse.digest(root/"diagnostics.jsonl") == marker["diagnostics_sha256"]
    assert reverse.pair_order_hash(frame) == marker["pair_order_sha256"]
    assert reverse.digest(Path(reverse.__file__)) == marker["feature_code_sha256"]
    expected=frame.set_index(reverse.KEYS)
    seen=[]
    components=("name","phonetic","address","digits","number_conflict","joint")
    for line in (root/"diagnostics.jsonl").open():
        row=json.loads(line)
        key=tuple(row[k] for k in reverse.KEYS)
        seen.append(key)
        features=expected.loc[key]
        assert key[0] not in row["rival_ids"]
        assert len(row["rival_ids"]) <= 3
        assert len(row["rival_ids"]) == len(row["rival_views"])
        assert row["shortlist_count"] <= 80 and row["rerank_count"] <= 8
        assert row["nonself_top8_count"] == row["rerank_count"]-int(row["self_excluded"])
        assert row["posting_count_nonself"] <= row["posting_count"]
        assert row["complete_postings"]
        for terms in row["selected_terms"].values():
            assert len(terms) <= 2
            assert terms == sorted(terms)
            cap=min(20_000,max(1,int(.02*row["country_records"])))
            assert all(0 < df <= cap for df,term in terms)
        if row["rival_views"]:
            np.testing.assert_allclose(features.iloc[6:12].to_numpy(),[row["rival_views"][0][c] for c in components],atol=1e-6)
        else:
            assert features.iloc[6:17].eq(-1).all()
        assert features.reverse_no_other_returned == float(not row["rival_ids"])
    assert seen == list(frame[reverse.KEYS].itertuples(index=False,name=None))
    result=dict(status="passed",rows=len(frame),routed_references=frame.source1_entity_id.nunique(),
                unique_target_ids=frame.candidate_entity_id.nunique(),distinct_text_queries=marker["unique_target_queries"],
                no_other_rows=int(frame.reverse_no_other_returned.sum()),no_query_rows=int((frame.reverse_query_fields_used==0).sum()),
                saturated_rows=int(frame.reverse_shortlist_saturated.sum()),seconds=marker["seconds"],
                source_tsv_sha256=marker["source_tsv_sha256"],feature_code_sha256=marker["feature_code_sha256"],
                labels_read=False)
    reverse.write_json(root/"verification.json",result)
    return result


if __name__ == "__main__":
    for directory in sys.argv[1:]: print(json.dumps(verify(directory)),flush=True)
