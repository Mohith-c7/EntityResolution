import csv
import importlib.util
import json
from pathlib import Path
import sqlite3
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("reverse_competition", ROOT/"scripts/reverse_competition.py")
reverse = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = reverse
spec.loader.exec_module(reverse)


def source(tmp_path, rows, name="source.tsv"):
    path = tmp_path/name
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=reverse.SOURCE_COLUMNS, delimiter="\t")
        writer.writeheader(); writer.writerows(rows)
    return path


def raw(entity_id, name="zebranyx systems", address="2719 quuxlane", country="US"):
    return dict(entity_id=entity_id,business_name=name,business_address=address,country=country)


@pytest.fixture
def index(tmp_path):
    rows = [raw(f"S1-{i:04}",name=f"generic filler {i}",address="common road") for i in range(500)]
    rows[:6] = [raw(f"S1-{i:04}") for i in range(6)]
    rows.extend([raw(f"S1-fr-{i:04}",name=f"ordinary filler {i}",address="rue commune",country="France") for i in range(500)])
    rows[500] = raw("S1-fr-0000",country="France")
    rows[501] = raw("S1-fr-0001",country="France")
    path = source(tmp_path,rows)
    output = tmp_path/"train.sqlite"
    reverse.build_index(path,output,"train",expected_count=len(rows),expected_sha256=reverse.digest(path))
    instance = reverse.ReverseIndex(output,corpus="train")
    yield instance
    instance.close()


def test_reload_country_df_and_source_provenance(index):
    assert index.meta["records"] == 1000
    assert index.meta["country_counts"] == {"us":500,"france":500}
    target = reverse.clean_record(raw("S2-t"))
    own = index.records(["S1-0000"])["S1-0000"]
    before = reverse.pair_features(index,target,own)
    reloaded = reverse.ReverseIndex(index.path,corpus="train")
    assert reverse.pair_features(reloaded,target,own) == before
    reloaded.close()
    with pytest.raises(ValueError,match="corpus mismatch"):
        reverse.ReverseIndex(index.path,corpus="test")


def test_rank_one_self_is_excluded_and_best_three_remain(index):
    target = reverse.clean_record(raw("S2-t"))
    own = index.records(["S1-0000"])["S1-0000"]
    features, diagnostic = reverse.pair_features(index,target,own)
    assert diagnostic["rival_ids"] == ["S1-0001","S1-0002","S1-0003"]
    assert features["reverse_own_in_top8"] == 1.
    assert features["reverse_no_other_returned"] == 0.
    assert diagnostic["nonself_top8_count"] == 5
    assert diagnostic["posting_count_nonself"] == diagnostic["posting_count"]-1
    own2 = index.records(["S1-0001"])["S1-0001"]
    _, diagnostic2 = reverse.pair_features(index,target,own2)
    assert diagnostic2["rival_ids"] == ["S1-0000","S1-0002","S1-0003"]
    assert len(index.cache) == 1


def test_open_country_france_and_unseen_are_not_global(index):
    own = index.records(["S1-0000"])["S1-0000"]
    france = reverse.clean_record(raw("S2-fr",country="France"))
    features, diagnostic = reverse.pair_features(index,france,own)
    assert diagnostic["rival_ids"] == ["S1-fr-0000","S1-fr-0001"]
    assert features["reverse_own_country_conflict"] == 1.
    unseen = reverse.clean_record(raw("S2-new",country="arbitrary unseen country"))
    features, diagnostic = reverse.pair_features(index,unseen,own)
    assert not diagnostic["rival_ids"]
    assert diagnostic["country_unavailable"]
    assert diagnostic["no_query"] and diagnostic["no_result"]
    assert features["reverse_target_country_missing"] == 0.
    assert all(features[f] == -1 for f in reverse.FEATURE_NAMES[6:17])
    missing = reverse.clean_record(raw("S2-missing",country=""))
    _, diagnostic = reverse.pair_features(index,missing,own)
    assert diagnostic["country_records"] == 1000
    assert diagnostic["rival_ids"]


def test_no_rare_terms_is_missing_evidence_and_common_terms_are_skipped(index):
    own = index.records(["S1-0000"])["S1-0000"]
    target = reverse.clean_record(raw("S2-common",name="generic filler",address="common road"))
    features, diagnostic = reverse.pair_features(index,target,own)
    assert features["reverse_query_fields_used"] == 0.
    assert features["reverse_high_df_terms_skipped"] > 0
    assert features["reverse_no_other_returned"] == 1.
    assert diagnostic["no_query"] and diagnostic["complete_postings"]


def test_same_rival_components_and_canonical_number_semantics(index):
    assert reverse.canonical_numbers("００１２ gui1len hurtad0 22b") == {"12"}
    target = reverse.clean_record(raw("S2-t",name="alpha",address="0012 lane"))
    left = reverse.clean_record(raw("S1-a",name="alpha",address="99 road"))
    right = reverse.clean_record(raw("S1-b",name="beta",address="12 lane"))
    views_left, views_right = reverse.raw_views(target,left),reverse.raw_views(target,right)
    class FakeIndex:
        def query(self,target,own_id):
            assert own_id == "S1-own"
            return [(right,views_right),(left,views_left)],dict(selected_terms={f:[] for f in reverse.FIELDS},
                query_fields=0,high_df_skipped=0,shortlist_saturated=False,self_excluded=False,rerank_count=2,posting_count=0)
    own = reverse.clean_record(raw("S1-own",name="alpha",address="12 lane"))
    features, diagnostic = reverse.pair_features(FakeIndex(),target,own)
    assert features["reverse_other_name_sort"] == views_right["name"]
    assert features["reverse_other_address_sort"] == views_right["address"]
    assert features["reverse_other_name_sort"] < views_left["name"]
    assert views_left["number_conflict"] == 1.


def test_build_refuses_labels_wrong_hash_count_and_duplicate_ids(tmp_path):
    path=source(tmp_path,[raw("S1-a")])
    with pytest.raises(ValueError,match="SHA256"):
        reverse.build_index(path,tmp_path/"bad.sqlite","train",expected_count=1,expected_sha256="bad")
    with pytest.raises(ValueError,match="row count"):
        reverse.build_index(path,tmp_path/"count.sqlite","train",expected_count=2)
    assert not (tmp_path/"count.sqlite").exists()
    path=source(tmp_path,[raw("S1-a"),raw("S1-a")],name="duplicate.tsv")
    with pytest.raises(sqlite3.IntegrityError):
        reverse.build_index(path,tmp_path/"duplicate.sqlite","train",expected_count=2)
    with pytest.raises(ValueError,match="Corpus"):
        reverse.build_index(path,tmp_path/"mixed.sqlite","combined",expected_count=2)


def test_ftsvocab_and_country_df_match_preserved_unicode(tmp_path):
    rows=[raw(f"S1-{i}",name=f"generic {i}",address="common road") for i in range(500)]
    rows[0]=raw("S1-0",name="École भारत",address="१२ rue")
    path=source(tmp_path,rows)
    output=tmp_path/"unicode.sqlite"
    reverse.build_index(path,output,"train",expected_count=len(rows))
    index=reverse.ReverseIndex(output)
    try:
        target=reverse.clean_record(raw("S2-u",name="École भारत",address="१२ rue"))
        assert "भारत" in target["name"]
        rivals,coverage=index.query(target,"S1-absent")
        assert rivals[0][0]["id"] == "S1-0"
        assert coverage["complete_postings"]
    finally: index.close()


def test_keyed_export_pinned_chunk_parity_and_labels_are_unread(index, tmp_path):
    import pandas as pd
    sys.path.insert(0,str(ROOT/"scripts"))
    import train_sibling_matcher as sibling
    frame=pd.DataFrame([{**dict(zip(reverse.KEYS,(f"S1-{i:04}",f"S2-{i}-{j}"))),
                         "probability":p,"first_stage":p,"label":j%2}
                        for i in range(10) for j,p in enumerate((.999,.7,.2))])
    target_source=source(tmp_path,[raw("S2-t")],name="target_source.tsv")
    prefix=tmp_path/"target"
    for source_name in ("S2","S3"):
        with sqlite3.connect(str(prefix)+"_"+source_name+".sqlite") as conn:
            conn.execute("CREATE TABLE records(id TEXT,name TEXT,address TEXT,country TEXT)")
            if source_name == "S2":
                conn.executemany("INSERT INTO records VALUES(?,?,?,?)",[(target,"zebranyx systems","2719 quuxlane","us") for target in frame.candidate_entity_id])
            conn.execute("CREATE TABLE metadata(key TEXT,value TEXT)")
            conn.execute("INSERT INTO metadata VALUES('build',?)",(json.dumps(dict(complete=True,fingerprint=dict(path=str(target_source)))),))
    route=sibling.select_route(frame)
    route_dir=tmp_path/"pinned";route_dir.mkdir()
    sibling.route_edges(frame,route).to_parquet(route_dir/"route.parquet",index=False)
    global_hash=reverse.pair_order_hash(frame)
    reverse.write_json(route_dir/"manifest.json",dict(status="complete",route=sibling.asdict(sibling.RouteConfig()),
        global_references=10,global_pair_order_sha256=global_hash,route_sha256=reverse.digest(route_dir/"route.parquet")))
    def export(data,directory):
        path=tmp_path/(directory+"_pairs.parquet");data.to_parquet(path,index=False)
        manifest_path=tmp_path/(directory+"_manifest.json")
        reverse.write_json(manifest_path,dict(status="complete",verified_current_pipeline=True,split="development",
            pairs_sha256=reverse.digest(path),pair_order_sha256=reverse.pair_order_hash(data),routing_universe_pair_order_sha256=global_hash))
        marker=reverse.generate_features(path,manifest_path,index.path,prefix,tmp_path/directory,route_dir=route_dir)
        result=pd.read_parquet(tmp_path/directory/"features.parquet")
        assert marker["labels_read"] is False
        assert marker["diagnostics_sha256"] == reverse.digest(tmp_path/directory/"diagnostics.jsonl")
        return result
    full=export(frame,"full")
    owners=full.source1_entity_id.unique()
    subset=frame[frame.source1_entity_id == owners[0]].copy().reset_index(drop=True)
    subset.label=1-subset.label
    chunk=export(subset,"chunk")
    pd.testing.assert_frame_equal(chunk,full[full.source1_entity_id == owners[0]].reset_index(drop=True))
    assert set(map(tuple,full[reverse.KEYS].to_numpy())) == set(map(tuple,frame.iloc[sorted(route)][reverse.KEYS].to_numpy()))
    bad=subset.iloc[:-1].reset_index(drop=True)
    with pytest.raises(ValueError,match="complete reference candidates"):
        export(bad,"missing_peer")
