"""Safety properties of the experimental bounded retrieval path."""
import sys
import sqlite3
import subprocess
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code/business_entity_resolution"))
from src.blocking.disk_index import DiskSearchConfig, build_disk_index, normalize_record
from src.blocking.postings_index import PostingsConfig, PostingsSourceIndex


def test_complete_small_lists_skip_oversized_and_preserve_open_country(tmp_path):
    source = tmp_path / "source.tsv"
    source.write_text("entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        "S2-1\tCommon Alpha\t10 Elm\tFrance\n"
        "S2-2\tCommon Beta\t20 Pine\tNew Country\n"
        "S2-3\tCommon Gamma\t30 Cedar\tNew Country\n", encoding="utf-8")
    path = tmp_path / "source.sqlite"
    build_disk_index(source, path, "S2")
    index = PostingsSourceIndex(path, DiskSearchConfig(top_k=2, path_top_k=4, character_mode="off"),
        postings_config=PostingsConfig(max_postings=2, cache_bytes=1024))
    try:
        assert index.postings("name", "common")[0].tolist() == []
        assert index.profile["skipped_large_lists"] == 1
        assert index.postings("name", "gamma")[0].tolist() == [3]
        assert index.postings("name", "gamma")[0].tolist() == [3]
        assert index.profile["cache_hits"] == 1
        reference = normalize_record(dict(entity_id="S1-1", business_name="Common Gamma",
            business_address="30 Cedar", country="New Country"))
        pairs = index.query(reference)
        assert pairs[0][0].candidate_entity_id == "S2-3"
        assert len({c.candidate_entity_id for c, _ in pairs}) == len(pairs)
        assert [c.rank_within_source for c, _ in pairs] == list(range(1, len(pairs)+1))
    finally:
        index.close()


def test_shortlist_ties_are_deterministic_and_no_zero_padding():
    ids = np.array([8, 3, 9, 4])
    scores = np.array([1., 1., 0., 1.])
    positions = PostingsSourceIndex.top_positions(scores, 2, ids)
    assert sorted(ids[positions]) == [3, 4]
    assert len(PostingsSourceIndex.top_positions(np.zeros(4), 2, ids)) == 0
    with pytest.raises(ValueError):
        PostingsConfig(max_postings=0)


def test_native_transfer_and_cached_scoring_equal_reference(tmp_path):
    root = Path(__file__).resolve().parents[1]
    library = tmp_path / "erpostings.dylib"
    subprocess.run([sys.executable, str(root / "scripts/build_postings_native.py"),
        "--output", str(library)], check=True, capture_output=True)
    connection = sqlite3.connect(":memory:")
    connection.enable_load_extension(True)
    connection.load_extension(str(library))
    connection.enable_load_extension(False)
    assert np.frombuffer(connection.execute(
        "SELECT er_pack(x) FROM (SELECT 3 AS x UNION ALL SELECT 17 UNION ALL SELECT 1000000)"
    ).fetchone()[0], dtype=np.int32).tolist() == [3, 17, 1000000]
    assert connection.execute("SELECT er_pack(x) FROM (SELECT 1 AS x WHERE 0)").fetchone()[0] == b""
    with pytest.raises(sqlite3.OperationalError, match="positive int32"):
        connection.execute("SELECT er_pack(2147483648)").fetchone()
    connection.close()
    source = tmp_path / "source.tsv"
    source.write_text("entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        "S2-1\tÉcole Alpha\t10 Rue du Pont Paris\tFrance\n"
        "S2-2\tAlpha\t10 Rue du Port Lyon\tFrance\n"
        "S2-3\tभारत\t12 Main Street\tIndia\n", encoding="utf-8")
    path = tmp_path / "source.sqlite"
    build_disk_index(source, path, "S2")
    from src.blocking.disk_index import DiskSourceIndex
    config = DiskSearchConfig(top_k=3, path_top_k=4, character_mode="off", reranker_version="v4")
    pure = PostingsSourceIndex(path, config)
    native = PostingsSourceIndex(path, config, native_extension=library)
    base = DiskSourceIndex(path, config)
    try:
        left = normalize_record(dict(entity_id="S1-1", business_name="Ecole Alpha",
            business_address="10 Rue du Pont Paris", country="France"))
        assert native.query(left) == pure.query(left)
        for rowid in (1, 2, 3):
            right = base.record(rowid)
            assert base.rerank(left, right, ("rare_name",)) == native.rerank(left, right, ("rare_name",))
            from src.blocking.cheap_ranker import rank_features, RANK_FEATURES
            from src.blocking.contracts import Candidate
            from src.features.pairwise_features import build_versioned_pair_features
            from src.features.registry import FEATURE_VERSION_V3
            full = build_versioned_pair_features(left, right, Candidate(left.entity_id, right.entity_id, "S2", .5, ()), base, FEATURE_VERSION_V3)
            np.testing.assert_array_equal(np.asarray(rank_features(left, right, base), dtype=np.float32),
                np.asarray([full[name] for name in RANK_FEATURES], dtype=np.float32))
    finally:
        pure.close(); native.close(); base.close()
