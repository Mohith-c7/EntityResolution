"""Statistics must count documents in the same view queried by alignment."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code/business_entity_resolution"))
spec = importlib.util.spec_from_file_location("alignment_builder", ROOT / "scripts/build_token_alignment_features.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)

from src.features.evidence import phonetic
from src.features.token_alignment import AlignmentStatistics, FEATURE_NAMES, STATISTICS_VERSION, _alignment


def source(path, names, addresses=None):
    addresses = addresses or ["10 Élysée Road"] * len(names)
    pd.DataFrame({"entity_id": [f"S1-{i}" for i in range(len(names))],
                  "business_name": names, "business_address": addresses,
                  "country": ["Freedonia"] * len(names)}).to_csv(path, sep="\t", index=False)
    return path


def test_common_accented_street_uses_folded_document_frequency(tmp_path):
    path = source(tmp_path / "source1.tsv", ["Café Services"] * 100)
    stats, _ = builder.build_statistics(path)
    assert stats.df("elysee", "freedonia") == 100
    assert stats.idf("elysee", "freedonia") == 1.0
    assert stats.df("cafe", "freedonia") == 100
    assert stats.df("élysee", "freedonia") == 0
    assert stats.idf("never_observed", "freedonia") > 4.0


def test_phonetic_frequency_counts_document_union_not_sum(tmp_path):
    # Both primary tokens transform to one token; one document counts once.
    assert phonetic("Cece") == phonetic("Keke")
    path = source(tmp_path / "source1.tsv", ["Cece Keke Services"] * 100)
    stats, _ = builder.build_statistics(path)
    token = phonetic("Cece")
    assert stats.df(token, "freedonia", "phonetic") == 100
    assert stats.df(phonetic("Services"), "freedonia", "phonetic") == 100
    assert stats.idf(phonetic("Services"), "freedonia", "phonetic") == 1.0
    assert stats.idf(token, "unknown_country", "phonetic") == 1.0


def test_reused_statistics_reject_changed_source_but_accept_relocated_identical_source(tmp_path):
    path = source(tmp_path / "source1.tsv", ["Café Services"])
    stats, _ = builder.build_statistics(path)
    builder.verify_statistics_source(stats, path)
    copy = tmp_path / "relocated.tsv"
    copy.write_bytes(path.read_bytes())
    builder.verify_statistics_source(stats, copy)
    source(copy, ["Other Corp"])
    with pytest.raises(ValueError, match="Source 1 hash"):
        builder.verify_statistics_source(stats, copy)


def test_statistics_roundtrip_preserves_both_views(tmp_path):
    path = source(tmp_path / "source1.tsv", ["Café Services", "Keke"])
    stats, _ = builder.build_statistics(path)
    restored = AlignmentStatistics.from_dict(stats.to_dict())
    assert restored == stats
    assert restored.version == STATISTICS_VERSION
    legacy = stats.to_dict()
    legacy["version"] = "token-alignment-stats-v1"
    with pytest.raises(ValueError, match="rebuild v2"):
        AlignmentStatistics.from_dict(legacy)


def test_fuzzy_cap_keeps_most_distinctive_token_and_is_order_invariant():
    left = [f"a{i}" for i in range(8)] + ["zephyr"]
    weight = lambda token: 9.0 if token.startswith("zeph") else 1.0
    forward = _alignment(left, ["zephir"], weight)
    backward = _alignment(list(reversed(left)), ["zephir"], weight)
    assert forward == backward
    assert forward[0] > 0.0
    assert forward[2] > 3.0


def test_typed_statistics_without_phonetic_view_fail_explicitly():
    stats = AlignmentStatistics(STATISTICS_VERSION, 1, {"services": 1})
    with pytest.raises(ValueError, match="Phonetic document frequencies are required"):
        stats.idf("servikes", "", "phonetic")


def test_batch_reuse_roundtrip_and_wrong_corpus_rejection(tmp_path):
    left = source(tmp_path / "source1.tsv", ["Café Services"])
    paths = []
    for target in ("S2", "S3"):
        path = source(tmp_path / (target + ".tsv"), ["Café Services"])
        data = pd.read_csv(path, sep="\t", dtype=str)
        data["entity_id"] = [target + "-0"]
        data.to_csv(path, sep="\t", index=False)
        paths.append(path)
    pairs = tmp_path / "pairs.parquet"
    pd.DataFrame({"source1_entity_id": ["S1-0", "S1-0"],
                  "candidate_entity_id": ["S2-0", "S3-0"]}).to_parquet(pairs, index=False)
    stats, _ = builder.build_statistics(left)
    stats_path = tmp_path / "stats.json"
    stats_path.write_text(json.dumps(stats.to_dict()))
    output = tmp_path / "features.parquet"
    command = [sys.executable, str(ROOT / "scripts/build_token_alignment_features.py"),
               "--pairs", str(pairs), "--source1", str(left), "--source2", str(paths[0]),
               "--source3", str(paths[1]), "--statistics", str(stats_path), "--output", str(output)]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    features = pd.read_parquet(output)
    assert list(features) == builder.PAIR_KEYS + FEATURE_NAMES
    assert features[builder.PAIR_KEYS].equals(pd.read_parquet(pairs))
    assert not features[FEATURE_NAMES].isna().any().any()
    # The command itself must reject reused statistics before any new output.
    source(left, ["Different Business"])
    rejected_output = tmp_path / "rejected.parquet"
    command[-1] = str(rejected_output)
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode != 0 and "Source 1 hash" in result.stderr
    assert not rejected_output.exists()
