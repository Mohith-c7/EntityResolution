"""Behavioral tests for retrieval, pruning, and model-input exports."""

import csv
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_CODE = Path(__file__).resolve().parents[1] / "code" / "business_entity_resolution"
sys.path.insert(0, str(PROJECT_CODE))

from src.blocking import BlockingConfig, CandidateGenerator, generate_scalable_candidates as generate_candidates
from src.blocking.cli import run_blocking
from src.blocking.contracts import BlockingRecord, records_from_dataframe
from src.blocking.identity_reranker import fused_score
from src.blocking.token_index import CorpusStatistics
from src.preprocessing.preprocess import preprocess_dataframe


def frame(source, values):
    rows = []
    for index, value in enumerate(values):
        name, address, country = value
        rows.append({"entity_id": f"{source}-{index + 1:03d}", "business_name": name,
                     "business_address": address, "country": country})
    return preprocess_dataframe(pd.DataFrame(rows, columns=("entity_id", "business_name", "business_address", "country")))


def targets():
    return frame("S2", []), frame("S3", [])


def test_character_path_recovers_typo_without_exact_name_token():
    s1 = frame("S1", [("Zenith", "", "France")])
    s2 = frame("S2", [("Zenitx", "", "France"), ("Unrelated", "", "France")])
    result = generate_candidates(s1, s2, targets()[1])
    assert result.candidate_entity_id.tolist() == ["S2-001"]
    assert "name_character" in result.iloc[0].blocking_paths
    assert "exact_name" not in result.iloc[0].blocking_paths


def test_empty_keys_do_not_create_matches():
    s1 = frame("S1", [("", "", "France")])
    s2 = frame("S2", [("", "", "France")])
    assert generate_candidates(s1, s2, targets()[1]).empty


def test_many_matches_and_per_source_caps():
    s1 = frame("S1", [("Atlas Print", "10 Oak Road", "Newlandia")])
    s2 = frame("S2", [("Atlas Print", "10 Oak Road", "Newlandia")] * 5)
    s3 = frame("S3", [("Atlas Print", "10 Oak Road", "Newlandia")] * 4)
    result = generate_candidates(s1, s2, s3, BlockingConfig(top_k=2))
    assert result.groupby("candidate_source").size().to_dict() == {"S2": 2, "S3": 2}
    assert result.rank_within_source.tolist() == [1, 2, 1, 2]
    assert not result.duplicated(["source1_entity_id", "candidate_entity_id"]).any()


def test_right_branch_survives_exact_block_truncation():
    s1 = frame("S1", [("Atlas", "910 Cedar Avenue", "France")])
    s2 = frame("S2", [("Atlas", "1 Birch Street", "France"), ("Atlas", "910 Cedar Avenue", "France")])
    result = generate_candidates(s1, s2, targets()[1], BlockingConfig(top_k=1, path_top_k=1))
    assert result.candidate_entity_id.tolist() == ["S2-002"]
    assert "exact_name" in result.iloc[0].blocking_paths


def test_shuffle_does_not_change_candidate_ids_scores_or_ranks():
    s1 = frame("S1", [("Atlas", "10 Oak Road", "France")])
    s2 = frame("S2", [("Atlas", "10 Oak Road", "France")] * 10)
    config = BlockingConfig(top_k=3, path_top_k=3)
    original = generate_candidates(s1, s2, targets()[1], config)
    shuffled = generate_candidates(s1, s2.sample(frac=1, random_state=7), targets()[1], config)
    pd.testing.assert_frame_equal(original, shuffled)


def test_missing_country_does_not_exclude_a_plausible_match():
    s1 = frame("S1", [("Atlas Print", "10 Oak Road", "France")])
    s2 = frame("S2", [("Atlas Print", "10 Oak Road", None)])
    assert generate_candidates(s1, s2, targets()[1]).candidate_entity_id.tolist() == ["S2-001"]


def test_numeric_tokens_preserve_zeros_and_field_provenance():
    mapping = {"entity_id": "S1-1", "norm_name": "studio 001", "name_core": "studio 001",
               "norm_address": "00123 oak road", "norm_country": "france",
               "name_digits": ("001",), "address_digits": ("00123",)}
    record = BlockingRecord.from_mapping(mapping, "S1")
    assert record.keys("digits", BlockingConfig()) == {"name:001", "address:00123"}
    assert record.postcode_candidates == {"00123"}
    mapping["postcode_candidates"] = ()
    assert not BlockingRecord.from_mapping(mapping, "S1").postcode_candidates


def test_explicit_postcode_name_path():
    s1 = frame("S1", [("Atlas", "North", "France")])
    s2 = frame("S2", [("Atlas", "South", "France")])
    s1["postcode_candidates"] = [("75001",)]
    s2["postcode_candidates"] = [("75001",)]
    result = generate_candidates(s1, s2, targets()[1])
    assert "postcode_name" in result.iloc[0].blocking_paths


def test_unicode_contract_preserves_supplied_text():
    row = {"entity_id": "S1-1", "name_norm": "दिल्ली café", "name_core": "दिल्ली café",
           "address_norm": "ଓଡ଼ିଶା", "country_norm": "newlandia"}
    record = BlockingRecord.from_mapping(row, "S1")
    assert record.name == "दिल्ली café"
    assert record.address == "ଓଡ଼ିଶା"
    assert BlockingRecord.from_dict(record.to_dict()) == record


def test_reranker_is_symmetric_for_the_same_corpus():
    left = records_from_dataframe(frame("S1", [("Atlas Print", "10 Oak Road", "France")]), "S1")[0]
    right = records_from_dataframe(frame("S2", [("Atlas Prints", "10 Oak Rd", "France")]), "S2")[0]
    stats = CorpusStatistics(BlockingConfig())
    stats.update([right])
    assert fused_score(left, right, ("rare_name",), stats) == fused_score(right, left, ("rare_name",), stats)


@pytest.mark.parametrize("updates", [{"top_k": 0}, {"top_k": 10, "path_top_k": 2},
                                     {"max_token_df_ratio": 0}, {"exact_name_bonus": float("nan")}])
def test_bad_configuration_is_rejected(updates):
    with pytest.raises(ValueError):
        BlockingConfig(**updates)


def test_duplicate_and_wrong_source_ids_are_rejected():
    s2 = frame("S2", [("Atlas", "", "France")])
    duplicate = pd.concat([s2, s2], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate"):
        CandidateGenerator(duplicate, targets()[1])
    s2.loc[0, "entity_id"] = "S1-wrong"
    with pytest.raises(ValueError, match="S2"):
        CandidateGenerator(s2, targets()[1])


def write_raw(path, source, rows):
    raw = frame(source, rows)[["entity_id", "business_name", "business_address", "country"]]
    raw.to_csv(path, sep="\t", index=False)


def cli_fixture(tmp_path):
    paths = [tmp_path / f"source{i}.tsv" for i in (1, 2, 3)]
    write_raw(paths[0], "S1", [("Atlas Print", "10 Oak Road", "France"), ("Quasar", "", "India"), ("", "", "Newlandia")])
    write_raw(paths[1], "S2", [("Atlas Print", "1 Birch Lane", "France"), ("Atlas Print", "10 Oak Road", "France")])
    write_raw(paths[2], "S3", [("Quasar", "", "India"), ("Atlas Print", "10 Oak Road", "France")])
    truth = tmp_path / "truth.tsv"
    truth.write_text("source1_entity_id\tmatched_entity_ids\nS1-001\tS2-002,S3-002\nS1-002\tS3-001\nS1-003\t\n")
    return paths, truth


def test_sharded_cli_exports_exact_model_input_and_all_reference_rows(tmp_path):
    paths, truth = cli_fixture(tmp_path)
    output = tmp_path / "output"
    report = run_blocking(*paths, output, BlockingConfig(top_k=1), shard_size=1, query_batch_size=2, ground_truth=truth)
    with (output / "candidate_pairs.tsv").open() as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    assert [row["source1_entity_id"] for row in rows] == ["S1-001", "S1-002", "S1-003"]
    assert rows[0]["candidate_entity_ids"] == "S2-002,S3-002"
    assert rows[-1]["candidate_entity_ids"] == ""
    scores = pd.read_csv(output / "candidate_scores.tsv", sep="\t")
    for row in rows:
        listed = set(row["candidate_entity_ids"].split(",")) - {""}
        actual = set(scores.loc[scores.source1_entity_id == row["source1_entity_id"], "candidate_entity_id"])
        assert actual == listed
    assert report["link_recall_after_top_k"] == 1.0
    assert report["oracle_macro_f05"] == 1.0
    assert report["matching_model_scored"] is False
    assert report["max_candidates_per_reference"] <= 2


def test_small_shards_and_unsharded_export_agree_on_fixture(tmp_path):
    paths, truth = cli_fixture(tmp_path)
    outputs = []
    for shard_size in (1, 100):
        output = tmp_path / str(shard_size)
        run_blocking(*paths, output, BlockingConfig(top_k=1), shard_size=shard_size, ground_truth=truth)
        outputs.append((output / "candidate_pairs.tsv").read_bytes())
    assert outputs[0] == outputs[1]


def test_cli_rejects_incomplete_truth_without_writing_exports(tmp_path):
    paths, truth = cli_fixture(tmp_path)
    truth.write_text("source1_entity_id\tmatched_entity_ids\nS1-001\tS2-002\n")
    with pytest.raises(ValueError, match="cover"):
        run_blocking(*paths, tmp_path / "out", ground_truth=truth)
    assert not (tmp_path / "out" / "candidate_pairs.tsv").exists()


def test_cross_process_exports_are_deterministic(tmp_path):
    paths, _ = cli_fixture(tmp_path)
    outputs = []
    for seed in ("1", "999"):
        import os
        output = tmp_path / seed
        command = [sys.executable, str(PROJECT_CODE / "run_blocking.py"),
                   "--source1", str(paths[0]), "--source2", str(paths[1]), "--source3", str(paths[2]),
                   "--output-dir", str(output), "--shard-size", "1", "--top-k", "1"]
        subprocess.run(command, check=True, capture_output=True, env={**os.environ, "PYTHONHASHSEED": seed})
        outputs.append((output / "candidate_scores.tsv").read_bytes())
    assert outputs[0] == outputs[1]
