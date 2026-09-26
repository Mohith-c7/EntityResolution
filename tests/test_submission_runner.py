"""Durable checkpoints and full finalization on a tiny offline fixture."""

import importlib.util
import fcntl
import json
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("submission_runner", ROOT / "scripts/create_submission.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
sys.path.insert(0, str(ROOT / "code/business_entity_resolution"))
from src.blocking.disk_index import DiskSearchConfig, build_disk_index
from src.features.registry import FEATURE_NAMES, FEATURE_VERSION


def test_checkpoint_preserves_empty_fields_and_commits_both_scored_sets(tmp_path):
    rows = [{"entity_id": "S1-A", "country": "France"}, {"entity_id": "S1-B", "country": "Newlandia"}]
    result = [("S1-A", ["S2-A", "S3-A"], ["S3-A"]), ("S1-B", [], [])]
    stats = runner.save_chunk(tmp_path, 0, rows, result)
    assert stats["scored_candidates"] == 2
    assert stats["predicted_links"] == 1
    assert stats["empty_predictions"] == 1
    assert stats["country"]["france"]["entities"] == 1
    assert (tmp_path / "0000000.matching.tsv").read_text() == "S1-A\tS3-A\nS1-B\t\n"
    assert (tmp_path / "0000000.candidate.tsv").read_text() == "S1-A\tS2-A,S3-A\nS1-B\t\n"
    assert json.loads((tmp_path / "0000000.json").read_text())["input_id_sha256"] == runner.id_digest(rows)
    with pytest.raises(ValueError, match="not a scored candidate"):
        runner.save_chunk(tmp_path, 1, rows[:1], [("S1-A", [], ["S2-X"])])
    assert not (tmp_path / "0000001.json").exists()


@pytest.mark.parametrize("postings", [False, True])
def test_frozen_inference_finalizes_validates_and_resumes_without_scoring(tmp_path, postings):
    submission = tmp_path / "output/submission_01"
    model = submission / "model"
    package = submission / "code/business_entity_resolution"
    model.mkdir(parents=True)
    shutil.copytree(ROOT / "code/business_entity_resolution", package,
        ignore=shutil.ignore_patterns("__pycache__", "tests"))
    test, indexes = tmp_path / "test", tmp_path / "indexes"
    test.mkdir(); indexes.mkdir()
    header = "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
    (test / "test_source1.tsv").write_text(header + "S1-A\tCafe\t00123 Rue\tFrance\nS1-B\tSolo\t42 Hill\tNewlandia\n")
    for n in (2, 3):
        path = test / f"test_source{n}.tsv"
        path.write_text(header + f"S{n}-A\tCafe\t00123 Rue\tFrance\n")
        build_disk_index(path, indexes / f"index_test_S{n}.sqlite", f"S{n}")
    x = np.zeros((4, 36), dtype=np.float32)
    ds = lgb.Dataset(x, label=[0, 1, 0, 1], feature_name=list(FEATURE_NAMES))
    booster = lgb.train({"objective": "binary", "verbosity": -1, "num_threads": 1}, ds, num_boost_round=1)
    booster.save_model(str(model / "model.txt"))
    report = {"experiment": "fixture_model", "search_config": asdict(DiskSearchConfig(character_mode="off")), "feature_version": FEATURE_VERSION,
        "threshold": .71, "holdout": {"macro_f05": .5}}
    if postings:
        from src.blocking.postings_index import PostingsConfig
        library = package / "native/erpostings.dylib"
        subprocess.run([sys.executable, str(ROOT / "scripts/build_postings_native.py"), "--output", str(library)], check=True)
        report.update(retrieval={"engine": "bounded-postings-v1", "options": asdict(PostingsConfig()),
            "native_extension": "native/erpostings.dylib"}, pipeline_holdout_macro_f05=None,
            pipeline_tuning_macro_f05=.5)
    (model / "report.json").write_text(json.dumps(report))
    (submission / "compatibility_report.json").write_text(json.dumps({"compatible": True}))
    (submission / "frozen_assets_sha256.json").write_text(json.dumps({"model/model.txt": runner.digest(model / "model.txt")}))
    command = [sys.executable, str(ROOT / "scripts/create_submission.py"), "--submission-dir", str(submission),
        "--test-dir", str(test), "--index-dir", str(indexes), "--workers", "1", "--batch-size", "1", "--mmap-bytes", "268435456"]
    with (submission / ".run.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        blocked = subprocess.run(command, capture_output=True, text=True)
        assert blocked.returncode != 0
        assert "already has an active runner" in blocked.stderr
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads((submission / "submission_report.json").read_text())
    assert summary["ready_for_upload"]
    assert summary["entities"] == 2
    assert summary["empty_predictions"] == 2
    assert summary["validation"] == {"strict": "PASS", "official": "PASS", "id_checking": True}
    assert summary["country"]["france"]["entities"] == 1
    assert summary["leaderboard_score"] is None
    assert summary["model"] == "fixture_model"
    assert summary["runtime_options"]["mmap_bytes"] == 268435456
    if postings:
        import pandas as pd
        frames = [pd.read_parquet(p) for p in (submission / "feature_cache").glob("*.parquet")]
        scored = pd.concat(frames)
        assert len(scored) == summary["scored_candidates"]
        assert set(scored.candidate_entity_id) == {"S2-A", "S3-A"}
        assert summary["local_validation_macro_f05"] is None
    expected = (submission / "matching_results.tsv").read_bytes()
    assert (submission.parent / "matching_results.tsv").read_bytes() == expected
    # Resumption validates each checkpoint and regenerates byte-identical outputs.
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert '"generated_this_run": 0' in result.stdout
    assert (submission / "matching_results.tsv").read_bytes() == expected
    if postings:
        # Cache evidence participates in the same resumability contract.
        cache_file = next((submission / "feature_cache").glob("*.parquet"))
        cache_file.write_bytes(b"corrupt")
        result = subprocess.run(command, capture_output=True, text=True)
        assert result.returncode != 0
        assert "corrupt scored feature cache" in result.stderr
