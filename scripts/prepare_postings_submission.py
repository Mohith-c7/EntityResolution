"""Freeze the tested postings pipeline without claiming a fresh holdout score."""
import argparse
import json
import shutil
import sqlite3
from pathlib import Path

from create_submission import digest, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifact-dir", type=Path, default=Path("models/features_v3_audit01"))
    p.add_argument("--submission-dir", type=Path, required=True)
    p.add_argument("--test-dir", type=Path, default=Path("dataset/test"))
    p.add_argument("--index-dir", type=Path, default=Path("models"))
    p.add_argument("--native-extension", type=Path, default=Path("models/runtime/erpostings.dylib"))
    args = p.parse_args()
    artifact, submission = args.artifact_dir, args.submission_dir
    if submission.exists(): raise FileExistsError("Use a fresh submission directory")
    selection = json.loads((artifact / "frozen_selection.json").read_text())
    for file, key in (("model.txt", "model_sha256"), ("name_aliases.json", "alias_sha256")):
        if digest(artifact / file) != selection[key]: raise ValueError("Verified model asset changed: " + file)
    reports = Path("reports/experiments/redesign")
    evidence = json.loads((reports / "feature_compatibility.json").read_text())
    if evidence["different_values"] or evidence["unchanged_feature_columns"] != 64 or not evidence["native_hash_matches_blob_predictions"]:
        raise ValueError("Feature or implementation compatibility failed")
    benchmark = json.loads((reports / "postings_test_10k.json").read_text())
    threshold = json.loads((reports / "wide_threshold_selection.json").read_text())["selection"]
    if not threshold or benchmark["references_per_second"] < 240:
        raise ValueError("Missing tuning selection or insufficient test throughput")
    indexes = {}
    for n, source in ((2, "S2"), (3, "S3")):
        raw = (args.test_dir / f"test_source{n}.tsv").resolve()
        path = (args.index_dir / f"index_test_{source}.sqlite").resolve()
        with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as conn:
            meta = json.loads(conn.execute("SELECT value FROM metadata WHERE key='build'").fetchone()[0])
        fp, stat = meta["fingerprint"], raw.stat()
        if not meta["complete"] or (fp["path"], fp["size"], fp["mtime_ns"], fp["source"]) != (str(raw), stat.st_size, stat.st_mtime_ns, source):
            raise ValueError("Index fingerprint mismatch: " + source)
        indexes[source] = meta
    package = submission / "code/business_entity_resolution"
    package.parent.mkdir(parents=True)
    shutil.copytree("code/business_entity_resolution", package, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (package / "native").mkdir(exist_ok=True)
    shutil.copyfile(args.native_extension, package / "native/erpostings.dylib")
    model = submission / "model"; model.mkdir()
    for filename in ("model.txt", "name_aliases.json", "feature_registry.json", "MODEL_LICENSE.txt",
                     "frozen_selection.json", "audit_report.json", "audit_protocol.json"):
        shutil.copyfile(artifact / filename, model / filename)
    report = json.loads((artifact / "report.json").read_text())
    report["previous_pipeline_threshold"] = report["threshold"]
    report["threshold"] = threshold["threshold"]
    report["experiment"] = "postings_v1_v3_matcher"
    report["pipeline_holdout_macro_f05"] = None
    report["pipeline_tuning_macro_f05"] = threshold["macro_f05"]
    report["retrieval"] = {"engine": "bounded-postings-v1", "options": benchmark["options"],
        "native_extension": "native/erpostings.dylib"}
    report["qualification"] = "Reuses verified v3 matcher weights; changed retrieval/threshold have tuning results only. The inherited holdout section applies only to the previous retrieval pipeline."
    write_json(model / "report.json", report)
    for filename in ("feature_compatibility.json", "postings_test_10k.json", "wide_threshold_selection.json", "postings_hash_wide_10k.json"):
        shutil.copyfile(reports / filename, submission / filename)
    shutil.copyfile("scripts/create_submission.py", submission / "create_submission.py")
    write_json(submission / "compatibility_report.json", {"compatible": True, "feature_schema": evidence,
        "model_sha256": selection["model_sha256"], "alias_sha256": selection["alias_sha256"],
        "test_indexes": indexes, "fresh_holdout_for_this_pipeline": None,
        "tuning": threshold, "leaderboard_score": None})
    frozen = {str(path.relative_to(submission)): digest(path) for path in sorted(submission.rglob("*")) if path.is_file()}
    write_json(submission / "frozen_assets_sha256.json", frozen)
    write_json(submission / "progress.json", {"status": "prepared", "ready_for_upload": False,
        "tuning_macro_f05": threshold["macro_f05"], "local_validation_macro_f05": None,
        "leaderboard_score": None, "threshold": threshold["threshold"]})
    print(json.dumps({"prepared": str(submission), "frozen_assets": len(frozen), "tuning": threshold}))


if __name__ == "__main__": main()
