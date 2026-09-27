#!/usr/bin/env python3
"""R1 release: reuse verified stored scores, append vetoes, decide globally.

This script never scores models, retrieves candidates, edits baseline artifacts,
repairs checkpoints, or publishes to a shared output filename. Build requires a
coordinator-frozen release descriptor and a new output directory.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import sqlite3
import subprocess
import sys
import time
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PAIR_KEYS = ["source1_entity_id", "candidate_entity_id"]
SCORE_COLUMNS = ["first_stage", "second_stage", "probability"]
VERSION = "stored-score-release-v1"


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def ids_digest(ids):
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


def pair_order_digest(frame):
    value = hashlib.sha256()
    for s, c in frame[PAIR_KEYS].itertuples(index=False, name=None):
        value.update((s + "\t" + c + "\n").encode())
    return value.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def read_source1(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    ids = [row["entity_id"] for row in rows]
    if any(not value or not value.startswith("S1-") or value != value.strip() for value in ids):
        raise ValueError("Malformed Source 1 IDs")
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate Source 1 IDs")
    return rows


def resolve_evidence(evidence, root):
    path = Path(evidence["path"])
    if not path.is_absolute():
        path = Path(root) / path
    if digest(path) != evidence["sha256"]:
        raise ValueError(f"Input/asset hash mismatch: {path}")
    return path


def validate_pairs(frame):
    required = PAIR_KEYS + SCORE_COLUMNS
    if not set(required) <= set(frame):
        raise ValueError(f"Missing score columns: {sorted(set(required) - set(frame))}")
    for column, prefix in zip(PAIR_KEYS, ("S1-", None)):
        values = frame[column]
        if values.isna().any() or not values.map(lambda x: isinstance(x, str) and x == x.strip() and bool(x)).all():
            raise ValueError("Malformed pair keys")
        if prefix and not values.str.startswith(prefix).all():
            raise ValueError("Invalid Source 1 pair key")
    if not frame.candidate_entity_id.str.startswith(("S2-", "S3-")).all():
        raise ValueError("Invalid candidate prefix")
    if frame.duplicated(PAIR_KEYS).any():
        raise ValueError("Duplicate pair keys")
    values = frame[SCORE_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Stored scores must be finite probabilities")


def load_verified_chunks(manifest, test_dir, root=ROOT):
    """Each original input slice includes zero-candidate owners explicitly."""
    if manifest.get("version") != VERSION:
        raise ValueError("Unsupported reuse manifest")
    for n in (1, 2, 3):
        evidence = manifest["test_files"][f"test_source{n}.tsv"]
        if digest(Path(test_dir) / f"test_source{n}.tsv") != evidence["sha256"]:
            raise ValueError("Test input hash mismatch")
    for evidence in manifest.get("assets", []):
        resolve_evidence(evidence, root)
    if "source1_universe" in manifest:
        resolve_evidence(manifest["source1_universe"], root)
    frozen_path = resolve_evidence(manifest["baseline_frozen"], root)
    report_path = resolve_evidence(manifest["baseline_report"], root)
    baseline = json.loads(report_path.read_text())
    validation = baseline.get("validation", {})
    validated = (validation.get("strict_issues") == [] and validation.get("official_returncode") == 0
                 or validation.get("strict") == validation.get("official") == "PASS" and validation.get("id_checking") is True)
    if (baseline.get("status") != "complete" or not validated
            or baseline.get("frozen_sha256") != digest(frozen_path)
            or baseline.get("entities") != manifest["entities"]
            or baseline.get("scored_candidates") != manifest["pairs"]):
        raise ValueError("Baseline is incomplete, unvalidated, or has different lineage")
    rows = read_source1(Path(test_dir) / "test_source1.tsv")
    if len(rows) != manifest["entities"] or ids_digest([r["entity_id"] for r in rows]) != manifest["input_ids_sha256"]:
        raise ValueError("Input order/count mismatch")
    frames, cursor = [], 0
    for number, evidence in enumerate(manifest["chunks"]):
        if evidence["chunk"] != number:
            raise ValueError("Chunks must be complete and contiguous")
        marker_path = resolve_evidence(evidence["marker"], root)
        parquet_path = resolve_evidence(evidence["parquet"], root)
        marker = json.loads(marker_path.read_text())
        chunk_rows = rows[cursor:cursor + marker["entities"]]
        chunk_ids = [r["entity_id"] for r in chunk_rows]
        if (marker["chunk"] != number or marker["entities"] <= 0 or len(chunk_rows) != marker["entities"]
                or marker["input_ids_sha256"] != ids_digest(chunk_ids)
                or marker["parquet_sha256"] != evidence["parquet"]["sha256"]):
            raise ValueError("Chunk input/order/hash mismatch")
        frame = pd.read_parquet(parquet_path)
        validate_pairs(frame)
        if len(frame) != marker["pairs"]:
            raise ValueError("Chunk pair count mismatch")
        ranks = {value: position for position, value in enumerate(chunk_ids)}
        positions = frame.source1_entity_id.map(ranks)
        if positions.isna().any() or not positions.is_monotonic_increasing:
            raise ValueError("Chunk pair order/reference membership mismatch")
        order = frame.groupby("source1_entity_id", sort=False).cumcount().to_numpy()
        if "candidate_order" in frame and not np.array_equal(frame.candidate_order.to_numpy(), order):
            raise ValueError("Stored candidate order mismatch")
        frame["candidate_order"] = order
        if pair_order_digest(frame) != evidence["pair_order_sha256"]:
            raise ValueError("Chunk pair order hash mismatch")
        frames.append(frame)
        cursor += len(chunk_rows)
    if cursor != len(rows):
        raise ValueError("Incomplete Source 1 chunk coverage")
    frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=PAIR_KEYS + SCORE_COLUMNS + ["candidate_order"])
    validate_pairs(frame)
    if len(frame) != manifest["pairs"] or pair_order_digest(frame) != manifest["pair_order_sha256"]:
        raise ValueError("Complete pair count/order hash mismatch")
    return rows, frame


def seal_reuse_manifest(baseline_dir, test_dir, frozen_path, asset_manifest, output, root=ROOT):
    """Read-only baseline inspection; only writes a new manifest outside it.

    Call after the baseline releases the machine. asset_manifest is a nonempty
    path -> known SHA-256 map for the immutable indexes and native library.
    Fresh checksums establish unchanged file identity, not historical lineage;
    the caller must preserve the original asset snapshot evidence separately.
    """
    baseline_dir, test_dir, frozen_path, output = map(Path, (baseline_dir, test_dir, frozen_path, output))
    if output.exists() or baseline_dir.resolve() in output.resolve().parents:
        raise ValueError("Reuse manifest must be a new file outside preserved baseline")
    if not asset_manifest:
        raise ValueError("Index/native asset evidence is required for exact reuse")
    frozen = json.loads(frozen_path.read_text())
    report_path = baseline_dir / "submission_report.json"
    report = json.loads(report_path.read_text())
    validation = report.get("validation", {})
    if (report.get("status") != "complete" or validation.get("strict_issues") != []
            or validation.get("official_returncode") != 0 or report.get("frozen_sha256") != digest(frozen_path)):
        raise ValueError("Baseline must be complete and both ID validators passed")
    rows = read_source1(test_dir / "test_source1.tsv")
    with sqlite3.connect(f"file:{(baseline_dir / 'universe_counts_test.sqlite').resolve()}?mode=ro", uri=True) as conn:
        raw = conn.execute("SELECT value FROM meta WHERE key='universe'").fetchone()
    universe = json.loads(raw[0]) if raw else {}
    if universe.get("sha256") != digest(test_dir / "test_source1.tsv") or universe.get("records") != len(rows):
        raise ValueError("Baseline Source 1 universe hash/count mismatch")
    assets = dict(asset_manifest)
    assets.update(frozen.get("code_sha256", {}))
    for key in ("first_stage", "second_stage", "aliases", "rules_source"):
        item = frozen[key]; assets[item["path"]] = item["sha256"]
    assets[frozen["bridge"]["ranking_model"]] = frozen["bridge"]["ranking_model_sha256"]
    for path, sha256 in assets.items():
        resolve_evidence({"path": path, "sha256": sha256}, root)
    evidence = lambda path: {"path": str(path.resolve()), "sha256": digest(path)}
    markers = sorted((baseline_dir / "chunks").glob("*.json"))
    chunks, cursor, pairs = [], 0, 0
    order_hash = hashlib.sha256()
    candidate_hash = hashlib.sha256(b"source1_entity_id\tcandidate_entity_ids\n")
    for number, path in enumerate(markers):
        if path.stem != f"{number:07d}":
            raise ValueError("Incomplete/noncontiguous baseline chunks")
        marker = json.loads(path.read_text())
        part = rows[cursor:cursor + marker["entities"]]
        part_ids = [row["entity_id"] for row in part]
        parquet = path.with_suffix(".parquet")
        if (marker["chunk"] != number or marker["entities"] <= 0 or len(part) != marker["entities"]
                or marker["input_ids_sha256"] != ids_digest(part_ids)
                or marker["parquet_sha256"] != digest(parquet)):
            raise ValueError("Baseline chunk input/count/hash mismatch")
        frame = pd.read_parquet(parquet)
        validate_pairs(frame)
        positions = frame.source1_entity_id.map({eid: i for i, eid in enumerate(part_ids)})
        if len(frame) != marker["pairs"] or positions.isna().any() or not positions.is_monotonic_increasing:
            raise ValueError("Baseline chunk pair count/order/membership mismatch")
        for s, c in frame[PAIR_KEYS].itertuples(index=False, name=None):
            order_hash.update((s + "\t" + c + "\n").encode())
        grouped = frame.groupby("source1_entity_id", sort=False).candidate_entity_id.agg(list).to_dict()
        for row in part:
            s = row["entity_id"]
            candidate_hash.update((s + "\t" + ",".join(grouped.get(s, [])) + "\n").encode())
        chunks.append({"chunk": number, "marker": evidence(path), "parquet": evidence(parquet),
                       "pair_order_sha256": pair_order_digest(frame)})
        cursor += len(part); pairs += len(frame)
    if cursor != len(rows) or report.get("entities") != cursor or report.get("scored_candidates") != pairs:
        raise ValueError("Baseline full input/pair coverage mismatch")
    for name, expected in report["output_sha256"].items():
        if digest(baseline_dir / name) != expected:
            raise ValueError("Completed baseline output hash mismatch")
    if candidate_hash.hexdigest() != report["output_sha256"]["candidate_pairs.tsv"]:
        raise ValueError("Stored chunks do not reproduce the exact baseline candidate export")
    result = {"version": VERSION, "entities": len(rows), "pairs": pairs,
              "input_ids_sha256": ids_digest([row["entity_id"] for row in rows]),
              "pair_order_sha256": order_hash.hexdigest(), "chunks": chunks,
              "baseline_frozen": evidence(frozen_path), "baseline_report": evidence(report_path),
              "test_files": {f"test_source{n}.tsv": evidence(test_dir / f"test_source{n}.tsv") for n in (1, 2, 3)},
              "assets": [{"path": path, "sha256": sha256} for path, sha256 in sorted(assets.items())],
              "source1_universe": evidence(baseline_dir / "universe_counts_test.sqlite")}
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    return result


def attach_vetoes(frame, gates, expected_order_hash):
    """Pair keys control alignment. Equal length or accidental row order cannot."""
    if pair_order_digest(frame) != expected_order_hash:
        raise ValueError("Gate candidate universe/order hash mismatch")
    if not set(PAIR_KEYS + ["gate_veto", "gate_reason"]) <= set(gates):
        raise ValueError("Gate table missing pair keys/veto/reason")
    if gates[PAIR_KEYS].isna().any().any() or gates.duplicated(PAIR_KEYS).any():
        raise ValueError("Duplicate/missing gate pair keys")
    if gates.gate_veto.isna().any() or not gates.gate_veto.map(lambda v: isinstance(v, (bool, np.bool_))).all():
        raise ValueError("Gate vetoes must be booleans")
    if gates.gate_reason.isna().any() or not gates.gate_reason.map(lambda v: isinstance(v, str)).all():
        raise ValueError("Gate reasons must be strings")
    left_keys = pd.MultiIndex.from_frame(frame[PAIR_KEYS])
    indexed = gates.set_index(PAIR_KEYS)
    if len(gates) != len(frame) or not left_keys.isin(indexed.index).all():
        raise ValueError("Gate pair universe mismatch")
    result = frame.copy()
    aligned = indexed.loc[left_keys]
    result["gate_veto"] = aligned.gate_veto.to_numpy()
    result["gate_reason"] = aligned.gate_reason.to_numpy()
    return result


def load_original_decide(runner_path):
    """Execute only the hash-verified original pure decision function."""
    tree = ast.parse(Path(runner_path).read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "decide"]
    if len(nodes) != 1:
        raise ValueError("Frozen runner has no unique decide function")
    namespace = {"np": np, "pd": pd}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(runner_path), "exec"), namespace)
    return namespace["decide"]


def make_decisions(frame, release, root=ROOT):
    """One complete claimant universe; base probabilities remain unchanged."""
    if release["policy"] == "original":
        decide = load_original_decide(resolve_evidence(release["original_runner"], root))
        effective = frame.copy()
        # Preserve all base columns in frame; only a transient eligibility view
        # removes vetoed claims before global incumbent arbitration/rescue.
        if "gate_veto" in effective:
            effective.loc[effective.gate_veto, "probability"] = -1.0
        if len(effective):
            chosen, lost = decide(effective, effective, release["decision_config"])
        else:
            chosen = lost = np.zeros(0, dtype=bool)
        details = {"policy_version": "original", "accepted_links": int(chosen.sum())}
    elif release["policy"] == "ownership_v2":
        sys.path.insert(0, str(ROOT / "scripts"))
        from decision_policy_v2 import decide_v2
        chosen, lost, diagnostics = decide_v2(frame, frame, release["decision_config"])
        details = diagnostics["summary"]
    else:
        raise ValueError("Unknown decision policy")
    chosen, lost = np.asarray(chosen, dtype=bool), np.asarray(lost, dtype=bool)
    if len(chosen) != len(frame) or len(lost) != len(frame):
        raise ValueError("Decision mask alignment mismatch")
    if "gate_veto" in frame and (chosen & frame.gate_veto.to_numpy()).any():
        raise ValueError("A vetoed pair was accepted")
    return chosen, lost, details


def export_outputs(rows, frame, chosen, output):
    candidates, matches = defaultdict(list), defaultdict(list)
    for s, c, keep in zip(frame.source1_entity_id, frame.candidate_entity_id, chosen):
        candidates[s].append(c)
        if keep:
            matches[s].append(c)
    country, histogram = {}, Counter()
    for row in rows:
        s, label = row["entity_id"], row.get("country", "").strip().casefold()
        state = country.setdefault(label, Counter())
        state.update(entities=1, scored_candidates=len(candidates[s]), predicted_links=len(matches[s]),
                     empty_predictions=int(not matches[s]), empty_candidates=int(not candidates[s]))
        histogram[len(candidates[s])] += 1
    for name, column, values in (("candidate_pairs.tsv", "candidate_entity_ids", candidates),
                                  ("matching_results.tsv", "matched_entity_ids", matches)):
        with (Path(output) / name).open("w", encoding="utf-8", newline="") as stream:
            stream.write("source1_entity_id\t" + column + "\n")
            for row in rows:
                stream.write(row["entity_id"] + "\t" + ",".join(values[row["entity_id"]]) + "\n")
    return {"country": country, "candidate_histogram": histogram,
            "empty_predictions": sum(v["empty_predictions"] for v in country.values())}


def validate_outputs(output, test_dir, strict_path=None, official_path=None):
    strict_start = time.perf_counter()
    strict_path = strict_path or ROOT / "code/business_entity_resolution/src/pipeline/export.py"
    official_path = official_path or ROOT / "code/business_entity_resolution/src/utils/organizer_validate_submission.py"
    import importlib.util
    spec = importlib.util.spec_from_file_location("release_strict_validator", strict_path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    matching, candidate = Path(output) / "matching_results.tsv", Path(output) / "candidate_pairs.tsv"
    issues = module.validate_submission(matching, candidate, Path(test_dir))
    write_json(Path(output) / "strict_validation.json", {"passed": not issues, "id_checking": True, "issues": issues})
    strict_seconds = time.perf_counter() - strict_start
    command = [sys.executable, "-S", str(official_path), "--matching", str(matching), "--candidate", str(candidate),
               "--test-dir", str(test_dir), "--check-ids"]
    official_start = time.perf_counter()
    with (Path(output) / "official_validation.log").open("w") as stream:
        returncode = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT).returncode
    official_seconds = time.perf_counter() - official_start
    result = {"strict": "PASS" if not issues else "FAIL", "official": "PASS" if not returncode else "FAIL",
              "strict_issues": issues, "official_returncode": returncode, "id_checking": True,
              "official_command": command, "validator_sha256": {"strict": digest(strict_path), "official": digest(official_path)},
              "stage_seconds": {"strict_validator": strict_seconds, "official_validator": official_seconds}}
    write_json(Path(output) / "validation.json", result)
    return result


def build(release_path, test_dir, output, root=ROOT):
    start = time.perf_counter()
    release_path, output = Path(release_path).resolve(), Path(output).resolve()
    release = json.loads(release_path.read_text())
    if release.get("status") != "candidate_frozen" or release.get("mode") != "R1" or not release.get("frozen_at"):
        raise ValueError("Build requires a coordinator-frozen R1 release")
    # These paths are never writable build destinations, including their children.
    forbidden = [Path(root) / "output" / name for name in ("submission_03", "submission_04")]
    if any(output == p.resolve() or p.resolve() in output.parents for p in forbidden):
        raise ValueError("Preserved baseline output directory is not a release destination")
    if output.exists():
        raise ValueError("Release output directory must be new/versioned")
    manifest_path = resolve_evidence(release["reuse_manifest"], root)
    pinned_code = {str((Path(root) / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()): sha
                   for path, sha in release["code_sha256"].items()}
    if pinned_code.get(str(Path(__file__).resolve())) != digest(__file__):
        raise ValueError("Release must pin this exact exporter implementation")
    if release["policy"] == "ownership_v2" and pinned_code.get(str(ROOT / "scripts/decision_policy_v2.py")) != digest(ROOT / "scripts/decision_policy_v2.py"):
        raise ValueError("Release must pin the selected ownership implementation")
    for validator in (ROOT / "code/business_entity_resolution/src/pipeline/export.py",
                      ROOT / "code/business_entity_resolution/src/utils/organizer_validate_submission.py"):
        if pinned_code.get(str(validator)) != digest(validator):
            raise ValueError("Release must pin both exact ID validator implementations")
    for path, sha256 in release["code_sha256"].items():
        resolve_evidence({"path": path, "sha256": sha256}, root)
    if release.get("candidate_freeze"):
        candidate_path = resolve_evidence(release["candidate_freeze"], root)
        candidate = json.loads(candidate_path.read_text())
        if candidate.get("status") != "candidate_frozen":
            raise ValueError("Referenced audit candidate is not frozen")
        for key in ("frozen_at", "parent_frozen_sha256", "policy", "decision_config", "gate_config"):
            if key in candidate and candidate[key] != release.get(key):
                raise ValueError(f"Release differs from audited candidate: {key}")
        candidate_code = {str((Path(root) / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()): sha
                          for path, sha in candidate["code_sha256"].items()}
        if any(candidate_code.get(path) != sha for path, sha in pinned_code.items()):
            raise ValueError("Release implementation is absent/different in audited candidate freeze")
        for group in ("model_sha256", "schema_sha256", "index_sha256", "native_sha256"):
            for path, sha in candidate.get(group, {}).items():
                resolve_evidence({"path": path, "sha256": sha}, root)
    manifest = json.loads(manifest_path.read_text())
    if release.get("parent_frozen_sha256") != manifest["baseline_frozen"]["sha256"]:
        raise ValueError("Release parent frozen lineage mismatch")
    if release["policy"] == "original":
        baseline_config = json.loads(resolve_evidence(manifest["baseline_frozen"], root).read_text())
        runner_sha = baseline_config.get("code_sha256", {}).get("scripts/run_frozen_pipeline.py")
        if not runner_sha or release["original_runner"]["sha256"] != runner_sha:
            raise ValueError("Original decision runner differs from the frozen baseline")
    descriptor_verified = time.perf_counter()
    rows, base = load_verified_chunks(manifest, test_dir, root)
    scores_loaded = time.perf_counter()
    frame = base.copy()
    if release.get("gate_table"):
        gate = release["gate_table"]
        frame = attach_vetoes(frame, pd.read_parquet(resolve_evidence(gate, root)), gate["pair_order_sha256"])
    gate_aligned = time.perf_counter()
    chosen, lost, decision = make_decisions(frame, release, root)
    if not frame[SCORE_COLUMNS].equals(base[SCORE_COLUMNS]):
        raise ValueError("Gate/decision changed stored base scores")
    decided = time.perf_counter()
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(release_path, output / "release.json")
    shutil.copy2(manifest_path, output / "reuse_manifest.json")
    statistics = export_outputs(rows, frame, chosen, output)
    exported = time.perf_counter()
    # Pair-keyed decisions preserve original scores plus a separate veto/mask.
    diagnostics = frame.copy()
    diagnostics["accepted"], diagnostics["ownership_lost"] = chosen, lost
    diagnostics.to_parquet(output / "pair_decisions.parquet", index=False)
    diagnostics_written = time.perf_counter()
    validation = validate_outputs(output, test_dir)
    validated = time.perf_counter()
    high_band = float(release["decision_config"].get("threshold", release["decision_config"]["t_rest"]))
    high_claims = frame.loc[frame.probability >= high_band].groupby("candidate_entity_id", sort=False).source1_entity_id.nunique()
    final_claims = frame.loc[chosen].groupby("candidate_entity_id", sort=False).source1_entity_id.nunique()
    # Recheck raw input bytes after the decision and validator passes.
    for n in (1, 2, 3):
        if digest(Path(test_dir) / f"test_source{n}.tsv") != manifest["test_files"][f"test_source{n}.tsv"]["sha256"]:
            raise ValueError("Test inputs changed during build")
    report = {"version": VERSION, "mode": "R1", "status": "complete" if validation["strict"] == validation["official"] == "PASS" else "validation_failed",
              "ready_for_upload": validation["strict"] == validation["official"] == "PASS", "entities": len(rows),
              "scored_candidates": len(frame), "predicted_links": int(chosen.sum()), "ownership_lost": int(lost.sum()),
              "gate_vetoes": int(frame.gate_veto.sum()) if "gate_veto" in frame else 0,
              "pre_decision_high_band_collision_targets": int((high_claims > 1).sum()),
              "pre_decision_high_band_threshold": high_band,
              "final_collision_targets": int((final_claims > 1).sum()),
              "final_max_owners_per_target": int(final_claims.max()) if len(final_claims) else 0,
              "decision": decision, "validation": validation, **statistics,
              "candidate_contract": "Exact unchanged stored candidate set in original per-reference retrieval order; final matches are subsets.",
              "base_score_contract": "Original first_stage, second_stage and probability bytes preserved in verified chunks; separate veto and decision fields.",
              "pair_order_sha256": manifest["pair_order_sha256"], "release_sha256": digest(release_path),
              "candidate_freeze_sha256": release.get("candidate_freeze", {}).get("sha256"),
              "output_sha256": {name: digest(output / name) for name in ("candidate_pairs.tsv", "matching_results.tsv", "pair_decisions.parquet")},
              "elapsed_seconds": time.perf_counter() - start, "peak_rss_platform_units": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "runtime_stage_seconds": {"descriptor_verification": descriptor_verified - start,
                                        "score_and_input_read_verification": scores_loaded - descriptor_verified,
                                        "gate_table_read_and_alignment": gate_aligned - scores_loaded,
                                        "global_decision_and_score_preservation": decided - gate_aligned,
                                        "tsv_export": exported - decided,
                                        "pair_diagnostics_write": diagnostics_written - exported,
                                        **validation["stage_seconds"],
                                        "collision_diagnostics_and_final_input_hashes": time.perf_counter() - validated},
              "leaderboard_score": None}
    histogram = statistics["candidate_histogram"]
    report["mean_candidates_per_s1"] = len(frame) / max(1, len(rows))
    report["max_candidates_per_s1"] = max(histogram, default=0)
    cumulative = 0
    report["p95_candidates_per_s1"] = 0
    for count, frequency in sorted(histogram.items()):
        cumulative += frequency
        if cumulative >= .95 * len(rows):
            report["p95_candidates_per_s1"] = count; break
    write_json(output / "submission_report.json", report)
    if not report["ready_for_upload"]:
        raise ValueError("Release validators failed; inspect versioned report/logs")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.release, args.test_dir, args.output), indent=2))


if __name__ == "__main__":
    main()
