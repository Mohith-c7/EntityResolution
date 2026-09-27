"""R1 artifacts: no scoring, no candidate filtering, complete owners and hashes."""
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("build_gated_submission", ROOT / "scripts/build_gated_submission.py")
r = importlib.util.module_from_spec(spec); spec.loader.exec_module(r)


def evidence(path):
    return {"path": str(path), "sha256": r.digest(path)}


@pytest.fixture
def bundle(tmp_path):
    test = tmp_path / "test"; test.mkdir()
    for source, rows in ((1, [("S1-A", "france"), ("S1-B", "india"), ("S1-C", "us")]),
                         (2, [("S2-X", "france"), ("S2-Y", "india")]), (3, [("S3-Z", "france")])):
        with (test / f"test_source{source}.tsv").open("w") as out:
            out.write("entity_id\tcountry\tbusiness_name\tbusiness_address\n")
            for eid, country in rows:
                out.write(f"{eid}\t{country}\tname\taddress\n")
    chunks = tmp_path / "chunks"; chunks.mkdir()
    frame = pd.DataFrame([("S1-A", "S2-X", .9, .91, .91), ("S1-A", "S3-Z", .71, .72, .72),
                          ("S1-B", "S2-X", .95, .96, .96), ("S1-B", "S2-Y", .81, .82, .82)],
                         columns=r.PAIR_KEYS + r.SCORE_COLUMNS)
    frame.to_parquet(chunks / "0000000.parquet", index=False)
    marker = {"chunk": 0, "entities": 3, "pairs": len(frame),
              "input_ids_sha256": r.ids_digest(["S1-A", "S1-B", "S1-C"]),
              "parquet_sha256": r.digest(chunks / "0000000.parquet")}
    r.write_json(chunks / "0000000.json", marker)
    frozen = tmp_path / "frozen.json"; r.write_json(frozen, {"threshold": .83,"code_sha256":{"scripts/run_frozen_pipeline.py":r.digest(ROOT / "scripts/run_frozen_pipeline.py")}})
    report = tmp_path / "base_report.json"; r.write_json(report, {"status": "complete", "entities":3,"scored_candidates":len(frame),"frozen_sha256":r.digest(frozen),"validation":{"strict_issues":[],"official_returncode":0}})
    manifest = {"version": r.VERSION, "entities": 3, "pairs": len(frame), "input_ids_sha256": marker["input_ids_sha256"],
                "pair_order_sha256": r.pair_order_digest(frame), "assets": [], "baseline_frozen": evidence(frozen),
                "baseline_report": evidence(report), "test_files": {f"test_source{i}.tsv": evidence(test / f"test_source{i}.tsv") for i in (1,2,3)},
                "chunks": [{"chunk":0,"marker":evidence(chunks / "0000000.json"),"parquet":evidence(chunks / "0000000.parquet"),"pair_order_sha256":r.pair_order_digest(frame)}]}
    manifest_path = tmp_path / "reuse.json"; r.write_json(manifest_path, manifest)
    runner = ROOT / "scripts/run_frozen_pipeline.py"
    release = {"status": "candidate_frozen", "frozen_at":"2026-09-27T11:37:34Z", "mode": "R1", "policy":"original", "parent_frozen_sha256":r.digest(frozen),
               "reuse_manifest":evidence(manifest_path),"code_sha256":{str(ROOT / path):r.digest(ROOT / path) for path in ("scripts/build_gated_submission.py","code/business_entity_resolution/src/pipeline/export.py","code/business_entity_resolution/src/utils/organizer_validate_submission.py")},
               "original_runner":evidence(runner), "decision_config":{"threshold":.83,"t_first":.7,"t_rest":.83}}
    release_path = tmp_path / "release.json"; r.write_json(release_path, release)
    return {"root":tmp_path,"test":test,"chunks":chunks,"frame":frame,"manifest":manifest,"release":release,
            "manifest_path":manifest_path,"release_path":release_path,"output":tmp_path / "submission_05"}


def rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def test_control_full_export_empty_owner_and_both_id_validators(bundle):
    before = r.digest(bundle["chunks"] / "0000000.parquet")
    report = r.build(bundle["release_path"], bundle["test"], bundle["output"], bundle["root"])
    assert report["validation"]["strict"] == report["validation"]["official"] == "PASS"
    assert report["validation"]["id_checking"] is True
    assert "--check-ids" in report["validation"]["official_command"]
    assert rows(bundle["output"] / "candidate_pairs.tsv") == [
        {"source1_entity_id":"S1-A","candidate_entity_ids":"S2-X,S3-Z"},
        {"source1_entity_id":"S1-B","candidate_entity_ids":"S2-X,S2-Y"},
        {"source1_entity_id":"S1-C","candidate_entity_ids":""}]
    assert rows(bundle["output"] / "matching_results.tsv") == [
        {"source1_entity_id":"S1-A","matched_entity_ids":"S3-Z"},
        {"source1_entity_id":"S1-B","matched_entity_ids":"S2-X"},
        {"source1_entity_id":"S1-C","matched_entity_ids":""}]
    decisions = pd.read_parquet(bundle["output"] / "pair_decisions.parquet")
    assert decisions[r.SCORE_COLUMNS].equals(bundle["frame"][r.SCORE_COLUMNS])
    assert r.digest(bundle["chunks"] / "0000000.parquet") == before
    assert report["country"]["us"]["empty_candidates"] == 1
    assert report["p95_candidates_per_s1"] == report["max_candidates_per_s1"] == 2
    with pytest.raises(ValueError, match="new/versioned"):
        r.build(bundle["release_path"], bundle["test"], bundle["output"], bundle["root"])


def test_pair_key_join_shuffled_gates_preserves_scores_and_candidates(bundle):
    frame = bundle["frame"]
    gates = frame[r.PAIR_KEYS].copy(); gates["gate_veto"] = [False,False,True,False]; gates["gate_reason"] = "fixture"
    path = bundle["root"] / "gates.parquet"; gates.iloc[::-1].to_parquet(path,index=False)
    bundle["release"]["gate_table"] = {**evidence(path),"pair_order_sha256":r.pair_order_digest(frame)}
    r.write_json(bundle["release_path"],bundle["release"])
    report = r.build(bundle["release_path"], bundle["test"], bundle["output"], bundle["root"])
    assert report["gate_vetoes"] == 1
    assert report["scored_candidates"] == 4
    assert rows(bundle["output"] / "matching_results.tsv")[0]["matched_entity_ids"] == "S2-X"
    assert pd.read_parquet(bundle["output"] / "pair_decisions.parquet")[r.SCORE_COLUMNS].equals(frame[r.SCORE_COLUMNS])


@pytest.mark.parametrize("kind",["file_hash","input_order","pair_order","pair_count","coverage","duplicate"])
def test_rejects_corruption_or_misalignment(bundle,kind):
    manifest=bundle["manifest"]; marker_path=bundle["chunks"] / "0000000.json"; parquet_path=bundle["chunks"] / "0000000.parquet"
    if kind == "file_hash":
        with parquet_path.open("ab") as out: out.write(b"corruption")
    elif kind in ("input_order","pair_count","coverage"):
        marker=json.loads(marker_path.read_text())
        marker[{"input_order":"input_ids_sha256","pair_count":"pairs","coverage":"entities"}[kind]] = {"input_order":"wrong","pair_count":9,"coverage":2}[kind]
        r.write_json(marker_path,marker);manifest["chunks"][0]["marker"]=evidence(marker_path)
    elif kind == "pair_order": manifest["chunks"][0]["pair_order_sha256"] = "wrong"
    else:
        frame=pd.concat([bundle["frame"],bundle["frame"].iloc[:1]],ignore_index=True);frame.to_parquet(parquet_path,index=False)
        marker=json.loads(marker_path.read_text());marker.update(pairs=len(frame),parquet_sha256=r.digest(parquet_path));r.write_json(marker_path,marker)
        manifest["chunks"][0].update(marker=evidence(marker_path),parquet=evidence(parquet_path))
    with pytest.raises(ValueError): r.load_verified_chunks(manifest,bundle["test"],bundle["root"])


@pytest.mark.parametrize("kind",["missing","extra","duplicate","wrong_hash","nonboolean"])
def test_rejects_bad_gate_universe(bundle,kind):
    frame=bundle["frame"];gates=frame[r.PAIR_KEYS].copy();gates["gate_veto"]=False;gates["gate_reason"]="ok"
    order=r.pair_order_digest(frame)
    if kind=="missing":gates=gates.iloc[:-1]
    elif kind=="extra":gates=pd.concat([gates,pd.DataFrame([{"source1_entity_id":"S1-C","candidate_entity_id":"S3-Z","gate_veto":False,"gate_reason":"ok"}])])
    elif kind=="duplicate":gates=pd.concat([gates.iloc[:-1],gates.iloc[:1]])
    elif kind=="wrong_hash":order="wrong"
    else:gates["gate_veto"]=1
    with pytest.raises(ValueError):r.attach_vetoes(frame,gates,order)


def test_freeze_and_preserved_directory_guards(bundle):
    bundle["release"]["status"]="development";r.write_json(bundle["release_path"],bundle["release"])
    with pytest.raises(ValueError,match="frozen"):r.build(bundle["release_path"],bundle["test"],bundle["output"],bundle["root"])
    bundle["release"]["status"]="candidate_frozen";r.write_json(bundle["release_path"],bundle["release"])
    with pytest.raises(ValueError,match="Preserved"):r.build(bundle["release_path"],bundle["test"],bundle["root"] / "output/submission_04/new",bundle["root"])


def test_v2_rescue_and_normal_global_collision(bundle):
    frame=bundle["frame"].copy(); frame["probability"]=[.72,.1,.91,.1]
    chosen,lost,details=r.make_decisions(frame,{"policy":"ownership_v2","decision_config":{"t_first":.7,"t_rest":.83}},bundle["root"])
    assert chosen.tolist()==[False,False,True,False]
    assert details["collision_targets"]==1
    assert details["rescue_proposals"]==1


def test_empty_pair_universe_has_all_s1_rows(tmp_path):
    frame=pd.DataFrame(columns=r.PAIR_KEYS+r.SCORE_COLUMNS)
    chosen,lost,details=r.make_decisions(frame,{"policy":"original","original_runner":evidence(ROOT / "scripts/run_frozen_pipeline.py"),"decision_config":{"threshold":.83,"t_first":.7,"t_rest":.83}})
    assert len(chosen)==0
    r.export_outputs([{"entity_id":"S1-empty","country":"france"}],frame,chosen,tmp_path)
    assert rows(tmp_path / "candidate_pairs.tsv")==[{"source1_entity_id":"S1-empty","candidate_entity_ids":""}]


def prepare_seal(bundle):
    import shutil
    import sqlite3
    baseline=bundle["root"] / "baseline";baseline.mkdir()
    shutil.copytree(bundle["chunks"],baseline / "chunks")
    asset=bundle["root"] / "immutable_asset";asset.write_text("fixture index/native/model")
    item=evidence(asset)
    frozen=bundle["root"] / "full_frozen.json"
    r.write_json(frozen,{"code_sha256":{},"first_stage":item,"second_stage":item,"aliases":item,"rules_source":item,
                         "bridge":{"ranking_model":str(asset),"ranking_model_sha256":r.digest(asset)}})
    source_rows=r.read_source1(bundle["test"] / "test_source1.tsv")
    chosen,_,_=r.make_decisions(bundle["frame"],bundle["release"],bundle["root"])
    r.export_outputs(source_rows,bundle["frame"],chosen,baseline)
    with sqlite3.connect(baseline / "universe_counts_test.sqlite") as conn:
        conn.execute("CREATE TABLE meta(key TEXT,value TEXT)")
        conn.execute("INSERT INTO meta VALUES ('universe',?)",(json.dumps({"sha256":r.digest(bundle["test"] / "test_source1.tsv"),"records":3}),))
    report={"status":"complete","validation":{"strict_issues":[],"official_returncode":0},"frozen_sha256":r.digest(frozen),
            "entities":3,"scored_candidates":4,"output_sha256":{name:r.digest(baseline / name) for name in ("matching_results.tsv","candidate_pairs.tsv")}}
    r.write_json(baseline / "submission_report.json",report)
    return baseline,frozen,{str(asset):r.digest(asset)}


def test_seal_proves_exact_candidate_export_without_baseline_mutations(bundle):
    baseline,frozen,assets=prepare_seal(bundle)
    before={str(path):r.digest(path) for path in baseline.rglob("*") if path.is_file()}
    path=bundle["root"] / "sealed.json"
    manifest=r.seal_reuse_manifest(baseline,bundle["test"],frozen,assets,path,bundle["root"])
    assert manifest["entities"]==3 and manifest["pairs"]==4
    owners,frame=r.load_verified_chunks(manifest,bundle["test"],bundle["root"])
    assert len(owners)==3 and frame[r.SCORE_COLUMNS].equals(bundle["frame"][r.SCORE_COLUMNS])
    assert before=={str(path):r.digest(path) for path in baseline.rglob("*") if path.is_file()}


def test_seal_rejects_candidate_export_not_equal_to_stored_scored_pairs(bundle):
    baseline,frozen,assets=prepare_seal(bundle)
    candidate=baseline / "candidate_pairs.tsv"
    candidate.write_text(candidate.read_text().replace("S2-X,S3-Z","S3-Z,S2-X"))
    report=json.loads((baseline / "submission_report.json").read_text())
    report["output_sha256"][candidate.name]=r.digest(candidate)
    r.write_json(baseline / "submission_report.json",report)
    with pytest.raises(ValueError,match="exact baseline candidate"):
        r.seal_reuse_manifest(baseline,bundle["test"],frozen,assets,bundle["root"] / "sealed.json",bundle["root"])


def test_execution_descriptor_keeps_audited_freeze_immutable(bundle):
    candidate=bundle["root"] / "candidate_freeze.json"
    r.write_json(candidate,bundle["release"])
    before=r.digest(candidate)
    bundle["release"]["candidate_freeze"]=evidence(candidate)
    bundle["release"]["decision_config"]["t_first"]=.6
    r.write_json(bundle["release_path"],bundle["release"])
    with pytest.raises(ValueError,match="differs from audited candidate"):
        r.build(bundle["release_path"],bundle["test"],bundle["output"],bundle["root"])
    assert r.digest(candidate)==before


@pytest.mark.parametrize("corruption", [None, "evidence", "copied_bytes"])
def test_portable_copy_requires_original_evidence_and_exact_bytes(bundle, corruption):
    spec = importlib.util.spec_from_file_location("verify_copied_seal", ROOT / "research/sprint_6h/release/verify_copied_seal.py")
    verifier = importlib.util.module_from_spec(spec); spec.loader.exec_module(verifier)
    baseline, frozen, assets = prepare_seal(bundle)
    manifest = r.seal_reuse_manifest(baseline, bundle["test"], frozen, assets,
                                     bundle["root"] / "local_seal.json", bundle["root"])

    def paths(value, portable):
        if isinstance(value, dict):
            result = {key: paths(item, portable) for key, item in value.items()}
            if isinstance(result.get("path"), str):
                relative = str(Path(result["path"]).relative_to(bundle["root"]))
                result["path"] = relative if portable else "/mnt/er/entity/" + relative
            return result
        if isinstance(value, list):
            return [paths(item, portable) for item in value]
        return value

    parent_path = bundle["root"] / "original_seal.json"
    r.write_json(parent_path, paths(manifest, False))
    portable = paths(manifest, True)
    portable["parent_manifest_sha256"] = r.digest(parent_path)
    if corruption == "evidence":
        portable["chunks"][0]["pair_order_sha256"] = "altered"
    elif corruption == "copied_bytes":
        with (baseline / "chunks/0000000.parquet").open("ab") as out:
            out.write(b"altered")
    portable_path = bundle["root"] / "portable_seal.json"
    r.write_json(portable_path, portable)
    output = bundle["root"] / "copy_verification.json"
    if corruption:
        with pytest.raises(ValueError, match="changed immutable|hash mismatch"):
            verifier.inspect(portable_path, parent_path, bundle["root"], output)
        assert not output.exists()
    else:
        result = verifier.inspect(portable_path, parent_path, bundle["root"], output)
        assert result["copy_byte_parity"] is True and result["pairs"] == 4
        assert result["validators_rerun"] is False
        assert result["pair_order_sha256"] == manifest["pair_order_sha256"]


def test_runtime_harness_measures_complete_stages_and_inflates_eta(bundle):
    spec = importlib.util.spec_from_file_location("benchmark_actual_blocks", ROOT / "research/sprint_6h/release/benchmark_actual_blocks.py")
    bench = importlib.util.module_from_spec(spec); spec.loader.exec_module(bench)
    metadata_paths = []
    for number in range(2):
        directory = bundle["root"] / f"runtime_input_{number}"; directory.mkdir()
        source = pd.read_csv(bundle["test"] / "test_source1.tsv", sep="\t")
        frame = bundle["frame"].copy()
        if number:
            replacements = {"S1-A":"S1-D", "S1-B":"S1-E", "S1-C":"S1-F"}
            source["entity_id"] = source.entity_id.map(replacements)
            frame["source1_entity_id"] = frame.source1_entity_id.map(replacements)
        source_path = directory / "source1.tsv"; source.to_csv(source_path, sep="\t", index=False)
        score_path = directory / "scores.parquet"; frame.to_parquet(score_path, index=False)
        gate_path = directory / "gate_table.parquet"
        gate = frame[r.PAIR_KEYS].copy(); gate["gate_veto"] = False; gate["gate_reason"] = "fixture"
        gate.to_parquet(gate_path, index=False)
        metadata = {"candidate_frozen_sha256":r.digest(bundle["release_path"]), "measured":True,
                    "source1":evidence(source_path), "scores":evidence(score_path), "gate_table":evidence(gate_path),
                    "references":3, "pairs":4, "country_counts":{"france":1,"india":1,"us":1},
                    "pair_order_sha256":r.pair_order_digest(frame), "eligible_signal_rows":4, "skipped_signal_rows":0,
                    "seconds":{stage:.01 for stage in bench.GATE_STAGES}}
        path = directory / "metadata.json"; r.write_json(path, metadata); metadata_paths.append(path)
    proof = bundle["root"] / "copy_proof.json"
    r.write_json(proof, {"status":"complete", "copy_byte_parity":True,
                        "baseline_frozen_sha256":bundle["release"]["parent_frozen_sha256"],
                        "pairs":8, "entities":6, "elapsed_seconds":.01})
    result = bench.benchmark(bundle["release_path"], metadata_paths, bundle["test"], proof,
                             bundle["root"] / "runtime_output", root=bundle["root"])
    assert all(set(bench.ALL_STAGES) <= set(block["seconds"]) for block in result["blocks"])
    assert all(block["validation"]["strict"] == block["validation"]["official"] == "PASS" for block in result["blocks"])
    assert result["estimated_processing_seconds"] == pytest.approx(result["estimated_before_safety_seconds"] * 1.15)
    assert result["candidate_frozen_sha256"] == r.digest(bundle["release_path"])


def test_original_floor_subset_preserves_global_accepted_pairs_with_vetoes(bundle):
    rng = np.random.default_rng(21)
    data = []
    for owner in range(100):
        for rank in range(10):
            probability = rng.choice([0., .4, .6999999999999997, .6999999999999998, .75, .83, .8300000000000004, .9])
            data.append((f"S1-{owner}", f"S2-{(owner + rank) % 35}", probability, probability, probability))
    frame = pd.DataFrame(data, columns=r.PAIR_KEYS + r.SCORE_COLUMNS)
    frame["gate_veto"] = rng.random(len(frame)) < .2
    release = {**bundle["release"], "decision_config":{"threshold":.83,"t_first":.6999999999999998,"t_rest":.8300000000000004}}
    chosen, lost, _ = r.make_decisions(frame, release)
    retained = frame.probability >= min(release["decision_config"]["t_first"], release["decision_config"]["t_rest"])
    subset = frame.loc[retained].reset_index(drop=True)
    subset_chosen, subset_lost, _ = r.make_decisions(subset, release)
    assert set(map(tuple, frame.loc[chosen, r.PAIR_KEYS].to_numpy())) == set(map(tuple, subset.loc[subset_chosen, r.PAIR_KEYS].to_numpy()))
    assert np.array_equal(lost[retained], subset_lost)


def test_french_diagnostic_requires_exact_baseline_bytes_and_inspectable_winners(bundle):
    import shutil
    spec = importlib.util.spec_from_file_location("diagnose_france_claims", ROOT / "research/sprint_6h/release/diagnose_france_claims.py")
    diag = importlib.util.module_from_spec(spec); spec.loader.exec_module(diag)
    baseline, frozen, assets = prepare_seal(bundle)
    runner = bundle["root"] / "scripts/run_frozen_pipeline.py"; runner.parent.mkdir(); shutil.copy2(ROOT / "scripts/run_frozen_pipeline.py", runner)
    config = json.loads(frozen.read_text()); config.update(bundle["release"]["decision_config"])
    config["code_sha256"] = {"scripts/run_frozen_pipeline.py":r.digest(runner)}; r.write_json(frozen, config)
    report_path = baseline / "submission_report.json"; report = json.loads(report_path.read_text())
    report.update(frozen_sha256=r.digest(frozen),predicted_links=2,country={"france":{"predicted_links":1},"india":{"predicted_links":1},"us":{"predicted_links":0}})
    r.write_json(report_path,report)
    manifest_path = bundle["root"] / "sealed.json"
    r.seal_reuse_manifest(baseline,bundle["test"],frozen,assets,manifest_path,bundle["root"])
    proof_path = bundle["root"] / "copy_proof.json"
    r.write_json(proof_path,{"copy_byte_parity":True,"status":"complete","portable_manifest_sha256":r.digest(manifest_path),"baseline_frozen_sha256":r.digest(frozen)})
    cache = bundle["root"] / "diagnostic_cache"
    parity = diag.prepare(manifest_path,proof_path,frozen,bundle["test"],cache,bundle["root"])
    assert parity["exact_matching_byte_reconstruction"] is True
    assert parity["baseline_matching_sha256"] == report["output_sha256"]["matching_results.tsv"]
    table = bundle["frame"].loc[bundle["frame"].source1_entity_id.eq("S1-A"),r.PAIR_KEYS].copy()
    table["gate_veto"] = [False,True]; table["gate_reason"] = ["keep","veto_fixture"]
    table_path = bundle["root"] / "french.parquet"; table.to_parquet(table_path,index=False)
    metadata_path = bundle["root"] / "metadata.json"
    r.write_json(metadata_path,{"table_sha256":r.digest(table_path),"baseline_frozen_sha256":r.digest(frozen),"pair_order_sha256":r.pair_order_digest(table)})
    destination = bundle["root"] / "veto_diagnostic"
    result = diag.compare(cache,table_path,metadata_path,frozen,bundle["test"],destination,bundle["root"],manifest_path=manifest_path)
    assert result["country_link_deltas"] == {"france":-1,"india":0,"us":0}
    assert result["removed_accepted_claims"] == result["changed_target_winners"] == 1
    changes = pd.read_parquet(destination / "changed_claims.parquet")
    assert changes.candidate_entity_id.tolist() == ["S3-Z"]
    assert changes.baseline_accepted.tolist() == [True] and changes.gated_accepted.tolist() == [False]
    assert result["ready_for_upload"] is False and result["labels_used"] is False


def test_changed_winner_derivation_preserves_unchanged_coowners(tmp_path):
    spec = importlib.util.spec_from_file_location("derive_changed_winners", ROOT / "research/sprint_6h/release/derive_changed_winners.py")
    derive = importlib.util.module_from_spec(spec); spec.loader.exec_module(derive)
    cache=tmp_path/"cache";cache.mkdir();comparison=tmp_path/"comparison";comparison.mkdir()
    frame=pd.DataFrame([("S1-A","S2-X"),("S1-B","S2-X")],columns=r.PAIR_KEYS)
    frame.to_parquet(cache/"eligible_claims.parquet",index=False)
    np.savez_compressed(cache/"baseline_masks.npz",chosen=np.array([True,True]),lost=np.array([False,False]))
    r.write_json(cache/"baseline_parity.json",{"status":"baseline_parity_verified",
                 "eligible_claims_sha256":r.digest(cache/"eligible_claims.parquet"),"baseline_masks_sha256":r.digest(cache/"baseline_masks.npz")})
    change=frame.iloc[:1].copy();change["baseline_accepted"]=True;change["gated_accepted"]=False
    change.to_parquet(comparison/"changed_claims.parquet",index=False)
    r.write_json(comparison/"diagnostic.json",{"status":"diagnostic_complete","baseline_parity_sha256":r.digest(cache/"baseline_parity.json"),
                 "changed_claims_sha256":r.digest(comparison/"changed_claims.parquet"),"changed_target_winners":1})
    result=derive.derive(cache,comparison)
    winners=json.loads((comparison/"changed_winners.json").read_text())
    assert winners==[{"candidate_entity_id":"S2-X","baseline_owner_ids":["S1-A","S1-B"],"gated_owner_ids":["S1-B"],
                      "removed_owner_ids":["S1-A"],"added_owner_ids":[],"unchanged_owner_ids":["S1-B"]}]
    assert result["arbitration_rerun"] is False and result["unchanged_coowners_preserved"] is True
