import ast
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from decision_policy_v2 import decide_v2, form_proposals
from evaluate_sprint import decide_control, paired_evaluate, verify_audit_freeze
from export_sprint_workbench import export_workbench, compare_replay, sha
spec = importlib.util.spec_from_file_location("reserve_splits", ROOT / "research/sprint_6h/validation/reserve_splits.py")
reserve_module = importlib.util.module_from_spec(spec); spec.loader.exec_module(reserve_module)
CONFIG = {"threshold": .83, "t_first": .70, "t_rest": .83}


@pytest.mark.parametrize("corruption", [None, "pairs", "coverage", "proof"])
def test_fixed_development_evaluation_requires_sealed_graph(tmp_path, corruption):
    spec = importlib.util.spec_from_file_location("fixed_evaluation", ROOT / "research/sprint_6h/validation/evaluate_ownership_fixed.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    frozen = tmp_path / "frozen.json"; frozen.write_text("{}")
    proof = tmp_path / "proof.json"; proof.write_text("{}")
    workbench = tmp_path / "workbench"; workbench.mkdir()
    rows = [{"entity_id": "S1-a"}, {"entity_id": "S1-empty"}]
    (workbench / "references.json").write_text(json.dumps(rows))
    claims = pd.DataFrame({"source1_entity_id": ["S1-a"], "candidate_entity_id": ["S2-x"],
                           "probability": [.9], "candidate_order": [0]})
    pairs = workbench / "pairs.parquet"; claims.to_parquet(pairs, index=False)
    manifest = {"status": "complete", "verified_current_pipeline": True, "split": "development",
                "frozen_sha256": sha(frozen), "audit_labels_read": False, "second_stage_owner_excluded": True,
                "pairs_sha256": sha(pairs), "reference_ids": [r["entity_id"] for r in rows], "entities": 2,
                "chunks": [{"entities": 2}], "pairs": 1,
                "pair_order_sha256": hashlib.sha256(b"S1-a\tS2-x\n").hexdigest()}
    for name in ("parity", "asset"):
        manifest[name + "_proof"] = str(proof); manifest[name + "_proof_sha256"] = sha(proof)
    if corruption == "pairs": claims.assign(probability=.8).to_parquet(pairs, index=False)
    if corruption == "coverage": manifest["chunks"] = [{"entities": 1}]
    if corruption == "proof": proof.write_text('{"changed":true}')
    (workbench / "manifest.json").write_text(json.dumps(manifest))
    if corruption:
        with pytest.raises(ValueError): module.verify_development(workbench, frozen)
    else:
        saved, references, scored = module.verify_development(workbench, frozen)
        assert len(references) == 2 and len(scored) == 1 and saved["entities"] == 2


@pytest.mark.parametrize("drop_empty_reference", [False, True])
def test_complete_heldout_evidence_counts_processed_empty_references(tmp_path, drop_empty_reference):
    spec = importlib.util.spec_from_file_location("heldout_evidence", ROOT / "research/sprint_6h/validation/build_heldout_evidence.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    frozen = tmp_path / "frozen.json"; frozen.write_text("{}")
    proof = tmp_path / "proof.json"; proof.write_text("{}")
    workbench = tmp_path / "workbench"; workbench.mkdir()
    rows = [{"entity_id": "S1-a"}, {"entity_id": "S1-empty"}]
    raw = tmp_path / "heldout_universe.json"; raw.write_text(json.dumps(rows))
    (workbench / "references.json").write_text(json.dumps(rows))
    claims = pd.DataFrame({"source1_entity_id": ["S1-a"], "candidate_entity_id": ["S2-x"]})
    pairs = workbench / "pairs.parquet"; claims.to_parquet(pairs, index=False)
    plan = {"claimant_universe": {"kind": "complete_heldout", "labels_read": False, "entities": 2, "records_sha256": sha(raw)}, "cohort_fingerprints": {}}
    for role, row in zip(("confirmation", "audit"), rows):
        directory = tmp_path / role; directory.mkdir(); p = directory / "references.json"; p.write_text(json.dumps([row]))
        plan["cohort_fingerprints"][role] = {"references_sha256": sha(p)}
    reservation = tmp_path / "plan.json"; reservation.write_text(json.dumps(plan))
    ids = [r["entity_id"] for r in rows]
    manifest = {"status": "complete", "verified_current_pipeline": True, "split": "cohort",
                "frozen_sha256": sha(frozen), "audit_labels_read": False, "second_stage_owner_excluded": True,
                "references_sha256": sha(raw), "pairs_sha256": sha(pairs), "reference_ids": ids, "entities": 2,
                "chunks": [{"chunk": 0, "entities": 2, "pairs": 1, "input_ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest()}],
                "pairs": 1, "pair_order_sha256": hashlib.sha256(b"S1-a\tS2-x\n").hexdigest(),
                "incumbent_fitted_owner_ids_sha256": "fit-hash", "capture_script_sha256": "capture-hash"}
    for name in ("parity", "asset"):
        manifest[name + "_proof"] = str(proof); manifest[name + "_proof_sha256"] = sha(proof)
    if drop_empty_reference:
        manifest["chunks"][0].update(entities=1,input_ids_sha256=hashlib.sha256(b"S1-a").hexdigest())
    (workbench / "manifest.json").write_text(json.dumps(manifest))
    if drop_empty_reference:
        with pytest.raises(ValueError): module.build(workbench, reservation, frozen)
    else:
        evidence = module.build(workbench, reservation, frozen)
        assert evidence["scored_references"] == 2 and evidence["pair_bearing_references"] == 1
        assert evidence["zero_candidate_references"] == 1 and evidence["scoring_complete"]


@pytest.mark.parametrize("corruption", [None, "self_rival", "mixed_component", "query_cap"])
def test_independent_reverse_feature_verifier_detects_contract_breaks(tmp_path, corruption):
    spec = importlib.util.spec_from_file_location("reverse_verifier", ROOT / "research/sprint_6h/validation/verify_reverse_features.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    own = [ .9, .8, .7, 1., 0., .86 ]; other = [ .6, .5, .4, .5, 0., .5 ]
    suffix = ["name_sort", "phonetic_sort", "address_sort", "number_jaccard", "number_contradiction", "joint"]
    row = {"source1_entity_id":"S1-own", "candidate_entity_id":"S2-t"}
    row.update({"reverse_own_"+s:v for s,v in zip(suffix,own)})
    row.update({"reverse_other_"+s:v for s,v in zip(suffix,other)})
    row.update({"reverse_margin_"+s:v for s,v in zip(["name","phonetic","address","number","joint"],[.3,.3,.3,.5,.36])})
    row.update(reverse_other_second_joint=-1.,reverse_other_third_joint=-1.,reverse_best_second_joint_gap=-1.,
               reverse_name_terms_used=1.,reverse_phonetic_terms_used=0.,reverse_address_terms_used=1.,reverse_query_fields_used=2.,
               reverse_high_df_terms_skipped=3.,reverse_shortlist_saturated=0.,reverse_own_in_top8=1.,reverse_no_other_returned=0.)
    diagnostic = {"source1_entity_id":"S1-own","candidate_entity_id":"S2-t","rival_ids":["S1-rival"],
                  "rival_views":[dict(zip(["name","phonetic","address","digits","number_conflict","joint"],other))],
                  "selected_terms":{"namecore":[[1,"rare"]],"phonetic":[],"address":[[1,"road"]]},
                  "country_records":100,"query_fields":2,"high_df_skipped":3,"shortlist_saturated":False,
                  "self_excluded":True,"complete_postings":True,"shortlist_count":2,"rerank_count":2,
                  "nonself_top8_count":1,"posting_count_nonself":1,"posting_count":2,"no_result":False,"no_query":False}
    if corruption == "self_rival": diagnostic["rival_ids"] = ["S1-own"]
    if corruption == "mixed_component": row["reverse_other_address_sort"] = .99
    if corruption == "query_cap": diagnostic["selected_terms"]["namecore"] = [[1,"a"],[1,"b"],[1,"c"]]
    features = tmp_path / "features.parquet"; pd.DataFrame([row]).to_parquet(features,index=False)
    diagnostics = tmp_path / "diagnostics.jsonl"; diagnostics.write_text(json.dumps(diagnostic)+"\n")
    (tmp_path / "manifest.json").write_text(json.dumps({"features_sha256":sha(features),"diagnostics_sha256":sha(diagnostics)}))
    if corruption:
        with pytest.raises(ValueError): module.verify(tmp_path)
    else:
        report = module.verify(tmp_path)
        assert report["rows"] == 1 and report["self_in_top8_rows"] == 1 and not report["labels_read"]


def frame(rows):
    result = pd.DataFrame(rows, columns=["source1_entity_id", "candidate_entity_id", "probability"])
    result["candidate_order"] = result.groupby("source1_entity_id", sort=False).cumcount()
    return result


def test_rescues_compete_with_ordinary_claims():
    f = frame([("S1-a", "S2-x", .8), ("S1-b", "S2-x", .9)])
    chosen, lost, detail = decide_v2(f, f, CONFIG)
    assert chosen.tolist() == [False, True] and lost.tolist() == [True, False]
    assert detail["summary"]["rescue_proposals"] == 1


def test_rescue_collision_ties_abstain():
    f = frame([("S1-a", "S2-x", .8), ("S1-b", "S2-x", .8)])
    chosen, _, detail = decide_v2(f, f, CONFIG)
    assert not chosen.any() and detail["summary"]["tie_targets"] == 1


def test_no_rescue_cascade_after_ownership_loss():
    f = frame([("S1-a", "S2-x", .8), ("S1-a", "S2-y", .75), ("S1-b", "S2-x", .9)])
    chosen, _, _ = decide_v2(f, f, CONFIG)
    assert chosen.tolist() == [False, False, True]


def test_veto_does_not_rerank_rescue():
    f = frame([("S1-a", "S2-x", .8), ("S1-a", "S2-y", .75)])
    f["gate_veto"] = [True, False]
    assert not decide_v2(f, f, CONFIG)[0].any()


def test_multiple_targets_per_owner_are_allowed():
    f = frame([("S1-a", "S2-x", .9), ("S1-a", "S3-y", .95)])
    assert decide_v2(f, f, CONFIG)[0].all()


def test_margin_can_select_no_owner_and_street_does_not_force_winner():
    f = frame([("S1-a", "S2-x", .91), ("S1-b", "S2-x", .90)])
    f["street_overlap"] = [0., 1.]
    assert not decide_v2(f, f, {**CONFIG, "ownership_margin": .02})[0].any()
    assert decide_v2(f, f, CONFIG)[0].tolist() == [True, False]


@pytest.mark.parametrize("key", ["ownership_margin", "tie_tolerance", "ownership_min_score"])
def test_nonfinite_ownership_config_is_rejected(key):
    f = frame([("S1-a", "S2-x", .9), ("S1-b", "S2-x", .9)])
    with pytest.raises(ValueError): decide_v2(f, f, {**CONFIG, key: float("nan")})


def test_row_order_does_not_change_owner_or_rescue():
    f = frame([("S1-a", "S2-x", .8), ("S1-a", "S2-y", .8), ("S1-b", "S2-x", .9)])
    expected = set(map(tuple, f.loc[decide_v2(f, f, CONFIG)[0], ["source1_entity_id", "candidate_entity_id"]].to_numpy()))
    shuffled = f.sample(frac=1, random_state=7).reset_index(drop=True)
    actual = set(map(tuple, shuffled.loc[decide_v2(shuffled, shuffled, CONFIG)[0], ["source1_entity_id", "candidate_entity_id"]].to_numpy()))
    assert actual == expected


def test_complete_graph_fast_path_matches_keyed_copy():
    f = frame([("S1-a", "S2-x", .8), ("S1-a", "S2-y", .75), ("S1-b", "S2-x", .9),
               ("S1-c", "S3-t", .85), ("S1-d", "S3-t", .85)])
    direct = decide_v2(f, f, CONFIG)
    keyed = decide_v2(f.copy(), f, CONFIG)
    assert np.array_equal(direct[0], keyed[0]) and np.array_equal(direct[1], keyed[1])
    pd.testing.assert_frame_equal(direct[2]["rows"], keyed[2]["rows"])
    assert direct[2]["summary"] == keyed[2]["summary"]
    reversed_columns = f[list(reversed(f.columns))]
    fast = decide_v2(reversed_columns, reversed_columns, CONFIG)
    slow = decide_v2(reversed_columns.copy(), reversed_columns, CONFIG)
    pd.testing.assert_frame_equal(fast[2]["rows"], slow[2]["rows"])


@pytest.mark.parametrize("case", ["duplicate", "nan", "foreign", "changed_score", "changed_veto"])
def test_invalid_claim_universe_fails(case):
    f = frame([("S1-a", "S2-x", .9)]); claims = f.copy()
    if case == "duplicate": claims = pd.concat([claims, claims])
    if case == "nan": claims.loc[0, "probability"] = np.nan
    if case == "foreign": f.loc[0, "candidate_entity_id"] = "S3-z"
    if case == "changed_score": f.loc[0, "probability"] = .91
    if case == "changed_veto": f["gate_veto"] = True; claims["gate_veto"] = False
    with pytest.raises(ValueError): decide_v2(f, claims, CONFIG)


def test_control_exact_parity_with_frozen_function():
    tree = ast.parse((ROOT / "scripts/run_frozen_pipeline.py").read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "decide")
    module = ast.Module(body=[node], type_ignores=[]); namespace = {"np": np, "pd": pd}
    exec(compile(module, "frozen_decide", "exec"), namespace)
    rng = np.random.default_rng(20260927)
    for _ in range(30):
        rows = [(f"S1-{s}", f"S2-{t}", float(rng.choice([.69, .7, .8, .83, .9, .99]))) for s in range(8) for t in range(5)]
        f = frame(rows).sample(frac=1, random_state=1).reset_index(drop=True)
        a = namespace["decide"](f, f, CONFIG); b = decide_control(f, f, CONFIG)
        assert all(np.array_equal(x, y) for x, y in zip(a, b))


def fixture_chunks(tmp_path):
    chunks = tmp_path / "chunks"; chunks.mkdir()
    rows = [{"entity_id": "S1-a", "country": "US"}, {"entity_id": "S1-empty", "country": "IN"}]
    refs = tmp_path / "refs.json"; refs.write_text(json.dumps(rows))
    f = frame([("S1-a", "S2-x", .9)])
    f["first_stage"] = .95; f["second_stage"] = .85
    p = chunks / "0000000.parquet"; f.drop(columns="candidate_order").to_parquet(p, index=False)
    m = {"chunk": 0, "entities": 2, "pairs": 1, "parquet_sha256": sha(p),
         "input_ids_sha256": hashlib.sha256("S1-a\nS1-empty".encode()).hexdigest()}
    p.with_suffix(".json").write_text(json.dumps(m))
    lineage = tmp_path / "lineage.json"; lineage.write_text(json.dumps({k: {} for k in ("input", "normalization", "aliases", "indexes", "native_library", "feature_schema", "models")}))
    return chunks, refs, lineage


def test_export_keeps_empty_references_and_seals_no_truth(tmp_path):
    chunks, refs, lineage = fixture_chunks(tmp_path)
    result, report = export_workbench(chunks, refs, tmp_path / "out", lineage)
    assert report["zero_candidate_references"] == 1 and report["references"] == 2
    assert "label" not in result and result.candidate_order.tolist() == [0]


def test_export_rejects_equal_length_wrong_reference_order(tmp_path):
    chunks, refs, lineage = fixture_chunks(tmp_path)
    refs.write_text(json.dumps(list(reversed(json.loads(refs.read_text())))))
    with pytest.raises(ValueError, match="order"): export_workbench(chunks, refs, tmp_path / "out", lineage)


def test_export_rejects_hash_corruption(tmp_path):
    chunks, refs, lineage = fixture_chunks(tmp_path)
    p = chunks / "0000000.parquet"; p.write_bytes(p.read_bytes() + b"x")
    with pytest.raises(ValueError, match="hash"): export_workbench(chunks, refs, tmp_path / "out", lineage)


def test_replay_must_verify_pair_keys_not_lengths():
    a = frame([("S1-a", "S2-x", .9)]); a["first_stage"] = .9; a["second_stage"] = .9
    b = a.copy(); b["candidate_entity_id"] = "S2-y"
    with pytest.raises(ValueError, match="IDs"): compare_replay(a, b)


def test_pair_metrics_empty_truth_country_and_link_removal():
    truth = {"S1-a": ["S2-x"], "S1-b": []}
    base = {"S1-a": ["S2-x", "S2-fp"], "S1-b": []}
    candidate = {"S1-a": ["S2-x"], "S1-b": []}
    result = paired_evaluate(truth, base, candidate, {"S1-a": "US", "S1-b": "IN"}, iterations=100)
    assert result["baseline"]["macro_f05"] == pytest.approx((1.25 / 2.25 + 1) / 2)
    assert result["candidate"]["macro_f05"] == 1
    assert result["links"]["false_removed"] == 1 and result["links"]["true_removed"] == 0


def test_paired_evaluation_refuses_missing_empty_prediction():
    with pytest.raises(ValueError, match="identical"):
        paired_evaluate({"S1-a": [], "S1-b": []}, {"S1-a": []}, {"S1-a": [], "S1-b": []}, {"S1-a": "US", "S1-b": "US"}, iterations=100)


def test_incomplete_universe_cannot_promote_ownership():
    result = paired_evaluate({"S1-a": []}, {"S1-a": []}, {"S1-a": []}, {"S1-a": "US"}, iterations=100,
                             ownership=True, universe={"kind": "exact_key_rivals", "complete": False})
    assert "global_ownership_universe_not_complete_and_heldout" in result["acceptance"]["failures"]


def test_relative_singleton_rule_has_no_historical_cap():
    truth = {f"S1-{i}": [] for i in range(40)}; countries = {e: "US" for e in truth}
    predictions = {e: ["S2-x"] for e in truth}
    result = paired_evaluate(truth, predictions, predictions, countries, iterations=100)
    assert result["acceptance"]["non_inferiority_pass"]


def test_exposure_scans_reference_columns_and_json_keys(tmp_path):
    artifacts = tmp_path / "artifacts"; artifacts.mkdir()
    (artifacts / "prior_truth.json").write_text(json.dumps({"S1-a": ["S2-z"], "heldout": [{"entity_id": "S1-b"}]}))
    pd.DataFrame({"source1_entity_id": ["S1-c"], "candidate_entity_id": ["S2-z"], "label": [1]}).to_parquet(artifacts / "scores.parquet")
    report = reserve_module.collect([artifacts], tmp_path / "ledger")
    assert report["references"] == 3
    assert json.loads((tmp_path / "ledger/exposed_ids.json").read_text()) == ["S1-a", "S1-b", "S1-c"]


def test_reservation_disjoint_excludes_exposure_and_never_reads_truth(tmp_path):
    source = tmp_path / "source.tsv"; source.write_text("entity_id\tcountry\tname\n" + "".join(f"S1-{i}\tUS\tN{i}\n" for i in range(1000)))
    eligible = [f"S1-{i}" for i in range(1000) if reserve_module.entity_fold(f"S1-{i}") == "holdout"]
    ledger = tmp_path / "ledger.json"; ledger.write_text(json.dumps(eligible[:3]))
    result = reserve_module.reserve(source, [ledger], tmp_path / "out", count=5, scope_reviewed=True)
    a = set(json.loads((tmp_path / "out/audit/entity_ids.json").read_text())); c = set(json.loads((tmp_path / "out/confirmation/entity_ids.json").read_text()))
    assert not a & c and not (a | c) & set(eligible[:3])
    assert result["status"].endswith("no_labels_read")


def test_audit_freeze_required_before_truth():
    # Failure is independent of any truth path: no label opening occurs.
    with pytest.raises(ValueError): verify_audit_freeze(Path(__file__), Path(__file__))


def test_valid_audit_freeze_and_changed_artifact(tmp_path):
    from datetime import datetime, timezone
    asset = tmp_path / 'asset.txt'; asset.write_text('frozen')
    refs = tmp_path / 'refs.json'; refs.write_text('[]')
    descriptor = {'status':'candidate_frozen', 'frozen_at':datetime.now(timezone.utc).isoformat(),
                  'parent_frozen_sha256':'baseline', 'split_ledger_sha256':'ledger',
                  'audit_sampled_ids_sha256':sha(refs)}
    for group in ('code_sha256','model_sha256','schema_sha256','index_sha256','native_sha256'):
        descriptor[group] = {str(asset):sha(asset)}
    frozen = tmp_path / 'frozen.json'; frozen.write_text(json.dumps(descriptor))
    assert verify_audit_freeze(frozen, refs)['status'] == 'candidate_frozen'
    asset.write_text('changed')
    with pytest.raises(ValueError, match='changed'): verify_audit_freeze(frozen, refs)


def test_audit_once_lock_is_tied_to_cohort_not_output_path(tmp_path, monkeypatch):
    from datetime import datetime, timezone
    import evaluate_sprint
    refs = tmp_path / 'refs.json'; refs.write_text(json.dumps([{'entity_id':'S1-a','country':'US'}]))
    truth = tmp_path / 'truth.json'; truth.write_text(json.dumps({'S1-a':[]}))
    baseline = tmp_path / 'baseline.json'; baseline.write_text(json.dumps({'S1-a':[]}))
    candidate = tmp_path / 'candidate.json'; candidate.write_text(json.dumps({'S1-a':[]}))
    universe = tmp_path / 'universe.json'; universe.write_text(json.dumps({'kind':'exact_key_rivals','complete':False}))
    asset = tmp_path / 'asset.txt'; asset.write_text('frozen')
    descriptor = {'status':'candidate_frozen', 'frozen_at':datetime.now(timezone.utc).isoformat(),
                  'parent_frozen_sha256':'baseline', 'split_ledger_sha256':'ledger', 'audit_sampled_ids_sha256':sha(refs)}
    for group in ('code_sha256','model_sha256','schema_sha256','index_sha256','native_sha256'):
        descriptor[group] = {str(asset):sha(asset)}
    descriptor['decision_config'] = CONFIG
    frozen = tmp_path / 'frozen.json'; frozen.write_text(json.dumps(descriptor))
    from evaluate_sprint import canonical_sha
    plan=tmp_path/'reservation.json';plan.write_text(json.dumps({'status':'confirmation_and_audit_reserved_no_labels_read','scope_reviewed':True,
         'reserved_at':datetime.now(timezone.utc).isoformat(),'exposure_ledgers':[],
         'cohort_fingerprints':{'audit':{'entities':1,'reference_ids_sha256':hashlib.sha256(b'S1-a').hexdigest(),'references_sha256':sha(refs)}}}))
    proof=tmp_path/'predictions.json';proof.write_text(json.dumps({'candidate_sha256':sha(candidate),'baseline_sha256':sha(baseline),'references_sha256':sha(refs),'universe_sha256':sha(universe),
          'candidate_frozen_sha256':sha(frozen),'baseline_frozen_sha256':'baseline','decision_config_sha256':canonical_sha(CONFIG)}))
    seconds={s:0.01 for s in ['reads','lookups','normalization','gate_or_features','predict','global_decision','diagnostic_write','tsv_export','strict_validator','official_validator']}
    runtime=tmp_path/'runtime.json';runtime.write_text(json.dumps({'measured':True,'passed':True,'mode':'R1','full_test_eta_supported':True,'estimated_processing_seconds':10,'processing_budget_seconds':7200,
        'safety_margin_fraction':.15,'candidate_frozen_sha256':sha(frozen),'blocks':[{'references':3,'country_stratified':True,'seconds':seconds}]*2}))
    args = ['evaluate_sprint.py','--truth',str(truth),'--references',str(refs),'--baseline',str(baseline),
            '--candidate',str(candidate),'--universe',str(universe),'--audit','--freeze',str(frozen),'--iterations','100',
            '--runtime',str(runtime),'--reservation-plan',str(plan),'--prediction-evidence',str(proof)]
    monkeypatch.setattr(sys,'argv',args+['--output',str(tmp_path/'first')]); evaluate_sprint.main()
    assert (tmp_path/'audit_evaluation_started.json').exists()
    monkeypatch.setattr(sys,'argv',args+['--output',str(tmp_path/'second')])
    with pytest.raises(FileExistsError): evaluate_sprint.main()
    assert not (tmp_path/'second').exists()


def test_country_harm_blocks_global_average_improvement():
    truth = {f'S1-US-{i}':['S2-t'] for i in range(100)}
    truth['S1-IN']=['S2-t']
    countries={e:('IN' if e=='S1-IN' else 'US') for e in truth}
    baseline={e:(['S2-t'] if e=='S1-IN' else ['S2-t','S3-fp']) for e in truth}
    candidate={e:([] if e=='S1-IN' else ['S2-t']) for e in truth}
    report=paired_evaluate(truth,baseline,candidate,countries,iterations=100)
    assert report['paired_macro_f05_delta']>0
    assert 'known_country_point_loss_below_minus_0.0005' in report['acceptance']['failures']


def test_singleton_false_predictions_relative_increase_blocks_improvement():
    truth={f'S1-{i}':['S2-t'] for i in range(100)};truth['S1-singleton']=[]
    countries={e:'US' for e in truth}
    baseline={e:([] if not truth[e] else ['S2-t','S3-fp']) for e in truth}
    candidate={e:(['S3-fp'] if not truth[e] else ['S2-t']) for e in truth}
    report=paired_evaluate(truth,baseline,candidate,countries,iterations=100)
    assert report['paired_macro_f05_delta']>0
    assert 'singleton_false_predictions_rose' in report['acceptance']['failures']


def test_empty_workbench_is_valid_and_has_no_proposals():
    f=frame([])
    chosen,lost,detail=decide_v2(f,f,CONFIG)
    assert len(chosen)==len(lost)==0 and detail['summary']['accepted_links']==0


def test_prediction_schema_rejects_legacy_nested_records():
    with pytest.raises(ValueError,match='target-ID'):
        paired_evaluate({'S1-a':[]},{'S1-a':{'predictions':[]}},{'S1-a':[]},{'S1-a':'US'},iterations=100)


def runtime_fixture():
    stages=['reads','lookups','normalization','gate_or_features','predict','global_decision','diagnostic_write','tsv_export','strict_validator','official_validator']
    return {'measured':True,'passed':True,'mode':'R1','full_test_eta_supported':True,'estimated_processing_seconds':100,
            'processing_budget_seconds':7200,'safety_margin_fraction':.15,
            'blocks':[{'references':100,'country_stratified':True,'seconds':{s:1. for s in stages}}]*2}


def test_runtime_blocks_are_not_optional_for_promotion():
    from evaluate_sprint import check_runtime
    valid=runtime_fixture();assert check_runtime(valid)['passed']
    bad={**valid,'blocks':valid['blocks'][:1]}
    assert not check_runtime(bad)['passed']
    assert not check_runtime({**valid,'synthetic_fixture_runtime':True})['passed']
    assert not check_runtime({**valid,'estimated_processing_seconds':8000})['passed']


def test_development_quality_and_fresh_promotion_are_distinct():
    truth={'S1-a':['S2-t']};base={'S1-a':['S2-t','S3-fp']};cand={'S1-a':['S2-t']};country={'S1-a':'US'}
    runtime=runtime_fixture();fresh={'reservation_verified':True,'candidate_preselected':True}
    dev=paired_evaluate(truth,base,cand,country,iterations=100,runtime=runtime,freshness=fresh)
    assert dev['acceptance']['quality_improvement_pass'] and not dev['acceptance']['promotion_pass']
    confirmation=paired_evaluate(truth,base,cand,country,iterations=100,role='confirmation',runtime=runtime,freshness=fresh)
    assert confirmation['acceptance']['promotion_pass']
    unknown=paired_evaluate(truth,base,cand,country,iterations=100,role='confirmation',runtime=runtime)
    assert not unknown['acceptance']['promotion_pass']


def test_incomplete_ownership_can_report_quality_without_promoting():
    result = paired_evaluate({'S1-a':['S2-t']}, {'S1-a':['S2-t','S3-fp']}, {'S1-a':['S2-t']},
                             {'S1-a':'US'}, iterations=100, ownership=True,
                             universe={'kind':'complete_development','complete':True})
    assert result['acceptance']['quality_improvement_pass']
    assert result['acceptance']['quality_non_inferiority_pass']
    assert not result['acceptance']['ownership_scope_eligible']
    assert not result['acceptance']['promotion_pass']


def test_audit_runtime_must_match_frozen_candidate():
    from evaluate_sprint import check_runtime
    evidence={**runtime_fixture(),'candidate_frozen_sha256':'one'}
    assert check_runtime(evidence,'one')['passed']
    assert not check_runtime(evidence,'two')['passed']


def test_complete_ownership_requires_scored_universe_not_only_declared_records():
    universe={'kind':'complete_heldout','complete':True,'supervised_training_excluded':True,'scoring_complete':True,'declared_references':2,'scored_references':2}
    kwargs={'truth':{'S1-a':[]},'baseline':{'S1-a':[]},'candidate':{'S1-a':[]},'countries':{'S1-a':'US'},'iterations':100,'ownership':True}
    valid=paired_evaluate(**kwargs,universe=universe)
    assert valid['acceptance']['non_inferiority_pass']
    for bad in ({**universe,'scoring_complete':False},{**universe,'scored_references':1}):
        assert 'global_ownership_universe_not_complete_and_heldout' in paired_evaluate(**kwargs,universe=bad)['acceptance']['failures']
