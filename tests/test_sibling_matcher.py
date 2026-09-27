import importlib.util
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("sibling_matcher", ROOT / "scripts/train_sibling_matcher.py")
sibling = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = sibling
spec.loader.exec_module(sibling)
role_spec = importlib.util.spec_from_file_location("sibling_roles", ROOT / "research/sprint_6h/sibling/build_role_workbenches.py")
roles_module = importlib.util.module_from_spec(role_spec)
role_spec.loader.exec_module(roles_module)
learned_spec = importlib.util.spec_from_file_location("learned_roles", ROOT / "research/sprint_6h/sibling/build_learned_workbench.py")
learned_module = importlib.util.module_from_spec(learned_spec)
learned_spec.loader.exec_module(learned_module)


def pairs():
    rows = []
    for i in range(10):
        for j, p in enumerate((.999, .70, .25, .05, .01, .40)):
            rows.append({"source1_entity_id": f"S1-{i}", "candidate_entity_id": f"S{2+j%2}-{i}-{j}",
                "probability": p, "first_stage": .999 if j == 0 else p, "accepted": p >= .83})
    return pd.DataFrame(rows)


def test_route_caps_and_cannot_support_itself():
    frame = pairs()
    route = sibling.select_route(frame)
    assert len({frame.source1_entity_id.iat[i] for i in route}) == 2
    assert len(route) <= 8
    assert sum(map(len, route.values())) <= 16
    for i, peers in route.items():
        assert all(frame.candidate_entity_id.iat[i] != frame.candidate_entity_id.iat[j] for j in peers)
    with pytest.raises(ValueError, match="budget"):
        sibling.select_route(frame, sibling.RouteConfig(fraction=.21))


def test_route_does_not_use_labels_and_order_is_deterministic():
    frame = pairs()
    frame["label"] = np.arange(len(frame)) % 2
    route = sibling.select_route(frame)
    selected = {(frame.source1_entity_id.iat[i], frame.candidate_entity_id.iat[i]) for i in route}
    shuffled = frame.sample(frac=1, random_state=42).reset_index(drop=True)
    shuffled.label = 1-shuffled.label
    other = sibling.select_route(shuffled)
    assert selected == {(shuffled.source1_entity_id.iat[i], shuffled.candidate_entity_id.iat[i]) for i in other}


def test_pinned_global_route_survives_subset_and_rejects_candidate_universe_change(tmp_path):
    frame = pairs()
    config = sibling.RouteConfig()
    route = sibling.select_route(frame)
    plan = tmp_path / "route"
    plan.mkdir()
    sibling.route_edges(frame, route).to_parquet(plan / "route.parquet", index=False)
    global_hash = sibling.pair_order_hash(frame)
    sibling.write_json(plan / "manifest.json", {"status": "complete", "route": sibling.asdict(config),
        "global_references": 10, "global_pair_order_sha256": global_hash,
        "route_sha256": sibling.digest(plan / "route.parquet")})
    owner = frame.source1_entity_id.iat[next(iter(route))]
    subset = frame[frame.source1_entity_id == owner].reset_index(drop=True)
    assert sibling.select_route(subset) == {}  # A one-reference local 20% cap would erase its global route.
    pinned, _ = sibling.load_pinned_route(subset, {"routing_universe_pair_order_sha256": global_hash}, plan)
    assert pinned
    expected = {(frame.candidate_entity_id.iat[i], frame.candidate_entity_id.iat[j])
        for i, peers in route.items() if frame.source1_entity_id.iat[i] == owner for j in peers}
    actual = {(subset.candidate_entity_id.iat[i], subset.candidate_entity_id.iat[j]) for i, peers in pinned.items() for j in peers}
    assert actual == expected
    with pytest.raises(ValueError, match="universe"):
        sibling.load_pinned_route(subset, {"routing_universe_pair_order_sha256": "wrong"}, plan)
    with pytest.raises(ValueError, match="complete reference"):
        sibling.load_pinned_route(subset[subset.probability < .99].reset_index(drop=True),
            {"routing_universe_pair_order_sha256": global_hash}, plan)


def test_zero_candidate_owners_are_counted_in_route_and_metric():
    frame = pairs()
    ids = [f"S1-{i}" for i in range(20)]
    route = sibling.select_route(frame, universe_ids=ids)
    assert len({frame.source1_entity_id.iat[i] for i in route}) == 4
    truth = {e: [] for e in ids}
    truth["S1-19"] = ["S2-unretrieved"]
    result = sibling.opportunity_audit(frame, truth, route)
    assert result["references"] == 20
    assert result["baseline_macro_f05"] == pytest.approx(9/20)


def test_perfect_route_changes_only_allowed_candidates():
    frame = pd.DataFrame([{"source1_entity_id": "a", "candidate_entity_id": "x", "accepted": False},
        {"source1_entity_id": "a", "candidate_entity_id": "wrong", "accepted": True},
        {"source1_entity_id": "b", "candidate_entity_id": "z", "accepted": False}])
    truth = {"a": ["x", "unretrieved"], "b": ["z"], "empty": []}
    result = sibling.opportunity_audit(frame, truth, {0: [1]}, {"a": "india", "b": "us", "empty": "us"})
    assert result["true_links_recoverable"] == 1
    assert result["false_links_correctable"] == 0
    assert result["route_perfect_correction_gain"] == pytest.approx((1.25/2.5)/3)
    assert result["country_gain"]["us"] == 0


def test_ideal_blend_diagnostic_respects_score_reach_and_predeclared_weights():
    from run_frozen_pipeline import decide
    frame = pd.DataFrame([{"source1_entity_id": "a", "candidate_entity_id": "S2-seed", "probability": .999},
        {"source1_entity_id": "a", "candidate_entity_id": "S3-hard", "probability": .50}])
    truth = {"a": ["S2-seed", "S3-hard"]}
    result = sibling.feasible_score_diagnostic(frame, truth, {1: [0]}, {"threshold": .83, "t_rest": .83, "t_first": .70}, decide)
    trials = result["trials"]
    assert [r["weight"] for r in trials] == [.25, .5, .75]
    assert trials[0]["gain_vs_same_cohort_policy"] == 0.
    assert trials[1]["gain_vs_same_cohort_policy"] == 0.
    assert trials[2]["gain_vs_same_cohort_policy"] > 0.
    assert [r["ordinary_positive_recovery_probability_floor"] for r in trials] == pytest.approx([.773333333, .66, .32])


def test_compatibility_features_are_symmetric_and_missing_is_not_contradiction():
    left = sibling.clean_target("École भारत", "12 rue de la paix", "S2")
    right = sibling.clean_target("Ecole भारत", "0012 paix rue", "S3")
    np.testing.assert_array_equal(sibling.sibling_features(left, right), sibling.sibling_features(right, left))
    result = dict(zip(sibling.FEATURE_NAMES, sibling.sibling_features(left, right)))
    assert result["numeric_jaccard"] == 1
    assert result["same_source"] == 0
    missing = sibling.clean_target("Ecole भारत", "", "S3")
    result = dict(zip(sibling.FEATURE_NAMES, sibling.sibling_features(left, missing)))
    assert result["numeric_disjoint"] == 0
    assert result["address_both_present"] == 0
    assert result["address_one_missing"] == 1
    assert "भारत" in left["name"]


def test_numeric_edit_types_and_phonetic_domain_views_preserve_primary_text():
    left = sibling.clean_target("मैक्स टेक", "123 road gui1len", "S2")
    dropped = sibling.clean_target("max tek com", "23 road hurtad0", "S3")
    result = dict(zip(sibling.FEATURE_NAMES, sibling.sibling_features(left, dropped)))
    assert result["numeric_disjoint"] == 1
    assert result["numeric_containment"] == 1
    assert result["numeric_relation_disjoint"] == 0
    assert result["collapsed_phonetic_ratio"] > result["name_ratio"]
    assert "मैक्स टेक" == left["name"]
    assert "com" in dropped["name"]
    substituted = sibling.clean_target("max tek", "124 road", "S3")
    result = dict(zip(sibling.FEATURE_NAMES, sibling.sibling_features(left, substituted)))
    assert result["numeric_single_substitution"] == 1
    assert left["canonical_digits"] == {"123"}
    np.testing.assert_array_equal(sibling.sibling_features(left, dropped), sibling.sibling_features(dropped, left))


def test_learned_claimant_workbench_excludes_residual_owners_and_recomputes_control(tmp_path):
    frame = pd.DataFrame([{"source1_entity_id": "fit", "candidate_entity_id": "S2-x", "probability": .999,
        "first_stage": .999, "accepted": True}, {"source1_entity_id": "selection", "candidate_entity_id": "S2-x",
        "probability": .99, "first_stage": .99, "accepted": False}])
    marker = {"pairs_sha256": "source50k", "reference_ids": ["fit", "selection"], "verified_current_pipeline": True}
    result = learned_module.make_learned(frame, marker, {"selection": ["S2-x"]},
        {"selection": "india"}, ["selection"], ["selection"], ["fit"],
        {"threshold": .83, "t_rest": .83, "t_first": .7}, tmp_path / "learned")
    assert result["evaluation_expected_macro_f05"] == 1.
    output = pd.read_parquet(tmp_path / "learned/pairs.parquet")
    assert output.source1_entity_id.tolist() == ["selection"]
    assert output.accepted.iat[0] and not output.accepted_source50k.iat[0]


def test_seed_errors_are_retained_as_natural_negative_pairs():
    frame = pd.DataFrame([{"source1_entity_id": "a", "candidate_entity_id": "S2-a", "first_stage": .7, "probability": .7},
        {"source1_entity_id": "a", "candidate_entity_id": "S3-b", "first_stage": .999, "probability": .999},
        {"source1_entity_id": "b", "candidate_entity_id": "S3-b", "first_stage": .999, "probability": .999}])
    truth = {"a": ["S2-a", "S3-a"], "b": ["S3-b"]}
    records = {c: sibling.clean_target("same name", "12 street", c[:2]) for targets in truth.values() for c in targets}
    examples = sibling.training_examples(frame, {0: [1]}, truth, records)
    mined = examples[examples.kind == "oof_neighborhood"]
    assert len(mined) == 1 and mined.label.iat[0] == 0
    assert {mined.left_owner.iat[0], mined.right_owner.iat[0]} == {"a", "b"}


def test_unknown_owner_does_not_become_negative_or_training_owner():
    frame = pd.DataFrame([{"source1_entity_id": "a", "candidate_entity_id": "S2-a"},
        {"source1_entity_id": "a", "candidate_entity_id": "S3-unknown"}])
    truth = {"a": ["S2-a", "S3-a"], "outside": ["S2-other", "S3-other"]}
    records = {c: sibling.clean_target("name", "street", c[:2]) for targets in truth.values() for c in targets}
    examples = sibling.training_examples(frame, {0: [1]}, truth, records)
    assert len(examples) == 1 and examples.label.iat[0] == 1
    assert set(examples.left_owner) == {"a"}


def test_owner_database_extends_natural_negatives_only_with_outer_train_labels(tmp_path):
    database = tmp_path / "owners.sqlite"
    with sqlite3.connect(database) as conn:
        conn.execute("CREATE TABLE owners(id TEXT PRIMARY KEY, owner TEXT, outer_fold TEXT)")
        conn.executemany("INSERT INTO owners VALUES(?,?,?)", [("S2-a", "a", "train"),
            ("S3-other", "outside", "train"), ("S3-dev", "dev", "tune"), ("S3-audit", "audit", "holdout")])
    external = sibling.lookup_training_owners(["S2-a", "S3-other", "S3-dev", "S3-audit", "S3-unknown"], database)
    assert external == {"S2-a": "a", "S3-other": "outside"}
    frame = pd.DataFrame([{"source1_entity_id": "a", "candidate_entity_id": "S2-a"},
        {"source1_entity_id": "a", "candidate_entity_id": "S3-other"}])
    truth = {"a": ["S2-a", "S3-a"]}
    records = {c: sibling.clean_target("same", "12 street", c[:2]) for c in ["S2-a", "S3-a", "S3-other"]}
    examples = sibling.training_examples(frame, {0: [1]}, truth, records, external_training_owners=external)
    assert len(examples) == 2
    assert examples.label.tolist() == [1, 0]
    assert "outside" in set(examples.right_owner)


def test_fold_fit_excludes_either_target_owner_and_validation_is_owner_disjoint():
    examples = pd.DataFrame({"left_owner": ["a", "b", "c", "d"], "right_owner": ["a", "c", "e", "d"],
        "left_fold": [4, 2, 4, 1], "right_fold": [4, 4, 3, 1]})
    fit, held = sibling.fold_masks(examples, 4)
    assert fit.tolist() == [False, False, False, True]
    assert held.tolist() == [True, False, False, False]


def test_support_preserves_baseline_keys_and_batches_model_prediction():
    frame = pairs()
    route = sibling.select_route(frame)
    records = {c: sibling.clean_target("name", "12 street", c[:2]) for c in frame.candidate_entity_id}
    class Model:
        calls = 0
        def predict(self, matrix, num_threads):
            self.calls += 1
            assert matrix.shape[1] == len(sibling.FEATURE_NAMES)
            return np.full(len(matrix), .8)
    model = Model()
    support = sibling.support_features(frame, route, records, model)
    assert model.calls == 1
    assert support[sibling.KEYS].equals(frame[sibling.KEYS])
    assert (support.loc[list(route), "sibling_max"] == np.float32(.8)).all()
    assert (support.drop(index=list(route)).sibling_count == 0).all()


def test_probability_table_requires_keys_and_owner_excluded_oof(tmp_path):
    frame = pairs()
    frame["oof_excluded_fold"] = frame.source1_entity_id.map(sibling.inner_fold)
    path = tmp_path / "pairs.parquet"
    frame.to_parquet(path, index=False)
    manifest = {"status": "complete", "verified_current_pipeline": True, "split": "train",
        "pairs_sha256": sibling.digest(path), "pair_order_sha256": sibling.pair_order_hash(frame),
        "first_stage_owner_excluded_oof": True}
    marker = tmp_path / "manifest.json"
    marker.write_text(json.dumps(manifest))
    sibling.load_pairs(path, marker, training=True)
    frame.loc[0, "oof_excluded_fold"] = (frame.oof_excluded_fold.iat[0]+1)%5
    frame.to_parquet(path, index=False)
    manifest["pairs_sha256"] = sibling.digest(path)
    marker.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="owner's fold"):
        sibling.load_pairs(path, marker, training=True)
    manifest["pairs_sha256"] = "wrong"
    marker.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="hash"):
        sibling.load_pairs(path, marker, training=True)


def test_support_rejects_equal_length_wrong_keys(tmp_path):
    frame = pairs()
    support = frame[sibling.KEYS].iloc[::-1].reset_index(drop=True)
    for name in sibling.SUPPORT_NAMES: support[name] = 0.
    path = tmp_path / "support.parquet"
    support.to_parquet(path, index=False)
    marker = tmp_path / "manifest.json"
    marker.write_text(json.dumps({"support_sha256": sibling.digest(path), "pair_order_sha256": sibling.pair_order_hash(frame),
        "feature_names": sibling.SUPPORT_NAMES}))
    with pytest.raises(ValueError, match="keys"):
        sibling.aligned_support(frame, path, marker)


def test_audit_inputs_allow_inference_but_never_open_truth(tmp_path, monkeypatch):
    frame = pairs()
    path = tmp_path / "pairs.parquet"
    frame.to_parquet(path, index=False)
    marker = tmp_path / "manifest.json"
    sibling.write_json(marker, {"status": "complete", "split": "audit", "verified_current_pipeline": True,
        "pairs_sha256": sibling.digest(path), "pair_order_sha256": sibling.pair_order_hash(frame)})
    sibling.load_pairs(path, marker)
    truth = tmp_path / "sealed_truth.json"
    truth.write_text("This file is deliberately not parseable and must never be read.")
    monkeypatch.setattr(sys, "argv", ["matcher", "support", "--pairs", str(path), "--manifest", str(marker),
        "--truth", str(truth), "--output", str(tmp_path / "out")])
    with pytest.raises(ValueError, match="label-free"):
        sibling.main()


def test_role_subset_preserves_outside_role_claims_and_zero_candidate_entities(tmp_path):
    frame = pd.DataFrame([{"source1_entity_id": "a", "candidate_entity_id": "S2-x", "probability": .99,
        "first_stage": .99, "accepted": False}, {"source1_entity_id": "b", "candidate_entity_id": "S2-x",
        "probability": .999, "first_stage": .999, "accepted": True}])
    manifest = {"status": "complete", "verified_current_pipeline": True, "split": "development",
        "pairs_sha256": "parent-table-hash", "reference_ids": ["a", "b", "c"]}
    reports = roles_module.build_roles(frame, manifest, {"a": ["S2-x"], "b": [], "c": []},
        {"a": "india", "b": "us", "c": "us"}, {"selection": ["a"], "residual_train": ["b"], "early_stop": ["c"]},
        tmp_path / "roles", "parent-manifest-hash")
    part, marker = sibling.load_pairs(tmp_path / "roles/selection/pairs.parquet", tmp_path / "roles/selection/manifest.json")
    assert not part.accepted.iat[0]
    assert marker["claims_universe_entities"] == 3
    assert marker["historical_full_ownership_replayed"] is False
    assert reports["selection"]["expected_macro_f05"] == 0.
    assert reports["early_stop"]["expected_macro_f05"] == 1.


def test_truth_lookup_uses_read_only_owner_database_and_keeps_empty_owners(tmp_path):
    database = tmp_path / "owners.sqlite"
    with sqlite3.connect(database) as conn:
        conn.execute("CREATE TABLE owners(id TEXT PRIMARY KEY, owner TEXT, outer_fold TEXT)")
        conn.executemany("INSERT INTO owners VALUES(?,?,?)", [("S2-a", "a", "tune"),
            ("S3-a", "a", "tune"), ("S3-sealed", "sealed", "holdout")])
    before = sibling.digest(database)
    assert roles_module.read_owner_truth(["a", "empty"], database) == {"a": ["S2-a", "S3-a"], "empty": []}
    assert sibling.digest(database) == before


def test_cli_disjoint_component_residual_and_inference_contract(tmp_path, monkeypatch):
    truths, tables = {}, {}
    record_rows = {"S2": [], "S3": []}
    for role in ("train", "dev"):
        truth, rows = {}, []
        for i in range(20):
            owner = f"{role}-{i}"
            ids = [f"S2-{role}-{i}-a", f"S3-{role}-{i}-b", f"S2-{role}-{i}-c"]
            truth[owner] = ids
            for cid in ids: record_rows[cid[:2]].append((cid, f"name {i}", f"{i+1} road"))
            candidates = [ids[0], ids[1], f"S2-{role}-{(i+1)%20}-a", ids[2]]
            for cid, p in zip(candidates, [.999, .7, .6, .02]):
                rows.append({"source1_entity_id": owner, "candidate_entity_id": cid,
                    "first_stage": p, "probability": p, "accepted": p >= .83,
                    "oof_excluded_fold": sibling.inner_fold(owner)})
        frame = pd.DataFrame(rows)
        if role == "train": frame.probability = .999  # In-sample p2 must not control negative mining.
        table = tmp_path / f"{role}.parquet"
        marker = tmp_path / f"{role}_manifest.json"
        truth_file = tmp_path / f"{role}_truth.json"
        frame.to_parquet(table, index=False)
        truth_file.write_text(json.dumps(truth))
        sibling.write_json(marker, {"status": "complete", "split": "train" if role == "train" else "development",
            "role": "component_train" if role == "train" else "residual_train",
            "verified_current_pipeline": True, "first_stage_owner_excluded_oof": role == "train",
            "pairs_sha256": sibling.digest(table), "pair_order_sha256": sibling.pair_order_hash(frame),
            "reference_ids": list(truth)})
        tables[role] = (table, marker)
        truths[role] = truth_file
    prefix = tmp_path / "index"
    for source, rows in record_rows.items():
        with sqlite3.connect(f"{prefix}_{source}.sqlite") as conn:
            conn.execute("CREATE TABLE records(id TEXT, name TEXT, address TEXT)")
            conn.executemany("INSERT INTO records VALUES(?,?,?)", rows)
    def run(command, *args):
        monkeypatch.setattr(sys, "argv", ["train_sibling_matcher.py", command, *map(str, args)])
        sibling.main()
    prepared, model, support, residual, scored = [tmp_path / n for n in ("prepared", "model", "support", "residual", "scored")]
    run("prepare", "--pairs", tables["train"][0], "--manifest", tables["train"][1], "--truth", truths["train"],
        "--index-prefix", prefix, "--output", prepared)
    prepared_manifest = json.loads((prepared / "manifest.json").read_text())
    assert prepared_manifest["negatives"] > 0
    assert prepared_manifest["route_probability_source"] == "owner_excluded_oof_first_stage"
    assert pd.read_parquet(tables["train"][0]).probability.eq(.999).all()
    run("fit", "--examples", prepared / "examples.parquet", "--trees", 2, "--output", model)
    run("support", "--pairs", tables["dev"][0], "--manifest", tables["dev"][1],
        "--model-file", model / "model.txt", "--index-prefix", prefix, "--output", support)
    support_manifest = json.loads((support / "manifest.json").read_text())
    assert support_manifest["owner_excluded"]
    run("fit-residual", "--pairs", tables["dev"][0], "--manifest", tables["dev"][1], "--truth", truths["dev"],
        "--support", support / "support.parquet", "--support-manifest", support / "manifest.json", "--trees", 2,
        "--output", residual)
    run("rescore", "--pairs", tables["dev"][0], "--manifest", tables["dev"][1],
        "--support", support / "support.parquet", "--support-manifest", support / "manifest.json",
        "--model-dir", residual, "--output", scored)
    result = pd.read_parquet(scored / "pairs.parquet")
    original = pd.read_parquet(tables["dev"][0])
    assert result[sibling.KEYS].equals(original[sibling.KEYS])
    routed = pd.read_parquet(support / "support.parquet").sibling_count > 0
    np.testing.assert_array_equal(result.loc[~routed, "probability"], original.loc[~routed, "probability"])
    assert (result.probability_original == original.probability).all()


def test_anchored_offset_zero_control_and_reload(tmp_path):
    import importlib.util
    import lightgbm as lgb
    path = Path(__file__).resolve().parents[1] / "research/sprint_6h/sibling/train_anchored_adapter.py"
    spec = importlib.util.spec_from_file_location("anchored_adapter", path)
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    base = np.array([.1, .5, .8, .99])
    np.testing.assert_allclose(adapter.corrected_probability(base, np.zeros(4)), base, atol=1e-15)
    assert len(adapter.FEATURES) == 113
    # Empirical labels .8 match init_score .8; zero correction must preserve .8.
    x = np.zeros((100, 113), dtype=np.float32)
    y = np.r_[np.ones(80), np.zeros(20)]
    model = lgb.train(dict(objective="binary", verbosity=-1, num_threads=1, boost_from_average=False),
        lgb.Dataset(x, label=y, init_score=adapter.base_logit(np.full(100, .8)), feature_name=adapter.FEATURES), num_boost_round=3)
    delta = model.predict(x, raw_score=True, num_threads=1)
    np.testing.assert_allclose(delta, 0., atol=1e-10)
    np.testing.assert_allclose(adapter.corrected_probability(np.full(100, .8), delta), .8, atol=1e-10)
    assert np.allclose(model.predict(x, num_threads=1), .5)  # Anchor is not serialized in Booster.
    model.save_model(str(tmp_path / "adapter.txt"))
    reloaded = lgb.Booster(model_file=str(tmp_path / "adapter.txt"))
    np.testing.assert_array_equal(delta, reloaded.predict(x, raw_score=True, num_threads=1))


def test_anchored_raw_features_labels_independent_no_self_and_chunk_parity():
    import importlib.util
    path = Path(__file__).resolve().parents[1] / "research/sprint_6h/sibling/train_anchored_adapter.py"
    spec = importlib.util.spec_from_file_location("anchored_adapter_raw", path)
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    frame = pd.DataFrame([{"source1_entity_id": e, "candidate_entity_id": c, "first_stage": p,
        "probability": q, "candidate_order": k, "accepted": False, "label": 0}
        for e in ["a", "b"] for k,(c,p,q) in enumerate([(f"S2-{e}",.95,.999),(f"S3-{e}",.6,.51)])])
    queries = {e: sibling.clean_target("name "+e, "1 road", "S1") for e in ["a", "b"]}
    targets = {c: sibling.clean_target("name "+c[-1], "1 road", c[:2]) for c in frame.candidate_entity_id}
    route = {1:[0], 3:[2]}
    full = adapter.raw_features(frame, route, queries, targets)
    changed = frame.assign(label=1, accepted=True)
    pd.testing.assert_frame_equal(full, adapter.raw_features(changed,route,queries,targets))
    chunk = adapter.raw_features(frame.iloc[2:].reset_index(drop=True),{1:[0]},queries,targets)
    pd.testing.assert_frame_equal(full.iloc[1:].reset_index(drop=True),chunk)
    with pytest.raises(ValueError, match="Selfsupport"):
        adapter.raw_features(frame,{1:[1]},queries,targets)
    class ZeroCorrection:
        def predict(self, x, **kwargs):
            assert kwargs["raw_score"] is True
            return np.zeros(len(x))
    # Keyed scatter remains correct when feature rows are reversed.
    scored = adapter.score_frame(frame,full.iloc[::-1],ZeroCorrection())
    np.testing.assert_allclose(scored.probability,frame.probability,atol=1e-15)
    pd.testing.assert_frame_equal(scored[sibling.KEYS],frame[sibling.KEYS])
    foreign = full.copy(); foreign.loc[0,"candidate_entity_id"]="S2-foreign"
    with pytest.raises(ValueError,match="outsidepairuniverse"):
        adapter.score_frame(frame,foreign,ZeroCorrection())


def reverse_adapter_module():
    import importlib.util
    path = Path(__file__).resolve().parents[1] / 'research/sprint_6h/sibling/train_reverse_adapter.py'
    spec = importlib.util.spec_from_file_location('reverse_adapter_fixture', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reverse_join_uses_all_p2_candidates_and_fixed_numeric_schema():
    adapter = reverse_adapter_module()
    frame = pd.DataFrame({'source1_entity_id':['a']*4,'candidate_entity_id':['S2-z','S2-a','S3-b','S3-c'],
        'first_stage':[.95,.9,.4,.8],'probability':[.8,.8,.7,.95],'candidate_order':[0,1,2,3]})
    assert adapter.frozen_p2_rank(frame).tolist()==[3,2,4,1]
    route={0:[1],2:[1]}
    reverse = pd.DataFrame([{**dict(zip(sibling.KEYS,('a',c))),**{n:0. for n in adapter.REVERSE_FEATURES},
        'rival1_id':'private-rival-id','rival_trained_score':.987} for c in ['S3-b','S2-z']])
    joined=adapter.join_reverse(frame,reverse,route)
    assert len(adapter.FEATURES)==36
    assert list(joined.columns)==[*sibling.KEYS,*adapter.FEATURES]
    assert joined.candidate_rank.tolist()==[3,4]
    pd.testing.assert_frame_equal(joined,adapter.join_reverse(frame.assign(label=1),reverse.iloc[::-1],route))
    with pytest.raises(ValueError,match='exactlycover'):
        adapter.join_reverse(frame,reverse.iloc[:1],route)
    duplicate=pd.concat([reverse,reverse.iloc[:1]],ignore_index=True)
    with pytest.raises(ValueError,match='Duplicate'):
        adapter.join_reverse(frame,duplicate,route)
    bad=reverse.copy();bad.loc[0,adapter.REVERSE_FEATURES[0]]=np.nan
    with pytest.raises(ValueError,match='Nonfinite'):
        adapter.join_reverse(frame,bad,route)


def test_reverse_anchor_reload_scatter_and_complete_reference_chunk(tmp_path):
    import lightgbm as lgb
    adapter=reverse_adapter_module()
    frame=pd.DataFrame({'source1_entity_id':['a','a','b','b'],'candidate_entity_id':['S2-a','S3-a','S2-b','S3-b'],
        'first_stage':[.99,.5,.99,.5],'probability':[.99,.6,.99,.6]})
    reverse=pd.DataFrame([{**dict(zip(sibling.KEYS,(e,c))),**{n:0. for n in adapter.REVERSE_FEATURES}}
        for e,c in [('a','S3-a'),('b','S3-b')]])
    features=adapter.join_reverse(frame,reverse,{1:[0],3:[2]})
    x=np.zeros((100,36),dtype=np.float32);y=np.r_[np.ones(80),np.zeros(20)]
    model=lgb.train(dict(objective='binary',verbosity=-1,num_threads=1,boost_from_average=False),
        lgb.Dataset(x,label=y,init_score=adapter.base_logit(np.full(100,.8)),feature_name=adapter.FEATURES),num_boost_round=3)
    model.save_model(str(tmp_path/'model.txt'));loaded=lgb.Booster(model_file=str(tmp_path/'model.txt'))
    np.testing.assert_array_equal(model.predict(x,raw_score=True,num_threads=1),loaded.predict(x,raw_score=True,num_threads=1))
    full=adapter.score_frame(frame,features.iloc[::-1],loaded)
    np.testing.assert_array_equal(full.probability,frame.probability)
    chunk=frame.iloc[2:].reset_index(drop=True)
    chunk_features=adapter.join_reverse(chunk,reverse.iloc[1:],{1:[0]})
    chunk_score=adapter.score_frame(chunk,chunk_features,loaded)
    pd.testing.assert_frame_equal(full.iloc[2:].reset_index(drop=True),chunk_score)


def test_reverse_manifest_requires_safe_completed_index_and_self_exclusion(tmp_path):
    adapter=reverse_adapter_module()
    index=tmp_path/'index_manifest.json'
    meta={'complete':True,'fingerprint':{'corpus':'train','source_sha256':'1'*64,'source_size':1,
        'version':'reverse-s1-v1','normalization':'shared-normalize/accent-fold/core/phonetic;fts-terms-v1'},
        'records':2206821,'labels_read':False,'test_pseudo_labels':False,'country_counts':{'us':2206821},'query_config':adapter.QUERY_CONFIG}
    root=Path(__file__).resolve().parents[1];code=root/'code/business_entity_resolution/src'
    import hashlib
    meta.update(index_sha256='4'*64,build_code_sha256=sibling.digest(root/'scripts/reverse_competition.py'),
        normalization_sha256=sibling.digest(code/'preprocessing/normalize.py'),phonetic_code_sha256=sibling.digest(code/'features/evidence.py'),
        df_policy_sha256=hashlib.sha256(json.dumps(adapter.QUERY_CONFIG,sort_keys=True).encode()).hexdigest())
    sibling.write_json(index,meta)
    reverse=pd.DataFrame([{**dict(zip(sibling.KEYS,('a','S2-a'))),**{n:0. for n in adapter.REVERSE_FEATURES}}])
    path=tmp_path/'reverse.parquet';reverse.to_parquet(path,index=False)
    edges=tmp_path/'route.parquet';pd.DataFrame([{'source1_entity_id':'a','candidate_entity_id':'S2-a','peer_entity_id':'S3-a'}]).to_parquet(edges,index=False)
    route_marker=tmp_path/'route_manifest.json';sibling.write_json(route_marker,{'status':'complete','route':sibling.asdict(sibling.RouteConfig()),
        'global_pair_order_sha256':'source-order-hash','route_sha256':sibling.digest(edges)})
    diag=tmp_path/'diagnostics.jsonl';diag.write_text(json.dumps({'source1_entity_id':'a','candidate_entity_id':'S2-a','rival_ids':['b']})+'\n')
    source={'split':'development','pairs_sha256':'source-pair-hash','pair_order_sha256':'source-order-hash'}
    root=Path(__file__).resolve().parents[1];code=root/'code/business_entity_resolution/src'
    manifest={'status':'complete','feature_version':'reverse-s1-v1','features':adapter.REVERSE_FEATURES,'features_sha256':sibling.digest(path),
        'pair_order_sha256':sibling.pair_order_hash(reverse),'labels_read':False,'trained_rival_scores_used':False,'audit_labels_read':False,'test_pseudo_labels':False,
        'rival_ids_as_features':False,'own_id_excluded':True,'source_pairs_sha256':'source-pair-hash',
        'source_pair_order_sha256':'source-order-hash','route_manifest_sha256':sibling.digest(route_marker),'route_manifest':str(route_marker),
        'index_manifest':str(index),'index_manifest_sha256':sibling.digest(index),'index_metadata':meta,'diagnostics_sha256':sibling.digest(diag),
        'source_tsv_sha256':{'S1':'1'*64,'S2':'2'*64,'S3':'3'*64},'query_config':adapter.QUERY_CONFIG,'feature_code_sha256':sibling.digest(root/'scripts/reverse_competition.py'),
        'normalization_sha256':sibling.digest(code/'preprocessing/normalize.py'),'phonetic_code_sha256':sibling.digest(code/'features/evidence.py'),
        'target_index_metadata':{'S2':{'complete':True},'S3':{'complete':True}}}
    assert adapter.validate_reverse_manifest(reverse,path,manifest,source,None)['split']=='train'
    with pytest.raises(ValueError,match='leakage'):
        adapter.validate_reverse_manifest(reverse,path,{**manifest,'trained_rival_scores_used':True},source,None)
    diag.write_text(json.dumps({'source1_entity_id':'a','candidate_entity_id':'S2-a','rival_ids':['a']})+'\n')
    with pytest.raises(ValueError,match='Selfrival'):
        adapter.validate_reverse_manifest(reverse,path,{**manifest,'diagnostics_sha256':sibling.digest(diag)},source,None)
