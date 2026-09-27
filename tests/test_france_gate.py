import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from france_gate import address_view, fit_signals, gate_veto, pair_signals, reduced_name


def record(entity_id, name="Atlas Ltd", address="12 rue Victor Hugo", country="open label", **extra):
    return {"entity_id": entity_id, "business_name": name, "business_address": address,
            "country": country, **extra}


def universe():
    rows = [record("S1_1"), record("S1_2", address="24 rue Jean Jaures"),
            record("S1_3", address="16 rue Paul Verlaine")]
    # Diverse supplied contexts support the structural prefix hypothesis. These
    # unrelated names do not inflate Atlas's owner-ambiguity count.
    rows += [record(f"S1_support_{i}", name=f"Unique {i}", address=f"{40+i} rue {word} Gardens")
             for i, word in enumerate(("William", "Arthur", "Henry", "George", "Thomas"))]
    return rows


def signals(target, rows=None):
    rows = universe() if rows is None else rows
    return pair_signals(rows[0], target, fit_signals({"open label": rows}))


def test_distinct_record_owners_and_exact_self_exclusion():
    result = signals(record("S2_1", address="38 rue Mac Arthur"))
    assert result["reference_original_name_rivals"] == 2
    assert result["target_reduced_name_rivals"] == 2
    assert result["street_overlap"] == 0
    assert gate_veto(result)["gate_veto"]


def test_unknown_and_missing_address_cannot_veto():
    for address in ("", None, "75001 Paris", "some words without supported syntax"):
        result = signals(record("S2_1", address=address))
        assert not gate_veto(result)["veto"]
    assert signals(record("S2_1", address=None))["address_overlap"] is None


def test_numeric_deletion_and_house_disagreement_preserve_same_street():
    for address in ("rue Victor Hugo", "21 rue Victor Hugo", "12b rue Victor Hugo"):
        result = signals(record("S2_1", address=address))
        assert result["street_overlap"] == 1
        assert not gate_veto(result)["veto"]
    dropped = signals(record("S2_1", address="rue Victor Hugo"))
    assert dropped["house_number_match"] is None
    assert signals(record("S2_1", address="21 rue Victor Hugo"))["house_number_conflict"]
    assert signals(record("S2_1", address="12b rue Victor Hugo"))["house_suffix_match"] is False


def test_postcode_is_not_a_house_number_and_raw_text_survives():
    target = record("S2_1", address="75001 rue Victor Hugo, Paris")
    result = signals(target)
    assert result["target_address"]["house_number"] is None
    assert result["target_address"]["postcodes"] == ["75001"]
    assert result["target_address"]["raw"] == target["business_address"]
    assert not gate_veto(result)["veto"]
    assert address_view(record("x", address="building 7 main street"))["house_number"] is None


def test_suffix_street_template_is_learned_from_supplied_records():
    rows = [record("S1_1", address="12 Blue Oak Road"), record("S1_2", address="23 Cedar Pine Road"),
            record("S1_3", address="41 Purple Elm Road")]
    stats = fit_signals({"open label": rows})
    assert stats["countries"]["open label"]["street_templates"]["suffix"] == ["road"]
    assert pair_signals(rows[0], record("S2_1", address="Blue Oak Road"), stats)["street_overlap"] == 1
    assert gate_veto(pair_signals(rows[0], record("S2_1", address="8 Red Birch Road"), stats))["veto"]


def test_shared_building_and_name_overlap_do_not_delete_street_content():
    rows = universe()
    rows[0]["business_name"] = "Victor Hugo Ltd"
    rows[1]["business_name"] = rows[2]["business_name"] = "Victor Hugo Ltd"
    result = signals(record("S2_1", name="Victor Hugo Ltd", address="21 rue Victor Hugo"), rows)
    assert result["reference_address"]["street_tokens"] == ["hugo", "victor"]
    assert result["reference_name_token_weights"] == {"victor": 0.5, "hugo": 0.5}
    assert not gate_veto(result)["veto"]


def test_frozen_legal_forms_only_strip_edges_and_empty_core_is_missing_key():
    assert reduced_name("private atlas limited") == "atlas"
    assert reduced_name("atlas private gardens") == "atlas private gardens"
    # No hand-written French vocabulary: this token remains in the base view.
    assert reduced_name("atlas sas") == "atlas sas"
    rows = universe()
    for row in rows[:3]:
        row["business_name"] = "Ltd"
    stats = fit_signals({"open label": rows})
    result = pair_signals(rows[0], record("S2_1", name="Ltd", address="38 rue Mac Arthur"), stats)
    assert result["reference_reduced_name"] == ""
    assert result["reference_reduced_name_rivals"] == -1
    assert "" not in stats["countries"]["open label"]["reduced_name_counts"]
    assert result["reference_original_name_rivals"] == 2


def test_indic_marks_and_latin_accents_are_preserved_in_appropriate_views():
    rows = universe()
    rows[0]["business_name"] = "श्री शक्ति"
    stats = fit_signals({"open label": rows})
    result = pair_signals(rows[0], record("S2_1", name="श्री शक्ति"), stats)
    assert result["reference_name"] == "श्री शक्ति"
    assert result["name_match"]
    accented = universe()
    accented[0]["business_name"] = "Café Ltd"
    result = signals(record("S2_1", name="Cafe Ltd"), accented)
    assert result["reference_name_raw"] == "Café Ltd"
    assert result["name_match"]


def test_no_name_key_missingness_is_distinct_from_zero_rivals():
    rows = universe()
    rows[0]["business_name"] = ""
    result = signals(record("S2_1", name=""), rows)
    assert result["reference_original_name_rivals"] == -1
    assert result["target_reduced_name_rivals"] == -1
    assert not result["name_match"]
    assert not gate_veto(result)["veto"]
    rows[0]["business_name"] = "Unique Ltd"
    assert signals(record("S2_1", name="Unique Ltd"), rows)["reference_original_name_rivals"] == 0


def test_fit_is_label_free_json_roundtrip_and_does_not_mutate_records():
    rows = universe()
    original = copy.deepcopy(rows)
    first = fit_signals({"open label": rows})
    for row in rows:
        row["truth_owner"] = "misleading shared supervised value"
    second = fit_signals({"open label": rows})
    assert first == second
    assert all(row[key] == original[i][key] for i, row in enumerate(rows) for key in original[i])
    restored = json.loads(json.dumps(first, sort_keys=True))
    assert pair_signals(rows[0], record("S2_1"), restored) == pair_signals(rows[0], record("S2_1"), first)


def test_outside_changed_duplicate_and_wrong_country_references_fail():
    rows = universe()
    stats = fit_signals({"open label": rows})
    for changed in (record("S1_other"), record("S1_1", name="changed")):
        with pytest.raises(ValueError, match="unchanged"):
            pair_signals(changed, record("S2_1"), stats)
    with pytest.raises(ValueError, match="unique"):
        fit_signals({"open label": rows + [rows[0]]})
    with pytest.raises(ValueError, match="Country"):
        fit_signals({"wrong country": rows})


def test_unproven_frequent_tokens_are_not_stripped_and_strict_core_is_auxiliary():
    rows = [record(f"S1_{i}", name=f"Common {word}") for i, word in enumerate(("Alpha", "Beta", "Gamma"))]
    state = fit_signals({"open label": rows})["countries"]["open label"]
    assert state["learned_edge_tokens"] == []
    bases = ["Alpha Factory", "Beta Factory", "Gamma Factory"]
    rows = []
    for base in bases:
        rows.extend([record(f"S1_{len(rows)}", name=base), record(f"S1_{len(rows)+1}", name=base),
                     record(f"S1_{len(rows)+2}", name=base + " Affix")])
    stats = fit_signals({"open label": rows})
    assert stats["countries"]["open label"]["learned_edge_tokens"] == ["affix"]
    result = pair_signals(rows[2], record("S2_1", name=bases[0]), stats)
    assert result["reference_learned_core"] == "alpha factory"
    assert result["reference_reduced_name"] == "alpha factory affix"
    assert not result["name_match"]
    assert not stats["learned_core_used_for_veto"]


def test_config_validation_and_veto_are_pure():
    result = signals(record("S2_1", address="38 rue Mac Arthur"))
    original = copy.deepcopy(result)
    assert not gate_veto(result, {"min_name_rivals": 3})["veto"]
    assert gate_veto(result)["veto"]
    assert result == original
    for config in ({"min_name_rivals": 0}, {"max_street_overlap": 1}, {"unknown": 1}):
        with pytest.raises(ValueError):
            gate_veto(result, config)


def test_orientation_conflict_requires_predeclared_tenfold_support():
    row = record("S1", address="12 marker Oak End")
    templates = {"prefix": ["marker"], "suffix": ["end"],
                 "support": {"prefix": {"marker": 100}, "suffix": {"end": 10}}}
    view = address_view(row, templates)
    assert view["street_reliable"]
    assert view["street_tokens"] == ["end", "oak"]
    assert view["raw"] == row["business_address"]
    templates["support"]["prefix"]["marker"] = 99
    assert not address_view(row, templates)["street_reliable"]
    multiple = record("S1", address="12 marker Oak End, 14 marker Cedar End")
    templates["support"]["prefix"]["marker"] = 100
    assert not address_view(multiple, templates)["street_reliable"]


def test_raw_lookup_is_atomic_preserves_text_and_rejects_changed_inputs(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research/sprint_6h/gate"))
    from build_raw_lookup import build_raw_lookup, lookup_manifest, lookup_raw_records, open_raw_lookup
    source = tmp_path / "source2.tsv"
    source.write_text("entity_id\tbusiness_name\tbusiness_address\tcountry\n"
                      "S2_1\tCafé शक्ति\t12 bis, Rue du Test\tNA\n"
                      "S2_2\tOther\t\tnew country\n")
    output = tmp_path / "lookup.sqlite"
    manifest = build_raw_lookup({"S2": source}, output, batch_size=1)
    assert manifest["records"] == 2
    assert lookup_manifest(output)["normalization"] == "none"
    assert output.with_name(output.name + ".complete.json").exists()
    marker = output.with_name(output.name + ".complete.json")
    marker_text = marker.read_text()
    marker.unlink()
    with pytest.raises(ValueError, match="complete marker"):
        open_raw_lookup(output)
    marker.write_text(marker_text)
    conn = open_raw_lookup(output)
    rows = lookup_raw_records(conn, ["S2_2", "S2_1", "S2_1"])
    assert rows["S2_1"]["business_name"] == "Café शक्ति"
    assert rows["S2_1"]["business_address"] == "12 bis, Rue du Test"
    assert rows["S2_1"]["country"] == "NA"
    assert rows["S2_2"]["business_address"] == ""
    with pytest.raises(ValueError, match="missing"):
        lookup_raw_records(conn, ["S2_missing"])
    conn.close()
    assert build_raw_lookup({"S2": source}, output)["records"] == 2
    source.write_text(source.read_text() + "S2_3\tNew\tNew address\tNA\n")
    with pytest.raises(ValueError, match="different"):
        build_raw_lookup({"S2": source}, output)
    with pytest.raises(ValueError, match="Expected"):
        build_raw_lookup({"S2": source}, tmp_path / "wrong_hash.sqlite", expected_hashes={"S2": "0" * 64})


def test_streamed_gate_table_keeps_all_pair_keys_and_checks_checkpoint_hash(tmp_path):
    import csv
    import hashlib
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research/sprint_6h/gate"))
    from build_gate_table import initialize, merge_gate_tables, process_chunk
    from build_raw_lookup import build_raw_lookup, sha256
    rows = universe()
    target_rows = [record("S2_1"), record("S2_2", address="38 rue Mac Arthur")]
    paths = {}
    for source, values in (("S1", rows), ("S2", target_rows)):
        path = tmp_path / (source + ".tsv")
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0]), delimiter="\t")
            writer.writeheader()
            writer.writerows(values)
        paths[source] = path
    lookup = tmp_path / "raw.sqlite"
    build_raw_lookup(paths, lookup)
    state = tmp_path / "state.json"
    state.write_text(json.dumps(fit_signals({"open label": rows})))
    input_path = tmp_path / "0000000.parquet"
    original = pa.table({"source1_entity_id": ["S1_1", "S1_1"],
                         "candidate_entity_id": ["S2_1", "S2_2"], "probability": [0.99, 0.9]})
    pq.write_table(original, input_path)
    original_hash = sha256(input_path)
    input_path.with_suffix(".json").write_text(json.dumps({"parquet_sha256": original_hash}))
    fingerprint = {"gate_code_sha256": sha256(Path(__file__).resolve().parents[1] / "scripts/france_gate.py"),
                   "stats_sha256": sha256(state), "config_sha256": "fixture"}
    initialize(state, lookup, None, fingerprint, 1)
    checkpoint = process_chunk((input_path, tmp_path / "gates"))
    result = pq.read_table(checkpoint["output"]).to_pydict()
    assert result["source1_entity_id"] == ["S1_1", "S1_1"]
    assert result["candidate_entity_id"] == ["S2_1", "S2_2"]
    assert result["gate_veto"] == [False, True]
    assert sha256(input_path) == original_hash
    expected = hashlib.sha256(b"S1_1\tS2_1\nS1_1\tS2_2\n").hexdigest()
    assert checkpoint["pair_order_sha256"] == expected
    assert process_chunk((input_path, tmp_path / "gates")) == checkpoint
    merged = merge_gate_tables([checkpoint], tmp_path / "gate_table.parquet", expected, 1)
    assert merged["rows"] == 2
    assert merged["pair_order_sha256"] == expected
    with pytest.raises(ValueError, match="pair order"):
        merge_gate_tables([checkpoint], tmp_path / "wrong_order.parquet", "0" * 64, 1)
    assert not (tmp_path / "wrong_order.parquet").exists()
    pq.write_table(original.take(pa.array([1, 0])), input_path)
    input_path.with_suffix(".json").write_text(json.dumps({"parquet_sha256": sha256(input_path)}))
    with pytest.raises(ValueError, match="different inputs"):
        process_chunk((input_path, tmp_path / "gates"))


def test_streaming_early_returns_match_versioned_pure_gate():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research/sprint_6h/gate"))
    from build_gate_table import batch_pair_decision, prepare_name_views
    from france_gate import DEFAULT_CONFIG
    rows = universe()
    targets = [record(f"S2_{i}", name=name, address=address) for i, (name, address) in enumerate([
        ("Atlas Ltd", "38 rue Mac Arthur"), ("Atlas Ltd", "21 rue Victor Hugo"),
        ("Atlas Ltd", ""), ("Atlas Ltd", "75001 Paris"), ("Unrelated", "38 rue Mac Arthur"),
        ("", ""), ("Private Ltd", "38 rue Mac Arthur"), ("Atlas LLC", "rue Victor Hugo")])]
    stats = fit_signals({"open label": rows})
    records = {row["entity_id"]: row for row in rows + targets}
    views = prepare_name_views(records)
    for minimum in (1, 2, 4):
        config = {**DEFAULT_CONFIG, "min_name_rivals": minimum}
        for left in rows:
            for target in targets:
                country, fast = batch_pair_decision(left["entity_id"], target["entity_id"], records, views, stats, config)
                pure = gate_veto(pair_signals(left, target, stats), config)
                assert fast["gate_veto"] == pure["gate_veto"]
                assert fast["gate_reason"] == pure["gate_reason"]
                assert country == "open label"


def test_r1_floor_skip_keeps_candidates_and_actual_decision_parity(tmp_path):
    import csv
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    pd = pytest.importorskip("pandas")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research/sprint_6h/gate"))
    from build_gate_table import initialize, process_chunk
    from build_raw_lookup import build_raw_lookup, sha256
    from decision_policy_v2 import decide_v2
    from run_frozen_pipeline import decide
    rows = universe()
    targets = [record("S2_1"), record("S2_2", address="38 rue Mac Arthur"),
               record("S2_3", address="38 rue Mac Arthur")]
    paths = {}
    for source, values in (("S1", rows), ("S2", targets)):
        path = tmp_path / (source + ".tsv")
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0]), delimiter="\t")
            writer.writeheader(); writer.writerows(values)
        paths[source] = path
    lookup, state = tmp_path / "raw.sqlite", tmp_path / "state.json"
    build_raw_lookup(paths, lookup)
    state.write_text(json.dumps(fit_signals({"open label": rows})))
    frame = pd.DataFrame({"source1_entity_id": ["S1_1", "S1_1", "S1_1", "S1_2"],
        "candidate_entity_id": ["S2_1", "S2_2", "S2_3", "S2_2"],
        "probability": [0.99, 0.985, 0.90, 0.95]})
    input_path = tmp_path / "0000000.parquet"
    frame.to_parquet(input_path, index=False)
    input_path.with_suffix(".json").write_text(json.dumps({"parquet_sha256": sha256(input_path)}))
    fingerprint = {"gate_code_sha256": sha256(Path(__file__).resolve().parents[1] / "scripts/france_gate.py")}
    initialize(state, lookup, None, fingerprint, 2)
    full = pq.read_table(process_chunk((input_path, tmp_path / "full"))["output"]).to_pandas()
    optimized_fingerprint = {**fingerprint, "execution_mode": "R1", "eligibility_floor": 0.95,
        "eligibility_proof": {"t_first": 0.95, "t_rest": 0.98, "score_column": "probability"}}
    initialize(state, lookup, None, optimized_fingerprint, 2)
    pruned = pq.read_table(process_chunk((input_path, tmp_path / "pruned"))["output"]).to_pandas()
    assert list(pruned.candidate_entity_id) == list(frame.candidate_entity_id)
    assert full.gate_veto.tolist() == [False, True, True, True]
    assert pruned.gate_veto.tolist() == [False, True, False, True]
    assert pruned.gate_reason.iloc[2] == "ineligible_r1_score"
    policy = {"threshold": 0.98, "t_first": 0.95, "t_rest": 0.98}
    chosen_original, chosen_v2 = [], []
    for table in (full, pruned):
        combined = frame.merge(table, on=["source1_entity_id", "candidate_entity_id"], validate="one_to_one")
        accepted, _, _ = decide_v2(combined, combined, policy)
        chosen_v2.append(accepted.tolist())
        # Original decision composition with separate eligibility veto mask.
        controlled = combined.copy()
        controlled.loc[controlled.gate_veto, "probability"] = 0.0
        accepted, _ = decide(controlled, controlled, policy)
        chosen_original.append(accepted.tolist())
    assert chosen_original[0] == chosen_original[1] == [True, False, False, False]
    assert chosen_v2[0] == chosen_v2[1] == [True, False, False, False]
    assert pq.read_table(input_path).to_pandas().probability.tolist() == [0.99, 0.985, 0.90, 0.95]


def test_original_gate_composition_releases_claim_before_rescue():
    pd = pytest.importorskip("pandas")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research/sprint_6h/gate"))
    from evaluate_gate_grid import gated_predictions
    frame = pd.DataFrame({"source1_entity_id": ["owner_a", "owner_a", "owner_b", "owner_c"],
        "candidate_entity_id": ["shared", "fallback", "shared", "low"],
        "probability": [0.95, 0.75, 0.90, 0.1]})
    gate = frame[["source1_entity_id", "candidate_entity_id"]].copy()
    gate["gate_veto"] = [True, False, False, False]
    config = {"threshold": 0.83, "t_first": 0.7, "t_rest": 0.83}
    predictions = gated_predictions(frame, gate, config, {"owner_a", "owner_b", "owner_c"})
    assert predictions == {"owner_a": {"fallback"}, "owner_b": {"shared"}, "owner_c": set()}
    assert frame.probability.tolist() == [0.95, 0.75, 0.90, 0.1]
    with pytest.raises(ValueError, match="Incomplete gate"):
        gated_predictions(frame, gate.iloc[:3], config, {"owner_a", "owner_b", "owner_c"})


def test_v3_house_agreement_abstains_without_changing_v2_signals():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research/sprint_6h/gate"))
    from house_agreement_gate_v3 import gate_veto as gate_veto_v3
    rows, stats = universe(), fit_signals({"open label": universe()})
    for address, veto in (("12 rue Mac Arthur", False), ("12b rue Mac Arthur", False),
                          ("21 rue Mac Arthur", True), ("rue Mac Arthur", True)):
        value = pair_signals(rows[0], record("S2_1", address=address), stats)
        saved = copy.deepcopy(value)
        v2, v3 = gate_veto(value), gate_veto_v3(value)
        assert v2["gate_veto"] is True
        assert v3["gate_veto"] is veto
        assert v3["gate_version"] == "conservative-street-v3-house-agreement"
        assert value == saved
    # No invented disagreement when a postcode was mistaken for a house.
    value = pair_signals(rows[0], record("S2_1", address="75001 rue Mac Arthur"), stats)
    assert value["house_number_match"] is None
    assert gate_veto_v3(value)["gate_veto"] == gate_veto(value)["gate_veto"]


def test_streaming_v3_parity_for_all_fixture_name_and_address_routes():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research/sprint_6h/gate"))
    import build_gate_table as builder
    from house_agreement_gate_v3 import gate_veto as gate_veto_v3
    rows = universe()
    targets = [record(f"S2_{i}", name=name, address=address) for i, (name, address) in enumerate([
        ("Atlas Ltd", "12 rue Mac Arthur"), ("Atlas Ltd", "21 rue Mac Arthur"),
        ("Atlas Ltd", "rue Mac Arthur"), ("Atlas Ltd", ""), ("Unrelated", "12 rue Mac Arthur"),
        ("Atlas Ltd", "12b rue Mac Arthur"), ("", ""), ("Ltd", "12 rue Mac Arthur")])]
    stats = fit_signals({"open label": rows})
    records = {row["entity_id"]: row for row in rows + targets}
    views = builder.prepare_name_views(records)
    previous = builder.PAIR_GATE
    try:
        builder.PAIR_GATE = gate_veto_v3
        for minimum in (1, 2, 4):
            config = {"min_name_rivals": minimum, "max_street_overlap": 0, "min_street_tokens": 2}
            for left in rows:
                for target in targets:
                    _, fast = builder.batch_pair_decision(left["entity_id"], target["entity_id"], records, views, stats, config)
                    pure = gate_veto_v3(pair_signals(left, target, stats), config)
                    assert (fast["gate_veto"], fast["gate_reason"]) == (pure["gate_veto"], pure["gate_reason"])
    finally:
        builder.PAIR_GATE = previous


def distinctive_universe():
    rows = universe()
    rows += [record("S1_extra_a", address="14 rue Blue Garden"), record("S1_extra_b", address="17 rue Red Forest")]
    rows[0]["business_address"] = "12 rue de Bethune"
    rows += [record(f"S1_context_{i}", name=f"Unique context {i}", address=f"{100+i} rue de Place{i}")
             for i in range(190)]
    return rows


def test_v4_distinctive_contradiction_preserves_common_tokens_and_house_guard():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research/sprint_6h/gate"))
    from distinctive_street_gate_v4 import gate_veto as gate_veto_v4, pair_signals as pair_signals_v4
    rows = distinctive_universe()
    stats = fit_signals({"open label": rows})
    for address, veto in (("21 rue de Thumesnil", True), ("21 rue Thumesnil", True), ("12 rue de Thumesnil", False),
                          ("21 rue de Bethun", False), ("21 rue de Bethune", False), ("", False)):
        result = pair_signals_v4(rows[0], record("S2_1", address=address), stats)
        assert gate_veto_v4(result)["gate_veto"] is veto
        assert result["reference_address"]["street_tokens"] == ["bethune", "de"]
        assert result["reference_distinctive_street_tokens"] == ["bethune"]
    result = pair_signals_v4(rows[0], record("S2_1", address="21 rue de Thumesnil"), stats)
    assert result["street_overlap"] > 0
    assert result["distinctive_street_contradiction"]


def test_v4_df_boundary_near_match_boundary_and_numerical_tokens():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research/sprint_6h/gate"))
    from distinctive_street_gate_v4 import near_match, pair_signals as pair_signals_v4
    assert near_match("harbor", "harbr")
    assert near_match("abcd", "abce")  # Exact .25 boundary.
    assert not near_match("abcd", "abef")
    assert not near_match("oak", "oax")  # Short strings require exact support.
    rows = distinctive_universe()
    stats = fit_signals({"open label": rows})
    state = stats["countries"]["open label"]
    assert state["records"] == 200
    state["address_df"]["bethune"] = 2
    assert pair_signals_v4(rows[0], record("S2_1"), stats)["reference_distinctive_street_tokens"] == ["bethune"]
    state["address_df"]["bethune"] = 3
    assert pair_signals_v4(rows[0], record("S2_1"), stats)["reference_distinctive_street_tokens"] == []


def test_streaming_v4_fast_views_match_pure_distinctive_gate():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research/sprint_6h/gate"))
    import build_gate_table as builder
    from distinctive_street_gate_v4 import gate_veto as veto, pair_signals as signal
    rows = distinctive_universe()
    targets = [record(f"S2_{i}", name=name, address=address) for i, (name, address) in enumerate([
        ("Atlas Ltd", "21 rue de Thumesnil"), ("Atlas Ltd", "12 rue de Thumesnil"),
        ("Atlas Ltd", "21 rue de Bethun"), ("Atlas Ltd", ""), ("Unrelated", "21 rue de Thumesnil")])]
    stats = fit_signals({"open label": rows})
    records = {row["entity_id"]: row for row in rows + targets}
    views = builder.prepare_name_views(records)
    previous = builder.PAIR_GATE, builder.PAIR_SIGNALS
    try:
        builder.PAIR_GATE, builder.PAIR_SIGNALS = veto, signal
        config = {"min_name_rivals": 4, "max_street_overlap": 0, "min_street_tokens": 2}
        for left in rows[:10]:
            for target in targets:
                _, fast = builder.batch_pair_decision(left["entity_id"], target["entity_id"], records, views, stats, config)
                pure = veto(signal(left, target, stats), config)
                assert (fast["gate_veto"], fast["gate_reason"]) == (pure["gate_veto"], pure["gate_reason"])
    finally:
        builder.PAIR_GATE, builder.PAIR_SIGNALS = previous


def test_unlabelled_capture_preserves_exact_floor_and_complete_country_scope(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research/sprint_6h/gate"))
    from diagnose_french_claims import capture
    from build_raw_lookup import sha256
    floor = 0.6999999999999998
    path = tmp_path / "0000000.parquet"
    table = pa.Table.from_pydict({"source1_entity_id": ["fr", "fr", "us"],
        "candidate_entity_id": ["a", "b", "c"], "probability": [floor, 0.6999999999999997, 0.9]})
    pq.write_table(table, path)
    source_hash = sha256(path)
    seal = {"chunks": [{"parquet": {"path": path.name, "sha256": source_hash}}],
            "pairs": 3, "pair_order_sha256": "complete_original_order"}
    stats = {"owner_keys": {"fr": ["france", "", ""], "us": ["us", "", ""]}}
    result = capture(seal, tmp_path, stats, floor, tmp_path / "diagnostic")
    assert result["counts"] == {"scanned_pairs": 3, "eligible_pairs": 2,
        "france_eligible_pairs": 1, "us_eligible_pairs": 1, "skipped_below_floor_pairs": 1}
    eligible = pq.read_table(result["all_country_eligible"]["path"]).to_pydict()
    assert eligible["candidate_entity_id"] == ["a", "c"]
    assert eligible["probability"] == [floor, 0.9]
    french = pq.read_table(result["paths"][0]).to_pydict()
    assert french["candidate_entity_id"] == ["a"]
    assert sha256(path) == source_hash
    with pytest.raises(ValueError, match="complete sealed"):
        capture({**seal, "pairs": 4}, tmp_path, stats, floor, tmp_path / "incomplete")
