"""Measure label-free hybrid blocks and project the verified-cache TSV release.

This reports saved-cache mode R2, not a cold reproduction benchmark. The complete
new neural and rescue preparation stages use actual full-test timings.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "research/final_2h")]
from preflight import read, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--freeze", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--complete-neural", action="store_true")
    a = p.parse_args()
    import numpy as np
    import pandas as pd
    import build_gated_submission as export
    from evaluate_sprint import check_runtime, decide_control
    from post_selection_exclusivity import exclusive_selected
    if a.output.exists(): raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    frozen = read(a.freeze)
    allrefs = read("research/sprint_6h/validation/fresh_splits/heldout_universe.json")
    by = {}
    for row in allrefs: by.setdefault(row["country"].strip().casefold(), []).append(row)
    if set(by) != {"india", "us"}: raise ValueError("Unexpected actual heldout countries")
    shares = {country: round(10000 * len(rows) / len(allrefs)) for country, rows in by.items()}
    shares["us"] += 10000 - sum(shares.values())
    started = time.monotonic()
    frame = pd.read_parquet("research/final_2h/hybrid_audit/heldout_scored_v1/pairs.parquet")
    read_seconds = time.monotonic() - started
    timings = []
    for number in (0, 1):
        refs = [r for country in sorted(by) for r in by[country][number*shares[country]:(number+1)*shares[country]]]
        ids = {r["entity_id"] for r in refs}
        begin = time.monotonic(); block = frame.loc[frame.source1_entity_id.isin(ids)].copy()
        seconds = {"reads": time.monotonic()-begin + read_seconds*len(ids)/331012,
                   "lookups": 0.0, "normalization": 0.0, "gate_or_features": 0.0, "predict": 0.0}
        if len(block) != 400000: raise ValueError("Incomplete actual block candidates")
        begin = time.monotonic(); chosen, _ = decide_control(block, block, frozen["decision_config"])
        chosen, removed = exclusive_selected(block, chosen)
        seconds["global_decision"] = time.monotonic()-begin
        out = a.output / f"block_{number}"; out.mkdir()
        begin = time.monotonic()
        (out / "diagnostic.json").write_text(json.dumps({"selected": int(chosen.sum()), "removed": int(removed.sum()), "labels_read": False})+"\n")
        seconds["diagnostic_write"] = time.monotonic()-begin
        begin = time.monotonic(); stats = export.export_outputs(refs, block, chosen, out)
        seconds["tsv_export"] = time.monotonic()-begin
        test = out / "test"; test.mkdir()
        pd.DataFrame(refs).to_csv(test / "test_source1.tsv", sep="\t", index=False)
        for s in (2, 3):
            (test / f"test_source{s}.tsv").symlink_to((ROOT / f"dataset/train/train_source{s}.tsv").resolve())
        validation = export.validate_outputs(out, test)
        if validation["strict"] != "PASS" or validation["official"] != "PASS": raise ValueError("Actual block ID validation failed")
        seconds.update(validation["stage_seconds"])
        timings.append({"references": len(refs), "pairs": len(block), "country_stratified": True,
                        "countries": shares, "seconds": seconds, "validation": validation, "statistics": stats})
    (a.output / "measured_blocks.json").write_text(json.dumps(timings, indent=2)+"\n")
    # Full-stage neural timing manifests may arrive after the fresh block work.
    prep = Path("models/final_2h/hybrid_test_rescue_v1/manifest.json")
    old = prep.parent / "neural_old/manifest.json"; fine = prep.parent / "neural_fine/manifest.json"
    main = Path("models/sprint_6h/neural/last8_full_test06/manifest.json")
    files = [prep, old, fine, main]
    if not all(f.exists() for f in files):
        print(json.dumps({"status": "blocks_complete_waiting_full_neural_manifests", "missing": [str(f) for f in files if not f.exists()]})); return
    stage = [read(f) for f in files]
    if any(x["status"] != "complete" or x.get("labels_read") is not False for x in stage): raise ValueError("Incomplete full stage evidence")
    fullrefs, fullpairs = 1732544, 69301760
    join = read("research/final_2h/hybrid_audit/heldout_scored_v1/manifest.json")["seconds"]
    projection = {"new_full_extra_route_reverse_and_raw_serialization": stage[0]["seconds"],
                  "new_full_main8_neural": stage[3]["seconds"],
                  "new_full_extra_old_neural": stage[1]["seconds"],
                  "new_full_extra_fine4_neural": stage[2]["seconds"],
                  "full_anchored_adapter_join_and_write": join * fullpairs/13240480 * 1.5,
                  "global_decision": max(b["seconds"]["global_decision"]/b["pairs"] for b in timings)*fullpairs*1.5,
                  "export": max(b["seconds"]["tsv_export"]/b["references"] for b in timings)*fullrefs,
                  "strict_and_official_id_validators_conservative": sum(max(b["seconds"][v] for b in timings) for v in ("strict_validator", "official_validator")) + 300,
                  "network_and_integrity_allowance": 300.0}
    eta = sum(projection.values())*1.15
    remaining_projection = {k: v for k, v in projection.items() if not k.startswith("new_full_")}
    remaining_eta = sum(remaining_projection.values())*1.15
    deadline = datetime(2026, 9, 27, 18, 10, tzinfo=timezone.utc)
    remaining_budget = (deadline-datetime.now(timezone.utc)).total_seconds()
    report = {"mode": "R2", "measured": True, "candidate_frozen_sha256": sha(a.freeze),
              "synthetic_fixture_runtime": False, "full_test_eta_supported": True,
              "estimated_processing_seconds": remaining_eta, "processing_budget_seconds": remaining_budget,
              "safety_margin_fraction": .15, "blocks": timings, "passed": remaining_eta <= remaining_budget,
              "total_incremental_pipeline_seconds_including_completed_stages": eta,
              "remaining_after_neural_seconds": remaining_eta, "operational_deadline_utc": deadline.isoformat(),
              "full_references": fullrefs, "full_pairs": fullpairs, "projection_seconds": projection,
              "scope": "Verified complete saved base/old-main-feature cache reuse; actual new full-test branches and two actual heldout country-stratified decision/export/ID-validator blocks. No cold reproduction claim.",
              "timing_note": "Reads/normalization/retrieval and original main old-head values are already sealed caches. Fresh full extra preparation includes its normalization and reverse lookups. Blocks time only cached candidate decisions/export/validators; new full-stage costs are separate, not omitted.",
              "source_sha256": sha(__file__), "stage_manifest_sha256": {str(f): sha(f) for f in files}}
    report["acceptance"] = check_runtime(report, sha(a.freeze))
    # Avoid embedding recursive duplicate runtime report inside acceptance.
    report["acceptance"].pop("evidence")
    (a.output / "runtime.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({"passed": report["acceptance"]["passed"], "estimated_processing_seconds": eta, "runtime_sha256": sha(a.output / "runtime.json")}))


if __name__ == "__main__": main()
