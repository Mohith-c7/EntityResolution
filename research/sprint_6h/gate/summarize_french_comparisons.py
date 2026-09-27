"""Hash-bound mechanical comparison summary; no French labels or selection."""
import argparse
import json
from pathlib import Path

from build_raw_lookup import sha256


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--comparisons", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    names = [f"v3_setting_{i}" for i in range(6)] + ["v4_fixed"]
    results = []
    for name in names:
        directory = a.comparisons / name
        path, winners = directory / "diagnostic.json", directory / "changed_winners.json"
        if not path.is_file() or not winners.is_file():
            raise ValueError("Require all seven completed comparisons and complete owner lists")
        q = json.loads(path.read_text())
        if q["labels_used"] is not False or q["scanned_full_pairs"] != 69301760:
            raise ValueError("Require unlabelled complete scored-universe diagnostics")
        results.append({"name": name, "gate_rule": q["gate_metadata"]["gate_rule"],
                        "gate_config": q["gate_metadata"]["gate_config"],
                        "veto_claims": q["veto_claims"], "vetoed_baseline_accepted": q["vetoed_baseline_accepted"],
                        "removed_accepted_claims": q["removed_accepted_claims"],
                        "added_accepted_claims": q["added_accepted_claims"],
                        "country_link_deltas": q["country_link_deltas"],
                        "changed_target_winners": q["changed_target_winners"],
                        "baseline_collisions": q["baseline_collisions"], "gated_collisions": q["gated_collisions"],
                        "baseline_country_collisions": q["baseline_country_collisions"],
                        "gated_country_collisions": q["gated_country_collisions"],
                        "paths": {"diagnostic": str(path.resolve()), "changed_winners": str(winners.resolve()),
                                  "changed_claims": str((directory / "changed_claims.parquet").resolve())},
                        "hashes": {"diagnostic": sha256(path), "changed_winners": sha256(winners),
                                   "changed_claims": sha256(directory / "changed_claims.parquet")}})
    report = {"status": "complete_unlabelled_mechanical_comparisons", "labels_used": False,
              "promotion": "none; root selection required; v4 rejected on development precision",
              "interpretation": "Mechanical claim/owner changes only. No France accuracy or links-count target.",
              "scanned_full_pairs": 69301760, "eligible_claims": 5919526,
              "below_floor_skipped": 63382234, "french_signal_rows": 997413,
              "results": results, "summary_code_sha256": sha256(__file__)}
    a.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps([{key: r[key] for key in ("name", "veto_claims", "removed_accepted_claims",
                                              "added_accepted_claims", "changed_target_winners")} for r in results]))


if __name__ == "__main__":
    main()
