"""Independent label-free arithmetic, self-exclusion and coverage checks."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from export_sprint_workbench import sha


def verify(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    if sha(directory / "features.parquet") != manifest["features_sha256"] or sha(directory / "diagnostics.jsonl") != manifest["diagnostics_sha256"]:
        raise ValueError("Reverse evidence artifact hash differs")
    frame = pd.read_parquet(directory / "features.parquet", use_threads=False)
    keys = ["source1_entity_id", "candidate_entity_id"]
    if frame.duplicated(keys).any(): raise ValueError("Duplicate reverse evidence keys")
    lookup = frame.set_index(keys); seen = set()
    no_other = no_query = self_in_pool = saturated = unavailable = 0
    fields = ("namecore", "phonetic", "address")
    components = ("name", "phonetic", "address", "digits", "number_conflict", "joint")
    suffix = ("name_sort", "phonetic_sort", "address_sort", "number_jaccard", "number_contradiction", "joint")
    with (directory / "diagnostics.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line); key = tuple(row[k] for k in keys)
            if key in seen or key not in lookup.index: raise ValueError("Foreign/duplicate diagnostic key")
            seen.add(key); feature = lookup.loc[key]
            rivals, views = row["rival_ids"], row["rival_views"]
            if len(rivals) > 3 or key[0] in rivals or len(set(rivals)) != len(rivals) or len(rivals) != len(views):
                raise ValueError("Self/repeated rival or mismatched views")
            def close(actual, expected):
                if not np.isclose(actual, expected, rtol=0, atol=1e-7): raise ValueError("Reverse arithmetic/coverage value differs")
            for component, name in zip(components, suffix):
                close(feature["reverse_other_" + name], views[0][component] if rivals else -1.)
            for component, name, own in zip(("name", "phonetic", "address", "digits", "joint"),
                                           ("name", "phonetic", "address", "number", "joint"),
                                           ("name_sort", "phonetic_sort", "address_sort", "number_jaccard", "joint")):
                close(feature["reverse_margin_" + name], feature["reverse_own_" + own] - views[0][component] if rivals else -1.)
            close(feature.reverse_other_second_joint, views[1]["joint"] if len(views) > 1 else -1.)
            close(feature.reverse_other_third_joint, views[2]["joint"] if len(views) > 2 else -1.)
            close(feature.reverse_best_second_joint_gap, views[0]["joint"] - views[1]["joint"] if len(views) > 1 else -1.)
            if any(views[i]["joint"] < views[i+1]["joint"] for i in range(len(views)-1)):
                raise ValueError("Rivals not joint-score ordered")
            selected = row["selected_terms"]
            bound = min(20000, max(1, math.floor(.02 * row["country_records"])))
            for field, name in zip(fields, ("name", "phonetic", "address")):
                terms = selected[field]
                if len(terms) > 2 or len({v[1] for v in terms}) != len(terms) or terms != sorted(terms) or any(not 0 < v[0] <= bound for v in terms):
                    raise ValueError("Unbounded or misordered country-local query terms")
                close(feature["reverse_" + name + "_terms_used"], len(terms))
            query_fields = sum(bool(selected[f]) for f in fields)
            close(feature.reverse_query_fields_used, query_fields)
            close(feature.reverse_high_df_terms_skipped, row["high_df_skipped"])
            close(feature.reverse_shortlist_saturated, row["shortlist_saturated"])
            close(feature.reverse_own_in_top8, row["self_excluded"])
            close(feature.reverse_no_other_returned, not rivals)
            if not row["complete_postings"] or row["shortlist_count"] > 80 or row["rerank_count"] > 8 or row["query_fields"] != query_fields:
                raise ValueError("Incomplete postings or shortlist cap violated")
            if row["nonself_top8_count"] != row["rerank_count"] - int(row["self_excluded"]) or not 0 <= row["posting_count_nonself"] <= row["posting_count"]:
                raise ValueError("Nonself coverage counts differ")
            if not query_fields and rivals: raise ValueError("Noquery incorrectly returns rivals")
            if bool(row["no_result"]) != (not rivals) or bool(row["no_query"]) != (not query_fields):
                raise ValueError("Missing-result/query flag differs")
            no_other += not rivals; no_query += not query_fields
            self_in_pool += row["self_excluded"]; saturated += row["shortlist_saturated"]
            unavailable += row.get("country_unavailable", False)
    if len(seen) != len(frame): raise ValueError("Diagnostic coverage incomplete")
    return {"status": "independent_reverse_features_verified_no_labels", "verified_at": datetime.now(timezone.utc).isoformat(),
            "rows": len(frame), "references": int(frame.source1_entity_id.nunique()), "no_other_rows": no_other,
            "no_query_rows": no_query, "self_in_top8_rows": self_in_pool, "shortlist_saturated_rows": saturated,
            "country_unavailable_rows": unavailable, "manifest_sha256": sha(directory / "manifest.json"),
            "labels_read": False, "coverage_limitation": "Only complete selected postings and bounded cachedtop8 rivals are verified; this does not establish exhaustive rival recall or uniqueness."}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(); report = verify(a.features)
    with a.output.open("x") as stream: json.dump(report, stream, indent=2); stream.write("\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__": main()
