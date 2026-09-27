"""Inspect every known true target rejected by the fixed development ablation."""
import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from build_raw_lookup import sha256


def main():
    p = argparse.ArgumentParser()
    for key in ("raw_report", "baseline", "candidate", "scores", "output"):
        p.add_argument("--" + key.replace("_", "-"), required=True, type=Path)
    a = p.parse_args()
    report = json.loads(a.raw_report.read_text())
    harms = report["variants"][0]["all_true_vetoes"]
    baseline, candidate = json.loads(a.baseline.read_text()), json.loads(a.candidate.read_text())
    ids = [item["source1_entity_id"] for item in harms]
    wanted = {(item["source1_entity_id"], item["candidate_entity_id"]) for item in harms}
    scores = {}
    selected_owners = pa.array(sorted(set(ids)))
    for batch in pq.ParquetFile(a.scores).iter_batches(batch_size=10000,
            columns=["source1_entity_id", "candidate_entity_id", "probability"]):
        batch = batch.filter(pc.is_in(batch["source1_entity_id"], value_set=selected_owners))
        frame = batch.to_pydict()
        for left, right, probability in zip(frame["source1_entity_id"], frame["candidate_entity_id"], frame["probability"]):
            if (left, right) in wanted:
                scores[left, right] = probability
    for item in harms:
        left, right = item["source1_entity_id"], item["candidate_entity_id"]
        item.update(probability=scores.get((left, right)), baseline_accepted=right in baseline[left],
                    v4_accepted=right in candidate[left])
    output = {"status": "rejected_for_promotion", "rule_changes_after_results": "none",
              "known_true_veto_count": len(harms),
              "accepted_true_removed": sum(item["baseline_accepted"] and not item["v4_accepted"] for item in harms),
              "all_harmed_links": harms,
              "provenance": {"raw_report_sha256": sha256(a.raw_report), "baseline_sha256": sha256(a.baseline),
                             "candidate_sha256": sha256(a.candidate), "scores_sha256": sha256(a.scores),
                             "exporter_sha256": sha256(__file__)}}
    a.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: output[key] for key in ("known_true_veto_count", "accepted_true_removed")}))


if __name__ == "__main__":
    main()
