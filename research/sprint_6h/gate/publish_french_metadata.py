"""Publish strict comparison bindings for completed unlabelled gate tables."""
import argparse
import json
from pathlib import Path

from build_raw_lookup import sha256


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--report", required=True, type=Path)
    a = p.parse_args()
    report = json.loads(a.report.read_text())
    if report["status"] != "complete" or report["provenance"]["labels"] != "none":
        raise ValueError("Require completed unlabelled French diagnostics")
    paths = []
    for table in report["tables"]:
        if sha256(table["path"]) != table["sha256"]:
            raise ValueError("French diagnostic gate table hash differs")
        metadata = {"status": "unlabelled_research_diagnostic_only",
                    "promotion": "rejected" if table["rule"] == "v4-distinctive" else "not_selected",
                    "path": table["path"], "table_sha256": table["sha256"],
                    "pair_order_sha256": table["pair_order_sha256"], "rows": table["rows"],
                    "baseline_frozen_sha256": report["provenance"]["decision_config_sha256"],
                    "gate_config": table["config"], "gate_rule": table["rule"],
                    "code_state_hashes": report["provenance"], "capture": report["capture"],
                    "coverage": report["coverage"], "reason_counts": table["reason_counts"],
                    "report_sha256": sha256(a.report), "publisher_sha256": sha256(__file__)}
        path = Path(table["path"]).with_name("release_metadata.json")
        path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        paths.append({"name": table["name"], "metadata": str(path.resolve()), "sha256": sha256(path)})
    index = {"status": "complete", "comparisons": paths, "labels": "none",
             "report_sha256": sha256(a.report)}
    a.report.with_name("comparison_metadata_index.json").write_text(json.dumps(index, indent=2) + "\n")
    print(json.dumps(index))


if __name__ == "__main__":
    main()
