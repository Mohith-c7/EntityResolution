"""Review Cursor ownership on a complete, already exposed development graph.

No labels are loaded until all policies have made predictions. Input split names
are deliberately restricted to early_stop and selection. This is development
evidence, not a fresh audit or an official-test claimant-density certificate.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

PAIR_KEYS = ["source1_entity_id", "candidate_entity_id"]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def admissible_lex(claims, config):
    """Propose ordinary links + one original rank-one rescue; arbitrate proposals.

    This predeclared mechanism comparator removes ineligible lower-band rivals.
    Scores win; exact ties use source IDs consistently with Cursor's policy.
    Losing a claim never promotes another previously ineligible rescue.
    """
    work = claims.reset_index(drop=True)
    probability = work.probability.to_numpy()
    codes, _ = pd.factorize(work.source1_entity_id)
    # lexsort preserves original input order for equal scores, matching control.
    order = np.lexsort((-probability, codes))
    first = np.zeros(len(work), dtype=bool)
    if len(work):
        first[order[np.r_[True, codes[order][1:] != codes[order][:-1]]]] = True
    proposed = (probability >= config["t_rest"]) | (first & (probability >= config["t_first"]))
    eligible = work.loc[proposed].sort_values(
        ["probability", "source1_entity_id"], ascending=[False, True], kind="stable")
    accepted = np.zeros(len(work), dtype=bool)
    accepted[eligible.groupby("candidate_entity_id", sort=False).head(1).index] = True
    return accepted, proposed & ~accepted


def load_policy(path):
    spec = importlib.util.spec_from_file_location("cursor_ownership_snapshot", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def collisions(frame, chosen):
    counts = frame.loc[chosen].groupby("candidate_entity_id").source1_entity_id.nunique()
    return int((counts > 1).sum())


def run(args):
    sys.path.insert(0, str(args.repo / "scripts"))
    from evaluate_sprint import decide_control, paired_evaluate, predictions_for, entity_f05
    from decision_policy_v2 import validate_claims

    if args.output.exists():
        raise FileExistsError(args.output)
    policy = load_policy(args.policy)
    config = json.loads(args.frozen.read_text())
    config = {k: config[k] for k in ("threshold", "t_first", "t_rest")}
    columns = PAIR_KEYS + ["probability", "candidate_order"]
    claims = pd.read_parquet(args.pairs, columns=columns).reset_index(drop=True)
    validate_claims(claims)
    ids_by_split = {}
    for split, count in (("early_stop", 10000), ("selection", 20000)):
        ids_by_split[split] = json.loads((args.workbenches / split / "reference_ids.json").read_text())
        if len(ids_by_split[split]) != count or len(set(ids_by_split[split])) != count:
            raise ValueError("Expected existing 10k early / 20k selection development cohorts")
    all_ids = set(ids_by_split["early_stop"]) | set(ids_by_split["selection"])
    if len(all_ids) != 30000 or set(claims.source1_entity_id) != all_ids:
        raise ValueError("Scored claim graph must cover exactly the complete existing 30k universe")
    masks = {}
    timings = {}
    policies = {
        "control": lambda: decide_control(claims, claims, config),
        "cursor_flags_off": lambda: policy.decide_v2(claims, claims, config),
        "cursor_tie_only": lambda: policy.decide_v2(claims, claims, {**config, "tie_break": True}),
        "cursor_cover_only": lambda: policy.decide_v2(claims, claims, {**config, "cover_rank_one": True}),
        "cursor_both": lambda: policy.decide_v2(claims, claims, {**config, "tie_break": True, "cover_rank_one": True}),
        "admissible_lex": lambda: admissible_lex(claims, config),
    }
    for name, operation in policies.items():
        before = time.perf_counter()
        masks[name], _ = operation()
        timings[name] = time.perf_counter() - before
        print(json.dumps({"stage": "predicted_without_truth", "policy": name,
                          "accepted": int(masks[name].sum()), "collisions": collisions(claims, masks[name]),
                          "seconds": timings[name]}), flush=True)
    if not np.array_equal(masks["control"], masks["cursor_flags_off"]):
        raise AssertionError("Cursor flags-off differs from frozen control")
    args.output.mkdir(parents=True)
    metadata = {"scope": "complete_existing_development30k", "fresh_labels_read": False,
                "pairs": len(claims), "claimants": len(all_ids), "config": config,
                "input_sha256": {"pairs": sha(args.pairs), "frozen": sha(args.frozen), "cursor_policy": sha(args.policy),
                                 "evaluator": sha(__file__)}, "timing_seconds": timings,
                "collision_targets": {name: collisions(claims, mask) for name, mask in masks.items()}}
    universe = {"kind": "complete_development", "complete": True, "scoring_complete": True,
                "declared_references": 30000, "scored_references": 30000, "pairs": len(claims),
                "scope": "Existing exposed development graph only; no heldout density certification"}
    for split in ("early_stop", "selection"):
        directory = args.workbenches / split
        # Only previously exposed development files are accepted.
        manifest = json.loads((directory / "manifest.json").read_text())
        if (manifest.get("status") != "complete" or manifest.get("split") != "development"
                or manifest.get("verified_current_pipeline") is not True
                or manifest.get("audit_labels_read") is not False
                or manifest.get("second_stage_owner_excluded") is not True
                or manifest.get("frozen_sha256") != sha(args.frozen)):
            raise ValueError("Require sealed existing development labels and frozen control")
        if manifest.get("truth_sha256") and manifest["truth_sha256"] != sha(directory / "truth.json"):
            raise ValueError("Previously exposed truth hash changed")
        if manifest.get("countries_sha256") != sha(directory / "countries.json"):
            raise ValueError("Previously exposed countries changed")
        truth = json.loads((directory / "truth.json").read_text())
        countries = json.loads((directory / "countries.json").read_text())
        ids = ids_by_split[split]
        if set(truth) != set(ids) or set(countries) != set(ids):
            raise ValueError("Truth/countries differ from existing selected development IDs")
        selected = claims.source1_entity_id.isin(ids).to_numpy()
        frame = claims.loc[selected]
        baseline = predictions_for(frame, masks["control"][selected], ids)
        for name in policies:
            if name in ("control", "cursor_flags_off"):
                continue
            candidate = predictions_for(frame, masks[name][selected], ids)
            report = paired_evaluate(truth, baseline, candidate, countries,
                                     universe=universe, ownership=True, role="development", iterations=args.iterations)
            report.update(model=args.model, policy=name, split=split,
                          metadata=metadata, changed_entities=sum(baseline[e] != candidate[e] for e in ids),
                          labels_sha256=sha(directory / "truth.json"))
            (args.output / (split + "_" + name + ".json")).write_text(json.dumps(report, indent=2) + "\n")
            changes = []
            for e in ids:
                if baseline[e] != candidate[e]:
                    correct = set(truth[e])
                    removed, added = baseline[e] - candidate[e], candidate[e] - baseline[e]
                    changes.append({"entity_id": e, "country": countries[e],
                                    "delta": entity_f05(correct, candidate[e]) - entity_f05(correct, baseline[e]),
                                    "true_removed": sorted(removed & correct), "false_removed": sorted(removed - correct),
                                    "true_added": sorted(added & correct), "false_added": sorted(added - correct)})
            (args.output / (split + "_" + name + "_changes.json")).write_text(json.dumps(changes, indent=2) + "\n")
            print(json.dumps({"stage": "evaluated", "model": args.model, "split": split, "policy": name,
                              "baseline_f05": report["baseline"]["macro_f05"], "candidate_f05": report["candidate"]["macro_f05"],
                              "delta": report["paired_macro_f05_delta"], "ci": report["paired_delta_95pct_ci"],
                              "links": report["links"]}), flush=True)
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--workbenches", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--iterations", type=int, default=1000)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
