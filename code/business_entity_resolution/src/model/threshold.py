"""Threshold selection uses tuning entities, never the final holdout."""

import numpy as np

from ..evaluation.metrics import score_matches


def select_matches(pairs, probabilities, threshold: float, all_source1_ids) -> dict[str, list[str]]:
    probabilities = np.asarray(probabilities)
    if probabilities.shape != (len(pairs),) or not np.all(np.isfinite(probabilities)):
        raise ValueError("Probability count must match candidate rows and values must be finite")
    if not 0 <= threshold <= 1 or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Threshold and probabilities must be in [0,1]")
    result = {eid: [] for eid in all_source1_ids}
    for eid, cid, probability in zip(pairs["source1_entity_id"], pairs["candidate_entity_id"], probabilities):
        if eid not in result:
            raise ValueError(f"Unknown reference {eid}")
        if probability >= threshold:
            result[eid].append(cid)
    return {eid: sorted(set(ids)) for eid, ids in result.items()}


def tune_threshold(pairs, probabilities, truth: dict) -> tuple[float, list[dict]]:
    rows = []
    for threshold in np.r_[np.arange(.30, .951, .01), .96, .97, .98, .99, .995, .999]:
        predictions = select_matches(pairs, probabilities, float(threshold), truth)
        rows.append({"threshold": round(float(threshold), 6), **score_matches(truth, predictions)})
    best = max(rows, key=lambda row: (row["macro_f05"], row["micro_precision"], row["threshold"]))
    return best["threshold"], rows
