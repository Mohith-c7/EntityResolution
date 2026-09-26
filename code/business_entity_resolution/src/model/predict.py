"""
Prediction and decision layer for the business entity resolution pipeline.

Provides:
- predict()         : score candidate pairs (Requirements 6.3)
- select_matches()  : one match per S1 per source, threshold-gated (Requirements 4.1, 4.2)
- write_outputs()   : write matching_results.tsv and candidate_pairs.tsv (Requirements 4.3, 4.4)
"""
from __future__ import annotations

import os

import lightgbm as lgb
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------

def predict(model: lgb.Booster, feature_matrix: np.ndarray) -> np.ndarray:
    """
    Return probability scores in [0, 1] for each row of feature_matrix.

    Parameters
    ----------
    model : lgb.Booster
        Trained LightGBM model.
    feature_matrix : np.ndarray shape [N, 36]
        Feature vectors for N candidate pairs.

    Returns
    -------
    np.ndarray shape [N] of float probabilities in [0, 1].

    Requirements: 6.3
    """
    return model.predict(feature_matrix)


# ---------------------------------------------------------------------------
# select_matches
# ---------------------------------------------------------------------------

def select_matches(
    candidate_pairs: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float,
    all_s1_ids: list | None = None,
) -> pd.DataFrame:
    """
    For each S1 entity, select ALL candidates with probability >= threshold
    per source (S1-S2 and S1-S3 independently).

    An S1 entity can match zero or many S2/S3 records. This emits exactly one
    row per unique S1 entity; matched_entity_ids is a comma-separated list of
    all IDs above threshold, empty string if none.

    Parameters
    ----------
    candidate_pairs : pd.DataFrame
        Must contain 's1_id', 's2_id' (or None), 's3_id' (or None), 'source'.
    probabilities : np.ndarray shape [N]
        Predicted match probabilities aligned to candidate_pairs rows.
    threshold : float
        Minimum probability required to select a match.
    all_s1_ids : list or None
        Complete list of S1 entity IDs (including singletons with no candidates).
        If provided, ensures every S1 ID gets a row even with no candidates.

    Returns
    -------
    pd.DataFrame with columns ['s1_id', 's2_id', 's3_id'] where s2_id and s3_id
    contain comma-separated match IDs or empty string.

    Requirements: 4.1, 4.2, 4.3
    """
    df = candidate_pairs.reset_index(drop=True).copy()
    df["_prob"] = probabilities

    # Collect all S1 IDs — from candidates plus any extra singletons
    candidate_s1_ids = list(df["s1_id"].unique())
    if all_s1_ids is not None:
        all_ids = list(dict.fromkeys(list(all_s1_ids) + candidate_s1_ids))
    else:
        all_ids = candidate_s1_ids

    # Separate S1-S2 and S1-S3 candidates above threshold
    above = df[df["_prob"] >= threshold]
    s1s2_above = above[above["source"].str.upper().str.contains("S1-S2", na=False)]
    s1s3_above = above[above["source"].str.upper().str.contains("S1-S3", na=False)]

    # Build {s1_id: comma-separated match IDs} for each source
    def _multi_matches(sub: pd.DataFrame, id_col: str) -> dict[str, str]:
        result: dict[str, str] = {}
        for s1_id, group in sub.groupby("s1_id"):
            ids = [str(v) for v in group[id_col].dropna() if str(v).strip()]
            if ids:
                result[s1_id] = ",".join(ids)
        return result

    s2_matches = _multi_matches(s1s2_above, "s2_id")
    s3_matches = _multi_matches(s1s3_above, "s3_id")

    rows = []
    for s1_id in all_ids:
        rows.append({
            "s1_id": s1_id,
            "s2_id": s2_matches.get(s1_id, ""),
            "s3_id": s3_matches.get(s1_id, ""),
        })

    return pd.DataFrame(rows, columns=["s1_id", "s2_id", "s3_id"])


# ---------------------------------------------------------------------------
# write_outputs
# ---------------------------------------------------------------------------

def write_outputs(
    matches: pd.DataFrame,
    candidate_pairs: pd.DataFrame,
    output_dir: str,
) -> None:
    """
    Write matching_results.tsv and candidate_pairs.tsv to output_dir.

    Release format (team plan section 11):
      matching_results.tsv  : source1_entity_id<TAB>matched_entity_ids
                              matched_entity_ids = comma-separated S2/S3 IDs, empty if no match
      candidate_pairs.tsv   : source1_entity_id<TAB>candidate_entity_ids
                              candidate_entity_ids = comma-separated candidate IDs

    Unmatched source fields are represented as empty strings.

    Requirements: 4.3, 4.4
    """
    os.makedirs(output_dir, exist_ok=True)

    matches_path = os.path.join(output_dir, "matching_results.tsv")
    candidates_path = os.path.join(output_dir, "candidate_pairs.tsv")

    # --- matching_results.tsv ---
    # Combine s2_id and s3_id into a comma-separated matched_entity_ids list per S1
    def _combine_ids(row: pd.Series) -> str:
        ids = []
        s2 = str(row.get("s2_id", "") or "").strip()
        s3 = str(row.get("s3_id", "") or "").strip()
        if s2:
            ids.append(s2)
        if s3:
            ids.append(s3)
        return ",".join(ids)

    matches_out = matches.copy()
    matches_out["matched_entity_ids"] = matches_out.apply(_combine_ids, axis=1)
    matches_out = matches_out.rename(columns={"s1_id": "source1_entity_id"})
    matches_out[["source1_entity_id", "matched_entity_ids"]].to_csv(
        matches_path, sep="\t", index=False
    )

    # --- candidate_pairs.tsv ---
    # Exactly 2 columns, exactly 1 row per S1 entity.
    # Aggregate all candidate IDs per S1 into a comma-separated list.
    cands_out = candidate_pairs.copy()
    for col in ("s2_id", "s3_id"):
        if col in cands_out.columns:
            cands_out[col] = cands_out[col].fillna("")

    cands_out["_cand_id"] = cands_out.apply(
        lambda r: str(r.get("s2_id", "") or r.get("s3_id", "") or "").strip(),
        axis=1,
    )

    # One row per S1: aggregate all candidate IDs
    cands_grouped = (
        cands_out[cands_out["_cand_id"] != ""]
        .groupby("s1_id")["_cand_id"]
        .apply(lambda ids: ",".join(ids))
        .reset_index()
        .rename(columns={"s1_id": "source1_entity_id", "_cand_id": "candidate_entity_ids"})
    )

    # Ensure every S1 from matches appears, even if no candidates
    all_s1 = matches_out[["source1_entity_id"]].drop_duplicates()
    cands_final = all_s1.merge(cands_grouped, on="source1_entity_id", how="left")
    cands_final["candidate_entity_ids"] = cands_final["candidate_entity_ids"].fillna("")

    cands_final[["source1_entity_id", "candidate_entity_ids"]].to_csv(
        candidates_path, sep="\t", index=False
    )
