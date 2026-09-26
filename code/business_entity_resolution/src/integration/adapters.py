"""
Integration adapters — bridge between teammates' column naming conventions
and the internal naming convention used throughout src/features/, src/model/, src/evaluation/.

Internal convention:
  candidate_pairs columns : s1_id, s2_id, s3_id, blocking_score, source, [rank]
  ground_truth columns    : s1_id, match_id  (one row per match)
  source_records keys     : "S1", "S2", "S3"  (DataFrames indexed by entity_id)

Harsha's convention (generate_candidates output):
  source1_entity_id, candidate_entity_id, candidate_source ("S2" or "S3"),
  blocking_score, [score_* path columns]

Mohit's convention (load_train_source* / load_ground_truth output):
  load_train_source*  -> DataFrame with columns: entity_id, business_name, business_address, country
  load_ground_truth   -> DataFrame with columns: source1_entity_id, matched_entity_ids (comma-sep str)
"""
from __future__ import annotations

import pandas as pd


# ---------------------------------------------------------------------------
# Harsha's blocking output → internal candidate_pairs
# ---------------------------------------------------------------------------

def adapt_candidates(harsha_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert Harsha's generate_candidates() output to the internal format.

    Harsha columns  →  Internal columns
    ─────────────────────────────────────
    source1_entity_id   →  s1_id
    candidate_entity_id →  s2_id  (if candidate_source == "S2")
                        →  s3_id  (if candidate_source == "S3")
    candidate_source    →  source  ("S1-S2" or "S1-S3")
    blocking_score      →  blocking_score  (unchanged)

    Parameters
    ----------
    harsha_df : pd.DataFrame
        Output of generate_candidates(source1, source2, source3).

    Returns
    -------
    pd.DataFrame with columns: s1_id, s2_id, s3_id, blocking_score, source
    """
    df = harsha_df.copy()

    df = df.rename(columns={"source1_entity_id": "s1_id"})

    df["source"] = df["candidate_source"].map({"S2": "S1-S2", "S3": "S1-S3"})

    df["s2_id"] = df.apply(
        lambda r: r["candidate_entity_id"] if r["candidate_source"] == "S2" else None,
        axis=1,
    )
    df["s3_id"] = df.apply(
        lambda r: r["candidate_entity_id"] if r["candidate_source"] == "S3" else None,
        axis=1,
    )

    return df[["s1_id", "s2_id", "s3_id", "blocking_score", "source"]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Mohit's loader output → internal source_records dict
# ---------------------------------------------------------------------------

def adapt_source_records(
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """
    Convert Mohit's load_train_source*() DataFrames into the source_records dict
    expected by build_feature_matrix().

    Sets entity_id as index and renames business_name → name,
    business_address → address to match feature builder field names.

    Returns
    -------
    dict with keys "S1", "S2", "S3"; each value is a DataFrame indexed by entity_id.
    """
    def _prep(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if "entity_id" in out.columns:
            out = out.set_index("entity_id")
        out = out.rename(columns={
            "business_name": "name",
            "business_address": "address",
        })
        return out

    return {
        "S1": _prep(s1_df),
        "S2": _prep(s2_df),
        "S3": _prep(s3_df),
    }


# ---------------------------------------------------------------------------
# Mohit's ground truth → internal ground_truth (one row per match)
# ---------------------------------------------------------------------------

def adapt_ground_truth(mohit_gt: pd.DataFrame) -> pd.DataFrame:
    """
    Convert Mohit's load_ground_truth() output to the internal format.

    Mohit's format  →  Internal format
    ────────────────────────────────────
    source1_entity_id, matched_entity_ids (comma-sep string)
    →  s1_id, match_id  (one row per individual match)

    Singletons (empty matched_entity_ids) produce no rows.

    Returns
    -------
    pd.DataFrame with columns: s1_id, match_id
    """
    rows = []
    for _, row in mohit_gt.iterrows():
        s1_id = row["source1_entity_id"]
        raw = row["matched_entity_ids"]

        if isinstance(raw, set):
            match_ids = raw
        elif isinstance(raw, str) and raw.strip():
            match_ids = {m.strip() for m in raw.split(",") if m.strip()}
        else:
            match_ids = set()

        for match_id in match_ids:
            rows.append({"s1_id": s1_id, "match_id": match_id})

    return pd.DataFrame(rows, columns=["s1_id", "match_id"])
