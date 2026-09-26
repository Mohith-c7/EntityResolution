"""
Decision threshold optimization and calibration.

Implements empirical threshold sweep across [0.30, 0.95] optimizing Macro F0.5
on validation predictions as specified in ARCHITECTURE.md Section 13.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.legacy.evaluation.metrics import compute_entity_f_beta

logger = logging.getLogger(__name__)


def _extract_ground_truth_dict(ground_truth: pd.DataFrame) -> dict[str, set[str]]:
    """
    Convert ground truth DataFrame to dict[s1_id, set[match_id]].
    Supports both internal format (s1_id, match_id) and loader format
    (source1_entity_id, matched_entity_ids).
    """
    gt_dict: dict[str, set[str]] = {}

    if "source1_entity_id" in ground_truth.columns and "matched_entity_ids" in ground_truth.columns:
        for _, row in ground_truth.iterrows():
            s1 = str(row["source1_entity_id"]).strip()
            raw = row["matched_entity_ids"]
            if isinstance(raw, set):
                gt_dict[s1] = {str(x).strip() for x in raw if str(x).strip()}
            elif isinstance(raw, str) and raw.strip():
                gt_dict[s1] = {str(x).strip() for x in raw.split(",") if str(x).strip()}
            else:
                gt_dict[s1] = set()
    elif "s1_id" in ground_truth.columns and "match_id" in ground_truth.columns:
        for s1, group in ground_truth.groupby("s1_id"):
            gt_dict[str(s1).strip()] = {
                str(m).strip() for m in group["match_id"].dropna() if str(m).strip()
            }
    else:
        logger.warning("Unrecognized ground truth format; returning empty dict.")

    return gt_dict


def sweep_threshold(
    candidate_pairs: pd.DataFrame,
    probabilities: np.ndarray,
    ground_truth: pd.DataFrame,
    min_threshold: float = 0.30,
    max_threshold: float = 0.95,
    step: float = 0.01,
    all_s1_ids: list[str] | set[str] | None = None,
    beta: float = 0.5,
) -> tuple[float, float]:
    """
    Empirically sweep decision thresholds across [min_threshold, max_threshold]
    to strictly maximize official validation Macro F0.5.

    Parameters
    ----------
    candidate_pairs : pd.DataFrame
        Candidate pairs DataFrame. Must contain 's1_id', and candidate ID in
        's2_id'/'s3_id' or 'candidate_entity_id'.
    probabilities : np.ndarray
        Predicted match probabilities aligned 1:1 with candidate_pairs rows.
    ground_truth : pd.DataFrame
        Ground truth matches.
    min_threshold : float
        Lower bound of sweep range (default 0.30).
    max_threshold : float
        Upper bound of sweep range (default 0.95).
    step : float
        Step size between threshold candidates (default 0.01).
    all_s1_ids : collection of str, optional
        Complete set of validation S1 IDs (including singletons without candidates).
    beta : float
        F-beta parameter (default 0.5).

    Returns
    -------
    (best_threshold: float, best_macro_f05: float)
    """
    gt_dict = _extract_ground_truth_dict(ground_truth)

    # Extract candidate IDs
    def _get_cand_id(row: pd.Series) -> str:
        c = row.get("candidate_entity_id") or row.get("s2_id") or row.get("s3_id")
        return str(c).strip() if c and not pd.isna(c) else ""

    s1_ids = candidate_pairs["s1_id"].astype(str).str.strip().tolist()
    cand_ids = [_get_cand_id(row) for _, row in candidate_pairs.iterrows()]
    probs = np.asarray(probabilities, dtype=float)

    if all_s1_ids is not None:
        eval_s1_ids = set(str(x).strip() for x in all_s1_ids)
    else:
        eval_s1_ids = set(s1_ids) | set(gt_dict.keys())

    if not eval_s1_ids:
        logger.warning("No evaluation S1 IDs found for threshold sweep. Defaulting to 0.65.")
        return 0.65, 0.0

    thresholds = np.arange(min_threshold, max_threshold + (step / 2.0), step)
    best_threshold = 0.65
    best_f_beta = -1.0

    # Pre-organize pairs for fast evaluation per threshold
    num_pairs = len(probs)

    for tau in thresholds:
        tau_val = round(float(tau), 4)
        pred_dict: dict[str, set[str]] = {s1: set() for s1 in eval_s1_ids}

        for i in range(num_pairs):
            if probs[i] >= tau_val:
                s1 = s1_ids[i]
                c_id = cand_ids[i]
                if s1 in pred_dict and c_id:
                    pred_dict[s1].add(c_id)

        # Compute Macro F_beta across all eval_s1_ids
        total_f = 0.0
        for s1 in eval_s1_ids:
            p_ids = pred_dict.get(s1, set())
            t_ids = gt_dict.get(s1, set())
            total_f += compute_entity_f_beta(p_ids, t_ids, beta=beta)

        macro_f = total_f / len(eval_s1_ids)

        if macro_f > best_f_beta:
            best_f_beta = macro_f
            best_threshold = tau_val

    logger.info("Threshold sweep: best_threshold=%.4f with Macro F%.1f=%.4f", best_threshold, beta, best_f_beta)
    return best_threshold, best_f_beta
