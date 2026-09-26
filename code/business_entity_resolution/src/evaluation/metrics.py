"""
Evaluation metrics implementation.

Implements precision, recall, F1, and competition-primary Macro F0.5 metrics.

Mathematical definition from ARCHITECTURE.md:
  Macro F_0.5 = (1 / |S1|) * sum_{s in S1} F_0.5(s)
  where for each entity s:
  F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
  Precision is weighted 2x over recall.
  Singletons: If true is empty and pred is empty, F_0.5 = 1.0. If true is empty and pred is non-empty, F_0.5 = 0.0.
  If true is non-empty and pred is empty, F_0.5 = 0.0.
"""
from __future__ import annotations

from typing import Any


def compute_entity_f_beta(
    pred_ids: set[str],
    true_ids: set[str],
    beta: float = 0.5,
) -> float:
    """
    Compute F_beta for a single Source 1 entity.

    Parameters
    ----------
    pred_ids : set[str]
        Predicted matching entity IDs for this S1 entity.
    true_ids : set[str]
        Ground truth matching entity IDs for this S1 entity (empty for singletons).
    beta : float
        Weight of precision vs recall (default 0.5 favors precision 2x).

    Returns
    -------
    float in [0.0, 1.0]
    """
    # Clean up empty strings or whitespace
    clean_pred = {str(x).strip() for x in pred_ids if str(x).strip()}
    clean_true = {str(x).strip() for x in true_ids if str(x).strip()}

    if not clean_true:
        # True singleton: correct if no matches predicted, else 0.0
        return 1.0 if not clean_pred else 0.0

    if not clean_pred:
        # Ground truth has matches but nothing predicted
        return 0.0

    tp = len(clean_pred & clean_true)
    if tp == 0:
        return 0.0

    precision = tp / len(clean_pred)
    recall = tp / len(clean_true)

    beta_sq = beta ** 2
    denom = (beta_sq * precision) + recall
    if denom == 0.0:
        return 0.0

    return ((1.0 + beta_sq) * precision * recall) / denom


def compute_macro_f_beta(
    predictions: dict[str, set[str]],
    ground_truth: dict[str, set[str]],
    all_s1_ids: set[str] | list[str] | None = None,
    beta: float = 0.5,
) -> float:
    """
    Compute Macro F_beta across all Source 1 entities.

    Parameters
    ----------
    predictions : dict[str, set[str]]
        Mapping s1_id -> set of predicted candidate entity IDs.
    ground_truth : dict[str, set[str]]
        Mapping s1_id -> set of true matching entity IDs.
    all_s1_ids : collection of str, optional
        Complete universe of S1 entity IDs. If None, uses union of keys from predictions and ground_truth.
    beta : float
        F-beta parameter (default 0.5).

    Returns
    -------
    float
        Macro F_beta score averaged across all S1 entities.
    """
    if all_s1_ids is None:
        eval_ids = set(predictions.keys()) | set(ground_truth.keys())
    else:
        eval_ids = set(all_s1_ids)

    if not eval_ids:
        return 0.0

    total_f = 0.0
    for s1_id in eval_ids:
        p_ids = predictions.get(s1_id, set())
        t_ids = ground_truth.get(s1_id, set())
        total_f += compute_entity_f_beta(p_ids, t_ids, beta=beta)

    return total_f / len(eval_ids)


def compute_macro_f05(
    predictions: dict[str, set[str]],
    ground_truth: dict[str, set[str]],
    all_s1_ids: set[str] | list[str] | None = None,
) -> float:
    """
    Convenience wrapper for Macro F0.5.
    """
    return compute_macro_f_beta(predictions, ground_truth, all_s1_ids=all_s1_ids, beta=0.5)


def compute_micro_precision_recall(
    predictions: dict[str, set[str]],
    ground_truth: dict[str, set[str]],
) -> tuple[float, float, float]:
    """
    Compute micro-averaged Precision, Recall, and F0.5 across all pairs.

    Returns
    -------
    (precision, recall, f05)
    """
    total_tp = 0
    total_pred = 0
    total_true = 0

    all_keys = set(predictions.keys()) | set(ground_truth.keys())
    for s1_id in all_keys:
        p = {str(x).strip() for x in predictions.get(s1_id, set()) if str(x).strip()}
        t = {str(x).strip() for x in ground_truth.get(s1_id, set()) if str(x).strip()}
        total_tp += len(p & t)
        total_pred += len(p)
        total_true += len(t)

    precision = total_tp / total_pred if total_pred > 0 else (1.0 if total_true == 0 else 0.0)
    recall = total_tp / total_true if total_true > 0 else 1.0

    denom = 0.25 * precision + recall
    f05 = (1.25 * precision * recall) / denom if denom > 0 else 0.0

    return precision, recall, f05
