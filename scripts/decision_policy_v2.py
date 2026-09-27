"""Version 2 ownership: fixed proposals, vetoes, all-claim arbitration, abstention.

No labels enter this module. The claims table must contain every candidate of
all references in the declared scored universe, including lower-band rescues.
"""
import numpy as np
import pandas as pd

PAIR_KEYS = ["source1_entity_id", "candidate_entity_id"]


def validate_claims(frame):
    required = set(PAIR_KEYS + ["probability"])
    if not required <= set(frame):
        raise ValueError(f"Missing columns: {sorted(required - set(frame))}")
    if frame[PAIR_KEYS].isna().any().any() or frame[PAIR_KEYS].astype(str).eq("").any().any():
        raise ValueError("Pair IDs must be nonempty")
    if frame.duplicated(PAIR_KEYS).any():
        raise ValueError("Duplicate pair keys")
    values = frame.probability.to_numpy(dtype=float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Probabilities must be finite and between zero and one")
    if "gate_veto" in frame and (frame.gate_veto.isna().any() or not frame.gate_veto.isin([True, False]).all()):
        raise ValueError("gate_veto must contain booleans")
    if "candidate_order" in frame:
        orders = frame.candidate_order.to_numpy(dtype=float)
        if not np.isfinite(orders).all() or (orders < 0).any() or (orders != np.floor(orders)).any():
            raise ValueError("Candidate order must contain nonnegative integers")
        if frame.duplicated(["source1_entity_id", "candidate_order"]).any():
            raise ValueError("Duplicate per-reference candidate order")


def form_proposals(claims, config):
    """Determine top candidate once, before veto/arbitration; never retry a loser."""
    validate_claims(claims)
    t_first, t_rest = float(config["t_first"]), float(config["t_rest"])
    if not 0 <= t_first <= t_rest <= 1:
        raise ValueError("Require 0 <= t_first <= t_rest <= 1")
    work = claims.reset_index(drop=True).copy()
    if "candidate_order" not in work:
        # Deterministic fallback for imports with no recorded retrieval order.
        work["candidate_order"] = work.groupby("source1_entity_id")["candidate_entity_id"].rank(method="dense").astype(int) - 1
    veto = work.get("gate_veto", pd.Series(False, index=work.index)).to_numpy(dtype=bool)
    ordinary = work.probability.to_numpy() >= t_rest
    top = work.sort_values(["source1_entity_id", "probability", "candidate_order", "candidate_entity_id"],
                           ascending=[True, False, True, True], kind="stable").groupby("source1_entity_id", sort=False).head(1).index
    first = np.zeros(len(work), dtype=bool); first[top] = True
    # A rescue is the single original top candidate below the ordinary band.
    rescue = first & ~ordinary & (work.probability.to_numpy() >= t_first)
    work["proposal_kind"] = np.where(ordinary, "ordinary", np.where(rescue, "rescue", "none"))
    work["proposed"] = (ordinary | rescue) & ~veto
    return work


def decide_v2(frame, claims, config):
    """Return (accepted mask, ownership-lost mask, row diagnostics + summary).

Score determines the owner only when unique and separated by the configured
margin. Street agreement alone never forces a winner. Every equal-score tie
abstains, even at margin zero. IDs make processing deterministic, not evidence.
"""
    validate_claims(frame)
    work = form_proposals(claims, config)
    margin = float(config.get("ownership_margin", 0.0))
    tolerance = float(config.get("tie_tolerance", 1e-12))
    minimum = float(config.get("ownership_min_score", 0.0))
    if not np.isfinite([margin, tolerance, minimum]).all() or margin < 0 or tolerance < 0 or not 0 <= minimum <= 1:
        raise ValueError("Invalid ownership margin, tie tolerance, or minimum score")
    work["accepted"] = False
    work["ownership_reason"] = np.where(work.proposed, "pending", np.where(work.get("gate_veto", False), "gate_veto", "ineligible"))
    proposed = work[work.proposed].sort_values(["candidate_entity_id", "probability", "source1_entity_id"],
                                            ascending=[True, False, True], kind="stable")
    # Vectorized top-two arbitration keeps complete submission universes practical.
    rank = proposed.groupby("candidate_entity_id", sort=False).cumcount()
    top = proposed[rank == 0]
    second = proposed[rank == 1].set_index("candidate_entity_id").probability
    rival = top.candidate_entity_id.map(second)
    difference = top.probability - rival
    tie = rival.notna() & (difference <= tolerance)
    too_close = rival.notna() & ~tie & (difference < margin)
    too_low = top.probability < minimum
    reason = pd.Series(np.select([too_low, tie, too_close],
                                 ["below_owner_minimum", "tie", "insufficient_margin"], default="owner"), index=top.index)
    winner_indices = top.index[reason == "owner"]
    work.loc[proposed.index, "ownership_reason"] = "lost_to_owner"
    work.loc[winner_indices, "accepted"] = True
    work.loc[winner_indices, "ownership_reason"] = "owner"
    rejected = top.loc[reason != "owner", ["candidate_entity_id"]].copy()
    rejected["reason"] = reason[reason != "owner"]
    abstain_reason = proposed.candidate_entity_id.map(rejected.set_index("candidate_entity_id").reason)
    rejected_positions = abstain_reason.index[abstain_reason.notna()]
    work.loc[rejected_positions, "ownership_reason"] = abstain_reason.loc[rejected_positions]
    collisions, ties, abstentions = len(second), int(tie.sum()), len(rejected)
    if frame is claims:
        # form_proposals preserves the complete input order. Avoid building two
        # enormous pair indexes for the common full-graph release operation.
        rows = work if list(work.columns[:2]) == PAIR_KEYS else work[PAIR_KEYS + [c for c in work if c not in PAIR_KEYS]]
    else:
        lookup = work.set_index(PAIR_KEYS)
        keys = pd.MultiIndex.from_frame(frame[PAIR_KEYS])
        if not keys.isin(lookup.index).all():
            raise ValueError("Evaluation frame contains pairs absent from claims")
        rows = lookup.loc[keys].reset_index()
    if not np.array_equal(rows.probability.to_numpy(), frame.probability.to_numpy()):
        raise ValueError("Frame scores differ from claimant universe")
    if "gate_veto" in frame and not np.array_equal(rows.get("gate_veto", pd.Series(False, index=rows.index)).to_numpy(), frame.gate_veto.to_numpy()):
        raise ValueError("Frame vetoes differ from claimant universe")
    chosen = rows.accepted.to_numpy(dtype=bool)
    lost = (rows.proposed & ~rows.accepted).to_numpy(dtype=bool)
    summary = {"policy_version": "ownership_v2", "claims_pairs": len(work), "claimants": int(work.source1_entity_id.nunique()),
               "ordinary_proposals": int(((work.proposal_kind == "ordinary") & work.proposed).sum()),
               "rescue_proposals": int(((work.proposal_kind == "rescue") & work.proposed).sum()),
               "accepted_links": int(work.accepted.sum()), "collision_targets": collisions,
               "tie_targets": ties, "no_owner_targets": abstentions,
               "rescues_accepted": int(((work.proposal_kind == "rescue") & work.accepted).sum()),
               "ownership_margin": margin, "tie_tolerance": tolerance, "ownership_min_score": minimum}
    return chosen, lost, {"rows": rows, "summary": summary}
