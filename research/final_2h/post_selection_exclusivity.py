"""Research-only exclusivity applied after the unchanged frozen decision policy."""
from pathlib import Path
import sys
import numpy as np


def exclusive_selected(frame, chosen):
    """Keep the best *selected* claim per target; preserve every other selection.

    Probability wins; an exact tie uses ascending Source1 ID. Unselected claims
    cannot participate. This operation never changes thresholds or promotes a
    rescue. Returns the final mask and only the selections removed here.
    """
    original = np.asarray(chosen, dtype=bool)
    if len(original) != len(frame):
        raise ValueError('Selection mask length differs from candidate rows')
    if frame.duplicated(['source1_entity_id', 'candidate_entity_id']).any():
        raise ValueError('Duplicate candidate pair')
    selected = frame.loc[original, ['source1_entity_id', 'candidate_entity_id', 'probability']].copy()
    if not np.isfinite(selected.probability.to_numpy()).all():
        raise ValueError('Selected probabilities must be finite')
    selected['_position'] = np.flatnonzero(original)
    winners = selected.sort_values(['probability', 'source1_entity_id'],
                                   ascending=[False, True], kind='stable')
    winners = winners.groupby('candidate_entity_id', sort=False).head(1)
    final = np.zeros(len(frame), dtype=bool)
    final[winners._position.to_numpy()] = True
    return final, original & ~final


def decide_post_selection(frame, claims, config):
    """Call the unchanged frozen control first, then arbitrate its selections."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
    from evaluate_sprint import decide_control
    original, original_lost = decide_control(frame, claims, config)
    final, exclusivity_lost = exclusive_selected(frame, original)
    return final, original_lost, exclusivity_lost
