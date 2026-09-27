"""One predeclared, development-only residual-strength comparison.

Select on exposed early10k, then evaluate that one winner on selection20k.
Every decision uses the full existing30k graph. No training or fresh labels.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'research/sprint_6h')]
from evaluate_sprint import decide_control, paired_evaluate, predictions_for
from export_sprint_workbench import sha
import neural_adapter as nn

STRENGTHS = [0.5, 0.75, 1., 1.25, 1.5]
KEYS = ['source1_entity_id', 'candidate_entity_id', 'candidate_order']
B = Path('research/sprint_6h/sibling')


def route_residuals(original, selected, features):
    if (original.duplicated(KEYS[:2]).any() or features.duplicated(KEYS[:2]).any()
            or not original[KEYS].equals(selected[KEYS])
            or not np.array_equal(original.probability.to_numpy(), selected.probability_original.to_numpy())):
        raise ValueError('Complete graph keys/order/frozen p2 changed')
    positions = pd.MultiIndex.from_frame(original[KEYS[:2]]).get_indexer(pd.MultiIndex.from_frame(features[KEYS[:2]]))
    if (positions < 0).any() or not np.array_equal(original.probability.to_numpy()[positions], features.probability.to_numpy()):
        raise ValueError('Foreign route or changed frozen anchor')
    mask = np.ones(len(original), dtype=bool); mask[positions] = False
    if not np.array_equal(original.probability.to_numpy()[mask], selected.probability.to_numpy()[mask]):
        raise ValueError('Selected model changed outside-route p2')
    base = original.probability.to_numpy()[positions]
    trial = selected.probability.to_numpy()[positions]
    if not np.isfinite(trial).all() or ((trial < 0) | (trial > 1)).any():
        raise ValueError('Invalid selected routed probability')
    changed = base != trial
    if ((trial[changed] <= 0) | (trial[changed] >= 1)).any():
        raise ValueError('Saturated probability cannot recover its saved residual')
    residual = np.zeros(len(trial), dtype=np.float64)
    # Invert the exact stored anchored probabilities, not a new model fit.
    # Keep unchanged rows at zero to preserve exact frozen 0/1 anchors.
    residual[changed] = np.log(trial[changed]) - np.log1p(-trial[changed]) - nn.reverse.base_logit(base[changed])
    return positions, residual


def apply_strength(original, selected, positions, residual, strength):
    if strength not in STRENGTHS:
        raise ValueError('Undeclared residual strength')
    result = original.copy(deep=False)
    probability = original.probability.to_numpy().copy()
    if strength == 1:
        # Avoid even a floating-point round trip for the selected control.
        probability[positions] = selected.probability.to_numpy()[positions]
    else:
        probability[positions] = nn.reverse.corrected_probability(probability[positions], strength * residual)
    result['probability'] = probability
    result['probability_original'] = original.probability.to_numpy()
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selected', type=Path, default=Path('research/final_2h/neural_last4_fine_v1'))
    p.add_argument('--output', type=Path, default=Path('research/final_2h/residual_strength_v1'))
    a = p.parse_args()
    if a.output.exists():
        raise ValueError('One new bounded experiment output required')
    a.output.mkdir(parents=True)
    predeclare = {'status': 'predeclared_before_label_reads', 'declared_at': datetime.now(timezone.utc).isoformat(),
        'strengths': STRENGTHS, 'early_selection': 'exposed early10k only',
        'eligibility': 'Precision >= selected strength1; singleton false positives <= selected strength1; no country delta below -0.0005',
        'tie_break': 'Highest early macro F0.5, then closest strength to1, then smaller strength',
        'selection_evaluation': 'Winner once, exposed selection20k versus selected strength1',
        'decision_scope': 'Complete existing30k claimant graph at unchanged original thresholds',
        'source_sha256': sha(__file__), 'original_pairs_sha256': sha(B / 'learned30k/pairs.parquet'),
        'selected_pairs_sha256': sha(a.selected / 'pairs.parquet'), 'selected_report_sha256': sha(a.selected / 'report.json'),
        'early_truth_path': str(B / 'workbenches50k/early_stop/truth.json'),
        'selection_truth_path': str(B / 'workbenches50k/selection/truth.json'),
        'fresh_audit_labels_read': False, 'confirmation_labels_read': False, 'new_training': False}
    (a.output / 'predeclare.json').write_text(json.dumps(predeclare, indent=2) + '\n')
    original = pd.read_parquet(B / 'learned30k/pairs.parquet')
    selected = pd.read_parquet(a.selected / 'pairs.parquet')
    features, fm = nn.load_features(B / 'neural_v1_features30k')
    report = json.loads((a.selected / 'report.json').read_text())
    if (len(fm['reference_ids']) != 30000 or set(original.source1_entity_id) & set(report['fit_owners'])
            or fm['input_pairs_sha256'] != predeclare['original_pairs_sha256']
            or report.get('mode') is not None or report.get('calibration_logit_shift', 0) != 0):
        raise ValueError('Wrong source graph, fitted owner overlap or non-anchored model')
    positions, residual = route_residuals(original, selected, features)
    config = json.loads(Path('research/sprint_6h/coordination/baseline_freeze/frozen.json').read_text())
    config = {k: config[k] for k in ['threshold', 't_first', 't_rest']}
    chosen, _ = decide_control(selected, selected, config)
    truth = json.loads((B / 'workbenches50k/early_stop/truth.json').read_text())
    countries = json.loads((B / 'workbenches50k/early_stop/countries.json').read_text())
    if len(truth) != 10000 or not set(truth) <= set(fm['reference_ids']) or set(truth) & set(report['fit_owners']):
        raise ValueError('Wrong existing early cohort')
    mask = original.source1_entity_id.isin(truth).to_numpy()
    baseline = predictions_for(selected.loc[mask], chosen[mask], truth)
    grid = []
    for strength in STRENGTHS:
        frame = apply_strength(original, selected, positions, residual, strength)
        keep, _ = decide_control(frame, frame, config)
        result = paired_evaluate(truth, baseline, predictions_for(frame.loc[mask], keep[mask], truth), countries, role='development')
        eligible = (result['candidate']['micro_precision'] >= result['baseline']['micro_precision']
            and result['candidate']['singleton_false_positives'] <= result['baseline']['singleton_false_positives']
            and min(result['country_deltas'].values()) >= -.0005)
        row = {'strength': strength, 'eligible': eligible, 'evaluation': result}
        grid.append(row)
        print(json.dumps({'stage': 'early_grid', 'strength': strength, 'eligible': eligible,
            'score': result['candidate']['macro_f05'], 'delta': result['paired_macro_f05_delta'],
            'precision': result['candidate']['micro_precision'], 'links': result['links']}), flush=True)
    eligible = [r for r in grid if r['eligible']]
    winner = min(eligible, key=lambda r: (-r['evaluation']['candidate']['macro_f05'], abs(r['strength'] - 1), r['strength']))
    selection = {'status': 'winner_selected_before_selection_labels', 'selected_at': datetime.now(timezone.utc).isoformat(),
        'strength': winner['strength'], 'grid': grid, 'decision_config': config,
        'predeclare_sha256': sha(a.output / 'predeclare.json'), 'selection_labels_read': False,
        'fresh_audit_labels_read': False, 'confirmation_labels_read': False}
    (a.output / 'early_selection.json').write_text(json.dumps(selection, indent=2) + '\n')
    # The only selection20k evaluation is below, after the winner seal exists.
    st = json.loads((B / 'workbenches50k/selection/truth.json').read_text())
    sc = json.loads((B / 'workbenches50k/selection/countries.json').read_text())
    if len(st) != 20000 or set(st) & set(truth) or set(st) & set(report['fit_owners']) or set(st) | set(truth) != set(fm['reference_ids']):
        raise ValueError('Wrong existing selection cohort')
    frame = apply_strength(original, selected, positions, residual, winner['strength'])
    keep, _ = decide_control(frame, frame, config)
    smask = original.source1_entity_id.isin(st).to_numpy()
    result = paired_evaluate(st, predictions_for(selected.loc[smask], chosen[smask], st),
        predictions_for(frame.loc[smask], keep[smask], st), sc, role='development')
    frame.to_parquet(a.output / 'pairs.parquet', index=False)
    final = {'status': 'bounded_development_experiment_complete', 'winner_strength': winner['strength'],
        'scope': 'One early10k-selected strength evaluated once on exposed selection20k; complete30k decisions; no fresh evidence',
        'selection_vs_selected_strength1': result, 'early_selection_sha256': sha(a.output / 'early_selection.json'),
        'selected_pairs_sha256': predeclare['selected_pairs_sha256'], 'winner_pairs_sha256': sha(a.output / 'pairs.parquet'),
        'residual_source': 'Inverse logit of saved anchored probabilities; saturated changed rows rejected; strength1 uses exact saved bytes',
        'routed_rows': len(positions), 'unrouted_p2_unchanged': True, 'fit_owners': report['fit_owners'],
        'source_sha256': sha(__file__), 'decision_config': config, 'fresh_audit_labels_read': False,
        'confirmation_labels_read': False, 'selected_freeze_or_exporter_amended': False}
    (a.output / 'report.json').write_text(json.dumps(final, indent=2) + '\n')
    print(json.dumps({'stage': 'selection_once', 'strength': winner['strength'], 'score': result['candidate']['macro_f05'],
        'delta': result['paired_macro_f05_delta'], 'ci': result['paired_delta_95pct_ci'],
        'links': result['links'], 'acceptance': result['acceptance']}), flush=True)


if __name__ == '__main__':
    main()
