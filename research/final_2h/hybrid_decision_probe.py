"""Bounded calibration on exposed early10k, with one selection20k comparison.

Preserves probabilities, candidate membership, ownership arbitration and the
original candidate-order tie rule. Never reads the extension audit.
"""
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'scripts')]
from evaluate_sprint import decide_control, predictions_for, paired_evaluate, entity_f05
from export_sprint_workbench import sha

B = Path('research/sprint_6h/sibling')
HYBRID = Path('research/final_2h/fixed_hybrid_v1')
OUT = Path('research/final_2h/hybrid_decision_probe_v1')
KEYS = ['source1_entity_id', 'candidate_entity_id']


def cached_decision_state(frame, config):
    original_keep, lost = decide_control(frame, frame, config)
    adjusted = np.where(lost, 0., frame.probability.to_numpy())
    codes, _ = pd.factorize(frame.source1_entity_id)
    order = np.lexsort((-adjusted, codes))
    first = np.zeros(len(frame), dtype=bool)
    if len(frame):
        first[order[np.r_[True, codes[order][1:] != codes[order][:-1]]]] = True
    if not np.array_equal(original_keep, select(adjusted, first, config)):
        raise ValueError('Cached decision differs from unchanged frozen semantics')
    return adjusted, first, original_keep


def select(adjusted, first, config):
    return (adjusted >= config['t_rest']) | (first & (adjusted >= config['t_first']))


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    config = json.loads(Path('research/sprint_6h/coordination/baseline_freeze/frozen.json').read_text())
    config = {k: config[k] for k in ['threshold', 't_rest', 't_first']}
    grid = [{**config, 't_rest': r, 't_first': f}
            for r in [.79, .81, config['t_rest'], .85, .87]
            for f in [.60, .65, config['t_first'], .75, .80]]
    protocol = {'status': 'declared_before_calibration_results',
        'declared_at': datetime.now(timezone.utc).isoformat(), 'source_sha256': sha(__file__),
        'hybrid_pairs_sha256': sha(HYBRID / 'pairs.parquet'),
        'baseline': 'Fixed hybrid using inherited thresholds; separate from already-passed model combination',
        'grid': grid, 'ownership_threshold_fixed': config['threshold'],
        'selection': 'Maximize early10k macro F0.5, break ties by nearest inherited settings then higher thresholds; score chosen setting once on exposed selection20k',
        'retain_changed_thresholds_only_if': {'selection_paired_ci_lower_above': 0.,
            'selection_minimum_point_gain_vs_fixed_hybrid': .0001,
            'country_delta_floor': -.0005},
        'no_model_route_or_feature_changes': True, 'extension_audit_labels_read': False}
    (OUT / 'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    frame = pd.read_parquet(HYBRID / 'pairs.parquet')
    if len(frame) != 1200000 or frame.source1_entity_id.nunique() != 30000 or frame.groupby('source1_entity_id').size().ne(40).any():
        raise ValueError('Require complete30k all40 graph')
    adjusted, first, original_keep = cached_decision_state(frame, config)
    early = json.loads((B/'workbenches50k/early_stop/truth.json').read_text())
    if len(early) != 10000: raise ValueError('Wrong calibration cohort')
    mask = frame.source1_entity_id.isin(early).to_numpy()
    part = frame[mask]
    rows = []
    for setting in grid:
        keep = select(adjusted, first, setting)
        pred = predictions_for(part, keep[mask], early)
        score = float(np.mean([entity_f05(early[e], pred[e]) for e in sorted(early)]))
        rows.append({'config': setting, 'macro_f05': score})
    def key(row):
        s = row['config']
        distance = abs(s['t_rest']-config['t_rest'])+abs(s['t_first']-config['t_first'])
        return (-row['macro_f05'], distance, -s['t_rest'], -s['t_first'])
    best = sorted(rows, key=key)[0]
    # Freeze the chosen setting before accessing the selection truth.
    (OUT/'chosen.json').write_text(json.dumps({'selected_on': 'early10k', **best}, indent=2)+'\n')
    truth = json.loads((B/'workbenches50k/selection/truth.json').read_text())
    countries = json.loads((B/'workbenches50k/selection/countries.json').read_text())
    if len(truth) != 20000 or set(early)&set(truth): raise ValueError('Wrong disjoint selection cohort')
    mask = frame.source1_entity_id.isin(truth).to_numpy()
    chosen = select(adjusted, first, best['config'])
    result = paired_evaluate(truth, predictions_for(frame[mask], original_keep[mask], truth),
        predictions_for(frame[mask], chosen[mask], truth), countries, role='development')
    result['legacy_guard_diagnostics'] = result.pop('acceptance')
    passed = (best['config'] != config and result['paired_macro_f05_delta'] >= .0001
        and result['paired_delta_95pct_ci'][0] > 0 and min(result['country_deltas'].values()) >= -.0005)
    report = {'status': 'bounded_decision_probe_complete', 'grid_early_results': rows,
        'chosen': best, 'selection': result, 'changed_thresholds_accepted': passed,
        'effective_config': best['config'] if passed else config,
        'protocol_sha256': sha(OUT/'protocol.json'), 'source_sha256': sha(__file__),
        'extension_audit_labels_read': False, 'production_changed': False}
    (OUT/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'chosen': best, 'selection_gain': result['paired_macro_f05_delta'],
        'ci': result['paired_delta_95pct_ci'], 'accepted': passed, 'effective_config': report['effective_config']}), flush=True)


if __name__ == '__main__': main()
