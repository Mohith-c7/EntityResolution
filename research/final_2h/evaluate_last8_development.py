"""Fixed anchored38 last-eight calibration; one exposed selection20k check.

Requires a previously reserved audit-only extension. Its labels are never read.
No thresholds, strengths, release freezes or submission models are changed.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'research/final_2h')]
from export_sprint_workbench import sha
from evaluate_sprint import decide_control, paired_evaluate, predictions_for

B = Path('research/sprint_6h/sibling')
N = Path('models/sprint_6h/neural')
FOUR = Path('research/final_2h/neural_last4_fine_v1')
EIGHT = Path('research/final_2h/neural_last8_fine_v1')
SELECTION = Path('research/final_2h/neural_last8_fine_v1_selection')
PROTOCOL = Path('research/final_2h/neural_last8_evaluation_protocol_v1.json')
HEAD = Path('models/final_2h/neural_last8_continue_v1/manifest.json')
FOUR_HEAD = Path('models/final_2h/neural_last4_continue_v1/manifest.json')
KEYS = ['source1_entity_id', 'candidate_entity_id', 'candidate_order']


def accepted(result):
    return (result['paired_macro_f05_delta'] >= .0003 and result['paired_delta_95pct_ci'][0] > 0
        and result['candidate']['micro_precision'] >= result['baseline']['micro_precision']
        and result['candidate']['singleton_false_positives'] <= result['baseline']['singleton_false_positives']
        and min(result['country_deltas'].values()) >= -.0005)


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reservation', type=Path, required=True)
    a = p.parse_args()
    if any(path.exists() for path in [PROTOCOL, EIGHT, SELECTION]):
        raise ValueError('One predeclared new candidate/evaluation required')
    reservation = json.loads((a.reservation / 'plan.json').read_text())
    ids = json.loads((a.reservation / 'entity_ids.json').read_text())
    if (reservation.get('status') != 'extension_audit_reserved_no_labels_read'
            or reservation.get('labels_read') is not False or reservation.get('ground_truth_read') is not False
            or reservation.get('exactly_one_new_cohort') is not True
            or reservation.get('confirmation_cohort_created') is not False
            or len(ids) != 10000 or len(set(ids)) != len(ids)
            or reservation['cohort_fingerprints']['audit']['references_sha256'] != sha(a.reservation / 'references.json')):
        raise ValueError('Missing prior fresh audit-only extension reservation')
    head = json.loads(HEAD.read_text()); old = json.loads(FOUR_HEAD.read_text())
    if (head.get('status') != 'complete' or head.get('completed_epochs') != 1
            or head.get('trainable_encoder_layers') != 8 or head.get('initializer_epochs') != 3
            or head.get('training_rows') != 120000 or set(head['training_owners']) != set(old['training_owners'])
            or head['training_input_sha256'] != old['training_input_sha256']
            or head['initializer_manifest_sha256'] != old['initializer_manifest_sha256']
            or set(ids) & set(head['training_owners'])
            or any(head.get(k) is not False for k in ['audit_labels_read', 'development_labels_read', 'test_records_read'])):
        raise ValueError('Eight-layer outer-training lineage or reservation exclusion differs')
    if datetime.fromisoformat(reservation['reserved_at']) > datetime.now(timezone.utc):
        raise ValueError('Invalid future reservation')
    protocol = {'status': 'predeclared_before_eight_layer_development_evaluation',
        'declared_at': datetime.now(timezone.utc).isoformat(), 'source_sha256': sha(__file__),
        'head_manifest_sha256': sha(HEAD), 'extension_plan_sha256': sha(a.reservation / 'plan.json'),
        'extension_labels_read': False, 'consumed_audit_or_confirmation_read': False,
        'calibration': 'Existing neural_peer fit-fine anchored38, 20k residual fit and 10k early stopping; fixed existing hyperparameters',
        'selection_evaluation': 'One full30k global decision replay; exposed selection20k once versus selected four-layer model',
        'acceptance': {'point_macro_f05_gain_at_least': .0003, 'paired_ci_lower_bound_above': 0.,
            'micro_precision_must_not_fall': True, 'singleton_false_predictions_must_not_rise': True,
            'country_delta_floor': -.0005}, 'no_strength_or_threshold_tuning': True,
        'four_layer_pairs_sha256': sha(FOUR / 'pairs.parquet'), 'four_layer_report_sha256': sha(FOUR / 'report.json')}
    write(PROTOCOL, protocol)
    cmd = [sys.executable, str(ROOT / 'research/final_2h/neural_peer.py'), 'fit-fine',
        '--features', str(B / 'neural_v1_features_fit'), '--dev-features', str(B / 'neural_v1_features30k'),
        '--fine-scores', str(N / 'last8_residual20k'), '--dev-fine-scores', str(N / 'last8_learned30k'),
        '--fine-input-manifest', str(N / 'neural_inputs/residual20k/manifest.json'),
        '--dev-fine-input-manifest', str(N / 'neural_inputs/learned30k/manifest.json'),
        '--fine-head-manifest', str(HEAD), '--output', str(EIGHT)]
    subprocess.run(cmd, cwd=ROOT, check=True)
    original = pd.read_parquet(B / 'learned30k/pairs.parquet')
    four = pd.read_parquet(FOUR / 'pairs.parquet'); eight = pd.read_parquet(EIGHT / 'pairs.parquet')
    report = json.loads((EIGHT / 'report.json').read_text())
    for frame in [four, eight]:
        if (not original[KEYS].equals(frame[KEYS])
                or not np.array_equal(original.probability.to_numpy(), frame.probability_original.to_numpy())):
            raise ValueError('Candidate graph/order/frozen p2 differs')
    if set(report['fit_owners']) & (set(original.source1_entity_id) | set(ids)):
        raise ValueError('Calibration owner entered development or extension audit')
    truth = json.loads((B / 'workbenches50k/selection/truth.json').read_text())
    countries = json.loads((B / 'workbenches50k/selection/countries.json').read_text())
    if len(truth) != 20000 or set(truth) & set(ids):
        raise ValueError('Wrong exposed selection20k or extension overlap')
    config = json.loads(Path('research/sprint_6h/coordination/baseline_freeze/frozen.json').read_text())
    config = {k: config[k] for k in ['threshold', 't_first', 't_rest']}
    predictions = []
    for frame in [four, eight]:
        chosen, _ = decide_control(frame, frame, config)
        mask = frame.source1_entity_id.isin(truth).to_numpy()
        predictions.append(predictions_for(frame.loc[mask], chosen[mask], truth))
    result = paired_evaluate(truth, predictions[0], predictions[1], countries, role='development')
    SELECTION.mkdir()
    final = {'status': 'one_development_selection_evaluation_complete', 'scope': 'Exposed selection20k; not a fresh audit',
        'vs_selected_four_layer': result, 'predeclared_acceptance_pass': accepted(result),
        'protocol_sha256': sha(PROTOCOL), 'head_manifest_sha256': sha(HEAD),
        'eight_layer_report_sha256': sha(EIGHT / 'report.json'),
        'extension_plan_sha256': sha(a.reservation / 'plan.json'), 'extension_labels_read': False,
        'consumed_audit_or_confirmation_read': False, 'source_sha256': sha(__file__)}
    write(SELECTION / 'report.json', final)
    print(json.dumps({'stage': 'selection_once', 'score': result['candidate']['macro_f05'],
        'gain': result['paired_macro_f05_delta'], 'ci': result['paired_delta_95pct_ci'],
        'precision': result['candidate']['micro_precision'], 'singleton_fp': result['candidate']['singleton_false_positives'],
        'acceptance': accepted(result), 'links': result['links'], 'country_deltas': result['country_deltas']}), flush=True)


if __name__ == '__main__':
    main()
