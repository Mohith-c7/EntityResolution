"""Prepare and confirm the preselected neural v2 on the full heldout graph.

Stages are separate. Preparation and prediction read no ownership labels;
only the explicit confirmation stage exports the reserved 10k labels once.
"""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'research/sprint_6h/validation'))
import numpy as np
import pandas as pd
from evaluate_sprint import decide_control, predictions_for, verify_reservation, verify_prediction_evidence
from export_sprint_workbench import sha, reference_rows

B = Path('research/sprint_6h/sibling')
V = Path('research/sprint_6h/validation')
N = Path('models/sprint_6h/neural')
C = Path('models/sprint_6h/heldout_cohort')
R = Path('research/sprint_6h/reverse_competition/heldout331k_v2')
F = Path('research/sprint_6h/coordination/baseline_freeze/frozen.json')
SELECT = Path('reports/sprint_6h/neural_v2_finalist_selection.json')
OUT = V / 'neural_v2_fresh_confirmation'


def run(script, *args):
    subprocess.run([sys.executable, script, *map(str, args)], check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['prepare', 'predict', 'evaluate'])
    a = p.parse_args()
    selected = json.loads(SELECT.read_text())
    if selected['candidate'] != 'neural_v2' or selected['fresh_labels_opened'] is not False:
        raise ValueError('Wrong preselected challenger')
    if a.stage == 'prepare':
        run('research/sprint_6h/sibling/train_reverse_adapter.py', 'join',
            '--pairs', C/'pairs.parquet', '--manifest', C/'manifest.json',
            '--reverse-features', R/'features.parquet', '--reverse-manifest', R/'manifest.json',
            '--route-plan', R/'global_route', '--output', B/'reverse_v1_features_heldout331k')
        run('research/sprint_6h/prepare_neural_inputs.py',
            '--features', B/'reverse_v1_features_heldout331k', '--data-dir', 'dataset/train',
            '--output', N/'neural_inputs/heldout331k_v2')
        return
    if a.stage == 'predict':
        run('research/sprint_6h/neural_adapter.py', 'join',
            '--features', B/'reverse_v1_features_heldout331k', '--neural', N/'heldout331k_v2',
            '--head-manifest', N/'frozen_head120k_v1/manifest.json',
            '--output', B/'neural_v2_features_heldout331k')
        model = json.loads((B/'neural_v2/manifest.json').read_text())
        fm = json.loads((B/'neural_v2_features_heldout331k/manifest.json').read_text())
        if model['neural_head_manifest_sha256'] != fm['neural_head_manifest_sha256'] or fm['neural_head_manifest_sha256'] != sha(N/'frozen_head120k_v1/manifest.json'):
            raise ValueError('Calibration and inference use different neural heads')
        run('research/sprint_6h/neural_adapter.py', 'rescore',
            '--pairs', C/'pairs.parquet', '--manifest', C/'manifest.json',
            '--features', B/'neural_v2_features_heldout331k', '--model', B/'neural_v2',
            '--output', B/'neural_v2_scored_heldout331k')
        source = json.loads((C/'manifest.json').read_text())
        ids = source['reference_ids']
        if set(ids) & set(model['training_owner_ids']): raise ValueError('Calibration fitted heldout owner')
        before = pd.read_parquet(C/'pairs.parquet')
        after = pd.read_parquet(B/'neural_v2_scored_heldout331k/pairs.parquet')
        keys = ['source1_entity_id', 'candidate_entity_id', 'candidate_order']
        if not before[keys].equals(after[keys]) or not np.array_equal(before.probability, after.probability_original):
            raise ValueError('Candidate set or baseline probabilities changed')
        frozen = json.loads(F.read_text()); config = {k:frozen[k] for k in ['threshold', 't_first', 't_rest']}
        base, _ = decide_control(before, before, config)
        trial, _ = decide_control(after, after, config)
        bp = predictions_for(before, base, ids); cp = predictions_for(after, trial, ids)
        OUT.mkdir(parents=True, exist_ok=False)
        for role in ['confirmation', 'audit']:
            refs = V/'fresh_splits'/role/'references.json'
            owners = [r['entity_id'] for r in reference_rows(refs)]
            for name, predictions in [('baseline', bp), ('candidate', cp)]:
                path = OUT/(role+'_'+name+'_predictions.json')
                path.write_text(json.dumps({e:sorted(predictions[e]) for e in owners})+'\n')
            evidence = {'status':'candidate_selected_before_confirmation',
                'selected_at':selected['selected_at'], 'predicted_at':datetime.now(timezone.utc).isoformat(),
                'candidate_sha256':sha(OUT/(role+'_candidate_predictions.json')),
                'baseline_sha256':sha(OUT/(role+'_baseline_predictions.json')),
                'references_sha256':sha(refs), 'universe_sha256':sha(V/'heldout_scoring_evidence.json'),
                'model_sha256':model['model_sha256'], 'selection_sha256':sha(SELECT),
                'full_claim_graph':{'references':len(ids),'pairs':len(before),
                    'baseline_collision_targets':int((before.loc[base].groupby('candidate_entity_id').source1_entity_id.nunique()>1).sum()),
                    'candidate_collision_targets':int((after.loc[trial].groupby('candidate_entity_id').source1_entity_id.nunique()>1).sum())},
                'source_manifest_sha256':sha(C/'manifest.json'), 'labels_read':False}
            (OUT/(role+'_prediction_evidence.json')).write_text(json.dumps(evidence,indent=2)+'\n')
        print('Complete graph predictions sealed; fresh labels remain unopened', flush=True)
        return
    # Verify sealed predictions and reservation BEFORE querying confirmation labels.
    refs = V/'fresh_splits/confirmation/references.json'
    plan = V/'fresh_splits/plan.json'
    args = argparse.Namespace(candidate=OUT/'confirmation_candidate_predictions.json',
        baseline=OUT/'confirmation_baseline_predictions.json', references=refs,
        universe=V/'heldout_scoring_evidence.json')
    verify_reservation(plan, refs, 'confirmation')
    verify_prediction_evidence(OUT/'confirmation_prediction_evidence.json', args)
    if (refs.parent/'confirmation_evaluation_started.json').exists():
        raise ValueError('Confirmation already consumed')
    label_dir = OUT/'confirmation_labels'; label_dir.mkdir(exist_ok=False)
    (label_dir/'export_started.json').write_text(json.dumps({'started_at':datetime.now(timezone.utc).isoformat(),
        'predictions_verified_before_truth':True,'references_sha256':sha(refs),
        'scope':'Only reserved confirmation owners; audit labels remain unopened'})+'\n')
    ids = [r['entity_id'] for r in reference_rows(refs)]
    truth = {e:[] for e in ids}
    owners = Path('models/scale_v1_plan/aliases/counts.sqlite')
    with sqlite3.connect(owners.resolve().as_uri()+'?mode=ro', uri=True) as conn:
        for start in range(0,len(ids),500):
            batch = ids[start:start+500]
            for target, owner in conn.execute('SELECT id,owner FROM owners WHERE owner IN ('+','.join('?'*len(batch))+')', batch):
                truth[owner].append(target)
    official = {}; wanted = set(ids)
    with Path('dataset/train/train_ground_truth.tsv').open(newline='') as stream:
        for row in csv.DictReader(stream, delimiter='\t'):
            owner = row['source1_entity_id']
            if owner in wanted:
                if owner in official: raise ValueError('Duplicate reserved truth owner')
                official[owner] = sorted(t for t in row['matched_entity_ids'].split(',') if t)
    if set(official) != wanted or any(sorted(truth[e]) != official[e] for e in ids):
        raise ValueError('Reserved ownership-index labels differ from official ground truth')
    path = label_dir/'truth.json'; path.write_text(json.dumps({e:sorted(v) for e,v in truth.items()})+'\n')
    run('scripts/evaluate_sprint.py', '--confirmation', '--truth', path, '--references', refs,
        '--baseline', args.baseline, '--candidate', args.candidate, '--universe', args.universe,
        '--reservation-plan', plan, '--prediction-evidence', OUT/'confirmation_prediction_evidence.json',
        '--output', OUT/'evaluation')


if __name__ == '__main__': main()
