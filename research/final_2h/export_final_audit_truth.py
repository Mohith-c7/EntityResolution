"""Open only the reserved audit truth after all frozen release gates pass."""
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from evaluate_sprint import (verify_audit_freeze, verify_reservation,
                             verify_prediction_evidence, check_runtime)
from export_sprint_workbench import reference_rows, sha


def main():
    base = ROOT / 'research/final_2h/neural_last4_sealed_final_audit_v2'
    refs = ROOT / 'research/sprint_6h/validation/fresh_splits/audit/references.json'
    freeze = ROOT / 'reports/final_2h/neural_last4_final_audit_freeze.json'
    universe = ROOT / 'research/sprint_6h/validation/heldout_scoring_evidence.json'
    plan = ROOT / 'research/sprint_6h/validation/fresh_splits/plan.json'
    runtime = ROOT / 'research/final_2h/neural_last4_runtime_final_v2/runtime.json'
    output = base / 'audit_labels'
    if output.exists() or (refs.parent / 'audit_evaluation_started.json').exists():
        raise ValueError('Audit label export/evaluation already started')
    frozen = verify_audit_freeze(freeze, refs)
    verify_reservation(plan, refs, 'final_audit')
    args = SimpleNamespace(candidate=base / 'candidate_predictions.json',
        baseline=base / 'baseline_predictions.json', references=refs,
        universe=universe, freeze=freeze)
    verify_prediction_evidence(base / 'manifest.json', args, frozen)
    if not check_runtime(json.loads(runtime.read_text()), sha(freeze))['passed']:
        raise ValueError('Runtime gate failed')
    ids = [r['entity_id'] for r in reference_rows(refs)]
    if len(ids) != 10000 or len(set(ids)) != 10000:
        raise ValueError('Wrong reserved audit')
    output.mkdir()
    (output / 'labels_opening.json').write_text(json.dumps({
        'started_at': datetime.now(timezone.utc).isoformat(),
        'candidate_frozen_sha256': sha(freeze), 'references_sha256': sha(refs),
        'all_label_free_gates_passed': True, 'code_sha256': sha(__file__),
        'labels_not_read_before_this_marker': True}, indent=2)+'\n')
    truth = {owner: [] for owner in ids}
    database = ROOT / 'models/scale_v1_plan/aliases/counts.sqlite'
    with sqlite3.connect(f'{database.resolve().as_uri()}?mode=ro', uri=True) as conn:
        for start in range(0, len(ids), 500):
            group = ids[start:start+500]
            placeholders = ','.join('?' for _ in group)
            rows = conn.execute(f'SELECT id,outer_fold FROM refs WHERE id IN ({placeholders})', group).fetchall()
            if len(rows) != len(group) or any(fold != 'holdout' for _, fold in rows):
                raise ValueError('Audit owner is not in the outer holdout')
            for target, owner in conn.execute(f'SELECT id,owner FROM owners WHERE owner IN ({placeholders})', group):
                truth[owner].append(target)
    official = ROOT / 'dataset/train/train_ground_truth.tsv'
    before = sha(official); seen = set()
    with official.open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream, delimiter='\t'):
            owner = row['source1_entity_id']
            if owner not in truth:
                continue
            if owner in seen:
                raise ValueError('Duplicate official audit ground truth')
            seen.add(owner)
            targets = [x.strip() for x in row['matched_entity_ids'].split(',') if x.strip()]
            if len(targets) != len(set(targets)) or set(targets) != set(truth[owner]):
                raise ValueError('SQLite audit truth differs from official ground truth')
    if seen != set(ids) or sha(official) != before:
        raise ValueError('Official ground truth coverage/integrity failed')
    truth = {owner: sorted(targets) for owner, targets in truth.items()}
    path = output / 'truth.json'
    path.write_text(json.dumps(truth, sort_keys=True)+'\n')
    (output / 'manifest.json').write_text(json.dumps({
        'status': 'official_reserved_audit_truth_verified', 'owners': len(truth),
        'truth_sha256': sha(path), 'official_truth_sha256': before,
        'references_sha256': sha(refs), 'candidate_frozen_sha256': sha(freeze),
        'code_sha256': sha(__file__)}, indent=2)+'\n')
    print(json.dumps({'status': 'audit_truth_verified', 'owners': len(truth)}), flush=True)


if __name__ == '__main__':
    main()
