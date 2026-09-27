"""Reserve exactly one new 10k audit extension from sealed Source1 records.

No ground truth, predictions or model quality are consulted. This is an audit
extension only; the consumed confirmation/audit reservation is not rewritten.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

DOMAIN = 'last8-extension-audit-20260927-v1'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def sample_ids(universe, excluded, count=10000):
    ids = list(universe)
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate original heldout owner')
    eligible = set(ids) - set(excluded)
    if len(eligible) < count:
        raise ValueError('Insufficient unused heldout owners')
    order = sorted(eligible, key=lambda e: (hashlib.sha256((DOMAIN + '\0' + e).encode()).digest(), e))
    return order[:count], len(eligible)


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dev-manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, default=Path('research/final_2h/extension_audit_last8_v1'))
    a = p.parse_args()
    if a.output.exists():
        raise ValueError('New extension directory required')
    original_plan_path = Path('research/sprint_6h/validation/fresh_splits/plan.json')
    original_plan = json.loads(original_plan_path.read_text())
    raw = Path('research/sprint_6h/validation/fresh_splits/heldout_universe.json')
    manifest_path = Path('models/sprint_6h/heldout_cohort/manifest.json')
    manifest = json.loads(manifest_path.read_text())
    if (sha(raw) != original_plan['claimant_universe']['records_sha256']
            or sha(raw) != manifest['references_sha256'] or len(manifest['reference_ids']) != 331012):
        raise ValueError('Original complete heldout Source1 records differ')
    records = json.loads(raw.read_text())
    ids = [r['entity_id'] for r in records]
    if set(ids) != set(manifest['reference_ids']) or len(ids) != 331012:
        raise ValueError('Source1 universe differs from complete scored graph')
    if any(set(r) != {'entity_id', 'business_name', 'business_address', 'country'} for r in records):
        raise ValueError('Only Source1 records may be read')
    excluded = set()
    for ledger in original_plan['exposure_ledgers']:
        if sha(ledger['path']) != ledger['sha256']:
            raise ValueError('Original exposure ledger changed')
        excluded.update(json.loads(Path(ledger['path']).read_text()))
    extra = set(); sources = []
    for role in ['confirmation', 'audit']:
        path = original_plan_path.parent / role / 'references.json'
        if sha(path) != original_plan['cohort_fingerprints'][role]['references_sha256']:
            raise ValueError('Consumed cohort changed')
        rows = json.loads(path.read_text()); extra.update(r['entity_id'] for r in rows)
        sources.append({'role': 'consumed_' + role, 'path': str(path), 'sha256': sha(path), 'owners': len(rows)})
    dev = json.loads(a.dev_manifest.read_text())
    if dev['status'] != 'complete' or len(dev['reference_ids']) != 30000:
        raise ValueError('Expected current complete30k development manifest')
    extra.update(dev['reference_ids'])
    sources.append({'role': 'current_early10k_plus_selection20k', 'path': str(a.dev_manifest),
        'sha256': sha(a.dev_manifest), 'owners': len(dev['reference_ids'])})
    fit_path = Path('research/final_2h/neural_last4_fine_v1/report.json')
    fit = json.loads(fit_path.read_text()); extra.update(fit['fit_owners'])
    sources.append({'role': 'current_residual20k_adapter_fit', 'path': str(fit_path), 'sha256': sha(fit_path), 'owners': len(fit['fit_owners'])})
    for path in [Path('models/sprint_6h/neural/frozen_head120k_v1/manifest.json'),
                 Path('models/final_2h/neural_last4_continue_v1/manifest.json')]:
        head = json.loads(path.read_text())
        if head['status'] != 'complete':
            raise ValueError('Incomplete fitted-head provenance')
        extra.update(head['training_owners'])
        sources.append({'role': 'neural_training_owners', 'path': str(path), 'sha256': sha(path), 'owners': len(head['training_owners'])})
    chosen, eligible = sample_ids(ids, excluded | extra)
    chosen_set = set(chosen)
    if chosen_set & excluded or chosen_set & extra or len(chosen_set) != 10000:
        raise ValueError('Extension audit contains exposed owners')
    by_id = {r['entity_id']: r for r in records}
    selected = [by_id[e] for e in chosen]
    a.output.mkdir(parents=True)
    write(a.output / 'additional_excluded_ids.json', sorted(extra))
    write(a.output / 'entity_ids.json', chosen)
    write(a.output / 'references.json', selected)
    additional = {'path': str(a.output / 'additional_excluded_ids.json'),
        'sha256': sha(a.output / 'additional_excluded_ids.json'), 'ids': len(extra)}
    plan = {'status': 'extension_audit_reserved_no_labels_read', 'kind': 'fresh_extension_audit_only',
        'reserved_at': datetime.now(timezone.utc).isoformat(), 'sampling_domain': DOMAIN, 'scope_reviewed': True,
        'exactly_one_new_cohort': True, 'confirmation_cohort_created': False,
        'reserved_before_new_head_evaluation': True, 'labels_read': False, 'ground_truth_read': False,
        'original_plan': {'path': str(original_plan_path), 'sha256': sha(original_plan_path)},
        'original_heldout_manifest': {'path': str(manifest_path), 'sha256': sha(manifest_path)},
        'source1_records': {'path': str(raw), 'sha256': sha(raw), 'owners': len(ids)},
        'eligible_unused_heldout_owners': eligible, 'exposure_ledgers': [*original_plan['exposure_ledgers'], additional],
        'additional_exclusion_sources': sources,
        'cohort_fingerprints': {'audit': {'entities': 10000,
            'reference_ids_sha256': hashlib.sha256('\n'.join(chosen).encode()).hexdigest(),
            'references_sha256': sha(a.output / 'references.json')}},
        'country_counts': dict(Counter(r['country'] for r in selected)),
        'new_head_lineage_requirement': 'Eight-layer head must use identical outer-training owners/input and three-epoch initializer as the excluded four-layer head',
        'audit_policy': 'Labels remain unread if the new head fails its predeclared development acceptance. Any later audit requires a new candidate freeze and predictions from all331012 claimants.',
        'verification_protocol': 'Distinct audit-extension plan; never claim this is the consumed confirmation-and-audit plan',
        'code_sha256': sha(__file__)}
    write(a.output / 'plan.json', plan)
    print(json.dumps({'status': plan['status'], 'plan_sha256': sha(a.output / 'plan.json'),
        'entities': 10000, 'eligible': eligible, 'countries': plan['country_counts'], 'labels_read': False}), flush=True)


if __name__ == '__main__':
    main()
