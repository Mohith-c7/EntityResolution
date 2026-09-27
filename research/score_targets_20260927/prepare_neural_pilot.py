"""Prepare an isolated text-matching pilot from existing training caches.

No test or audit data is read. Candidate budgets are unchanged. This pilot uses
retrieval-rank hard negatives, not in-sample model scores. Validation retains
all cached candidates and full per-reference truth, including empty lists.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import pandas as pd
import pyarrow as pa


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for part in iter(lambda: handle.read(2**20), b''):
            h.update(part)
    return h.hexdigest()


def priority(entity):
    return hashlib.sha256(('neural-pilot-v1:' + entity).encode()).digest()


def collect(cache, wanted, training):
    manifest = json.loads((cache / 'manifest.json').read_text())
    if manifest['split'] != ('train' if training else 'tune'):
        raise ValueError('Wrong cache split')
    frames, truth, hashes = [], {}, {}
    for path in sorted(cache.glob('[0-9]*.json')):
        metadata = json.loads(path.read_text())
        rows = [r for r in metadata['entity_rows'] if r['entity_id'] in wanted]
        if not rows:
            continue
        parquet = path.with_suffix('.parquet')
        sha = digest(parquet)
        if sha != metadata['parquet_sha256']:
            raise ValueError(f'Corrupt cache: {parquet}')
        hashes[str(parquet)] = sha
        columns = ['source1_entity_id', 'candidate_entity_id', 'label', 'blocking_score']
        frame = pd.read_parquet(parquet, columns=columns)
        frame = frame[frame.source1_entity_id.isin(wanted)]
        if frame.duplicated(['source1_entity_id', 'candidate_entity_id']).any():
            raise ValueError('Duplicate candidate pairs')
        for row in rows:
            entity = row['entity_id']
            if entity in truth:
                raise ValueError('Reference occurs in multiple chunks')
            truth[entity] = row['true_ids']
        for entity, group in frame.groupby('source1_entity_id', sort=False):
            actual = group.candidate_entity_id.isin(truth[entity]).astype(int)
            if not (actual.to_numpy() == group.label.to_numpy()).all():
                raise ValueError('Cached labels disagree with full truth')
        if training:
            positive = frame[frame.label == 1]
            negative = frame[frame.label == 0]
            hard = negative.sort_values(['blocking_score', 'candidate_entity_id'],
                                        ascending=[False, True]).groupby('source1_entity_id').head(4)
            remainder = negative.drop(index=hard.index).copy()
            remainder['sample_order'] = [priority(a + ':' + b).hex() for a, b in zip(
                remainder.source1_entity_id, remainder.candidate_entity_id)]
            easy = remainder.sort_values('sample_order').groupby('source1_entity_id').head(2)
            frame = pd.concat([positive, hard, easy[columns]], ignore_index=True)
        frames.append(frame)
    if set(truth) != wanted:
        raise ValueError(f'Missing {len(wanted-set(truth))} reference records')
    return pd.concat(frames, ignore_index=True), truth, hashes


def serialize(record):
    # Keep both names and numbers before the longer address field.
    return ('name: ' + record['business_name'] + ' country: ' + record['country']
            + ' address: ' + record['business_address'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--train-references', type=int, default=20000)
    parser.add_argument('--validation-references', type=int, default=5000)
    args = parser.parse_args()
    os.nice(10)
    pa.set_cpu_count(1)
    args.output.mkdir(parents=True, exist_ok=False)
    reservation = Path('models/scale_v1_plan/sampled_references.json')
    data = json.loads(reservation.read_text())
    train = sorted(data['train'], key=lambda r: priority(r['entity_id']))[:args.train_references]
    # Reuse the already designated early-stopping pool; no new audit is consumed.
    early_pool = sorted(data['tune'], key=lambda r: hashlib.sha256(
        ('context-split-v1:' + r['entity_id']).encode()).digest())[20000:30000]
    valid = sorted(early_pool, key=lambda r: priority(r['entity_id']))[:args.validation_references]
    if len(train) != args.train_references or len(valid) != args.validation_references:
        raise ValueError('Requested more references than available')
    groups = {'train': train, 'validation': valid}
    frames, truths, provenance = {}, {}, {}
    for split, records in groups.items():
        frame, truth, hashes = collect(Path('models/scale_v1_run/cache_' + (
            'train' if split == 'train' else 'tune')), {r['entity_id'] for r in records}, split == 'train')
        frames[split], truths[split], provenance[split] = frame, truth, hashes
        print(json.dumps({'stage': 'pairs_collected', 'split': split, 'pairs': len(frame),
                          'references': len(truth)}), flush=True)
    if set(truths['train']) & set(truths['validation']):
        raise ValueError('Entity split leakage')
    valid_owned = {target for values in truths['validation'].values() for target in values}
    if set(frames['train'].candidate_entity_id) & valid_owned:
        raise ValueError('Validation-owned target leaked into training')
    wanted = set().union(*(set(f.candidate_entity_id) for f in frames.values()))
    targets = {}
    for source in (2, 3):
        path = Path(f'dataset/train/train_source{source}.tsv')
        for chunk in pd.read_csv(path, sep='\t', dtype=str, keep_default_na=False, chunksize=100000):
            for row in chunk[chunk.entity_id.isin(wanted)].to_dict('records'):
                if row['entity_id'] in targets:
                    raise ValueError('Duplicate target ID')
                targets[row['entity_id']] = serialize(row)
    if wanted != set(targets):
        raise ValueError('Missing target records')
    summaries = {}
    for split, records in groups.items():
        references = {r['entity_id']: serialize(r) for r in records}
        frame = frames[split]
        frame['text_left'] = frame.source1_entity_id.map(references)
        frame['text_right'] = frame.candidate_entity_id.map(targets)
        frame['country'] = frame.source1_entity_id.map({r['entity_id']: r['country'] for r in records})
        path = args.output / f'{split}.parquet'
        frame.to_parquet(path, index=False)
        (args.output / f'{split}_truth.json').write_text(json.dumps(truths[split]) + '\n')
        summaries[split] = {'references': len(records), 'pairs': len(frame),
                            'positive_pairs': int(frame.label.sum()), 'sha256': digest(path)}
    report = {'status': 'prepared_not_trained', 'summaries': summaries,
              'negative_sampling': 'four highest blocking-score negatives plus two deterministic random remaining negatives per training entity',
              'validation': 'all cached candidates for 5000 references from the existing early-stopping pool; not a fresh audit',
              'retrieval': 'scale_v1_run existing pre-bridge cache; this pilot is not a comparison to the final bridge pipeline',
              'raw_text': True, 'test_or_audit_data_read': False,
              'reservation_sha256': digest(reservation), 'cache_hashes': provenance}
    (args.output / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'stage': 'complete', 'summaries': summaries}), flush=True)


if __name__ == '__main__':
    main()
