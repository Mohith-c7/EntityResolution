"""Serialize the sealed full-test route from the official TSVs, without labels."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'research/sprint_6h'))
import pandas as pd
from prepare_neural_inputs import serialize, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['features', 'data_dir', 'output']:
        parser.add_argument('--' + name.replace('_', '-'), type=Path, required=True)
    args = parser.parse_args(); started = time.monotonic()
    marker_path = args.features / 'manifest.json'
    marker = json.loads(marker_path.read_text())
    path = args.features / 'features.parquet'
    if (marker.get('status') != 'complete' or marker.get('split') != 'test'
            or marker.get('labels_read') is not False
            or sha(path) != marker.get('features_sha256')):
        raise ValueError('Requires sealed, label-free official-test features')
    keys = ['source1_entity_id', 'candidate_entity_id']
    frame = pd.read_parquet(path, columns=keys)
    if frame[keys].isna().any().any() or frame.duplicated(keys).any():
        raise ValueError('Missing or duplicate input pair keys')
    owners = set(frame.source1_entity_id); targets = set(frame.candidate_entity_id)
    if (any(not x.startswith('S1-') for x in owners)
            or any(not x.startswith(('S2-', 'S3-')) for x in targets)):
        raise ValueError('Invalid source prefixes')
    records = {}; hashes = {}
    for source in [1, 2, 3]:
        raw = args.data_dir / f'test_source{source}.tsv'
        hashes[f'S{source}'] = sha(raw)
        wanted = owners if source == 1 else targets
        for batch in pd.read_csv(raw, sep='\t', dtype=str, keep_default_na=False,
                                 chunksize=100000):
            for row in batch[batch.entity_id.isin(wanted)].to_dict('records'):
                if row['entity_id'] in records:
                    raise ValueError('Duplicate official record ID')
                records[row['entity_id']] = serialize(row)
    if set(records) != owners | targets:
        raise ValueError('Missing official routed record')
    args.output.mkdir(parents=True, exist_ok=False)
    out = args.output / 'pairs.jsonl'
    with out.open('x') as stream:
        for owner, target in frame.itertuples(index=False, name=None):
            stream.write(json.dumps({keys[0]: owner, keys[1]: target,
                'text_left': records[owner], 'text_right': records[target]},
                ensure_ascii=False) + '\n')
    result = {'status': 'complete', 'rows': len(frame), 'labels_read': False,
        'input_sha256': sha(out), 'feature_manifest_sha256': sha(marker_path),
        'feature_file_sha256': sha(path), 'source_tsv_sha256': hashes,
        'code_sha256': sha(__file__), 'role': marker.get('role'),
        'source_split': 'test', 'seconds': time.monotonic()-started}
    (args.output / 'manifest.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
