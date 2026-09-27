"""Run the existing reverse feature builder on a single pinned global route.

Only execution is partitioned. Candidate groups, route selection, query code,
index, and feature definitions remain unchanged. No labels are read.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
import numpy as np
import pandas as pd
import pyarrow as pa
import reverse_competition as reverse
import train_sibling_matcher as sibling


def task(args):
    pa.set_cpu_count(1)
    pairs, manifest, index, prefix, output, route = args
    return reverse.generate_features(pairs, manifest, index, prefix, output,
                                     route_dir=route)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['pairs', 'manifest', 'index', 'target_index_prefix', 'output']:
        p.add_argument('--' + name.replace('_', '-'), type=Path, required=True)
    p.add_argument('--workers', type=int, default=16)
    a = p.parse_args()
    if not 1 <= a.workers <= 32 or a.output.exists():
        raise ValueError('Require bounded workers and unused output')
    pa.set_cpu_count(1)
    start = time.monotonic()
    frame, marker = sibling.load_pairs(a.pairs, a.manifest)
    ids = marker['reference_ids']
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate declared reference')
    counts = frame.groupby('source1_entity_id', sort=False).size()
    if list(counts.index) != ids or counts.nunique() != 1:
        raise ValueError('This execution wrapper requires contiguous equal-sized groups')
    count = int(counts.iloc[0])
    if not np.array_equal(frame.source1_entity_id.to_numpy().reshape(-1, count)[:, 0], ids):
        raise ValueError('Unexpected source group order')
    a.output.mkdir(parents=True)
    route_dir = a.output / 'global_route'; route_dir.mkdir()
    route = sibling.select_route(frame, universe_ids=ids)
    edges = sibling.route_edges(frame, route)
    edges.to_parquet(route_dir / 'route.parquet', index=False)
    sibling.write_json(route_dir / 'manifest.json', {
        'status': 'complete', 'route': asdict(sibling.RouteConfig()),
        'global_references': len(ids), 'global_pair_order_sha256': marker['pair_order_sha256'],
        'route_sha256': sibling.digest(route_dir / 'route.parquet'),
        'frozen_sha256': marker['frozen_sha256'], 'labels_read': False})
    jobs = []
    for number, positions in enumerate(np.array_split(np.arange(len(ids)), a.workers)):
        if not len(positions): continue
        part = a.output / 'parts' / str(number); part.mkdir(parents=True)
        sub = frame.iloc[int(positions[0])*count:(int(positions[-1])+1)*count].reset_index(drop=True)
        path = part / 'pairs.parquet'; sub.to_parquet(path, index=False)
        sibling.write_json(part / 'manifest.json', {
            **marker, 'reference_ids': [ids[int(i)] for i in positions],
            'entities': len(positions), 'pairs': len(sub), 'pairs_sha256': sibling.digest(path),
            'pair_order_sha256': sibling.pair_order_hash(sub),
            'parent_pairs_sha256': marker['pairs_sha256'],
            'parent_manifest_sha256': sibling.digest(a.manifest),
            'routing_universe_pair_order_sha256': marker['pair_order_sha256'],
            'labels_read': False})
        jobs.append((path, part / 'manifest.json', a.index, a.target_index_prefix,
                     part / 'features', route_dir))
    del frame, sub, route
    print(json.dumps({'stage': 'global_route_pinned', 'routed_pairs': edges[sibling.KEYS].drop_duplicates().shape[0],
                      'workers': len(jobs), 'seconds': time.monotonic()-start}), flush=True)
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        manifests = list(pool.map(task, jobs))
    data = pd.concat([pd.read_parquet(j[4] / 'features.parquet') for j in jobs], ignore_index=True)
    if data.duplicated(sibling.KEYS).any() or set(map(tuple, data[sibling.KEYS].to_numpy())) != set(map(tuple, edges[sibling.KEYS].drop_duplicates().to_numpy())):
        raise ValueError('Parallel result does not exactly cover the global route')
    data.to_parquet(a.output / 'features.parquet', index=False)
    with (a.output / 'diagnostics.jsonl').open('x') as stream:
        for j in jobs:
            with (j[4] / 'diagnostics.jsonl').open() as source:
                for line in source: stream.write(line)
    first = manifests[0]
    for m in manifests[1:]:
        for field in ['index_manifest_sha256', 'query_config', 'source_tsv_sha256',
                      'feature_code_sha256', 'normalization_sha256', 'phonetic_code_sha256']:
            if m[field] != first[field]: raise ValueError('Part provenance differs')
    sibling.write_json(a.output / 'manifest.json', {
        **first, 'features_sha256': sibling.digest(a.output / 'features.parquet'),
        'pair_order_sha256': sibling.pair_order_hash(data),
        'source_pairs_sha256': marker['pairs_sha256'],
        'source_pair_order_sha256': marker['pair_order_sha256'],
        'source_manifest_sha256': sibling.digest(a.manifest),
        'route_manifest': str(route_dir / 'manifest.json'),
        'diagnostics_sha256': sibling.digest(a.output / 'diagnostics.jsonl'),
        'rows': len(data), 'routed_references': data.source1_entity_id.nunique(),
        'unique_target_queries': sum(m['unique_target_queries'] for m in manifests),
        'parallel_wrapper_sha256': sibling.digest(__file__),
        'parallel_parts': [sibling.digest(j[4] / 'manifest.json') for j in jobs],
        'workers': a.workers, 'seconds': time.monotonic()-start})
    print(json.dumps({'stage': 'complete', 'rows': len(data), 'seconds': time.monotonic()-start}), flush=True)


if __name__ == '__main__': main()
