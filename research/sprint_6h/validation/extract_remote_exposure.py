"""Bounded remote exposure extraction: JSON manifests or projected pair IDs only."""
import argparse
import hashlib
import json
import re
from pathlib import Path
ID = re.compile(r"^S1-[A-Za-z0-9_.:-]+$")


def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""): h.update(chunk)
    return h.hexdigest()


def extract_json(v):
    if isinstance(v, str) and ID.fullmatch(v): yield v
    elif isinstance(v, dict):
        for k, item in v.items():
            if ID.fullmatch(str(k)): yield k
            yield from extract_json(item)
    elif isinstance(v, list):
        for item in v: yield from extract_json(item)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True); p.add_argument('--known', type=Path); p.add_argument('--path', action='append'); p.add_argument('--include-parquet', action='store_true'); args = p.parse_args()
    known = set(json.loads(args.known.read_text())) if args.known else set()
    paths = [args.root / v for v in args.path] if args.path else [f for root in ['models','reports','research'] for f in (args.root/root).rglob('*.json') if f.is_file() and '/aliases/' not in str(f) and f.name != 'name_aliases.json']
    if args.include_parquet and not args.path:
        paths += [f for root in ["models", "reports", "research"] for f in (args.root/root).rglob("*.parquet") if f.is_file()]
    ids, sources, accounted = set(), [], []
    for path in sorted(paths):
        digest = sha(path)
        if digest in known:
            accounted.append({'path':str(path),'sha256':digest,'evidence':'identical locally scanned artifact'});continue
        if path.suffix == '.parquet':
            import pyarrow as pa
            import pyarrow.parquet as pq
            pa.set_cpu_count(1)
            parquet = pq.ParquetFile(path)
            cols = [c for c in ['source1_entity_id','entity_id','reference_id','owner_id','owner'] if c in parquet.schema_arrow.names]
            if not cols: raise ValueError('Unaccounted scored parquet has no reference ID column: '+str(path))
            found = set()
            for batch in parquet.iter_batches(columns=cols, batch_size=65536, use_threads=False):
                for col in cols: found.update(v for v in batch.column(col).to_pylist() if isinstance(v,str) and ID.fullmatch(v))
        else: found = set(extract_json(json.loads(path.read_text())))
        ids.update(found);sources.append({'path':str(path),'sha256':digest,'reference_ids':len(found),'method':'projected_reference_columns' if path.suffix=='.parquet' else 'recursive_json_reference_ids'})
    print(json.dumps({'root':str(args.root),'ids':sorted(ids),'sources':sources,'accounted_by_local_hash':accounted},separators=(',',':')))
if __name__=='__main__':main()
