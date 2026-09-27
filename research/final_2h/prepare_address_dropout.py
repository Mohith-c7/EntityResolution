"""Deterministic target-only training augmentation with sealed original lineage."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

DOMAIN = 'target_address_dropout_balanced_v1'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load(path):
    return [json.loads(line) for line in Path(path).open()]

def masked(text):
    prefix, separator, address = text.rpartition(' address: ')
    if not separator:
        raise ValueError('Missing structural address suffix')
    return prefix + separator, bool(address.strip())

def selection(original, street, count, seed, original_sha, street_sha):
    if len(original) != len(street):
        raise ValueError('Parent row count changed')
    keys = [(r['source1_entity_id'], r['candidate_entity_id']) for r in street]
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate parent keys')
    eligible = []
    decoys = set()
    populations = Counter((r['text_left'], r['text_right'], r['label']) for r in street)
    for i, (a, b) in enumerate(zip(original, street)):
        if any(a[k] != b[k] for k in ('source1_entity_id', 'candidate_entity_id', 'label')):
            raise ValueError('Parent keys/order/labels changed')
        if a != b:
            decoys.add(i)
            continue
        right, nonempty = masked(b['text_right'])
        if nonempty:
            payload = json.dumps([DOMAIN, seed, original_sha, street_sha, *keys[i]], separators=(',', ':'))
            eligible.append((hashlib.sha256(payload.encode()).hexdigest(), keys[i], i, right))
    chosen = {}; counts = Counter(); collision_skips = 0
    for _, _, i, right in sorted(eligible):
        row = street[i]; label = row['label']
        if counts[label] >= count:
            continue
        if populations[(row['text_left'], right, 1-label)]:
            collision_skips += 1
            continue
        populations[(row['text_left'], row['text_right'], label)] -= 1
        populations[(row['text_left'], right, label)] += 1
        chosen[i] = right; counts[label] += 1
    if counts != {0: count, 1: count}:
        raise ValueError(f'Insufficient contradiction-free eligible rows: {dict(counts)}')
    return chosen, decoys, collision_skips

def derive(original, street, chosen):
    return [{**r, 'text_right': chosen[i]} if i in chosen else r for i, r in enumerate(street)]

def verify(input_path, marker):
    if marker.get('augmentation_kind') != DOMAIN or marker['builder_sha256'] != sha(__file__):
        raise ValueError('Wrong augmentation builder provenance')
    for name, descriptor in marker['parents'].items():
        if sha(descriptor['input']) != descriptor['input_sha256'] or sha(descriptor['manifest']) != descriptor['manifest_sha256']:
            raise ValueError('Parent input or manifest changed')
        parent = json.loads(Path(descriptor['manifest']).read_text())
        if parent['input_sha256'] != descriptor['input_sha256']:
            raise ValueError('Parent seal mismatch')
    a = marker['parents']['original']; b = marker['parents']['street']
    original, street = load(a['input']), load(b['input'])
    chosen, decoys, skipped = selection(original, street, marker['masked_per_class'], marker['seed'], a['input_sha256'], b['input_sha256'])
    actual = load(input_path)
    if actual != derive(original, street, chosen) or sha(input_path) != marker['input_sha256']:
        raise ValueError('Derived rows are not the deterministic target-address-only transform')
    if len(decoys) != marker['excluded_modified_pairs'] or skipped != marker['contradiction_skips']:
        raise ValueError('Transform diagnostics changed')
    if hashlib.sha256(json.dumps(sorted(chosen), separators=(',', ':')).encode()).hexdigest() != marker['chosen_row_indices_sha256']:
        raise ValueError('Chosen row seal changed')
    return a, b

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--original', type=Path, required=True); p.add_argument('--original-manifest', type=Path, required=True)
    p.add_argument('--street', type=Path, required=True); p.add_argument('--street-manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--count-per-class', type=int, default=10000); p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    parents = {}
    for name in ('original', 'street'):
        path = getattr(args, name); manifest = getattr(args, name+'_manifest')
        parents[name] = {'input': str(path), 'manifest': str(manifest), 'input_sha256': sha(path), 'manifest_sha256': sha(manifest)}
    original, street = load(args.original), load(args.street)
    chosen, decoys, skipped = selection(original, street, args.count_per_class, args.seed, parents['original']['input_sha256'], parents['street']['input_sha256'])
    output = args.output/'pairs.jsonl'
    with output.open('x') as stream:
        for row in derive(original, street, chosen):
            stream.write(json.dumps(row, ensure_ascii=True)+'\n')
    marker = json.loads(args.street_manifest.read_text())
    marker.update({'input_sha256': sha(output), 'augmentation_kind': DOMAIN, 'augmentation': 'Deterministic balanced target-address suffix deletion only; all modified street-decoys retained unchanged; reject exact text contradictions', 'masked_per_class': args.count_per_class, 'seed': args.seed, 'parents': parents, 'builder_sha256': sha(__file__), 'excluded_modified_pairs': len(decoys), 'contradiction_skips': skipped, 'chosen_row_indices_sha256': hashlib.sha256(json.dumps(sorted(chosen), separators=(',', ':')).encode()).hexdigest(), 'source_labels_unchanged': True, 'keys_order_owners_unchanged': True})
    manifest = args.output/'manifest.json'; manifest.write_text(json.dumps(marker, indent=2, sort_keys=True)+'\n')
    verify(output, marker)
    print(json.dumps({'rows': len(street), 'masked_per_class': args.count_per_class, 'excluded_modified_pairs': len(decoys), 'contradiction_skips': skipped, 'input_sha256': marker['input_sha256'], 'manifest_sha256': sha(manifest)}))

if __name__ == '__main__':
    main()
