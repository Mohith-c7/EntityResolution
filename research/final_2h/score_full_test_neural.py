"""Run the frozen scorers in batch-aligned, sealed inference partitions.

Only execution is partitioned. Model weights, tokenizer, precision, batch
sizes, pair order and scorer source remain the audited implementation.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'research/sprint_6h'))
from prepare_neural_inputs import sha


def write(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def verify_part(directory, head, head_key, code, input_path, manifest_path, rows):
    """A scorer can return zero after its time limit without completing."""
    path = directory / 'manifest.json'
    if not path.is_file():
        raise ValueError('Neural scorer returned without a completed partition: ' + str(directory))
    marker = json.loads(path.read_text())
    if (marker.get('status') != 'complete' or marker.get('labels_read') is not False
            or marker.get('rows') != rows or marker.get(head_key) != sha(head / 'manifest.json')
            or marker.get('code_sha256') != sha(code)
            or marker.get('scores_sha256') != sha(directory / 'scores.jsonl')
            or marker.get('input_sha256') != sha(input_path)
            or ('input_manifest_sha256' in marker
                and marker['input_manifest_sha256'] != sha(manifest_path))):
        raise ValueError('Neural partition completion/provenance differs')
    return marker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--parts', type=Path, required=True)
    parser.add_argument('--old-output', type=Path, required=True)
    parser.add_argument('--fine-output', type=Path, required=True)
    parser.add_argument('--part-rows', type=int, default=393216)
    args = parser.parse_args(); started = time.monotonic()
    if args.part_rows <= 0 or args.part_rows % 512:
        raise ValueError('Partitions must preserve both 32- and 512-row batch boundaries')
    if any(p.exists() for p in [args.parts, args.old_output, args.fine_output]):
        raise ValueError('Require unused inference outputs')
    full = json.loads((args.inputs / 'manifest.json').read_text())
    raw = args.inputs / 'pairs.jsonl'
    if (full.get('status') != 'complete' or full.get('labels_read') is not False
            or full.get('source_split') != 'test' or sha(raw) != full['input_sha256']):
        raise ValueError('Requires sealed official test inputs without labels')
    old_head = ROOT / 'models/sprint_6h/neural/frozen_head120k_v1'
    fine_head = ROOT / 'models/final_2h/neural_last4_continue_v1'
    old_code = ROOT / 'research/sprint_6h/neural_frozen_head.py'
    fine_code = ROOT / 'research/final_2h/neural_layers.py'
    args.parts.mkdir(parents=True)
    entries = []; total = 0

    def score(number, lines):
        directory = args.parts / str(number); directory.mkdir()
        input_path = directory / 'pairs.jsonl'; input_path.write_text(''.join(lines))
        manifest_path = directory / 'manifest.json'
        write(manifest_path, {**full, 'rows': len(lines), 'input_sha256': sha(input_path),
            'parent_input_manifest_sha256': sha(args.inputs / 'manifest.json'),
            'parent_input_sha256': full['input_sha256'],
            'execution_subset': True, 'batch_boundary': 512})
        subprocess.run([sys.executable, str(old_code), 'score',
            '--input', str(input_path), '--manifest', str(manifest_path),
            '--base', str(ROOT / 'models/sprint_6h/neural/minilm_base'),
            '--head', str(old_head), '--output', str(directory / 'old')],
            cwd=ROOT, check=True)
        verify_part(directory / 'old', old_head, 'head_manifest_sha256', old_code,
                    input_path, manifest_path, len(lines))
        subprocess.run([sys.executable, str(fine_code), 'score',
            '--input', str(input_path), '--input-manifest', str(manifest_path),
            '--model', str(fine_head / 'checkpoint'), '--max-minutes', '30',
            '--output', str(directory / 'fine')], cwd=ROOT, check=True)
        verify_part(directory / 'fine', fine_head, 'checkpoint_manifest_sha256',
                    fine_code, input_path, manifest_path, len(lines))
        entries.append(directory)
        print(json.dumps({'stage': 'both_heads_part_complete', 'part': number,
                          'rows': len(lines), 'seconds': time.monotonic()-started}), flush=True)

    lines = []
    with raw.open() as stream:
        for line in stream:
            lines.append(line); total += 1
            if len(lines) == args.part_rows:
                score(len(entries), lines); lines = []
    if lines:
        score(len(entries), lines)
    if total != full['rows']:
        raise ValueError('Incomplete input coverage')
    for name, target, head, head_key, code in [
        ('old', args.old_output, old_head, 'head_manifest_sha256', old_code),
        ('fine', args.fine_output, fine_head, 'checkpoint_manifest_sha256', fine_code)]:
        target.mkdir(parents=True)
        markers = []; rows = 0
        with (target / 'scores.jsonl').open('x') as out:
            for part in entries:
                path = part / name
                marker = json.loads((path / 'manifest.json').read_text())
                if (marker.get('status') != 'complete' or marker.get('labels_read') is not False
                        or marker[head_key] != sha(head / 'manifest.json')
                        or marker.get('code_sha256') != sha(code)
                        or marker['scores_sha256'] != sha(path / 'scores.jsonl')
                        or marker['input_sha256'] != sha(part / 'pairs.jsonl')):
                    raise ValueError('Neural partition seal or source differs')
                with (path / 'scores.jsonl').open() as inp:
                    for line in inp:
                        out.write(line); rows += 1
                markers.append(sha(path / 'manifest.json'))
        if rows != total:
            raise ValueError('Score count differs from full input')
        write(target / 'manifest.json', {'status': 'complete', 'rows': total,
            'labels_read': False, head_key: sha(head / 'manifest.json'),
            'scores_sha256': sha(target / 'scores.jsonl'),
            'input_sha256': full['input_sha256'],
            'input_manifest_sha256': sha(args.inputs / 'manifest.json'),
            'code_sha256': sha(code), 'execution_driver_sha256': sha(__file__),
            'partition_manifest_sha256': markers, 'batch_boundary': 512,
            'seconds': time.monotonic()-started})
    print(json.dumps({'stage': 'complete', 'rows': total,
                      'seconds': time.monotonic()-started}), flush=True)


if __name__ == '__main__':
    main()
