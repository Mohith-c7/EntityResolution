"""Score sealed heldout/test rescue inputs with the two frozen four-layer heads.

Orchestration only: both underlying scorers and exact-key manifest verification
are reused unchanged. No labels, fitting, thresholds or submission outputs.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'research/final_2h'))
from score_full_test_neural import sha, verify_part


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory', type=Path, required=True)
    p.add_argument('--input-sha256', required=True)
    a = p.parse_args()
    inputs = a.directory/'neural_inputs'
    raw, marker = inputs/'pairs.jsonl', inputs/'manifest.json'
    m = json.loads(marker.read_text())
    if m.get('status')!='complete' or m.get('labels_read') is not False or sha(raw)!=a.input_sha256 or m['input_sha256']!=a.input_sha256:
        raise ValueError('Requires exact sealed label-free input')
    old_head = ROOT/'models/sprint_6h/neural/frozen_head120k_v1'
    fine_head = ROOT/'models/final_2h/neural_last4_continue_v1'
    if sha(old_head/'manifest.json')!='f2b75c8a1ca5bcb7293b893c61791b619c02d5d7f38e75c7da197423b87dfd94' or sha(fine_head/'manifest.json')!='60bca75bf41628f43eff487f8bdbb9b2f2f5ad54967af467807d7bc55417083c':
        raise ValueError('Frozen rescue head lineage changed')
    old_code = ROOT/'research/sprint_6h/neural_frozen_head.py'
    fine_code = ROOT/'research/final_2h/neural_layers.py'
    if sha(old_code)!='a619ed6d1a9c7859e6d39001c9145dcb5ee6ff5cafea7510202a635eb2eb5f9d' or sha(fine_code)!='34ecb591d1bdc0035b927ab65cc6ddcb9326ff9651d57046ea7ee98f3106a373':
        raise ValueError('Frozen neural scoring implementation changed')
    started = time.monotonic()
    subprocess.run([sys.executable, str(old_code), 'score', '--input', str(raw), '--manifest', str(marker),
        '--base', str(ROOT/'models/sprint_6h/neural/minilm_base'), '--head', str(old_head),
        '--output', str(a.directory/'neural_old')], cwd=ROOT, check=True)
    verify_part(a.directory/'neural_old', old_head, 'head_manifest_sha256', old_code, raw, marker, m['rows'])
    subprocess.run([sys.executable, str(fine_code), 'score', '--input', str(raw), '--input-manifest', str(marker),
        '--model', str(fine_head/'checkpoint'), '--output', str(a.directory/'neural_fine')], cwd=ROOT, check=True)
    verify_part(a.directory/'neural_fine', fine_head, 'checkpoint_manifest_sha256', fine_code, raw, marker, m['rows'])
    print(json.dumps({'status':'both_frozen_heads_verified', 'rows':m['rows'],
        'directory':str(a.directory), 'seconds':time.monotonic()-started}), flush=True)


if __name__=='__main__': main()
