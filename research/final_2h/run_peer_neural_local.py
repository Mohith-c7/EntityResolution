"""Serialize MPS jobs after the direct neural experiment releases the device."""
import json
from pathlib import Path
import subprocess
import sys
import time
N=Path('models/sprint_6h/neural')
M=Path('models/final_2h')
started=time.monotonic()


def run(*args):
    print(json.dumps({'stage':'start','args':list(map(str,args)),'seconds':time.monotonic()-started}),flush=True)
    subprocess.run([sys.executable,'research/sprint_6h/neural_frozen_head.py',*map(str,args)],check=True)


while not all((N/name/'manifest.json').exists() for name in ['last2x3_residual20k','last2x3_learned30k']):
    if time.monotonic()-started>15*60:raise RuntimeError('Direct neural barrier exceeded15minutes')
    print(json.dumps({'stage':'waiting_for_direct_scores','seconds':time.monotonic()-started}),flush=True)
    time.sleep(30)
for name in ['last2x3_residual20k','last2x3_learned30k']:
    marker=json.loads((N/name/'manifest.json').read_text())
    if marker['status']!='complete' or marker['labels_read'] is not False:raise ValueError('GPU barrier is not completed inference')
training=M/'neural_peer_training_v1';head=M/'neural_peer_head_v1'
run('fit','--input',training/'pairs.jsonl','--manifest',training/'manifest.json',
    '--base',N/'minilm_base','--output',head)
for role in ['residual20k','learned30k']:
    inputs=M/'neural_peer_inputs'/role
    run('score','--input',inputs/'pairs.jsonl','--manifest',inputs/'manifest.json',
        '--base',N/'minilm_base','--head',head,'--output',M/'neural_peer_scores'/role)
for role in ['baseline','upstreamquota']:
    inputs=M/'retrieval_neural_inputs'/role
    if inputs.exists():
        run('score','--input',inputs/'pairs.jsonl','--manifest',inputs/'manifest.json',
            '--base',N/'minilm_base','--head',N/'frozen_head120k_v1','--output',M/'retrieval_neural_scores'/role)
print(json.dumps({'stage':'peer_neural_complete','seconds':time.monotonic()-started}),flush=True)
