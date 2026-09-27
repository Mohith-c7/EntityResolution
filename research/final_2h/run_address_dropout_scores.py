"""Wait for complete owned dropout fit, then score two sealed raw S1 inputs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--training-pid',type=int,required=True);a=p.parse_args()
    model=Path('models/final_2h/neural_address_dropout8_v1');deadline=time.monotonic()+16*60
    while True:
        try:os.kill(a.training_pid,0)
        except ProcessLookupError:break
        if time.monotonic()>deadline:raise TimeoutError('Training barrier exceeded; no scoring started')
        time.sleep(3)
    marker=json.loads((model/'manifest.json').read_text())
    if marker.get('status')!='complete' or marker.get('completed_epochs')!=1 or marker.get('processed_training_rows')!=120000 or marker.get('trainable_encoder_layers')!=8:raise ValueError('Incomplete model cannot be scored')
    if marker['code_sha256']!=sha('research/final_2h/neural_address_dropout.py') or marker['training_input_sha256']!=sha('models/final_2h/neural_address_dropout_training_v1/pairs.jsonl') or marker['training_manifest_sha256']!=sha('models/final_2h/neural_address_dropout_training_v1/manifest.json'):raise ValueError('Training/source lineage changed')
    if any(marker.get(k) is not False for k in ('audit_labels_read','development_labels_read','test_records_read')):raise ValueError('Training scope differs')
    for name,digest in marker['checkpoint_files'].items():
        if sha(model/'checkpoint'/name)!=digest:raise ValueError('Checkpoint changed')
    for role,expected in [('residual20k',7389),('learned30k',11144)]:
        inputs=Path('models/sprint_6h/neural/neural_inputs')/role
        output=Path('models/sprint_6h/neural')/('address_dropout_'+role)
        im=json.loads((inputs/'manifest.json').read_text())
        if im['rows']!=expected or im['input_sha256']!=sha(inputs/'pairs.jsonl') or im.get('labels_read') is not False:raise ValueError('Wrong sealed scoring input')
        subprocess.run([sys.executable,'-u','research/final_2h/neural_layers.py','score','--model',str(model/'checkpoint'),'--input',str(inputs/'pairs.jsonl'),'--input-manifest',str(inputs/'manifest.json'),'--output',str(output)],check=True)
        score=json.loads((output/'manifest.json').read_text())
        if score['rows']!=expected or score['checkpoint_manifest_sha256']!=sha(model/'manifest.json') or score['input_sha256']!=im['input_sha256'] or score['code_sha256']!=sha('research/final_2h/neural_layers.py') or sha(output/'scores.jsonl')!=score['scores_sha256']:raise ValueError('Score provenance changed')
        inputs_keys=[(r['source1_entity_id'],r['candidate_entity_id']) for r in map(json.loads,(inputs/'pairs.jsonl').open())]
        scores_keys=[(r['source1_entity_id'],r['candidate_entity_id']) for r in map(json.loads,(output/'scores.jsonl').open())]
        if inputs_keys!=scores_keys or len(set(scores_keys))!=expected:raise ValueError('Score keys/order incomplete')
        print(json.dumps({'stage':'score_verified','role':role,'output':str(output),'rows':expected}),flush=True)

if __name__=='__main__':main()
