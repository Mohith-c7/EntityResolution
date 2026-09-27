"""Execute the single authorized 36-feature fit and frozen blend grid on VM2."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
os.chdir(ROOT)
sys.path.insert(0, str(Path(__file__).parent))
from build_role_workbenches import read_owner_truth
import train_sibling_matcher as sibling

BASE = Path('research/sprint_6h/sibling')
THREADS = 16
TRAIN = Path('models/sprint_6h/training50k_oof')
train_manifest = json.loads((TRAIN/'manifest.json').read_text())
if not train_manifest['verified_current_pipeline'] or not train_manifest['first_stage_owner_excluded_oof']:
    raise ValueError('Coordinator must seal current training OOF capture')
truth_path = BASE/'training50k_truth.json'
if truth_path.exists():
    raise FileExistsError(truth_path)
truth = read_owner_truth(train_manifest['reference_ids'], Path('models/scale_v1_plan/aliases/counts.sqlite'))
sibling.write_json(truth_path, truth)

def run(command, *args):
    started = time.time()
    print('START', command, ' '.join(map(str,args)), flush=True)
    subprocess.run([sys.executable, 'scripts/train_sibling_matcher.py', command, '--threads', str(THREADS), *map(str,args)], check=True)
    print('DONE', command, round(time.time()-started, 3), flush=True)

run('prepare', '--pairs',TRAIN/'pairs.parquet','--manifest',TRAIN/'manifest.json','--truth',truth_path,
    '--owner-db','models/scale_v1_plan/aliases/counts.sqlite','--index-prefix','models/index_train','--output',BASE/'prepared')
print('PREPARED',json.dumps({k:v for k,v in json.loads((BASE/'prepared/manifest.json').read_text()).items() if k in ['examples','positives','negatives','compared_targets','compared_targets_with_known_train_owner','seconds']}),flush=True)
run('fit','--examples',BASE/'prepared/examples.parquet','--trees',250,'--output',BASE/'compatibility')
role = BASE/'workbenches50k/residual_train'
run('support','--pairs',role/'pairs.parquet','--manifest',role/'manifest.json','--index-prefix','models/index_train',
    '--model-file',BASE/'compatibility/model.txt','--output',BASE/'residual_support')
run('fit-residual','--pairs',role/'pairs.parquet','--manifest',role/'manifest.json','--truth',role/'truth.json',
    '--support',BASE/'residual_support/support.parquet','--support-manifest',BASE/'residual_support/manifest.json',
    '--trees',100,'--output',BASE/'residual')
role = BASE/'learned30k'
run('support','--pairs',role/'pairs.parquet','--manifest',role/'manifest.json','--index-prefix','models/index_train',
    '--route-plan',BASE/'learned30k_route','--model-file',BASE/'compatibility/model.txt','--output',BASE/'learned30k_support')
for weight in sibling.LEARNED_WEIGHTS:
    run('rescore','--pairs',role/'pairs.parquet','--manifest',role/'manifest.json',
        '--support',BASE/'learned30k_support/support.parquet','--support-manifest',BASE/'learned30k_support/manifest.json',
        '--route-plan',BASE/'learned30k_route','--model-dir',BASE/'residual','--weight',weight,
        '--output',BASE/('learned_weight_'+str(weight).replace('.','p')))
print('COMPLETE frozen learned grid',flush=True)
