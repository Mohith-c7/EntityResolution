"""Run the predeclared head-calibration pilot, early10k labels only."""
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[2]
S=Path('research/sprint_6h/neural_adapter.py')
B=Path('research/sprint_6h/sibling')
N=Path('models/sprint_6h/neural')


def run(command,*args):
    print('START',command,flush=True)
    subprocess.run([sys.executable,str(S),command,*map(str,args)],check=True)
    print('DONE',command,flush=True)


for role,features,out in [('residual20k','reverse_v1_features_fit','neural_v1_features_fit'),('learned30k','reverse_v1_features30k','neural_v1_features30k')]:
    run('join','--features',B/features,'--neural',N/role,'--head-manifest',N/'frozen_head120k_v1/manifest.json','--output',B/out)
run('fit','--features',B/'neural_v1_features_fit','--early-features',B/'neural_v1_features30k','--train-truth',B/'workbenches50k/residual_train/truth.json',
    '--early-truth',B/'workbenches50k/early_stop/truth.json','--output',B/'neural_v1')
run('rescore','--pairs',B/'learned30k/pairs.parquet','--manifest',B/'learned30k/manifest.json',
    '--features',B/'neural_v1_features30k','--model',B/'neural_v1','--output',B/'neural_v1_scored30k')
subprocess.run([sys.executable,'research/sprint_6h/validation/evaluate_adapter_pilot.py','--family','neural',
    '--source',str(B/'learned30k'),'--candidate',str(B/'neural_v1_scored30k'),'--model',str(B/'neural_v1'),
    '--features',str(B/'neural_v1_features30k'),'--early-truth',str(B/'workbenches50k/early_stop/truth.json'),
    '--early-countries',str(B/'workbenches50k/early_stop/countries.json'),'--frozen','research/sprint_6h/coordination/baseline_freeze/frozen.json',
    '--output','research/sprint_6h/validation/neural_v1_early10k_independent'],check=True)
