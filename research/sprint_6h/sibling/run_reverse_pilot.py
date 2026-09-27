"""Launch exactly one authorized reverse pilot once sealed keyed features arrive."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[3];os.chdir(ROOT)
p=argparse.ArgumentParser();p.add_argument('--fit-reverse',type=Path,required=True);p.add_argument('--learned-reverse',type=Path,required=True);a=p.parse_args()
B=Path('research/sprint_6h/sibling');SCRIPT=B/'train_reverse_adapter.py'
def run(command,*args):
 print('START',command,flush=True)
 subprocess.run([sys.executable,str(SCRIPT),command,'--threads','16',*map(str,args)],check=True)
 print('DONE',command,flush=True)
R=B/'workbenches50k/residual_train';L=B/'learned30k'
run('join','--pairs',R/'pairs.parquet','--manifest',R/'manifest.json','--reverse-features',a.fit_reverse/'features.parquet','--reverse-manifest',a.fit_reverse/'manifest.json','--output',B/'reverse_v1_features_fit')
run('join','--pairs',L/'pairs.parquet','--manifest',L/'manifest.json','--route-plan',B/'learned30k_route','--reverse-features',a.learned_reverse/'features.parquet','--reverse-manifest',a.learned_reverse/'manifest.json','--output',B/'reverse_v1_features30k')
run('fit','--feature-dir',B/'reverse_v1_features_fit','--early-feature-dir',B/'reverse_v1_features30k','--train-truth',R/'truth.json','--early-truth',B/'workbenches50k/early_stop/truth.json','--output',B/'reverse_v1')
run('rescore','--pairs',L/'pairs.parquet','--manifest',L/'manifest.json','--feature-dir',B/'reverse_v1_features30k','--model-dir',B/'reverse_v1','--output',B/'reverse_v1_scored30k')
run('early-evaluate','--pairs',L/'pairs.parquet','--manifest',L/'manifest.json','--feature-dir',B/'reverse_v1_features30k','--model-dir',B/'reverse_v1','--early-truth',B/'workbenches50k/early_stop/truth.json','--decision-config','research/sprint_6h/coordination/baseline_freeze/frozen.json','--output',B/'reverse_v1_early_report')
