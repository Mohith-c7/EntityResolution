"""One authorized anchored adapter pilot, early10k only; selected20k not evaluated."""
from pathlib import Path
import os
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[3];os.chdir(ROOT)
B=Path('research/sprint_6h/sibling');S=B/'train_anchored_adapter.py'
def run(command,*args):
 print('START',command,flush=True)
 subprocess.run([sys.executable,str(S),command,'--threads','16',*map(str,args)],check=True)
 print('DONE',command,flush=True)
R=B/'workbenches50k/residual_train';L=B/'learned30k';Q=Path('models/sprint_6h/development50k/references.json')
run('build','--pairs',R/'pairs.parquet','--manifest',R/'manifest.json','--queries',Q,'--index-prefix','models/index_train','--output',B/'adapter_v1_features_fit')
run('build','--pairs',L/'pairs.parquet','--manifest',L/'manifest.json','--queries',Q,'--route-plan',B/'learned30k_route','--index-prefix','models/index_train','--output',B/'adapter_v1_features30k')
run('fit','--feature-dir',B/'adapter_v1_features_fit','--early-feature-dir',B/'adapter_v1_features30k','--train-truth',R/'truth.json','--early-truth',B/'workbenches50k/early_stop/truth.json','--output',B/'adapter_v1')
run('rescore','--pairs',L/'pairs.parquet','--manifest',L/'manifest.json','--feature-dir',B/'adapter_v1_features30k','--model-dir',B/'adapter_v1','--output',B/'adapter_v1_scored30k')
run('early-evaluate','--pairs',L/'pairs.parquet','--manifest',L/'manifest.json','--feature-dir',B/'adapter_v1_features30k','--model-dir',B/'adapter_v1','--early-truth',B/'workbenches50k/early_stop/truth.json','--decision-config','research/sprint_6h/coordination/baseline_freeze/frozen.json','--output',B/'adapter_v1_early_report')
