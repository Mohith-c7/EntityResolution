"""Describe routed fitting-role evidence; no model or selection changes."""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pyarrow as pa
import lightgbm as lgb
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'scripts'))
import train_sibling_matcher as sibling
B=ROOT/'research/sprint_6h/sibling';R=B/'workbenches50k/residual_train'
pa.set_cpu_count(1)
f,m=sibling.load_pairs(R/'pairs.parquet',R/'manifest.json')
full,_=sibling.aligned_support(f,B/'residual_support/support.parquet',B/'residual_support/manifest.json')
t=json.loads((R/'truth.json').read_text())
full=full[full.sibling_count>0].copy()
full['label']=[int(c in t[s]) for s,c in full[sibling.KEYS].itertuples(index=False,name=None)]
model=lgb.Booster(model_file=str(B/'residual/residual.txt'))
full['residual_p']=model.predict(full[sibling.RESIDUAL_NAMES].to_numpy(dtype=np.float32),num_threads=1)
full['p_bin']=pd.cut(full.probability,[-1,.32,.66,.83,1.01],labels=['<.32','.32-.66','.66-.83','>=.83'],right=False)
result=[]
for (label,bin),g in full.groupby(['label','p_bin'],observed=True):
 result.append({'label':int(label),'p2_bin':str(bin),'count':len(g),
 'mean_p1':float(g.first_stage.mean()),'mean_p2':float(g.probability.mean()),'mean_residual':float(g.residual_p.mean()),
 'mean_TT_max':float(g.sibling_max.mean()),'mean_TT_min':float(g.sibling_min.mean()),
 'TT_max_ge_90':int((g.sibling_max>=.9).sum()),'TT_min_ge_90':int((g.sibling_min>=.9).sum()),
 'residual_ge_83':int((g.residual_p>=.83).sum())})
sibling.write_json(B/'residual_training_diagnostic.json',{'scope':'In-sample residual-training role diagnostic; not an achieved held-out score. No selection or fresh audit labels consumed.','routed_pairs':len(full),'buckets':result})
print(json.dumps(result),flush=True)
