"""Label-free exact-key, role, head-exclusion and original-feature pilot gates."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'research/sprint_6h/sibling')]
import train_reverse_adapter as adapter
from reverse_competition import KEYS,pair_order_hash
from export_sprint_workbench import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,default=Path('research/final_2h/empty_rescue_v1'))
    a=p.parse_args();d=a.directory
    marker=json.loads((d/'manifest.json').read_text())
    protocol=json.loads((d/'protocol.json').read_text())
    if marker['status']!='complete' or sha(d/'protocol.json')!=marker['protocol_sha256']:
        raise ValueError('Pilot preparation incomplete')
    if marker['baseline_selection_macro_f05']!=.9802609267254342:
        raise ValueError('Baseline differs')
    heads={}
    for name,path in [('old','models/sprint_6h/neural/frozen_head120k_v1/manifest.json'),('fine','models/final_2h/neural_last4_continue_v1/manifest.json')]:
        if sha(path)!=protocol['heads'][name]:raise ValueError('Head seal differs')
        heads[name]=set(json.loads(Path(path).read_text())['training_owners'])
    previous_raw=json.loads(Path('models/sprint_6h/neural/neural_inputs/residual20k/manifest.json').read_text())
    b=Path('research/sprint_6h/sibling')
    old={name:pd.read_parquet(b/directory/'features.parquet').set_index(KEYS) for name,directory in [('fit','neural_v1_features_fit'),('dev','neural_v1_features30k')]}
    result={}
    for role in ['residual_train','early_stop','selection']:
        directory=d/role;fm=json.loads((directory/'manifest.json').read_text())
        im=json.loads((directory/'neural_inputs/manifest.json').read_text())
        f=pd.read_parquet(directory/'features.parquet')
        if fm['role']!=role or im['role']!=role or fm['features']!=adapter.FEATURES:
            raise ValueError('Role/schema differs')
        if any(x['labels_read'] is not False for x in [fm,im]):raise ValueError('Label-bearing inputs')
        if im['source_tsv_sha256']!=previous_raw['source_tsv_sha256']:
            raise ValueError('Provided training source hashes differ from sealed original raw input')
        if any(set(f.source1_entity_id)&owners for owners in heads.values()):raise ValueError('Head-fit owner entered pilot')
        role_ids=json.loads((b/f'workbenches50k/{role}/reference_ids.json').read_text())
        if not set(f.source1_entity_id)<=set(role_ids) or sha(b/f'workbenches50k/{role}/reference_ids.json')!=fm['reference_ids_sha256']:
            raise ValueError('Pilot crossed role boundary')
        for path,key,source in [(directory/'features.parquet','features_sha256',fm),(directory/'neural_inputs/pairs.jsonl','input_sha256',im)]:
            if sha(path)!=source[key]:raise ValueError('Artifact bytes differ')
        if im['feature_manifest_sha256']!=sha(directory/'manifest.json') or pair_order_hash(f)!=fm['pair_order_sha256']:
            raise ValueError('Feature/input lineage differs')
        records=[json.loads(line) for line in (directory/'neural_inputs/pairs.jsonl').open()]
        if len(records)!=len(f) or [(r[KEYS[0]],r[KEYS[1]]) for r in records]!=list(f[KEYS].itertuples(index=False,name=None)):
            raise ValueError('Raw key order differs')
        if any(set(r)!={*KEYS,'text_left','text_right'} or not all(isinstance(r[k],str) for k in r) for r in records):
            raise ValueError('Unexpected/label-bearing serialization')
        prior=old['fit' if role=='residual_train' else 'dev']
        overlap=f.set_index(KEYS).index.intersection(prior.index)
        if len(overlap) and not np.array_equal(f.set_index(KEYS).loc[overlap,adapter.FEATURES].to_numpy(),prior.loc[overlap,adapter.FEATURES].to_numpy()):
            raise ValueError('Direct rebuild differs from exact original36 overlap features')
        result[role]={'rows':len(f),'empty_owners':f.source1_entity_id.nunique(),
            'overlapping_existing38_keys':len(overlap),'new_extra_keys':len(f)-len(overlap),
            'exact36_overlap_parity_passed':True,'neural_head_training_owners_excluded':True,
            'feature_manifest_sha256':sha(directory/'manifest.json'),
            'input_manifest_sha256':sha(directory/'neural_inputs/manifest.json')}
    report={'status':'passed','roles':result,'all_source_hashes_match_original_supplied_train':True,
        'original_route_corrections_must_be_preserved':True,'apply_only_new_extra_keys':True,
        'audit_labels_read':False,'extension_labels_read':False,'pilot_selection_results_read':False,
        'source_sha256':sha(__file__),'preparation_manifest_sha256':sha(d/'manifest.json')}
    (d/'validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
