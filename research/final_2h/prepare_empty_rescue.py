"""Separate diagnostic empty-prediction route, with real reverse33/raw inputs.

No pilot labels/evaluation/training or production changes. The route differs
explicitly from the audited sibling route: every empty-prediction owner, frozen
p2 top4 candidates, deterministic target-ID ties, no seed or peer requirement.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sqlite3
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'research/sprint_6h'),str(ROOT/'research/sprint_6h/sibling'),str(ROOT/'research/final_2h')]
import reverse_competition as reverse
import train_reverse_adapter as adapter
import neural_adapter as nn
import neural_peer
from prepare_neural_inputs import serialize,sha
from evaluate_sprint import decide_control,predictions_for,entity_f05
B=Path('research/sprint_6h/sibling')
N=Path('models/sprint_6h/neural')
KEYS=reverse.KEYS


def select_empty_top4(frame,chosen,reference_ids):
    if len(frame)!=40*len(reference_ids) or set(frame.source1_entity_id)!=set(reference_ids):
        raise ValueError('Complete40-candidate owner graph required')
    if frame.groupby('source1_entity_id').size().ne(40).any() or frame.duplicated(KEYS).any():
        raise ValueError('Truncated/duplicate candidate groups')
    accepted=set(frame.loc[chosen,'source1_entity_id'])
    empty=set(reference_ids)-accepted
    ranked=frame[frame.source1_entity_id.isin(empty)].sort_values(
        ['source1_entity_id','probability','candidate_entity_id'],ascending=[True,False,True],kind='stable')
    return ranked.groupby('source1_entity_id',sort=False).head(4).sort_values(KEYS,kind='stable')


def write(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('research/final_2h/empty_rescue_v1'))
    p.add_argument('--max-seconds',type=int,default=850)
    a=p.parse_args();start=time.monotonic()
    def cap():
        if time.monotonic()-start>a.max_seconds:raise TimeoutError('Preparation cap; incomplete artifacts must not be used')
    if a.output.exists():raise FileExistsError(a.output)
    index_path=Path('research/sprint_6h/reverse_competition/train_s1.sqlite')
    data=Path('dataset/train')
    required=[index_path,Path(str(index_path)+'.complete.json'),*[Path(f'models/index_train_S{s}.sqlite') for s in [2,3]],*[data/f'train_source{s}.tsv' for s in [1,2,3]]]
    if any(not x.exists() for x in required):raise ValueError('Missing actual reverse index/target index/raw provided records')
    a.output.mkdir(parents=True)
    model_dir=Path('research/final_2h/neural_last4_fine_v1')
    model_report=json.loads((model_dir/'report.json').read_text())
    if sha(model_dir/'adapter.txt')!=model_report['model_sha256']:raise ValueError('Selected adapter hash changed')
    heads={'old':Path('models/sprint_6h/neural/frozen_head120k_v1/manifest.json'), 'fine':Path('models/final_2h/neural_last4_continue_v1/manifest.json')}
    protocol={'status':'predeclared_diagnostic_preparation_before_pilot_evaluation','declared_at_epoch':time.time(),
        'route':'All final selected-four empty-prediction owners; original40 top4 by frozenp2 descending, targetID ascending. No confident sibling seed or peer.',
        'original_route_changed':True,'production_route_changed':False,'selection_pilot_results_read':False,
        'allowed_fit':'Only residual20k; any learned rescue early-stop only early10k, predeclare before selection results',
        'first_candidate':'Existing selected four-layer correction from originalp2 anchor once, unchanged global decide, separate pilot only',
        'required_acceptance':'Exact baseline selection replay .9802609267254342, paired gain CI lower>0, precision must not fall, singleton false positives must not rise, no country delta<-.0005',
        'claimant_graphs':'Development30k shared by early10k/selection20k; residual20k separate diagnostic global graph, in-sample adapter predictions. No full331k parity claim.',
        'selected_adapter_sha256':sha(model_dir/'adapter.txt'),'selected_report_sha256':sha(model_dir/'report.json'),
        'heads':{k:sha(v) for k,v in heads.items()},'source_sha256':sha(__file__),'audit_labels_read':False,'extension_labels_read':False}
    write(a.output/'protocol.json',protocol)
    config=json.loads(Path('research/sprint_6h/coordination/baseline_freeze/frozen.json').read_text())
    config={k:config[k] for k in ['threshold','t_first','t_rest']}
    dev_base=pd.read_parquet(B/'learned30k/pairs.parquet')
    dev=pd.read_parquet(model_dir/'pairs.parquet')
    if not dev[KEYS].equals(dev_base[KEYS]) or not np.array_equal(dev.probability_original,dev_base.probability):raise ValueError('Development frozen anchors/order changed')
    dev_chosen,_=decide_control(dev,dev,config)
    roles={r:json.loads((B/f'workbenches50k/{r}/reference_ids.json').read_text()) for r in ['residual_train','early_stop','selection']}
    if [len(roles[r]) for r in roles]!=[20000,10000,20000] or len(set().union(*map(set,roles.values())))!=50000:raise ValueError('Authorized roles differ')
    # Baseline-only verification uses already consumed selection labels; no
    # rescue probabilities or pilot selection results are evaluated here.
    truth=json.loads((B/'workbenches50k/selection/truth.json').read_text())
    mask=dev.source1_entity_id.isin(truth).to_numpy()
    pred=predictions_for(dev[mask],dev_chosen[mask],truth)
    baseline=float(np.mean([entity_f05(truth[e],pred[e]) for e in sorted(truth)]))
    if baseline!=.9802609267254342:raise ValueError('Exact selected selection replay differs')
    train_path=Path('research/sprint_6h/reverse_competition/inputs/residual_train/pairs.parquet')
    if not train_path.exists():
        train_path=B/'workbenches50k/residual_train/pairs.parquet'
    train_base=pd.read_parquet(train_path)
    train_features,fm=nn.load_features(B/'neural_v1_features_fit')
    if sha(train_path)!=fm['input_pairs_sha256']:raise ValueError('Residual graph differs')
    train_features=neural_peer.add_fine(train_features,fm,N/'last4_residual20k',N/'neural_inputs/residual20k/manifest.json',heads['fine'])
    names=model_report['features']
    model=lgb.Booster(model_file=str(model_dir/'adapter.txt'))
    if model.feature_name()!=names:raise ValueError('Selected feature schema differs')
    positions=pd.MultiIndex.from_frame(train_base[KEYS]).get_indexer(pd.MultiIndex.from_frame(train_features[KEYS]))
    if (positions<0).any() or not np.array_equal(train_base.probability.to_numpy()[positions],train_features.probability):raise ValueError('Residual feature anchors differ')
    train=train_base.copy()
    train.loc[positions,'probability']=adapter.corrected_probability(train_features.probability,model.predict(train_features[names].to_numpy(dtype=np.float32),raw_score=True,num_threads=4))
    train_chosen,_=decide_control(train,train,config)
    selected={}
    for role,ids in roles.items():
        original,chosen=(train_base,train_chosen) if role=='residual_train' else (dev_base,dev_chosen)
        # Determine emptiness globally on the full role claimant graph; then
        # retain role-owned keys from its original40 candidates.
        allids=list(original.source1_entity_id.unique())
        routed=select_empty_top4(original,chosen,allids)
        selected[role]=routed[routed.source1_entity_id.isin(ids)].copy()
    cap()
    index=reverse.ReverseIndex(index_path,corpus='train')
    if index.meta['records']!=2206821:raise ValueError('Incomplete actual train reverse corpus')
    allkeys=pd.concat(selected.values(),ignore_index=True)
    own=index.records(allkeys.source1_entity_id.unique())
    if set(own)!=set(allkeys.source1_entity_id):raise ValueError('Missing actual source1 indexed record')
    targets={};target_metadata={}
    for source in [2,3]:
        target_path=Path(f'models/index_train_S{source}.sqlite').resolve()
        with sqlite3.connect(f'{target_path.as_uri()}?mode=ro',uri=True) as conn:
            meta=json.loads(conn.execute("SELECT value FROM metadata WHERE key='build'").fetchone()[0])
            if not meta['complete']:raise ValueError('Incomplete actual target index')
            target_metadata[f'S{source}']=meta
            ids=allkeys.loc[allkeys.candidate_entity_id.str.startswith(f'S{source}-'),'candidate_entity_id'].unique()
            for begin in range(0,len(ids),500):
                batch=list(ids[begin:begin+500])
                for e,n,ad,c in conn.execute('SELECT id,name,address,country FROM records WHERE id IN ('+','.join('?' for _ in batch)+')',batch):
                    targets[e]=reverse.clean_record(dict(entity_id=e,business_name=n,business_address=ad,country=c))
    if set(targets)!=set(allkeys.candidate_entity_id):raise ValueError('Missing real target index records')
    outputs={}
    for role,selected_frame in selected.items():
        directory=a.output/role;directory.mkdir()
        rows=[];diagnostics=[]
        for n,(e,c) in enumerate(selected_frame[KEYS].itertuples(index=False,name=None)):
            cap();features,diag=reverse.pair_features(index,targets[c],own[e])
            if e in diag['rival_ids']:raise ValueError('Own record returned as rival')
            rows.append(dict(zip(KEYS,[e,c]),**features));diagnostics.append(dict(zip(KEYS,[e,c]),**diag))
        raw=pd.DataFrame(rows,columns=[*KEYS,*reverse.FEATURE_NAMES])
        raw[reverse.FEATURE_NAMES]=raw[reverse.FEATURE_NAMES].astype(np.float32)
        original=train_base if role=='residual_train' else dev_base
        joined=adapter.join_reverse(original,raw,set(selected_frame.index))
        joined.to_parquet(directory/'features.parquet',index=False)
        raw.to_parquet(directory/'reverse33.parquet',index=False)
        with (directory/'diagnostics.jsonl').open('x') as stream:
            for row in diagnostics:stream.write(json.dumps(row)+'\n')
        marker={'status':'complete','version':'empty-rescue-direct-v1-36','features':adapter.FEATURES,
            'feature_manifest_kind':'Diagnostic new direct-key route; no fabricated peer/seed; not original sibling route',
            'features_sha256':sha(directory/'features.parquet'),'pair_order_sha256':reverse.pair_order_hash(joined),
            'rows':len(joined),'empty_owners':joined.source1_entity_id.nunique(),'role':role,'split':'development',
            'reference_ids_sha256':sha(B/f'workbenches50k/{role}/reference_ids.json'),
            'baseline_claims_sha256':sha(train_path if role=='residual_train' else B/'learned30k/pairs.parquet'),
            'selected_four_claims_sha256':sha(model_dir/'pairs.parquet') if role!='residual_train' else None,
            'baseline_selection_macro_f05':baseline,'protocol_sha256':sha(a.output/'protocol.json'),
            'reverse33_sha256':sha(directory/'reverse33.parquet'),'diagnostics_sha256':sha(directory/'diagnostics.jsonl'),
            'index_manifest_sha256':sha(Path(str(index_path)+'.complete.json')),'index_metadata':index.index_manifest,
            'target_index_metadata':target_metadata,'query_config':asdict(reverse.QueryConfig()),
            'feature_code_sha256':sha(reverse.__file__),'join_code_sha256':sha(adapter.__file__),
            'labels_used_for_route':False,'audit_labels_read':False,'extension_labels_read':False,'labels_read':False}
        write(directory/'manifest.json',marker);outputs[role]=marker
        print(json.dumps({'stage':'reverse_complete','role':role,'rows':len(joined),'seconds':time.monotonic()-start}),flush=True)
    index.close();cap()
    records={};hashes={};owners=set(allkeys.source1_entity_id);targets_ids=set(allkeys.candidate_entity_id)
    for source in [1,2,3]:
        path=data/f'train_source{source}.tsv';hashes[f'S{source}']=sha(path)
        wanted=owners if source==1 else targets_ids
        for batch in pd.read_csv(path,sep='\t',dtype=str,keep_default_na=False,chunksize=100000):
            cap()
            for row in batch[batch.entity_id.isin(wanted)].to_dict('records'):
                if row['entity_id'] in records:raise ValueError('Repeated raw provided record')
                records[row['entity_id']]=serialize(row)
    if set(records)!=owners|targets_ids:raise ValueError('Missing raw provided record; do not substitute index text')
    for role in roles:
        directory=a.output/role;out=directory/'neural_inputs';out.mkdir()
        f=pd.read_parquet(directory/'features.parquet',columns=KEYS)
        with (out/'pairs.jsonl').open('x') as stream:
            for e,c in f.itertuples(index=False,name=None):
                stream.write(json.dumps(dict(zip(KEYS,[e,c]),text_left=records[e],text_right=records[c]),ensure_ascii=False)+'\n')
        write(out/'manifest.json',{'status':'complete','rows':len(f),'labels_read':False,'role':role,'source_split':'development',
            'input_sha256':sha(out/'pairs.jsonl'),'feature_manifest_sha256':sha(directory/'manifest.json'),
            'feature_file_sha256':sha(directory/'features.parquet'),'source_tsv_sha256':hashes,'code_sha256':sha(__file__),
            'protocol_sha256':sha(a.output/'protocol.json'),'audit_labels_read':False,'extension_labels_read':False})
    write(a.output/'manifest.json',{'status':'complete','source_sha256':sha(__file__),'seconds':time.monotonic()-start,
        'protocol_sha256':sha(a.output/'protocol.json'),'baseline_selection_macro_f05':baseline,
        'roles':{r:{'rows':outputs[r]['rows'],'empty_owners':int(outputs[r]['empty_owners']),
            'feature_manifest_sha256':sha(a.output/r/'manifest.json'),'input_manifest_sha256':sha(a.output/r/'neural_inputs/manifest.json')} for r in roles},
        'audit_labels_read':False,'extension_labels_read':False,'pilot_selection_results_read':False,'production_outputs_scored':False})
    print((a.output/'manifest.json').read_text(),flush=True)


if __name__=='__main__':main()
