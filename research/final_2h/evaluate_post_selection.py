"""Bounded development-only review of pure post-selection exclusivity.

The complete pre-existing30k graph is used for all decisions. Only the already
exposed early_stop/selection workbenches can supply labels, if decisions change.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(8<<20),b''): h.update(part)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for field in ['repo','pairs','output','helper']:p.add_argument('--'+field,type=Path,required=True)
    a=p.parse_args(); started=time.monotonic()
    if a.output.exists():raise FileExistsError(a.output)
    sys.path[:0]=[str(a.repo/'scripts'),str(a.repo/'code/business_entity_resolution')]
    from evaluate_sprint import decide_control,paired_evaluate,predictions_for
    spec=importlib.util.spec_from_file_location('post_selected',a.helper)
    helper=importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
    workbenches=a.repo/'research/sprint_6h/sibling/workbenches50k'
    ids={}; markers={}
    for split,size in [('early_stop',10000),('selection',20000)]:
        directory=workbenches/split
        marker=json.loads((directory/'manifest.json').read_text())
        owners=json.loads((directory/'reference_ids.json').read_text())
        if len(owners)!=size or len(set(owners))!=size or marker.get('split')!='development' or marker.get('audit_labels_read') is not False:
            raise ValueError('Require previously exposed development cohort only')
        ids[split]=owners; markers[split]=marker
    frame=pd.read_parquet(a.pairs)
    all_ids=set(ids['early_stop'])|set(ids['selection'])
    if len(all_ids)!=30000 or set(frame.source1_entity_id)!=all_ids or frame.duplicated(['source1_entity_id','candidate_entity_id']).any():
        raise ValueError('Require exact complete30k graph')
    report_path=a.repo/'research/final_2h/neural_last4_fine_v1/report.json'
    fitted=set(json.loads(report_path.read_text())['fit_owners'])
    for path in ['models/sprint_6h/neural/frozen_head120k_v1/manifest.json','models/final_2h/neural_last4_continue_v1/manifest.json']:
        fitted.update(json.loads((a.repo/path).read_text())['training_owners'])
    if all_ids&fitted:raise ValueError('Model-fitted owner entered development evaluation')
    frozen=a.repo/'research/sprint_6h/coordination/baseline_freeze/frozen.json'
    config=json.loads(frozen.read_text())
    original,lost=decide_control(frame,frame,config)
    chosen,removed=helper.exclusive_selected(frame,original)
    counts=frame.loc[original].candidate_entity_id.value_counts()
    report={'status':'complete', 'scope':'Existing exposed complete30k four-layer development graph',
      'all_claimant_owners':30000,'pairs':len(frame),'original_accepted_pairs':int(original.sum()),
      'post_selection_accepted_pairs':int(chosen.sum()),'removed_pairs':int(removed.sum()),
      'added_pairs':int((chosen&~original).sum()),'original_duplicate_targets':int((counts>1).sum()),
      'post_selection_duplicate_targets':int((frame.loc[chosen].candidate_entity_id.value_counts()>1).sum()),
      'labels_read':False,'fresh_or_audit_labels_read':False,'release_policy_changed':False,
      'input_sha256':{'pairs':sha(a.pairs),'helper':sha(a.helper),'frozen':sha(frozen),'selected_model_report':sha(report_path)},
      'evaluation':{}}
    if not removed.any():
        report['conclusion']='Masks identical; every quality metric is identical without reading any labels.'
    else:
        for split,owners in ids.items():
            directory=workbenches/split; marker=markers[split]
            for name in ['truth','countries']:
                path=directory/(name+'.json')
                if marker.get(name+'_sha256') and sha(path)!=marker[name+'_sha256']:
                    raise ValueError('Sealed development evidence changed')
            truth=json.loads((directory/'truth.json').read_text())
            countries=json.loads((directory/'countries.json').read_text())
            if set(truth)!=set(owners) or set(countries)!=set(owners):raise ValueError('Development label coverage differs')
            take=frame.source1_entity_id.isin(owners).to_numpy()
            baseline=predictions_for(frame.loc[take],original[take],owners)
            candidate=predictions_for(frame.loc[take],chosen[take],owners)
            report['evaluation'][split]=paired_evaluate(truth,baseline,candidate,countries,role='development')
        report['labels_read']=True
        report['conclusion']='Development evidence only; no final policy adoption or audit claim.'
    report['seconds']=time.monotonic()-started
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='evaluation'}),flush=True)

if __name__=='__main__':main()
