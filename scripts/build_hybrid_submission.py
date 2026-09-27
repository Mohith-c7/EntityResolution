#!/usr/bin/env python3
"""Export one frozen NN8-original-route plus NN4-empty-extra hybrid, TSV only.

No labels, fitting, route search, candidate replacement. The candidate decision configuration is explicit.
Both residual margins use ORIGINAL p2 exactly once, on disjoint keyed routes.
"""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import shutil
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'research/sprint_6h'),str(ROOT/'research/sprint_6h/sibling'),str(ROOT/'research/final_2h')]
import lightgbm as lgb
import numpy as np
import pandas as pd
import build_gated_submission as reuse
import build_neural_last4_submission as four
import neural_adapter as nn
import prepare_hybrid_test_rescue as prep
import post_selection_exclusivity as exclusivity
KEYS=four.KEYS
FEATURES=four.FEATURES
ADAPTER8='b462edc1d2a6cae9efdaa5ccb2f9463a49d53f8377049fbaa87ee23ae3da57ef'
HEAD8='5626ab23f1dda7cab60192654cb6835c1d17fc2f87e64b31a3b8dbfef0ef549c'
CANDIDATE='hybrid8_empty4_calibrated_exclusive_v1'
FINE_SCORE_CODE=ROOT/'research/final_2h/neural_layers.py'
CRITERIA={'minimum_point_macro_gain':.0003,'paired_ci_lower_bound_strictly_above':0.,'country_macro_delta_floor':-.0005,'precision_singletons_diagnostic_only':True}

def pinned(path,frozen,group):
    name=Path(path).resolve().relative_to(ROOT).as_posix()
    if frozen[group].get(name)!=reuse.digest(path):raise ValueError('Unpinned dependency: '+name)
    return json.loads(Path(path).read_text()) if str(path).endswith('.json') else None

def verify_freeze(path):
    f=json.loads(path.read_text())
    if f.get('status')!='candidate_frozen' or f.get('candidate')!=CANDIDATE or f.get('mode')!='R2' or not f.get('frozen_at'):
        raise ValueError('Requires explicit frozen hybrid candidate')
    for group in ['code_sha256','model_sha256','schema_sha256','index_sha256','native_sha256']:
        if not f.get(group):raise ValueError('Missing immutable evidence group '+group)
        for file,digest in f[group].items():reuse.resolve_evidence({'path':file,'sha256':digest},ROOT)
    for module in [sys.modules[__name__],reuse,four,nn,nn.reverse,nn.sibling,prep,exclusivity]:pinned(module.__file__,f,'code_sha256')
    return f

def verify_audit(path,frozen,freeze_path):
    """Use the new prospective fresh-audit result; retain old rejected results."""
    gate=pinned(path,frozen,'schema_sha256')
    acceptance=gate.get('hybrid_prospective_acceptance',{})
    if (gate.get('status')!='paired_extension_audit_complete'
            or acceptance.get('passed') is not True or acceptance.get('criteria')!=CRITERIA
            or frozen.get('extension_acceptance_criteria')!=CRITERIA
            or gate.get('baseline_frozen_sha256')!=frozen['baseline_frozen_sha256']
            or gate.get('reservation_plan_sha256')!=frozen['reservation_plan_sha256']):
        raise ValueError('Fresh hybrid audit absent, failed, or wrong comparator/reservation')
    if gate.get('candidate_frozen_sha256') not in [reuse.digest(freeze_path),frozen.get('audit_freeze_sha256')]:
        raise ValueError('Fresh audit belongs to another scoring freeze')
    runtime=gate.get('runtime',{});universe=gate.get('claimant_universe',{})
    if (runtime.get('passed') is not True
            or runtime.get('evidence',{}).get('candidate_frozen_sha256')!=gate['candidate_frozen_sha256']
            or gate.get('full_claim_graph')!={'references':331012,'pairs':13240480}
            or universe.get('kind')!='complete_heldout'
            or any(universe.get(k) is not True for k in ['complete','scoring_complete','supervised_training_excluded'])
            or universe.get('declared_references')!=331012 or universe.get('scored_references')!=331012
            or universe.get('pairs')!=13240480):
        raise ValueError('Fresh audit runtime/full claimant graph evidence incomplete')
    return gate

def read_scores(directory,input_manifest,head_sha,head_field,code,frozen,rows):
    im=json.loads(input_manifest.read_text());sm=json.loads((directory/'manifest.json').read_text())
    if (im.get('status')!='complete' or sm.get('status')!='complete'
            or im.get('labels_read') is not False or sm.get('labels_read') is not False
            or im.get('source_split')!='test' or sm.get('rows')!=rows or im.get('rows')!=rows
            or im['input_sha256']!=reuse.digest(input_manifest.parent/'pairs.jsonl')
            or sm.get('input_sha256')!=im['input_sha256'] or sm.get(head_field)!=head_sha
            or sm.get('scores_sha256')!=reuse.digest(directory/'scores.jsonl')
            or sm.get('code_sha256')!=reuse.digest(code)
            or ('input_manifest_sha256' in sm and sm['input_manifest_sha256']!=reuse.digest(input_manifest))):
        raise ValueError('Neural score/input/head/code seal differs: '+str(directory))
    pinned(code,frozen,'code_sha256')
    values=pd.read_json(directory/'scores.jsonl',lines=True,dtype={k:str for k in KEYS})
    if (set(values)!=set(KEYS+['neural_probability']) or values.duplicated(KEYS).any()
            or values[KEYS].isna().any().any() or not values.neural_probability.between(0,1).all()
            or not np.isfinite(values.neural_probability).all()):raise ValueError('Invalid score keys/probabilities')
    return values,sm,im

def add_old(frame,values):
    if frame.duplicated(KEYS).any() or values.duplicated(KEYS).any():raise ValueError('Duplicate direct keys')
    positions=pd.MultiIndex.from_frame(values[KEYS]).get_indexer(pd.MultiIndex.from_frame(frame[KEYS]))
    if len(frame)!=len(values) or (positions<0).any():raise ValueError('Old score key universe differs')
    result=frame.copy(deep=False);result['neural_probability']=values.neural_probability.to_numpy()[positions]
    return result

def apply_hybrid(base,route,extra,model8,model4,threads=8,base_verified=False):
    """One union scatter from original anchors; overlap or double anchoring fails."""
    if pd.MultiIndex.from_frame(route[KEYS]).isin(pd.MultiIndex.from_frame(extra[KEYS])).any():raise ValueError('Hybrid routes overlap')
    corrections=[]
    for features,model in [(route,model8),(extra,model4)]:
        if model.feature_name()!=FEATURES or model.num_feature()!=38:raise ValueError('Native38 schema/order differs')
        inputs=features[FEATURES].to_numpy(dtype=np.float32)
        if not np.isfinite(inputs).all():raise ValueError('Nonfinite38 input')
        corrections.append(model.predict(inputs,raw_score=True,num_threads=threads))
    combined=pd.concat([route,extra],ignore_index=True)
    return four.scatter_corrections(base,combined,np.concatenate(corrections),base_verified=base_verified)

def validate_extra_route(base,old,extra,blue,protocol):
    if protocol.get('blue_matching_sha256')!=reuse.digest(blue/'matching_results.tsv'):raise ValueError('Global blue gate changed')
    matching=pd.read_csv(blue/'matching_results.tsv',sep='\t',dtype=str,keep_default_na=False)
    if matching.source1_entity_id.duplicated().any() or set(matching.source1_entity_id)!=set(base.source1_entity_id):raise ValueError('Blue global owner universe differs')
    empty=set(matching.loc[matching.matched_entity_ids.eq(''),'source1_entity_id'])
    contexts=[]
    for begin in range(0,len(base),2000000):
        part=base.iloc[begin:begin+2000000]
        contexts.append(part.loc[part.source1_entity_id.isin(empty)])
    context=pd.concat(contexts,ignore_index=True)
    expected=prep.select_extra(context,set(old[KEYS].itertuples(index=False,name=None)))
    if not expected[KEYS].equals(extra.sort_values(KEYS,kind='stable').reset_index(drop=True)[KEYS]):raise ValueError('Empty top4-after-exclusion route differs')
    return len(empty)

def export_matching(rows,frame,chosen,blue,output):
    matches=defaultdict(list)
    for e,c in frame.loc[chosen,KEYS].itertuples(index=False,name=None):matches[e].append(c)
    countries={}
    with (output/'matching_results.tsv').open('x',encoding='utf-8',newline='') as stream:
        stream.write('source1_entity_id\tmatched_entity_ids\n')
        for row in rows:
            e=row['entity_id'];values=matches[e]
            stream.write(e+'\t'+','.join(values)+'\n')
            label=row.get('country','').strip().casefold()
            countries.setdefault(label,Counter()).update(entities=1,scored_candidates=40,predicted_links=len(values),empty_predictions=int(not values),empty_candidates=0)
    # Preserve the verified candidate TSV verbatim for both official validators.
    # Only matching_results.tsv is the user-requested delivery.
    shutil.copy2(blue/'candidate_pairs.tsv',output/'candidate_pairs.tsv')
    return {'country':countries,'candidate_histogram':{40:len(rows)},'empty_predictions':sum(c['empty_predictions'] for c in countries.values())}

def build(a):
    start=time.monotonic();stages={}
    preserved=[(ROOT/'output'/f'submission_{v}').resolve() for v in ['03','04','05']]
    if a.output.exists() or any(a.output.resolve()==p or p in a.output.resolve().parents for p in preserved):raise ValueError('Require new versioned output')
    if a.threads<=0:raise ValueError('Positive threads required')
    frozen=verify_freeze(a.freeze);gate=verify_audit(a.audit_evidence,frozen,a.freeze)
    reuse_manifest=json.loads(a.reuse_manifest.read_text())
    if reuse_manifest['baseline_frozen']['sha256']!=frozen['parent_frozen_sha256']:raise ValueError('Original04 parent differs')
    baseline=json.loads(reuse.resolve_evidence(reuse_manifest['baseline_frozen'],ROOT).read_text())
    if frozen['baseline_decision_config']!={k:baseline[k] for k in ['threshold','t_first','t_rest']}:raise ValueError('Baseline decision config differs')
    if frozen['decision_config']!={'threshold':.83,'t_first':.6999999999999998,'t_rest':.79} or frozen.get('candidate_post_selection_policy')!='selected_only_probability_then_source1_id':raise ValueError('Fixed candidate decision/exclusivity differs')
    t=time.monotonic();rows,base=reuse.load_verified_chunks(reuse_manifest,a.test_dir);stages['verified_base_load']=time.monotonic()-t
    if len(rows)!=1732544 or len(base)!=69301760:raise ValueError('Incomplete official TEST universe')
    blue_report=json.loads((a.blue/'submission_report.json').read_text())
    if blue_report['candidate_frozen_sha256']!=frozen['baseline_frozen_sha256']:raise ValueError('Blue05 comparator freeze differs')
    if blue_report['status']!='complete' or blue_report['validation']['strict']!='PASS' or blue_report['validation']['official']!='PASS':raise ValueError('Blue05 verification failed')
    for n in ['matching_results.tsv','candidate_pairs.tsv']:
        if reuse.digest(a.blue/n)!=blue_report['files_sha256'][n]:raise ValueError('Blue05 TSV changed')
    old,fm=nn.load_features(a.features)
    if len(old)!=801380 or fm['split']!='test' or fm['input_pair_order_sha256']!=reuse_manifest['pair_order_sha256'] or reuse.digest(a.features/'features.parquet')!=blue_report['evidence_sha256']['features']:raise ValueError('Original801380 route/features differ')
    fitted=set()
    for directory,expected,head,head_sha in [(a.model8,ADAPTER8,a.head8_manifest,HEAD8),(a.model4,four.ADAPTER_SHA,a.head4_manifest,four.FINE_HEAD_SHA)]:
        report=pinned(directory/'report.json',frozen,'model_sha256');pinned(directory/'adapter.txt',frozen,'model_sha256');hm=pinned(head,frozen,'model_sha256')
        if (reuse.digest(directory/'adapter.txt')!=expected or report['model_sha256']!=expected or report['features']!=FEATURES
                or report.get('fine_head_sha256')!=head_sha or reuse.digest(head)!=head_sha
                or report.get('original_neural_head_sha256')!=four.OLD_HEAD_SHA
                or report.get('mode') not in [None,'anchored_neural'] or report.get('calibration_logit_shift',0)!=0
                or report.get('peer_head_sha256') is not None or hm.get('status')!='complete'):raise ValueError('Anchored native/head schema differs')
        fitted.update(report['fit_owners']);fitted.update(hm['training_owners'])
    oh=pinned(a.old_head_manifest,frozen,'model_sha256')
    if reuse.digest(a.old_head_manifest)!=four.OLD_HEAD_SHA or fm['neural_head_manifest_sha256']!=four.OLD_HEAD_SHA:raise ValueError('Old head lineage changed')
    fitted.update(oh['training_owners'])
    if fitted & set(r['entity_id'] for r in rows):raise ValueError('Fitted owner entered TEST graph')
    score8,sm8,im8=read_scores(a.fine8_scores,a.fine8_input_manifest,HEAD8,'checkpoint_manifest_sha256',FINE_SCORE_CODE,frozen,len(old))
    source_hashes={f'S{i}':reuse_manifest['test_files'][f'test_source{i}.tsv']['sha256'] for i in [1,2,3]}
    if im8['source_tsv_sha256']!=source_hashes or im8['feature_manifest_sha256']!=fm['reverse_feature_manifest_sha256']:raise ValueError('Original route raw test lineage differs')
    original38=four.assemble_fine(old,score8)
    em=json.loads((a.extra_features/'manifest.json').read_text());protocol=json.loads((a.extra_features/'protocol.json').read_text())
    if (em.get('status')!='complete' or em.get('version')!=prep.VERSION or em.get('features')!=nn.reverse.FEATURES
            or em.get('split')!='test' or em.get('labels_read') is not False
            or em['source_sha256']!=reuse.digest(prep.__file__) or em['features_sha256']!=reuse.digest(a.extra_features/'features.parquet')
            or em['protocol_sha256']!=reuse.digest(a.extra_features/'protocol.json')
            or protocol['original_pair_order_sha256']!=reuse_manifest['pair_order_sha256']
            or protocol['blue_frozen_sha256']!=blue_report['candidate_frozen_sha256']
            or em['source_tsv_sha256']!=source_hashes):raise ValueError('New direct TEST feature seal differs')
    extra=pd.read_parquet(a.extra_features/'features.parquet')
    if em['feature_pair_order_sha256']!=nn.sibling.pair_order_hash(extra):raise ValueError('Direct feature pair order changed')
    t=time.monotonic();empty_count=validate_extra_route(base,old,extra,a.blue,protocol);stages['global_blue_route_replay']=time.monotonic()-t
    input_manifest=a.extra_features/'neural_inputs/manifest.json'
    if em['input_manifest_sha256']!=reuse.digest(input_manifest):raise ValueError('Extra raw input manifest changed')
    scoreold,smo,imo=read_scores(a.extra_old_scores,input_manifest,four.OLD_HEAD_SHA,'head_manifest_sha256',ROOT/'research/sprint_6h/neural_frozen_head.py',frozen,len(extra))
    scorefine,smf,imf=read_scores(a.extra_fine4_scores,input_manifest,four.FINE_HEAD_SHA,'checkpoint_manifest_sha256',FINE_SCORE_CODE,frozen,len(extra))
    if imo['source_tsv_sha256']!=source_hashes or imo['route_sha256']!=em['route_sha256'] or imo['pair_order_sha256']!=em['feature_pair_order_sha256']:raise ValueError('Extra raw route/test lineage differs')
    extra38=four.assemble_fine(add_old(extra,scoreold),scorefine)
    t=time.monotonic();frame=apply_hybrid(base,original38,extra38,lgb.Booster(model_file=str(a.model8/'adapter.txt')),lgb.Booster(model_file=str(a.model4/'adapter.txt')),a.threads,True);stages['native_and_scatter']=time.monotonic()-t
    runner=reuse.resolve_evidence(frozen['original_runner'],ROOT);pinned(runner,frozen,'code_sha256')
    t=time.monotonic();chosen,lost=reuse.load_original_decide(runner)(frame,frame,frozen['decision_config']);chosen=np.asarray(chosen,dtype=bool);lost=np.asarray(lost,dtype=bool)
    if len(chosen)!=len(frame) or len(lost)!=len(frame):raise ValueError('Global decision alignment changed')
    chosen,exclusivity_lost=exclusivity.exclusive_selected(frame,chosen)
    stages['global_decision_then_selected_exclusivity']=time.monotonic()-t
    a.output.mkdir(parents=True);shutil.copy2(a.freeze,a.output/'frozen.json')
    t=time.monotonic();statistics=export_matching(rows,frame,chosen,a.blue,a.output);stages['matching_tsv_export']=time.monotonic()-t
    expected=blue_report['files_sha256']['candidate_pairs.tsv']
    if reuse.digest(a.output/'candidate_pairs.tsv')!=expected:raise ValueError('Candidate TSV changed')
    validation=reuse.validate_outputs(a.output,a.test_dir)
    result={'status':'complete' if validation['strict']==validation['official']=='PASS' else 'validation_failed','candidate':CANDIDATE,'mode':'R2',
        'entities':len(rows),'scored_candidates':len(frame),'original_route_rows':len(old),'extra_route_rows':len(extra),
        'blue05_global_empty_owners':empty_count,'candidate_frozen_sha256':reuse.digest(a.freeze),'fresh_audit_gate_sha256':reuse.digest(a.audit_evidence),
        'candidate_pair_order_sha256':reuse_manifest['pair_order_sha256'],'candidate_tsv_identical_to_submission05':True,
        'original_anchor_once_disjoint_routes':True,'original_global_decide_source_unchanged':True,'candidate_decision_config':frozen['decision_config'],'baseline_decision_config':frozen['baseline_decision_config'],'post_selection_exclusivity':True,'exclusivity_removed_links':int(exclusivity_lost.sum()),'labels_read':False,'audit_labels_read':False,
        'predicted_links':int(chosen.sum()),'ownership_lost_pairs':int(lost.sum()),'statistics':statistics,'validation':validation,
        'stage_seconds':stages,'seconds':time.monotonic()-start,'source_sha256':reuse.digest(__file__),
        'evidence_sha256':{'features37':reuse.digest(a.features/'features.parquet'),'extra_features36':em['features_sha256'],
            'extra_protocol':reuse.digest(a.extra_features/'protocol.json'),'head8_scores':sm8['scores_sha256'],
            'extra_old_scores':smo['scores_sha256'],'extra_fine4_scores':smf['scores_sha256'],
            'adapter8':ADAPTER8,'adapter4':four.ADAPTER_SHA,'head8_manifest':HEAD8,'head4_manifest':four.FINE_HEAD_SHA,'old_head_manifest':four.OLD_HEAD_SHA},
        'files_sha256':{n:reuse.digest(a.output/n) for n in ['matching_results.tsv','candidate_pairs.tsv']},'user_deliverable':'matching_results.tsv'}
    reuse.write_json(a.output/'submission_report.json',result)
    if result['status']!='complete':raise ValueError('Submission validation failed')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['freeze','audit_evidence','reuse_manifest','features','fine8_scores','fine8_input_manifest','extra_features','extra_old_scores','extra_fine4_scores','model8','model4','head8_manifest','head4_manifest','old_head_manifest','blue','test_dir','output']:
        p.add_argument('--'+name.replace('_','-'),type=Path,required=True)
    p.add_argument('--threads',type=int,default=8)
    print(json.dumps(build(p.parse_args()),indent=2))
