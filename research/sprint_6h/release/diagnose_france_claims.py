#!/usr/bin/env python3
"""Unlabeled French veto diagnostics with complete original-policy competition.

Streams every sealed candidate, keeps every baseline-floor-eligible claim across
all countries, proves exact baseline matching TSV byte reconstruction, then
compares experimental French veto tables. No submission TSV is written.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import pyarrow as pa

ROOT=Path(__file__).resolve().parents[3]
spec=importlib.util.spec_from_file_location('release_builder',ROOT/'scripts/build_gated_submission.py')
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)


def source_rows(path):
    with path.open(encoding='utf-8-sig',newline='') as stream:
        return [{'entity_id':row['entity_id'],'country':row['country'].strip().casefold()}
                for row in csv.DictReader(stream,delimiter='\t')]


def collision_summary(frame, chosen, veto, threshold):
    result={}
    for name, mask in (('eligible',~veto),('high',(~veto)&(frame.probability.to_numpy()>=threshold)),('final',chosen)):
        counts=frame.loc[mask,'candidate_entity_id'].value_counts(sort=False)
        result[name+'_collision_targets']=int((counts>1).sum())
        result[name+'_max_owners']=int(counts.max()) if len(counts) else 0
    return result


def country_links(frame, chosen, countries):
    return dict(Counter(frame.loc[chosen,'source1_entity_id'].map(countries)))


def country_collision_summary(frame,chosen,veto,threshold,countries):
    result={country:{} for country in sorted(set(countries.values()))}
    for name,mask in (('eligible',~veto),('high',(~veto)&(frame.probability.to_numpy()>=threshold)),('final',chosen)):
        claims=frame.loc[mask,r.PAIR_KEYS]
        counts=claims.candidate_entity_id.value_counts(sort=False)
        competing=claims.loc[claims.candidate_entity_id.map(counts).gt(1)].copy()
        competing['country']=competing.source1_entity_id.map(countries)
        for country in result:
            subset=competing.loc[competing.country.eq(country)]
            result[country][name+'_collision_targets']=int(subset.candidate_entity_id.nunique())
            result[country][name+'_references_with_rivals']=int(subset.source1_entity_id.nunique())
    return result


def reconstruction_sha(rows,frame,chosen):
    matches=defaultdict(list)
    for s,c in frame.loc[chosen,r.PAIR_KEYS].itertuples(index=False,name=None):matches[s].append(c)
    h=hashlib.sha256(b'source1_entity_id\tmatched_entity_ids\n')
    for row in rows:h.update((row['entity_id']+'\t'+','.join(matches.get(row['entity_id'],[]))+'\n').encode())
    return h.hexdigest()


def prepare(manifest_path,proof_path,frozen_path,test_dir,output,root=ROOT):
    pa.set_cpu_count(1);pa.set_io_thread_count(1)
    manifest_path,proof_path,frozen_path,test_dir,output=map(Path,(manifest_path,proof_path,frozen_path,test_dir,output))
    if output.exists():raise ValueError('Diagnostic preparation destination must be new/versioned')
    if any((Path(root)/'output'/name).resolve()==output.resolve() or (Path(root)/'output'/name).resolve() in output.resolve().parents for name in ('submission_03','submission_04')):
        raise ValueError('Preserved baseline is not a diagnostic destination')
    start=time.perf_counter();m=json.loads(manifest_path.read_text());proof=json.loads(proof_path.read_text());config=json.loads(frozen_path.read_text())
    if proof.get('copy_byte_parity') is not True or proof.get('status')!='complete' or proof['portable_manifest_sha256']!=r.digest(manifest_path):
        raise ValueError('Exact complete copied baseline proof required')
    if r.digest(frozen_path)!=m['baseline_frozen']['sha256'] or proof['baseline_frozen_sha256']!=r.digest(frozen_path):
        raise ValueError('Frozen baseline differs from copied seal')
    runner=Path(root)/'scripts/run_frozen_pipeline.py'
    runner_sha=config['code_sha256']['scripts/run_frozen_pipeline.py']
    if r.digest(runner)!=runner_sha:raise ValueError('Original runner AST hash mismatch')
    floor=min(config['t_first'],config['t_rest']);threshold=config['threshold']
    if not 0<floor<=threshold or not np.isfinite([floor,threshold,config['t_first'],config['t_rest']]).all():
        raise ValueError('Eligible-subset proof requires finite positive floor <= high threshold')
    source=test_dir/'test_source1.tsv'
    if r.digest(source)!=m['test_files']['test_source1.tsv']['sha256']:raise ValueError('Source1 hash mismatch')
    rows=source_rows(source);countries={row['entity_id']:row['country'] for row in rows}
    if len(rows)!=m['entities'] or len(countries)!=len(rows) or r.ids_digest([row['entity_id'] for row in rows])!=m['input_ids_sha256']:
        raise ValueError('Source1 full coverage/order mismatch')
    report_path=r.resolve_evidence(m['baseline_report'],root);baseline=json.loads(report_path.read_text())
    if baseline['status']!='complete' or baseline['validation']['official_returncode']!=0 or baseline['validation']['strict_issues']!=[]:
        raise ValueError('Baseline report is incomplete/unvalidated')
    parts=[];scanned=0
    for number,item in enumerate(m['chunks']):
        if item['chunk']!=number:raise ValueError('Noncontiguous sealed chunks')
        path=r.resolve_evidence(item['parquet'],root)
        part=pd.read_parquet(path,columns=r.PAIR_KEYS+r.SCORE_COLUMNS)
        scanned+=len(part)
        part['candidate_order']=part.groupby('source1_entity_id',sort=False).cumcount().to_numpy()
        parts.append(part.loc[part.probability>=floor])
        if (number+1)%1000==0:print(json.dumps({'scanned_chunks':number+1,'scanned_pairs':scanned}),flush=True)
    if scanned!=m['pairs']:raise ValueError('Incomplete full candidate scan')
    frame=pd.concat(parts,ignore_index=True);del parts
    if frame.duplicated(r.PAIR_KEYS).any():raise ValueError('Duplicate retained pairs')
    release={'policy':'original','decision_config':config,'original_runner':{'path':str(runner),'sha256':runner_sha}}
    read_end=time.perf_counter();chosen,lost,details=r.make_decisions(frame,release,root);decided=time.perf_counter()
    reconstructed=reconstruction_sha(rows,frame,chosen)
    expected=baseline['output_sha256']['matching_results.tsv']
    if reconstructed!=expected or int(chosen.sum())!=baseline['predicted_links']:
        raise ValueError('Eligible-only replay differs from complete sealed submission04')
    links=country_links(frame,chosen,countries)
    for country,state in baseline['country'].items():
        if links.get(country,0)!=state['predicted_links']:raise ValueError('Country baseline replay mismatch')
    output.mkdir(parents=True)
    frame.to_parquet(output/'eligible_claims.parquet',index=False)
    np.savez_compressed(output/'baseline_masks.npz',chosen=chosen,lost=lost)
    result={'status':'baseline_parity_verified','scope':'Unlabeled diagnostic; no submission TSVs; complete scanned candidate universe with exact floor subset for arbitration.',
            'manifest_sha256':r.digest(manifest_path),'copy_proof_sha256':r.digest(proof_path),'frozen_sha256':r.digest(frozen_path),
            'runner_sha256':runner_sha,'script_sha256':r.digest(__file__),'all_references':len(rows),'scanned_pairs':scanned,
            'retained_pairs':len(frame),'excluded_below_floor_pairs':scanned-len(frame),'floor':floor,'high_threshold':threshold,
            'pair_order_sha256':m['pair_order_sha256'],'retained_pair_order_sha256':r.pair_order_digest(frame),
            'eligible_claims_sha256':r.digest(output/'eligible_claims.parquet'),'baseline_masks_sha256':r.digest(output/'baseline_masks.npz'),
            'baseline_matching_sha256':reconstructed,'exact_matching_byte_reconstruction':True,
            'baseline_links':int(chosen.sum()),'baseline_ownership_lost':int(lost.sum()),'country_links':links,
            'collisions':collision_summary(frame,chosen,np.zeros(len(frame),dtype=bool),threshold),
            'seconds':{'scan_and_subset':read_end-start,'global_original_decision':decided-read_end,'total':time.perf_counter()-start},
            'subset_equivalence':'All high-threshold rivals remain because threshold>=floor. Any acceptable adjusted score is>=floor; omitted scores cannot outrank it. When all retained claims are vetoed/lost, neither retained nor omitted belowfloor claims can pass either acceptance threshold. Original candidate relative order remains unchanged.'}
    r.write_json(output/'baseline_parity.json',result);return result


def compare(cache,table_path,metadata_path,frozen_path,test_dir,output,root=ROOT,*,manifest_path):
    pa.set_cpu_count(1);pa.set_io_thread_count(1)
    cache,table_path,metadata_path,frozen_path,test_dir,output=map(Path,(cache,table_path,metadata_path,frozen_path,test_dir,output))
    if output.exists():raise ValueError('Veto diagnostic output must be new/versioned')
    if cache.resolve() in output.resolve().parents:raise ValueError('Diagnostic comparison must preserve baseline cache')
    if any((Path(root)/'output'/name).resolve()==output.resolve() or (Path(root)/'output'/name).resolve() in output.resolve().parents for name in ('submission_03','submission_04')):
        raise ValueError('Preserved baseline is not a diagnostic destination')
    start=time.perf_counter();proof=json.loads((cache/'baseline_parity.json').read_text());meta=json.loads(metadata_path.read_text());config=json.loads(frozen_path.read_text())
    if proof['status']!='baseline_parity_verified' or proof['frozen_sha256']!=r.digest(frozen_path):raise ValueError('Exact baseline replay proof required')
    if r.digest(manifest_path)!=proof['manifest_sha256']:raise ValueError('Baseline reuse manifest changed')
    manifest=json.loads(Path(manifest_path).read_text())
    if r.digest(test_dir/'test_source1.tsv')!=manifest['test_files']['test_source1.tsv']['sha256']:
        raise ValueError('Source1 changed after verified baseline replay')
    if meta['table_sha256']!=r.digest(table_path) or meta['baseline_frozen_sha256']!=proof['frozen_sha256']:
        raise ValueError('Gate metadata differs from table or baseline descriptor')
    if r.digest(cache/'eligible_claims.parquet')!=proof['eligible_claims_sha256'] or r.digest(cache/'baseline_masks.npz')!=proof['baseline_masks_sha256']:
        raise ValueError('Baseline diagnostic cache changed')
    frame=pd.read_parquet(cache/'eligible_claims.parquet');table=pd.read_parquet(table_path)
    if not set(r.PAIR_KEYS+['gate_veto','gate_reason'])<=set(table) or table.duplicated(r.PAIR_KEYS).any():raise ValueError('Malformed French pair gate table')
    if table.gate_veto.isna().any() or not table.gate_veto.map(lambda v:isinstance(v,(bool,np.bool_))).all():raise ValueError('Nonboolean veto')
    if table.gate_reason.isna().any() or not table.gate_reason.map(lambda v:isinstance(v,str)).all():raise ValueError('Malformed veto reason')
    countries={row['entity_id']:row['country'] for row in source_rows(test_dir/'test_source1.tsv')}
    french=frame.source1_entity_id.map(countries).eq('france')
    expected=pd.MultiIndex.from_frame(frame.loc[french,r.PAIR_KEYS]);indexed=table.set_index(r.PAIR_KEYS)
    if meta['pair_order_sha256']!=r.pair_order_digest(frame.loc[french]):raise ValueError('French gate pair-order lineage mismatch')
    if len(table)!=int(french.sum()) or not expected.isin(indexed.index).all():raise ValueError('French eligible table is not exact full expected pair universe')
    frame['gate_veto']=False;frame['gate_reason']='signal_skipped_non_france'
    aligned=indexed.loc[expected]
    frame.loc[french,'gate_veto']=aligned.gate_veto.to_numpy();frame.loc[french,'gate_reason']=aligned.gate_reason.to_numpy()
    runner=Path(root)/'scripts/run_frozen_pipeline.py'
    release={'policy':'original','decision_config':config,'original_runner':{'path':str(runner),'sha256':proof['runner_sha256']}}
    with np.load(cache/'baseline_masks.npz') as masks:baseline_chosen=masks['chosen'];baseline_lost=masks['lost']
    prepared=time.perf_counter();reused=not frame.gate_veto.any()
    if reused:chosen,lost=baseline_chosen.copy(),baseline_lost.copy()
    else:chosen,lost,details=r.make_decisions(frame,release,root)
    decided=time.perf_counter()
    changed=baseline_chosen!=chosen
    ownership_changed=baseline_lost!=lost
    inspect_mask=changed|frame.gate_veto.to_numpy()|ownership_changed
    diagnostics=frame.loc[inspect_mask].copy()
    diagnostics['baseline_accepted']=baseline_chosen[inspect_mask]
    diagnostics['gated_accepted']=chosen[inspect_mask]
    diagnostics['baseline_ownership_lost']=baseline_lost[inspect_mask]
    diagnostics['gated_ownership_lost']=lost[inspect_mask]
    output.mkdir(parents=True);diagnostics.to_parquet(output/'changed_claims.parquet',index=False)
    links=country_links(frame,chosen,countries)
    result={'status':'diagnostic_complete','ready_for_upload':False,'labels_used':False,
            'baseline_parity_sha256':r.digest(cache/'baseline_parity.json'),'frozen_sha256':r.digest(frozen_path),
            'diagnostic_script_sha256':r.digest(__file__),'global_decision_reused_verified_baseline':reused,
            'table_sha256':r.digest(table_path),'metadata_sha256':r.digest(metadata_path),'gate_metadata':meta,
            'scanned_full_pairs':proof['scanned_pairs'],'retained_complete_country_claims':len(frame),
            'excluded_below_floor_pairs':proof['excluded_below_floor_pairs'],'french_signal_rows':int(french.sum()),
            'non_france_skipped_signal_rows':int((~french).sum()),'floor':proof['floor'],
            'veto_claims':int(frame.gate_veto.sum()),'vetoed_baseline_accepted':int((frame.gate_veto.to_numpy()&baseline_chosen).sum()),
            'removed_accepted_claims':int((baseline_chosen&~chosen).sum()),'added_accepted_claims':int((~baseline_chosen&chosen).sum()),
            'changed_claims':int(changed.sum()),'changed_target_winners':int(frame.loc[changed,'candidate_entity_id'].nunique()),
            'changed_ownership_states':int(ownership_changed.sum()),'inspectable_claim_rows':int(inspect_mask.sum()),
            'baseline_links':int(baseline_chosen.sum()),'gated_links':int(chosen.sum()),
            'baseline_country_links':proof['country_links'],'gated_country_links':links,
            'country_link_deltas':{c:links.get(c,0)-proof['country_links'].get(c,0) for c in sorted(set(countries.values()))},
            'baseline_collisions':proof['collisions'],'gated_collisions':collision_summary(frame,chosen,frame.gate_veto.to_numpy(),config['threshold']),
            'baseline_country_collisions':country_collision_summary(frame,baseline_chosen,np.zeros(len(frame),dtype=bool),config['threshold'],countries),
            'gated_country_collisions':country_collision_summary(frame,chosen,frame.gate_veto.to_numpy(),config['threshold'],countries),
            'reason_counts':{str(k):int(v) for k,v in frame.loc[french,'gate_reason'].value_counts().items()},
            'changed_claims_sha256':r.digest(output/'changed_claims.parquet'),
            'seconds':{'read_and_alignment':prepared-start,'global_original_decision':decided-prepared,'total':time.perf_counter()-start},
            'interpretation':'Mechanical link/collision deltas only; France quality unknown. Skipped signal rows are not observed street agreements. Complete stored candidates/exports remain unchanged.'}
    r.write_json(output/'diagnostic.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prepare');a.add_argument('--manifest',type=Path,required=True);a.add_argument('--copy-proof',type=Path,required=True)
    a.add_argument('--frozen',type=Path,required=True);a.add_argument('--test-dir',type=Path,required=True);a.add_argument('--output',type=Path,required=True)
    a=sub.add_parser('compare');a.add_argument('--cache',type=Path,required=True);a.add_argument('--table',type=Path,required=True)
    a.add_argument('--manifest',type=Path,required=True)
    a.add_argument('--metadata',type=Path,required=True);a.add_argument('--frozen',type=Path,required=True);a.add_argument('--test-dir',type=Path,required=True);a.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.command=='prepare':result=prepare(args.manifest,args.copy_proof,args.frozen,args.test_dir,args.output)
    else:result=compare(args.cache,args.table,args.metadata,args.frozen,args.test_dir,args.output,manifest_path=args.manifest)
    print(json.dumps(result,indent=2),flush=True)
