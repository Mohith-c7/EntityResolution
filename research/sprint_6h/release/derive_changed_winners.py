#!/usr/bin/env python3
"""Derive complete changed-target owner lists without rerunning decisions.

Every affected target starts with all verified baseline accepted owners. Accepted
pair deltas remove/add owners; unchanged co-owners remain in both complete lists.
Writes separate new artifacts, never changes original comparisons or caches.
"""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa

ROOT=Path(__file__).resolve().parents[3]
spec=importlib.util.spec_from_file_location('release_builder',ROOT/'scripts/build_gated_submission.py')
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)


def derive(cache,comparison):
    pa.set_cpu_count(1);pa.set_io_thread_count(1)
    cache,comparison=map(Path,(cache,comparison))
    outputs=[comparison/name for name in ('changed_winners.json','changed_winners.parquet','changed_winners_manifest.json')]
    if any(path.exists() for path in outputs):raise ValueError('Changed winner outputs must be new')
    parity_path=cache/'baseline_parity.json';parity=json.loads(parity_path.read_text())
    report_path=comparison/'diagnostic.json';report=json.loads(report_path.read_text())
    if parity['status']!='baseline_parity_verified' or report['status']!='diagnostic_complete' or report['baseline_parity_sha256']!=r.digest(parity_path):
        raise ValueError('Exact paired baseline diagnostic evidence required')
    if r.digest(cache/'eligible_claims.parquet')!=parity['eligible_claims_sha256'] or r.digest(cache/'baseline_masks.npz')!=parity['baseline_masks_sha256']:
        raise ValueError('Baseline cache changed')
    changes_path=comparison/'changed_claims.parquet'
    if r.digest(changes_path)!=report['changed_claims_sha256']:raise ValueError('Changed claim bytes differ from diagnostic')
    changes=pd.read_parquet(changes_path)
    deltas=changes.loc[changes.baseline_accepted!=changes.gated_accepted]
    if deltas.duplicated(r.PAIR_KEYS).any():raise ValueError('Duplicate accepted pair deltas')
    targets=set(deltas.candidate_entity_id)
    if len(targets)!=report['changed_target_winners']:raise ValueError('Changed winner target coverage differs')
    frame=pd.read_parquet(cache/'eligible_claims.parquet',columns=r.PAIR_KEYS)
    with np.load(cache/'baseline_masks.npz') as masks:baseline_chosen=masks['chosen']
    if len(frame)!=len(baseline_chosen):raise ValueError('Baseline mask alignment mismatch')
    owners={target:set() for target in targets}
    for owner,target in frame.loc[baseline_chosen&frame.candidate_entity_id.isin(targets).to_numpy(),r.PAIR_KEYS].itertuples(index=False,name=None):
        owners[target].add(owner)
    gated={target:set(values) for target,values in owners.items()}
    for owner,target,before,after in deltas[r.PAIR_KEYS+['baseline_accepted','gated_accepted']].itertuples(index=False,name=None):
        if bool(before)!=(owner in owners[target]):raise ValueError('Pair delta conflicts with verified baseline owner list')
        if before:gated[target].remove(owner)
        if after:gated[target].add(owner)
    rows=[{'candidate_entity_id':target,'baseline_owner_ids':sorted(owners[target]),'gated_owner_ids':sorted(gated[target]),
           'removed_owner_ids':sorted(owners[target]-gated[target]),'added_owner_ids':sorted(gated[target]-owners[target]),
           'unchanged_owner_ids':sorted(owners[target]&gated[target])} for target in sorted(targets)]
    columns=['candidate_entity_id','baseline_owner_ids','gated_owner_ids','removed_owner_ids','added_owner_ids','unchanged_owner_ids']
    pd.DataFrame(rows,columns=columns).to_parquet(outputs[1],index=False)
    r.write_json(outputs[0],rows)
    result={'status':'complete','target_rows':len(rows),'accepted_pair_delta_rows':len(deltas),'unchanged_coowners_preserved':True,
            'arbitration_rerun':False,'baseline_parity_sha256':r.digest(parity_path),'diagnostic_sha256':r.digest(report_path),
            'changed_claims_sha256':report['changed_claims_sha256'],'derivation_code_sha256':r.digest(__file__),
            'output_sha256':{path.name:r.digest(path) for path in outputs[:2]}}
    r.write_json(outputs[2],result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cache',type=Path,required=True);p.add_argument('--comparison',type=Path,required=True)
    a=p.parse_args();print(json.dumps(derive(a.cache,a.comparison),indent=2))
