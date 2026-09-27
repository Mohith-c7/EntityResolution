"""Mechanism diagnostic on fixed failed sample's 115 existing missing-address misses.

Apply the already declared two/source quota upstream of learned top20 only in
conditioned error rows. This is not a population retrieval measurement.
"""
import json,sys,time
from pathlib import Path
from dataclasses import replace
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'code/business_entity_resolution'))
import run_frozen_pipeline as runner
from src.blocking.disk_index import normalize_record
from src.blocking.cheap_ranker import rank_features

out=Path(__file__).with_name('result');started=time.perf_counter()
data=json.loads(Path(__file__).with_name('sample.json').read_text())
refs={r['entity_id']:r for r in data['references']}
misses=pd.read_parquet(Path(__file__).with_name('misses.parquet'))
misses=misses[(misses['class']=='missing_address')&misses.source1_entity_id.isin(refs)]
assert len(misses)==115
config=json.loads((ROOT/'models/frozen_v4_checkpoint_repair/frozen.json').read_text())
runner.init(config,ROOT/config['aliases']['path'],str(ROOT/'models/index_train'),ROOT/'models/frozen_v4_checkpoint_repair/replay/universe_counts_train.sqlite')
indexes={i.source:i for i in runner.WORKER[0]};rows=[];lookups=0
for e,part in misses.groupby('source1_entity_id',sort=True):
    ref=normalize_record(refs[e]);name_only=replace(ref,address='',address_tokens=frozenset(),address_digits=frozenset(),postcode_candidates=frozenset())
    for source,group in part.groupby(part.candidate_entity_id.str[:2]):
        ix=indexes[source];lookups+=1
        raw_original={c.candidate_entity_id for c,t in ix.query(ref,return_all=True)}
        pool=ix.query(name_only,return_all=True)
        raw={c.candidate_entity_id for c,t in pool}
        probs=ix.ranking_model.predict(np.asarray([rank_features(name_only,t,ix) for c,t in pool],dtype=np.float32),num_threads=1)
        ordered=sorted(zip(pool,probs),key=lambda p:(-p[1],p[0][0].candidate_entity_id))
        top20={c.candidate_entity_id for (c,t),p in ordered[:20]}
        current=set(data['baseline_candidates'][e])
        eligible=[(c,t,p) for (c,t),p in ordered if not t.address and c.candidate_entity_id not in current and ref.core and t.core and fuzz.token_set_ratio(ref.core,t.core)>=80]
        quota={c.candidate_entity_id for c,t,p in eligible[:2]}
        missing_quota={c.candidate_entity_id for (c,t),p in [p for p in ordered if not p[0][1].address][:2]}
        elig={c.candidate_entity_id for c,t,p in eligible}
        ranks={c.candidate_entity_id:i+1 for i,((c,t),p) in enumerate(ordered)}
        eligible_ranks={c.candidate_entity_id:i+1 for i,(c,t,p) in enumerate(eligible)}
        # Reconstruct the actual name/core postings union before fused/field caps.
        terms=ix.rare_terms('name',ref.name_tokens,ix.postings_config.max_postings,ix.postings_config.tokens_per_field)
        pieces=[ix.postings('name',term)[0] for term in terms]
        for core in (ix.aliases.variants(ref.core) if ix.aliases else [ref.core]):
            if core:pieces.append(ix.postings('core',core)[0])
        ids=np.unique(np.concatenate(pieces)) if pieces else np.empty(0,dtype=np.int32)
        empty_targets=[]
        for offset in range(0,len(ids),500):
            chunk=[int(v) for v in ids[offset:offset+500]]
            empty_targets.extend(ix.record_values(row) for row in ix.connection.execute(
                'SELECT id,name,core,address,country,name_digits,address_digits,postcodes,folded FROM records WHERE address=\'\' AND rowid IN ('+','.join('?' for v in chunk)+')',chunk))
        all_raw_empty={t.entity_id for t in empty_targets}
        quota_targets=[t for t in empty_targets if t.entity_id not in current and ref.core and t.core and fuzz.token_set_ratio(ref.core,t.core)>=80]
        ix.prefetch_frequencies([name_only,*quota_targets])
        quota_prob=ix.ranking_model.predict(np.asarray([rank_features(name_only,t,ix) for t in quota_targets],dtype=np.float32),num_threads=1) if quota_targets else []
        full_quota_order=sorted(zip(quota_targets,quota_prob),key=lambda p:(-p[1],p[0].entity_id))
        full_quota={t.entity_id for t,p in full_quota_order[:2]}
        full_eligible={t.entity_id for t,p in full_quota_order}
        for row in group.itertuples(index=False):
            cid=row.candidate_entity_id
            rows.append({'source1_entity_id':e,'candidate_entity_id':cid,'country':row.country,
                'target_address_missing':not bool(row.candidate_address),
                'in_original_direct_raw_pool':cid in raw_original,'in_name_only_raw_pool':cid in raw,
                'in_name_only_learned_top20':cid in top20,'eligible_in_name_only_raw_pool':cid in elig,
                'name_only_full_pool_learned_rank':ranks.get(cid,0),
                'name_only_eligible_missing_address_rank':eligible_ranks.get(cid,0),
                'in_existing_two_quota_before_top20':cid in quota,
                'in_unfiltered_missing_address_two_quota':cid in missing_quota,
                'name_only_raw_pool_size':len(pool),'eligible_missing_address_pool_size':len(eligible),
                'in_actual_name_core_postings_union':cid in all_raw_empty,
                'eligible_in_actual_postings_union':cid in full_eligible,
                'in_existing_two_quota_before_fused_field_caps':cid in full_quota,
                'actual_name_core_postings_union_size':len(ids),'actual_missing_address_postings_pool_size':len(empty_targets),
                'actual_eligible_missing_address_pool_size':len(quota_targets)})
frame=pd.DataFrame(rows);frame.to_parquet(out/'missing_address_quota_diagnostic.parquet',index=False)
def counts(f):
    return {'links':len(f),**{k:int(f[k].sum()) for k in ['target_address_missing','in_original_direct_raw_pool','in_name_only_raw_pool','in_name_only_learned_top20','eligible_in_name_only_raw_pool','in_existing_two_quota_before_top20','in_unfiltered_missing_address_two_quota','in_actual_name_core_postings_union','eligible_in_actual_postings_union','in_existing_two_quota_before_fused_field_caps']}}
report={'scope':'conditioned mechanism diagnosis on115 already exposed misses; no second population variant or parameter search','references':misses.source1_entity_id.nunique(),'reference_source_lookups':lookups,'frozen_spec':'Same core token_set>=80 and2 new/source quota inspected before learned top20 and before fused/field caps','counts':counts(frame),'country_counts':{c:counts(f) for c,f in frame.groupby('country')},'seconds':time.perf_counter()-started,'limitation':'Original/name-only bounded pools are after fused128/field24, before learned top20. Actual name/core postings union is before these caps, retaining existing token-list100000 cap and six rare terms. All115 targets have empty addresses, so address/digit postings cannot add them beyond this same name/core union. Quota totals are conditioned on known misses and cannot estimate population candidate counts, losses, or oracle/runtime gains. Current failed5k report unchanged.'}
(out/'missing_address_quota_diagnostic.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
