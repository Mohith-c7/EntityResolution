"""Post-run description of 14 recovered exposed-development links only."""
import json,sys
from pathlib import Path
from dataclasses import replace
import pandas as pd
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'code/business_entity_resolution'))
import run_frozen_pipeline as runner
from src.blocking.disk_index import normalize_record
out=Path(__file__).with_name('result')
data=json.loads(Path(__file__).with_name('sample.json').read_text())
refs={r['entity_id']:r for r in data['references']}
config=json.loads((ROOT/'models/frozen_v4_checkpoint_repair/frozen.json').read_text())
runner.init(config,ROOT/config['aliases']['path'],str(ROOT/'models/index_train'),ROOT/'models/frozen_v4_checkpoint_repair/replay/universe_counts_train.sqlite')
indexes={i.source:i for i in runner.WORKER[0]}
new=pd.read_parquet(out/'new_true_links.parquet');rows=[]
for e,part in new.groupby('source1_entity_id',sort=True):
    ref=normalize_record(refs[e]);name_only=replace(ref,address='',address_tokens=frozenset(),address_digits=frozenset(),postcode_candidates=frozenset())
    for source,group in part.groupby(part.candidate_entity_id.str[:2]):
        ix=indexes[source]
        raw_original={c.candidate_entity_id for c,t in ix.query(ref,return_all=True)}
        raw_name_only={c.candidate_entity_id for c,t in ix.query(name_only,return_all=True)}
        for cid in group.candidate_entity_id:
            rows.append({'source1_entity_id':e,'candidate_entity_id':cid,'present_in_original_direct_raw_shortlist':cid in raw_original,'present_in_name_only_raw_shortlist':cid in raw_name_only})
frame=pd.DataFrame(rows);frame.to_parquet(out/'new_true_shortlist_diagnostic.parquet',index=False)
report={'scope':'post-run characterization of recovered links only; not a second retrieval variant','new_true_links':len(frame),'already_in_original_direct_raw_shortlist':int(frame.present_in_original_direct_raw_shortlist.sum()),'absent_from_original_direct_raw_shortlist':int((~frame.present_in_original_direct_raw_shortlist).sum()),'in_name_only_raw_shortlist':int(frame.present_in_name_only_raw_shortlist.sum()),'limitation':'Original-direct shortlist is before learned rerank. Bridge raw shortlists are not replayed; this does not identify the full never-generated class or overall ranker loss.'}
(out/'new_true_shortlist_diagnostic.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
