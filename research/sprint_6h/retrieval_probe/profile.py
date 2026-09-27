"""Read-only exposed-development candidate residual inventory and fixed sample."""
import json, hashlib, sqlite3, sys, time
from pathlib import Path
from collections import Counter, defaultdict
import pandas as pd
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'code/business_entity_resolution'))
from src.blocking.disk_index import normalize_record
from src.evaluation.metrics import score_matches

out=Path(__file__).parent
dev=ROOT/'models/sprint_6h/development50k'
bench=ROOT/'research/sprint_6h/sibling/workbenches50k'
refs={r['entity_id']:r for r in json.loads((dev/'references.json').read_text())}
truth=json.loads((bench/'truth50k.json').read_text())
countries=json.loads((bench/'countries50k.json').read_text())
start=time.perf_counter()
frame=pd.read_parquet(dev/'pairs.parquet',columns=['source1_entity_id','candidate_entity_id'])
cands=frame.groupby('source1_entity_id',sort=False).candidate_entity_id.agg(set).to_dict()
misses=[(e,c) for e in sorted(refs) for c in sorted(set(truth[e])-cands.get(e,set()))]
cons={s:sqlite3.connect((ROOT/f'models/index_train_{s}.sqlite').resolve().as_uri()+'?mode=ro',uri=True) for s in ['S2','S3']}
rows=[]
for e,c in misses:
 r=normalize_record(refs[e]); row=cons[c[:2]].execute('SELECT name,address,country FROM records WHERE id=?',(c,)).fetchone()
 t=normalize_record({'entity_id':c,'business_name':row[0],'business_address':row[1],'country':row[2]})
 shared_name=len(r.name_tokens&t.name_tokens);shared_address=len(r.address_tokens&t.address_tokens)
 nonascii=any(x.isalpha() and not x.isascii() for x in r.name+t.name)
 if not t.address or not r.address: cls='missing_address'
 elif not shared_name and nonascii: cls='script_changed_name'
 elif not shared_name and shared_address: cls='address_led_name_changed'
 elif shared_name: cls='name_overlap_rank_or_shortlist'
 else: cls='no_token_overlap'
 rows.append({'source1_entity_id':e,'candidate_entity_id':c,'country':r.country,'class':cls,'name_overlap':shared_name,'address_overlap':shared_address,'nonascii_name':nonascii,'reference_name':r.name,'candidate_name':t.name,'reference_address':r.address,'candidate_address':t.address})
pd.DataFrame(rows).to_parquet(out/'misses.parquet',index=False)
# Fixed 10% country-stratified sample, deterministic content-independent hash.
selected=[]
for country in sorted(set(countries.values())):
 ids=[e for e in refs if countries[e]==country]
 selected+=sorted(ids,key=lambda e:hashlib.sha256(('retrieval-probe-v1:'+e).encode()).digest())[:max(1,round(len(ids)*.1))]
selected=sorted(selected)
sample={'references':[refs[e] for e in selected],'truth':{e:truth[e] for e in selected},'countries':{e:countries[e] for e in selected},'baseline_candidates':{e:sorted(cands.get(e,set())) for e in selected}}
(out/'sample.json').write_text(json.dumps(sample))
counts=np.array([len(cands.get(e,set())) for e in refs])
report={'scope':'already_exposed_development_only','entities':len(refs),'missed_links':len(misses),'affected_references':len(set(e for e,c in misses)),'dominant_classes':dict(Counter(r['class'] for r in rows)),'country_misses':dict(Counter(r['country'] for r in rows)),'baseline_oracle':score_matches(truth,{e:set(truth[e])&cands.get(e,set()) for e in refs},countries),'counts':{'mean':float(counts.mean()),'p95':float(np.quantile(counts,.95)),'max':int(counts.max())},'sample_entities':len(selected),'sample_ids_sha256':hashlib.sha256('\n'.join(selected).encode()).hexdigest(),'seconds':time.perf_counter()-start}
(out/'profile.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report),flush=True)
