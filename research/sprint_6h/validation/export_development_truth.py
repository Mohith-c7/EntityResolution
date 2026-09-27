"""Export only already-exposed tune-owner labels and their supplied target records."""
import argparse
import json
import sqlite3
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'scripts'))
sys.path.insert(0,str(ROOT/'code/business_entity_resolution'))
from export_sprint_workbench import reference_rows,sha
from src.evaluation.validation import entity_fold


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--references',type=Path,required=True);p.add_argument('--exposed',type=Path,required=True)
    p.add_argument('--owners',type=Path,default=Path('models/scale_v1_plan/aliases/counts.sqlite'))
    p.add_argument('--index-prefix',default='models/index_train');p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();rows=reference_rows(args.references);ids=[r['entity_id'] for r in rows]
    exposed=set(json.loads(args.exposed.read_text()))
    if not set(ids)<=exposed:raise ValueError('Requested IDs include unexposed references; no labels opened')
    if any(entity_fold(e)!='tune' for e in ids):raise ValueError('This exporter accepts only the consumed tune cohort, never fresh outer holdout')
    if args.output.exists():raise FileExistsError(args.output)
    truth={e:[] for e in ids}
    with sqlite3.connect(args.owners.resolve().as_uri()+'?mode=ro',uri=True) as conn:
        for start in range(0,len(ids),500):
            part=ids[start:start+500]
            for target,owner in conn.execute('SELECT id,owner FROM owners WHERE owner IN ('+','.join('?'*len(part))+')',part):truth[owner].append(target)
    targets=[];known={t for values in truth.values() for t in values}
    for source in ('S2','S3'):
        wanted=sorted(t for t in known if t.startswith(source+'-'))
        with sqlite3.connect(Path(f'{args.index_prefix}_{source}.sqlite').resolve().as_uri()+'?mode=ro',uri=True) as conn:
            for start in range(0,len(wanted),500):
                part=wanted[start:start+500]
                for eid,name,address,country in conn.execute('SELECT id,name,address,country FROM records WHERE id IN ('+','.join('?'*len(part))+')',part):
                    targets.append({'entity_id':eid,'business_name':name,'business_address':address,'country':country})
    if len(targets)!=len(known):raise ValueError('Missing truth target records in supplied index')
    args.output.mkdir(parents=True)
    (args.output/'truth.json').write_text(json.dumps({e:sorted(v) for e,v in truth.items()})+'\n')
    (args.output/'target_records.json').write_text(json.dumps(targets,ensure_ascii=False)+'\n')
    report={'status':'already_exposed_development_truth_exported','references':len(ids),'truth_links':len(known),
            'input_hashes':{'references':sha(args.references),'exposure_ids':sha(args.exposed)},
            'scope':'Consumed outer-tune owners only; no fresh confirmation/audit labels opened',
            'target_records_source':'models/index_train_{S2,S3}.sqlite records table; normalized names/addresses, punctuation/case not preserved',
            'target_records_raw':False,
            'outputs':{n:sha(args.output/n) for n in ['truth.json','target_records.json']}}
    (args.output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
