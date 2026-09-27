"""Compare complete matching TSVs to supplied S1 country records, no labels."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'research/sprint_6h'))
from prepare_neural_inputs import sha

def changes(old,new):
    a=old.split(',') if old else [];b=new.split(',') if new else []
    if len(set(a))!=len(a) or len(set(b))!=len(b):raise ValueError('Repeated matched target in one owner')
    return a,b,{'owners':1,'changed_owners':int(set(a)!=set(b)),'old_predicted_links':len(a),'new_predicted_links':len(b),
        'added_links':len(set(b)-set(a)),'removed_links':len(set(a)-set(b)),
        'old_empty_predictions':int(not a),'new_empty_predictions':int(not b),
        'empty_to_nonempty':int(not a and bool(b)),'nonempty_to_empty':int(bool(a) and not b)}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,default=Path('output/submission_05'))
    p.add_argument('--candidate',type=Path,default=Path('output/submission_06'))
    p.add_argument('--source1',type=Path,default=Path('dataset/test/test_source1.tsv'))
    a=p.parse_args();reports={}
    for kind,directory in [('baseline',a.baseline),('candidate',a.candidate)]:
        r=json.loads((directory/'submission_report.json').read_text())
        if r['status']!='complete' or r['validation']['strict']!='PASS' or r['validation']['official']!='PASS' or sha(directory/'matching_results.tsv')!=r['files_sha256']['matching_results.tsv']:raise ValueError('Complete matching TSV not verified')
        reports[kind]=r
    totals=Counter();countries={};oldtargets=Counter();newtargets=Counter()
    with a.source1.open(encoding='utf-8-sig',newline='') as raw,(a.baseline/'matching_results.tsv').open(newline='') as old,(a.candidate/'matching_results.tsv').open(newline='') as new:
        source=csv.DictReader(raw,delimiter='\t');before=csv.DictReader(old,delimiter='\t');after=csv.DictReader(new,delimiter='\t')
        for row in source:
            x=next(before,None);y=next(after,None)
            if x is None or y is None or row['entity_id']!=x['source1_entity_id'] or row['entity_id']!=y['source1_entity_id']:raise ValueError('Global source1 TSV order/coverage differs')
            aa,bb,delta=changes(x['matched_entity_ids'],y['matched_entity_ids'])
            totals.update(delta);countries.setdefault(row['country'].strip().casefold(),Counter()).update(delta)
            oldtargets.update(aa);newtargets.update(bb)
        if next(before,None) is not None or next(after,None) is not None:raise ValueError('Unexpected extra matching owners')
    if totals['owners']!=1732544:raise ValueError('Incomplete official TEST owners')
    collisions={'baseline_collision_targets':sum(v>1 for v in oldtargets.values()),'candidate_collision_targets':sum(v>1 for v in newtargets.values()),
        'baseline_excess_claims':sum(max(0,v-1) for v in oldtargets.values()),'candidate_excess_claims':sum(max(0,v-1) for v in newtargets.values())}
    if collisions['candidate_collision_targets']:raise ValueError('Selected-only exclusivity failed')
    result={'status':'complete_label_free_matching_comparison','baseline_matching_sha256':reports['baseline']['files_sha256']['matching_results.tsv'],
        'candidate_matching_sha256':reports['candidate']['files_sha256']['matching_results.tsv'],'source1_sha256':sha(a.source1),
        'global':dict(totals),'country':{k:dict(v) for k,v in countries.items()},'ownership':collisions,
        'actual_precision_recall_f05_unavailable_without_test_labels':True,'portal_gain_not_inferred_from_link_counts':True,
        'labels_read':False,'audit_labels_read':False,'source_sha256':sha(__file__)}
    output=a.candidate/'label_free_vs05.json'
    if output.exists():raise ValueError('Do not overwrite existing output comparison')
    output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':main()
