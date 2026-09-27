"""Construct sibling training pairs solely within sealed outer-training owners."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(2**20),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['input','manifest','output']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();m=json.loads(a.manifest.read_text())
    if m['status']!='complete' or m['role']!='outer_train' or m.get('owners_disjoint_development') is not True or sha(a.input)!=m['input_sha256']:
        raise ValueError('Requires sealed outer-training pairs')
    if a.output.exists():raise ValueError('New output required')
    groups=defaultdict(list)
    for line in a.input.open():
        row=json.loads(line);groups[row['source1_entity_id']].append(row)
    a.output.mkdir(parents=True);pairs=0;positive=0;owners=set()
    with (a.output/'pairs.jsonl').open('x') as stream:
        for owner,rows in sorted(groups.items()):
            seeds=sorted((r for r in rows if r['label']==1),key=lambda r:r['candidate_entity_id'])[:2]
            for row in rows:
                other=next((s for s in seeds if s['candidate_entity_id']!=row['candidate_entity_id']),None)
                if other is None:continue
                item={**row,'text_left':other['text_right']}
                stream.write(json.dumps(item,ensure_ascii=False)+'\n');pairs+=1;positive+=int(row['label']);owners.add(owner)
    marker={**m,'rows':pairs,'positive_pairs':positive,'negative_pairs':pairs-positive,
        'input_sha256':sha(a.output/'pairs.jsonl'),'source_input_sha256':sha(a.input),
        'source_manifest_sha256':sha(a.manifest),'code_sha256':sha(__file__),
        'construction':'Within an outer-training business, replace Source1 text by another supplied positive target; keep each candidate original label. No inference seeds or labels read.',
        'training_owners':sorted(owners),'development_labels_read':False,'audit_labels_read':False,'test_records_read':False}
    (a.output/'manifest.json').write_text(json.dumps(marker,indent=2)+'\n')
    print(json.dumps({'status':'complete','rows':pairs,'positives':positive,'negatives':pairs-positive,'owners':len(owners)}))


if __name__=='__main__':main()
