"""Independently validate target-only masking of the sealed training curriculum."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda:stream.read(8*1024*1024),b''):h.update(data)
    return h.hexdigest()


def rows(path):
    with path.open() as stream:return [json.loads(line) for line in stream]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--masked',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--original',type=Path,default=Path('models/sprint_6h/neural/train120k.jsonl'))
    p.add_argument('--street',type=Path,default=Path('models/sprint_6h/neural/train120k_street_decoys.jsonl'))
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    if sha(a.original)!='25b5904fca2d1a6b88754b34adab52cbc2889fc3327eba13cb76d83135ed0ad6':raise ValueError('Original training bytes changed')
    if sha(a.street)!='e2e232a81dd3b5d041e5a08b00f6cc53c03ed73d909d6782894e59498aa8e1eb':raise ValueError('Street curriculum bytes changed')
    original,street,masked=[rows(path) for path in (a.original,a.street,a.masked)]
    if len(original)!=120000 or len(street)!=120000 or len(masked)!=120000:raise ValueError('Row count changed')
    changes=Counter();missing=Counter();labels=Counter();owners=set();decoys=masked_decoys=0
    for raw,before,after in zip(original,street,masked):
        for key in ('source1_entity_id','candidate_entity_id','label'):
            if raw[key]!=before[key] or before[key]!=after[key]:raise ValueError('Training row key/order/label changed')
        if before['text_left']!=after['text_left']:raise ValueError('Source1 text changed')
        altered=raw['text_left']!=before['text_left'] or raw['text_right']!=before['text_right']
        decoys+=altered;owners.add(after['source1_entity_id']);label=int(after['label']);labels[label]+=1
        left,prefix_right=after['text_left'],after['text_right']
        if not left.rpartition('address:')[2].strip():raise ValueError('Unexpected missing Source1 training address')
        missing[label]+=not prefix_right.rpartition('address:')[2].strip()
        if before['text_right']!=after['text_right']:
            if altered:masked_decoys+=1
            pre,marker,address=before['text_right'].rpartition('address:')
            post,post_marker,post_address=after['text_right'].rpartition('address:')
            if not marker or not post_marker or not address.strip() or pre!=post or post_address.strip():raise ValueError('Not originally nonempty target-only address removal')
            changes[label]+=1
    if dict(changes)!={0:10000,1:10000} or masked_decoys or decoys!=7338:raise ValueError('Wrong balanced mask or decoy exclusions')
    if dict(labels)!={0:52990,1:67010} or dict(missing)!={0:10122,1:11969} or len(owners)!=20000:raise ValueError('Population or missing-address counts differ')
    manifest=json.loads(a.manifest.read_text())
    if manifest['input_sha256']!=sha(a.masked) or manifest['rows']!=120000:raise ValueError('Transform manifest seal differs')
    out={'status':'independent_target_address_dropout_validation_passed','rows':120000,'training_owners':20000,'owner_keys_labels_order_preserved':True,'source1_text_preserved':True,'source1_missing_addresses':0,'masked_target_pairs_by_label':dict(changes),'target_missing_total_by_label':dict(missing),'street_decoy_pairs_excluded':decoys,'street_decoy_pairs_changed':masked_decoys,'labels_outer_training_only':True,'input_sha256':{'original':sha(a.original),'street':sha(a.street),'masked':sha(a.masked),'manifest':sha(a.manifest)},'source_sha256':sha(Path(__file__))}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))


if __name__=='__main__':main()
