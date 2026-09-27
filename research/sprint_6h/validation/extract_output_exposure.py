"""Stream Source 1 IDs from prior output TSVs; verify matching/candidate order."""
import argparse
import hashlib
import json
from pathlib import Path


def inspect(path, collect=False):
    before=path.stat(); filehash=hashlib.sha256(); orderhash=hashlib.sha256(); ids=set(); count=0
    with path.open('rb') as stream:
        header=stream.readline();filehash.update(header)
        if not header.startswith(b'source1_entity_id\t'):raise ValueError('Unexpected output header')
        for line in stream:
            filehash.update(line); eid=line.split(b'\t',1)[0]
            if not eid.startswith(b'S1-'):raise ValueError('Invalid output reference ID')
            orderhash.update(eid+b'\n');count+=1
            if collect:ids.add(eid.decode())
    after=path.stat()
    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError('Output changed during scan')
    return ids,{'path':str(path),'sha256':filehash.hexdigest(),'source1_order_sha256':orderhash.hexdigest(),'rows':count,'reference_ids':len(ids) if collect else count}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--matching',type=Path,required=True);p.add_argument('--candidates',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    ids,a=inspect(args.matching,True);_,b=inspect(args.candidates)
    if a['rows']!=b['rows'] or a['source1_order_sha256']!=b['source1_order_sha256']:raise ValueError('Output reference universes/order differ')
    if len(ids)!=a['rows']:raise ValueError('Duplicate matching output reference IDs')
    args.output.mkdir(parents=True,exist_ok=False);(args.output/'exposed_ids.json').write_text(json.dumps(sorted(ids))+'\n')
    report={'status':'prior_submission_source_ids_accounted','references':len(ids),'sources':[a,b],'labels_read':False}
    (args.output/'ledger.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
