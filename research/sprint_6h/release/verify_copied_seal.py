#!/usr/bin/env python3
"""Verify a copied baseline against immutable seal byte evidence, one CPU.

No score recomputation, model prediction, retrieval, or baseline writes occur.
Original source seal and a separately derived portable seal are both retained.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time


def digest(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):result.update(block)
    return result.hexdigest()


def inspect(manifest_path,parent_path,root,output):
    manifest_path,parent_path,root,output=map(Path,(manifest_path,parent_path,root,output))
    if output.exists():raise ValueError('Copy verification output must be new/versioned')
    start=time.perf_counter()
    manifest=json.loads(manifest_path.read_text())
    parent=json.loads(parent_path.read_text())
    if manifest.get('version')!='stored-score-release-v1':raise ValueError('Unsupported reuse seal')
    if manifest.get('parent_manifest_sha256')!=digest(parent_path):raise ValueError('Parent seal hash mismatch')
    def rebase(value):
        if isinstance(value,dict):
            result={k:rebase(v) for k,v in value.items()}
            if isinstance(result.get('path'),str) and result['path'].startswith('/mnt/er/entity/'):
                result['path']=result['path'][len('/mnt/er/entity/'):]
            return result
        if isinstance(value,list):return [rebase(v) for v in value]
        return value
    expected=rebase(parent)
    for old,new in manifest.get('path_rebindings',{}).items():
        if old!='models/frozen_v4_checkpoint_repair/frozen.json' or new!='research/sprint_6h/coordination/baseline_freeze/frozen.json':
            raise ValueError('Unrecognized frozen proof path rebinding')
        if expected['baseline_frozen']['path']==old:expected['baseline_frozen']['path']=new
    actual={k:v for k,v in manifest.items() if k not in ('parent_manifest_sha256','evidence_path_binding','path_rebindings')}
    if actual!=expected:raise ValueError('Portable seal changed immutable byte/score/order evidence')
    checked=0
    def verify(evidence):
        nonlocal checked
        path=Path(evidence['path'])
        if path.is_absolute():raise ValueError('Copied seal must use root-relative evidence paths')
        path=(root/path).resolve()
        if root.resolve() not in path.parents:raise ValueError('Evidence path escapes isolated root')
        if digest(path)!=evidence['sha256']:raise ValueError('Copied baseline hash mismatch: '+evidence['path'])
        checked+=1
        return path
    frozen=verify(manifest['baseline_frozen'])
    report_path=verify(manifest['baseline_report'])
    report=json.loads(report_path.read_text())
    if (report.get('status')!='complete' or report.get('validation',{}).get('strict_issues')!=[]
        or report.get('validation',{}).get('official_returncode')!=0 or report.get('frozen_sha256')!=digest(frozen)
        or report.get('entities')!=manifest['entities'] or report.get('scored_candidates')!=manifest['pairs']):
        raise ValueError('Copied report is not the sealed, complete validated baseline')
    for evidence in manifest['assets']:verify(evidence)
    for evidence in manifest['test_files'].values():verify(evidence)
    verify(manifest['source1_universe'])
    for number,chunk in enumerate(manifest['chunks']):
        if number!=chunk['chunk']:raise ValueError('Noncontiguous copied chunks')
        verify(chunk['marker']);verify(chunk['parquet'])
        if (number+1)%1000==0:print(json.dumps({'verified_chunks':number+1,'of':len(manifest['chunks'])}),flush=True)
    directory=report_path.parent
    for name,expected_hash in report['output_sha256'].items():
        if digest(directory/name)!=expected_hash:raise ValueError('Copied output hash mismatch: '+name)
        checked+=1
    result={'status':'complete','copy_byte_parity':True,'entities':manifest['entities'],'pairs':manifest['pairs'],
            'chunks':len(manifest['chunks']),'checked_files':checked,'pair_order_sha256':manifest['pair_order_sha256'],
            'portable_manifest_sha256':digest(manifest_path),'original_manifest_sha256':digest(parent_path),
            'baseline_frozen_sha256':digest(frozen),'output_sha256':report['output_sha256'],
            'validation_basis':'Exact byte/hash parity with sealed baseline validated by strict and official --check-ids; no validators rerun on copy.',
            'validators_rerun':False,'elapsed_seconds':time.perf_counter()-start}
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary=output.with_name(output.name+'.partial')
    temporary.write_text(json.dumps(result,indent=2)+'\n');temporary.replace(output)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--parent-manifest',type=Path,required=True)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(inspect(args.manifest,args.parent_manifest,args.root,args.output),indent=2))
