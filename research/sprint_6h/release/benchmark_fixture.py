#!/usr/bin/env python3
"""Synthetic release mechanics timings; these are not full-test ETA evidence."""
import argparse
import importlib.util
import json
from pathlib import Path
import resource
import tempfile
import time

import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
spec=importlib.util.spec_from_file_location("release_builder",ROOT / "scripts/build_gated_submission.py")
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output",type=Path,required=True)
parser.add_argument("--sizes",type=int,nargs="+",default=[20000,30000])
args=parser.parse_args()
blocks=[]
for size in args.sizes:
    start=time.perf_counter()
    rows=[{"entity_id":f"S1-{i}","country":("france","india","us")[i%3]} for i in range(size)]
    data=[]
    for i in range(size):
        for rank,prob in enumerate((.9 if i%2 else .91,.72,.4,.1,.01)):
            target=f"S2-shared-{i//2}" if rank==0 else f"S3-{i}-{rank}"
            data.append((f"S1-{i}",target,prob,prob,prob,rank))
    frame=pd.DataFrame(data,columns=r.PAIR_KEYS+r.SCORE_COLUMNS+["candidate_order"])
    prepared=time.perf_counter()
    gates=frame[r.PAIR_KEYS].copy();gates["gate_veto"]=False;gates["gate_reason"]="fixture"
    frame=r.attach_vetoes(frame,gates.iloc[::-1],r.pair_order_digest(frame))
    gated=time.perf_counter()
    chosen,lost,details=r.make_decisions(frame,{"policy":"ownership_v2","decision_config":{"t_first":.7,"t_rest":.83}})
    decided=time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="r1-fixture-") as directory:
        directory=Path(directory);test=directory / "test";test.mkdir();output=directory / "output";output.mkdir()
        pd.DataFrame(rows).to_csv(test / "test_source1.tsv",sep="\t",index=False)
        for source in (2,3):
            ids=frame.loc[frame.candidate_entity_id.str.startswith(f"S{source}-"),"candidate_entity_id"].drop_duplicates()
            pd.DataFrame({"entity_id":ids}).to_csv(test / f"test_source{source}.tsv",sep="\t",index=False)
        r.export_outputs(rows,frame,chosen,output)
        export_end=time.perf_counter()
        validation=r.validate_outputs(output,test)
    end=time.perf_counter()
    assert validation["strict"]==validation["official"]=="PASS"
    blocks.append({"references":size,"pairs":len(frame),"prepare_seconds":prepared-start,"gate_alignment_seconds":gated-prepared,
                   "global_decision_seconds":decided-gated,"input_fixture_and_export_seconds":export_end-decided,
                   "both_validator_seconds":end-export_end,"total_seconds":end-start,"decision":details,
                   "fixture_mechanics_references_per_second":size/(end-prepared),"validation":{"strict":"PASS","official":"PASS","id_checking":True}})
report={"kind":"synthetic_fixture_runtime","full_test_eta_supported":False,"blocks":blocks,
        "peak_rss_platform_units":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "code_sha256":{str(path.relative_to(ROOT)):r.digest(path) for path in (ROOT / "scripts/build_gated_submission.py",ROOT / "scripts/decision_policy_v2.py")}}
r.write_json(args.output,report)
print(json.dumps(report,indent=2))
