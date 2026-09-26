"""Compare retrieval on a fixed tuning sample against complete target indexes."""

import argparse
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"code/business_entity_resolution"))
from src.blocking.anchor_index import build_anchor_index
from src.blocking.disk_index import DiskSearchConfig,DiskSourceIndex,normalize_record
from src.evaluation.metrics import score_matches
from src.model.name_aliases import NameAliases


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir",type=Path,required=True)
    parser.add_argument("--index-dir",type=Path,default=Path("models"))
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--entities",type=int,default=100)
    parser.add_argument("--retrieval-mode",choices=("wide","anchored"),default="anchored")
    parser.add_argument("--top-k",type=int,default=20)
    parser.add_argument("--path-top-k",type=int,default=80)
    args=parser.parse_args()
    if args.entities<1:
        parser.error("--entities must be positive")
    report=json.loads((args.artifact_dir/"report.json").read_text())
    config=replace(DiskSearchConfig(**report["search_config"]),retrieval_mode=args.retrieval_mode,
        top_k=args.top_k,path_top_k=args.path_top_k)
    aliases=NameAliases.load(args.artifact_dir/"name_aliases.json") if config.use_name_aliases else None
    references=json.loads((args.artifact_dir/"sampled_references.json").read_text())["tune"][:args.entities]
    truth=json.loads((args.artifact_dir/"sampled_truth.json").read_text())
    truth={r["entity_id"]:set(truth[r["entity_id"]]) for r in references}
    indexes=[]
    try:
        for source in ("S2","S3"):
            path=args.index_dir/f"index_train_{source}.sqlite"
            if config.retrieval_mode=="anchored":
                build_anchor_index(path,aliases)
            indexes.append(DiskSourceIndex(path,config,aliases=aliases))
        start=time.perf_counter()
        covered,countries,counts={}, {}, []
        for number,raw in enumerate(references,1):
            ref=normalize_record(raw)
            pairs=[p for idx in indexes for p in idx.query(ref)]
            ids={c.candidate_entity_id for c,t in pairs}
            covered[ref.entity_id]=truth[ref.entity_id]&ids
            countries[ref.entity_id]=ref.country
            counts.append(len(ids))
            if number%100==0:
                print(json.dumps({"references":number,"seconds":round(time.perf_counter()-start,1)}),flush=True)
        result={"policy":"Tuning-only retrieval diagnostic. Oracle assumes perfect matching of retrieved positives; it is not a model score.",
            "config":asdict(config),"target_pool":{idx.source:idx.record_count for idx in indexes},
            "oracle":score_matches(truth,covered,countries),"mean_candidates":sum(counts)/len(counts),
            "seconds":time.perf_counter()-start,"missing_links":{eid:sorted(truth[eid]-covered[eid]) for eid in covered if truth[eid]-covered[eid]}}
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2)+"\n")
        print(json.dumps({k:v for k,v in result.items() if k!="missing_links"}),flush=True)
    finally:
        for index in indexes:
            index.close()


if __name__=="__main__":
    main()
