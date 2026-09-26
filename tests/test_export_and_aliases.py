"""Adversarial output checks and train-only learned normalization contracts."""

import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

import lightgbm as lgb
import pandas as pd
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"code/business_entity_resolution"))
from src.blocking.contracts import Candidate
from src.blocking.disk_index import DiskSearchConfig,normalize_record
from src.blocking.disk_index import DiskSourceIndex,build_disk_index
from src.blocking.anchor_index import build_anchor_index
from src.evaluation.validation import entity_fold
from src.features.pairwise_features import build_pair_features
from src.features.registry import FEATURE_NAMES,FEATURE_VERSION
from src.model.name_aliases import NameAliases,fit_name_aliases
from src.pipeline.export import validate_submission
from src.pipeline.inference import run_inference


@pytest.mark.parametrize("matching,candidates,s2,expected",[
    ("S1-A\tS2-A\nS1-B\t\n","S1-B\t\nS1-A\tS2-A,S3-A\n",{"S2-A"},True),
    ("S1-A\nS1-B\t\n","S1-A\t\nS1-B\t\n",{"S2-A"},False),
    ("S1-A\t,S2-A,,\nS1-B\t\n","S1-A\tS2-A\nS1-B\t\n",{"S2-A"},False),
    ("S1-A\tS2-ghost\nS1-B\t\n","S1-A\tS2-ghost\nS1-B\t\n",set(),False),
    ("S1-A\tS1-B\nS1-B\t\n","S1-A\tS1-B\nS1-B\t\n",{"S2-A"},False),
    ("S1-A\tS2-A,S2-A\nS1-B\t\n","S1-A\tS2-A\nS1-B\t\n",{"S2-A"},False),
    ("S1-A\tS2-A\nS1-B\t\n","S1-A\t\nS1-B\t\n",{"S2-A"},False),
    ("S1-A\t\n","S1-A\t\nS1-B\t\n",{"S2-A"},False),
    ("S1-A\t\nS1-A\t\nS1-B\t\n","S1-A\t\nS1-B\t\n",{"S2-A"},False),
])
def test_strict_output_checks(tmp_path,matching,candidates,s2,expected):
    m,c=tmp_path/"matching_results.tsv",tmp_path/"candidate_pairs.tsv"
    m.write_text("source1_entity_id\tmatched_entity_ids\n"+matching)
    c.write_text("source1_entity_id\tcandidate_entity_ids\n"+candidates)
    errors=validate_submission(m,c,test_s1_ids={"S1-A","S1-B"},test_s2_ids=s2,test_s3_ids={"S3-A"})
    assert (not errors)==expected


def test_name_channel_excludes_heldout_counterparts_and_preserves_ambiguity(tmp_path):
    train=next(f"S1-{n}" for n in range(100) if entity_fold(f"S1-{n}")=="train")
    heldout=next(f"S1-{n}" for n in range(100) if entity_fold(f"S1-{n}")=="holdout")
    header="entity_id\tbusiness_name\tbusiness_address\tcountry\n"
    (tmp_path/"train_source1.tsv").write_text(header+f"{train}\tAlpha Ltd\t123 Main Street\tNewlandia\n{heldout}\tBeta Ltd\t456 Other Street\tFrance\n")
    (tmp_path/"train_source2.tsv").write_text(header+"S2-A\tLearned Variant\t123 Main Street\tNewlandia\nS2-B\tLearned Variant\t123 Main Street\tNewlandia\nS2-C\tHeldout Variant\t456 Other Street\tFrance\nS2-D\tHeldout Variant\t456 Other Street\tFrance\n")
    (tmp_path/"train_source3.tsv").write_text(header)
    (tmp_path/"train_ground_truth.tsv").write_text("source1_entity_id\tmatched_entity_ids\n"+f"{train}\tS2-A,S2-B\n{heldout}\tS2-C,S2-D\n")
    output=tmp_path/"aliases.json"
    aliases=fit_name_aliases(tmp_path,output)
    assert aliases.resolve("learned variant")[0]=="alpha"
    assert aliases.resolve("heldout variant")[0]=="heldout variant"
    assert aliases.metadata["fitted_targets"]==2
    assert train not in output.read_text() and heldout not in output.read_text()
    ambiguity=NameAliases({},aliases.metadata,{"shared alias":[["alpha",3,5],["beta",2,5]]})
    assert ambiguity.resolve("shared alias")[0]=="shared alias"
    assert ambiguity.evidence("shared alias","beta")==(.4,2)
    assert "shared alias" in ambiguity.variants("beta")
    (tmp_path/"train_source2.tsv").write_text(header)
    with pytest.raises(ValueError):
        fit_name_aliases(tmp_path,output)


def test_compound_retrieval_learned_name_numbers_and_open_country(tmp_path):
    source=tmp_path/"source.tsv"
    source.write_text("entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        "S2-A\tNom Alternatif\t123 Rue Mozart 75001\tFrance\n"
        "S2-B\tCafe Lumiere\t456 Rue Mozart 75001\tFrance\n"
        "S2-C\tCafe Lumiere\t123 Rue Mozart 75001\tNewlandia\n")
    path=tmp_path/"index.sqlite"
    build_disk_index(source,path,"S2")
    aliases=NameAliases({"nom alternatif":["cafe lumiere",3,3]}, {})
    saved=build_anchor_index(path,aliases)
    assert build_anchor_index(path,aliases)==saved
    cfg=DiskSearchConfig(top_k=1,path_top_k=5,character_mode="off",reranker_version="v4",
        use_name_aliases=True,retrieval_mode="anchored")
    index=DiskSourceIndex(path,cfg,aliases=aliases)
    ref=normalize_record({"entity_id":"S1-A","business_name":"Cafe Lumiere",
        "business_address":"00123 Rue Mozart 75001","country":"France"})
    pairs=index.query(ref)
    assert pairs[0][0].candidate_entity_id=="S2-A"
    assert "family_numeric" in pairs[0][0].blocking_paths
    assert "S2-C" not in {candidate.candidate_entity_id for candidate,target in index.query(ref,return_all=True)}
    index.close()
    with pytest.raises(ValueError,match="fingerprint"):
        build_anchor_index(path,NameAliases({"nom alternatif":["other business",3,3]}, {}))


@pytest.mark.parametrize("workers",[1,2])
def test_batched_inference_exports_exact_scored_pool(tmp_path,workers):
    test=tmp_path/"test";test.mkdir()
    header="entity_id\tbusiness_name\tbusiness_address\tcountry\n"
    (test/"test_source1.tsv").write_text(header+"S1-A\tCafe Lumiere\t00123 Rue Mozart 75001\tFrance\nS1-B\tUnique Singleton\t\tNewlandia\n")
    for source in (2,3):
        (test/f"test_source{source}.tsv").write_text(header+f"S{source}-A\tCafe Lumiere\t00123 Rue Mozart 75001\tFrance\n")
    raw={"entity_id":"S1-A","business_name":"Cafe Lumiere","business_address":"00123 Rue Mozart 75001","country":"France"}
    ref=normalize_record(raw)
    examples=[]
    for name,address,label in [("Cafe Lumiere","00123 Rue Mozart 75001",1),("Other Business","00567 Elm Road 560001",0)]:
        target=normalize_record({**raw,"entity_id":"S2-A","business_name":name,"business_address":address})
        examples.append((build_pair_features(ref,target,Candidate("S1-A","S2-A","S2",.8 if label else .2,("rare_name",))),label))
    frame=pd.DataFrame([values for values,label in examples]*20,columns=FEATURE_NAMES)
    labels=[label for values,label in examples]*20
    model=lgb.LGBMClassifier(n_estimators=8,min_child_samples=1,num_leaves=3,n_jobs=1,verbosity=-1).fit(frame,labels)
    artifact=tmp_path/"model";artifact.mkdir()
    model.booster_.save_model(str(artifact/"model.txt"))
    config=DiskSearchConfig(top_k=2,path_top_k=5,character_mode="off")
    (artifact/"report.json").write_text(json.dumps({"feature_version":FEATURE_VERSION,"search_config":asdict(config),"threshold":.5}))
    out=tmp_path/f"out{workers}"
    report=run_inference(test,tmp_path/"indexes",artifact,out,workers=workers,batch_size=1)
    assert report["ready_for_submission"]
    with (out/"candidate_pairs.tsv").open() as stream:
        candidates=list(csv.DictReader(stream,delimiter="\t"))
    assert len(candidates)==2
    counts=sum(len(row["candidate_entity_ids"].split(",")) if row["candidate_entity_ids"] else 0 for row in candidates)
    assert counts==report["scored_pairs"]
    assert not validate_submission(out/"matching_results.tsv",out/"candidate_pairs.tsv",test)
