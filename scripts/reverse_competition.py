"""Label-free, corpus-local S1 reverse competition evidence.

This deliberately does not use DiskSourceIndex's learned retrieval reranker.
Only complete postings below a country-local DF bound enter the raw text views.
An absent query/result is missing evidence, never a uniqueness claim.
"""
from __future__ import annotations

import argparse
from collections import Counter, OrderedDict
from dataclasses import asdict, dataclass
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code/business_entity_resolution"))
from src.preprocessing.normalize import accent_fold, extract_digits, name_core, normalize_address, normalize_country, normalize_name
from src.features.evidence import phonetic
from rapidfuzz import fuzz

VERSION = "reverse-s1-v1"
KEYS = ["source1_entity_id", "candidate_entity_id"]
FIELDS = ("namecore", "phonetic", "address")
SOURCE_COLUMNS = ("entity_id", "business_name", "business_address", "country")
RECORD_FIELDS = ("id", "name", *FIELDS, "primary_address", "country")
GLOBAL = "__global__"
EXPECTED_COUNTS = {"train": 2_206_821, "test": 1_732_544}
FEATURE_NAMES = [
    "reverse_own_name_sort", "reverse_own_phonetic_sort", "reverse_own_address_sort",
    "reverse_own_number_jaccard", "reverse_own_number_contradiction", "reverse_own_joint",
    "reverse_other_name_sort", "reverse_other_phonetic_sort", "reverse_other_address_sort",
    "reverse_other_number_jaccard", "reverse_other_number_contradiction", "reverse_other_joint",
    "reverse_margin_name", "reverse_margin_phonetic", "reverse_margin_address", "reverse_margin_number",
    "reverse_margin_joint", "reverse_other_second_joint", "reverse_other_third_joint",
    "reverse_best_second_joint_gap", "reverse_name_terms_used", "reverse_phonetic_terms_used",
    "reverse_address_terms_used", "reverse_query_fields_used", "reverse_high_df_terms_skipped",
    "reverse_shortlist_saturated", "reverse_own_in_top8", "reverse_no_other_returned",
    "reverse_target_name_missing", "reverse_target_address_missing", "reverse_target_numbers_missing",
    "reverse_target_country_missing", "reverse_own_country_conflict"]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def fts_terms(value):
    # Index precisely these tokens as well as querying them; Unicode combining
    # marks are excluded by unicode61, while preserved in raw similarity text.
    return frozenset(re.findall(r"[^\W_]+", accent_fold(value), re.UNICODE))


def clean_record(row):
    name = normalize_name(row["business_name"])
    core = accent_fold(name_core(name) or name)
    primary_address = normalize_address(row["business_address"])
    return {"id": str(row["entity_id"]), "name": name, "namecore": core,
            "phonetic": phonetic(core), "address": accent_fold(primary_address), "primary_address": primary_address,
            "country": normalize_country(row["country"])}


@dataclass(frozen=True)
class QueryConfig:
    country_df_fraction: float = .02
    max_df: int = 20_000
    terms_per_field: int = 2
    field_top_k: int = 16
    joint_top_k: int = 32
    rerank_top_k: int = 8
    rivals: int = 3

    def validate(self):
        if asdict(self) != asdict(QueryConfig()):
            raise ValueError("Reverse query policy is frozen; alternate bounds require a new version")


def build_index(source_path, output_path, corpus, *, expected_sha256=None, expected_count=None, batch_size=20_000):
    if corpus not in EXPECTED_COUNTS:
        raise ValueError("Corpus must be separate train or test S1")
    source_path, output_path = Path(source_path).resolve(), Path(output_path).resolve()
    source_hash = digest(source_path)
    if expected_sha256 and source_hash != expected_sha256:
        raise ValueError("Raw S1 source SHA256 mismatch")
    expected_count = EXPECTED_COUNTS[corpus] if expected_count is None else expected_count
    fingerprint = {"corpus": corpus, "source_sha256": source_hash, "source_size": source_path.stat().st_size,
                   "version": VERSION, "normalization": "shared-normalize/accent-fold/core/phonetic;fts-terms-v1"}
    if output_path.exists():
        with sqlite3.connect(f"{output_path.as_uri()}?mode=ro", uri=True) as conn:
            meta = json.loads(conn.execute("SELECT value FROM metadata WHERE key='build'").fetchone()[0])
        if meta["fingerprint"] != fingerprint or not meta["complete"] or meta["records"] != expected_count:
            raise ValueError("Existing index lineage/count differs; choose a fresh output")
        return meta
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + f".building-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(temporary)
    started = time.perf_counter()
    code_root = Path(__file__).resolve().parents[1]/"code/business_entity_resolution/src"
    build_provenance = {"build_code_sha256": digest(__file__),
                        "normalization_sha256": digest(code_root/"preprocessing/normalize.py"),
                        "phonetic_code_sha256": digest(code_root/"features/evidence.py"),
                        "df_policy_sha256": hashlib.sha256(json.dumps(asdict(QueryConfig()),sort_keys=True).encode()).hexdigest()}
    conn = sqlite3.connect(temporary)
    conn.executescript("""
        PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF; PRAGMA temp_store=FILE; PRAGMA cache_size=-262144;
        CREATE TABLE records(id TEXT NOT NULL, name TEXT NOT NULL, namecore TEXT NOT NULL,
          phonetic TEXT NOT NULL, address TEXT NOT NULL, primary_address TEXT NOT NULL, country TEXT NOT NULL);
        CREATE VIRTUAL TABLE words USING fts5(namecore, phonetic, address, content='', tokenize='unicode61 remove_diacritics 2');
        CREATE TABLE frequencies(country TEXT, field TEXT, term TEXT, df INTEGER, PRIMARY KEY(country,field,term)) WITHOUT ROWID;
        CREATE TABLE countries(country TEXT PRIMARY KEY, records INTEGER) WITHOUT ROWID;
        CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT);
    """)
    frequencies, countries = Counter(), Counter()
    count = 0
    def flush(batch):
        nonlocal count
        conn.executemany("INSERT INTO records VALUES(?,?,?,?,?,?,?)", [r[0] for r in batch])
        conn.executemany("INSERT INTO words(rowid,namecore,phonetic,address) VALUES(?,?,?,?)",
                         [(count+i+1, *r[1]) for i, r in enumerate(batch)])
        count += len(batch)
        conn.commit()
        if count % 100_000 == 0:
            print(json.dumps({"stage": "reverse_index_build", "records": count,
                              "seconds": round(time.perf_counter()-started, 2)}), flush=True)
    try:
        with source_path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream, delimiter="\t")
            if tuple(reader.fieldnames or ()) != SOURCE_COLUMNS:
                raise ValueError("Unexpected S1 source header")
            batch = []
            for row in reader:
                if None in row or any(v is None for v in row.values()):
                    raise ValueError(f"Malformed S1 source row {reader.line_num}")
                record = clean_record(row)
                if not record["id"].startswith("S1-"):
                    raise ValueError("Non-S1 row in reverse corpus")
                countries[record["country"]] += 1
                tokens = [fts_terms(record[field]) for field in FIELDS]
                for field, terms in zip(FIELDS, tokens):
                    frequencies.update((record["country"], field, term) for term in terms)
                batch.append((tuple(record[key] for key in RECORD_FIELDS),
                              tuple(" ".join(sorted(terms)) for terms in tokens)))
                if len(batch) >= batch_size:
                    flush(batch); batch.clear()
            if batch: flush(batch)
        if count != expected_count:
            raise ValueError(f"S1 row count mismatch: {count} != {expected_count}")
        if digest(source_path) != source_hash:
            raise ValueError("Raw S1 source changed during build")
        print(json.dumps({"stage": "reverse_index_finalize", "records": count, "frequency_keys": len(frequencies)}), flush=True)
        conn.executemany("INSERT INTO frequencies VALUES(?,?,?,?)", ((*key, value) for key, value in frequencies.items()))
        conn.executemany("INSERT INTO countries VALUES(?,?)", countries.items())
        conn.execute("INSERT INTO frequencies SELECT ?,field,term,SUM(df) FROM frequencies GROUP BY field,term", (GLOBAL,))
        conn.execute("CREATE UNIQUE INDEX records_id ON records(id)")
        conn.execute("INSERT INTO words(words) VALUES('optimize')")
        meta = {"complete": True, "fingerprint": fingerprint, "records": count,
                "country_counts": dict(countries), "query_config": asdict(QueryConfig()),
                "labels_read": False, "test_pseudo_labels": False, "seconds": time.perf_counter()-started,
                "sqlite_version": sqlite3.sqlite_version}
        conn.execute("INSERT INTO metadata VALUES('build',?)", (json.dumps(meta),))
        conn.commit(); conn.close()
        temporary.replace(output_path)
        write_json(str(output_path)+".complete.json", {**meta, **build_provenance, "index_sha256":digest(output_path)})
        return meta
    except BaseException:
        conn.close()
        raise


def canonical_numbers(address):
    return {t.lstrip("0") or "0" for t in extract_digits(" ".join(t for t in address.split() if t.isdecimal()))}


def raw_views(target, rival):
    name = fuzz.token_sort_ratio(target["namecore"], rival["namecore"])/100 if target["namecore"] and rival["namecore"] else 0.
    phone = fuzz.token_sort_ratio(target["phonetic"], rival["phonetic"])/100 if target["phonetic"] and rival["phonetic"] else 0.
    address = fuzz.token_sort_ratio(target["address"], rival["address"])/100 if target["address"] and rival["address"] else 0.
    left, right = canonical_numbers(target["address"]), canonical_numbers(rival["address"])
    digits = len(left & right)/len(left | right) if left or right else 0.
    contradiction = bool(left and right and not left & right)
    joint = max(.58*max(name, phone)+.35*address+.07*digits, .92*address+.08*digits)-.08*contradiction
    return {"name": name, "phonetic": phone, "address": address, "digits": digits,
            "number_conflict": float(contradiction), "joint": joint}


class ReverseIndex:
    def __init__(self, path, *, corpus=None, config=None):
        self.path = Path(path).resolve()
        self.config = config or QueryConfig(); self.config.validate()
        self.conn = sqlite3.connect(f"{self.path.as_uri()}?mode=ro", uri=True)
        self.conn.execute("PRAGMA cache_size=-262144")
        self.meta = json.loads(self.conn.execute("SELECT value FROM metadata WHERE key='build'").fetchone()[0])
        if not self.meta["complete"] or self.meta["fingerprint"]["version"] != VERSION:
            raise ValueError("Incomplete or incompatible reverse index")
        if corpus and self.meta["fingerprint"]["corpus"] != corpus:
            raise ValueError("Train/test reverse corpus mismatch")
        if self.meta["query_config"] != asdict(self.config):
            raise ValueError("Reverse index/query policy mismatch")
        self.index_manifest = json.loads(Path(str(self.path)+".complete.json").read_text())
        if not self.index_manifest.get("complete") or self.index_manifest["fingerprint"] != self.meta["fingerprint"] or self.index_manifest["records"] != self.meta["records"]:
            raise ValueError("Reverse index completion seal mismatch")
        if self.index_manifest.get("index_sha256") and digest(self.path) != self.index_manifest["index_sha256"]:
            raise ValueError("Reverse index binary SHA256 mismatch")
        self.cache = OrderedDict()

    def close(self): self.conn.close()

    def records(self, ids):
        ids = list(dict.fromkeys(ids)); result = {}
        for start in range(0, len(ids), 500):
            chunk = ids[start:start+500]
            for row in self.conn.execute("SELECT id,name,namecore,phonetic,address,primary_address,country FROM records WHERE id IN ("+",".join("?" for _ in chunk)+")", chunk):
                result[row[0]] = dict(zip(RECORD_FIELDS, row))
        return result

    def rare_terms(self, target, field):
        country = target["country"] if target["country"] else GLOBAL
        n = self.meta["records"] if country == GLOBAL else self.meta["country_counts"].get(country, 0)
        bound = min(self.config.max_df, max(1, math.floor(self.config.country_df_fraction*n)))
        values = []
        for term in fts_terms(target[field]):
            row = self.conn.execute("SELECT df FROM frequencies WHERE country=? AND field=? AND term=?",
                                    (country, field, term)).fetchone()
            if row and 0 < row[0] <= bound:
                values.append((row[0], term))
        return sorted(values)[:self.config.terms_per_field]

    def high_df_skipped(self, target):
        country = target["country"] if target["country"] else GLOBAL
        n = self.meta["records"] if country == GLOBAL else self.meta["country_counts"].get(country, 0)
        bound = min(self.config.max_df, max(1, math.floor(self.config.country_df_fraction*n)))
        skipped = 0
        for field in FIELDS:
            for term in fts_terms(target[field]):
                row = self.conn.execute("SELECT df FROM frequencies WHERE country=? AND field=? AND term=?", (country, field, term)).fetchone()
                skipped += bool(row and row[0] > bound)
        return skipped

    def query(self, target, own_id):
        """Complete postings -> overlap shortlist -> raw top8 -> excluded top3."""
        cache_key = tuple(target[key] for key in ("name", *FIELDS, "country"))
        if cache_key in self.cache:
            pool, coverage = self.cache[cache_key]
            self.cache.move_to_end(cache_key)
        else:
            selected, hits, rows = {}, {}, {}
            country = target["country"] if target["country"] else GLOBAL
            n = self.meta["records"] if country == GLOBAL else self.meta["country_counts"].get(country, 0)
            for field in FIELDS:
                selected[field] = self.rare_terms(target, field)
                hits[field] = {}
                total_weight = 0.
                for expected_df, term in selected[field]:
                    weight = 1.+math.log((n+1)/(expected_df+1))
                    total_weight += weight
                    sql = "SELECT words.rowid,records.id FROM words JOIN records ON records.rowid=words.rowid WHERE words MATCH ?"
                    params = [field+':"'+term.replace('"','""')+'"']
                    if country != GLOBAL:
                        sql += " AND records.country=?"; params.append(country)
                    found = list(self.conn.execute(sql, params))
                    if len(found) != expected_df:
                        raise ValueError("FTS posting/DF mismatch: cannot certify complete postings")
                    for rowid, entity_id in found:
                        rows[rowid] = entity_id
                        hits[field][rowid] = hits[field].get(rowid, 0.)+weight
                if total_weight:
                    hits[field] = {r: v/total_weight for r, v in hits[field].items()}
            overlap_joint = {r: max(.65*max(hits["namecore"].get(r,0.),hits["phonetic"].get(r,0.))+.35*hits["address"].get(r,0.),
                                    .92*hits["address"].get(r,0.)) for r in rows}
            shortlist = set()
            for field in FIELDS:
                shortlist.update(sorted(hits[field], key=lambda r: (-hits[field][r],rows[r]))[:self.config.field_top_k])
            shortlist.update(sorted(overlap_joint, key=lambda r: (-overlap_joint[r],rows[r]))[:self.config.joint_top_k])
            candidates = []
            rowids = sorted(shortlist)
            for start in range(0, len(rowids), 500):
                chunk = rowids[start:start+500]
                for row in self.conn.execute("SELECT id,name,namecore,phonetic,address,primary_address,country FROM records WHERE rowid IN ("+",".join("?" for _ in chunk)+")", chunk):
                    record = dict(zip(RECORD_FIELDS, row))
                    candidates.append((record, raw_views(target, record)))
            pool = sorted(candidates, key=lambda r: (-r[1]["joint"],r[0]["id"]))[:self.config.rerank_top_k]
            coverage = {"query_fields": sum(bool(selected[f]) for f in FIELDS),
                        "query_terms": sum(len(selected[f]) for f in FIELDS), "posting_count": len(rows),
                        "shortlist_count": len(shortlist), "country_records": n,
                        "known_country": country != GLOBAL and country in self.meta["country_counts"], "country_unavailable": country != GLOBAL and country not in self.meta["country_counts"], "selected_terms": selected, "complete_postings": True,
                        "high_df_skipped": self.high_df_skipped(target),
                        "shortlist_saturated": any(len(hits[f]) > self.config.field_top_k for f in FIELDS) or len(overlap_joint) > self.config.joint_top_k}
            self.cache[cache_key] = (pool, coverage)
            if len(self.cache) > 25_000: self.cache.popitem(last=False)
        rivals = [(r, v) for r, v in pool if r["id"] != own_id]
        return rivals[:self.config.rivals], {**coverage, "rerank_count": len(pool),
                   "no_query": not coverage["query_terms"], "no_result": not rivals,
                   "self_excluded": any(r["id"] == own_id for r, v in pool)}


def pair_features(index, target, own):
    rivals, coverage = index.query(target, own["id"])
    views = raw_views(target, own)
    components = ("name", "phonetic", "address", "digits", "number_conflict", "joint")
    values = [views[k] for k in components]
    values.extend(rivals[0][1][k] if rivals else -1. for k in components)
    values.extend(views[k]-rivals[0][1][k] if rivals else -1. for k in ("name", "phonetic", "address", "digits", "joint"))
    values.extend([rivals[i][1]["joint"] if len(rivals)>i else -1. for i in (1,2)])
    values.append(rivals[0][1]["joint"]-rivals[1][1]["joint"] if len(rivals)>1 else -1.)
    values.extend(len(coverage["selected_terms"][field]) for field in FIELDS)
    values.extend([coverage["query_fields"], coverage["high_df_skipped"], coverage["shortlist_saturated"],
                   coverage["self_excluded"], not rivals, not target["namecore"], not target["address"],
                   not canonical_numbers(target["address"]), not target["country"],
                   bool(target["country"] and own["country"] and target["country"] != own["country"])])
    assert len(values) == len(FEATURE_NAMES)
    return dict(zip(FEATURE_NAMES, map(float, values))), {
        **coverage, "rival_ids": [record["id"] for record, view in rivals],
        "rival_views": [view for record, view in rivals], "posting_count_nonself": coverage["posting_count"]-int((not target["country"] or target["country"] == own["country"]) and any(any(term in fts_terms(own[field]) for df,term in coverage["selected_terms"][field]) for field in FIELDS)), "nonself_top8_count": coverage["rerank_count"]-int(coverage["self_excluded"])}


def pair_order_hash(frame):
    h = hashlib.sha256()
    for source, target in frame[KEYS].itertuples(index=False, name=None):
        h.update((str(source)+"\t"+str(target)+"\n").encode())
    return h.hexdigest()


def generate_features(pairs_path, manifest_path, index_path, target_index_prefix, output, *, route_dir=None, reference_ids=None):
    import numpy as np
    import pandas as pd
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import train_sibling_matcher as sibling
    output = Path(output)
    if output.exists(): raise FileExistsError(output)
    output.mkdir(parents=True)
    started = time.perf_counter()
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest.get("status") != "complete" or not manifest.get("verified_current_pipeline"):
        raise ValueError("Need verified complete frozen pair source")
    if digest(pairs_path) != manifest["pairs_sha256"]: raise ValueError("Source pair hash differs")
    # Read no labels, truth, or accepted decisions into the reverse mechanism.
    frame = pd.read_parquet(pairs_path, columns=[*KEYS,"probability","first_stage"]).reset_index(drop=True)
    if frame[KEYS].isna().any().any() or frame.duplicated(KEYS).any(): raise ValueError("Missing/duplicate source keys")
    if pair_order_hash(frame) != manifest["pair_order_sha256"]: raise ValueError("Source pair order differs")
    if not np.isfinite(frame[["probability","first_stage"]]).all().all() or not frame[["probability","first_stage"]].apply(lambda x:x.between(0,1)).all().all():
        raise ValueError("Invalid frozen probability")
    if route_dir:
        route, route_marker = sibling.load_pinned_route(frame, manifest, route_dir)
        route_manifest_path = Path(route_dir)/"manifest.json"
    else:
        universe = json.loads(Path(reference_ids).read_text()) if reference_ids else manifest.get("reference_ids")
        if universe is None: raise ValueError("Need declared complete reference universe for residual route")
        route = sibling.select_route(frame, universe_ids=universe)
        edges = sibling.route_edges(frame, route)
        edges.to_parquet(output/"route.parquet", index=False)
        route_marker = {"status":"complete","route":asdict(sibling.RouteConfig()),"global_references":len(universe),
                        "global_pair_order_sha256":manifest["pair_order_sha256"],"route_sha256":digest(output/"route.parquet"),
                        "frozen_sha256":manifest.get("frozen_sha256")}
        write_json(output/"route_manifest.json",route_marker)
        route_manifest_path = output/"route_manifest.json"
    selected = frame.iloc[sorted(route)].reset_index(drop=True)
    corpus = "test" if manifest.get("split") == "test" else "train"
    index = ReverseIndex(index_path, corpus=corpus)
    own = index.records(selected.source1_entity_id.unique())
    if set(selected.source1_entity_id)-set(own): raise ValueError("Missing own S1 source record")
    targets, target_meta, target_hashes = {}, {}, {}
    for source in ("S2","S3"):
        path = Path(str(target_index_prefix)+"_"+source+".sqlite").resolve()
        with sqlite3.connect(f"{path.as_uri()}?mode=ro",uri=True) as conn:
            marker = json.loads(conn.execute("SELECT value FROM metadata WHERE key='build'").fetchone()[0])
            if not marker["complete"]: raise ValueError("Incomplete frozen target index")
            target_meta[source] = marker
            source_path = Path(marker["fingerprint"]["path"])
            if not source_path.exists():
                source_path = Path(__file__).resolve().parents[1]/"dataset"/corpus/(corpus+"_source"+source[-1]+".tsv")
            target_hashes[source] = digest(source_path)
            ids = selected.loc[selected.candidate_entity_id.str.startswith(source+"-"),"candidate_entity_id"].unique().tolist()
            for start in range(0,len(ids),500):
                chunk=ids[start:start+500]
                for entity_id,name,address,country in conn.execute("SELECT id,name,address,country FROM records WHERE id IN ("+",".join("?" for _ in chunk)+")",chunk):
                    targets[entity_id]=clean_record(dict(entity_id=entity_id,business_name=name,business_address=address,country=country))
    if set(selected.candidate_entity_id)-set(targets): raise ValueError("Missing routed target source record")
    rows, diagnostics = [], []
    for i,(source,target) in enumerate(selected[KEYS].itertuples(index=False,name=None)):
        feature, diagnostic = pair_features(index, targets[target], own[source])
        rows.append({KEYS[0]:source,KEYS[1]:target,**feature})
        diagnostics.append({KEYS[0]:source,KEYS[1]:target,**diagnostic})
        if (i+1)%1000 == 0:
            print(json.dumps({"stage":"reverse_features","rows":i+1,"unique_target_queries":len(index.cache),"seconds":round(time.perf_counter()-started,2)}),flush=True)
    result = pd.DataFrame(rows,columns=[*KEYS,*FEATURE_NAMES])
    result[FEATURE_NAMES]=result[FEATURE_NAMES].astype(np.float32)
    result.to_parquet(output/"features.parquet",index=False)
    with (output/"diagnostics.jsonl").open("w") as stream:
        for row in diagnostics: stream.write(json.dumps(row)+"\n")
    index_manifest = str(index_path)+".complete.json"
    code_root = Path(__file__).resolve().parents[1]/"code/business_entity_resolution/src"
    marker={"status":"complete","features":FEATURE_NAMES,"feature_version":VERSION,"features_sha256":digest(output/"features.parquet"),
            "pair_order_sha256":pair_order_hash(result),"source_pairs_sha256":manifest["pairs_sha256"],
            "diagnostics_sha256":digest(output/"diagnostics.jsonl"),
            "source_pair_order_sha256":manifest["pair_order_sha256"],"source_manifest_sha256":digest(manifest_path),
            "route_manifest_sha256":digest(route_manifest_path),"route_manifest":str(route_manifest_path),
            "index_manifest":index_manifest,"index_manifest_sha256":digest(index_manifest),"index_metadata":index.index_manifest,
            "source_tsv_sha256":{"S1":index.meta["fingerprint"]["source_sha256"],**target_hashes},
            "normalization_sha256":digest(code_root/"preprocessing/normalize.py"),"phonetic_code_sha256":digest(code_root/"features/evidence.py"),
            "feature_code_sha256":digest(__file__),"target_index_metadata":target_meta,"query_config":asdict(QueryConfig()),
            "rows":len(result),"routed_references":result.source1_entity_id.nunique(),"unique_target_queries":len(index.cache),
            "frozen_sha256":manifest.get("frozen_sha256"),"labels_read":False,"trained_rival_scores_used":False,
            "own_id_excluded":True,"rival_ids_as_features":False,"test_pseudo_labels":False,"audit_labels_read":False,
            "seconds":time.perf_counter()-started}
    write_json(output/"manifest.json",marker)
    index.close()
    return marker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--source", required=True); build.add_argument("--output", required=True)
    build.add_argument("--corpus", choices=EXPECTED_COUNTS, required=True)
    build.add_argument("--expected-sha256", required=True)
    features = sub.add_parser("features")
    for name in ("pairs","manifest","index","target-index-prefix","output"):
        features.add_argument("--"+name,required=True)
    features.add_argument("--route-dir"); features.add_argument("--reference-ids")
    args = parser.parse_args()
    if args.command == "build":
        print(json.dumps(build_index(args.source, args.output, args.corpus, expected_sha256=args.expected_sha256)), flush=True)
    else:
        marker=generate_features(args.pairs,args.manifest,args.index,args.target_index_prefix,args.output,
                                 route_dir=args.route_dir,reference_ids=args.reference_ids)
        print(json.dumps({k:v for k,v in marker.items() if k in ("status","rows","unique_target_queries","seconds")}),flush=True)


if __name__ == "__main__": main()
