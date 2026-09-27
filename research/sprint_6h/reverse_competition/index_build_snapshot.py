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
        write_json(str(output_path)+".complete.json", meta)
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
        country = target["country"] if target["country"] and target["country"] in self.meta["country_counts"] else GLOBAL
        n = self.meta["records"] if country == GLOBAL else self.meta["country_counts"][country]
        bound = min(self.config.max_df, max(1, math.floor(self.config.country_df_fraction*n)))
        values = []
        for term in fts_terms(target[field]):
            row = self.conn.execute("SELECT df FROM frequencies WHERE country=? AND field=? AND term=?",
                                    (country, field, term)).fetchone()
            if row and 0 < row[0] <= bound:
                values.append((row[0], term))
        return sorted(values)[:self.config.terms_per_field]

    def query(self, target, own_id):
        """Complete postings -> overlap shortlist -> raw top8 -> excluded top3."""
        cache_key = tuple(target[key] for key in ("name", *FIELDS, "country"))
        if cache_key in self.cache:
            pool, coverage = self.cache[cache_key]
            self.cache.move_to_end(cache_key)
        else:
            selected, hits, rows = {}, {}, {}
            country = target["country"] if target["country"] and target["country"] in self.meta["country_counts"] else GLOBAL
            n = self.meta["records"] if country == GLOBAL else self.meta["country_counts"][country]
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
                        "known_country": country != GLOBAL, "selected_terms": selected, "complete_postings": True}
            self.cache[cache_key] = (pool, coverage)
            if len(self.cache) > 25_000: self.cache.popitem(last=False)
        rivals = [(r, v) for r, v in pool if r["id"] != own_id]
        return rivals[:self.config.rivals], {**coverage, "rerank_count": len(pool),
                   "no_query": not coverage["query_terms"], "no_result": not rivals,
                   "self_excluded": any(r["id"] == own_id for r, v in pool)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--source", required=True); build.add_argument("--output", required=True)
    build.add_argument("--corpus", choices=EXPECTED_COUNTS, required=True)
    build.add_argument("--expected-sha256", required=True)
    args = parser.parse_args()
    if args.command == "build":
        print(json.dumps(build_index(args.source, args.output, args.corpus, expected_sha256=args.expected_sha256)), flush=True)


if __name__ == "__main__": main()
