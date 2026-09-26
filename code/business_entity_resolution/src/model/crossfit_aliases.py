"""Disk-backed alias fitting with entity-disjoint inner folds.

Target ownership is retained for training exclusion and label lookup. Only outer
training entities contribute alias counts. Each inner-fold model excludes all
targets owned by that fold, including retrieval aliases, not just pair features.
"""
import csv
import hashlib
import json
import sqlite3
import time
from pathlib import Path

from ..evaluation.validation import entity_fold, stable_hash
from ..preprocessing.normalize import normalize_name, name_core
from .name_aliases import ALIAS_VERSION


def inner_fold(entity_id, folds=5):
    return stable_hash(entity_id, "alias-inner-v1") % folds


def source_fingerprint(train_dir, index_dir, folds):
    files = {}
    for name in ("train_source1.tsv", "train_ground_truth.tsv"):
        path = (Path(train_dir) / name).resolve()
        stat = path.stat()
        h = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8*1024*1024), b""): h.update(block)
        files[name] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": h.hexdigest()}
    indexes = {}
    for source in ("S2", "S3"):
        path = (Path(index_dir) / f"index_train_{source}.sqlite").resolve()
        with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as conn:
            meta = json.loads(conn.execute("SELECT value FROM metadata WHERE key='build'").fetchone()[0])
        raw = (Path(train_dir) / f"train_source{source[-1]}.tsv").resolve()
        stat = raw.stat(); fp = meta["fingerprint"]
        if not meta["complete"] or (fp["path"], fp["size"], fp["mtime_ns"]) != (str(raw), stat.st_size, stat.st_mtime_ns):
            raise ValueError("Source/index fingerprint mismatch")
        indexes[source] = fp
    return {"version": "crossfit-alias-counts-v1", "inner_folds": folds, "seed": 42, "files": files, "indexes": indexes}


def build_counts(train_dir, index_dir, output, folds=5):
    train_dir, index_dir, output = map(Path, (train_dir, index_dir, output))
    if folds < 2: raise ValueError("At least two inner folds are required")
    fingerprint = source_fingerprint(train_dir, index_dir, folds)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        with sqlite3.connect(output) as conn:
            saved = json.loads(conn.execute("SELECT value FROM metadata WHERE key='build'").fetchone()[0])
        if saved["fingerprint"] != fingerprint or not saved["complete"]: raise ValueError("Count database mismatch")
        return saved
    temporary = output.with_suffix(".building.sqlite")
    if temporary.exists(): raise FileExistsError("Partial count database exists; preserve it and use a fresh output path")
    started = time.perf_counter()
    conn = sqlite3.connect(temporary.resolve().as_uri(), uri=True)
    conn.executescript("""
        PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF;
        PRAGMA temp_store=FILE; PRAGMA cache_size=-65536;
        CREATE TABLE refs(id TEXT PRIMARY KEY, core TEXT NOT NULL, outer_fold TEXT NOT NULL,
                          inner_fold INTEGER NOT NULL) WITHOUT ROWID;
        CREATE TABLE owners(id TEXT PRIMARY KEY, owner TEXT NOT NULL, outer_fold TEXT NOT NULL) WITHOUT ROWID;
        CREATE TABLE counts(alias TEXT, canonical TEXT, inner_fold INTEGER, n INTEGER NOT NULL,
                            PRIMARY KEY(alias,canonical,inner_fold)) WITHOUT ROWID;
        CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT);
    """)
    def progress(stage, **values):
        print(json.dumps({"stage": stage, "seconds": time.perf_counter()-started, **values}), flush=True)
    try:
        batch = []; references = 0
        with (train_dir / "train_source1.tsv").open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream, delimiter="\t"):
                eid = row["entity_id"]
                batch.append((eid, name_core(normalize_name(row["business_name"])), entity_fold(eid), inner_fold(eid, folds)))
                if len(batch) >= 20000:
                    conn.executemany("INSERT INTO refs VALUES(?,?,?,?)", batch); references += len(batch); batch.clear()
                    conn.commit()
                    if references % 200000 == 0: progress("alias_references", references=references)
        if batch: conn.executemany("INSERT INTO refs VALUES(?,?,?,?)", batch); references += len(batch)
        conn.commit(); progress("alias_references_ready", references=references)
        batch = []; targets = 0
        with (train_dir / "train_ground_truth.tsv").open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream, delimiter="\t"):
                eid = row["source1_entity_id"]; fold = entity_fold(eid)
                for target in filter(None, row["matched_entity_ids"].split(",")):
                    batch.append((target, eid, fold))
                if len(batch) >= 20000:
                    conn.executemany("INSERT INTO owners VALUES(?,?,?)", batch); targets += len(batch); batch.clear(); conn.commit()
        if batch: conn.executemany("INSERT INTO owners VALUES(?,?,?)", batch); targets += len(batch)
        conn.execute("CREATE INDEX owners_by_reference ON owners(owner)"); conn.commit()
        progress("alias_ownership_ready", targets=targets)
        for source in ("S2", "S3"):
            path = (index_dir / f"index_train_{source}.sqlite").resolve()
            conn.execute("ATTACH DATABASE ? AS target", (path.as_uri() + "?mode=ro",))
            conn.execute("PRAGMA target.mmap_size=2147418112")
            conn.execute("""INSERT INTO counts(alias,canonical,inner_fold,n)
                SELECT t.core,r.core,r.inner_fold,count(*)
                FROM owners o JOIN target.records t ON t.id=o.id JOIN refs r ON r.id=o.owner
                WHERE o.id>=? AND o.id<? AND o.outer_fold='train' AND r.core<>'' AND t.core<>''
                GROUP BY t.core,r.core,r.inner_fold
                ON CONFLICT(alias,canonical,inner_fold) DO UPDATE SET n=counts.n+excluded.n
                """, (source + "-", source + "."))
            conn.commit(); conn.execute("DETACH DATABASE target")
            progress("alias_source_counts_ready", source=source)
        if source_fingerprint(train_dir, index_dir, folds) != fingerprint: raise ValueError("Inputs changed during fitting")
        metadata = {"fingerprint": fingerprint, "complete": True, "references": references, "owned_targets": targets,
            "fitted_targets": conn.execute("SELECT sum(n) FROM counts").fetchone()[0] or 0,
            "count_rows": conn.execute("SELECT count(*) FROM counts").fetchone()[0], "seconds": time.perf_counter()-started}
        conn.execute("INSERT INTO metadata VALUES('build',?)", (json.dumps(metadata),)); conn.commit(); conn.close()
        temporary.replace(output); progress("alias_counts_complete", **metadata)
        return metadata
    except BaseException:
        conn.close()
        raise


def export_aliases(database, output, excluded_fold=None, min_support=2, min_dominance=.98):
    database, output = Path(database).resolve(), Path(output)
    conn = sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)
    build = json.loads(conn.execute("SELECT value FROM metadata WHERE key='build'").fetchone()[0])
    folds = build["fingerprint"]["inner_folds"]
    if excluded_fold is not None and not 0 <= excluded_fold < folds: raise ValueError("Invalid inner fold")
    metadata = {"version": ALIAS_VERSION, "seed": 42, "fitted_fold": "train", "excluded_inner_fold": excluded_fold,
        "inner_folds": folds, "crossfit_input": build["fingerprint"], "min_support": min_support,
        "min_dominance": min_dominance, "license": "MIT",
        "policy": "Only outer-training targets contribute. All targets owned by the excluded inner entity fold are omitted from retrieval and feature aliases."}
    if output.exists():
        old = json.loads(output.read_text())
        if any(old["metadata"].get(k) != v for k, v in metadata.items()): raise ValueError("Alias cache mismatch")
        conn.close(); return old["metadata"]
    query = "SELECT alias,canonical,sum(n) FROM counts"
    parameters = ()
    if excluded_fold is not None: query += " WHERE inner_fold<>?"; parameters = (excluded_fold,)
    query += " GROUP BY alias,canonical ORDER BY alias,canonical"
    entries, alternatives, current, histogram = {}, {}, None, {}
    observed = 0; targets = 0
    def finish(alias, values):
        if alias is None: return
        canonical, support = min(values.items(), key=lambda item: (-item[1], item[0]))
        total = sum(values.values())
        if support >= min_support and support/total >= min_dominance:
            entries[alias] = [canonical, support, total]
        else:
            possibilities = [[name, count, total] for name, count in sorted(values.items(), key=lambda item: (-item[1], item[0]))[:3]
                if count >= min_support and count/total >= .10]
            if possibilities: alternatives[alias] = possibilities
    for alias, canonical, count in conn.execute(query, parameters):
        if alias != current:
            finish(current, histogram); current = alias; histogram = {}; observed += 1
        histogram[canonical] = count; targets += count
    finish(current, histogram)
    count_query = "SELECT count(*) FROM refs WHERE outer_fold='train' AND core<>''"
    if excluded_fold is not None: count_query += " AND inner_fold<>?"
    metadata.update(alias_strings=len(entries), ambiguous_alias_strings=len(alternatives), observed_strings=observed,
        fitted_targets=targets, fitted_references=conn.execute(count_query, parameters).fetchone()[0])
    conn.close(); output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".partial")
    temporary.write_text(json.dumps({"metadata": metadata, "entries": entries, "alternatives": alternatives}, ensure_ascii=False))
    temporary.replace(output)
    print(json.dumps({"stage": "crossfit_alias_export", "file": str(output), "excluded_fold": excluded_fold,
        "fitted_targets": targets, "aliases": len(entries)}), flush=True)
    return metadata
