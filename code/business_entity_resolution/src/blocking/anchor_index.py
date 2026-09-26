"""Bounded compound-key retrieval from corpus text and a fitted name channel."""

import hashlib
import json
import os
import sqlite3
import time
from functools import lru_cache
from pathlib import Path

from ..preprocessing.normalize import accent_fold

ANCHOR_VERSION = "compound-anchor-v2"


def token(prefix, text):
    return prefix + hashlib.blake2b(text.encode("utf-8"), digest_size=12).hexdigest()


@lru_cache(maxsize=100_000)
def family_key(core):
    folded = " ".join(sorted(set(accent_fold(core).split())))
    return token("f", folded) if folded else ""


def family_keys(core, aliases=None, *, reference=False):
    names = {core}
    if aliases and not reference:
        canonical, _, _ = aliases.resolve(core)
        names.add(canonical)
        names.update(row[0] for row in aliases.alternatives.get(core, ()))
    return sorted({family_key(name) for name in names if name})


def fingerprint(base_metadata, aliases):
    # Hash fitted state rather than its filename: different weights cannot
    # silently reuse the same derived index.
    alias_state = {"entries": aliases.entries, "alternatives": aliases.alternatives} if aliases else None
    digest = hashlib.sha256(json.dumps(alias_state, sort_keys=True, ensure_ascii=False,
        separators=(",", ":")).encode("utf-8")).hexdigest()
    return {"version": ANCHOR_VERSION, "base": base_metadata["fingerprint"], "alias_sha256": digest}


def anchor_path(base_path):
    return Path(str(base_path) + ".anchors-v2.sqlite")


def build_anchor_index(base_path, aliases=None):
    base_path = Path(base_path).resolve()
    output = anchor_path(base_path)
    with sqlite3.connect(f"{base_path.as_uri()}?mode=ro", uri=True) as base:
        metadata = json.loads(base.execute("SELECT value FROM metadata WHERE key='build'").fetchone()[0])
        expected = fingerprint(metadata, aliases)
        if output.exists():
            with sqlite3.connect(f"{output.as_uri()}?mode=ro", uri=True) as existing:
                saved = json.loads(existing.execute("SELECT value FROM metadata").fetchone()[0])
            if saved["complete"] and saved["fingerprint"] == expected:
                return saved
            raise ValueError(f"Compound index fingerprint differs: {output}; choose a fresh index directory")
        temporary = output.with_name(output.name + f".building-{os.getpid()}")
        if temporary.exists():
            raise FileExistsError(temporary)
        start = time.perf_counter()
        connection = sqlite3.connect(temporary)
        connection.executescript("""
            PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF; PRAGMA cache_size=-65536;
            CREATE VIRTUAL TABLE anchors USING fts5(family,address,digits,scope,
                content='',tokenize='unicode61 remove_diacritics 2');
            CREATE TABLE metadata(value TEXT);
        """)
        count = 0
        try:
            cursor = base.execute("SELECT rowid,core,address,address_digits,country FROM records ORDER BY rowid")
            while rows := cursor.fetchmany(20_000):
                records = [(rowid, " ".join(family_keys(core, aliases)), address,
                    " ".join(sorted({"a" + (t.lstrip("0") or "0") for t in digits.split()})),
                    token("c", country) if country else "missingcountry") for rowid,core,address,digits,country in rows]
                connection.executemany("INSERT INTO anchors(rowid,family,address,digits,scope) VALUES(?,?,?,?,?)", records)
                connection.commit()
                count += len(records)
                if count % 500_000 == 0:
                    print(json.dumps({"stage":"anchor_index","source":metadata["fingerprint"]["source"],
                        "records":count,"seconds":round(time.perf_counter()-start,1)}),flush=True)
            connection.execute("INSERT INTO anchors(anchors) VALUES('optimize')")
            saved = {"fingerprint":expected,"complete":True,"records":count,"seconds":time.perf_counter()-start}
            connection.execute("INSERT INTO metadata VALUES(?)",(json.dumps(saved),))
            connection.commit()
            connection.close()
            temporary.replace(output)
            return saved
        except BaseException:
            connection.close()
            raise


class AnchorIndex:
    def __init__(self, base_path, aliases=None):
        self.connection = sqlite3.connect(f"{anchor_path(Path(base_path).resolve()).as_uri()}?mode=ro",uri=True)
        self.connection.execute("PRAGMA cache_size=-65536")
        self.aliases = aliases
        self.metadata = json.loads(self.connection.execute("SELECT value FROM metadata").fetchone()[0])
        if not self.metadata["complete"]:
            raise ValueError("Incomplete compound index")

    def close(self):
        self.connection.close()

    def search(self, expression, limit):
        return [row[0] for row in self.connection.execute(
            "SELECT rowid FROM anchors WHERE anchors MATCH ? ORDER BY rank LIMIT ?",(expression,limit))]

    def query(self, reference, base):
        paths = {}
        cfg = base.config
        expr = base.expression
        family = expr("family",family_keys(reference.core, reference=True)) if reference.core else ""
        scope = expr("scope",[token("c",reference.country),"missingcountry"]) if reference.country else ""
        def find(path, expression, limit=None):
            if scope:
                expression += " AND " + scope
            paths[path] = self.search(expression,limit or cfg.path_top_k)
        digits = ["a" + t for t in sorted({t.lstrip("0") or "0" for t in reference.address_digits})]
        address = base.rare_terms("address",reference.address_tokens,cfg.max_address_df,cfg.query_tokens)
        # Compound keys retrieve branches before truncation; shared chain names
        # alone cannot exhaust every retrieval path.
        if family and digits:
            find("family_numeric",family + " AND " + expr("digits",digits))
        if family and address:
            find("family_address",family + " AND " + expr("address",address))
        if digits:
            locations = list(dict.fromkeys(accent_fold(t) for t in reference.address.split()[-6:]
                if len(t)>=3 and not t.isdecimal() and 0<base.df("address",accent_fold(t))<=base.record_count*.5))[-2:]
            ordered = sorted(reference.address_digits,key=lambda t:(base.df("digits","a"+t) or base.record_count,t))[:2]
            numeric = " AND ".join(expr("digits",["a"+(t.lstrip("0") or "0")]) for t in ordered)
            for i, location in enumerate(locations):
                find(f"numeric_location_{i}",numeric + " AND " + expr("address",[location]))
        if len(address)>=2:
            # Two rare components support names absent from the learned channel.
            find("address_compound",expr("address",address[:1])+" AND "+expr("address",address[1:3]))
        if family and len({row for values in paths.values() for row in values}) < cfg.top_k:
            find("family_fallback",family,min(cfg.path_top_k,max(8,cfg.top_k)))
        if not any(paths.values()):
            names=base.rare_terms("name",reference.name_tokens,cfg.max_word_df,2)
            if names:
                # Fallback is bounded; word retrieval uses the base source index.
                paths["name_fallback"]=base.search(expr("name",names),country=reference.country)
            if address:
                find("address_fallback",expr("address",address[:2]))
        return paths
