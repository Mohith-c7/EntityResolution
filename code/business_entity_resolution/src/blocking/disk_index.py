"""Corpus-local SQLite retrieval over a complete target source.

Indexes contain no labels. Query limits bound the candidates passed to reranking;
they do not restrict the target universe to a sampled pool.
"""

from __future__ import annotations

import csv
import json
import math
import os
import sqlite3
import time
from collections import OrderedDict, defaultdict
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

from rapidfuzz import fuzz

from ..preprocessing.normalize import accent_fold, extract_digits, name_core, normalize_address, normalize_country, normalize_name
from .contracts import BlockingConfig, BlockingRecord, Candidate
from .frequency_index import ensure_frequency_index, frequency_path

INDEX_VERSION = "sqlite-fts-v1"
COLUMNS = ("entity_id", "business_name", "business_address", "country")


def normalize_record(row: dict[str, str]) -> BlockingRecord:
    name = normalize_name(row["business_name"])
    address = normalize_address(row["business_address"])
    return BlockingRecord(
        row["entity_id"], name, name_core(name), address, normalize_country(row["country"]),
        frozenset(name.split()), frozenset(address.split()),
        frozenset(extract_digits(name)), frozenset(extract_digits(address)),
        frozenset(t for t in extract_digits(address) if len(t) >= 4), accent_fold(name),
    )


@dataclass(frozen=True)
class DiskSearchConfig:
    top_k: int = 20
    path_top_k: int = 80
    query_tokens: int = 6
    max_word_df: int = 100_000
    max_address_df: int = 10_000
    character_terms: int = 8
    max_character_df: int = 20_000
    character_mode: str = "always"
    reranker_version: str = "v1"
    numeric_conjunctions: bool = False
    numeric_location_queries: bool = False
    use_name_aliases: bool = False
    retrieval_mode: str = "wide"

    def __post_init__(self):
        for field in ("top_k", "path_top_k", "query_tokens", "max_word_df", "max_address_df", "character_terms", "max_character_df"):
            if type(getattr(self, field)) is not int or getattr(self, field) < 1:
                raise ValueError(f"{field} must be a positive integer")
        if self.path_top_k < self.top_k:
            raise ValueError("path_top_k must be >= top_k")
        if self.character_mode not in ("always", "fallback", "off"):
            raise ValueError("character_mode must be always, fallback, or off")
        if self.reranker_version not in ("v1", "v2", "v3", "v4"):
            raise ValueError("Unsupported reranker version")
        if self.retrieval_mode not in ("wide", "anchored"):
            raise ValueError("retrieval_mode must be wide or anchored")
        for field in ("numeric_conjunctions", "numeric_location_queries", "use_name_aliases"):
            if type(getattr(self, field)) is not bool:
                raise ValueError(f"{field} must be boolean")


def build_disk_index(source_path: str | Path, output_path: str | Path, source: str, *, batch_size: int = 20_000) -> dict:
    if source not in ("S2", "S3"):
        raise ValueError("Index source must be S2 or S3")
    source_path, output_path = Path(source_path).resolve(), Path(output_path).resolve()
    stat = source_path.stat()
    fingerprint = {"path": str(source_path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "source": source, "version": INDEX_VERSION}
    if output_path.exists():
        with sqlite3.connect(f"{output_path.as_uri()}?mode=ro", uri=True) as existing:
            meta = json.loads(existing.execute("SELECT value FROM metadata WHERE key='build'").fetchone()[0])
        if meta["fingerprint"] == fingerprint and meta["complete"]:
            ensure_frequency_index(output_path)
            return meta
        raise ValueError(f"Index fingerprint differs: {output_path}. Choose a fresh index path.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + f".building-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"Partial build exists: {temporary}")
    start = time.perf_counter()
    connection = sqlite3.connect(temporary)
    connection.executescript("""
        PRAGMA journal_mode=OFF;
        PRAGMA synchronous=OFF;
        PRAGMA temp_store=FILE;
        PRAGMA cache_size=-65536;
        CREATE TABLE records(id TEXT, name TEXT, core TEXT, address TEXT, country TEXT,
            digits TEXT, name_digits TEXT, address_digits TEXT, postcodes TEXT, prefix TEXT, folded TEXT);
        CREATE VIRTUAL TABLE words USING fts5(name, address, digits,
            content='records', content_rowid='rowid', tokenize='unicode61 remove_diacritics 2');
        CREATE VIRTUAL TABLE chars USING fts5(folded, content='records', content_rowid='rowid', tokenize='trigram', detail='none');
        CREATE VIRTUAL TABLE word_vocab USING fts5vocab(words, 'col');
        CREATE VIRTUAL TABLE char_vocab USING fts5vocab(chars, 'row');
        CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT);
    """)
    count = 0
    def insert_batch(batch):
        nonlocal count
        old_count = count
        connection.executemany("INSERT INTO records VALUES (?,?,?,?,?,?,?,?,?,?,?)", batch)
        count += len(batch)
        connection.execute("INSERT INTO words(rowid,name,address,digits) SELECT rowid,name,address,digits FROM records WHERE rowid>?", (old_count,))
        connection.execute("INSERT INTO chars(rowid,folded) SELECT rowid,folded FROM records WHERE rowid>?", (old_count,))
        connection.commit()
        if count % 100_000 == 0:
            print(json.dumps({"stage": "index_build", "source": source, "records": count, "seconds": round(time.perf_counter() - start, 1)}), flush=True)
    try:
        with source_path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream, delimiter="\t")
            if tuple(reader.fieldnames or ()) != COLUMNS:
                raise ValueError(f"Unexpected source header: {reader.fieldnames}")
            batch = []
            for row in reader:
                if None in row or any(value is None for value in row.values()):
                    raise ValueError(f"Malformed TSV row at line {reader.line_num}")
                record = normalize_record(row)
                if not record.entity_id.startswith(source + "-"):
                    raise ValueError(f"Unexpected ID {record.entity_id}")
                nd, ad = sorted(record.name_digits), sorted(record.address_digits)
                batch.append((record.entity_id, record.name, record.core, record.address, record.country,
                    " ".join(["n" + t for t in nd] + ["a" + t for t in ad]), " ".join(nd), " ".join(ad),
                    " ".join(sorted(record.postcode_candidates)), "".join(record.core.split())[:8], record.retrieval_name))
                if len(batch) >= batch_size:
                    insert_batch(batch)
                    batch.clear()
            if batch:
                insert_batch(batch)
        after = source_path.stat()
        if (after.st_size, after.st_mtime_ns) != (stat.st_size, stat.st_mtime_ns):
            raise ValueError("Source file changed during indexing")
        print(json.dumps({"stage": "index_finalize", "source": source, "records": count}), flush=True)
        connection.executescript("""
            CREATE UNIQUE INDEX records_id ON records(id);
            CREATE INDEX records_name ON records(name);
            CREATE INDEX records_core ON records(core);
            CREATE INDEX records_prefix ON records(prefix);
        """)
        connection.execute("INSERT INTO words(words) VALUES ('optimize')")
        connection.execute("INSERT INTO chars(chars) VALUES ('optimize')")
        meta = {"fingerprint": fingerprint, "complete": True, "records": count, "seconds": time.perf_counter() - start,
                "sqlite_version": sqlite3.sqlite_version, "frequency_policy": "label-free, local to this full target source"}
        connection.execute("INSERT INTO metadata VALUES ('build',?)", (json.dumps(meta),))
        connection.commit()
        connection.close()
        temporary.replace(output_path)
        ensure_frequency_index(output_path)
        print(json.dumps({"stage": "index_ready", "source": source, "records": count, "seconds": round(meta["seconds"], 1)}), flush=True)
        return meta
    except BaseException:
        connection.close()
        raise


def trigrams(text: str) -> frozenset[str]:
    return frozenset(text[i:i+3] for i in range(max(0, len(text) - 2)))


class DiskSourceIndex:
    def __init__(self, path: str | Path, config: DiskSearchConfig | None = None, *, aliases=None):
        self.path, self.config = Path(path).resolve(), config or DiskSearchConfig()
        self.aliases = aliases
        if self.config.use_name_aliases and aliases is None:
            raise ValueError("The fitted name alias model is required by this search configuration")
        self.connection = sqlite3.connect(f"{self.path.as_uri()}?mode=ro", uri=True)
        self.connection.execute("PRAGMA cache_size=-65536")
        self.connection.execute("PRAGMA temp_store=FILE")
        self.meta = json.loads(self.connection.execute("SELECT value FROM metadata WHERE key='build'").fetchone()[0])
        if not self.meta["complete"]:
            raise ValueError("Incomplete target index")
        self.source, self.record_count = self.meta["fingerprint"]["source"], self.meta["records"]
        self._df_cache = OrderedDict()
        self.frequency_table = "word_vocab"
        if frequency_path(self.path).exists():
            self.connection.execute("ATTACH DATABASE ? AS stats",(f"{frequency_path(self.path).as_uri()}?mode=ro",))
            self.frequency_table = "stats.frequencies"
        self.anchors = None
        if self.config.retrieval_mode == "anchored":
            from .anchor_index import AnchorIndex
            self.anchors = AnchorIndex(self.path, aliases)

    def close(self):
        self.connection.close()
        if self.anchors is not None:
            self.anchors.close()

    def df(self, column: str, term: str) -> int:
        key = (column, term)
        if key in self._df_cache:
            self._df_cache.move_to_end(key)
            return self._df_cache[key]
        if column == "character":
            row = self.connection.execute("SELECT doc FROM char_vocab WHERE term=?", (term,)).fetchone()
        else:
            row = self.connection.execute(f"SELECT doc FROM {self.frequency_table} WHERE term=? AND col=?", (term, column)).fetchone()
        value = row[0] if row else 0
        self.cache_frequency(key, value)
        return value

    def cache_frequency(self, key, value):
        self._df_cache[key] = value
        self._df_cache.move_to_end(key)
        if len(self._df_cache) > 100_000:
            self._df_cache.popitem(last=False)

    def prefetch_frequencies(self, records):
        keys = {(column, accent_fold(token)) for record in records
                for column, tokens in (("name", record.name_tokens), ("address", record.address_tokens)) for token in tokens}
        unknown = {key for key in keys if key not in self._df_cache}
        terms = sorted({term for column, term in unknown})
        found = {}
        for start in range(0, len(terms), 500):
            chunk = terms[start:start+500]
            placeholders = ",".join("?" for _ in chunk)
            for term, column, frequency in self.connection.execute(f"SELECT term,col,doc FROM {self.frequency_table} WHERE term IN ({placeholders})", chunk):
                found[(column, term)] = frequency
        for key in sorted(unknown):
            self.cache_frequency(key, found.get(key, 0))

    def idf(self, column: str, term: str) -> float:
        return math.log((self.record_count + 1) / (self.df(column, term) + 1)) + 1

    def rare_terms(self, column: str, terms, cap: int, limit: int) -> list[str]:
        # FTS unicode61 accent folding is used for word lookup as well.
        values = {accent_fold(t) if column != "character" else t for t in terms}
        eligible = [(self.df(column, t), t) for t in values]
        return [t for df, t in sorted(eligible) if 0 < df <= cap][:limit]

    def search(self, expression: str, *, character: bool = False, country: str | None = None) -> list[int]:
        table = "chars" if character else "words"
        if country:
            return [row[0] for row in self.connection.execute(
                f"SELECT {table}.rowid FROM {table} JOIN records ON records.rowid={table}.rowid WHERE {table} MATCH ? AND (records.country=? OR records.country='') ORDER BY {table}.rank LIMIT ?",
                (expression, country, self.config.path_top_k))]
        return [row[0] for row in self.connection.execute(
            f"SELECT rowid FROM {table} WHERE {table} MATCH ? ORDER BY rank LIMIT ?", (expression, self.config.path_top_k))]

    @staticmethod
    def expression(column: str, terms: list[str]) -> str:
        prefix = column + " : " if column else ""
        return prefix + "(" + " OR ".join('"' + t.replace('"', '""') + '"' for t in terms) + ")"

    def record(self, rowid: int) -> BlockingRecord:
        row = self.connection.execute("SELECT id,name,core,address,country,name_digits,address_digits,postcodes,folded FROM records WHERE rowid=?", (rowid,)).fetchone()
        return self.record_values(row)

    @staticmethod
    def record_values(row) -> BlockingRecord:
        return BlockingRecord(row[0], row[1], row[2], row[3], row[4], frozenset(row[1].split()), frozenset(row[3].split()),
            frozenset(row[5].split()), frozenset(row[6].split()), frozenset(row[7].split()), row[8])

    def weighted_overlap(self, left, right, column: str) -> float:
        if not left or not right:
            return 0.0
        union = sorted(left | right)
        weights = {t: self.idf(column, accent_fold(t)) for t in union}
        return sum(weights[t] for t in sorted(left & right)) / sum(weights.values())

    def address_containment(self, left: str, right: str) -> float:
        # Preserve primary Unicode views. This alternate evidence view allows
        # shared Latin/numeric components to survive a transliterated component.
        a = {t for t in accent_fold(left).split() if t.isascii()}
        b = {t for t in accent_fold(right).split() if t.isascii()}
        shared = a & b
        if len(shared) < 2:
            return 0.0
        weights = {t: self.idf("address", t) for t in sorted(a | b)}
        if max(weights[t] for t in shared) < 4:
            return 0.0
        denominator = min(sum(weights[t] for t in sorted(a)), sum(weights[t] for t in sorted(b)))
        return sum(weights[t] for t in sorted(shared)) / denominator if denominator else 0.0

    @lru_cache(maxsize=10_000)
    def family_count(self, canonical: str) -> int:
        variants = self.aliases.variants(canonical) if self.aliases else [canonical]
        placeholders = ",".join("?" for _ in variants)
        return self.connection.execute(f"SELECT count(*) FROM records WHERE core IN ({placeholders})", variants).fetchone()[0]

    def resolved_name(self, record: BlockingRecord):
        core = record.core or record.name
        if self.aliases and not record.entity_id.startswith("S1-"):
            return self.aliases.resolve(core)
        return core, 0.0, 0

    def rerank(self, left: BlockingRecord, right: BlockingRecord, paths: tuple[str, ...]) -> float:
        name = self.weighted_overlap(left.name_tokens, right.name_tokens, "name")
        address = self.weighted_overlap(left.address_tokens, right.address_tokens, "address")
        a, b = trigrams(left.retrieval_name), trigrams(right.retrieval_name)
        char = len(a & b) / len(a | b) if a and b else 0.0
        digits = len(left.address_digits & right.address_digits) / len(left.address_digits | right.address_digits) if left.address_digits and right.address_digits else 0.0
        if self.config.reranker_version in ("v2", "v3", "v4"):
            # Name equality cannot identify a branch. Strong address evidence
            # must survive when the name is transliterated or a trade name.
            name_score = .4 * name + .3 * char + .3 * fuzz.token_sort_ratio(left.name, right.name) / 100
            name_score = max(name_score, .9 * fuzz.token_sort_ratio(left.core, right.core) / 100 if left.core and right.core else 0)
            address_score = (.35 * address + .35 * fuzz.token_set_ratio(left.address, right.address) / 100
                + .30 * fuzz.token_sort_ratio(left.address, right.address) / 100) if left.address and right.address else 0
            if self.config.reranker_version in ("v3", "v4"):
                address_score = max(address_score, .9 * self.address_containment(left.address, right.address))
            if self.config.reranker_version == "v4":
                lc, _, _ = self.resolved_name(left)
                rc, _, _ = self.resolved_name(right)
                if lc and rc:
                    name_score = max(name_score, .95 * fuzz.token_set_ratio(lc, rc) / 100)
                if self.aliases:
                    probability, _ = self.aliases.evidence(right.core, left.core)
                    if probability:
                        name_score = max(name_score, .8 + .15 * probability)
            ld = {t.lstrip("0") or "0" for t in left.address_digits}
            rd = {t.lstrip("0") or "0" for t in right.address_digits}
            number_score = len(ld & rd) / len(ld | rd) if ld and rd else 0
            score = max(.58 * name_score + .35 * address_score + .07 * number_score, .92 * address_score + .08 * number_score)
            score -= .08 * bool(ld and rd and not ld & rd)
            score += .02 * bool(left.name and left.name == right.name) + .02 * bool(left.core and left.core == right.core)
            score += .01 * math.log1p(len(paths))
            score -= .05 * bool(left.country and right.country and left.country != right.country)
            return float(score)
        score = .25 * name + .25 * char + .20 * fuzz.token_sort_ratio(left.name, right.name) / 100
        score += .15 * address + .10 * fuzz.token_sort_ratio(left.address, right.address) / 100 + .05 * digits
        score += .15 * bool(left.name and left.name == right.name) + .10 * bool(left.core and left.core == right.core)
        score += .02 * math.log1p(len(paths))
        score -= .05 * bool(left.country and right.country and left.country != right.country)
        return float(score)

    def query(self, reference: BlockingRecord, *, return_all: bool = False) -> list[tuple[Candidate, BlockingRecord]]:
        cfg = self.config
        paths = defaultdict(set)
        def add(path, rows):
            for rowid in rows:
                paths[rowid].add(path)
        if self.anchors is not None:
            for path, rows in self.anchors.query(reference, self).items():
                add(path, rows)
            return self.finish_query(reference, paths, return_all=return_all)
        for path, column, value in (("exact_name", "name", reference.name), ("exact_core", "core", reference.core), ("core_prefix", "prefix", "".join(reference.core.split())[:8])):
            if value:
                # Bounded exact lookup; lexical/address paths provide additional
                # branch evidence when identical names occur in large blocks.
                add(path, [r[0] for r in self.connection.execute(f"SELECT rowid FROM records WHERE {column}=? LIMIT ?", (value, cfg.path_top_k))])
        names = self.rare_terms("name", reference.name_tokens, cfg.max_word_df, cfg.query_tokens)
        addresses = self.rare_terms("address", reference.address_tokens, cfg.max_address_df, cfg.query_tokens)
        numbers = ["n" + t for t in reference.name_digits] + ["a" + t for t in reference.address_digits]
        digits = self.rare_terms("digits", numbers, cfg.max_address_df, cfg.query_tokens)
        for path, col, terms in (("rare_name", "name", names), ("rare_address", "address", addresses), ("rare_digits", "digits", digits)):
            if terms:
                add(path, self.search(self.expression(col, terms)))
        if cfg.numeric_conjunctions:
            # Common house-number fragments can be distinctive jointly even
            # when neither passes the single-token rarity cutoff.
            all_digits = sorted((self.df("digits", "a" + t), "a" + t) for t in reference.address_digits)
            joint = [term for frequency, term in all_digits if frequency > 0][:2]
            if len(joint) == 2:
                add("rare_digits", self.search(" AND ".join(self.expression("digits", [term]) for term in joint)))
        if cfg.numeric_location_queries and reference.address_digits:
            known = [(self.df("digits", "a" + t), "a" + t) for t in reference.address_digits]
            number_terms = [t for frequency, t in sorted(known) if frequency > 0][:2]
            # Address-tail tokens are component-order evidence, not a city
            # dictionary. Every country label is accepted, including new labels.
            location_terms = list(dict.fromkeys(accent_fold(t) for t in reference.address.split()[-6:]
                if len(t) >= 3 and not t.isdecimal() and 0 < self.df("address", accent_fold(t)) <= self.record_count * .5))[-4:]
            if number_terms:
                numeric_expression = " AND ".join(self.expression("digits", [t]) for t in number_terms)
                for location in location_terms:
                    add("numeric_location", self.search(numeric_expression + " AND " + self.expression("address", [location]), country=reference.country))
        if self.aliases and cfg.use_name_aliases and reference.core:
            variants = self.aliases.variants(reference.core)
            if len(variants) > 1:
                # Each alias is a conjunction of words; legal-form insertion
                # and word transposition need not preserve an exact phrase.
                families = ["name : (" + " AND ".join('"' + t.replace('"','""') + '"' for t in alias.split()) + ")" for alias in variants if alias]
                family = "(" + " OR ".join(families) + ")"
                numbers = ["a" + t for t in sorted(reference.address_digits)]
                if numbers:
                    add("name_alias_numeric", self.search(family + " AND " + self.expression("digits", numbers), country=reference.country))
                if addresses:
                    add("name_alias_address", self.search(family + " AND " + self.expression("address", addresses), country=reference.country))
        if names and reference.postcode_candidates:
            pterms = ["a" + t for t in sorted(reference.postcode_candidates)]
            add("postcode_name", self.search(self.expression("name", names) + " AND " + self.expression("digits", pterms)))
        do_chars = cfg.character_mode == "always" or (cfg.character_mode == "fallback" and not names)
        if do_chars:
            grams = [g for g in trigrams(reference.retrieval_name) if all(c.isalnum() for c in g)]
            grams = self.rare_terms("character", grams, cfg.max_character_df, cfg.character_terms)
            if grams:
                add("name_character", self.search(self.expression("", grams), character=True))
        return self.finish_query(reference, paths, return_all=return_all)

    def finish_query(self, reference, paths, *, return_all=False):
        cfg = self.config
        candidates = []
        if not paths:
            return []
        positions = sorted(paths)
        placeholders = ",".join("?" for _ in positions)
        targets = {row[0]: self.record_values(row[1:]) for row in self.connection.execute(
            f"SELECT rowid,id,name,core,address,country,name_digits,address_digits,postcodes,folded FROM records WHERE rowid IN ({placeholders})", positions)}
        self.prefetch_frequencies([reference, *targets.values()])
        for rowid, provenance in paths.items():
            target = targets[rowid]
            path_tuple = tuple(sorted(provenance))
            candidates.append((Candidate(reference.entity_id, target.entity_id, self.source, self.rerank(reference, target, path_tuple), path_tuple), target))
        candidates.sort(key=lambda pair: (-pair[0].blocking_score, pair[0].candidate_entity_id))
        return [(Candidate(c.source1_entity_id, c.candidate_entity_id, c.candidate_source, c.blocking_score, c.blocking_paths, rank + 1), target)
                for rank, (c, target) in enumerate(candidates if return_all else candidates[:cfg.top_k])]
