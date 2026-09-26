"""Experimental bounded postings retrieval; never selected by the frozen runner.

Read complete small token lists without a per-query BM25 sort. Accumulate
field-specific evidence in NumPy, then rerank a bounded union of shortlists.
The cache is bounded by bytes. Oversized token lists are skipped, not truncated
by record order. All target records remain searchable through other keys.
"""
from collections import OrderedDict
from dataclasses import dataclass
import math
import time
from pathlib import Path
from functools import lru_cache
import ctypes

import numpy as np

from .disk_index import DiskSourceIndex
from ..preprocessing.normalize import accent_fold


@lru_cache(maxsize=100000)
def folded_token(token):
    return accent_fold(token)


@lru_cache(maxsize=20000)
def ascii_address_tokens(address):
    return frozenset(t for t in accent_fold(address).split() if t.isascii())


@dataclass(frozen=True)
class PostingsConfig:
    max_postings: int = 20000
    tokens_per_field: int = 6
    field_budget: int = 24
    fused_budget: int = 64
    cache_bytes: int = 64 * 1024**2

    def __post_init__(self):
        if any(type(v) is not int or v <= 0 for v in vars(self).values()):
            raise ValueError("Postings limits must be positive integers")


class PostingsSourceIndex(DiskSourceIndex):
    def __init__(self, path, config, *, aliases=None, postings_config=None, native_extension=None, ranking_model=None):
        super().__init__(path, config, aliases=aliases)
        self.postings_config = postings_config or PostingsConfig()
        self.ranking_model = ranking_model
        self._postings = OrderedDict()
        self._postings_bytes = 0
        self.native_extension = native_extension
        if native_extension:
            self.connection.enable_load_extension(True)
            try:
                self.connection.load_extension(str(Path(native_extension).resolve()))
            finally:
                self.connection.enable_load_extension(False)
            self._native = ctypes.CDLL(str(Path(native_extension).resolve()))
            self._native.er_reduce.restype = ctypes.c_int64
            self._native.er_reduce.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int64,
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_int64]
        self.profile = dict(retrieve_seconds=0., rerank_seconds=0., queries=0,
                            accumulated_records=0, reranked_records=0, cache_hits=0,
                            cache_misses=0, skipped_large_lists=0)
        self.connection.execute("PRAGMA main.mmap_size=2147418112")
        if self.frequency_table.startswith("stats."):
            self.connection.execute("PRAGMA stats.mmap_size=2147418112")

    @lru_cache(maxsize=100000)
    def idf(self, column, term):
        return super().idf(column, term)

    def weighted_overlap(self, left, right, column):
        if not left or not right:
            return 0.
        weights = {t: self.idf(column, folded_token(t)) for t in sorted(left | right)}
        return sum(weights[t] for t in sorted(left & right)) / sum(weights.values())

    def address_containment(self, left, right):
        a, b = ascii_address_tokens(left), ascii_address_tokens(right)
        shared = a & b
        if len(shared) < 2:
            return 0.
        weights = {t: self.idf("address", t) for t in sorted(a | b)}
        if max(weights[t] for t in shared) < 4:
            return 0.
        denominator = min(sum(weights[t] for t in sorted(a)), sum(weights[t] for t in sorted(b)))
        return sum(weights[t] for t in sorted(shared)) / denominator if denominator else 0.

    def prefetch_frequencies(self, records):
        keys = {(column, folded_token(token)) for record in records
                for column, tokens in (("name", record.name_tokens), ("address", record.address_tokens)) for token in tokens}
        unknown = {key for key in keys if key not in self._df_cache}
        terms = sorted({term for column, term in unknown})
        found = {}
        for start in range(0, len(terms), 500):
            chunk = terms[start:start + 500]
            placeholders = ",".join("?" for _ in chunk)
            for term, column, frequency in self.connection.execute(
                f"SELECT term,col,doc FROM {self.frequency_table} WHERE term IN ({placeholders})", chunk):
                found[(column, term)] = frequency
        for key in sorted(unknown):
            self.cache_frequency(key, found.get(key, 0))

    def finish_query(self, reference, paths, *, return_all=False):
        if self.ranking_model is None or return_all:
            return super().finish_query(reference, paths, return_all=return_all)
        from .cheap_ranker import rank_features
        from .contracts import Candidate
        positions = sorted(paths)
        if not positions:
            return []
        placeholders = ",".join("?" for _ in positions)
        targets = {row[0]: self.record_values(row[1:]) for row in self.connection.execute(
            f"SELECT rowid,id,name,core,address,country,name_digits,address_digits,postcodes,folded FROM records WHERE rowid IN ({placeholders})", positions)}
        features = np.asarray([rank_features(reference, targets[rowid], self) for rowid in positions], dtype=np.float32)
        probabilities = self.ranking_model.predict(features, num_threads=1)
        if not np.isfinite(probabilities).all(): raise ValueError("Nonfinite ranker scores")
        chosen = sorted(zip(positions, probabilities), key=lambda pair: (-pair[1], targets[pair[0]].entity_id))[:self.config.top_k]
        self.prefetch_frequencies([reference, *[targets[rowid] for rowid, _ in chosen]])
        result = []
        for rank, (rowid, probability) in enumerate(chosen, 1):
            target = targets[rowid]
            provenance = tuple(sorted(paths[rowid]))
            result.append((Candidate(reference.entity_id, target.entity_id, self.source,
                self.rerank(reference, target, provenance), provenance, rank), target))
        return result

    def postings(self, column, term):
        key = column, term
        if key in self._postings:
            self.profile["cache_hits"] += 1
            self._postings.move_to_end(key)
            return self._postings[key]
        self.profile["cache_misses"] += 1
        cap = self.postings_config.max_postings
        if column in ("name", "address", "digits"):
            # Exact FTS token semantics, including Unicode tokenizer behavior.
            expression = self.expression(column, [term])
            sql = "SELECT rowid FROM words WHERE words MATCH ? LIMIT ?"
            params = expression, cap + 1
        elif column == "core":
            sql = "SELECT rowid FROM records WHERE core=? LIMIT ?"
            params = term, cap + 1
        else:
            raise ValueError(column)
        if self.native_extension:
            blob = self.connection.execute("SELECT er_pack(rowid) FROM (" + sql + ")", params).fetchone()[0]
            rows = np.frombuffer(blob, dtype=np.int32)
        else:
            rows = np.fromiter((row[0] for row in self.connection.execute(sql, params)), dtype=np.int32)
        if len(rows) > cap:
            self.profile["skipped_large_lists"] += 1
            rows = np.empty(0, dtype=np.int32)
        # Account for empty arrays and Python/cache-key overhead as well.
        size = rows.nbytes + 256 + len(term.encode("utf-8"))
        self._postings[key] = rows, size
        self._postings_bytes += size
        while self._postings_bytes > self.postings_config.cache_bytes and len(self._postings) > 1:
            _, (_, old_size) = self._postings.popitem(last=False)
            self._postings_bytes -= old_size
        return rows, size

    @staticmethod
    def top_positions(scores, limit, ids):
        positive = np.flatnonzero(scores > 0)
        if len(positive) > limit:
            # Deterministic threshold ties: rowid is a tie break, never evidence.
            cutoff = np.partition(scores[positive], -limit)[-limit]
            high = positive[scores[positive] > cutoff]
            tied = positive[scores[positive] == cutoff]
            positive = np.concatenate((high, tied[np.argsort(ids[tied])[:limit-len(high)]]))
        return positive

    def query(self, reference, *, return_all=False):
        start = time.perf_counter()
        cfg = self.postings_config
        self.prefetch_frequencies([reference])
        pieces, field_ids, weights = [], [], []
        field_weight = np.zeros(4)
        def add(field, col, term, weight):
            rows, _ = self.postings(col, term)
            if len(rows):
                pieces.append(rows)
                field_ids.append(field)
                weights.append(float(np.float32(weight)))
                field_weight[field] += weight
        for field, col, terms in ((0, "name", reference.name_tokens),
                                  (1, "address", reference.address_tokens),
                                  (2, "digits", ["a" + t for t in reference.address_digits])):
            for term in self.rare_terms(col, terms, cfg.max_postings, cfg.tokens_per_field):
                add(field, col, term, math.log((self.record_count + 1) / (self.df(col, term) + 1)))
        variants = self.aliases.variants(reference.core) if self.aliases else [reference.core]
        for core in variants:
            if core:
                add(3, "core", core, 1.)
        if not pieces:
            self.profile["queries"] += 1
            self.profile["retrieve_seconds"] += time.perf_counter() - start
            return []
        if self.native_extension:
            ptrs = np.asarray([p.ctypes.data for p in pieces], dtype=np.uintp)
            lengths = np.asarray([len(p) for p in pieces], dtype=np.int64)
            fields = np.asarray(field_ids, dtype=np.int32)
            values = np.asarray(weights, dtype=np.float64)
            total = int(lengths.sum())
            capacity = 1 << (2 * total - 1).bit_length()
            slots = np.zeros(capacity, dtype=np.uint32)
            ids = np.empty(min(total, self.record_count), dtype=np.int32)
            evidence = np.empty((len(ids), 4), dtype=np.float64)
            count = self._native.er_reduce(ptrs.ctypes.data, lengths.ctypes.data,
                fields.ctypes.data, values.ctypes.data, len(pieces), ids.ctypes.data,
                evidence.ctypes.data, slots.ctypes.data, capacity, self.record_count)
            if count < 0:
                raise ValueError("Native accumulator rejected invalid postings")
            ids, evidence = ids[:count], evidence[:count]
        else:
            ids, inverse = np.unique(np.concatenate(pieces), return_inverse=True)
            fields = np.concatenate([np.full(len(p), f, dtype=np.int8) for p, f in zip(pieces, field_ids)])
            values = np.concatenate([np.full(len(p), w) for p, w in zip(pieces, weights)])
            evidence = np.bincount(inverse * 4 + fields, weights=values,
                                   minlength=len(ids) * 4).reshape(-1, 4)
        evidence[:, :3] /= np.maximum(field_weight[:3], 1e-9)
        evidence[:, 3] = evidence[:, 3] > 0
        name = np.maximum(evidence[:, 0], evidence[:, 3])
        address, numeric = evidence[:, 1], evidence[:, 2]
        fused = np.maximum(.58 * name + .35 * address + .07 * numeric,
                           .92 * address + .08 * numeric)
        selected = set(self.top_positions(fused, cfg.fused_budget, ids))
        for score in (name, address, name * numeric, evidence[:, 3]):
            selected.update(self.top_positions(score, cfg.field_budget, ids))
        paths = {}
        for pos in sorted(selected):
            paths[int(ids[pos])] = {label for column, label in enumerate(
                ("rare_name", "rare_address", "rare_digits", "exact_core")) if evidence[pos, column] > 0}
        self.profile["queries"] += 1
        self.profile["accumulated_records"] += len(ids)
        self.profile["reranked_records"] += len(paths)
        self.profile["retrieve_seconds"] += time.perf_counter() - start
        start = time.perf_counter()
        pairs = self.finish_query(reference, paths, return_all=return_all)
        self.profile["rerank_seconds"] += time.perf_counter() - start
        return pairs
