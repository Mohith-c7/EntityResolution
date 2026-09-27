"""Rare-token soft alignment evidence for a candidate pair.

The matcher already has token-set and sorted-string similarities. Those treat
every shared token alike, so a shared city or generic business word can look as
strong as a shared distinctive word, and a one-character typo in a distinctive
word can lose all of its support. This module adds a one-to-one soft token
alignment view that:

* weights each token by a label-free document frequency, so common tokens
  (cities, generic business vocabulary) carry less weight than rare ones;
* lets a close spelling (a typo or transliteration variant) support a token,
  scored below an exact match;
* enforces one-to-one alignment, so one generic word cannot cover several
  different words on the other side;
* keeps a separate address view that excludes tokens repeated in either
  record's business name, so a name word echoed in the address is not mistaken
  for street identity.

No labels are read. Document frequencies come from the supplied Source 1 texts
only, separately per observed country with a global fallback. Country is an
open string; there are no country-specific word lists or decision rules.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

from rapidfuzz import fuzz

from ..preprocessing.normalize import accent_fold
from .evidence import phonetic

STATISTICS_VERSION = "token-alignment-stats-v1"
FEATURE_VERSION = "token-alignment-v1"

# Fixed feature order. The batch writer and any consumer must use this order.
FEATURE_NAMES = [
    # Name alignment, primary (accent-folded) view.
    "align_name_coverage_l2r",
    "align_name_coverage_r2l",
    "align_name_coverage_sym",
    "align_name_rarest_supported",
    "align_name_unmatched_distinctive",
    # Name alignment, phonetic view.
    "align_name_phon_coverage_sym",
    "align_name_phon_unmatched_distinctive",
    # Address alignment.
    "align_addr_coverage_l2r",
    "align_addr_coverage_r2l",
    "align_addr_coverage_sym",
    "align_addr_unmatched_distinctive",
    # Address view excluding tokens repeated in either business name.
    "align_addrx_coverage_sym",
    "align_addrx_unmatched_distinctive",
    # Missing-evidence flags (1 = evidence absent, never a contradiction).
    "align_name_missing",
    "align_addr_missing",
    "align_addrx_missing",
]

# A token pair is a candidate soft match at or above this similarity; an exact
# match always qualifies. Below this the tokens are treated as different.
SOFT_MATCH_THRESHOLD = 80.0
# Score given to a soft (non-exact) match, as a fraction of an exact match.
SOFT_MATCH_DISCOUNT = 0.5
# Bound on the per-side token count that enters the fuzzy matrix. Exact matches
# are resolved first; only the remaining tokens (at most this many per side)
# are compared pairwise, keeping per-pair work bounded.
MAX_FUZZY_TOKENS = 8
# A token is "distinctive" once its inverse document weight reaches this. Used
# for the unmatched-distinctive features.
DISTINCTIVE_IDF = 4.0


@dataclass(frozen=True)
class AlignmentStatistics:
    """Label-free per-country document frequencies with a global fallback."""

    version: str
    record_count: int
    # term -> document frequency over the whole supplied corpus.
    global_df: Mapping[str, int]
    # country -> (record_count, {term -> df}). Country keys are open strings.
    country_df: Mapping[str, tuple[int, Mapping[str, int]]] = field(default_factory=dict)
    # source file hashes for provenance.
    source_sha256: Mapping[str, str] = field(default_factory=dict)

    def df(self, term: str, country: str) -> int:
        if country and country in self.country_df:
            return self.country_df[country][1].get(term, 0)
        return self.global_df.get(term, 0)

    def count(self, country: str) -> int:
        if country and country in self.country_df:
            return self.country_df[country][0]
        return self.record_count

    def idf(self, term: str, country: str) -> float:
        return math.log((self.count(country) + 1) / (self.df(term, country) + 1)) + 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "record_count": self.record_count,
            "global_df": dict(self.global_df),
            "country_df": {c: {"record_count": n, "df": dict(df)} for c, (n, df) in self.country_df.items()},
            "source_sha256": dict(self.source_sha256),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AlignmentStatistics":
        if value.get("version") != STATISTICS_VERSION:
            raise ValueError(f"Unsupported statistics version {value.get('version')!r}")
        return cls(
            version=value["version"],
            record_count=int(value["record_count"]),
            global_df={k: int(v) for k, v in value["global_df"].items()},
            country_df={c: (int(d["record_count"]), {k: int(v) for k, v in d["df"].items()})
                        for c, d in value.get("country_df", {}).items()},
            source_sha256=dict(value.get("source_sha256", {})),
        )


def _tokens(text: str) -> list[str]:
    return [t for t in accent_fold(text).split() if t]


def _alignment(left_tokens, right_tokens, weight) -> tuple[float, float, float, float]:
    """One-to-one soft alignment between two token multisets.

    weight(term) is the per-token importance (IDF). Returns
    (coverage_l2r, coverage_r2l, rarest_supported_weight, unmatched_distinctive_weight).

    Exact matches are resolved first; then a bounded fuzzy matrix over the
    remaining tokens. Each token supports at most one token on the other side.
    """
    if not left_tokens or not right_tokens:
        return 0.0, 0.0, 0.0, 0.0

    # Multiset remaining counts, exact matches removed one-to-one.
    from collections import Counter

    left_counts = Counter(left_tokens)
    right_counts = Counter(right_tokens)
    matched_weight_left = 0.0
    matched_weight_right = 0.0
    supported_weights: list[float] = []
    for token in sorted(left_counts & right_counts):
        n = min(left_counts[token], right_counts[token])
        w = weight(token)
        matched_weight_left += n * w
        matched_weight_right += n * w
        supported_weights.extend([w] * n)
        left_counts[token] -= n
        right_counts[token] -= n

    left_remaining = [t for t in sorted(left_counts) for _ in range(left_counts[t])][:MAX_FUZZY_TOKENS]
    right_remaining = [t for t in sorted(right_counts) for _ in range(right_counts[t])][:MAX_FUZZY_TOKENS]

    # Bounded fuzzy matrix over remaining tokens; greedy best-first one-to-one.
    pairs = []
    for i, lt in enumerate(left_remaining):
        for j, rt in enumerate(right_remaining):
            score = fuzz.ratio(lt, rt)
            if score >= SOFT_MATCH_THRESHOLD:
                pairs.append((score, min(weight(lt), weight(rt)), i, j))
    pairs.sort(key=lambda row: (-row[0], -row[1], row[2], row[3]))
    used_left: set[int] = set()
    used_right: set[int] = set()
    for score, w, i, j in pairs:
        if i in used_left or j in used_right:
            continue
        used_left.add(i)
        used_right.add(j)
        soft = SOFT_MATCH_DISCOUNT * w * (score / 100.0)
        matched_weight_left += soft
        matched_weight_right += soft
        supported_weights.append(soft)

    total_left = sum(weight(t) for t in left_tokens)
    total_right = sum(weight(t) for t in right_tokens)
    coverage_l2r = matched_weight_left / total_left if total_left else 0.0
    coverage_r2l = matched_weight_right / total_right if total_right else 0.0

    rarest_supported = max(supported_weights) if supported_weights else 0.0

    # Distinctive tokens (high IDF) that found no support on the other side.
    matched_left_tokens = set()
    matched_right_tokens = set()
    lc = Counter(left_tokens)
    rc = Counter(right_tokens)
    for token in lc & rc:
        for _ in range(min(lc[token], rc[token])):
            matched_left_tokens.add(token)
            matched_right_tokens.add(token)
    for _, _, i, j in [(s, w, i, j) for s, w, i, j in pairs if i in used_left and j in used_right]:
        matched_left_tokens.add(left_remaining[i])
        matched_right_tokens.add(right_remaining[j])
    unmatched = [weight(t) for t in set(left_tokens) | set(right_tokens)
                 if weight(t) >= DISTINCTIVE_IDF and t not in matched_left_tokens and t not in matched_right_tokens]
    unmatched_distinctive = max(unmatched) if unmatched else 0.0

    return coverage_l2r, coverage_r2l, rarest_supported, unmatched_distinctive


def build_alignment_features(left, right, statistics: AlignmentStatistics) -> dict[str, float]:
    """Build the fixed alignment feature vector for one candidate pair.

    left and right are BlockingRecord-like objects with ``name``, ``address``
    and ``country`` attributes. Returns a dict keyed by FEATURE_NAMES. Missing
    fields set explicit flags and are never recorded as contradictions.
    """
    country = getattr(left, "country", "") or ""

    def weight_view(view: str):
        def weight(term: str) -> float:
            return statistics.idf(term, country)
        return weight

    name_weight = weight_view("name")

    left_name = _tokens(getattr(left, "name", "") or "")
    right_name = _tokens(getattr(right, "name", "") or "")
    left_addr = _tokens(getattr(left, "address", "") or "")
    right_addr = _tokens(getattr(right, "address", "") or "")

    features: dict[str, float] = {}

    # Name, primary view.
    name_missing = not left_name or not right_name
    if name_missing:
        features.update(dict.fromkeys(
            ["align_name_coverage_l2r", "align_name_coverage_r2l", "align_name_coverage_sym",
             "align_name_rarest_supported", "align_name_unmatched_distinctive"], 0.0))
    else:
        l2r, r2l, rarest, unmatched = _alignment(left_name, right_name, name_weight)
        features["align_name_coverage_l2r"] = l2r
        features["align_name_coverage_r2l"] = r2l
        features["align_name_coverage_sym"] = min(l2r, r2l)
        features["align_name_rarest_supported"] = rarest
        features["align_name_unmatched_distinctive"] = unmatched
    features["align_name_missing"] = float(name_missing)

    # Name, phonetic view (transliteration-robust).
    left_phon = [t for t in phonetic(getattr(left, "name", "") or "").split() if t]
    right_phon = [t for t in phonetic(getattr(right, "name", "") or "").split() if t]
    if not left_phon or not right_phon:
        features["align_name_phon_coverage_sym"] = 0.0
        features["align_name_phon_unmatched_distinctive"] = 0.0
    else:
        l2r, r2l, _, unmatched = _alignment(left_phon, right_phon, name_weight)
        features["align_name_phon_coverage_sym"] = min(l2r, r2l)
        features["align_name_phon_unmatched_distinctive"] = unmatched

    # Address, full view.
    addr_missing = not left_addr or not right_addr
    if addr_missing:
        features.update(dict.fromkeys(
            ["align_addr_coverage_l2r", "align_addr_coverage_r2l", "align_addr_coverage_sym",
             "align_addr_unmatched_distinctive"], 0.0))
    else:
        l2r, r2l, _, unmatched = _alignment(left_addr, right_addr, name_weight)
        features["align_addr_coverage_l2r"] = l2r
        features["align_addr_coverage_r2l"] = r2l
        features["align_addr_coverage_sym"] = min(l2r, r2l)
        features["align_addr_unmatched_distinctive"] = unmatched
    features["align_addr_missing"] = float(addr_missing)

    # Address view excluding tokens repeated in either record's business name.
    name_token_set = set(left_name) | set(right_name)
    left_addrx = [t for t in left_addr if t not in name_token_set]
    right_addrx = [t for t in right_addr if t not in name_token_set]
    addrx_missing = addr_missing or not left_addrx or not right_addrx
    if addrx_missing:
        features["align_addrx_coverage_sym"] = 0.0
        features["align_addrx_unmatched_distinctive"] = 0.0
    else:
        l2r, r2l, _, unmatched = _alignment(left_addrx, right_addrx, name_weight)
        features["align_addrx_coverage_sym"] = min(l2r, r2l)
        features["align_addrx_unmatched_distinctive"] = unmatched
    features["align_addrx_missing"] = float(addrx_missing)

    # Guarantee the fixed schema, finiteness and order.
    return {name: float(features.get(name, 0.0)) for name in FEATURE_NAMES}
