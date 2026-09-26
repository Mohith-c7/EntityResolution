"""
Pairwise feature builder — produces a 36-element numeric feature vector per candidate pair.

Feature layout (0-indexed):
  0–9   Name similarity (10)
  10–17 Address similarity (8)
  18–23 Digit/number (6)
  24–27 Agreement flags (4)
  28–31 Length covariates (4)
  32–35 Blocking / meta (4)
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from src.preprocessing.normalize import (
    extract_digits,
    name_core,
    normalize_address,
    normalize_country,
    normalize_name,
)

# ---------------------------------------------------------------------------
# Optional heavy imports — graceful degradation during unit tests if missing
# ---------------------------------------------------------------------------
try:
    import jellyfish
    _HAS_JELLYFISH = True
except ImportError:
    _HAS_JELLYFISH = False

try:
    from rapidfuzz import fuzz as rfuzz
    from rapidfuzz.distance import Levenshtein as rfLev
    _HAS_RAPIDFUZZ = True
except ImportError:
    _HAS_RAPIDFUZZ = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


# ---------------------------------------------------------------------------
# Feature names — exported constant
# ---------------------------------------------------------------------------
FEATURE_NAMES: list[str] = [
    # Name (0–9)
    "exact_name_match",
    "jaro_winkler_name",
    "token_sort_ratio",
    "token_set_ratio",
    "tfidf_cosine_name",
    "lcs_ratio_name",
    "levenshtein_norm_name",
    "qgram_similarity_name",
    "phonetic_match",
    "initials_match",
    # Address (10–17)
    "exact_address_match",
    "jaro_winkler_address",
    "token_overlap_address",
    "street_name_similarity",
    "city_postal_exact",
    "address_component_agreement",
    "tfidf_cosine_address",
    "postcode_exact",
    # Digit/numeric (18–23) — derived from name and address only
    "name_digit_exact_match",
    "name_digit_levenshtein_norm",
    "addr_digit_exact_match",
    "addr_digit_levenshtein_norm",
    "name_digit_token_overlap",
    "addr_digit_token_overlap",
    # Agreement (24–27)
    "country_match",
    "country_contradiction",
    "source_indicator",
    "name_country_combined",
    # Length (28–31)
    "s1_name_length",
    "cand_name_length",
    "name_length_diff",
    "name_length_ratio",
    # Meta (32–35)
    "blocking_score",
    "distance_rank_bin",
    "is_s1_s2",
    "is_s1_s3",
]

assert len(FEATURE_NAMES) == 36, f"Expected 36 features, got {len(FEATURE_NAMES)}"

# Default values per feature index
_DEFAULTS: list[float] = [
    # Name
    0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0,
    # Address
    0, 0.0, 0.0, 0.0, 0, 0, 0.0, 0,
    # Digit/numeric (from name+address only)
    0, 0.0, 0, 0.0, 0.0, 0.0,
    # Agreement
    -1, 0, 0, 0.0,
    # Length
    0, 0, 0, 0.0,
    # Meta
    0.0, 3, 0, 0,
]

assert len(_DEFAULTS) == 36


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe(value: Any, default: float) -> float:
    """Return value as float, or default if None/NaN."""
    if value is None:
        return default
    try:
        f = float(value)
        return default if math.isnan(f) else f
    except (ValueError, TypeError):
        return default


def _safe_str(value: Any) -> str:
    """Return str(value) or '' if None."""
    if value is None:
        return ""
    s = str(value).strip()
    return "" if s.lower() in ("nan", "none") else s


def _jaro_winkler(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if _HAS_JELLYFISH:
        return jellyfish.jaro_winkler_similarity(a, b)
    # Fallback: exact match only
    return 1.0 if a == b else 0.0


def _token_sort_ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if _HAS_RAPIDFUZZ:
        return rfuzz.token_sort_ratio(a, b) / 100.0
    ta, tb = sorted(a.split()), sorted(b.split())
    return 1.0 if ta == tb else 0.0


def _token_set_ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if _HAS_RAPIDFUZZ:
        return rfuzz.token_set_ratio(a, b) / 100.0
    sa, sb = set(a.split()), set(b.split())
    if not sa or not sb:
        return 0.0
    intersection = sa & sb
    return len(intersection) / max(len(sa), len(sb))


def _tfidf_cosine(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if not _HAS_SKLEARN:
        return 1.0 if a == b else 0.0
    try:
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3))
        mat = vec.fit_transform([a, b])
        return float((mat[0] * mat[1].T).toarray()[0][0])
    except Exception:
        return 0.0


def _lcs_ratio(a: str, b: str) -> float:
    """Longest common subsequence ratio."""
    if not a or not b:
        return 0.0
    m, n = len(a), len(b)
    # O(mn) DP — fine for typical business name lengths
    prev = [0] * (n + 1)
    for c in a:
        curr = [0] * (n + 1)
        for j, d in enumerate(b, 1):
            curr[j] = prev[j - 1] + 1 if c == d else max(curr[j - 1], prev[j])
        prev = curr
    lcs_len = prev[n]
    return (2 * lcs_len) / (m + n)


def _levenshtein_norm(a: str, b: str) -> float:
    """Normalized Levenshtein similarity (1 - edit_dist / max_len)."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    if _HAS_RAPIDFUZZ:
        dist = rfLev.distance(a, b)
    else:
        # Simple DP fallback
        la, lb = len(a), len(b)
        dp = list(range(lb + 1))
        for i, ca in enumerate(a, 1):
            ndp = [i] + [0] * lb
            for j, cb in enumerate(b, 1):
                ndp[j] = dp[j - 1] if ca == cb else 1 + min(dp[j], ndp[j - 1], dp[j - 1])
            dp = ndp
        dist = dp[lb]
    max_len = max(len(a), len(b))
    return 1.0 - dist / max_len


def _qgram_similarity(a: str, b: str, q: int = 3) -> float:
    if len(a) < q or len(b) < q:
        return 1.0 if a == b else 0.0
    def qgrams(s):
        return [s[i:i+q] for i in range(len(s) - q + 1)]
    qa, qb = qgrams(a), qgrams(b)
    sa, sb = set(qa), set(qb)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def _phonetic_match(a: str, b: str) -> int:
    """1 if Soundex or Metaphone codes match, 0 otherwise, -1 if unavailable."""
    if not a or not b:
        return -1
    if _HAS_JELLYFISH:
        try:
            return int(jellyfish.soundex(a) == jellyfish.soundex(b) or
                       jellyfish.metaphone(a) == jellyfish.metaphone(b))
        except Exception:
            return -1
    return -1


def _initials(name: str) -> str:
    return "".join(w[0] for w in name.split() if w)


def _token_overlap(a: str, b: str) -> float:
    sa, sb = set(a.split()), set(b.split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _levenshtein_norm_digits(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    return _levenshtein_norm(a, b)


def _distance_rank_bin(rank: Any) -> int:
    """Map candidate rank to bin: 1 → 1, 2–5 → 2, 6–20 → 3."""
    try:
        r = int(rank)
    except (TypeError, ValueError):
        return 3
    if r == 1:
        return 1
    if r <= 5:
        return 2
    return 3


# ---------------------------------------------------------------------------
# Name features (indices 0–9)
# ---------------------------------------------------------------------------

def _name_features(s1: dict, cand: dict) -> list[float]:
    raw_s1 = _safe_str(s1.get("name"))
    raw_cand = _safe_str(cand.get("name"))

    n1 = normalize_name(raw_s1)
    n2 = normalize_name(raw_cand)

    exact = float(n1 == n2 and n1 != "")
    jw = _jaro_winkler(n1, n2)
    tsr = _token_sort_ratio(n1, n2)
    tse = _token_set_ratio(n1, n2)
    tfidf = _tfidf_cosine(n1, n2)
    lcs = _lcs_ratio(n1, n2)
    lev = _levenshtein_norm(n1, n2)
    qg = _qgram_similarity(n1, n2)
    phon = float(_phonetic_match(name_core(raw_s1), name_core(raw_cand)))
    init = float(_initials(n1) == _initials(n2) and bool(_initials(n1)))

    return [exact, jw, tsr, tse, tfidf, lcs, lev, qg, phon, init]


# ---------------------------------------------------------------------------
# Address features (indices 10–17)
# ---------------------------------------------------------------------------

def _address_features(s1: dict, cand: dict) -> list[float]:
    raw_s1 = _safe_str(s1.get("address"))
    raw_cand = _safe_str(cand.get("address"))

    a1 = normalize_address(raw_s1)
    a2 = normalize_address(raw_cand)

    exact = float(a1 == a2 and a1 != "")
    jw = _jaro_winkler(a1, a2)
    tok_overlap = _token_overlap(a1, a2)

    # Street name: first comma-delimited component
    street1 = a1.split(",")[0] if "," in a1 else a1
    street2 = a2.split(",")[0] if "," in a2 else a2
    street_sim = _jaro_winkler(street1, street2)

    # Postal/city evidence extracted from the address string itself
    addr_tokens_s1 = set(a1.split())
    addr_tokens_cand = set(a2.split())

    # Digit tokens from address (proxy for postcodes/numbers)
    postal1 = extract_digits(raw_s1)
    postal2 = extract_digits(raw_cand)
    city_postal = float(postal1 == postal2 and postal1 != "")

    # Address component agreement: matching non-empty token subsets
    # Use first, middle and last tokens as proxies for street/city/postcode
    def _part(tokens: set, idx: int) -> str:
        lst = sorted(tokens)
        return lst[idx] if len(lst) > abs(idx) else ""

    parts_s1 = sorted(addr_tokens_s1)
    parts_cand = sorted(addr_tokens_cand)
    agree = sum(
        1 for i in [0, len(parts_s1)//2, -1]
        if parts_s1 and parts_cand and
        _part(addr_tokens_s1, i) == _part(addr_tokens_cand, i) and
        _part(addr_tokens_s1, i) != ""
    )

    tfidf = _tfidf_cosine(a1, a2)
    postcode = float(postal1 == postal2 and postal1 != "")

    return [exact, jw, tok_overlap, street_sim, city_postal, float(agree), tfidf, postcode]


# ---------------------------------------------------------------------------
# Digit/numeric features (indices 18–23)
# Derived strictly from business_name and business_address digit tokens.
# The challenge dataset has no phone/tax_id/employee_count/year_founded columns.
# ---------------------------------------------------------------------------

def _digit_features(s1: dict, cand: dict) -> list[float]:
    """
    Extract numeric tokens from name and address fields only.
    Features:
      18: name digit exact match flag
      19: name digit Levenshtein norm
      20: address digit exact match flag
      21: address digit Levenshtein norm
      22: name digit token Jaccard overlap
      23: address digit token Jaccard overlap
    """
    name_digits_s1 = extract_digits(_safe_str(s1.get("name")))
    name_digits_cand = extract_digits(_safe_str(cand.get("name")))
    addr_digits_s1 = extract_digits(_safe_str(s1.get("address")))
    addr_digits_cand = extract_digits(_safe_str(cand.get("address")))

    name_exact = float(name_digits_s1 == name_digits_cand and name_digits_s1 != "")
    name_lev = _levenshtein_norm(name_digits_s1, name_digits_cand) if (name_digits_s1 or name_digits_cand) else 0.0

    addr_exact = float(addr_digits_s1 == addr_digits_cand and addr_digits_s1 != "")
    addr_lev = _levenshtein_norm(addr_digits_s1, addr_digits_cand) if (addr_digits_s1 or addr_digits_cand) else 0.0

    # Token Jaccard overlap on digit token sets
    def _digit_jaccard(a: str, b: str) -> float:
        sa, sb = set(a.split()), set(b.split())
        sa.discard("")
        sb.discard("")
        if not sa and not sb:
            return 0.0
        return len(sa & sb) / len(sa | sb)

    name_jaccard = _digit_jaccard(name_digits_s1, name_digits_cand)
    addr_jaccard = _digit_jaccard(addr_digits_s1, addr_digits_cand)

    return [name_exact, name_lev, addr_exact, addr_lev, name_jaccard, addr_jaccard]


# ---------------------------------------------------------------------------
# Agreement flag features (indices 24–27)
# ---------------------------------------------------------------------------

def _agreement_features(s1: dict, cand: dict, source: str = "") -> list[float]:
    c1 = normalize_country(_safe_str(s1.get("country")))
    c2 = normalize_country(_safe_str(cand.get("country")))

    if not c1 or not c2:
        country_match = -1.0
        country_contradiction = 0.0
    elif c1 == c2:
        country_match = 1.0
        country_contradiction = 0.0
    else:
        country_match = 0.0
        country_contradiction = 1.0

    src = source.upper() if source else _safe_str(cand.get("source", "")).upper()
    source_indicator = 1.0 if "S1-S2" in src else 0.0

    # Combined signal: name similarity boosted by country agreement
    n1 = normalize_name(_safe_str(s1.get("name")))
    n2 = normalize_name(_safe_str(cand.get("name")))
    name_sim = _jaro_winkler(n1, n2)
    name_country_combined = name_sim * (1.0 + 0.1 * country_match) if country_match >= 0 else name_sim

    return [country_match, country_contradiction, source_indicator, name_country_combined]


# ---------------------------------------------------------------------------
# Length covariate features (indices 28–31)
# ---------------------------------------------------------------------------

def _length_features(s1: dict, cand: dict) -> list[float]:
    n1 = normalize_name(_safe_str(s1.get("name")))
    n2 = normalize_name(_safe_str(cand.get("name")))

    l1 = float(len(n1))
    l2 = float(len(n2))
    diff = abs(l1 - l2)
    ratio = min(l1, l2) / max(l1, l2) if max(l1, l2) > 0 else 0.0

    return [l1, l2, diff, ratio]


# ---------------------------------------------------------------------------
# Blocking / meta features (indices 32–35)
# ---------------------------------------------------------------------------

def _meta_features(s1: dict, cand: dict,
                   blocking_score: float = 0.0,
                   rank: Any = None,
                   source: str = "") -> list[float]:
    bs = _safe(blocking_score, 0.0)
    rank_bin = float(_distance_rank_bin(rank))
    src = source.upper() if source else ""
    is_s1s2 = 1.0 if "S1-S2" in src else 0.0
    is_s1s3 = 1.0 if "S1-S3" in src else 0.0
    return [bs, rank_bin, is_s1s2, is_s1s3]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_pair_features(
    s1_record: dict,
    candidate_record: dict,
    blocking_score: float = 0.0,
    rank: Any = None,
    source: str = "",
) -> list[float]:
    """
    Return a 36-element feature vector for one candidate pair.

    Parameters
    ----------
    s1_record : dict
        S1 business record fields.
    candidate_record : dict
        Candidate (S2 or S3) business record fields.
    blocking_score : float
        Relevance score from the blocking module (default 0.0).
    rank : int or None
        Rank of this candidate among candidates for this S1 entity.
    source : str
        "S1-S2" or "S1-S3".

    Returns
    -------
    list[float]
        Exactly 36 numeric values.
    """
    if s1_record is None:
        s1_record = {}
    if candidate_record is None:
        candidate_record = {}

    try:
        name_feats = _name_features(s1_record, candidate_record)
    except Exception:
        name_feats = _DEFAULTS[0:10]

    try:
        addr_feats = _address_features(s1_record, candidate_record)
    except Exception:
        addr_feats = _DEFAULTS[10:18]

    try:
        digit_feats = _digit_features(s1_record, candidate_record)
    except Exception:
        digit_feats = _DEFAULTS[18:24]

    try:
        agree_feats = _agreement_features(s1_record, candidate_record, source)
    except Exception:
        agree_feats = _DEFAULTS[24:28]

    try:
        len_feats = _length_features(s1_record, candidate_record)
    except Exception:
        len_feats = _DEFAULTS[28:32]

    try:
        meta_feats = _meta_features(s1_record, candidate_record, blocking_score, rank, source)
    except Exception:
        meta_feats = _DEFAULTS[32:36]

    vector = name_feats + addr_feats + digit_feats + agree_feats + len_feats + meta_feats
    assert len(vector) == 36, f"Feature vector length {len(vector)} != 36; names={FEATURE_NAMES}"
    return vector


def build_feature_matrix(
    candidate_pairs: pd.DataFrame,
    source_records: dict[str, pd.DataFrame],
) -> tuple[np.ndarray, list[str]]:
    """
    Build a feature matrix for all candidate pairs.

    Parameters
    ----------
    candidate_pairs : pd.DataFrame
        Columns: s1_id, s2_id (or s3_id), blocking_score, source, and optionally rank.
    source_records : dict[str, pd.DataFrame]
        Keys are source names (e.g. "S1", "S2", "S3"); values are DataFrames indexed by record ID.

    Returns
    -------
    (np.ndarray of shape [N, 36], list[str] of 36 feature names)
    """
    s1_df = source_records.get("S1", pd.DataFrame())
    s2_df = source_records.get("S2", pd.DataFrame())
    s3_df = source_records.get("S3", pd.DataFrame())

    rows = []
    for _, row in candidate_pairs.iterrows():
        s1_id = row.get("s1_id", "")
        s2_id = row.get("s2_id", None)
        s3_id = row.get("s3_id", None)
        source = row.get("source", "")
        blocking_score = float(row.get("blocking_score", 0.0))
        rank = row.get("rank", None)

        s1_rec = s1_df.loc[s1_id].to_dict() if (s1_id in s1_df.index) else {}
        if s2_id and not pd.isna(s2_id) if isinstance(s2_id, float) else s2_id:
            cand_rec = s2_df.loc[s2_id].to_dict() if (s2_id in s2_df.index) else {}
        elif s3_id and not pd.isna(s3_id) if isinstance(s3_id, float) else s3_id:
            cand_rec = s3_df.loc[s3_id].to_dict() if (s3_id in s3_df.index) else {}
        else:
            cand_rec = {}

        fv = build_pair_features(s1_rec, cand_rec, blocking_score, rank, source)
        rows.append(fv)

    matrix = np.array(rows, dtype=np.float64) if rows else np.empty((0, 36), dtype=np.float64)
    return matrix, FEATURE_NAMES
