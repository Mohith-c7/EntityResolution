"""Cheap supervised pair evidence for experimental pre-matcher ranking."""
from functools import lru_cache

from rapidfuzz import fuzz

from ..features.evidence import phonetic, TLD
from ..preprocessing.normalize import extract_digits

RANK_FEATURES = (
    "name_token_sort", "name_token_set", "address_token_sort", "address_token_set",
    "name_length_ratio", "address_length_ratio", "candidate_address_missing",
    "exact_name_core", "country_exact_match", "canonical_name_set", "canonical_core_exact",
    "candidate_alias_confidence", "numeric_canonical_jaccard", "first_numeric_canonical_exact", "concat_ratio",
)


@lru_cache(maxsize=20000)
def first_number(address):
    numbers = extract_digits(address)
    return (numbers[0].lstrip("0") or "0") if numbers else None


def rank_features(left, right, index):
    lc, _, _ = index.resolved_name(left)
    rc, confidence, _ = index.resolved_name(right)
    if index.aliases:
        confidence, _ = index.aliases.evidence(right.core, left.core)
    ld = {t.lstrip("0") or "0" for t in left.address_digits}
    rd = {t.lstrip("0") or "0" for t in right.address_digits}
    lf, rf = first_number(left.address), first_number(right.address)
    cl = "".join(phonetic(left.core or left.name).split())
    cr = "".join(t for t in phonetic(right.core or right.name).split() if t not in TLD)
    def ratio(a, b): return min(len(a), len(b)) / max(len(a), len(b)) if a and b else 0.
    return [
        fuzz.token_sort_ratio(left.name, right.name) / 100 if left.name and right.name else 0.,
        fuzz.token_set_ratio(left.name, right.name) / 100 if left.name and right.name else 0.,
        fuzz.token_sort_ratio(left.address, right.address) / 100 if left.address and right.address else 0.,
        fuzz.token_set_ratio(left.address, right.address) / 100 if left.address and right.address else 0.,
        ratio(left.name, right.name), ratio(left.address, right.address), float(not right.address),
        float(bool(left.core) and left.core == right.core),
        float(bool(left.country and right.country) and left.country == right.country),
        fuzz.token_set_ratio(lc, rc) / 100 if lc and rc else 0., float(bool(lc) and lc == rc), confidence,
        len(ld & rd) / len(ld | rd) if ld and rd else 0., float(lf is not None and rf is not None and lf == rf),
        fuzz.ratio(cl, cr) / 100 if cl and cr else 0.,
    ]
