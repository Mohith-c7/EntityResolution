"""Versioned pair evidence with explicit missingness and stable legacy columns."""

import math
from functools import lru_cache

import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein

from ..blocking.contracts import BlockingRecord, Candidate
from ..preprocessing.normalize import accent_fold, extract_digits
from .registry import (FEATURE_NAMES, FEATURE_NAMES_V2, FEATURE_NAMES_V3_EXTRA,
                       FEATURE_VERSION, FEATURE_VERSION_V2, FEATURE_VERSION_V3)
from .evidence import phonetic, grams, numeric_relation, TLD


def jaccard(left, right) -> float:
    return len(left & right) / len(left | right) if left and right else 0.0


def similarity_features(left: str, right: str, left_tokens, right_tokens) -> list[float]:
    if not left or not right:
        return [0.0] * 7
    return [
        JaroWinkler.similarity(left, right), Levenshtein.normalized_similarity(left, right),
        fuzz.token_sort_ratio(left, right) / 100, fuzz.token_set_ratio(left, right) / 100,
        fuzz.partial_ratio(left, right) / 100, jaccard(left_tokens, right_tokens),
        min(len(left), len(right)) / max(len(left), len(right)),
    ]


@lru_cache(maxsize=10_000)
def name_grams(name: str) -> frozenset[str]:
    return frozenset(name[i:i+n] for n in (3, 4) for i in range(max(0, len(name) - n + 1)))


def build_pair_features(left: BlockingRecord, right: BlockingRecord, candidate: Candidate) -> dict[str, float]:
    a, b = name_grams(left.retrieval_name), name_grams(right.retrieval_name)
    char_cosine = len(a & b) / math.sqrt(len(a) * len(b)) if a and b else 0.0
    nl = {"n:" + t for t in left.name_digits} | {"a:" + t for t in left.address_digits}
    nr = {"n:" + t for t in right.name_digits} | {"a:" + t for t in right.address_digits}
    first_left, first_right = extract_digits(left.address), extract_digits(right.address)
    values = [
        *similarity_features(left.name, right.name, left.name_tokens, right.name_tokens), char_cosine,
        *similarity_features(left.address, right.address, left.address_tokens, right.address_tokens),
        jaccard(left.address_digits, right.address_digits),
        jaccard(nl, nr), bool(nl and nl == nr), bool(left.postcode_candidates & right.postcode_candidates),
        bool(first_left and first_right and first_left[0] == first_right[0]),
        bool(left.name and left.name == right.name), bool(left.core and left.core == right.core),
        bool(left.name_tokens & right.name_tokens), bool(left.name_tokens & right.address_tokens),
        bool(left.address_tokens & right.name_tokens), bool(left.country and left.country == right.country),
        not bool(right.address), not bool(left.postcode_candidates and right.postcode_candidates),
        not bool(left.country and right.country),
        len(left.name), len(right.name), len(left.address), len(right.address),
        candidate.blocking_score, candidate.candidate_source == "S2", candidate.candidate_source == "S3",
    ]
    if len(values) != len(FEATURE_NAMES):
        raise RuntimeError("Feature registry mismatch")
    return dict(zip(FEATURE_NAMES, map(float, values)))


def feature_vector(left: BlockingRecord, right: BlockingRecord, candidate: Candidate) -> np.ndarray:
    return np.fromiter(build_pair_features(left, right, candidate).values(), dtype=np.float32, count=36)


def build_extended_pair_features(left, right, candidate, index):
    base = build_pair_features(left, right, candidate)
    lc, _, _ = index.resolved_name(left)
    rc, confidence, support = index.resolved_name(right)
    if index.aliases:
        evidence, evidence_support = index.aliases.evidence(right.core, left.core)
        confidence, support = evidence, evidence_support
    a, b = name_grams(lc), name_grams(rc)
    ld = {t.lstrip("0") or "0" for t in left.address_digits}
    rd = {t.lstrip("0") or "0" for t in right.address_digits}
    lf, rf = extract_digits(left.address), extract_digits(right.address)
    extras = [
        fuzz.token_sort_ratio(lc, rc) / 100 if lc and rc else 0,
        fuzz.token_set_ratio(lc, rc) / 100 if lc and rc else 0,
        bool(lc and lc == rc), len(a & b) / math.sqrt(len(a)*len(b)) if a and b else 0,
        min(len(lc),len(rc))/max(len(lc),len(rc)) if lc and rc else 0,
        math.log1p(index.family_count(lc)), confidence, math.log1p(support),
        index.address_containment(left.address,right.address),
        bool(lf and rf and (lf[0].lstrip("0") or "0") == (rf[0].lstrip("0") or "0")),
        jaccard(ld,rd), index.weighted_overlap(left.name_tokens,right.name_tokens,"name"),
        index.weighted_overlap(left.address_tokens,right.address_tokens,"address"),
        max((index.idf("address",t) for t in left.address_tokens & right.address_tokens), default=0),
    ]
    base.update(zip(FEATURE_NAMES_V2[len(FEATURE_NAMES):],map(float,extras)))
    return base


def build_v3_evidence(left, right, index):
    """Fifteen additional features; preserve all previous 50 feature meanings."""
    # Only standalone decimal tokens in this view: gui1len/hurtad0 are not
    # street-number evidence. Keep B12/12A intact in the original numeric view.
    ld = [t.lstrip("0") or "0" for t in extract_digits(" ".join(t for t in left.address.split() if t.isdecimal()))]
    rd = [t.lstrip("0") or "0" for t in extract_digits(" ".join(t for t in right.address.split() if t.isdecimal()))]
    pl, pr = phonetic(left.core or left.name), phonetic(right.core or right.name)
    gl, gr = grams(pl), grams(pr)
    cl = "".join(pl.split())
    cr = "".join(t for t in pr.split() if t not in TLD)
    rmin = min((index.df("name", accent_fold(t)) for t in right.name_tokens), default=0)
    lmin = min((index.df("name", accent_fold(t)) for t in left.name_tokens), default=0)
    at = {accent_fold(t) for t in left.address_tokens if not t.isdecimal()}
    bt = {accent_fold(t) for t in right.address_tokens if not t.isdecimal()}
    name_tokens = {accent_fold(t) for t in left.name_tokens | right.name_tokens}
    if at and bt:
        weights = {t: index.idf("address", t) for t in sorted(at | bt)}
        lu = sum(weights[t] for t in sorted(at - bt)) / sum(weights[t] for t in sorted(at))
        ru = sum(weights[t] for t in sorted(bt - at)) / sum(weights[t] for t in sorted(bt))
        mismatch = float(any(len(t) >= 3 and weights[t] >= 6 for t in at - bt - name_tokens)
                         and any(len(t) >= 3 and weights[t] >= 6 for t in bt - at - name_tokens))
    else:
        lu = ru = mismatch = -1.0
    values = [*numeric_relation(ld, rd),
        fuzz.token_set_ratio(pl, pr) / 100 if pl and pr else 0,
        fuzz.token_sort_ratio(pl, pr) / 100 if pl and pr else 0,
        jaccard(gl, gr), fuzz.ratio(cl, cr) / 100 if cl and cr else 0,
        fuzz.partial_ratio(cl, cr) / 100 if cl and cr and min(len(cl), len(cr)) >= 4 else 0,
        math.log1p(rmin), math.log1p(lmin), lu, ru, mismatch]
    return dict(zip(FEATURE_NAMES_V3_EXTRA, map(float, values)))


def build_versioned_pair_features(left, right, candidate, index, version):
    if version == FEATURE_VERSION:
        return build_pair_features(left, right, candidate)
    if version not in (FEATURE_VERSION_V2, FEATURE_VERSION_V3):
        raise ValueError(f"Unsupported feature version: {version}")
    features = build_extended_pair_features(left, right, candidate, index)
    if version == FEATURE_VERSION_V3:
        features.update(build_v3_evidence(left, right, index))
    return features
