"""Experimental one-hop expansion inside blocking, before final matcher scoring."""
from dataclasses import replace

import numpy as np
from rapidfuzz import fuzz

from .cheap_ranker import rank_features
from .contracts import Candidate
from ..features.evidence import phonetic


def retrieve_with_bridges(reference, indexes, ranker, min_probability=.99, max_seeds=2, informative_only=False):
    """Use blocker-selected seeds, then return at most top_k per source.

    Seed confidence comes from the blocking ranker, not the final matching model.
    No seed is automatically declared a match. Every final pair is scored later
    against the original S1 reference with the normal matching feature schema.
    """
    if not 0 <= min_probability <= 1 or max_seeds < 0: raise ValueError("Invalid bridge settings")
    base = {index.source: index.query(reference) for index in indexes}
    scored = []
    for index in indexes:
        pairs = base[index.source]
        if not pairs: continue
        matrix = np.asarray([rank_features(reference, t, index) for _, t in pairs], dtype=np.float32)
        probabilities = ranker.predict(matrix, num_threads=1)
        if not np.isfinite(probabilities).all(): raise ValueError("Nonfinite seed confidence")
        scored.extend((float(p), c.candidate_entity_id, t, index.source) for (c,t),p in zip(pairs,probabilities)
                      if p >= min_probability)
    seeds = []; seen_sources = set()
    for p,cid,target,source in sorted(scored, key=lambda item: (-item[0], item[1])):
        if source in seen_sources: continue
        if target.core == reference.core and target.address == reference.address: continue
        if informative_only:
            different_name = fuzz.token_sort_ratio(phonetic(reference.core), phonetic(target.core)) < 85
            fuller_address = len(target.address) > 1.4 * max(1, len(reference.address))
            if not (different_name or fuller_address): continue
        seeds.append(target); seen_sources.add(source)
        if len(seeds) >= max_seeds: break
    if max_seeds == 0: seeds = []
    unions = {s: {c.candidate_entity_id: (c,t) for c,t in pairs} for s,pairs in base.items()}
    for seed in seeds:
        query = replace(seed, entity_id=reference.entity_id)
        for index in indexes:
            for _,target in index.query(query):
                if target.entity_id not in unions[index.source]:
                    # The seed's exact-name/address provenance is not evidence about S1.
                    candidate = Candidate(reference.entity_id, target.entity_id, index.source,
                        index.rerank(reference,target,("bridge",)), ("bridge",))
                    unions[index.source][target.entity_id] = candidate,target
    result = {}
    for index in indexes:
        pairs = list(unions[index.source].values())
        if not pairs: result[index.source] = []; continue
        index.prefetch_frequencies([reference, *[t for _,t in pairs]])
        matrix = np.asarray([rank_features(reference,t,index) for _,t in pairs], dtype=np.float32)
        probabilities = ranker.predict(matrix,num_threads=1)
        if not np.isfinite(probabilities).all(): raise ValueError("Nonfinite final blocking ranks")
        ordered = sorted(zip(pairs,probabilities), key=lambda item: (-item[1], item[0][0].candidate_entity_id))[:index.config.top_k]
        result[index.source] = [(replace(c,rank_within_source=i+1),t) for i,((c,t),_) in enumerate(ordered)]
    return result
