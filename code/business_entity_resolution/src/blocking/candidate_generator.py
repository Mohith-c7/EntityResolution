"""Seven-path deterministic candidate generation for entity resolution."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from .address_index import RareAddressIndex
from .digit_index import RareDigitIndex
from .exact_blocks import (
    build_exact_index,
    build_postcode_name_index,
    build_prefix_index,
    core_prefix,
    lookup_exact,
    lookup_postcode_name,
)
from .name_index import RareNameIndex
from .reranker import rerank_candidates


PATH_COLUMNS = (
    "rare_name_tokens",
    "rare_address_tokens",
    "rare_digit_tokens",
    "exact_norm_name",
    "exact_name_core",
    "name_core_prefix",
    "postcode_name_token",
)


def _add_scores(target: dict[str, dict[str, float]], path: str, scores: dict[str, float]) -> None:
    for candidate_id, score in scores.items():
        target[candidate_id][path] += score


def generate_candidates(
    source1: Any,
    source2: Any,
    source3: Any,
    *,
    max_token_df: float = 0.10,
    top_k: int = 20,
    id_column: str = "entity_id",
) -> Any:
    """Generate top-k S2/S3 candidates per S1 record from seven blocking paths.

    Inputs are expected to already contain the preprocessing columns.  Source
    fields remain untouched; the returned dataframe is a candidate-pair table
    for downstream feature computation, not a match decision.
    """
    import pandas as pd

    required_columns = {
        id_column,
        "norm_name",
        "name_core",
        "norm_address",
        "digit_tokens",
        "norm_country",
    }
    for label, frame in (("source1", source1), ("source2", source2), ("source3", source3)):
        missing = sorted(required_columns.difference(frame.columns))
        if missing:
            raise KeyError(f"{label} is missing required columns: {', '.join(missing)}")

    candidates = pd.concat([source2, source3], ignore_index=True)
    candidate_sources = {
        str(record_id): "S2" for record_id in source2[id_column]
    }
    candidate_sources.update({str(record_id): "S3" for record_id in source3[id_column]})

    name_index = RareNameIndex(max_token_df).build(candidates, id_column=id_column)
    address_index = RareAddressIndex(max_token_df).build(candidates, id_column=id_column)
    digit_index = RareDigitIndex(max_token_df).build(candidates, id_column=id_column)
    exact_name_index = build_exact_index(candidates, "norm_name", id_column=id_column)
    exact_core_index = build_exact_index(candidates, "name_core", id_column=id_column)
    prefix_index = build_prefix_index(candidates, id_column=id_column)
    postcode_index = build_postcode_name_index(candidates, id_column=id_column)

    rows: list[dict[str, Any]] = []
    for _, record in source1.sort_values(id_column, kind="stable").iterrows():
        path_scores: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        _add_scores(path_scores, "rare_name_tokens", name_index.query(record["norm_name"]))
        _add_scores(path_scores, "rare_address_tokens", address_index.query(record["norm_address"]))
        _add_scores(path_scores, "rare_digit_tokens", digit_index.query(record["digit_tokens"]))
        _add_scores(path_scores, "exact_norm_name", lookup_exact(exact_name_index, record["norm_name"]))
        _add_scores(path_scores, "exact_name_core", lookup_exact(exact_core_index, record["name_core"]))
        _add_scores(path_scores, "name_core_prefix", lookup_exact(prefix_index, core_prefix(record["name_core"])))
        _add_scores(
            path_scores,
            "postcode_name_token",
            lookup_postcode_name(postcode_index, record["digit_tokens"], record["norm_name"]),
        )

        for candidate_id, scores, blocking_score in rerank_candidates(path_scores, top_k=top_k):
            rows.append(
                {
                    "source1_entity_id": str(record[id_column]),
                    "candidate_entity_id": candidate_id,
                    "candidate_source": candidate_sources[candidate_id],
                    **{f"score_{path}": scores.get(path, 0.0) for path in PATH_COLUMNS},
                    "blocking_score": blocking_score,
                }
            )

    columns = [
        "source1_entity_id",
        "candidate_entity_id",
        "candidate_source",
        *(f"score_{path}" for path in PATH_COLUMNS),
        "blocking_score",
    ]
    return pd.DataFrame(rows, columns=columns)
