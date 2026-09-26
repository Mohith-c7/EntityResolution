"""Focused tests for deterministic seven-path candidate blocking."""

import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_CODE = Path(__file__).resolve().parents[1] / "code" / "business_entity_resolution"
if str(PROJECT_CODE) not in sys.path:
    sys.path.insert(0, str(PROJECT_CODE))

from src.blocking.candidate_generator import generate_candidates  # noqa: E402
from src.blocking.name_index import RareNameIndex  # noqa: E402
from src.blocking.reranker import rerank_candidates  # noqa: E402


def _frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


class BlockingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source1 = _frame([
            {
                "entity_id": "S1-1", "norm_name": "acme services",
                "name_core": "acme services", "norm_address": "10 oak road springfield",
                "digit_tokens": ["10", "12345"], "norm_country": "exampleland",
            },
        ])
        self.source2 = _frame([
            {
                "entity_id": "S2-1", "norm_name": "acme services",
                "name_core": "acme services", "norm_address": "10 oak road springfield",
                "digit_tokens": ["10", "12345"], "norm_country": "exampleland",
            },
            {
                "entity_id": "S2-2", "norm_name": "other group",
                "name_core": "other group", "norm_address": "77 unique plaza",
                "digit_tokens": ["77777"], "norm_country": "exampleland",
            },
        ])
        self.source3 = _frame([
            {
                "entity_id": "S3-1", "norm_name": "acme service partners",
                "name_core": "acme service partners", "norm_address": "99 remote lane",
                "digit_tokens": ["99999"], "norm_country": "exampleland",
            },
            {
                "entity_id": "S3-2", "norm_name": "numeric holdings",
                "name_core": "numeric holdings", "norm_address": "12345 quiet lane",
                "digit_tokens": ["12345"], "norm_country": "exampleland",
            },
        ])

    def test_rare_name_index_excludes_common_tokens(self) -> None:
        records = _frame([
            {"entity_id": "A", "norm_name": "common alpha"},
            {"entity_id": "B", "norm_name": "common beta"},
            {"entity_id": "C", "norm_name": "gamma"},
        ])
        index = RareNameIndex(max_token_df=0.5).build(records)
        self.assertNotIn("common", index.postings)
        self.assertEqual(index.query("common alpha"), {"A": 1.0})

    def test_exact_name_and_core_paths_receive_scores_and_bonuses(self) -> None:
        result = generate_candidates(self.source1, self.source2, self.source3, max_token_df=1.0)
        match = result.loc[result["candidate_entity_id"] == "S2-1"].iloc[0]
        self.assertEqual(match["score_exact_norm_name"], 1.0)
        self.assertEqual(match["score_exact_name_core"], 1.0)
        self.assertGreaterEqual(match["blocking_score"], 5.5)
        self.assertEqual(match["candidate_source"], "S2")

    def test_prefix_and_digit_paths_retrieve_candidates(self) -> None:
        result = generate_candidates(self.source1, self.source2, self.source3, max_token_df=1.0)
        prefix_match = result.loc[result["candidate_entity_id"] == "S3-1"].iloc[0]
        digit_match = result.loc[result["candidate_entity_id"] == "S3-2"].iloc[0]
        self.assertEqual(prefix_match["score_name_core_prefix"], 1.0)
        self.assertEqual(digit_match["score_rare_digit_tokens"], 1.0)
        self.assertEqual(digit_match["candidate_source"], "S3")

    def test_candidate_union_keeps_candidates_from_distinct_paths(self) -> None:
        result = generate_candidates(self.source1, self.source2, self.source3, max_token_df=1.0)
        self.assertEqual(set(result["candidate_entity_id"]), {"S2-1", "S3-1", "S3-2"})
        self.assertIn("blocking_score", result.columns)

    def test_reranking_applies_top_k_and_stable_tie_breaking(self) -> None:
        ranked = rerank_candidates(
            {"S3-2": {"rare_name_tokens": 1.0}, "S2-9": {"rare_name_tokens": 1.0}},
            top_k=1,
        )
        self.assertEqual(ranked[0][0], "S2-9")

    def test_candidate_output_is_deterministic(self) -> None:
        first = generate_candidates(self.source1, self.source2, self.source3, max_token_df=1.0)
        second = generate_candidates(self.source1, self.source2, self.source3, max_token_df=1.0)
        pd.testing.assert_frame_equal(first, second)


if __name__ == "__main__":
    unittest.main()
