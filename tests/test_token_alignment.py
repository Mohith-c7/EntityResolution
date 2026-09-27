"""Tests for rare-token soft alignment evidence."""

import sys
import unittest
from pathlib import Path

PROJECT_CODE = Path(__file__).resolve().parents[1] / "code" / "business_entity_resolution"
if str(PROJECT_CODE) not in sys.path:
    sys.path.insert(0, str(PROJECT_CODE))

from src.blocking.disk_index import normalize_record  # noqa: E402
from src.features.token_alignment import (  # noqa: E402
    FEATURE_NAMES,
    AlignmentStatistics,
    build_alignment_features,
)


def _record(entity_id, name, address, country="US"):
    return normalize_record(
        {"entity_id": entity_id, "business_name": name,
         "business_address": address, "country": country}
    )


def _stats() -> AlignmentStatistics:
    # A small corpus where "springfield" and "services" are common and the
    # distinctive street/name words are rare.
    global_df = {
        "acme": 1, "services": 100, "springfield": 100,
        "oak": 2, "road": 50, "10": 5,
        "birch": 1, "avenue": 40,
    }
    return AlignmentStatistics(
        version="token-alignment-stats-v1",
        record_count=200,
        global_df=global_df,
        country_df={"US": (200, dict(global_df))},
        source_sha256={},
    )


class SchemaTests(unittest.TestCase):
    def test_fixed_schema_and_finite(self) -> None:
        stats = _stats()
        left = _record("S1-1", "Acme Services", "10 Oak Road, Springfield")
        right = _record("S2-1", "Acme Services", "10 Oak Road, Springfield")
        features = build_alignment_features(left, right, stats)
        self.assertEqual(list(features), FEATURE_NAMES)
        for value in features.values():
            self.assertTrue(0.0 <= value, f"non-finite/negative: {features}")

    def test_empty_address_is_missing_not_contradictory(self) -> None:
        stats = _stats()
        left = _record("S1-1", "Acme Services", "10 Oak Road, Springfield")
        right = _record("S2-1", "Acme Services", "")
        features = build_alignment_features(left, right, stats)
        self.assertEqual(features["align_addr_missing"], 1.0)
        self.assertEqual(features["align_addr_coverage_sym"], 0.0)
        # Missing address must not read as a distinctive mismatch.
        self.assertEqual(features["align_addr_unmatched_distinctive"], 0.0)


class AlignmentBehaviorTests(unittest.TestCase):
    def test_same_generic_name_different_street_weak_street_support(self) -> None:
        stats = _stats()
        left = _record("S1-1", "Acme Services", "10 Oak Road, Springfield")
        right = _record("S2-1", "Acme Services", "99 Birch Avenue, Springfield")
        features = build_alignment_features(left, right, stats)
        # The name-excluded address view should show weak support: the shared
        # city (springfield) is common and the distinctive streets differ.
        exact = build_alignment_features(left, _record("S2-2", "Acme Services", "10 Oak Road, Springfield"), stats)
        self.assertLess(features["align_addrx_coverage_sym"], exact["align_addrx_coverage_sym"])
        self.assertGreater(features["align_addr_unmatched_distinctive"], 0.0)

    def test_one_char_typo_still_supports(self) -> None:
        stats = _stats()
        left = _record("S1-1", "Acme Services", "10 Oak Road, Springfield")
        right = _record("S2-1", "Acme Services", "10 OaK Road, Springfield")  # case fold
        typo = _record("S2-2", "Acme Services", "10 Oak Raod, Springfield")   # transposed letters
        f_exact = build_alignment_features(left, right, stats)
        f_typo = build_alignment_features(left, typo, stats)
        # A one-character typo retains most of the address support.
        self.assertGreater(f_typo["align_addr_coverage_sym"], 0.5 * f_exact["align_addr_coverage_sym"])

    def test_transposition_and_accents_deterministic(self) -> None:
        stats = _stats()
        left = _record("S1-1", "Cafe Acme", "10 Oak Road")
        right = _record("S2-1", "Café Acme", "10 Oak Road")  # accented
        a = build_alignment_features(left, right, stats)
        b = build_alignment_features(left, right, stats)
        self.assertEqual(a, b)
        # Accent-folded name should align strongly.
        self.assertGreater(a["align_name_coverage_sym"], 0.5)

    def test_generic_word_cannot_cover_several(self) -> None:
        stats = _stats()
        # One shared generic token on the left, several on the right.
        left = _record("S1-1", "Services", "Springfield")
        right = _record("S2-1", "Services Services Services", "Springfield Springfield")
        features = build_alignment_features(left, right, stats)
        # One-to-one alignment caps coverage: the single left token can only
        # cover one right token, so right-to-left coverage stays low.
        self.assertLess(features["align_name_coverage_r2l"], 0.6)


if __name__ == "__main__":
    unittest.main()
