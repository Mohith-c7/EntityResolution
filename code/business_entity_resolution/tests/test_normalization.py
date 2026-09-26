"""Regression tests for Unicode-safe normalization and dataframe preprocessing."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_CODE = Path(__file__).resolve().parents[1]
if str(PROJECT_CODE) not in sys.path:
    sys.path.insert(0, str(PROJECT_CODE))

from src.preprocessing.normalize import (  # noqa: E402
    accent_fold,
    extract_digits,
    normalize_address,
    normalize_country,
    normalize_name,
)
from src.preprocessing.preprocess import preprocess_dataframe  # noqa: E402


class NormalizationTests(unittest.TestCase):
    def test_primary_normalization_preserves_indic_combining_marks(self) -> None:
        self.assertEqual(normalize_name("दिल्ली"), "दिल्ली")
        self.assertEqual(normalize_name("ଓଡ଼ିଶା"), "ଓଡ଼ିଶା")

    def test_composed_and_decomposed_accents_share_a_folded_view(self) -> None:
        self.assertEqual(normalize_name("Café"), "café")
        self.assertEqual(normalize_name("Cafe\u0301"), "café")
        self.assertEqual(accent_fold("Café"), "cafe")
        self.assertEqual(accent_fold("Cafe\u0301"), "cafe")
        self.assertEqual(accent_fold("दिल्ली"), "दिल्ली")

    def test_primary_address_does_not_expand_ambiguous_abbreviations(self) -> None:
        self.assertEqual(normalize_address("10 Main St, Miami, FL"), "10 main st miami fl")
        self.assertEqual(normalize_address("12 St John Rd"), "12 st john rd")

    def test_missing_scalar_variants_normalize_to_empty(self) -> None:
        for value in (np.float64("nan"), np.float32("nan"), float("nan"), None, pd.NA):
            self.assertEqual(normalize_name(value), "")
            self.assertEqual(normalize_address(value), "")
            self.assertEqual(normalize_country(value), "")
            self.assertEqual(extract_digits(value), [])

    def test_preprocess_preserves_raw_data_and_has_complete_schema(self) -> None:
        raw = pd.DataFrame({
            "entity_id": ["S1-1", "S1-2"],
            "business_name": ["Café LLC", pd.NA],
            "business_address": ["00123 Main St, 00999", pd.NA],
            "country": ["Newlandia", pd.NA],
        })
        original = raw.copy(deep=True)
        result = preprocess_dataframe(raw)

        pd.testing.assert_frame_equal(raw, original)
        self.assertEqual(result.loc[0, "business_name"], "Café LLC")
        self.assertEqual(result.loc[0, "norm_address"], "00123 main st 00999")
        self.assertEqual(result.loc[0, "accent_folded_name"], "cafe llc")
        self.assertEqual(result.loc[0, "name_numeric_tokens"], [])
        self.assertEqual(result.loc[0, "address_numeric_tokens"], ["00123", "00999"])
        self.assertEqual(result.loc[0, "postcode_candidates"], ["00123", "00999"])
        self.assertEqual(result.loc[1, "name_tokens"], [])
        self.assertEqual(result.loc[1, "address_tokens"], [])
        self.assertTrue(result.loc[1, "name_missing"])
        self.assertTrue(result.loc[1, "address_missing"])
        self.assertTrue(result.loc[1, "country_missing"])

        expected_columns = [
            "entity_id", "business_name", "business_address", "country",
            "norm_name", "name_core", "norm_address", "digit_tokens", "norm_country",
            "accent_folded_name", "accent_folded_address", "name_tokens", "address_tokens",
            "name_missing", "address_missing", "country_missing", "name_numeric_tokens",
            "address_numeric_tokens", "postcode_candidates",
        ]
        self.assertEqual(list(result.columns), expected_columns)

    def test_preprocess_rejects_missing_required_columns(self) -> None:
        with self.assertRaisesRegex(KeyError, "business_address"):
            preprocess_dataframe(pd.DataFrame({"business_name": ["Acme"], "country": ["US"]}))

    def test_empty_values_have_stable_empty_representations(self) -> None:
        result = preprocess_dataframe(pd.DataFrame({
            "business_name": [""], "business_address": [""], "country": [""],
        }))
        self.assertEqual(result.loc[0, "norm_name"], "")
        self.assertEqual(result.loc[0, "accent_folded_address"], "")
        self.assertEqual(result.loc[0, "digit_tokens"], [])
        self.assertEqual(result.loc[0, "postcode_candidates"], [])
        self.assertFalse(result.loc[0, "name_missing"])


if __name__ == "__main__":
    unittest.main()
