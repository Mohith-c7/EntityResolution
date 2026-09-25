"""Focused tests for deterministic preprocessing normalization."""

import sys
from pathlib import Path


PROJECT_CODE = Path(__file__).resolve().parents[1] / "code" / "business_entity_resolution"
if str(PROJECT_CODE) not in sys.path:
    sys.path.insert(0, str(PROJECT_CODE))

from src.preprocessing.normalize import (  # noqa: E402
    extract_digits,
    name_core,
    normalize_address,
    normalize_country,
    normalize_name,
)


def test_normalize_name_expands_business_abbreviations_and_noise() -> None:
    assert normalize_name("Callicoat & Dailey Inc") == "callicoat and dailey incorporated"
    assert normalize_name("Pvt. EFS Print Ventures Ltd.") == "private efs print ventures limited"
    assert normalize_name("-- Holloway Peak Inc Seafood") == "holloway peak incorporated seafood"


def test_normalize_name_handles_unicode_and_full_width_text() -> None:
    assert normalize_name("LLC Moncada Léarning Center") == "llc moncada léarning center"
    assert normalize_name("Ｆｏｏ　LLC") == "foo llc"


def test_normalize_address_expands_observed_abbreviations() -> None:
    assert normalize_address("105 ELM ST, MORGANTON, NC") == "105 elm street morganton nc"
    assert normalize_address("H.NO.16-11-23/37/A, 2Nd Floor, Flat No.207") == (
        "house number 16 11 23 37 a 2nd floor flat number 207"
    )
    assert normalize_address("5780 Fawn Ct, Fort Worth, Texas") == "5780 fawn court fort worth texas"


def test_null_and_empty_values_are_safe() -> None:
    assert normalize_name(None) == ""
    assert normalize_address("") == ""
    assert normalize_country(float("nan")) == ""
    assert extract_digits(None) == []
    assert name_core("") == ""


def test_country_normalization_is_open_set_formatting_only() -> None:
    assert normalize_country("  Côte d'Ivoire  ") == "côte d ivoire"
    assert normalize_country("Newlandia") == "newlandia"


def test_extract_digits_returns_ordered_address_tokens() -> None:
    assert extract_digits("KH NO. -570/13, NEW DELHI") == ["570", "13"]
    assert extract_digits("GREENSBORO, NC, 19 1/2 STARDUST TRAIL") == ["19", "1", "2"]


def test_name_core_removes_legal_form_tokens() -> None:
    assert name_core("custom wealth services llc") == "custom wealth services"
    assert name_core("llc moncada léarning center") == "moncada léarning center"
    assert name_core("international south consultants private limited") == "international south consultants"
