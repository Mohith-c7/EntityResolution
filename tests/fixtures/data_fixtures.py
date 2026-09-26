"""
Lightweight synthetic test fixtures for Data Contract and Schema validation.
Avoids loading full multi-gigabyte datasets during unit tests.
"""

from __future__ import annotations

import pandas as pd


def get_valid_source1_df() -> pd.DataFrame:
    """Return a valid minimal Source 1 DataFrame."""
    return pd.DataFrame([
        {
            "entity_id": "S1-101",
            "business_name": "Apex Technology Services",
            "business_address": "100 Innovation Way, Austin, TX",
            "country": "US",
        },
        {
            "entity_id": "S1-102",
            "business_name": "Apex Technology Services",  # Duplicate name across locations is allowed
            "business_address": "200 Market Street, Dallas, TX",
            "country": "US",
        },
        {
            "entity_id": "S1-103",
            "business_name": "Sharma Textiles Pvt Ltd",
            "business_address": "Plot 42, GIDC Estate, Surat, Gujarat",
            "country": "India",
        },
        {
            "entity_id": "S1-104",
            "business_name": "Atelier du Pain SAS",
            "business_address": "15 Rue de Rivoli, Paris",
            "country": "France",  # Open-set country test
        },
        {
            "entity_id": "S1-105",
            "business_name": "Solitary Enterprise Inc",
            "business_address": "999 Remote Ridge, Helena, MT",
            "country": "US",
        },
    ])


def get_valid_source2_df() -> pd.DataFrame:
    """Return a valid minimal Source 2 DataFrame with nullable address support."""
    return pd.DataFrame([
        {
            "entity_id": "S2-201",
            "business_name": "Apex Tech Services",
            "business_address": "100 INNOVATION WAY, AUSTIN, TX",
            "country": "US",
        },
        {
            "entity_id": "S2-202",
            "business_name": "Sharma Textiles Private Limited",
            "business_address": "Plot 42 GIDC Estate Surat",
            "country": "India",
        },
        {
            "entity_id": "S2-203",
            "business_name": "Atelier Pain",
            "business_address": None,  # Permitted null address in S2
            "country": "France",
        },
    ])


def get_valid_source3_df() -> pd.DataFrame:
    """Return a valid minimal Source 3 DataFrame."""
    return pd.DataFrame([
        {
            "entity_id": "S3-301",
            "business_name": "Apex Technology",
            "business_address": "100 Innovation Way, Austin, Texas",
            "country": "US",
        },
        {
            "entity_id": "S3-302",
            "business_name": "Sharma Textiles",
            "business_address": "GIDC Estate, Surat, Gujarat",
            "country": "India",
        },
    ])


def get_valid_ground_truth_df() -> pd.DataFrame:
    """
    Return a valid minimal Ground Truth DataFrame with singletons, single matches, and multi matches.
    """
    return pd.DataFrame([
        {
            "source1_entity_id": "S1-101",
            "matched_entity_ids": "S2-201,S3-301",  # Multi-match (S2 and S3)
        },
        {
            "source1_entity_id": "S1-102",
            "matched_entity_ids": "S2-201",         # Single match
        },
        {
            "source1_entity_id": "S1-103",
            "matched_entity_ids": "S2-202,S3-302",  # Multi-match
        },
        {
            "source1_entity_id": "S1-104",
            "matched_entity_ids": "S2-203",         # Match with null address candidate
        },
        {
            "source1_entity_id": "S1-105",
            "matched_entity_ids": "",               # Singleton (0 matches, valid)
        },
    ])


def get_missing_value_regression_df() -> pd.DataFrame:
    """Return fixture containing None, empty string, whitespace string, and valid text."""
    return pd.DataFrame({
        "entity_id": ["S1-1", "S1-2", "S1-3", "S1-4"],
        "test_col": [None, "", "   ", "valid text"],
        "business_name": ["Name1", "Name2", "Name3", "Name4"],
        "business_address": ["Addr1", "Addr2", "Addr3", "Addr4"],
        "country": ["US", "US", "US", "US"],
    })


def get_source_with_null_address_df() -> pd.DataFrame:
    """Return source records with permitted null address."""
    return pd.DataFrame([
        {"entity_id": "S2-10", "business_name": "Corp A", "business_address": None, "country": "US"},
        {"entity_id": "S2-11", "business_name": "Corp B", "business_address": "123 Main St", "country": "India"},
    ])


def get_source_with_whitespace_id_df() -> pd.DataFrame:
    """Return source records containing whitespace-only entity ID."""
    return pd.DataFrame([
        {"entity_id": "   ", "business_name": "Corp A", "business_address": "123 Main St", "country": "US"},
        {"entity_id": "S1-20", "business_name": "Corp B", "business_address": "456 Oak St", "country": "US"},
    ])


def get_source_with_duplicate_id_df() -> pd.DataFrame:
    """Return source records containing duplicate entity ID."""
    return pd.DataFrame([
        {"entity_id": "S1-30", "business_name": "Corp A", "business_address": "123 Main St", "country": "US"},
        {"entity_id": "S1-30", "business_name": "Corp B", "business_address": "456 Oak St", "country": "US"},
    ])


def get_source_with_wrong_prefix_df() -> pd.DataFrame:
    """Return source records with wrong prefix."""
    return pd.DataFrame([
        {"entity_id": "S2-40", "business_name": "Corp A", "business_address": "123 Main St", "country": "US"},
    ])


def get_source_with_missing_fields_df() -> pd.DataFrame:
    """Return source records with null or whitespace-only name and country."""
    return pd.DataFrame([
        {"entity_id": "S1-50", "business_name": "  ", "business_address": "123 Main St", "country": "US"},
        {"entity_id": "S1-51", "business_name": "Corp B", "business_address": "456 Oak St", "country": None},
    ])


def get_ground_truth_with_empty_matches_df() -> pd.DataFrame:
    """Return GT records that are pure singletons (empty matches)."""
    return pd.DataFrame([
        {"source1_entity_id": "S1-101", "matched_entity_ids": ""},
        {"source1_entity_id": "S1-102", "matched_entity_ids": "   "},
    ])


def get_ground_truth_with_singleton_matches_df() -> pd.DataFrame:
    """Return GT records each matching exactly one candidate."""
    return pd.DataFrame([
        {"source1_entity_id": "S1-101", "matched_entity_ids": "S2-201"},
        {"source1_entity_id": "S1-102", "matched_entity_ids": "S3-301"},
    ])


def get_ground_truth_with_multiple_matches_df() -> pd.DataFrame:
    """Return GT records with multiple matched candidates."""
    return pd.DataFrame([
        {"source1_entity_id": "S1-101", "matched_entity_ids": "S2-201,S3-301,S2-202"},
    ])


def get_ground_truth_with_duplicate_candidate_df() -> pd.DataFrame:
    """Return GT records with duplicate candidates in the same row."""
    return pd.DataFrame([
        {"source1_entity_id": "S1-101", "matched_entity_ids": "S2-201,S2-201"},
    ])


def get_ground_truth_with_s1_in_candidates_df() -> pd.DataFrame:
    """Return GT records where S1 ID appears in candidate list."""
    return pd.DataFrame([
        {"source1_entity_id": "S1-101", "matched_entity_ids": "S1-102,S2-201"},
    ])


def get_ground_truth_with_invalid_prefix_df() -> pd.DataFrame:
    """Return GT records with invalid prefix in candidate list."""
    return pd.DataFrame([
        {"source1_entity_id": "S1-101", "matched_entity_ids": "S4-999,S2-201"},
    ])


def get_ground_truth_with_nonexistent_s2_df() -> pd.DataFrame:
    """Return GT records with S2 candidate not present in Source 2."""
    return pd.DataFrame([
        {"source1_entity_id": "S1-101", "matched_entity_ids": "S2-99999"},
    ])


def get_ground_truth_with_nonexistent_s3_df() -> pd.DataFrame:
    """Return GT records with S3 candidate not present in Source 3."""
    return pd.DataFrame([
        {"source1_entity_id": "S1-101", "matched_entity_ids": "S3-99999"},
    ])


def get_ground_truth_with_mismatched_s1_df() -> pd.DataFrame:
    """Return GT records where S1 set does not match source 1."""
    return pd.DataFrame([
        {"source1_entity_id": "S1-999", "matched_entity_ids": "S2-201"},
    ])
