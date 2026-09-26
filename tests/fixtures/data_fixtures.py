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
