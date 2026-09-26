"""
Tests verifying that generic ingestion retains leading zeros and literal string labels.
"""
import sys
from pathlib import Path

import pytest

# Ensure business_entity_resolution is in sys.path
CODE_DIR = Path(__file__).resolve().parents[1] / "code" / "business_entity_resolution"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from src.data.loader import load_dataset, load_tsv


@pytest.mark.parametrize("loader", [load_dataset, load_tsv])
def test_generic_loader_preserves_raw_strings(tmp_path, loader):
    path = tmp_path / "source.tsv"
    path.write_text("entity_id\tbusiness_name\tbusiness_address\tcountry\nS1-A\t007\t00123\tNA\n", encoding="utf-8")
    
    # By default, strings with leading zeros and literal 'NA' are preserved
    row = loader(path).iloc[0]
    assert row["business_name"] == "007"
    assert row["business_address"] == "00123"
    assert row["country"] == "NA"

    # Explicit dtype=None triggers pandas type inference
    inferred_df = load_tsv(path, dtype=None)
    assert inferred_df.iloc[0]["business_address"] == 123
