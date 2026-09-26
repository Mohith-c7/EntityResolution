"""Generic ingestion must retain leading zeros and literal country labels."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"code/business_entity_resolution"))
from src.data.loader import load_dataset,load_tsv


@pytest.mark.parametrize("loader",[load_dataset,load_tsv])
def test_generic_loader_preserves_raw_strings(tmp_path,loader):
    path=tmp_path/"source.tsv"
    path.write_text("entity_id\tbusiness_name\tbusiness_address\tcountry\nS1-A\t007\t00123\tNA\n")
    row=loader(path).iloc[0]
    assert row["business_name"]=="007"
    assert row["business_address"]=="00123"
    assert row["country"]=="NA"
    assert load_tsv(path,dtype=None).iloc[0]["business_address"]==123
