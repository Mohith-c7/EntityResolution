"""
Data ingestion, schema validation, and loading subpackage for Amazon Business Entity Resolution 2026.
"""

from src.data.loader import (
    load_dataset,
    load_ground_truth,
    load_test_source1,
    load_test_source2,
    load_test_source3,
    load_train_source1,
    load_train_source2,
    load_train_source3,
    load_tsv,
)
from src.data.profiling import (
    compute_exact_median_from_histogram,
    find_ground_truth_examples,
    print_profiling_summary,
    profile_ground_truth,
    profile_source_dataset,
)
from src.data.schema import (
    DatasetSummary,
    inspect_dataframe,
    inspect_ground_truth,
    validate_ground_truth_contract,
    validate_schema,
    validate_source_dataset,
)

__all__ = [
    "load_tsv",
    "load_dataset",
    "load_train_source1",
    "load_train_source2",
    "load_train_source3",
    "load_ground_truth",
    "load_test_source1",
    "load_test_source2",
    "load_test_source3",
    "DatasetSummary",
    "inspect_dataframe",
    "inspect_ground_truth",
    "validate_schema",
    "validate_source_dataset",
    "validate_ground_truth_contract",
    "profile_source_dataset",
    "profile_ground_truth",
    "compute_exact_median_from_histogram",
    "find_ground_truth_examples",
    "print_profiling_summary",
]
