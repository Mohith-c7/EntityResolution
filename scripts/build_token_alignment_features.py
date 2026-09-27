"""Build rare-token alignment features for a supplied set of candidate pairs.

Reads pair keys from a parquet file and the official source TSVs, fetches the
needed records in one scan per source, and writes one feature row per supplied
pair with the same keys. No labels are read or allowed.

Statistics are label-free document frequencies built from the supplied Source 1
texts, separately per observed country with a global fallback. For test
inference pass the test Source 1 TSV so the statistics reflect the test corpus,
not training identities.

Example:
    python scripts/build_token_alignment_features.py \
        --pairs pairs.parquet \
        --source1 dataset/test/test_source1.tsv \
        --source2 dataset/test/test_source2.tsv \
        --source3 dataset/test/test_source3.tsv \
        --statistics-out research/.../token_alignment_stats.json \
        --output research/.../token_alignment_features.parquet
"""

import argparse
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "business_entity_resolution"))

from src.blocking.disk_index import normalize_record  # noqa: E402
from src.preprocessing.normalize import accent_fold  # noqa: E402
from src.features.evidence import phonetic  # noqa: E402
from src.features.token_alignment import (  # noqa: E402
    FEATURE_NAMES,
    FEATURE_VERSION,
    STATISTICS_VERSION,
    AlignmentStatistics,
    build_alignment_features,
)

PAIR_KEYS = ["source1_entity_id", "candidate_entity_id"]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def build_statistics(source1_path: Path) -> tuple[AlignmentStatistics, float]:
    """Label-free per-country document frequencies over Source 1 name+address."""
    started = time.perf_counter()
    global_df: Counter = Counter()
    country_df: dict[str, Counter] = defaultdict(Counter)
    country_count: Counter = Counter()
    phonetic_global_df: Counter = Counter()
    phonetic_country_df: dict[str, Counter] = defaultdict(Counter)
    total = 0
    frame = pd.read_csv(source1_path, sep="\t", dtype=str)
    for name, address, country in zip(
        frame["business_name"], frame["business_address"], frame["country"]
    ):
        record = normalize_record(
            {"entity_id": "S1-x", "business_name": name or "",
             "business_address": address or "", "country": country or ""}
        )
        terms = set(accent_fold(record.name).split()) | set(accent_fold(record.address).split())
        phonetic_terms = set(phonetic(record.name).split())
        country_key = record.country or ""
        for term in terms:
            global_df[term] += 1
            country_df[country_key][term] += 1
        for term in phonetic_terms:
            phonetic_global_df[term] += 1
            phonetic_country_df[country_key][term] += 1
        country_count[country_key] += 1
        total += 1
    stats = AlignmentStatistics(
        version=STATISTICS_VERSION,
        record_count=total,
        global_df=dict(global_df),
        country_df={c: (country_count[c], dict(df)) for c, df in country_df.items()},
        source_sha256={str(source1_path): digest(source1_path)},
        phonetic_global_df=dict(phonetic_global_df),
        phonetic_country_df={c: (country_count[c], dict(phonetic_country_df[c])) for c in country_count},
    )
    return stats, time.perf_counter() - started


def verify_statistics_source(stats: AlignmentStatistics, source1_path: Path) -> None:
    """Reject statistics from a different Source 1 corpus before generating rows.

    Paths may differ across machines; the content hash must match exactly.
    """
    expected = set(stats.source_sha256.values())
    if len(expected) != 1 or digest(source1_path) not in expected:
        raise ValueError("Statistics Source 1 hash does not match the supplied Source 1 corpus; rebuild statistics")


def _load_records(paths: dict[str, Path], wanted_ids: set[str]) -> dict[str, object]:
    """One scan per source; keep only the records referenced by the pairs."""
    records: dict[str, object] = {}
    for source, path in paths.items():
        frame = pd.read_csv(path, sep="\t", dtype=str)
        for entity_id, name, address, country in zip(
            frame["entity_id"], frame["business_name"], frame["business_address"], frame["country"]
        ):
            if entity_id in wanted_ids:
                records[entity_id] = normalize_record(
                    {"entity_id": entity_id, "business_name": name or "",
                     "business_address": address or "", "country": country or ""}
                )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--source1", type=Path, required=True)
    parser.add_argument("--source2", type=Path, required=True)
    parser.add_argument("--source3", type=Path, required=True)
    parser.add_argument("--statistics", type=Path, default=None,
                        help="Existing statistics JSON to reuse (skip rebuild).")
    parser.add_argument("--statistics-out", type=Path, default=None,
                        help="Write the built statistics here.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    pairs = pd.read_parquet(args.pairs)
    missing = [k for k in PAIR_KEYS if k not in pairs.columns]
    if missing:
        raise ValueError(f"Pairs file is missing key columns {missing}")
    if pairs.duplicated(PAIR_KEYS).any():
        raise ValueError("Pairs file contains duplicated pair keys")

    if args.statistics is not None:
        stats = AlignmentStatistics.from_dict(json.loads(args.statistics.read_text()))
        verify_statistics_source(stats, args.source1)
        stats_seconds = 0.0
    else:
        stats, stats_seconds = build_statistics(args.source1)
        if args.statistics_out is not None:
            write_json(args.statistics_out, stats.to_dict())

    wanted = set(pairs["source1_entity_id"]) | set(pairs["candidate_entity_id"])
    records = _load_records(
        {"S1": args.source1, "S2": args.source2, "S3": args.source3}, wanted)

    started = time.perf_counter()
    rows = []
    for s1, cand in zip(pairs["source1_entity_id"], pairs["candidate_entity_id"]):
        left = records.get(s1)
        right = records.get(cand)
        if left is None or right is None:
            raise ValueError(f"Missing record for pair ({s1}, {cand})")
        features = build_alignment_features(left, right, stats)
        rows.append({"source1_entity_id": s1, "candidate_entity_id": cand, **features})
    pair_seconds = time.perf_counter() - started

    output = pd.DataFrame(rows, columns=PAIR_KEYS + FEATURE_NAMES)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(args.output, index=False)

    manifest = {
        "feature_version": FEATURE_VERSION,
        "statistics_version": STATISTICS_VERSION,
        "statistics_sha256": (digest(args.statistics) if args.statistics is not None else
                              hashlib.sha256(json.dumps(stats.to_dict(), sort_keys=True, separators=(",", ":")).encode()).hexdigest()),
        "feature_names": FEATURE_NAMES,
        "pair_count": int(len(output)),
        "pair_key_sha256": hashlib.sha256(
            pd.util.hash_pandas_object(output[PAIR_KEYS], index=False).to_numpy().tobytes()
        ).hexdigest(),
        "source_sha256": {
            "pairs": digest(args.pairs),
            "source1": digest(args.source1),
            "source2": digest(args.source2),
            "source3": digest(args.source3),
        },
        "code_sha256": {
            "token_alignment": digest(ROOT / "code/business_entity_resolution/src/features/token_alignment.py"),
            "builder": digest(Path(__file__).resolve()),
        },
        "seconds": {"statistics": stats_seconds, "pair_features": pair_seconds},
        "pairs_per_second": (len(output) / pair_seconds) if pair_seconds else None,
        "labels_read": False,
    }
    write_json(args.output.with_suffix(".manifest.json"), manifest)
    print(json.dumps({"pairs": len(output), "pairs_per_second": manifest["pairs_per_second"],
                      "statistics_seconds": stats_seconds, "pair_seconds": pair_seconds}, indent=2))


if __name__ == "__main__":
    main()
