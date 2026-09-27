"""Controlled CPU pilot for new pair features on the existing pre-bridge pool.

The ``prepare`` command joins the already-prepared neural pilot pair IDs to the
immutable 65-feature caches and adds a small set of independently computed raw
text features.  The ``train`` command compares the original 65 columns with
the augmented columns using identical LightGBM settings.

This is exploratory development evidence.  Its validation rows come from the
old early-stopping pool, not a fresh audit, and it is not comparable to the
0.97982 final-pipeline result.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
import unicodedata

import numpy as np
import pandas as pd


FEATURE_NAMES_65 = (
    "name_jaro_winkler", "name_levenshtein", "name_token_sort", "name_token_set",
    "name_partial", "name_jaccard", "name_length_ratio", "name_character_cosine",
    "address_jaro_winkler", "address_levenshtein", "address_token_sort", "address_token_set",
    "address_partial", "address_jaccard", "address_length_ratio", "address_numeric_overlap",
    "digit_token_jaccard", "exact_numeric_match", "postcode_agreement", "street_number_match",
    "exact_name", "exact_name_core", "any_name_token_match", "name_token_in_address",
    "address_token_in_name", "country_exact_match", "candidate_address_missing",
    "postcode_missing", "country_missing", "source1_name_length", "candidate_name_length",
    "source1_address_length", "candidate_address_length", "blocking_score", "source_is_s2",
    "source_is_s3", "canonical_name_sort", "canonical_name_set", "canonical_core_exact",
    "canonical_name_cosine", "canonical_name_length_ratio", "query_family_log_frequency",
    "candidate_alias_confidence", "candidate_alias_log_support", "address_ascii_containment",
    "first_numeric_canonical_exact", "numeric_canonical_jaccard", "name_weighted_jaccard",
    "address_weighted_jaccard", "strongest_shared_address_idf", "num_containment",
    "num_single_substitution", "num_disjoint_contradiction", "num_left_only", "num_right_only",
    "phonetic_name_set", "phonetic_name_sort", "phonetic_char_jaccard", "concat_ratio",
    "concat_partial", "cand_name_min_log_df", "ref_name_min_log_df", "addr_left_unmatched_idf",
    "addr_right_unmatched_idf", "addr_distinct_mismatch",
)

NEW_FEATURE_NAMES = (
    "cross_name_recovery", "cross_address_recovery", "bidirectional_field_swap",
    "cross_contamination_asymmetry", "unicode_legal_core_exact", "unicode_legal_core_jaccard",
    "accent_only_core_variant", "dotted_legal_ending_equivalent", "numeric_boundary_jaccard",
    "numeric_boundary_conflict", "numeric_boundary_sequence_exact", "numeric_field_role_jaccard",
)

LEGAL_ENDINGS = {
    "ag", "as", "bv", "co", "company", "corp", "corporation", "gmbh", "inc",
    "incorporated", "kg", "kk", "limited", "llc", "llp", "ltd", "nv", "oy",
    "plc", "pte", "pty", "sa", "sarl", "sas", "spa", "srl",
}
GENERIC_ADDRESS = {
    "address", "avenue", "ave", "boulevard", "blvd", "building", "city", "floor",
    "highway", "lane", "ln", "road", "rd", "street", "st", "suite", "unit",
}
PAIR_KEYS = ["source1_entity_id", "candidate_entity_id"]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _characters(value: str, preserve_dots: bool = False) -> str:
    value = unicodedata.normalize("NFC", str(value or "")).casefold()
    result = []
    for char in value:
        category = unicodedata.category(char)
        if category[0] in ("L", "N") or category.startswith("M") or (preserve_dots and char == "."):
            result.append(char)
        else:
            result.append(" ")
    return " ".join("".join(result).split())


def accent_fold(value: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", value)
                   if not unicodedata.combining(c))


def legal_name_parts(value: str) -> tuple[tuple[str, ...], str, bool]:
    """Return a Unicode-preserving legal core, canonical suffix and dot flag."""
    dotted = _characters(value, preserve_dots=True).split()
    tokens = []
    dot_flags = []
    for token in dotted:
        collapsed = token.replace(".", "")
        if collapsed:
            tokens.append(collapsed)
            dot_flags.append("." in token)
    ending = ""
    ending_dotted = False
    while tokens and tokens[-1] in LEGAL_ENDINGS:
        ending = tokens.pop() if not ending else tokens.pop() + " " + ending
        ending_dotted = ending_dotted or dot_flags.pop()
    return tuple(tokens), ending, ending_dotted


def _token_set(value: str, *, address: bool = False) -> frozenset[str]:
    tokens = _characters(value).split()
    excluded = GENERIC_ADDRESS if address else LEGAL_ENDINGS
    return frozenset(t for t in tokens if len(t) > 1 and not t.isdecimal() and t not in excluded)


def _jaccard(left, right) -> float:
    left, right = set(left), set(right)
    return len(left & right) / len(left | right) if left and right else 0.0


_NUMERIC = re.compile(r"\d+")


def numeric_signatures(value: str) -> tuple[tuple[str, str], ...]:
    """Extract canonical number plus whether letters touch either boundary."""
    text = accent_fold(_characters(value)).replace(" ", "_")
    signatures = []
    for match in _NUMERIC.finditer(text):
        before = text[match.start() - 1] if match.start() else "_"
        after = text[match.end()] if match.end() < len(text) else "_"
        left = "a" if before.isalpha() else "b"
        right = "a" if after.isalpha() else "b"
        number = match.group().lstrip("0") or "0"
        signatures.append((number, left + right))
    return tuple(signatures)


@dataclass(frozen=True)
class RawRecord:
    name: str
    address: str
    country: str


def pair_features(left: RawRecord, right: RawRecord) -> dict[str, float]:
    ln, rn = _token_set(left.name), _token_set(right.name)
    la, ra = _token_set(left.address, address=True), _token_set(right.address, address=True)
    direct_name, direct_address = _jaccard(ln, rn), _jaccard(la, ra)
    name_any = _jaccard(ln, rn | ra)
    address_any = _jaccard(la, ra | rn)
    lr, rl = _jaccard(ln, ra), _jaccard(la, rn)

    lc, le, ldotted = legal_name_parts(left.name)
    rc, re_, rdotted = legal_name_parts(right.name)
    unicode_exact = bool(lc and lc == rc)
    folded_equal = bool(lc and rc and tuple(map(accent_fold, lc)) == tuple(map(accent_fold, rc)))

    left_all = numeric_signatures(left.name + " " + left.address)
    right_all = numeric_signatures(right.name + " " + right.address)
    left_set, right_set = set(left_all), set(right_all)
    common_values = {v for v, _ in left_all} & {v for v, _ in right_all}
    conflicts = sum(bool({b for v, b in left_all if v == number}.isdisjoint(
                         {b for v, b in right_all if v == number})) for number in common_values)
    ln_num, rn_num = set(numeric_signatures(left.name)), set(numeric_signatures(right.name))
    la_num, ra_num = set(numeric_signatures(left.address)), set(numeric_signatures(right.address))
    role_left = {("n",) + item for item in ln_num} | {("a",) + item for item in la_num}
    role_right = {("n",) + item for item in rn_num} | {("a",) + item for item in ra_num}

    values = (
        max(0.0, name_any - direct_name), max(0.0, address_any - direct_address),
        math.sqrt(lr * rl), abs(lr - rl), float(unicode_exact), _jaccard(lc, rc),
        float(folded_equal and not unicode_exact),
        float(bool(le and le == re_ and ldotted != rdotted)), _jaccard(left_set, right_set),
        conflicts / len(common_values) if common_values else 0.0,
        float(bool(left_all and left_all == right_all)), _jaccard(role_left, role_right),
    )
    return dict(zip(NEW_FEATURE_NAMES, map(float, values)))


def deterministic_reference_split(ids, early_count: int, selection_count: int, seed: int = 70007):
    ids = list(ids)
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate validation reference IDs")
    if early_count + selection_count != len(ids):
        raise ValueError("Early-stop and selection counts must exhaust validation references")
    ordered = sorted(ids, key=lambda item: hashlib.sha256(
        f"cpu-feature-pilot:{seed}:{item}".encode()).digest())
    return ordered[:early_count], ordered[early_count:]


def validate_truth(frame: pd.DataFrame, truth: dict[str, list[str]], split: str) -> None:
    if frame.duplicated(PAIR_KEYS).any():
        raise ValueError(f"Duplicate pair IDs in {split}")
    refs = set(frame.source1_entity_id)
    if refs != set(truth):
        raise ValueError(f"{split} pair/truth reference ID mismatch")
    actual = np.fromiter((candidate in set(truth[reference]) for reference, candidate in
                          frame[PAIR_KEYS].itertuples(index=False, name=None)), dtype=np.uint8)
    labels = frame.label.to_numpy(dtype=np.uint8)
    if not np.array_equal(actual, labels):
        raise ValueError(f"{split} labels disagree with full truth")


def align_cache(pairs: pd.DataFrame, cached: pd.DataFrame, split: str) -> pd.DataFrame:
    """Strict one-to-one pair join, with hard failures for every alignment error."""
    if pairs.duplicated(PAIR_KEYS).any() or cached.duplicated(PAIR_KEYS).any():
        raise ValueError(f"Duplicate pair IDs while aligning {split}")
    joined = pairs[PAIR_KEYS + ["label", "blocking_score"]].merge(
        cached, on=PAIR_KEYS, how="left", validate="one_to_one", suffixes=("_pilot", "_cache"),
        indicator=True, sort=False)
    if not (joined._merge == "both").all() or len(joined) != len(pairs):
        raise ValueError(f"Missing cached pair IDs while aligning {split}")
    if not np.array_equal(joined.label_pilot.to_numpy(), joined.label_cache.to_numpy()):
        raise ValueError(f"Label mismatch while aligning {split}")
    if not np.allclose(joined.blocking_score_pilot, joined.blocking_score_cache,
                       rtol=1e-6, atol=1e-7):
        raise ValueError(f"Blocking-score mismatch while aligning {split}")
    # Keep the cached value under the registry's feature name after checking it
    # against the independently prepared pilot value.
    joined = joined.drop(columns=["_merge", "label_cache", "blocking_score_pilot"]).rename(
        columns={"label_pilot": "label", "blocking_score_cache": "blocking_score"})
    return joined


def load_cached_pairs(cache: Path, references: set[str], split: str) -> pd.DataFrame:
    manifest = json.loads((cache / "manifest.json").read_text())
    progress = json.loads((cache / "progress.json").read_text())
    if manifest.get("split") != split or manifest.get("feature_version") != "pairwise-v3-65":
        raise ValueError(f"Wrong cache contract for {split}")
    if progress.get("status") != "complete":
        raise ValueError(f"Incomplete {split} cache")
    columns = [*FEATURE_NAMES_65, *PAIR_KEYS, "label"]
    frames, found = [], set()
    for number in range(progress["chunks"]):
        marker_path = cache / f"{number:06d}.json"
        marker = json.loads(marker_path.read_text())
        chunk_refs = {row["entity_id"] for row in marker["entity_rows"]} & references
        if not chunk_refs:
            continue
        parquet = marker_path.with_suffix(".parquet")
        if digest(parquet) != marker["parquet_sha256"]:
            raise ValueError(f"Corrupt cache chunk: {parquet}")
        frame = pd.read_parquet(parquet, columns=columns)
        frame = frame[frame.source1_entity_id.isin(chunk_refs)]
        frames.append(frame)
        found.update(chunk_refs)
    if found != references:
        raise ValueError(f"Missing {len(references - found)} references in {split} cache")
    return pd.concat(frames, ignore_index=True)


def source_records(pilot: Path, frames: dict[str, pd.DataFrame]) -> dict[str, RawRecord]:
    reservation_path = Path("models/scale_v1_plan/sampled_references.json")
    reservation = json.loads(reservation_path.read_text())
    wanted_refs = set().union(*(set(frame.source1_entity_id) for frame in frames.values()))
    records = {}
    for fold in ("train", "tune"):
        for row in reservation[fold]:
            if row["entity_id"] in wanted_refs:
                records[row["entity_id"]] = RawRecord(row["business_name"], row["business_address"], row["country"])
    if set(records) != wanted_refs:
        raise ValueError("Missing raw source1 records from the reservation")

    wanted_targets = set().union(*(set(frame.candidate_entity_id) for frame in frames.values()))
    for source in (2, 3):
        path = Path(f"dataset/train/train_source{source}.tsv")
        source_ids = {item for item in wanted_targets if item.startswith(f"S{source}-")}
        for chunk in pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, chunksize=100_000):
            for row in chunk[chunk.entity_id.isin(source_ids)].itertuples(index=False):
                if row.entity_id in records:
                    raise ValueError(f"Duplicate raw record: {row.entity_id}")
                records[row.entity_id] = RawRecord(row.business_name, row.business_address, row.country)
    if not wanted_targets.issubset(records):
        raise ValueError(f"Missing {len(wanted_targets - set(records))} raw target records")

    # The serialized text is only an integrity check; feature computation uses raw fields above.
    for split, frame in frames.items():
        left_text = frame.source1_entity_id.map(lambda item: "name: " + records[item].name +
            " country: " + records[item].country + " address: " + records[item].address)
        right_text = frame.candidate_entity_id.map(lambda item: "name: " + records[item].name +
            " country: " + records[item].country + " address: " + records[item].address)
        if not left_text.equals(frame.text_left) or not right_text.equals(frame.text_right):
            raise ValueError(f"Raw TSV/reservation records disagree with serialized {split} text")
    return records


def add_new_features(frame: pd.DataFrame, records: dict[str, RawRecord]) -> pd.DataFrame:
    unique_ids = set(frame.source1_entity_id) | set(frame.candidate_entity_id)
    prepared = {item: records[item] for item in unique_ids}
    rows = [pair_features(prepared[left], prepared[right]) for left, right in
            frame[PAIR_KEYS].itertuples(index=False, name=None)]
    return pd.concat([frame.reset_index(drop=True), pd.DataFrame(rows, columns=NEW_FEATURE_NAMES)], axis=1)


def prepare(args) -> None:
    if args.output.exists():
        raise FileExistsError("Use a fresh preparation directory")
    import pyarrow as pa
    pa.set_cpu_count(min(8, os.cpu_count() or 1))
    args.output.mkdir(parents=True)
    manifest_path = args.pilot / "manifest.json"
    pilot_manifest = json.loads(manifest_path.read_text())
    frames, truths = {}, {}
    for split in ("train", "validation"):
        path = args.pilot / f"{split}.parquet"
        if digest(path) != pilot_manifest["summaries"][split]["sha256"]:
            raise ValueError(f"Pilot {split} hash mismatch")
        frames[split] = pd.read_parquet(path)
        truths[split] = json.loads((args.pilot / f"{split}_truth.json").read_text())
        validate_truth(frames[split], truths[split], split)
    train_refs, validation_refs = set(truths["train"]), set(truths["validation"])
    if train_refs & validation_refs:
        raise ValueError("Train/validation reference fold leakage")
    train_truth = set().union(*map(set, truths["train"].values()))
    validation_truth = set().union(*map(set, truths["validation"].values()))
    if train_truth & validation_truth:
        raise ValueError("Train/validation truth-target leakage")
    if set(frames["train"].candidate_entity_id) & validation_truth:
        raise ValueError("Validation-owned target leaked into training candidates")

    cached_train = load_cached_pairs(args.train_cache, train_refs, "train")
    cached_validation = load_cached_pairs(args.tune_cache, validation_refs, "tune")
    aligned = {
        "train": align_cache(frames["train"], cached_train, "train"),
        "validation": align_cache(frames["validation"], cached_validation, "validation"),
    }
    records = source_records(args.pilot, frames)
    early_ids, selection_ids = deterministic_reference_split(
        validation_refs, args.early_references, len(validation_refs) - args.early_references, args.seed)
    fold_ids = {"train": sorted(train_refs), "early_stop": early_ids, "selection": selection_ids}
    owned_targets = {
        fold: set().union(*(set(truths["train" if fold == "train" else "validation"][item])
                            for item in ids))
        for fold, ids in fold_ids.items()
    }
    if any(owned_targets[left] & owned_targets[right] for left, right in
           (("train", "early_stop"), ("train", "selection"), ("early_stop", "selection"))):
        raise ValueError("Owned truth targets overlap across prepared folds")
    fold_frames = {
        "train": aligned["train"],
        "early_stop": aligned["validation"][aligned["validation"].source1_entity_id.isin(early_ids)].copy(),
        "selection": aligned["validation"][aligned["validation"].source1_entity_id.isin(selection_ids)].copy(),
    }
    outputs = {}
    for fold, frame in fold_frames.items():
        if set(frame.source1_entity_id) != set(fold_ids[fold]):
            raise ValueError(f"Incomplete {fold} fold")
        frame = add_new_features(frame, records)
        counts = frame.groupby("source1_entity_id").label.transform("size")
        frame["sample_weight"] = (1.0 / counts).astype(np.float32)
        frame["country"] = frame.source1_entity_id.map(lambda item: records[item].country)
        columns = [*PAIR_KEYS, "label", "sample_weight", "country", *FEATURE_NAMES_65, *NEW_FEATURE_NAMES]
        frame = frame[columns]
        if not np.isfinite(frame[[*FEATURE_NAMES_65, *NEW_FEATURE_NAMES, "sample_weight"]].to_numpy()).all():
            raise ValueError(f"Nonfinite values in {fold}")
        path = args.output / f"{fold}.parquet"
        frame.to_parquet(path, index=False)
        truth = {item: truths["train" if fold == "train" else "validation"][item] for item in fold_ids[fold]}
        truth_path = args.output / f"{fold}_truth.json"
        write_json(truth_path, truth)
        outputs[fold] = {"references": len(truth), "pairs": len(frame), "positive_pairs": int(frame.label.sum()),
                         "parquet_sha256": digest(path), "truth_sha256": digest(truth_path)}
        print(json.dumps({"stage": "fold_prepared", "fold": fold, **outputs[fold]}), flush=True)

    output_manifest = {
        "status": "prepared_not_trained", "feature_schema": "pairwise-v3-65+cpu-pilot-12",
        "baseline_features": list(FEATURE_NAMES_65), "new_features": list(NEW_FEATURE_NAMES),
        "folds": outputs, "seed": args.seed, "early_stop_references": args.early_references,
        "preparation_thread_cap": 8,
        "pilot_manifest_sha256": digest(manifest_path), "source_sha256": digest(Path(__file__)),
        "data_scope": "existing pre-bridge caches and old early-stopping pool; no audit or test data",
        "comparison_scope": "exploratory only; not evidence of improvement over the 0.97982 final pipeline",
        "raw_record_sources": ["models/scale_v1_plan/sampled_references.json",
                               "dataset/train/train_source2.tsv", "dataset/train/train_source3.tsv"],
    }
    write_json(args.output / "manifest.json", output_manifest)
    print(json.dumps({"stage": "prepare_complete", "output": str(args.output)}), flush=True)


def entity_metrics(probabilities, threshold, frame, truth, reference_ids=None):
    ids = list(reference_ids) if reference_ids is not None else sorted(truth)
    if set(ids) != set(truth):
        raise ValueError("Metric reference universe differs from full truth")
    index = {item: number for number, item in enumerate(ids)}
    groups = frame.source1_entity_id.map(index)
    if groups.isna().any():
        raise ValueError("Scored pair lies outside metric reference universe")
    groups = groups.to_numpy(dtype=np.int64)
    expected = np.array([len(truth[item]) for item in ids], dtype=np.int64)
    labels = frame.label.to_numpy(dtype=bool)
    accepted = np.asarray(probabilities) >= threshold
    predicted = np.bincount(groups[accepted], minlength=len(ids))
    tp = np.bincount(groups[accepted & labels], minlength=len(ids))
    denominator = predicted + .25 * expected
    values = np.divide(1.25 * tp, denominator, out=np.zeros(len(ids)), where=denominator > 0)
    values[(expected == 0) & (predicted == 0)] = 1.0
    result = {"threshold": float(threshold), "macro_f05": float(values.mean()), "entities": len(ids),
              "true_links": int(expected.sum()), "predicted_links": int(predicted.sum()),
              "true_positive_links": int(tp.sum()), "micro_precision": float(tp.sum() / max(1, predicted.sum())),
              "micro_recall": float(tp.sum() / max(1, expected.sum())),
              "singleton_entities": int((expected == 0).sum()),
              "singleton_false_positives": int(((expected == 0) & (predicted > 0)).sum()),
              "non_singleton_empty_predictions": int(((expected > 0) & (predicted == 0)).sum())}
    return result, values


def tune_threshold(probabilities, frame, truth):
    """Find the exact selection-set optimum by sweeping distinct probabilities."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    ids = sorted(truth)
    index = {item: number for number, item in enumerate(ids)}
    groups = frame.source1_entity_id.map(index).to_numpy(dtype=np.int64)
    labels = frame.label.to_numpy(dtype=np.uint8)
    expected = np.array([len(truth[item]) for item in ids], dtype=np.int64)
    predicted = np.zeros(len(ids), dtype=np.int64)
    tp = np.zeros(len(ids), dtype=np.int64)
    values = np.zeros(len(ids), dtype=np.float64)
    values[expected == 0] = 1.0
    total = float(values.sum())
    total_tp = total_pred = 0
    best = (total / len(ids), 1.0, np.nextafter(probabilities.max(initial=0.0), np.inf))
    order = np.argsort(-probabilities, kind="stable")
    position = 0
    while position < len(order):
        end = position + 1
        score = probabilities[order[position]]
        while end < len(order) and probabilities[order[end]] == score:
            end += 1
        chosen = order[position:end]
        affected = np.unique(groups[chosen])
        total -= values[affected].sum()
        np.add.at(predicted, groups[chosen], 1)
        np.add.at(tp, groups[chosen], labels[chosen])
        total_pred += len(chosen)
        total_tp += int(labels[chosen].sum())
        denominator = predicted[affected] + .25 * expected[affected]
        values[affected] = np.divide(1.25 * tp[affected], denominator,
                                     out=np.zeros(len(affected)), where=denominator > 0)
        total += values[affected].sum()
        macro = total / len(ids)
        precision = total_tp / max(1, total_pred)
        candidate = (macro, precision, score)
        if candidate[:2] > best[:2]:
            best = candidate
        position = end
    result, values = entity_metrics(probabilities, best[2], frame, truth, ids)
    return result, values


def paired_interval(difference: np.ndarray, seed: int, replicates: int = 5000) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    means = np.empty(replicates)
    for start in range(0, replicates, 250):
        size = min(250, replicates - start)
        samples = rng.integers(0, len(difference), size=(size, len(difference)))
        means[start:start + size] = difference[samples].mean(axis=1)
    low, high = np.quantile(means, [.025, .975])
    return {"mean_gain": float(difference.mean()), "paired_bootstrap_95_low": float(low),
            "paired_bootstrap_95_high": float(high), "replicates": replicates}


def load_prepared(data: Path):
    manifest = json.loads((data / "manifest.json").read_text())
    frames, truths = {}, {}
    for fold in ("train", "early_stop", "selection"):
        path = data / f"{fold}.parquet"
        truth_path = data / f"{fold}_truth.json"
        if digest(path) != manifest["folds"][fold]["parquet_sha256"] or \
                digest(truth_path) != manifest["folds"][fold]["truth_sha256"]:
            raise ValueError(f"Prepared {fold} hash mismatch")
        frames[fold] = pd.read_parquet(path)
        truths[fold] = json.loads(truth_path.read_text())
        validate_truth(frames[fold], truths[fold], fold)
    sets = [set(truths[fold]) for fold in ("train", "early_stop", "selection")]
    if any(sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Prepared reference folds overlap")
    return manifest, frames, truths


def train(args) -> None:
    if args.output.exists():
        raise FileExistsError("Use a fresh model output directory")
    if not 1 <= args.threads <= 8:
        raise ValueError("This experiment requires 1..8 LightGBM threads")
    args.output.mkdir(parents=True)
    os.nice(10)
    import lightgbm as lgb

    started = time.perf_counter()
    manifest, frames, truths = load_prepared(args.data)
    params = dict(objective="binary", metric="binary_logloss", learning_rate=.04, num_leaves=31,
                  min_data_in_leaf=30, lambda_l2=2., feature_fraction=.9, bagging_fraction=.9,
                  bagging_freq=1, max_bin=127, seed=args.seed, num_threads=args.threads,
                  deterministic=True, force_col_wise=True, verbosity=-1)
    protocol = {
        "status": "exploratory_old_pool_only", "parameters": params, "maximum_trees": args.trees,
        "early_stopping_rounds": args.early_stopping_rounds,
        "arms": {"baseline65": list(FEATURE_NAMES_65),
                 "baseline65_plus_cpu12": [*FEATURE_NAMES_65, *NEW_FEATURE_NAMES]},
        "threshold_selection": "exact probability sweep on selection fold only, independently per arm",
        "prepared_manifest_sha256": digest(args.data / "manifest.json"),
        "source_sha256": digest(Path(__file__)), "threads_per_job": args.threads,
        "scope": "existing pre-bridge cache and old early-stopping pool; not fresh audit; no test data",
        "comparison": "not comparable to the final-pipeline 0.97982 development result",
    }
    write_json(args.output / "protocol.json", protocol)
    outcomes, selection_values, probabilities = {}, {}, {}
    for arm, columns in protocol["arms"].items():
        train_frame, early_frame = frames["train"], frames["early_stop"]
        training = lgb.Dataset(train_frame[columns].to_numpy(dtype=np.float32),
            label=train_frame.label, weight=train_frame.sample_weight, feature_name=columns)
        validation = lgb.Dataset(early_frame[columns].to_numpy(dtype=np.float32),
            label=early_frame.label, weight=early_frame.sample_weight, feature_name=columns,
            reference=training)
        model = lgb.train(params, training, num_boost_round=args.trees, valid_sets=[validation],
            valid_names=["early_stop"], callbacks=[lgb.early_stopping(args.early_stopping_rounds),
                                                   lgb.log_evaluation(100)])
        model_path = args.output / f"{arm}.txt"
        model.save_model(str(model_path))
        selection_p = model.predict(frames["selection"][columns].to_numpy(dtype=np.float32),
                                    num_threads=args.threads)
        probabilities[arm] = selection_p
        chosen, values = tune_threshold(selection_p, frames["selection"], truths["selection"])
        selection_values[arm] = values
        early_p = model.predict(early_frame[columns].to_numpy(dtype=np.float32), num_threads=args.threads)
        early_fixed, _ = entity_metrics(early_p, chosen["threshold"], early_frame, truths["early_stop"])
        outcomes[arm] = {"best_iteration": model.best_iteration, "selection": chosen,
                         "early_stop_at_selection_threshold": early_fixed,
                         "model_sha256": digest(model_path),
                         "feature_importance_gain": dict(zip(columns, map(float, model.feature_importance("gain"))))}
        print(json.dumps({"stage": "arm_complete", "arm": arm, "best_iteration": model.best_iteration,
                          **chosen}), flush=True)

    ids = sorted(truths["selection"])
    country_by_reference = frames["selection"].drop_duplicates("source1_entity_id").set_index(
        "source1_entity_id").country.to_dict()
    countries = np.array([country_by_reference[item] for item in ids])
    difference = selection_values["baseline65_plus_cpu12"] - selection_values["baseline65"]
    slices = {}
    for country in sorted(set(countries)):
        selected = countries == country
        slices[country] = {"entities": int(selected.sum()),
                           "baseline65_macro_f05": float(selection_values["baseline65"][selected].mean()),
                           "augmented_macro_f05": float(selection_values["baseline65_plus_cpu12"][selected].mean()),
                           "gain": float(difference[selected].mean())}
    oracle, _ = entity_metrics(frames["selection"].label.to_numpy(), .5, frames["selection"],
                               truths["selection"])
    report = {
        "status": "exploratory_old_prebridge_pool_not_fresh_audit", "arms": outcomes,
        "paired_selection": paired_interval(difference, args.seed), "country_slices": slices,
        "retained_candidate_oracle": oracle, "selection_references": len(ids),
        "selection_pairs": len(frames["selection"]), "fresh_audit_evaluated": False,
        "final_pipeline_0_97982_comparison_valid": False, "seconds": time.perf_counter() - started,
    }
    score_frame = frames["selection"][PAIR_KEYS + ["label", "country"]].copy()
    for arm, values in probabilities.items():
        score_frame[f"{arm}_probability"] = values
    score_frame.to_parquet(args.output / "selection_scores.parquet", index=False)
    write_json(args.output / "report.json", report)
    print(json.dumps({"stage": "complete", "paired_selection": report["paired_selection"],
                      "output": str(args.output)}), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prep = subparsers.add_parser("prepare", help="Join pilot IDs to cache features and raw records")
    prep.add_argument("--pilot", type=Path, default=Path("models/neural_pilot_20260927_e70007"))
    prep.add_argument("--train-cache", type=Path, default=Path("models/scale_v1_run/cache_train"))
    prep.add_argument("--tune-cache", type=Path, default=Path("models/scale_v1_run/cache_tune"))
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--early-references", type=int, default=2500)
    prep.add_argument("--seed", type=int, default=70007)
    prep.set_defaults(func=prepare)
    fit = subparsers.add_parser("train", help="Fit and compare baseline and augmented LightGBM arms")
    fit.add_argument("--data", type=Path, required=True)
    fit.add_argument("--output", type=Path, required=True)
    fit.add_argument("--threads", type=int, default=8)
    fit.add_argument("--trees", type=int, default=1200)
    fit.add_argument("--early-stopping-rounds", type=int, default=75)
    fit.add_argument("--seed", type=int, default=70007)
    fit.set_defaults(func=train)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if getattr(args, "early_references", 1) < 1 or getattr(args, "trees", 1) < 1 or \
            getattr(args, "early_stopping_rounds", 1) < 1:
        raise ValueError("Counts must be positive")
    args.func(args)


if __name__ == "__main__":
    main()
