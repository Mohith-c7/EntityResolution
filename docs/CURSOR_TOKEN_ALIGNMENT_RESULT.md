# Rare-Token Alignment Evidence — Result

This document records the original v1 implementation at `43e14c633f45193611c8b2ea96d094dac957a919`. For corrected frequency views, v2 artifact compatibility, provenance checks and current commands, see [CURSOR_TOKEN_ALIGNMENT_V2_REVIEW.md](CURSOR_TOKEN_ALIGNMENT_V2_REVIEW.md).

Owner: Cursor session (parallel task). Parent session owns model experiments,
validation, VMs and release. This module adds evidence only; it changes no
existing model, registry, or output.

## What was delivered

A feature module that gives the matcher a one-to-one **soft token alignment**
view of names and addresses, weighted by label-free document frequency. It is
aimed at two failure modes:

* **True links with misspelled / transliterated words.** A one-character typo
  in a distinctive street or name still contributes support (scored below an
  exact match), instead of losing the token entirely.
* **False links whose shared words are only a city or generic business
  vocabulary.** Common tokens carry little weight, so a shared city/generic
  word yields weak street support and cannot masquerade as street identity.

## Entry points

* `code/business_entity_resolution/src/features/token_alignment.py`
  * `build_alignment_features(left, right, statistics) -> dict[str, float]`
    accepts the existing `BlockingRecord` contract (`.name`, `.address`,
    `.country`). Returns the fixed 16-feature vector below.
  * `AlignmentStatistics` — versioned label-free per-country document
    frequencies with a global fallback; `to_dict` / `from_dict`.
* `scripts/build_token_alignment_features.py` — batch builder.
* `tests/test_token_alignment.py` — behavior and robustness checks.

## Run command

```bash
python scripts/build_token_alignment_features.py \
    --pairs pairs.parquet \
    --source1 dataset/test/test_source1.tsv \
    --source2 dataset/test/test_source2.tsv \
    --source3 dataset/test/test_source3.tsv \
    --statistics-out token_alignment_stats.json \
    --output token_alignment_features.parquet
```

`--pairs` is a parquet with `source1_entity_id`, `candidate_entity_id`. Records
are fetched in one scan per source; output is one row per supplied pair with
the same keys. Pass `--statistics <json>` to reuse a built statistics artifact
(skips the rebuild). For test inference the statistics are built from the
**test** Source 1 corpus, not training identities. No labels are read.

## Feature schema (fixed order, `FEATURE_NAMES`, version `token-alignment-v1`)

Name, primary (accent-folded) view:
1. `align_name_coverage_l2r` — directional weighted coverage, left → right.
2. `align_name_coverage_r2l` — directional weighted coverage, right → left.
3. `align_name_coverage_sym` — symmetric coverage `min(l2r, r2l)`.
4. `align_name_rarest_supported` — largest weight among supported tokens.
5. `align_name_unmatched_distinctive` — largest weight among distinctive
   (high-IDF) tokens with no support on the other side.

Name, phonetic view (via existing `phonetic()`):
6. `align_name_phon_coverage_sym`
7. `align_name_phon_unmatched_distinctive`

Address, full view:
8. `align_addr_coverage_l2r`
9. `align_addr_coverage_r2l`
10. `align_addr_coverage_sym`
11. `align_addr_unmatched_distinctive`

Address view excluding tokens repeated in either record's business name:
12. `align_addrx_coverage_sym`
13. `align_addrx_unmatched_distinctive`

Missing-evidence flags (1 = evidence absent; never a contradiction):
14. `align_name_missing`
15. `align_addr_missing`
16. `align_addrx_missing`

## Method

* **Statistics.** Label-free document frequencies over the supplied Source 1
  `name + address` tokens, separately per observed country (country is an open
  string) with a global fallback. IDF = `log((N+1)/(df+1)) + 1`. Versioned
  artifact (`token-alignment-stats-v1`) with source SHA-256 hashes.
* **Alignment.** Exact token matches are resolved first (one-to-one over
  multisets). Remaining tokens (at most `MAX_FUZZY_TOKENS = 8` per side) enter a
  bounded fuzzy matrix using `rapidfuzz.fuzz.ratio`; pairs at or above
  `SOFT_MATCH_THRESHOLD = 80` are matched greedily best-first, one-to-one, at
  `SOFT_MATCH_DISCOUNT = 0.5` of the token weight scaled by similarity. A token
  supports at most one token on the other side, so one generic word cannot
  cover several different words.
* **Distinctive tokens.** A token is distinctive at IDF ≥ `DISTINCTIVE_IDF =
  4.0`; the unmatched-distinctive features surface a distinctive word that
  found no support.
* **Missing evidence.** An empty name/address sets an explicit flag and zero
  coverage; it is never recorded as contradictory.

No country-specific word lists or decision rules are used.

## Commit

* Branch: `cursor/token-alignment` (based on `origin/Harsha`).
* Commit: `43e14c633f45193611c8b2ea96d094dac957a919` (pushed to origin).
* Owned files only: the two source files, the test file, and this document.

## Runtime (local, Apple Silicon, `/tmp/entity-review-venv` Python)

Measured on 12,000 supplied-record pairs (test Source 1 × test Source 2):

* Statistics build over the full 1,732,544-row test Source 1 corpus:
  **49.4 s** (one-time; reusable via `--statistics`).
* Pair-feature time: **2.05 s** for 12,000 pairs → **≈ 5,853 pairs/second**.
* Zero-pair input: 0 rows, 18 columns, no error.

## Checks performed

* Same city/generic name + different distinctive streets → weak street support
  and a positive unmatched-distinctive signal.
* One-character typo in a distinctive street retains most address support.
* Accents fold deterministically; repeated runs give identical features.
* A single generic token cannot cover several tokens on the other side
  (one-to-one cap verified).
* Output: 12,000 rows, no duplicated pair keys, no NaN or infinity; zero-pair
  input produces an empty frame with the full schema.

## Limitations

* Statistics build is single-threaded and scales with the Source 1 corpus size
  (~49 s for 1.7M rows); reuse the artifact for repeated runs.
* The fuzzy matrix is capped at 8 remaining tokens per side; very long
  addresses rely on exact matches plus the 8 most salient leftovers.
* Features are evidence only. Model selection, validation, and any decision
  rule remain the parent session's job; no labels were read or used.
