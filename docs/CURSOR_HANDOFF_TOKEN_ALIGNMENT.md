# Parallel task: rare-token alignment evidence

Owner: Cursor session. Parent session owns model experiments, validation, VMs and release.

## Goal

Add evidence for true links with misspelled or transliterated words, and for false links whose shared words are only a city or generic business vocabulary. Current fresh confirmation is macro F0.5 0.979496, precision 0.995907 and recall 0.954860. These are local figures; the current portal score is 0.968.

Deliver a working feature module within 30 minutes. Do not spend this window writing another plan.

## Git coordination

- Base your separate worktree on the latest `origin/Harsha`; branch `cursor/token-alignment`.
- Own only `code/business_entity_resolution/src/features/token_alignment.py`, `scripts/build_token_alignment_features.py`, `tests/test_token_alignment.py`, and `docs/CURSOR_TOKEN_ALIGNMENT_RESULT.md`.
- Do not edit the parent session's files, switch its working directory's branch, reset its work, or touch frozen model artifacts.
- Push the branch when the first working implementation and checks pass. Record the commit, command, schema, runtime and limitations in the result document. Parent will fetch the commit and integrate it.
- Do not launch a full submission or use the VMs; parent owns their running jobs.

## Implementation

Expose `build_alignment_features(left, right, statistics) -> dict[str, float]` accepting the existing `BlockingRecord` contract. Keep the existing normalization and feature registries unchanged.

Build label-free document frequencies from the provided Source 1 texts, separately per observed country, with a global fallback. Handle country as a string. Country statistics for test inference must use the test Source 1 corpus, not training identities. Save a small versioned statistics artifact with source hashes.

Add a fixed, documented feature list for:

- One-to-one soft token alignment of names, using accent folding and a second view with the existing `phonetic()` function. Include directional weighted coverage, symmetric coverage, rarest supported token weight and unmatched distinctive token weight.
- One-to-one soft token alignment of addresses, including directional weighted coverage and unmatched distinctive tokens. Compare tokens with `rapidfuzz`; distinguish an exact token from a close spelling. Common tokens should have less weight, rather than being treated as street identity.
- A separate address view excluding tokens repeated in either record's business name. Missing evidence must have explicit flags; an empty address must not be recorded as contradictory.

Bound the alignment work per pair. Prefer exact matching first, then matching a bounded remaining token matrix. A token may support only one token on the other side. Do not let one generic word cover several different words. Do not add country-specific word lists or decision rules.

The batch script must accept pair keys in a parquet file and official source TSV paths, fetch the needed records in one scan per source, and write one row per supplied pair with the same keys. Include a manifest with feature order, source hashes, pair-key hash and code hash. No labels are needed or allowed in feature construction.

## Checks

- Same city/generic name but different distinctive streets yields weak street support.
- A one-character typo in a distinctive street or name can still provide support.
- Token transposition, accents, Indian-script phonetic views and missing addresses behave deterministically.
- No duplicated/foreign pair keys; no NaN or infinity; zero-pair input works.
- Measure at least 10,000 pair evaluations and report throughput. Prefer supplied-record samples; separate statistics-building time from pair-feature time.

## Inputs and boundaries

Official TSV files are under `/Users/sriharsha/Downloads/` and linked from `dataset/`. Existing modules: `src/features/evidence.py`, `src/preprocessing/normalize.py`, `src/blocking/contracts.py`.

Use only provided data. No business lookups, geocoding, LLM labeling, external augmentation or test pseudo-labels. Do not read `research/sprint_6h/validation/fresh_splits/audit/` labels, create new audit labels, or tune on the consumed fresh confirmation. Model selection remains the parent session's job.

If a required cache exists only on a VM, finish the module and fixture benchmark locally and push; do not wait for the full cache.
