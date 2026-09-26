# Business entity resolution

## Current implementation

The runnable component is candidate generation: eight retrieval paths, bounded integer postings, symmetric reranking, per-source top-K, sharded target processing, a disk-backed merge, and blocking diagnostics. Model training and final matching are separate stages. No real-data recall or competition score has been measured yet.

The candidate list emitted here is ready for the matcher. It becomes the submission candidate file only when the matcher actually scores precisely that set. No `matching_results.tsv` is produced by this command.

The original seven-path baseline remains available through `src.blocking.generate_candidates` and its existing keyword arguments. Its top-K cap now applies separately to S2 and S3. The eight-path in-memory API is `src.blocking.generate_scalable_candidates` or `CandidateGenerator`; the CLI below uses this implementation. The two retrieval scores have different definitions and should not share a trained model without regenerating features and recalibrating.

## Environment

Use Python 3.11 or 3.12 and the pinned requirements. The fixture suite was verified on Python 3.12; a Python 3.11 run is still required before choosing that release environment.

From this directory:

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

Use the virtual environment's Python executable for subsequent commands. From the repository root, development tests use:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest tests -q
```

## Candidate generation

```bash
python run_blocking.py \
  --source1 /path/to/dataset/train/train_source1.tsv \
  --source2 /path/to/dataset/train/train_source2.tsv \
  --source3 /path/to/dataset/train/train_source3.tsv \
  --ground-truth /path/to/dataset/train/train_ground_truth.tsv \
  --output-dir /path/to/blocking-validation \
  --top-k 20 \
  --shard-size 100000 \
  --query-batch-size 1000
```

For genuine validation, supply the agreed held-out source files and their ground truth. Running against full training files is useful for profiling but is not an independent validation result. Blocking never reads labels to choose candidates.

For test inference, use the three test source files and omit `--ground-truth`.

Outputs:

- `candidate_pairs.tsv`: one list-format row per supplied Source 1 entity.
- `candidate_scores.tsv`: the exact retained pair table, with source, score, rank, and paths, for model feature construction.
- `blocking_diagnostics.json`: configuration, sizes, candidate counts, runtime, peak process memory where supported, and optional labeled retrieval metrics.

No exact-match key is created from empty text. Countries are unrestricted and never used as a hard filter. Numeric tokens retain leading zeros and field provenance. Upstream postcode candidates are preferred; the legacy fallback considers 4–6 digit address tokens as uncertain candidates, not verified postcodes.

## Retrieval and reranking

Paths include rare name tokens, address tokens, numeric tokens, exact names, exact cores, core prefixes, postcode/name keys, and name character n-grams. Query postings are bounded by relative and absolute document-frequency limits. Character retrieval uses a bounded subset of rare grams with binary TF-IDF normalization; it is approximate retrieval, not a full matrix multiplication.

The union is reranked using IDF-weighted name/address/character overlap, fuzzy name/address comparisons, numeric overlap, exactness bonuses, path agreement, and a soft country-disagreement penalty. This is a retrieval score, not a match probability.

The first pass computes label-free target-corpus frequencies. The second pass builds one target shard at a time. Retained candidates are merged into per-reference/per-source top-K on disk. Source 1 normalized records are also stored on disk and reused across shards.

Corpora are processed independently for S2 and S3. Corpus statistics are shared across shards to keep weights consistent. Changing shard size can still change path-level truncation/provenance; compare it on validation before fixing a release configuration.

Global vocabulary counters remain in memory, and SQLite work files use disk space. Sharding limits target-record and posting memory; it does not guarantee a fixed total RAM footprint. Measure vocabulary sizes, runtime, and storage on the actual dataset before scaling up. The Python implementation is a baseline whose full-corpus throughput still needs profiling.

## Development limits and diagnostics

`--reference-limit N` profiles a subset of references against the supplied targets. `--target-limit N` also shrinks the distractor pool and can inflate apparent retrieval performance; do not report it as full-pool recall.

`--config /path/to/config.json` overrides `BlockingConfig` settings. Unknown settings and invalid budgets fail. Compare K=5, 10, 20, and 40 against the same held-out pools, and record matching score as well as retrieval coverage and candidate counts. An initial 95% link-recall milestone is insufficient evidence for a near-0.99 matching score.

## Submission package

Follow `docs/TEAM_PLAN.md` and `ARCHITECTURE.md` in the repository for ownership, evaluation, and release requirements. Include this directory, both final output TSVs, and the completed organizer methodology template in the final archive. Use the organizer-provided validator; the repository's existing validator placeholder is not a release check.
