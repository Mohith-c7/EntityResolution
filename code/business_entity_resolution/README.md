# Business entity resolution

## Environment and inputs

Tested with Python 3.12.12 and the pinned requirements. SQLite must support FTS5 and its trigram tokenizer. Python 3.11 has not been verified for this release.

From this code directory:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On macOS, install OpenMP with `brew install libomp` first if LightGBM's native build needs it. Windows activation is `.venv\Scripts\Activate.ps1`.

Put the four training TSVs under `dataset/train/` and the three test TSVs under `dataset/test/`. Preserve their organizer filenames. Source columns are exactly `entity_id`, `business_name`, `business_address`, `country`. Ground-truth columns are `source1_entity_id`, `matched_entity_ids`. All reads use explicit tabs; numeric evidence remains strings.

The commands below run from the repository root. In an extracted submission package, run from the directory containing `code/`. Supply explicit data paths if data is elsewhere. Raw data, indexes, and model artifacts are not committed.

## Reproduce the measured pilot

```bash
python code/business_entity_resolution/run_training.py \
  --train-dir dataset/train --index-dir models \
  --artifact-dir models/pilot_alias_v4 \
  --train-entities 2000 --tune-entities 500 --holdout-entities 500 \
  --top-k 20 --character-mode off --reranker-version v4 \
  --numeric-conjunctions --numeric-location-queries --use-name-aliases \
  --max-address-df 100000 --workers 4 --threads 4
```

Use a fresh artifact directory for each completed experiment. Matching input indexes may be reused. Changed data or fitted name-channel weights require fresh derived indexes. Input fingerprints use paths, sizes, and modification times; moving data can require a rebuild.

The pipeline builds complete S2/S3 SQLite FTS indexes, retrieves bounded candidates, generates pair features, trains LightGBM with retrieved hard negatives, tunes a macro F₀.₅ threshold, and evaluates held-out S1 entities. Missing positives are never inserted into candidates using labels. Each training S1 has equal total pair weight. Held-out targets are excluded from classifier-training examples, including negatives.

Artifacts include `model.txt`, `report.json`, `name_aliases.json`, feature registry, full split manifest, sampled references/labels, pair Parquet files, threshold sweep, holdout predictions/errors, and model license. New runs also emit source hashes and input/index fingerprints in `reproduction.json`.

The measured improved pilot achieved **0.9267946 macro F₀.₅**, **0.9758643 precision**, and **0.8717949 recall** on 500 held-out S1 entities against all 10,320,219 training targets. Its threshold was 0.74. Its classifier used 2,000 training S1 entities; its name channel used all 1,544,990 eligible training-fold S1 names and their 5,348,430 matched targets. Threshold tuning used another 500 S1 entities. These are exploratory local results, not leaderboard scores.

## Name channel and features

The name channel learns reusable name strings from only matched training-fold targets. It does not use entity IDs as model features. Confident mappings require support ≥2 and dominance ≥0.98. Ambiguous names retain up to three supported alternatives as conditional evidence without forced canonical rewriting.

The base registry contains 36 features: name/address similarity, numeric evidence, exactness/missingness flags, lengths, source flags, country agreement, and retrieval score. The versioned 50-feature extension adds canonical-name similarity, alias confidence/support, family frequency, address containment, canonical numeric agreement, and IDF evidence. Train/inference validate exact feature names and order.

## Validation protocol

Stable seed-42 S1 hashing creates 70% training, 15% tuning, and 15% holdout folds. Labeled targets follow their S1 owner. The complete audit found no target owned by multiple S1 entities; overlapping labels would require component grouping. S1 sampling scans the full file and is independent of row order; it is not exact stratification.

Validation retrieval uses the complete training target pool as a stress test. No held-out target identity contributes to supervised fitting. Target-corpus frequencies are label-free. France has no training labels, so this protocol cannot establish French accuracy. Pilot holdouts inspected during development remain exploratory; use a larger fresh audit population after freezing settings.

For nonempty truth, `F0.5 = 1.25*TP / (0.25*true_count + predicted_count)`. Empty truth and empty prediction score 1; a singleton with any prediction scores 0. Average over every S1 entity. Sweep thresholds from 0.30–0.95 plus high-threshold points; select tuning macro F₀.₅, then precision and higher threshold on ties. Model outputs are classification scores, without a probability-calibration claim.

## Retrieval experiments

The measured pilot uses lexical, numeric/location, and learned-name retrieval, followed by address-aware reranking. It retains at most 20 candidates per source. Query caps do not restrict the indexed target universe.

`--retrieval-mode anchored` is an experimental alternative using name-family hashes, address tokens, canonical numbers, and dynamic country scope in a derived FTS index. It includes bounded fallbacks. Refit and validate the matcher for a selected retrieval change; do not silently substitute different candidates into an existing model.

From the repository root:

```bash
python scripts/probe_retrieval.py \
  --artifact-dir models/pilot_alias_v4 \
  --output reports/retrieval_probe_anchored.json \
  --entities 100 --retrieval-mode anchored --path-top-k 40
```

Retrieval oracle scores assume perfect classification and are ceilings, not achieved model scores. Compare matching quality, recall, mean/p95 candidates, and runtime for K=5/10/20. Do not prune candidates after model scoring to shrink the reported export.

The earlier in-memory/sharded interfaces and `run_blocking.py` remain separate baselines with different retrieval-score definitions.

## Generate test outputs

After selecting and validating an artifact:

```bash
python code/business_entity_resolution/run_inference.py \
  --test-dir dataset/test --index-dir models \
  --artifact-dir models/pilot_alias_v4 \
  --output-dir output --workers 4 --batch-size 250
```

Inference reads saved search settings, feature version, name channel, and threshold. It streams every S1 entity through bounded batches and exports the exact scored candidate set. No match is forced.

A positive `--reference-limit` creates incomplete smoke outputs, marked unsafe for submission. Use a separate output directory. Full inference with the current wide configuration needs throughput improvement before release.

## Test, validate, and package

From the repository root:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q

python3 utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv --test-dir dataset/test

python3 utils/organizer_validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv --test-dir dataset/test --check-ids
```

Both validators use only the standard library. The strict validator enforces physical row shape, nonempty ID tokens, coverage, membership, duplicates, and candidate subsets. The official helper is unmodified; target-ID membership is optional unless `--check-ids` is supplied.

The final archive includes both complete TSVs under `output/`, this code directory with pinned requirements, and the completed `Documentation_template.md`. No full official-test result, competition-ready archive, or leaderboard upload is claimed yet.

## License and data rules

No external identity lookup, geocoding, business registries, or hosted model services are used. Unicode primary views are preserved; Latin-accent and leading-zero alternatives are explicit. Country labels are unrestricted strings. No pretrained neural model is used. LightGBM 4.3.0 is MIT-licensed; generated tree-model and name-channel artifacts carry `src/model/MODEL_LICENSE.txt`.
