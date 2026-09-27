# Business entity resolution: four-layer neural release

Run commands from the extracted archive directory containing `code/` and
`output/`. The two submitted files are `output/matching_results.tsv` and
`output/candidate_pairs.tsv`. Candidate output contains every forward-model
scored pair before decision thresholds are applied.

All inference source is included under `code/business_entity_resolution/src/`.
`src/final_pipeline/` preserves the original `scripts/`, `research/`, and core
source relationships so the saved code and feature hashes remain valid. The
additional inner core copy is intentional: moving original modules or rewriting
imports would invalidate their provenance checks. The entry point below locates
this runtime itself; it does not require the development repository.

Offline model assets include the selected tree models, learned aliases, original
frozen encoder head, multilingual MiniLM base checkpoint/tokenizer/license, and
the selected four-layer continuation checkpoint/tokenizer. Raw datasets, labels,
training caches, private credentials, and target indices are excluded. Target
and Source1 reverse indices can be rebuilt from the organizer-provided TSVs.
No network calls occur during inference; Transformers uses local files only.

## Environments

CPU indexing, features, tree inference, export and validation use Python 3.12.
The pinned CPU requirements are in `requirements.txt`, including LightGBM 4.3.0.
The neural scorer uses a separate Apple Silicon environment tested with
Python 3.13.12, PyTorch 2.14.0, Transformers 4.57.6, and MPS. CPU neural fallback is
disabled. A Python 3.12 MPS environment has not been independently verified.

```bash
python3.12 -m venv .venv-cpu
.venv-cpu/bin/python -m pip install -r code/business_entity_resolution/requirements.txt
python3.13 -m venv .venv-neural
.venv-neural/bin/python -m pip install -r code/business_entity_resolution/requirements-neural.txt
```

On macOS install OpenMP, a C compiler, and loadable-extension SQLite headers
before indexing (for example `brew install libomp sqlite`). Linux CPU workers
need a C compiler and SQLite development headers. SQLite must provide FTS5,
the trigram tokenizer, and loadable extensions. The neural stage requires an
Apple Silicon Mac with MPS; the CPU stages may run separately on a Linux host.
Dependencies can be installed from a local wheelhouse with `--no-index
--find-links /path/to/wheels`. Model weights need no download.

## Verify the extracted package

```bash
.venv-cpu/bin/python code/business_entity_resolution/src/run_final_submission.py verify-assets
.venv-cpu/bin/python code/business_entity_resolution/src/run_final_submission.py verify-sample \
  --output /absolute/path/to/new/subset-verification
```

The real numeric fixture contains 100 exposed runtime owners, all 4,000 original
candidate pairs, and 221 routed pairs. It has no raw business records or truth.
The command executes the production four-layer adapter function and unchanged
decision function, checks exact pair/order preservation, unchanged unrouted
scores, and parity with sealed reference probabilities and selection masks.
This checks packaged sparse scoring and decisions; it does not establish a cold
full-test reproduction, retrieval recall, neural throughput, or quality score.

## Reproduce inference from the supplied test data

Place only the three organizer test TSVs in an external directory: filenames
`test_source1.tsv`, `test_source2.tsv`, `test_source3.tsv`, with columns
`entity_id`, `business_name`, `business_address`, `country`. Preserve strings,
tabs, Unicode, and original row order. The submitted package contains no dataset.
Use a new external work directory with sufficient disk space and a CPU machine
large enough for the complete 69,301,760-pair graph (128 GB RAM recommended).
Index construction and cold full inference take longer than sparse cached
rescoring; no cold full-run duration is claimed from the subset check.

```bash
.venv-cpu/bin/python code/business_entity_resolution/src/run_final_submission.py reproduce \
  --test-dir /absolute/path/to/provided/test \
  --work-dir /absolute/path/to/new/reproduction \
  --cpu-python /absolute/path/to/.venv-cpu/bin/python \
  --neural-python /absolute/path/to/.venv-neural/bin/python \
  --workers 16 --sqlite-include /opt/homebrew/opt/sqlite/include
```

The command rebuilds full S2/S3 indices and frequencies, full Source1 reverse
index, and the native postings extension; scores the frozen forward pipeline;
pins one global reverse/neural route; reconstructs 33 reverse plus 3 score/rank
features; serializes supplied test text; scores both offline neural branches in
bounded 10,240-pair partitions aligned to both original inference batch sizes; joins 38 features; applies the selected native anchored
adapter; decides once on the complete Source1 claimant graph; and runs both
strict and unmodified organizer validators with target-ID checks.

The same stages can be run separately as `indexes`, `base`, `features`,
`neural-scores`, and `export`, with the same path/environment arguments. For separate hosts, copy `neural/neural_inputs/test/` and `combined_reverse/`
to the MPS work directory, run `neural-scores`, and copy `neural/test/`,
`neural/fine_test/`, and `combined_neural/` back to the original CPU work
directory before `export`. Run the packaged entrypoint separately on each host
with its local executable paths. Keep the CPU work directory and its sealed
absolute-path reuse manifest on the original host; do not relocate it.
Completed outputs are under `<work-dir>/final_output/` and always use a fresh
destination; the packaged `output/` files are never overwritten.

Rebuilt SQLite metadata includes paths and build timings, and a newly compiled
native library can have different bytes. The reproduction freeze therefore
keeps exact selected weights, source, schema, and routing pins, and independently
attests the rebuilt derived indices/native binary. It links to the original
release freeze. It is not a new model selection, training run, or audit.
The final TSV hashes can be compared against the archived submission report.

The operative release freeze retains the audited candidate lineage. A
validator-only correction accepts several validated peer triplets per scored
pair; the scoring functions, models, route policy, and decisions are unchanged.
The original audited exporter and comparison proof are included.

The sealed selected freeze, aggregate final audit report, measured runtime
projection, and final freeze verification are included under
`src/final_pipeline/`. These evidence files contain no audit truth table or raw
business records. The numeric subset has passed under the pinned Python 3.12
and LightGBM 4.3.0 environment; cold full reproduction remains unverified.

## Validation and licenses

```bash
.venv-cpu/bin/python code/business_entity_resolution/src/utils/organizer_validate_submission.py \
  --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv \
  --test-dir /absolute/path/to/provided/test --check-ids
```

The runtime exporter also runs the stricter source validator before marking its
report complete. Both outputs must cover every 1,732,544 official Source1 record,
including empty predictions. Accepted targets must be subsets of scored targets.

The supplied `microsoft/Multilingual-MiniLM-L12-H384` model is MIT-licensed. The
four-layer binary-classification checkpoint contains 117,654,530 parameters.
Both neural branches reuse this model family; the base encoder and fine-tuned
checkpoint are stored separately for exact offline inference. The base model
license and revision provenance are included alongside its files. LightGBM is
MIT-licensed. Generated tree and alias artifacts carry the included
`src/model/MODEL_LICENSE.txt` notice. No external business registry, geocoding,
identity API, external entity augmentation, or hosted model inference is used.
