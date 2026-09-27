# Fixed5k retrieval recovery combination

Preparation completed in 107.56 seconds on VM1 with eight feature workers. The
experiment reuses the sealed, already exposed 5,000-query upstream-quota cache;
it does not retrieve another sample, fit another model, or inspect fresh/audit
labels. The prepared artifacts are ready for the root's new 60-feature adapter.

| Check | Result |
|---|---:|
| Original frozen-model-scored pairs preserved | 200,000 / 200,000 |
| Extended frozen-model-scored pairs | 217,048 |
| Additions | 17,048, maximum four/query |
| Routed baseline / extended pairs | 198,604 / 215,652 |
| Additions routed to the new adapter | 17,048 / 17,048 |
| Supplied raw queries matching Source1 index normalization | 5,000 / 5,000 |
| Distinct supplied raw added targets matching cached normalization | 13,019 / 13,019 |
| Baseline first-stage, second-stage, final score parity | Bit-for-bit exact |

`added_raw_text.parquet` contains every addition's supplied Source1 and target
name/address, normalized feature text, retrieval path, target source, and rank.
`manifest.json` seals all six VM1 pair/text/feature files with SHA256 hashes.
Full artifacts are on VM1 at `/mnt/er/final_retrieval/broad60/`:
`baseline_pairs.parquet`, `upstreamquota_pairs.parquet`,
`baseline_features.parquet`, `upstreamquota_features.parquet`,
`added_pairs.parquet`, and `added_raw_text.parquet`.

The previous measured candidate oracle delta is +0.001195760, recovering 61
true links with no original pair removed. It is potential coverage, **not a new
matching gain**. Previous unchanged matcher decisions improved only +0.000143303
while precision fell and singleton errors rose, so retrieval alone was rejected.
Its measured additive retrieval worker-wall time was 521.094 seconds, or
104.219 ms/query. This preparation reused that retrieval and did not measure a
new retrieval speedup.

The subsequent equivalent retrieval benchmark **did** reduce extension work.
It preloads the read-only empty-address records and uses an immutable rowid bitmap
to filter the same capped posting union before the unchanged name-only ranker.
All 17,048 added pairs match the sealed variant exactly, including candidate
assembly. Measured extension worker-wall time is 103.582 seconds, or 20.716
ms/query; the eight-worker run takes 26.605 wall seconds including worker model
and metadata startup. See `bitmap_benchmark.json`. This is approximately one
fifth of the previous extension worker work. The baseline40 lists are cached in
this benchmark, and it does not measure matching, fulltest throughput, or ETA.

`retrieval_recovery.py score` compares broad-adapter40 against
broad-adapter44, using each list's own rebuilt frozen scores, ranks, and seeds.
Every final pair has frozen first/second-stage scores; every routed pair receives
the new adapter. Claims from the other45k remain frozen. These are exposed5k
combination diagnostics and cannot establish a fresh or full-Source1 promotion
result. New scoring commands require a sealed clean evaluation-owner subset.
The earlier broad_v1 report below used all5k and may overlap fitted owners; it
is retained only as the historical diagnostic and is not clean quality evidence.

The default weight1 `broad_v1` combination run completed successfully on VM1 at
`/mnt/er/final_retrieval/broad_v1_scored/` and was read through Azure's control
plane after intermittent SSH timeouts. **Reject the combination:** macro F0.5
changes from 0.9798674864 to 0.9799887117 (+0.0001212252, paired95%CI
[-0.0004597689, 0.0006762427]), adding15 true and2 false links. Precision falls
from 0.99655193 to 0.99643461; singleton false positives rise from5 to6.
Prediction takes0.475/0.521 seconds. See `broad_v1_combination_report.json`.
The requested 77-feature alignment variant is prepared on VM1 at
`/mnt/er/final_retrieval/alignment77/` in 15.688 seconds, using the root's sealed
full Source1 v2 statistics (`ed071284...`) and exact `p2>=.003 or rank_p1<=4` route.
Alignment covers 26,253 baseline and 27,863 quota pairs; the remaining rows have
zero alignment features and marker 0. The original 60 feature values and pair keys
are bit-for-bit unchanged across both complete tables. A direct read-only index
recomputation of 1,024 aligned pairs matches root semantics exactly. See
`alignment77_manifest.json` and `alignment77_parity.json`. These 77-feature tables
are ready for the root's adapter; no 77-model gain is claimed before scoring.
The approved alignment module is loaded from an isolated snapshot because VM1's
release lacks that module, preserving shared core files. The scoring command
accepts either sealed 60 or 77 features.

Five fixture tests passed with `/tmp/entity-review-venv/bin/python -m pytest
tests/test_final_retrieval_recovery.py -q`, covering preservation, duplicate
rejection, the four-addition cap, sealed baseline identity, and seed-free routing.

The standard neural_v2 preparation is on VM1 at
`/mnt/er/final_retrieval/neural_v2_prepare/`. It preserves all200,000 original
pairs in the217,048-pair quota graph. Independent complete5k routing yields
1,864 baseline and1,923 quota rows, with exact first-stage/p2/rank anchors and
TRAIN Source1 reverse-index provenance. Standard36 reverse inputs and supplied
raw keyed neural JSONL took48.616 seconds to prepare. The two raw inputs and
manifests were copied to Mac `models/final_2h/retrieval_neural_inputs/`, scored by
the unchanged frozen120k head, returned toVM1, and validated with the standard
37-input join and native neural_v2 adapter rescoring. No model was fitted here.
Independent routing is a fair paired5k pipeline comparison, but it is neither
an addition-only ablation nor parity with the historical global50k neural route.

**Reject the native neural_v2 quota combination.** The clean2,952-owner subset
excludes all2,048 neural_v2-fitted owners present in the exposed5k and all frozen
head fitted owners (zero overlaps). Decisions still retain complete5k plus
the preserved45k frozen rival claims. Macro F0.5 is .9821388498 baseline versus
.9818532780 quota, delta -.0002855718, paired95%CI[-.0011417823,+.0002609256].
The quota adds10 true and3 false links, removing none. Precision falls
.99635554 to .99605702 and singleton false predictions rise4 to5. See
`neural_v2_clean_evaluation.json` and the clean subset seal under
`neural_v2_prepare/clean_evaluation_subset.json`. No fresh/audit truth was read.
