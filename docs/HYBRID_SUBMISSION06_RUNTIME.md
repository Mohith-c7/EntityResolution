# Submission06 hybrid TSV runtime

This candidate is `hybrid8_empty4_calibrated_exclusive_v1`. It uses native NN8
on the original 801,380 neural-route pairs, native NN4 on disjoint extra pairs,
and ORIGINAL frozen p2 as the anchor once for each correction. The candidate
calls the original decision source with threshold 0.83, t_first
0.6999999999999998, and t_rest 0.79, then calls the existing
`post_selection_exclusivity.exclusive_selected` helper. Submission05 remains
unchanged and is the audit comparator.

`prepare_hybrid_test_rescue.py` verifies submission05's matching/report/freeze
hashes and full original 69,301,760-pair graph. Its gate is the submission05
GLOBAL empty mask (99,399 owners). It chooses original-p2 top4, with target-ID
ties, then removes original neural-route keys without replacing any overlap
with rank5. Actual supplied TEST records and the complete TEST reverse index
produce reverse33 plus original p1, p2 and full40 candidate rank. The new typed
feature/input manifests explicitly describe this direct route; they do not
claim the old sibling route. The resulting 393,220-pair raw input is independent
of labels and serialized concurrently with the 32 CPU feature workers.

Run preparation once into an unused output:

```sh
/mnt/er/venv/bin/python research/final_2h/prepare_hybrid_test_rescue.py --workers 32
```

Use the existing frozen neural scorers on
`models/final_2h/hybrid_test_rescue_v1/neural_inputs`, preserving 512/32 batch
boundaries. The old/fine4 outputs are separate from original-route NN8 scores.
No head is fitted here. The raw manifest is immutable and binds route keys,
raw source hashes, serialization source, role, split and input bytes.

The NEW exporter is `scripts/build_hybrid_submission.py`. It accepts:

```sh
/mnt/er/venv/bin/python scripts/build_hybrid_submission.py \
  --freeze NEW_RELEASE_FREEZE.json \
  --audit-evidence NEW_FRESH_AUDIT_REPORT.json \
  --reuse-manifest VERIFIED_ORIGINAL04_REUSE_MANIFEST.json \
  --features models/final_2h/test_combined_neural05 \
  --fine8-scores models/sprint_6h/neural/last8_full_test06 \
  --fine8-input-manifest ORIGINAL05_NEURAL_INPUT_MANIFEST.json \
  --extra-features models/final_2h/hybrid_test_rescue_v1 \
  --extra-old-scores EXTRA_OLD_SCORE_DIRECTORY \
  --extra-fine4-scores EXTRA_FINE4_SCORE_DIRECTORY \
  --model8 research/final_2h/neural_last8_fine_v1 \
  --model4 research/final_2h/neural_last4_fine_v1 \
  --head8-manifest models/final_2h/neural_last8_continue_v1/manifest.json \
  --head4-manifest models/final_2h/neural_last4_continue_v1/manifest.json \
  --old-head-manifest models/sprint_6h/neural/frozen_head120k_v1/manifest.json \
  --blue output/submission_05 --test-dir dataset/test \
  --output output/submission_06 --threads 16
```

The candidate/release freeze must pin this exporter, preparation builder and
imported unchanged scoring/decision dependencies in `code_sha256`; native
models, reports and all three head manifests in `model_sha256`; the new audit
report in `schema_sha256`; and complete immutable model/index/native evidence
in the remaining verified groups. Its `parent_frozen_sha256` is original04
68d6148b..., whereas `baseline_frozen_sha256` is submission05 release39c00889....
`baseline_decision_config` retains original05 settings. Set
`candidate_post_selection_policy` to
`selected_only_probability_then_source1_id`, `reservation_plan_sha256` to the
fresh extension plan hash, and `extension_acceptance_criteria` to:

```json
{"minimum_point_macro_gain":0.0003,"paired_ci_lower_bound_strictly_above":0.0,"country_macro_delta_floor":-0.0005,"precision_singletons_diagnostic_only":true}
```

The audit report must have status `paired_extension_audit_complete`, matching
candidate/baseline/reservation hashes, and passed `hybrid_prospective_acceptance`
with those exact criteria. The runtime must pass and bind the exact audit
freeze. The complete heldout claimant graph must contain 331,012 references and
13,240,480 pairs. A release amendment may use explicit `audit_freeze_sha256` to
retain the original audit binding. Old NN8 and empty-rescue component rejections
are preserved; their legacy acceptance results are not promotion evidence.

The exporter replays the entire direct top4 route from original anchors and
verified global blue emptiness; joins neural outputs by exact key; verifies
native38 order, anchors, full40 ranks and disjoint routes; scatters the union
once; and decides globally on all 69,301,760 candidates. It writes matching TSV
and retains the verified candidate TSV byte-for-byte for both official/strict
validators. The user deliverable is only `matching_results.tsv`; no ZIP is
created. A new output directory is mandatory. No label file is read by this
exporter.

Verification: eight scoped tests cover overlap-after-top4/no refill,
deterministic ties/full40 context, original-anchor single scatter, outside-route
invariance, duplicate/changed anchors/ranks, exact keyed joins, route replay and
wrong-freeze/legacy-audit gate rejection. An actual eight-layer score-manifest
regression binds generic `neural_layers.py` inference (training used
`neural_layers8.py`). Independent completed full-test validation passed all
393,220 diagnostics, input/feature order, disjoint routes and 32 part seals;
its aggregate evidence is `reports/final_2h/hybrid_test_rescue_validation.json`.
Full production export must wait for
the fresh audit gate.
