# Entity resolution handover — 27 September 2026

Current release update: 15:11 UTC / 20:41 IST. The detailed historical sections below retain the earlier 13:55 UTC handover. Access details are in the private Downloads copy.

## Start here

The selected candidate is **`neural_last4_fine_v1`**. Its fresh, single-use 10,000-reference audit scored **0.9799622272**, against **0.9793161919** for submission_04 on the same references. Gain **+0.0006460352**, paired 95% CI **[+0.0002035806, +0.0010892613]**; precision and recall improved, singleton errors stayed at 12, and both countries improved. Promotion passed. The audit is consumed. Development selection remains **0.9802609267**; submission_04's user-reported portal score remains **0.968**. No 0.985 result is verified.

Full submission preparation is running: the complete 1,732,544-reference / 69,301,760-candidate base graph, all exact reverse features and official test text are finished. Both audited neural heads are scoring the pinned 801,380 pairs on Mac MPS, started 15:05 UTC. The automatic finalizer waits for their completed integrity-checked manifests, then joins and exports on VM1, runs strict and official ID validators, copies verified TSVs to Downloads/EntityResolution_submission_05, and builds Downloads/EntityResolution_submission_05.zip. Exactly two Sol agents handle separate eight-layer calibration on VM2 and offline package checks. The safe four-layer submission will not wait for that experiment.

The 38-feature exporter is implemented and subset parity is tested. Two real country-stratified runtime blocks passed strict and organizer validators; conservative projection is 146 minutes with 15% margin. A validator-only fix distinguishes sibling edges from distinct candidate pairs in release freeze v3 SHA 39c00889453f513099ed01746f627cdb43f7a49d300d6f01718b5a12f3532af7; the original audited freeze is preserved. Final TSVs and ZIP are still pending full inference. The correction-strength grid chose the existing strength 1; post-selection exclusivity changed no links on the development graph.

Repository: `https://github.com/Mohith-c7/EntityResolution`, branch **Harsha**. Do not reset, force-push, switch the shared worktree, overwrite existing outputs, or merge unvalidated experimental models. Models, datasets and large caches are not in Git; a clone alone cannot reproduce the current state.

## Scores and what they mean

| Artifact | Evaluation | Macro F0.5 | Status |
| --- | --- | ---: | --- |
| Submission 03 | Portal | 0.9399 | Historical submission |
| Submission 04 | Portal | 0.968 | Current submitted baseline; team-reported |
| Submission 04 | Consumed earlier audit | 0.978547 | Historical audit; do not tune on it |
| `neural_v2` | Fresh 10k confirmation | 0.9794958377 | Best completed fresh result |
| `neural_v2` | Existing selection20k | 0.9800089319 | Development comparison baseline |
| `neural_fine_v1` | Existing selection20k | 0.9801408948 | Three-epoch head; gain over neural_v2 not significant |
| `neural_direct_v1` | Existing selection20k | 0.9802697541 | Slightly higher point score, but fails precision guard |
| **`neural_last4_fine_v1`** | **Existing selection20k** | **0.9802609267** | **Best development candidate passing quality guards; final audit pending** |
| `neural_last4_direct_v1` | Existing early10k | 0.9799338986 | Selection20k not evaluated |

For `neural_last4_fine_v1` versus neural_v2 on the same selection20k:

- Gain: **+0.0002519948**, paired 95% bootstrap CI **[+0.0000440171, +0.0004706429]**.
- Micro precision: **0.9966268851**; micro recall: **0.9530063497**.
- 156 true links added, 91 removed; 15 false links added, 19 removed.
- Singleton false predictions: **34 / 1,130 singleton references**; no increase.
- India macro F0.5: **0.9750013858**; US: **0.9837388904**.
- Development guards pass, but `promotion_pass=false` correctly remains because this is exposed development data and release validation is incomplete.

For neural_v2 fresh confirmation, gain over the incumbent is +0.0005644251, CI [+0.00014095, +0.00102835]. Precision and recall both improved; singleton false predictions fell from 16 to 15. That confirmation set is consumed.

No experiment has demonstrated 0.99 locally or on the portal. France has no supplied labels. Its score cannot be measured from links per business or inferred exactly from the public portal aggregate.

## Portal gap: first priority for the next operator

The team's latest steering is to close the validation-to-portal gap before spending more time on tiny local gains. Submission 03's approximate gap was 0.9594 - 0.9399 = **0.0195**; Submission 04's consumed audit-to-portal gap is 0.978547 - 0.968 = **0.010547**. These use different cohorts and are descriptive comparisons, not a stable correction factor.

Closing the current gap alone would place the present model around 0.979–0.980, rather than 0.985. Reaching portal 0.985 also needs roughly another 0.005–0.006 of genuine quality gain. The last team-reported top100 cutoff was 0.9858 and may move. The strategic target remains 0.99; no new full run is authorized by this handover.

Priority checks: reproduce exact model/index/normalization/feature/decision parity; reproduce output metric semantics including all singletons; measure country-composition effects; audit train/test differences in name collisions, street evidence, missing fields and scripts; then test country-generic fixes on owner-disjoint labeled transfer/collision slices. France is a plausible major source of loss, not a measured country score. Do not subtract 0.01 from every future local result or claim that lower collision counts establish French accuracy.

## Runtime pipeline and new model

The existing pipeline uses posting-list retrieval with a native extension, bridge retrieval, pairwise LightGBM, a context/competition stage, and ownership plus rank-one rescue decisions. Existing thresholds are `t_rest=0.8300000000000004`, `t_first=0.6999999999999998`, ownership threshold `0.83`.

The neural route uses at most 20% of references and four uncertain candidates per routed reference, with predicted high-confidence siblings required by the fixed routing contract. Auxiliary neural inference does not change the official candidate set. The full test has **1,732,544 references / 69,301,760 candidate pairs**; the measured frozen route contains **801,380 pairs**.

The new candidate uses the old 37 features, plus **`neural_fine_probability`**, in exactly that order. It retains both the original frozen neural head and the new fine-tuned head.

- Base checkpoint: `microsoft/Multilingual-MiniLM-L12-H384`, pinned offline revision recorded in `origin.json`, MIT; **117,654,530 parameters**.
- Three-epoch parent: 120,000 supplied outer-training pairs, batch 32, maximum combined sequence length 128, PyTorch/Transformers on Mac MPS; last two encoder layers plus pooler/head trainable; 360,000 presentations in 22m47s.
- Continuation: one additional epoch over the same sealed 120k pairs; last **four** layers plus pooler/head trainable, **7,246,466 trainable parameters**; 3,750 updates in **480.402s**, mean training loss 0.1051879.
- Learning rates: encoder/pooler 1e-5, classifier 1e-4, AdamW, warmup and linear decay. Full details and file hashes are in the model manifest and training plan.
- Calibration adapter: LightGBM, 38 features, 31 leaves, minimum leaf 40, L2=1, learning rate .05, max_bin=127, seed 42, 8 threads, 800-tree cap with early stopping 40; selected **103 trees**.
- It is an **anchored residual** adapter: `sigmoid(logit(original_p2) + raw_adapter_correction)`. Apply the p2 anchor exactly once. This model does not use the direct classifier's calibration shift.
- All outside-route p2 scores must remain unchanged. Candidate IDs, candidate order and original p2 must match the sealed full graph exactly.

## Essential artifacts

Paths below are relative to the repository root on the machine that holds them.

| Artifact | Location | Availability |
| --- | --- | --- |
| New adapter and report | `research/final_2h/neural_last4_fine_v1/{adapter.txt,report.json}` | VM2 and Mac |
| New selection result | `research/final_2h/neural_last4_fine_v1_selection/report.json` | VM2 and Mac |
| New full checkpoint | `models/final_2h/neural_last4_continue_v1/{checkpoint,manifest.json,training_plan.json,direct_scores_verification.json}` | Mac |
| Three-epoch initializer | `models/final_2h/neural_last2_3epoch_v1/` | Mac |
| New development scores | `models/sprint_6h/neural/{last4_residual20k,last4_learned30k}/` | Mac and VM2 |
| Old neural head | `models/sprint_6h/neural/frozen_head120k_v1/` | Mac; preserve for inference |
| Original residual37 features | `research/sprint_6h/sibling/neural_v1_features_fit/` | VM2 |
| Original development37 features | `research/sprint_6h/sibling/neural_v1_features30k/` | VM2 |
| Complete development claims | `research/sprint_6h/sibling/learned30k/pairs.parquet` | VM2; 1.2M pairs / 30k references |
| Existing development labels | `research/sprint_6h/sibling/workbenches50k/{residual_train,early_stop,selection}/` | VM2; 20k / 10k / 20k references |
| Complete heldout graph | `models/sprint_6h/heldout_cohort/{pairs.parquet,manifest.json}` | VM1 and VM2; 331,012 references / 13,240,480 pairs |
| Old37 heldout features | `research/sprint_6h/sibling/neural_v2_features_heldout331k/` | VM1; 123,600 routed pairs |
| Raw heldout neural input | `models/sprint_6h/neural/neural_inputs/heldout331k_v2/{pairs.jsonl,manifest.json}` | Mac; sealed, label-free |
| Reserved final audit | `research/sprint_6h/validation/fresh_splits/audit/references.json` | 10k references; labels unopened |
| Split ledger | `research/sprint_6h/validation/fresh_splits/plan.json` | VM1 and Mac |
| Whole-graph evidence | `research/sprint_6h/validation/heldout_scoring_evidence.json` | VM1 and Mac |
| Frozen incumbent | `research/sprint_6h/coordination/baseline_freeze/frozen.json` | Both VMs and Mac |
| Fresh confirmation result | `research/sprint_6h/validation/neural_v2_fresh_confirmation/evaluation/report.json` | VM1 and Mac |
| NNv2 actual runtime report | `research/final_2h/neural_runtime_verified.json` | Mac; binds NNv2 only |
| Actual runtime blocks | `research/sprint_6h/release/neural_v2_actual_blocks/` | VM1; two 10k blocks with France included |
| Raw runtime neural inputs | `models/sprint_6h/neural/neural_inputs/runtime_block_{0,1}/` | Mac |
| Submission 04 | `output/submission_04/` | Mac; keep intact |

Critical hashes:

```text
new adapter:          625268050ec1774ec2eb1c0c8d38b89bf13c284d3e7474b564bb8264eec0dac1
new head manifest:    60bca75bf41628f43eff487f8bdbb9b2f2f5ad54967af467807d7bc55417083c
new model weights:    7583c37180bf5bd9cdf1bf997769e36adc05947827f8bead7a325d06b49179e7
parent head manifest: 7342207a6e501d998a7a75fcca58155126aa68085d57119a3e597086c53a7a20
continuation source:  34ecb591d1bdc0035b927ab65cc6ddcb9326ff9651d57046ea7ee98f3106a373
```

The complete new model weights have not been transferred to either VM. Neural inference was run on the Mac. Small adapters and reports were copied back from VM2, but large claim graphs remain on the VMs.

## Remaining work, in execution order

1. Inventory artifact hashes and confirm the final audit lock is absent on VM1. Commit and freeze one selected candidate before reading any audit truth. Do not select a different model after the audit is read.
2. Run the new head over the sealed 123,600 heldout input pairs. Use `research/final_2h/neural_layers.py score`; a complete command is in the private handover. It reads no labels and is not a submission run.
3. Implement a new 38-feature scorer. Reuse `neural_adapter.load_features()` and `neural_peer.add_fine()` contracts. Check exact keys/order, neural score and input hashes, head lineage, native model feature names, finite probabilities, and exclusion of every adapter/head training owner from the heldout universe.
4. Scatter corrected scores onto all 13.24M heldout claims, leaving every other score untouched. Replay decisions on **all 331,012 owners**, then slice the reserved audit10k. Scoring only audit references misses competing owners and is invalid.
5. Measure the actual new head, feature join, adapter, decision, export and both validators on the two existing country-stratified 10k test blocks. Freeze the new candidate first and bind runtime evidence to that exact freeze. The old NNv2 timing alone cannot certify the new model.
6. Seal audit prediction evidence with candidate/baseline prediction hashes, reference and universe hashes, `candidate_frozen_sha256`, `baseline_frozen_sha256`, and canonical `decision_config_sha256`.
7. Verify every gate, then export only reserved-owner truth and invoke `scripts/evaluate_sprint.py --audit` **once**. Existing confirmation-style audit evidence lacks final freeze/config bindings and needs a new wrapper. A generic audit-preflight helper has **not** been written.
8. Report the resulting fresh score honestly. The standing instruction is **no new full submission until 0.99**. A developer must not silently replace that instruction with permission to submit a lower-scoring model.
9. If a full run is authorized later: create a new versioned output directory; preserve every S1 including France and empty singleton predictions; candidates must be exactly those scored; matches must be a subset; run strict and organizer validation with ID checking; save all hashes/configuration and copy artifacts before VM cleanup.

The NNv2 measured projection is **8,139.5 seconds / 135.7 minutes**, including 15% margin and validator costs, against a three-hour budget. This is a projection, not a completed new run. Adding the fine head requires a new measurement. Prefer reusing verified Submission 04 candidate/probability chunks rather than rebuilding identical retrieval, after proving package compatibility and parity.

## What remains toward 0.99

The candidate improvement is real on development but small. Going from fresh 0.9795 to 0.99 requires removing about **half of the remaining macro loss**; going from portal 0.968 to 0.99 requires removing about **69%**. More VM RAM or tree count alone has not supplied that evidence.

After completing the current candidate's honest audit, prioritize:

1. **Recompute the current error ledger.** Separate unretrieved true links, retrieved-but-rejected links, empty non-singletons, singleton false links and ownership errors under the final rules. Old loss numbers refer to earlier models.
2. **France and repeated-name precision.** Examine same-name/different-street collisions using country-generic street and ambiguity signals. Validate on labeled US/India collision and country-transfer slices. France monitors are diagnostic only; do not tune toward a desired link-count distribution. Public portal sampling prevents an exact France-score calculation from country averages.
3. **Stronger routed evidence.** The four-layer head is the best current lead. Train against real hard negatives and missed true variants, using owner-disjoint folds and out-of-fold sibling seeds. Consider widening its route only after measuring the achievable correction ceiling and inference cost.
4. **Retrieval only where missing links justify it.** Bridge retrieval historically raised the development oracle to about 0.9935. Add paths for missing-address and cross-script variants only if the complete matcher improves and candidate budget/runtime remain acceptable.
5. **Collective decisions.** Post-selection exclusivity is implemented and tested; separately measure it before integrating. Sibling-model additions in this sprint did not beat the simpler winner, so do not assume collective complexity is automatically beneficial.

Run independent labeled development experiments on VM2 while VM1 preserves caches and performs release validation. Keep at most two or three parallel workers on nonoverlapping code/artifacts. Keep the final audit sealed until all model/threshold choices are settled. A credible 0.99 portal result cannot be promised from the current evidence or a two-hour deadline.

## Completed experiments to avoid repeating

- Broad 77-feature anchored/direct/hard-negative/macro-objective variants: no acceptable gain; calibrated direct variant worsened score by about 0.00223.
- Missing-address candidate quota: modest retrieval ceiling improvement, but downstream score/precision guards failed; NNv2 clean evaluation delta about -0.000286. Do not promote it from oracle recall alone.
- Dedicated sibling neural head: peer40 and combined41 variants did not beat the simpler candidate with quality guards intact.
- Cursor ownership policies: zero changes/gain on the existing complete 30k graph. One variant can let an unaccepted rival suppress a valid owner; do not merge blindly.
- Token alignment corrected branch `cursor/token-alignment-fix`, commit `ee1e0b5`: reviewed, 13 tests passed. Existing v1 frequency statistics must be rebuilt for v2; keep normalization/index/model compatibility explicit.

## Ownership correction

Submission 04 has **1,198 multiply owned accepted targets / 2,425 claims / 1,227 excess claims**. Of those targets, 966 are France, 176 India, 56 US. This is a measured output property, not a measured score gain.

`research/final_2h/post_selection_exclusivity.py` deduplicates **only already selected claims**, retains highest probability with deterministic lexical S1 tie-breaking, adds no links and preserves all unrelated choices. Label-free replay removed the 1,227 excess claims and produced zero duplicate owners. It is research only and is not part of the audited NNv2 or new 38 decision configuration.

## Code, tests and preservation

Main entrypoints:

- `research/final_2h/neural_layers.py`: sealed continuation and offline scoring.
- `research/final_2h/neural_peer.py`: 38-feature keyed assembly and anchored adapter fitting.
- `research/final_2h/train_neural_direct.py`: separate direct classifier experiment; do not confuse its shift with anchored inference.
- `research/final_2h/compare_saved.py`: existing selection20k comparison, never a fresh audit.
- `research/final_2h/benchmark_neural.py`: measured **old NNv2** benchmark only.
- `research/sprint_6h/run_neural_confirmation.py`: existing confirmation workflow; needs a separate final-audit wrapper for new 38.
- `scripts/evaluate_sprint.py`: final freeze, reservation, prediction, runtime and audit gates.

Latest targeted verification: **16 tests passed** across neural assembly, epoch scheduling, macro metric, retrieval recovery and post-selection exclusivity.

```bash
python -m pytest tests/test_final_post_selection_exclusivity.py \
  tests/test_final_retrieval_recovery.py tests/test_neural_peer.py \
  tests/test_neural_epochs.py tests/test_early_macro_metric.py -q
```

Keep Submission 03 and 04 intact. `output/matching_results.tsv` still refers to an older root output; use the versioned directory when identifying an upload. Model weights, raw data and private connection details must remain outside public Git. Before deallocating or deleting VM1, copy needed `/mnt/er` artifacts: its temporary disk is not durable.

Challenge constraints: no external business identities, registries, geocoding or augmentation; no hosted LLM oracle; use only supplied challenge data for fitting; every learned model must independently satisfy the permitted MIT/Apache 2.0 license and <=8B-parameter rule. Keep country labels open and emit every official test S1 record, including France. Unlabeled statistics are not ground truth. French pseudo-label training was not implemented or cleared by this handover.
