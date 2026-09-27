# Entity resolution handover — 27 September 2026

Current state: **16:13 UTC / 21:43 IST**. The user reports **Submission 05 portal macro F0.5 = 0.970**, up **0.002** from Submission 04's 0.968. This is the best reported portal result. The result is user-reported, not independently retrieved from the portal; the submission receipt and unrounded score were not provided. The completed release files and ZIP remain unchanged. The user explicitly authorized the four-layer submission below .99; the earlier .99 submission restriction was superseded.

## Release to upload

Use **`Downloads/EntityResolution_submission.zip`**. Its preserved versioned copy is `Downloads/EntityResolution_submission_05.zip`. Both are byte-identical, **1,069,212,134 bytes / 289 members**, SHA-256:

```text
7db0a38c93e1f828ce1033a3619e85803933f7631da058d0d279917e2e42730d
```

Both validated TSVs and the output report are in `Downloads/EntityResolution_submission_05/`; a separate `UPLOAD_README.md` describes the upload. The ZIP contains exactly `output/matching_results.tsv` and `output/candidate_pairs.tsv`, complete inference source under `code/business_entity_resolution/src/`, pinned environments, offline model weights/licenses, a completed `Documentation_template.md`, release report and package verification. No raw dataset or audit truth is packaged. Reproduction rebuilds indices from the provided external test TSVs.

| Completed official-test output | Count |
| --- | ---: |
| Source 1 references | 1,732,544 |
| Scored forward candidates | 69,301,760; exactly 40 per reference |
| Auxiliary neural-route pairs | 801,380 |
| Accepted links | 5,772,254 |
| Empty references | 99,399; 5.73717% |
| Strict and unmodified organizer validators with ID checks | Both PASS |

The candidate TSV is byte-identical to Submission 04. Exact validated SHA-256 values:

```text
matching_results.tsv: 3705f27dbc21a693cde2633c9d00a495c1458d11bac4c92bf3c66ceba837de0e
candidate_pairs.tsv:  cfc74f994f0a00c15cae7002c0fef1882e26bfb91ac349b4b23e3e0c5e74c84f
```

Repository: `https://github.com/Mohith-c7/EntityResolution`, branch **Harsha**. Preserve Submission 03/04/05 and the completed archives. The older root `output/matching_results.tsv` is not the current upload. Do not reset or force-push the shared branch, overwrite release artifacts, or merge an experimental model into the completed release. Private VM access history remains only in the Downloads status document. Copy required temporary VM artifacts before cleanup; a Git clone alone omits the datasets, weights and large caches.

## Scores and frozen evidence

The selected candidate is **`neural_last4_fine_v1`**, the anchored 38-feature adapter with both the original frozen neural head and four-layer continuation.

| Evaluation | Four-layer release | Control | Meaning |
| --- | ---: | ---: | --- |
| Single-use final audit, 10k owners with 331,012-owner claim graph | **.9799622272** | Submission 04 .9793161919 | Consumed final audit; all promotion guards passed |
| Exposed selection20k with complete 30k claim graph | .9802609267 | Submission 04 .9794740787 | Development selection |
| Portal | **.970** | Submission 04 **.968** | Team-reported results; displayed gain +.002 |

The final-audit gain is **+.0006460352**, paired 95% CI **[+.0002035806,+.0010892613]**. Precision increased .9952899973→.9958352196 and recall .9544283274→.9553548163. Singleton false predictions stayed 12/528. US and India improved. This is not a leaderboard score or evidence of French accuracy. No verified .985 or .99 portal result exists; local-to-portal gaps are not a fixed adjustment.

Submission 05's local-audit-to-portal difference is approximately **0.009962**, comparing different populations. It is descriptive, not an estimated calibration correction or a measured France score. The remaining gap from the reported portal result to 0.985 is **0.015**. Portal outcome provenance and release hashes are recorded in `reports/final_2h/submission_05_portal_result.json`.

The final audit was opened once after candidate, complete-claim predictions and runtime evidence were sealed. It is consumed and must not be reused for tuning. The selected release freeze is `reports/final_2h/neural_last4_release_freeze_v3.json`, SHA:

```text
39c00889453f513099ed01746f627cdb43f7a49d300d6f01718b5a12f3532af7
```

The original audited descriptor `reports/final_2h/neural_last4_final_audit_freeze.json` remains unchanged, SHA `eee9067df380a21cf3d87ace97b55aeb2c84ab50045ec00b3fde892d10cf8982`. Release v3 fixes only export validation of 1,473,940 peer triplets versus 801,380 distinct scored pairs. The six scoring functions, global decision call, weights, thresholds and route are unchanged, proven by `reports/final_2h/neural_last4_release_v3_validation_proof.json` and the preserved audited exporter snapshot. All69,301,760 base keys and routed/peer keys were checked.

The measured two-block runtime projection is **8,768.913 seconds / 146.15 minutes**, including 15% allowance, below the three-hour gate. It is a projection, not measured cold package reproduction. Full official-test inference/export and both validators have now completed. Package verification executed the actual numeric 100-owner/4,000-pair subset: zero score error, exact selected masks, 221 routed and3,779 unchanged unrouted pairs. A synthetic cold CPU index/native/forward smoke also passed. Cold full-test reproduction of the archive was not run.

## Current experiments, separate from the release

The **empty-rescue fixed38 pilot was rejected** after development-only evaluation on VM2 with the frozen adapter. Selection20k scored .9805633077, gain +.0003023810 versus selected4, CI [+.000025,+.0006023810], adding 9 true and 2 false links. Precision fell .9966268851→.9965971991 and singleton false predictions rose 34→36, failing both guards. It is not release policy; no model fitting, production or final/extension-audit changes resulted. Aggregate report: `research/final_2h/empty_rescue_fixed38_v1/report.json`.

The **last-eight-layer continuation was rejected**. On exposed selection20k it scored .9804945666 versus .9802609267 for the selected four-layer model, gain +.0002336399 with 95% CI [+.0000459690,+.0004213959]. It failed the predeclared minimum gain +.0003 and nonfalling-precision guards: precision .9966268851→.9964807347; false links 223→233. Singleton false predictions 34→33 and country guards passed. Report: `research/final_2h/neural_last8_fine_v1_selection/report.json`. Its early10k report is an earlier development artifact, not the final selection result.

The separate 10k extension audit in `research/final_2h/extension_audit_last8_v1/` remains reserved and **unopened**. Plan SHA `7e0ae6bd98c6023690935f5317b3ea09550fcb5b117a23058db17b61566e8f2a`. Last8 failure does not justify opening or reusing it. Any future eligible audit requires its own frozen candidate and complete 331,012-owner prediction graph.

## Label-free Submission 04→05 differences

Aggregate report: `research/final_2h/submission05_unlabeled_diff_v1/report.json`. Both matching hashes were verified against their validated reports; all owner rows aligned. Only official Source 1 country attributes were used, with no truth or external entity lookup.

| Country | Added links | Removed links | Changed owners | Newly empty | Newly nonempty | Duplicate accepted targets, 04→05 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| US | 7,801 | 3,564 | 10,812 | 17 | 7 | 56→51 |
| India | 11,021 | 3,969 | 14,200 | 50 | 18 | 176→169 |
| France | 3,638 | 14,834 | 16,939 | 661 | 143 | 966→695 |
| Total | 22,460 | 22,367 | 41,951 | 728 | 168 | 1,198→915 |

Net accepted links increased 93 and empty references increased 560. Shared-target owner sets changed for 1,969 targets, including 1,343 claimant replacements. There were20,920 newly claimed and20,532 newly unclaimed targets. Excess duplicate claims fell 1,227→932; no cross-country duplicate targets were observed. These output properties do not establish accuracy, especially for France. The research-only post-selection exclusivity helper is not release policy.

## Runtime method and essential artifacts

The pipeline uses native posting-list/bridge retrieval, frozen first-stage and context/competition LightGBM, exact reverse-corpus evidence, offline neural probabilities, the anchored adapter and unchanged complete-graph decisions. Model: `microsoft/Multilingual-MiniLM-L12-H384`, MIT, **117,654,530 parameters**. Four-layer continuation uses the sealed 120,000 outer-training pairs; adapter fitting and heldout owner populations are separate.

The 38 inputs are33 reverse features, originalp1, p2 and full-candidate rank, original frozen-head probability and four-layer probability. Apply `sigmoid(logit(original_p2)+correction)` once on the fixed sparse route; preserve every outside-route score and all 40 candidates per owner. Original decisions retain ownership threshold .83, ordinary .8300000000000004 and rank-one rescue .6999999999999998.

| Artifact | Repository-relative location |
| --- | --- |
| Final audit aggregate | `research/final_2h/neural_last4_final_audit_v1/report.json` |
| Runtime gate | `research/final_2h/neural_last4_runtime_final_v2/runtime.json` |
| Release freeze and validator-only proof | `reports/final_2h/neural_last4_release_freeze_v3.json`, `neural_last4_release_v3_validation_proof.json` |
| Adapter | `research/final_2h/neural_last4_fine_v1/{adapter.txt,report.json}` |
| Fine checkpoint and lineage | `models/final_2h/neural_last4_continue_v1/` |
| Original frozen neural head | `models/sprint_6h/neural/frozen_head120k_v1/` |
| Complete heldout claim graph | `models/sprint_6h/heldout_cohort/`; 331,012 owners/13,240,480 pairs |
| Final exporter | `scripts/build_neural_last4_submission.py` |
| Full-test preparation/scoring | `research/final_2h/{prepare_full_test_base,parallel_test_reverse,prepare_test_neural_inputs,score_full_test_neural}.py` |
| Package assembler and self-contained entrypoint source | `scripts/{package_final_submission,run_packaged_submission}.py` |
| Package handoff | `research/final_2h/package_checks/release_v3_handoff.json` |
| Numeric fixture | `research/final_2h/package_fixture100_v1/`; numeric evidence only |

CPU runtime is Python3.12/LightGBM4.3.0; measured Mac neural runtime is Python3.13.12/PyTorch2.14.0/Transformers4.57.6 on MPS. Package requirements pin both environments. All inference loads are offline. Exact source/model bytes, feature names/order, head lineage, route and input/output hashes are sealed.

## Historical work and lessons

The earlier13:55/ 15:11 UTC handovers described an incomplete audit/run/archive and the then-active .99 release restriction. Those states are superseded by the completed release and explicit authorization above.

- Submission 03 portal .9399 and Submission 04 portal .968 are historical team reports. Submission 04's older consumed audit .978547 and neural_v2 fresh confirmation .9794958377 use different cohorts; do not tune on them or subtract a universal portal gap.
- Neural_v2 selection .9800089319; three-epoch fine model .9801408948. Direct calibration's .9802697541 point estimate failed the precision guard. These were development comparisons, not newer releases.
- Broad77-feature, hard-negative and direct variants did not pass acceptable-gain guards. Missing-address retrieval quota failed downstream precision/singleton guards despite oracle recall; clean NNv2 delta approximately−.000286. Do not claim oracle gains as matcher gains.
- Dedicated sibling/peer neural additions did not beat the simpler eligible winner. Preselection ownership can suppress valid accepted owners; the separate pure post-selection helper adds no links but remains experimental. Correction strength1 stayed selected; no policy adjustment resulted.
- Token-alignment fix `cursor/token-alignment-fix`, commit `ee1e0b5`, was reviewed with 13 tests; corresponding frequency statistics must be rebuilt before using its new schema. It is not permission to mutate frozen normalization or indices.

Challenge constraints remain: only supplied entity evidence for fitting, no external business identities/registries/geocoding/augmentation or hosted inference, permitted model licenses/size, every official Source 1 reference including France and empty predictions emitted. Unlabeled statistics are diagnostics. The final audit is consumed; the extension audit is unopened. Preserve Submission 05 as the 0.970 portal baseline; any further candidate needs separate development evidence, a frozen evaluation and new versioned outputs. Do not alter the validated final ZIP merely to update an external score log.
