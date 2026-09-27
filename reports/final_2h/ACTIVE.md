# Current release status

Updated 27 September 2026, 17:21 UTC /22:51 IST.

Submission 05 is complete and preserved, with user-reported portal macro F0.5 **0.970** versus Submission 04 **0.968**. The user now requests only the new upload TSV; no new ZIP is being built. Submission06 export is **paused** at the user’s request while model accuracy research continues. Its portal result is unknown. The supplied qualification screenshot shows **0.989279**; no result at that level has been demonstrated.

The locked candidate **`hybrid8_empty4_calibrated_exclusive_v1` passed its single-use fresh extension audit** after root explicitly authorized opening at17:06 UTC. All331,012 heldout owners and13,240,480 candidates participated in decisions before slicing the reserved10,000 owners. The extension audit is now consumed and must not select or tune another candidate.

| Same fresh10k cohort | Submission05 comparator | Hybrid candidate |
| --- | ---: | ---: |
| Macro F0.5 | .9803742791 | **.9811416633** |
| Precision | .9956300509 | .9953443954 |
| Recall | .9552856035 | .9578128141 |
| Singleton false predictions | 16 | 17 |

Macro gain **+.0007673842**, paired95% CI **[+.0002402509,+.0013411254]**. India gained+.0009361420 andUS+.0006567417. The exact prospectively declared rule required gain≥.0003, confidence lower bound>0 and country delta≥−.0005; all passed. Precision fell and singleton false predictions rose, explicitly reported as diagnostics under this new rule. Historical standalone last8 and empty-rescue rejections under their original guards remain unchanged. The new extension provides no French accuracy measurement.

Current processes: root requested cancellation of the capacity-only12-layer job andpreparation of an8-layer continuation with balanced target-address dropout on the same120k training keys/labels/owners. No new accuracy gain is claimed. The existing Submission06 exporter onVM1, PID139161, is SIGSTOP-paused with state preserved (read-only process check `Tsl`). Validation anddownload remain pending; no new submission run has been started. All801,380 original-route NN8 scores and393,220 disjoint extra old/fixed4 scores are complete and verified. Submission05's outputs, models and archive remain immutable.

The candidate applies NN8 on the original neural route and fixed4 on disjoint original-p2 top4 candidates of globally empty Submission05 owners. Exclusion is after top4, without fifth-candidate refill. Both native corrections anchor once to ORIGINALp2. Candidate thresholds are ownership.83, ordinary.79, first-candidate.6999999999999998, followed by highest-probability **actually selected** target ownership with Source1ID tie-break. Blue05 retains its original rules. All40 candidates per owner are preserved.

The cache-reuse R2 runtime evidence uses actual full new-stage timings and two actual US/India-stratified10k decision/export/ID-validator blocks. Both validators passed. Remaining processing was conservatively estimated **1,372.68seconds /22.88minutes including15% margin**, versus3,939.68seconds available to18:10UTC at measurement. Total incremental time including completed CPU/MPS stages is3,583.03seconds. This is not a cold reproduction benchmark; full-test neural timing includesFrance, without an accuracy claim.

| Sealed artifact | Path / SHA prefix |
| --- | --- |
| Audited freeze | `reports/final_2h/hybrid_extension_audit_freeze_v2.json`; `ab2506d2…` |
| Operative release, metadata amendment only | `reports/final_2h/hybrid_release_freeze_v1.json`; `31bc9745…` |
| Fresh aggregate report | `research/final_2h/hybrid_audit/extension_evaluation_v1/report.json`; `60cdfda2…` |
| Complete-graph prediction evidence | `research/final_2h/hybrid_audit/heldout_predictions_v2/manifest.json`; `92df351b…` |
| Runtime | `research/final_2h/hybrid_audit/runtime_v1/runtime.json`; `80d9fc25…` |
| Root release review | `reports/final_2h/hybrid_release_root_review_v1.json` |

Consumed-audit diagnostics identify906 retrieved true links rejected (682 alreadyNN8-routed),563 absent fromoriginal40 and156 accepted false links. Target addresses are missing in558/906 missed retrieved pairs. Fine8≥.95 unselected/anchored-p<.79 contains494 true and1271 false pairs, so confidence alone does not justify anoverride. Training coverage is thin:1969/67010 positive pairs and122/52990 negative pairs have target-address missing, while Source1 addresses are always present. These are retrospective diagnostics, not fresh validation or a deployed rule. Aggregate reports are `hybrid_consumed_error_summary_v1.json`, `hybrid_consumed_pair_traits_v1.json`, `hybrid_consumed_neural_confidence_v1.json` and `neural_training_address_coverage_v1.json`.

All97 source pins were reviewed by root; source/model/config/prediction lineage is unchanged in the release amendment. Production exporter `verify_freeze` and `verify_audit` passed before export. Eleven audit-workflow guards and root's36 scoped checks passed; an independent256-case metric check matches the official formula to floating-point precision. These are verification evidence, not accuracy gains.

## Historical starting snapshot

The statements below describe the earlier research window, not current process or audit status.

# Final improvement window

Started after the user reopened model research at 12:22 UTC on 27 September 2026. Working cutoff: 14:22 UTC, followed by the user's separate submission-generation window.

## Verified starting point

- Portal submission_04: 0.968, reported by the user.
- Neural v2: selection20k macro F0.5 0.980008932.
- Fresh confirmation10k: 0.979495838 versus incumbent 0.978931413. Both precision and recall improve. This cohort is consumed and must not select later experiments.
- Separate reserved final audit10k remains unopened.

## Parallel work

1. Parent: seed-independent broad correction, fit on the existing residual20k and early-stop on the existing early10k. It retains the original candidate universe. Original and augmented street/number/name signals are tested on already exposed development data.
2. Cursor coordination: review ownership/retrieval code and add corrected corpus-weighted token alignment. Ownership changes produced zero changes on both complete30k development graphs. Full training Source1 v2 statistics and thirteen alignment checks completed.
3. Retrieval recovery: retain all original40 candidates and test up to four missing-address additions per reference on the previously exposed fixed5k diagnostic. The oracle is potential only; the new broad model must demonstrate an actual gain.

## Measurements so far

- Fused150-feature residual: early gain +0.0005513, selection score 0.9799258. This is below neural v2 0.9800089; it is not promoted.
- Neural v2 selected20k remaining errors: 543 retrieved true links outside its route, 1,504 routed links still rejected, 1,267 true links not retrieved, 227 false links. Error categories overlap with the per-entity macro loss; counts are not score gains.
- Broad60 and corrected alignment77: approximately two million existing pair rows, prepared without labels. Both log-loss-stopped adapters lost recall and were rejected. Removing the probability anchor also lost score. Training-error weighting and macro-F0.5 stopping produced no convincing gain. Compact measurements are in `objective_results.json`.
- Full v2 alignment statistics: all 2,206,821 supplied Source1 training rows, 136.15 seconds, no labels read.
- Neural v2 runtime gate passed on two actual country-stratified10k blocks, including France: projected full processing135.7 minutes with15% safety. This is a runtime estimate, not a new full submission or a portal score.
- Three-epoch last-two-layer neural training is running on the Mac over the sealed120k outer-training pairs. The lower encoder layers remain frozen. No accuracy claim before keyed inference and development evaluation.
- New sibling-neural evidence compares each routed candidate against predicted first-stage seeds. Training pairs for a dedicated sibling head use only the same outer-training businesses:115,530 pairs from18,823 owners. Calibration and evaluation retain the original Source1/target candidate keys. Inference seeds use predicted scores, never truth.
- Missing-address quota retrieval with broad60 recovered15 true and2 false links on the old5k diagnostic but reduced precision and raised singleton errors; it is rejected. Later neural-quota checks must exclude every adapter/head-fitted owner from the labeled subset.

## Release boundaries

No experiment is called a fresh score unless measured on an unexposed entity cohort. Select using development data, freeze one candidate, then evaluate the separate audit once. Do not reuse the consumed confirmation to select a new candidate. Country statistics use only the provided records and support unseen countries. No external business data, LLM labeling or test pseudo-labels are used.

Submission_03 and submission_04, their models and checkpoints remain intact. No full new test submission has started in this window. Preserve code and measured artifacts on both the Mac and the VMs before cleanup.
