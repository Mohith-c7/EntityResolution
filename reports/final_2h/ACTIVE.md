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
