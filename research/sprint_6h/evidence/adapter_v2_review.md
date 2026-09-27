# Compact anchored adapter: evidence, release path, and stress check

Read-only review of the failed sibling experiment and the coordinator-authorized v2 pilot. No model fit, shared source edit, or new audit label access was performed.

## One precise experiment

Keep the frozen retrieval, candidate IDs, global route, and decision thresholds. Fit a small additive logit correction on the exposed residual-training 20k role, with the disjoint early 10k role supplying early stopping. Use **113 compact features**: direct S1-to-target raw 36 views; per-feature mean and maximum over the existing at-most-two routed TT seed comparisons (72); and p1, p2, frozen candidate rank, seed count, and maximum seed p1 (5). Do not refit the TT classifier or use its scalar probabilities. Only routed rows change.

For a routed pair, define `b = logit(clip(frozen_p2, 1e-6, 1-1e-6))` and train binary boosting with per-row `init_score=b`. The output is `sigmoid(b + learned_raw_margin)`. Use the same offset in the early Dataset. Preserve existing shallow residual scale (at most 150 trees, small leaves, regularization, one fixed configuration) and entity-normalized training weights. No threshold or blend-weight grid is proposed. Include unchanged control as the fallback if the early-stage paired decision replay fails the existing promotion guards. Evaluate the selected 20k only once after the specification is fixed; the complete heldout claimant cohort remains required for final ownership validation.

This adds *typed evidence to the adapter*: it can distinguish good name/phonetic agreement under missing addresses from actual numeric/address contradiction, conditioned on the direct S1-target match. Anchoring makes zero correction reproduce the incumbent and asks the learner to explain remaining errors rather than learn a new probability scale from scratch. It does not guarantee recall or prevent all harmful corrections.

## Evidence for the change

The failed 30k-claimant grid loses 188–236 true links while removing 47–59 false links, for selected macro changes -0.000180 to -0.000220. Its nine-feature residual is a fresh binary classifier, with only p1/p2 and pooled TT probability summaries. On the residual-training role, the 471 true rows with p2 in [0.66,0.83) have mean p2 0.751906 and mean residual probability 0.641561; none receive residual >=0.83, although 229 have TT maximum >=0.90. The model mainly down-calibrates difficult pairs.

The TT classifier itself was trained on 218,916 positive versus 12,560 negative examples. Its high probabilities are not interchangeable with S1-target match probabilities. The sibling implementer's independent early-role diagnosis finds 120 same-name/same-address negative TT edges with mean compatibility 0.975, including 113 >=0.90. All 784 early routed false negatives have a true seed, and seed correctness is 99.63%; wrong seeds are therefore not the principal failure. Missing-address early false negatives have mean TT maximum 0.740 versus 0.757 for corresponding false positives. These findings support retaining raw evidence types and direct S1 credibility rather than another TT rebalancing experiment.

Current selected20k error evidence also identifies the intended opportunity: 1,106 of 2,051 rejected retrieved true links have a missing candidate address; 752 are routed. Numeric contradiction is present on 267 false negatives versus 36 false positives. These are diagnostic counts already exposed earlier, not a justification for reopening selected labels during the v2 pilot.

## Available data and exact release reconstruction

VM2 has complete keyed development50k and training50k OOF workbenches, each with 2,000,000 pairs and approximately 247 MB of captured 92-feature chunks. Residual20k, early10k, and selection20k role files exist with separate truth and manifests. The training50k first-stage scores are owner-excluded OOF, but its second-stage scores are explicitly in-sample: **do not use those p2 values as honest training offsets**. The authorized outer-tune residual20k owners are disjoint from incumbent parameter training and are suitable for the offset adapter.

The original frozen stage-two script trains a binary-logloss model on the 65 direct, 20 context, and seven competition features, with per-reference weights and separate early owners. Its current model has 1,569 trees, 255 leaves, and 300k training references. The VM1 deployment does not retain the full original 92-feature caches; full-test chunks retain p1/p2 only. Reconstructing all 92 features would therefore create a new full-test capture cost. The compact proposal avoids it:

1. Read each stored test score chunk and select the globally pinned routed references/candidates using their frozen p1/p2 and exact IDs. Rank is over all retained candidates for a reference, not just the four routed rows.
2. Fetch only each routed candidate and its at-most-two seed records from the frozen test target indexes, and the corresponding supplied Source 1 records.
3. Build direct and TT raw36 views using identical frozen normalization and feature code; predict the small correction; preserve every unrouted probability and every candidate key.
4. Replay the complete declared global claimant decisions after score updates, then evaluate/export. Never reuse the pre-adapter accepted column.

If the gate's *raw* target lookup is used instead of the frozen normalized indexes, apply `normalize_name` and `normalize_address` identically before `clean_target`. `clean_target` alone does not expand name abbreviations or turn `&` into `and`, while the training index already did. This is a concrete train/test parity risk.

One-CPU read-only benchmark used 2,000 label-free hash-sampled routed residual candidates, 3,634 TT comparisons, 5,031 target records and 1,677 Source 1 records. Target lookup took 0.660 s, reference normalization 0.149 s and direct+TT raw36 generation 0.302 s: **1.111 s total**. The benchmark pooled max/min instead of the authorized max/mean; its 108 raw feature values and computational scale are otherwise equivalent. At the hard full-test cap of about 1.386 million routed rows this extrapolates optimistically to about 12.8 CPU-minutes of warm lookup/feature work. It excludes full score scanning, cold test-index I/O, global routing, prediction, hashing, and global replay; a measured test-batch end-to-end benchmark remains mandatory before release. No training runtime is claimed from this no-fit probe, but the routed train table is only approximately 7,389 rows, rather than a new multi-million-row backbone fit.

## Independent design checks

Mean/max pooling can combine the best name evidence from one seed with the best address evidence from another, although no individual peer jointly supports both. This was already possible in the older context features; the base p2 partly captures the original context, but the new raw adapter can learn such synthetic combinations. Keep seed count and report this mixed-seed pattern in the stress slice. Do not silently add new features after seeing pilot outcomes.

Anchoring does not solve the binary-logloss versus entity-F0.5 difference. The learner can still prefer a downward correction when probabilities are overconfident on the routed distribution. Early logloss decides stopping; final paired early-role decision replay decides whether the candidate deserves the one selected-role evaluation. The no-seed, singleton, and unretrieved-link blindspots remain outside the fixed route.

LightGBM documents `Dataset.init_score` as the starting score and `Booster.predict(raw_score=True)` as raw prediction output. Its Booster prediction API has no per-row offset argument, so the external incumbent logit must be explicitly added at inference; using transformed standalone Booster predictions would discard the intended anchor. Binary dataset persistence also omits `init_score`, so reloads must restore it. See the official [Dataset API](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.Dataset.html), [Booster API](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.Booster.html), and [binary-objective parameters](https://lightgbm.readthedocs.io/en/stable/Parameters.html#boost_from_average). Set `sigmoid=1` and disable an additional mean-label initialization. Require an offset-identity check plus saved/reloaded correction parity before the actual pilot; no semantic test fit was run in this review.

## One country-transfer risk check, without extra fits

Predeclare a **country-stratified collision slice on early10k only** before inspecting v2 outcomes. Build its membership from records/features alone, using the same globally pinned 30k route and claimant graph for both control and candidate:

- TT support collision: maximum name-set >=0.90 and maximum address-set >=0.90.
- Direct collision: direct S1-target name-set >=0.90 and address-set >=0.90.
- Generic/shared-address stress: intersection of those conditions, with the normalized/reduced Source 1 name occurring at least twice in `models/sprint_6h/training50k_oof/references.json`, the already-declared outer-training text universe. Count exact normalized Source 1 address repeats in that same 50k training text universe as a second flag, without reading owner truth for feature membership. These are diagnostic membership flags, not new model features.

For each slice, report US and India separately: number of routed pairs/references, true/false pair counts, mean score correction by true/false class, true links added/removed, false links added/removed, singleton false-prediction entities, and paired entity macro change after complete same-cohort decisions. Also report how often name/address maxima come from different seeds. **Fail this stress check if either country's generic/shared-address slice has a net increase in accepted false links, or if singleton false-prediction entities rise**. Apply the ordinary country, precision, singleton and paired-macro guards across the whole early role too. Sparse slices must show counts rather than a reassuring percentage.

This directly probes the known TT same-name/same-address hard negatives: the raw adapter should not lift them merely because a peer looks highly compatible. The same fixed model is checked in both observed countries; this is a transport-risk stress check, not a genuine train-on-one-country experiment or proof of French accuracy. No additional model fit, threshold choice, selected-role search, or French labels are required. Later, label-free French feature coverage may identify how often the deployment enters this slice, but cannot establish its accuracy.

Findings and parity pitfalls were sent to the coordinator and sibling implementer. The proposal remains one compact anchored adapter; the typed views and offset are both essential parts of that single experiment.
