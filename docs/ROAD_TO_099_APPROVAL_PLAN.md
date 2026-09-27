# Business Entity Resolution: plan toward 0.99

Date: 27 September 2026

Status: **proposed; implementation requires approval**. This document supersedes the experiment order in the earlier target plan. The research review changed no models, decision rules, cloud resources, or running jobs.

## Recommendation

Keep the current two-stage pipeline as the control. Build one reproducible development workbench, then test three distinct sources of improvement: learned compatibility between a difficult record and its sibling records; retrieval rules that cover the remaining misses; and better handling of competing owners. Test country transfer throughout. Carry the successful components into one frozen pipeline and evaluate it once on a new audit.

The next substantial experiment should add **new matching evidence**. Another round of random seeds, a lower global threshold, or a larger generic tree model is not a credible primary route from 0.9785 to 0.99. A small integration test for the recent 12 features is still worthwhile, but its measured effect is too small to anchor the strategy.

No method reviewed establishes that this dataset can reach 0.99 under the permitted inputs, much less that a particular change will achieve it. The plan is designed to find out quickly which remaining errors are recoverable, without confusing an oracle, development score, or local audit with a leaderboard result.

## 1. The verified starting point

| Measurement | Macro F0.5 | What it means |
| --- | ---: | --- |
| Submission 03 portal | 0.9399 | User-reported leaderboard score; the local record says the uploaded artifact association was not independently verified |
| Current final development pipeline | 0.979820 | Selected on the same 20,000 development references used to compare rules and models |
| Current fresh audit | **0.978547** | Frozen pipeline on 10,000 previously unused US/India references, with the ownership-coverage limits described below |
| Current development candidate oracle | 0.993451 | Perfect labels applied only to retained candidates; not model performance |
| India development candidate oracle | **0.988462** | Current retrieval cannot reach 0.99 on this India cohort, even with perfect decisions |

The fresh audit reports precision **0.996726**, recall **0.949690**, 15 singleton false predictions, and 50 non-singletons with empty predictions. Country scores are US **0.982320** and India **0.972883**. Submission 04 uses this frozen pipeline; it has no new portal result established by this review.

From the audited 0.978547, reaching 0.985 requires +0.006453; reaching 0.99 requires +0.011453. The latter removes **53.4% of the remaining macro loss**. The development candidate ceiling leaves only 0.003451 above 0.99. At unchanged retrieval, the development matcher would need to recover about **74.7% of its gap to that ceiling**. These calculations use their respective cohorts; they are not interchangeable audit and development ceilings.

Current model files:

- First stage: `models/bridge_v1/first_stage/model.txt`.
- Second stage: `models/bridge_v1/stacked_l255s42/model.txt`.
- Original freeze: `models/frozen_v4/frozen.json`.
- Submission resume uses a separate freeze containing the metadata-only checkpoint repair. The score replay was identical over 250 references and 10,000 pairs; model parameters were unchanged.
- Fresh audit report: `/mnt/er/entity/models/audit_v2/run/report.json` on the submission VM.

## 2. What the experiment history actually supports

The review covered the 78 JSON reports under `reports/experiments/`, the three recent large CPU feature reports, the direct-evidence experiment, the saved audit report, and the relevant retrieval, context, ownership, and validation code. Many reports are runtime or diagnostic probes, not separate model improvements.

| Stage or experiment | Result | Implication |
| --- | --- | --- |
| Initial pilot | 0.846216 on 500 holdout references | Historical starting point; too small for fine comparisons |
| Alias/retrieval pilot | 0.926795 on 500 holdout references | Strong early evidence that candidate quality and alias evidence matter |
| Larger alias model | 0.951057 on 2,000 holdout references; paired gain +0.017460 versus the earlier model | More training and improved evidence helped together; do not attribute the entire gain to sample size alone |
| Version 3 name/numeric features | Fresh 5,000-reference audit: 0.948512 → 0.960616; gain +0.012104, CI [0.009178, 0.015012] | A real evidence improvement, including script/phonetic and numeric comparisons |
| Larger first-stage training | Fresh 10,000-reference audit: 0.957786 → 0.965888; gain +0.008103, CI [0.006429, 0.009761] | Scaling supervised training was useful at that stage |
| Competition context | 0.973632 on the fixed 20k development selection | Candidate context and S1 competition added useful information |
| OOF second stage trained on 300k | 0.977126 on that selection, +0.003494 versus the 20k context incumbent | Larger, out-of-fold context training gave a meaningful gain |
| Learned ranker and bridge retrieval | Development oracle rose from about 0.9889 to 0.993451 | Retrieval headroom improved; downstream retraining was needed to use it |
| Both stages refitted on bridge candidates | 0.979077 with a single threshold | Current strongest base matcher |
| Existing exclusivity and rank-one rules | 0.979077 → 0.979820 on development | Development gain was largely not reproduced on fresh audit |
| Full current pipeline versus submission 03 | Same fresh audit: 0.957745 → 0.978547; gain +0.020802, CI [0.018827, 0.022710] | Strong verified local improvement, pending a corresponding portal result |
| Recent 12 features, 100k training references | Gains +0.000126, +0.000757, +0.000692 across three seeds | Small pre-bridge component gains; cannot be added to the current full-pipeline score |
| Direct-evidence expert | Confirmation gain +0.000223, CI [-0.000290, 0.000719] | Failed acceptance; do not promote the global blend |

The three recent seeds share the same entities. They are not three independent validation populations. Their mean gain, about +0.000525, is descriptive only. Two selection intervals were positive; one was not. All used the older candidate set and a single-stage classifier.

Ideas already tested, and why they should not lead the next round:

- **Independent expected-F0.5 decisions:** no eligible result under the precision/singleton constraints. A nominal raw gain increased singleton mistakes from 28 to 69. A joint structured variant would be a new experiment, but needs stronger probability/dependency evidence first.
- **Broad empty-prediction rescue:** small or negative gains and additional singleton errors. An empty prediction is not proof that a match should be forced.
- **Wider retrieval without matcher adaptation:** sometimes raised the oracle while reducing the actual score. Compare matched controls and retrain on the changed candidate distribution.
- **Generic tree capacity or seed ensembles:** some earlier gains, but the current 255-leaf context model stopped at 1,569 trees, below its 4,000-tree allowance. There is no evidence that simply adding trees to that incumbent solves the remaining errors.
- **More ordinary peer similarities:** already present. The context model compares each target with up to four other high-confidence candidates.
- **Removing alias/retrieval evidence globally:** the newest direct-expert test did not pass. Keep useful existing evidence and test complementary evidence on its actual failure cases.

Evidence paths: `reports/experiments/larger_model_comparison.json`, `features_v3_fresh_audit.json`, `round2/scale_fresh_audit.json`, `round2/competition_context_model.json`, `round3/stacked_c127.json`, `round3/bridge_stacked_l255s42.json`, and `round3/bridge_decision_rules_exclusive_stacked_l255s42.json`. Recent reports are under `models/cpu_scale_results_20260927_e70007_seed*/report.json` and `models/direct_evidence_expert_20260927_e70010/report.json`.

## 3. Corrections that must shape the plan

### The current detailed loss table predates the final rules

`reports/experiments/round3/bridge_losses_stacked_l255s42.json` describes the **0.979077 single-threshold development result**, not the final 0.979820 decisions and not the fresh audit:

| Loss category | Reported macro loss |
| --- | ---: |
| Retrieved true links rejected on nonempty predictions | 0.00704 |
| Empty prediction despite an available true candidate | 0.00350 |
| Unretrieved links on nonempty predictions | 0.00450 |
| Empty prediction with no available true candidate | 0.00195 |
| False links on matched entities | 0.00263 |
| Singleton false predictions | 0.00130 |

The useful broad priorities are retrieved misses, retrieval misses, then false merges. These are an ordered attribution of **macro F0.5 loss**, not independent gains to add together. The false-positive buckets already incorporate the metric's precision weighting; multiplying them by four would double-count it. Changing decisions or retrieval can move examples between categories.

Recompute the breakdown on exact final **development** decisions first. The existing audit may explain a discrepancy, but it is consumed and must not become the new tuning set. A new final audit remains untouched until the next freeze.

### The portal gap is not only a France question

The official S1 test file contains 809,986 India, 663,106 US, and 259,452 France references: **46.75%, 38.27%, and 14.98%**. Training is approximately 40% India and 60% US. India, the weaker labeled slice, has more weight in test.

If US and India retained their current audit scores on test, even assigning France a perfect score would give about **0.980556** over the complete test-country mixture. This is conditional arithmetic, not a forecast, an estimate of France accuracy, or a bound on the organizer's public/private subset if its country mix differs. It demonstrates why the strategy must improve US/India as well as address France.

France has no supplied labels. Prediction-count and probability distributions can expose anomalies, but cannot measure French accuracy. Do not force France to have the same link-count distribution as another country or treat the old portal gap as a fixed subtraction from future local scores.

### Ownership and audit do not have identical coverage

The test runner already groups the complete set of scored test claims by target. The audit/development replay instead discovers outside rivals using exact shared competition keys. It can miss a rival found by another retrieval path. Some replayed competitors are also in the model's training population, unlike test references.

The current decision function arbitrates only claims at or above 0.83, drops a claim only for a strictly higher rival, then rescues a reference's best candidate down to 0.70. Rescued claims and equal-score ties can therefore leave a target with multiple owners. The function's name describes a stronger condition than it enforces.

On the fresh audit, the second-stage threshold scored 0.978462; the final rules scored 0.978547: **+0.000085**, while singleton mistakes rose from 7 to 15. This is evidence to test a better decision policy, not permission to retune against that audit.

### Reproducibility is an immediate dependency

The bridge candidate/context/competition caches are absent from the checked local workspace and submission VM locations. Saved first-stage, second-stage and five OOF boosters, reports, and some probability arrays survived. This is not proof that no archive exists elsewhere.

Do not align a newly generated candidate table with an old probability array merely because lengths match. Rebuild or recover a pair-keyed export and regenerate probabilities with the saved boosters. Freeze index versions, native-extension hashes, normalization, aliases, feature order, model files, candidate ordering and decision parameters. The current freeze does not pin every native/index artifact.

## 4. Proposed work, in execution order

### Phase A — Reconstruct a trustworthy workbench

**Purpose:** make every later comparison directly answer whether the current pipeline improved.

After approval:

1. Inventory the existing archives and retained files. Recover the bridge caches if a valid copy exists. Otherwise transfer the necessary official training inputs, indexes, aliases and saved models to VM 2 and rebuild there. Leave the submission VM's running files unchanged.
2. Start with the fixed 20k development selection and enough training data for a paired smoke experiment. Reuse all five saved first-stage OOF models to regenerate training context; do not retrain them just because the probability arrays are missing. Refit them only if their inputs or learned preprocessing change.
3. Verify exact pair IDs and ordering, feature values, candidates and scores against recoverable reference artifacts. Reproduce 0.979820 within numerical tolerance when using the same saved development claims. If outside-rival artifacts are unavailable, reproduce the 0.979077 threshold result first and report that limitation explicitly; do not silently call it the final baseline.
4. Export one durable table containing S1 ID, target ID, source, candidate provenance, first-stage and second-stage scores, owner-related decisions, rank-one rescues and final acceptance. Store full truth separately, including singletons and true links absent from retrieval.
5. Compute the final development error budget and perfect-correction bounds for label-free routes: uncertain candidates, empty references' top candidates, script-mismatched names, missing addresses, shared-building cases and contested targets. A perfect-correction result is a feasibility ceiling, never a model score.
6. Check observational ambiguity: identical usable fields assigned to distinct owners, contradictory normalizations, and highly ambiguous missing-address records. Quantify the difficult mass rather than silently relabeling examples or memorizing IDs.

**Exit condition:** a reproduced baseline with pair-level provenance; current error categories; a data-exposure ledger; and measured estimates for the next experiments. If recovery or reconstruction exceeds the first timebox, report that before committing to a large run.

### Phase B — One controlled integration of the recent features

**Purpose:** determine whether the small CPU feature gain survives in the actual pipeline.

Keep bridge retrieval and the first-stage model unchanged. Add the recent features to the second-stage inputs, so first-stage OOF probabilities remain reusable. Compare the same 92-feature control against the augmented context model using identical training references, capacity, stopping set and candidate IDs. Start with one seed; additional seeds require a promising result.

Review the new feature semantics before integration. The research helper's generic accent folding is not the same as the production Latin-only accent folding; it must not remove meaningful Indic marks. Legal-form handling should be an alternate trailing-suffix view, preserving raw names. This is a versioned feature change, not a silent mutation of the frozen normalizer or index.

Train on the full existing 300k reference set for the promotion comparison. A smaller paired run is a smoke test, not proof that it beats the 300k incumbent. Recompute model-dependent context only when the corresponding scorer changes.

**Stop rule:** if the gain is below +0.0005 on the matched development comparison, or precision/singleton behavior deteriorates, do not spend further seed/capacity trials on this branch. A gain above that threshold remains provisional until confirmation. This branch is a low-cost integration check, not the projected source of a full percentage point.

### Phase C — Learned sibling compatibility: the primary new matcher experiment

**Hypothesis:** a difficult target can be closer to another genuine variant of the business than to S1. A learned target-to-target scorer may supply evidence that the existing maximum fuzzy similarities miss.

What is already implemented: up to four peers with first-stage probability at least 0.95; name/address token-set and token-sort comparisons; numeric overlap; and a few maxima. Simply increasing peer count would mostly repeat that design.

What is new:

- Build positive S2/S3 variant pairs from the **same labeled training owner**, and hard negatives from distinct training owners with similar names, shared addresses, or shared blocks. Limit pairs per owner so businesses with many variants do not dominate. Targets without a known owner are not automatically a valid target-to-target negative label.
- Train a compact field-aware compatibility model. Start with text, number-role, token-alignment and corruption-type features. Use the recently useful features where justified; retain contradiction evidence.
- At training time, select seeds using OOF model predictions, including their mistakes. Never give the context stage perfect ground-truth seeds that it will not have at inference. Exclude a candidate from supporting itself.
- Aggregate learned support from one or two seeds, agreement across target sources, support strength and conflicting name/number evidence. Count independent support carefully; nearly identical duplicates are correlated evidence.
- Feed these scores into a new second-stage model alongside the existing evidence. A strong seed is evidence, not an automatic label; do not merge an entire connected component or apply unconditional transitive closure.

**Bound the work before building it:** choose a route without labels, then measure its error coverage on development labels. A proposed starting budget is at most 20% of references, four difficult candidates and two seeds: at most eight extra comparisons per routed reference, about 2.77 million comparisons over the full test set if that routing rate holds. Batch model prediction; avoid one Python/model invocation per comparison. These are limits to test, not demonstrated throughput or coverage.

**Go/no-go:** proceed only if the route's perfect-correction gain has enough headroom to justify the branch—proposed minimum +0.003—and its measured runtime fits the end-to-end budget. Aim for a full-pipeline development gain of at least +0.001 before larger scaling. Reject a scorer that only improves isolated pair accuracy or reproduces existing peer features.

Research basis: learnable field-specific similarities were studied in [Bilenko and Mooney, KDD 2003](https://www.cs.utexas.edu/~ai-lab/pub-view.php?PubID=51499); scalable neighborhood evidence was studied in [Rastogi et al., PVLDB 2011](https://www.vldb.org/pvldb/vol4/p208-rastogi.pdf). The former motivates learning noise-specific comparisons; the latter motivates bounded collective evidence. Neither paper establishes the effectiveness of this exact business-sibling design.

### Phase D — Raise retrieval coverage under a candidate budget

**Objective:** target a retained-candidate macro oracle of at least **0.997 overall** and at least **0.995 for India** on development. These are engineering targets, not achieved results.

First distinguish links never generated, links discarded by an intermediate shortlist, and links removed by final top-K. The remedies differ. The old missing-candidate profile is useful history but must be refreshed on current bridge retrieval.

Test individually, in the order supported by the new miss counts:

1. **A ranker trained on the actual pre-cap candidate distribution.** The current cheap ranker was trained on negatives retained by an older blocker. Train from wider training-fold shortlists, including the difficult examples it currently discards; calibrate comparisons on the same source population.
2. **Selective script/name retrieval.** Add bounded keys from alternate name views, collapsed names, and learned character correspondences where the existing approximate romanization fails. Similarity features alone do not retrieve a record with no shared indexed key.
3. **Missing-address and address-led paths.** Preserve a small number of strong name-only candidates when address is absent, and address/number conjunction candidates for changed trade names. Require measured coverage; do not assume a shared building proves identity.
4. **Learned multi-pass predicates.** Choose combinations of existing allowed keys based on new training-fold true-link coverage per additional candidate and per unit of runtime. Fit on training misses, choose budgets on tuning, and evaluate all references, not only the misses used to design the rule.
5. **Adaptive candidate allocation.** Compare against the fixed 20-per-source control. Use ambiguity/rank margins to spend fewer candidates on easy cases and more on difficult ones. Start at mean final candidates no greater than 40, p95 no greater than 60, hard maximum 80 per reference; these are proposed test limits, not organizer rules. Do not enforce them when they contradict measured match multiplicities.

Every added path must report recovered true links **and lost previously retrieved links**, macro oracle, pair recall, mean/p95/max candidates, stage-specific pair volume and runtime. A higher oracle without a downstream gain is not a promotion. Refit affected matchers/context on the new candidate distribution before the final comparison.

Research basis: [Michelson and Knoblock, AAAI 2006](https://cdn.aaai.org/AAAI/2006/AAAI06-070.pdf) learns combinations of blocking predicates; [Sparkly, PVLDB 2023](https://pages.cs.wisc.edu/~anhai/papers1/sparkly-vldb2023.pdf) studies top-K lexical blocking and its recall/output-size/runtime tradeoff. The proposal extends our existing postings engine; it does not assume that replacing it with Spark/Lucene will improve runtime.

### Phase E — Competing owners and no-match decisions

This is a bounded companion experiment, not the assumed source of the entire missing score.

1. Form all proposed links, including rank-one rescues, before final ownership arbitration.
2. For each target, compute its best and second-best owner scores, the margin, claimant count, and reference/target reciprocal ranks. Keep a no-owner option. S1 retains unlimited target capacity; do not impose one-to-one business matching.
3. Compare deterministic unique-owner selection with an ambiguity-margin abstention rule. Include equal-score ties and cases where a false high-scoring rival steals a true match. Do not repeatedly rescue new candidates after arbitration without defining and testing termination/consistency.
4. If these features are learned, generate their training inputs from OOF claimant scores. Training-fold rivals scored by a model that fitted them must not masquerade as inference-like uncertainty.
5. Evaluate on a declared complete pseudo-test reference universe, scoring **every reference within that universe**, with full target pools and no labels from those reference owners in the fitted model. This avoids the exact-key rival-discovery shortcut. Its density may still differ from the full official test universe; report the difference rather than claim identical coverage.

Before a large pseudo-test run, measure disputed-target prevalence and its perfect-correction bound. If too little weighted error is affected, keep only the correctness/consistency change and spend the training budget elsewhere. A structural consistency rule can still lower F0.5, so it must be compared with the unchanged control.

Research basis: [Sadinle, Bayesian Estimation of Bipartite Matchings](https://arxiv.org/abs/1601.06630) studies dependent linkage and uncertainty, but its two-sided one-to-one setting is different from this challenge. It motivates coherent claims, not a drop-in Hungarian assignment. [Dembczynski et al., ICML 2013](https://proceedings.mlr.press/v28/dembczynski13.html) studies F-measure decisions from probabilistic models; it does not justify rerunning our failed independent-probability rule unchanged.

### Phase F — Country transfer and learned text alignment

Country transfer is an evaluation track for the earlier phases, not something to postpone until after a high aggregate development number.

- Report per-country macro F0.5, candidate oracle, precision/recall, singleton errors and empty non-singletons. Also report US/India scores reweighted to the **known-country** test proportions; label this as a sensitivity check, not a complete test estimate.
- Run train-US/evaluate-India and the reverse on bounded cohorts. Refit every supervised component—including aliases, ranker and any learned edit mappings—without held-out-country labels. Permit label-free target indexing as inference requires it, and record that explicitly.
- Add name-family and shared-building stress cohorts to expose memorization and address-only false merges. Keep near-identical owners/variants in the same fold where defining such a cohort.
- Preserve original Unicode and create alternate views for dotted suffixes, accents and alphanumeric boundaries. Do not destroy Indic marks or blindly strip short words such as `sa` everywhere.
- If current misses support it, learn character/subword edit and alignment costs from official training variants, rather than adding another generic edit-distance feature or relying only on exact whole-name aliases. Address and name channels should remain separate. Train learned mappings inside the appropriate folds and test on naturally occurring errors.
- Test training-only, label-preserving perturbations with learned noise rates if needed. Synthetic improvements do not establish natural-error or France performance. Do not use generated pseudo-labels for French business identities.

Research basis: [the NAACL 2024 blocking generalization study](https://aclanthology.org/2024.naacl-long.483/) evaluates in- and out-of-distribution behavior and shows why small within-domain blocking tests are insufficient. The [learned similarity work](https://www.cs.utexas.edu/~ml/papers/marlin-kdd-03.pdf) motivates learning field-specific edit costs. Neither gives a France accuracy estimate for this dataset.

### Phase G — Optional neural fallback

Only consider this if diagnostics show enough recoverable error and the CPU evidence/retrieval branches plateau.

Candidate: a small offline multilingual cross-encoder fine-tuned only on official training pairs. [Ditto](https://www.vldb.org/pvldb/vol14/p50-li.pdf) supplies a relevant pair-classification approach. [Multilingual DistilBERT](https://huggingface.co/distilbert/distilbert-base-multilingual-cased) is listed as Apache-2.0; [XLM-R base](https://huggingface.co/FacebookAI/xlm-roberta-base) is listed as MIT. Verify the exact checkpoint, parameter count and license at selection; do not inherit a library license as proof of a model's license.

Start with a measured uncertainty route that includes suspicious accepted links as well as rejected ones. Limit input length only after measuring truncation. Include difficult negatives and out-of-fold supervision. Evaluate full-entity macro F0.5 and end-to-end runtime, not only a balanced pair benchmark.

GPU quota requests did not yield a usable Azure GPU in the prior checks. This branch is not on the immediate critical path, and this plan does not authorize another VM, an external paid provider, or CPU inference over all roughly 69 million candidate pairs. Neural blocker approaches such as [Sudowoodo](https://arxiv.org/abs/2207.04122) are relevant alternatives, but encoding and indexing the full target collection would be a separate measured project, not a quick substitution.

## 5. Validation and promotion protocol

1. **Unit of splitting:** the real training owner/S1 identity, with all labeled variants together. Fit learned mappings and seed/competitor scorers without the evaluated owner. Review all supervised preprocessing dependencies, not merely the classifier's fit rows.
2. **Development:** use the existing 50k pool for diagnosis, early stopping, and predeclared selection roles. Its repeated use makes resulting scores exploratory. Keep all comparisons on identical reference IDs, full truth and candidate policies appropriate to each arm.
3. **Confirmation:** reserve a new entity-disjoint cohort, proposed 20k, from eligible unused tuning identities before fitting new branches. Log which labels were accessed. Use it once for the selected combined candidate; if it informs another change, it becomes development data.
4. **Final audit:** reserve a separate proposed 20k cohort from unused holdout identities, excluding every prior training, tuning, inspected and audit-label population. Freeze code, models, indexes, candidate rules, thresholds and claimant coverage before reading its labels. Do not reuse the submission 04 audit as the new pass/fail set.
5. **Ownership:** use a complete, explicitly defined claimant universe for ownership experiments. If full coverage is unaffordable, report the approximation and do not claim the corresponding result has full-test parity. Scoring unlabeled audit-universe rivals is allowed only with fold-safe models; audit-owner labels remain sealed until the final comparison.
6. **Statistical comparison:** bootstrap paired per-S1 macro differences, retain zero-candidate references, and report confidence intervals alongside absolute precision/recall and error counts. A bootstrap on a repeatedly tuned selection set is not independent confirmation. Inspect component-cluster uncertainty as a sensitivity check when ownership decisions create dependence between references.
7. **Proposed combined-candidate gate:** positive lower bound on the confirmation macro-gain interval; precision loss no more than 0.001; no increase in singleton false-prediction rate; no material India regression; and acceptable candidate/runtime cost. Predeclare any tolerances before seeing the confirmation result. A useful local milestone is audited 0.985, then audited 0.99; a portal 0.99 is a separate achievement.
8. **No forced gain:** do not add gains from overlapping branches. Measure the combination. If a branch fails, retain its report and restore the control.

A model may improve some metrics and fail a constraint. Report that plainly. Do not equate micro precision near 0.997 with macro F0.5 near 0.997.

## 6. Compute, runtime and checkpoints

Use the existing **VM 2 only** for approved new work: 32 vCPUs, about 125 GiB usable RAM. No new machine is required for the initial phases. Submission 04 and its VM remain unchanged.

The following are **work budgets**, not promises of completion or score:

| Checkpoint after approval | Initial budget | Required decision |
| --- | --- | --- |
| Cache inventory, transfer plan, 1k–5k parity/rebuild benchmark | 30–60 minutes | Establish throughput and whether a full rebuild fits the session |
| Reproducible development export and current error/route bounds | 30–90 additional minutes, conditional on transfer/rebuild | Rank the actual error opportunities; stop unpromising routes |
| Paired feature integration and learned-sibling pilot | 1–2 hours after inputs are ready | Compare against the complete control; retain only useful branches |
| Targeted retrieval/owner/transfer probes | 1–2 hours, prioritised by preceding results | Choose one combination; do not run every possible grid |
| Larger confirmation and frozen audit | Schedule from measured throughput | Report required time before committing beyond the session budget |

Use the first five to six hours as a bounded research session, with a checkpoint after Phase A. If cache recovery consumes that budget, reduce experiment scope rather than pretend all phases fit. No automatic shutdown is added; resource state remains under the user's explicit instructions.

For a two-hour full-test output target, 1,732,544 references require approximately 241 references/second even before merge and validation. Reserving 15 minutes for finalization requires roughly **275 references/second during scoring**. Benchmark the complete proposed pipeline, including retrieval, feature construction, model prediction, ownership, export and both validators. Report throughput by country, routed-pair fraction, peak memory and tail batch times. Additional RAM alone is not a runtime solution.

Store caches, model files, exact pair IDs, predictions, thresholds, logs and provenance on durable storage. A probability array without its pair-key order is not a reusable cache. Save a resumable checkpoint at each expensive stage and verify hashes on reuse. Do not place the only copy of training artifacts on a temporary VM disk.

## 7. Submission and challenge constraints

This plan does not start another full-test submission or change submission 04. A later approved submission must preserve all existing versions, score every official S1 including France, keep singleton predictions empty when appropriate, and pass both strict and official validators with ID checking.

`candidate_pairs.tsv` must account for the exact S1-target candidate set considered by the final matching pipeline, including candidates introduced by later retrieval. Do not shrink the reported set after scoring. Keep separate diagnostics for wider preliminary ranker pools and auxiliary target-target comparisons, and describe their costs and roles clearly. Mean, tail candidate counts and blocking recall all matter under the organizer's updated review criteria.

Research pages supply methods and model documentation only. No business registry, geocoder, identity API, external business dataset, hosted LLM oracle, or external identity lookup is used. Any pretrained model must independently satisfy the stated MIT/Apache-2.0 and at-most-8-billion-parameter requirements and run offline; any fine-tuning uses only organizer-provided training data.

## Approval scope

Recommended initial approval: **Phases A–C on VM 2**, with the Phase A checkpoint allowed to reorder or reject the bounded sibling experiment based on measured error coverage. Phase B is one controlled comparison, not an open-ended sweep. Country diagnostics accompany those phases.

Phases D–G describe the subsequent decision tree. They are not a commitment to run every experiment. At the checkpoint, present the measured score delta, remaining error budget and runtime before expanding work. No new cloud provisioning or submission generation is included.

Success for the next approved session is a reproducible improvement over the exact current pipeline—or clear evidence rejecting an approach—not a promised 0.99. The final target remains **0.99 macro F0.5**, supported first by honest local evaluation and ultimately by the portal.
