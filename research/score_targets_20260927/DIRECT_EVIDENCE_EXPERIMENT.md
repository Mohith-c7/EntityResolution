# Complementary direct-evidence matcher

## Hypothesis

A matcher trained without blocking score, alias-derived similarities, or corpus-frequency features may make different errors from the existing matcher. A small probability blend may recover true matches that the existing matcher rejects because retrieval or alias evidence is weak. This is a hypothesis, not a measured gain or a claim of improved France accuracy.

The current code resolves canonical names through the alias index in `build_extended_pair_features`. Therefore merely removing alias confidence would leave other alias-derived evidence. This experiment removes all five canonical-name features, alias confidence/support, name-family frequency, blocking score, and eight corpus-frequency/IDF features: 17 columns in total. It retains 60 direct-comparison features, including the 12 new text/number features being tested separately.

## Research and choice

- [Generalizing to Unseen Domains: A Survey on Domain Generalization](https://arxiv.org/abs/2103.03097) describes distribution shift and learning strategies including ensembles. It motivates testing dependence on training-domain evidence; it does not establish this particular feature-removal method.
- [Ditto: Deep Entity Matching with Pre-Trained Language Models](https://arxiv.org/abs/2004.00584) studies pairwise text evidence and augmentation for entity matching. It supports investigating complementary evidence, but this experiment does not implement Ditto or use a transformer.
- [LightGBM 4.3.0 parameters](https://lightgbm.readthedocs.io/en/v4.3.0/Parameters.html) documents binary classification and ranking objectives. Binary classification retains the no-match option directly; a pure ranker would not by itself decide whether any candidate should be accepted. Use the installed MIT-licensed LightGBM; no additional model download or external business data.

## Fixed protocol

1. Reuse the hash-verified development cache already on VM 2: 100,000 training entities, 10,000 early-stopping entities, and 20,000 old development entities. Candidate retrieval is unchanged and predates bridge retrieval.
2. Train one 60-feature expert with seed 70007, 31 leaves, learning rate 0.04, at most 2,400 trees, and early stopping after 75 rounds. Use six threads, leaving two of the eight previously spare VM threads for overhead.
3. Use the already-running 77-feature model with seed 70007 as the fixed comparator. Do not choose the best seed after seeing results. Verify its data and model hashes before comparing.
4. Split the 20,000 development entities deterministically into 10,000 for blend/threshold selection and 10,000 for confirmation. Keep all candidates of each entity together. The confirmation subset belongs to a previously used development pool and is **not a fresh audit**.
5. Try expert weights 0, 0.10, 0.25, 0.50 and 1. Select the weight and threshold only on the blend-selection subset. Independently select the comparator threshold on that same subset. Freeze the decisions before evaluating confirmation labels.
6. Evaluate full-truth macro F0.5, precision, recall, singleton false positives, empty non-singletons, and an entity bootstrap confidence interval. Report country, no-alias-evidence and weak-retrieval slices using all candidates and full truth for each selected entity.
7. A useful pilot requires a positive lower bound on the confirmation gain, precision loss no larger than 0.002, and no increase in singleton false positives. A positive pilot must still be tested in the complete two-stage pipeline before promotion.

## Limits

Removing features does not remove their influence on which candidates were retrieved. This is a conditional matcher experiment, not an independent retrieval system or a clean leave-country-out evaluation. Country slices contain only the provided labeled countries. Neither the current audit nor official test labels are used. No submission is generated or changed.

## Run and artifacts

Run `bash research/score_targets_20260927/cloud/run_direct_evidence.sh` on VM 2 only. The script refuses an existing experiment output directory.

Output: `models/direct_evidence_expert_20260927_e70010/` contains the protocol and split IDs, model, exact pair-keyed probabilities, frozen decision parameters, hashes, and final report. The expert saves before waiting for the comparator to finish; that wait has a one-hour limit. The job does not provision, stop, or modify either VM.
