# ML Challenge 2026: Business Entity Resolution

**Team:** EntityResolution
**Members:** Mohith, Sanhitha, Harsha, Sahasra
**Date:** 26 September 2026
**Status:** Development methodology. Final output counts and selected release artifact are pending.

## 1. Executive Summary

The current system combines complete disk-backed target indexes, bounded candidate retrieval, a training-only learned name channel, 65 versioned pair features, and a LightGBM classifier. After selection on 10,000 tuning entities, the latest model achieved 0.9606165 local macro F₀.₅ on 5,000 unused audit entities. The frozen v2 baseline scores 0.9485120 on those same entities. These are local validation results, not leaderboard scores or a completed submission.

## 2. Methodology

### 2.1 Problem Analysis

All seven provided TSV files were scanned with explicit tab parsing. Training counts: S1 2,206,821, S2 5,034,616, S3 5,285,603. Test counts: S1 1,732,544, S2 4,887,273, S3 5,082,316.

Ground truth contains 7,638,365 links; every target exists, S1 coverage is complete, and no target is labeled for multiple S1 entities. There are 123,247 training singletons (5.5848%). No duplicate source IDs or malformed rows were found, and train/test source IDs are disjoint.

The training countries are US and India. Test S1 includes 259,452 French records (14.9752%). Country labels remain unrestricted strings. Many S2/S3 addresses are blank, making name-only branch disambiguation a material limitation.

### 2.2 Solution Strategy

**Approach:** Blocking followed by a supervised pair classifier.

Primary normalization preserves Unicode combining marks, numbers, and raw fields. Latin accent folding is an alternate view. Ambiguous address abbreviations are preserved. Legal-form-free names and separated name/address numeric tokens provide additional evidence.

The learned name channel maps noisy target name strings to reference name cores using only training-fold matches. It learns text transformations rather than entity-ID features. Confident aliases require at least two examples and at least 0.98 dominance. Ambiguous names retain conditional alternatives.

## 3. Candidate Generation

The measured pilot indexes every training S2/S3 record with SQLite FTS5 and exact-name/core B-trees. Paths use rare names, rare addresses, numeric tokens, exact names/cores, core prefixes, postcode/name combinations, numeric/location conjunctions, and training-only alias families. Character retrieval exists but is disabled in the selected pilot.

Candidates are unioned and deterministically reranked using name, address, numeric, IDF, alias, and path evidence. At most 20 candidates per source are retained. The pilot averaged 40 candidates per S1 and achieved 0.9680378 link recall on the fresh audit; its oracle macro F₀.₅ ceiling was 0.9893222.

Compound family/address indexing is an experiment, not the selected system: smaller candidate sets caused unacceptable tuning recall losses.

**Final test candidate count:** pending full inference. Candidate exports must contain exactly the records scored by the matching model, before thresholding; they are not pruned afterward.

## 4. Matching Model

**Architecture:** LightGBM 4.3.0 binary gradient-boosted trees; 1,200 maximum rounds, learning rate 0.04, 31 leaves, minimum child samples 30, L2 regularization 2, feature and row fractions 0.9, seed 42. Tuning log loss controls early stopping with patience 75.

**Features:** a registered 36-feature baseline, 14 version-2 additions and 15 version-3 additions. Version 3 adds standalone numeric edit types, approximate Unicode/phonetic names, collapsed/domain-style name comparison, corpus-local name rarity and unmatched address-token evidence. Primary normalization and all earlier feature definitions remain unchanged. Evidence includes fuzzy name/address similarities, token overlaps, character cosine, numeric agreement, uncertain postcode overlap, length/missingness flags, country equality, source flags, retrieval score, canonical-name similarity, alias support/confidence, family frequency, address containment, and IDF-weighted agreement. IDs, label ownership, and folds are excluded from model features.

**Training:** all retrieved positives and difficult retrieved nonmatches; no label-based positive rescue. Each S1 has equal total pair weight. Held-out counterparts are excluded from training even when retrieved as negatives.

**Decision:** select a global threshold using tuning macro F₀.₅; empty predictions are allowed and no match is forced. The frozen v2 baseline selected 0.71. Version 3 selects 0.76 on 10,000 tuning entities, maximizing macro F₀.₅ subject to precision no more than 0.002 below baseline and no increase in singleton false positives. Outputs are classification scores without a probability-calibration claim.

## 5. Results and Error Analysis

| Metric on identical 5,000 audit entities | Frozen v2 | Version 3 |
|---|---:|---:|
| Macro F₀.₅ | 0.9485120 | 0.9606165 |
| Link precision | 0.9817793 | 0.9883041 |
| Link recall | 0.9015173 | 0.9165176 |
| Singleton false positives / 258 | 27 | 19 |
| US macro F₀.₅ | 0.9594630 | 0.9684397 |
| India macro F₀.₅ | 0.9310631 | 0.9481513 |

Paired gain: 0.0121045, with 95% entity-bootstrap interval [0.0091784, 0.0150119]. Version 3's macro interval is [0.9572219, 0.9639274]. All predefined fresh-audit gates pass: positive lower gain bound, precision loss at most 0.002, and no increase in singleton errors. Retuning the v2 baseline on the same tuning set retained threshold 0.71; this control was frozen before audit scoring.

Classifier fitting uses 20,000 training entities and 582,358 retained training pairs. The new tuning and audit populations contain 10,000 and 5,000 entities, excluded from previously recorded experiment reference manifests. The name channel is unchanged and fitted only on the training fold. Its training examples may benefit from their own contributions to alias statistics; out-of-fold alias fitting remains a separate experiment.

The audit's mean candidate count is 40; retrieval recall is 0.9680378 and candidate oracle macro F₀.₅ is 0.9893222. Better ranking is required to reach a 0.99 oracle on this cohort. On tuning, version 3 still loses 1,102 true links in blocking, rejects 1,757 retrieved true links and accepts 424 incorrect links. These tuning diagnostics guide subsequent work; the completed audit is not reused for tuning.

A repeated 600-reference, eight-worker runtime probe measured 17.74 references/second for baseline, 19.99 with memory mapping and 21.92 with numeric/location queries also removed. Memory mapping preserved exact candidates and predictions. Removing the path lowered matcher macro F₀.₅ from 0.958426 to 0.956390 despite a slight oracle increase; it was not adopted. These are short v2-model probe timings, not measured full-test runtime or version-3 throughput.

Stable whole-file hashing creates 70/15/15 entity folds; targets follow their owners. Fresh sampling uses a separate documented hash domain and is not exact stratification. Earlier pilot holdouts remain exploratory. France has no labels, so neither French accuracy nor a leaderboard rank is inferred. Its large predicted-match tail is a diagnostic warning, not evidence for imposing a cardinality cap or a target prediction distribution.

## 6. Release Status

The offline training and batched inference code are runnable. The seven-file integrity audit and synthetic metric, Unicode, alias-isolation, retrieval, and output-contract checks pass. Submission 01 was stopped at 37,000 completed references, preserving its frozen v2 model and checkpoints. Full official-test prediction, retrieval runtime improvements, final candidate-budget selection, complete output validation, and the final archive remain pending. No leaderboard upload has occurred.

## Appendix A. Code and Reproduction

Source is under `code/business_entity_resolution/src/`. `run_training.py` fits/evaluates the pipeline; `run_inference.py` builds complete test indexes and exports both required TSVs. The package README provides exact commands and pinned dependencies.

Run both `utils/validate_submission.py` and the unmodified organizer helper `utils/organizer_validate_submission.py`; include `--check-ids` for the official helper. Limited inference is explicitly marked unsafe for submission.

Large raw files and local artifacts remain out of Git. New artifacts record split IDs, feature version, search configuration, parameters, threshold, source hashes, input/index fingerprints, and the generated model license. Final output files must be identical to the files used for the portal and archive.

## Appendix B. Fair Play and Licenses

All identity evidence comes from the provided dataset. No external business registry, geocoding, identity API, external augmentation, or hosted model call is used. No pretrained neural model is used. LightGBM is MIT-licensed; generated tree and name-channel artifacts carry the project's model-artifact MIT notice. Each future added model must independently satisfy the organizer's license, size, offline, and data-use restrictions.
