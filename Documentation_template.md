# ML Challenge 2026: Business Entity Resolution

**Team:** EntityResolution
**Members:** Mohith, Sanhitha, Harsha, Sahasra
**Date:** 26 September 2026
**Status:** Development methodology. Final output counts and selected release artifact are pending.

## 1. Executive Summary

The current system combines complete disk-backed target indexes, bounded candidate retrieval, a training-only learned name channel, 50 pair features, and a LightGBM classifier. A full-target pilot achieved 0.9267946 local macro F₀.₅ on 500 held-out S1 entities. This is exploratory validation evidence, not a leaderboard score or a completed submission.

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

Candidates are unioned and deterministically reranked using name, address, numeric, IDF, alias, and path evidence. At most 20 candidates per source are retained. The pilot averaged 40 candidates per S1 and achieved 0.9662005 held-out link recall; its oracle macro F₀.₅ ceiling was 0.9871680.

Compound family/address indexing is an experiment, not the selected system: smaller candidate sets caused unacceptable tuning recall losses.

**Final test candidate count:** pending full inference. Candidate exports must contain exactly the records scored by the matching model, before thresholding; they are not pruned afterward.

## 4. Matching Model

**Architecture:** LightGBM 4.3.0 binary gradient-boosted trees; 1,200 maximum rounds, learning rate 0.04, 31 leaves, minimum child samples 30, L2 regularization 2, feature and row fractions 0.9, seed 42. Tuning log loss controls early stopping with patience 75.

**Features:** a registered 36-feature baseline plus 14 versioned additions. Evidence includes fuzzy name/address similarities, token overlaps, character cosine, numeric agreement, uncertain postcode overlap, length/missingness flags, country equality, source flags, retrieval score, canonical-name similarity, alias support/confidence, family frequency, address containment, and IDF-weighted agreement. IDs, label ownership, and folds are excluded from model features.

**Training:** all retrieved positives and difficult retrieved nonmatches; no label-based positive rescue. Each S1 has equal total pair weight. Held-out counterparts are excluded from training even when retrieved as negatives.

**Decision:** select a global threshold using tuning macro F₀.₅; empty predictions are allowed and no match is forced. The measured pilot selected 0.74. Outputs are classification scores without a probability-calibration claim.

## 5. Results and Error Analysis

| Metric | Improved pilot |
|---|---:|
| Held-out macro F₀.₅ | 0.9267946 |
| Link precision | 0.9758643 |
| Link recall | 0.8717949 |
| Retrieval recall | 0.9662005 |
| Oracle macro F₀.₅ | 0.9871680 |
| Mean candidates per S1 | 40 |
| Holdout US macro F₀.₅ | 0.9387401 |
| Holdout India macro F₀.₅ | 0.9069702 |

Entity-bootstrap 95% interval: 0.9107670–0.9412080. Two of 17 held-out singletons received a false match; this sample is too small for a stable singleton estimate. Larger training and evaluation are in progress.

Errors include branch-name ambiguity, missing addresses, severe name changes absent from the alias channel, and true links removed by candidate caps or the matching threshold. The retrieval ceiling substantially exceeds achieved matching quality, so classifier coverage is a priority.

These results use 2,000 classifier-training S1 entities, 500 tuning, and 500 holdout, against the complete training target pool. The name channel is fitted separately on 1,544,990 eligible training-fold names and 5,348,430 matched targets. A pilot comparison changed several components jointly and is not a clean component ablation.

Stable whole-file hashing creates 70/15/15 S1 folds; targets follow their owners. Sampling is row-order independent, not exact stratification. Pilot holdout inspection during development makes these exploratory results. A fresh larger audit should follow configuration freezing. No French accuracy, private leaderboard performance, or final competition rank is inferred.

## 6. Release Status

The offline training and batched inference code are runnable. The seven-file integrity audit and synthetic metric, Unicode, alias-isolation, retrieval, and output-contract checks pass. Full official-test prediction, final candidate-budget selection, complete output validation, and the final archive remain pending.

## Appendix A. Code and Reproduction

Source is under `code/business_entity_resolution/src/`. `run_training.py` fits/evaluates the pipeline; `run_inference.py` builds complete test indexes and exports both required TSVs. The package README provides exact commands and pinned dependencies.

Run both `utils/validate_submission.py` and the unmodified organizer helper `utils/organizer_validate_submission.py`; include `--check-ids` for the official helper. Limited inference is explicitly marked unsafe for submission.

Large raw files and local artifacts remain out of Git. New artifacts record split IDs, feature version, search configuration, parameters, threshold, source hashes, input/index fingerprints, and the generated model license. Final output files must be identical to the files used for the portal and archive.

## Appendix B. Fair Play and Licenses

All identity evidence comes from the provided dataset. No external business registry, geocoding, identity API, external augmentation, or hosted model call is used. No pretrained neural model is used. LightGBM is MIT-licensed; generated tree and name-channel artifacts carry the project's model-artifact MIT notice. Each future added model must independently satisfy the organizer's license, size, offline, and data-use restrictions.
