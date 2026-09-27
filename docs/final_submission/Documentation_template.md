# ML Challenge 2026: Business Entity Resolution

**Team:** EntityResolution  
**Members:** Mohith, Sanhitha, Harsha, Sahasra  
**Date:** 27 September 2026  
**Selected system:** Frozen bridge pipeline with a four-layer neural continuation
and an anchored 38-feature residual adapter.

## 1. Executive Summary

The system retrieves at most 20 candidates from each complete target source,
scores the resulting candidates with two tree models, and applies additional
reverse-corpus and offline neural evidence to a fixed sparse route. The selected
continuation updates only the final four encoder layers, pooler, and classifier
of a supplied multilingual MiniLM checkpoint. A 38-input LightGBM adapter adjusts
the frozen second-stage score once. All final decisions use the complete Source 1
claim graph; no candidate is removed from the candidate export by a threshold.

On the already exposed 20,000-owner selection cohort, with a complete 30,000-owner
decision graph, the four-layer system achieves macro F₀.₅ .9802609267 versus
.9794740787 for the original control. The paired gain is .0007868480 with 95%
interval [.0005175642,.0010832995]. These are development results. They do not
establish French quality, full-test ownership quality, or a leaderboard score.
After the selected candidate and complete 331,012-owner prediction graph were
frozen, the single-use final 10,000-owner audit achieved macro F₀.₅
.9799622272 versus .9793161919 for Submission 04. The paired gain was
.0006460352, with 95% interval [.0002035806, .0010892613]. Link precision
rose from .9952899973 to .9958352196 and recall from .9544283274 to
.9553548163. Singleton false predictions stayed at 12 of 528. These US/India
audit results do not establish French or leaderboard performance. The release
archive includes the validated output report and sealed package verification.
The audited candidate freeze SHA-256 is
`eee9067df380a21cf3d87ace97b55aeb2c84ab50045ec00b3fde892d10cf8982`.

## 2. Methodology

### 2.1 Problem Analysis

Training sources contain 2,206,821 Source 1, 5,034,616 Source 2, and 5,285,603 Source 3
records. The test contains 1,732,544 Source 1, 4,887,273 Source 2, and 5,082,316 Source 3
records. Supplied ground truth contains 7,638,365 links and 123,247 training
singletons. Source IDs are disjoint between train and test. Strings and explicit
tab delimiters preserve leading zeros and Unicode.

Training countries are the US and India. The test includes 259,452 French Source 1
records, or 14.9752%, without French training labels. Missing target addresses and
similar names across business branches are important sources of ambiguity.

### 2.2 Solution Strategy

Candidate retrieval uses label-free target indices plus learned training-only
name aliases and a trained cheap reranker. Primary normalization preserves
Unicode and numeric evidence; accent-folded, legal-form-free, phonetic, and
collapsed-name views supply alternatives. Ambiguous address abbreviations are
preserved. No external entity lookup or identity augmentation is used.

Pair classification is followed by candidate-context and Source 1 competition
evidence. A bounded reverse query against the complete Source 1 corpus provides
numeric summaries of other plausible claimants. Rival IDs do not enter learned
features. The query excludes the current Source 1 owner and retains explicit
missing/saturated evidence rather than treating absent results as uniqueness.

## 3. Candidate Generation

Complete SQLite FTS 5 target indices support rare-name/address terms, numeric
tokens, exact names and name cores, prefixes, postcode/name combinations,
numeric/location conjunctions, and learned alias families. The selected search
keeps 20 candidates per target source, path cap 80, at most 6 query tokens, and
maximum name/address term DF 100,000. Character retrieval is disabled.

The bridge pass uses the saved cheap tree ranker with a .99 seed cutoff and at
most two bridge seeds. Deterministic reranking preserves the selected candidate
budget. Every forward candidate receives first- and second-stage scores before
any final acceptance rule. Candidate membership and order remain unchanged by
the neural adapter. The official test candidate graph contains 69,301,760 pairs,
40 for each 1,732,544 Source 1 owner.

The neural route is computed once on the complete forward graph. It is capped at
20% of references and at most four uncertain candidates per routed reference,
uses candidates between .03 and .995 near the .83 decision region, and permits
up to two strong first-stage seeds at .90. The fixed test route contains 801,380
pairs. Auxiliary evidence is scored for that exact route; it does not add new
forward candidates.

## 4. Matching Model

### Forward models

The frozen first stage uses 65 registered pair features. The second stage adds
within-candidate context and complete Source 1 key-competition counts. Both are
LightGBM tree models. Saved native feature names/order, model hashes, alias
hashes, and source hashes are checked before inference.

### Offline neural evidence

The base model is `microsoft/Multilingual-MiniLM-L12-H384`, with its supplied
MIT license and pinned revision recorded in the included origin manifest.
The binary-classification checkpoint has 117,654,530 parameters. Input text is
the supplied name, country, and address, serialized identically for both sides
of a candidate pair. Tokenization uses longest-first truncation and maximum
length 128. No remote inference, downloads, or external business evidence occur.

One branch freezes the encoder and computes a 1536-dimensional pair vector from
left/right pooled representations, their absolute difference and product. A
trained 1536→256→64→2 head supplies the original neural probability. The second
branch starts from the sealed three-epoch two-layer checkpoint and continues
one epoch on the same 120,000 outer-training pairs, unfreezing the final four
encoder layers plus pooler and classifier. Each neural branch's supervised
owners are the same sealed 20,000-owner outer-training population.

### Anchored 38-feature adapter

The adapter receives 33 numeric reverse-evidence features, frozen first-stage
score, frozen second-stage score, rank within all 40 original candidates,
original frozen-head probability, and four-layer continuation probability.
Entity IDs, raw rival IDs, label ownership and folds are excluded from inputs.
The native tree adapter uses 103 selected boosting rounds. It predicts an
additive margin, applied as `sigmoid(logit(frozen_p2) + correction)` on routed
pairs only. A zero correction preserves the original score exactly. Every
unrouted score remains unchanged. There is no direct-probability substitution
or additional calibration shift in the selected release.

### Decisions

The unchanged control decision first rejects a high-band claim only when
another Source 1 owner claims that target with a strictly higher score at the
.83 competition threshold. It then recomputes rank one after those losses.
Ordinary acceptance uses .8300000000000004; a rank-one rescue uses
.6999999999999998. Empty outputs are allowed. A post-selection exclusivity
helper was evaluated separately but is not the selected release policy.

## 5. Results and Error Analysis

| Exposed selection 20k, complete 30k claimant graph | Original control | Four-layer system |
|---|---:|---:|
| Macro F₀.₅ | .9794740787 | .9802609267 |
| Link precision | .9958844623 | .9966268851 |
| Link recall | .9520083313 | .9530063497 |
| Singleton false predictions /1130 | 34 | 34 |
| US macro F₀.₅ | .9829811927 | .9837388904 |
| India macro F₀.₅ | .9741704553 | .9750013858 |

Relative to the control, the selected development run adds 205 true and 21 false
links and removes 136 true and 70 false links. One previously nonempty
non-singleton becomes empty. Candidate collisions on this development cohort
are zero for both systems. These exposed development measurements support
selection only; the reserved final audit is managed independently by a
single-use, frozen-candidate evaluation procedure.

| Reserved final audit: 10,000 owners, complete 331,012-owner graph | Submission 04 control | Four-layer system |
|---|---:|---:|
| Macro F₀.₅ | .9793161919 | .9799622272 |
| Link precision | .9952899973 | .9958352196 |
| Link recall | .9544283274 | .9553548163 |
| Singleton false predictions /528 | 12 | 12 |
| US macro F₀.₅ | .9813752782 | .9821165891 |
| India macro F₀.₅ | .9761544713 | .9766542110 |

The audit added 100 true and 17 false links and removed 68 true and 35 false
links. Neither system produced duplicate target claims within this audit graph.
No audit-driven parameter or policy change was made.

The tree adapter fits only the declared 20,000 residual-training owners, with
10,000 separate early-stopping owners and 20,000 separate selection owners.
Neural fitting uses only the sealed outer-training pairs. Model-fitted owners
are excluded from heldout evaluation, and all rival claimants remain in the
declared complete 331,012-owner heldout graph. Audit 10k truth is opened only after
candidate, predictions, and measured runtime evidence have been frozen.

The optional missing-address retrieval quota was rejected after a clean 2,952-owner
experiment: native neural_v 2 F₀.₅ fell by .0002855718, precision fell, and singleton
false predictions increased. It is not included. Alternative direct calibration,
peer-neural evidence, and preselection ownership changes are also not part of the
selected four-layer system.

## 6. Release and Verification

The archive contains exactly named complete TSVs under `output/`, all inference
source under `code/business_entity_resolution/src/`, offline model weights,
pinned requirements, and this completed method document. A separate release freeze links to the audited freeze and records a correction
to export validation: peer triplets are validated before taking their distinct
scored pair keys. This accommodates multiple peer edges for one scored pair.
The six scoring functions, global decision call, weights, thresholds and
routing policy remain unchanged, with source comparison proof and the original
audited exporter included. No further audit evaluation or tuning follows this
validation correction. The final output report
records input/output hashes, candidate counts, frozen candidate hash, stages,
and strict/official validator results. Submission 04 is preserved separately.

Package verification uses a sealed exposed runtime subset of 100 owners,
4,000 original candidates, and 221 neural-route rows. It executes the same
38-feature assembly, native adapter, keyed scatter, and original decision
function, checking reference probabilities and selection masks. The fixture
contains only keyed numeric evidence, no raw business records or truth.
This is an actual subset check, not a claim that cold full-test reproduction was
executed during package preparation. No leaderboard performance is inferred.

The sealed full-graph runtime projection is 8,768.913 seconds (146.15 minutes),
including 15% allowance, and passes the declared three-hour runtime gate. It is
an extrapolation from measured execution blocks, not a measured cold package
reproduction. The final runtime evidence is
`research/final_2h/neural_last4_runtime_final_v2/runtime.json`.

## Appendix A. Code and Reproduction

`code/business_entity_resolution/src/run_final_submission.py` verifies packaged
assets, runs the actual numeric subset, and reproduces inference from the three
provided external test TSVs. It rebuilds target/Source 1 indices rather than
bundling raw datasets. Existing module bytes and relative layout are preserved
inside `src/final_pipeline/` so native feature/source provenance remains valid.
The README provides commands for separate CPU and MPS environments and stage
transfer. Both strict and unmodified organizer validators check target IDs.

CPU indexing/features/tree inference use Python 3.12 with LightGBM 4.3.0.
The measured neural environment is Python 3.13.12, PyTorch 2.14.0 and
Transformers 4.57.6 on Apple Silicon MPS. All neural model/tokenizer loads use
local files. Index builds use SQLite FTS 5, trigram tokenization, complete source
coverage, and the included MIT-licensed native postings extension.

## Appendix B. Fair Play and Licenses

All entity evidence comes from organizer-provided data. No external business
registry, geocoding, identity API, external entity augmentation, hosted inference,
or test pseudo-labeling is used. The pretrained multilingual MiniLM base model
and its continuation remain below the stated model-size limit and carry the
included MIT model license. LightGBM is MIT-licensed. Learned tree and alias
artifacts include the project's MIT artifact notice. Pretrained origin and
fine-tuning lineage hashes are recorded in the packaged model manifests.
