# Model improvement after the first portal result

## Target and submission restriction

The target is **0.99 macro F0.5**. Submission generation is paused at the user's request. Work in this round is limited to training, retrieval experiments, evaluation and documentation. Preserve `output/submission_03/` and the published output files. Do not create another submission while the target is unmet.

A local 0.99 result would not establish a portal 0.99 result. Freeze the chosen pipeline before evaluating a new entity-level audit; review the result before considering another submission. The earlier 5,000- and 10,000-entity audits are consumed and must not be reused for selection.

## Starting evidence

- First portal result: **0.9399**, reported by the team. The discussion associates it with submission 03; the upload hash has not been independently checked against the portal.
- Submitted pipeline's tuning result: **0.959411**. The observed difference is **0.019511**, about 1.95 percentage points. It is not a reliable conversion factor for future models.
- Larger matcher: **0.965888** on a fresh 10,000-entity audit, versus **0.957786** for the old matcher on those exact entities. Paired gain: **0.008103**, 95% interval **[0.006429, 0.009761]**. This model has not been submitted.
- Larger matcher's full 50,000-entity tuning result: **0.966662**. Candidate oracle on that set: **0.988912**. Its current retrieval therefore cannot reach 0.99 on that set, even with perfect matching.
- France has no supplied labels. Its contribution to the portal gap is a hypothesis, not a measured country score. Match-count differences alone cannot establish correctness.

## Research translated into experiments

**Evaluate blocking separately from matching.** Sparkly studies scalable top-k lexical blocking; DIAL highlights that blocking and matching need different training objectives. Measure candidate recall, oracle macro F0.5, candidate count and runtime alongside actual matching accuracy. A higher oracle alone does not establish a better pipeline. [Sparkly](https://doi.org/10.14778/3583140.3583163), [DIAL](https://research.ibm.com/publications/deep-indexed-active-learning-for-matching-heterogeneous-entity-representations).

**Treat generalization as a separate experiment.** Published blocking evaluations examine both in-distribution and out-of-distribution behavior. Plan leave-one-training-country-out experiments with aliases fitted only on the permitted training countries and a label-free shared record index. Do not hard-code a test-country vocabulary or force France's cardinality distribution to resemble another country. [NAACL reproducibility study](https://aclanthology.org/2024.naacl-long.483/).

**Use a bounded multilingual matcher pilot if lexical/context features plateau.** Ditto demonstrates fine-tuning pretrained language models for pair classification with hard examples. Any local pilot would use only organizer records and labels for fine-tuning, retain entity-disjoint validation, and benchmark an uncertain-pair inference budget before integration. XLM-R base is listed as MIT and roughly 0.3B parameters; multilingual DistilBERT is listed as Apache-2.0. These are researched options, not downloaded or trained components of the current pipeline. [Ditto](https://arxiv.org/abs/2004.00584), [XLM-R model card](https://huggingface.co/FacebookAI/xlm-roberta-base), [DistilBERT model card](https://huggingface.co/distilbert/distilbert-base-multilingual-cased).

**Respect the challenge's one-to-many structure.** Research on one-to-one ER assumes both sources are deduplicated. That assumption does not fit this challenge: S1 can have many targets. A future ownership layer may constrain each target to at most one S1, while leaving the number of targets per S1 unrestricted. Evaluate it with competing references present and predictions produced without training on those identities. [Matching-algorithm analysis](https://research.ibm.com/publications/an-analysis-of-one-to-one-matching-algorithms-for-entity-resolution).

**Optimize the actual metric, but test probability assumptions.** F-measure decision research motivates choosing a prediction set per reference. The implemented prototype assumes independent calibrated candidate labels, unlike the general algorithms that model dependencies. It uses numerical integration of expected F0.5, includes the empty-set utility, and optionally models missing links with a Poisson count. Unit tests agree with exhaustive enumeration on small cases. This is a tested approximation to reality, not a claim of Bayes optimality for these records. [F-measure maximization](https://proceedings.mlr.press/v28/dembczynski13.pdf).

## Experiments completed

### Candidate ranking and bridge expansion

All rows below use the same 10,000 previously inspected tuning references and the frozen 300k matcher. The final candidate budget remains 20 per target source, 40 per S1.

| Blocking configuration | Oracle macro F0.5 | Matcher F0.5 at 0.75 | Recovered / lost true candidates versus baseline |
|---|---:|---:|---:|
| Existing shortlist and hand-weighted reranking | 0.988514 | 0.965922 | — |
| Wider shortlist, same reranker | 0.988512 | 0.966135 | 88 / 68 |
| Wider shortlist, learned ranker | 0.992332 | 0.966633 | 490 / 16 |
| Wider shortlist, learned ranker, one-hop bridges | 0.993052 | 0.967066 | 594 / 18 |

Bridge seeds come from the blocking ranker, not from accepted final matches. No seed is automatically declared a match. Additional retrieval results are compared against the original S1, deduplicated and narrowed to 20 per source before the final matcher scores them. A seed's exact-name provenance is never presented as exact-name evidence about S1. Runtime needs a separate full-budget benchmark before any eventual submission.

The first learned-ranker run finished scoring but failed during optional baseline report loading. All expected chunks were checked for entity coverage, pair uniqueness, feature finiteness and fixed-budget coverage before recovering the cache. The benchmark now supports a missing optional saved baseline without discarding scored results.

### Candidate-context model

Use first-stage predictions that are out of sample with respect to first-stage training labels. Divide the existing 50,000 development references into 20,000 second-stage training, 10,000 early-stopping and 20,000 selection references. The previous audit does not enter this experiment.

Add 20 features: candidate probability/rank, confidence of other candidates, agreement with up to four other high-confidence candidates, cross-source support, and name comparisons after removing words shared with each record's own address. A candidate cannot support itself. Fit a second LightGBM model, then select a probability blend and threshold.

On the 20,000 selection references:

| Metric | Frozen first stage | Context blend |
|---|---:|---:|
| Macro F0.5 | 0.967178 | 0.969199 |
| Link precision | 0.989505 | 0.992367 |
| Link recall | 0.928736 | 0.930862 |
| Singleton false predictions | 60 | 28 |
| Non-singleton empty predictions | 109 | 152 |

Selected blend: 75% context model, 25% first stage; threshold 0.73. The additional missed non-singletons are a remaining weakness. The paired interval on selection data is exploratory because that data selected the rule. A new audit is required before treating this as a verified gain.

### Expected-F0.5 decision rule

Rejected for now. Its best raw macro result was 0.969567 versus 0.969199 for the context threshold, but singleton errors rose from 28 to 69. Isotonic-calibrated variants also increased singleton errors. None passed the acceptance constraints. The prototype remains available for reproducibility and is not part of the selected pipeline.

## Work in progress and next decisions

The combined experiment completed on the same 20,000 context selection references, excluding all second-stage training and early-stop identities. At the previous context threshold it scores 0.970965, with 29 singleton errors. The conservative selection at threshold 0.76 retains 28 singleton errors and scores **0.970635**, with precision **0.992710** and recall **0.933639**. Its candidate oracle is **0.993451**. This is development evidence only, not a new audited best or a portal score.

The selective-bridge probe completed: only expanding materially different names or substantially fuller addresses reduced search calls from 44,244 to 27,982 over 10,000 references. Oracle F0.5 fell from 0.993052 to 0.992547; frozen first-stage matching fell from 0.967066 to 0.966760. This establishes a quality/cost tradeoff, not a chosen replacement. Timings were approximately 101 versus 84 seconds with six workers and should not be extrapolated to a full run without a dedicated benchmark.

1. The 300k first stage reached its 1,200-tree limit without early stopping. An isolated 2,400-tree-cap comparison is now training on the same cached 300k/50k data, with the existing 300k model as baseline. Its source snapshot and log are under `models/next_round/extended_training_job/`; output goes to `models/next_round/extended_model/`. This job only trains and selects on tuning. It does not evaluate an audit or generate submissions.
2. Analyze remaining rejected matches and false merges on that selection set. Retain changes only when gains survive precision and singleton checks.
3. Prioritize missing-address and Indian-script retrieval, independent name evidence, and country-transfer evaluation. Test a small offline multilingual pair classifier if the inexpensive evidence additions plateau.
4. Validate target ownership with all relevant competing references present before applying it. Per-reference samples do not establish its benefit.
5. Once a materially improved pipeline is frozen, reserve a new audit and evaluate once. Keep local development, fresh audit and portal numbers separate. No new submission generation in this round.

## Reproduction and artifacts

- `scripts/benchmark_postings.py`: bounded, learned-ranking and bridge retrieval experiments.
- `scripts/evaluate_retrieval_cache.py`: same-matcher, same-reference comparisons.
- `scripts/build_context_features.py` and `scripts/train_context_model.py`: second-stage evidence and fitting.
- `scripts/probe_expected_f05.py`: rejected per-reference decision experiment.
- `scripts/evaluate_context_retrieval.py`: combined retrieval/context evaluation.
- `reports/experiments/round2/`: aggregate reports and the user-reported portal result.
- `models/next_round/`: ignored caches, models, selection IDs and detailed predictions.

All identity evidence comes from the organizer's files. Research pages supply methods and model documentation only; no external business lookup or augmentation is used.
