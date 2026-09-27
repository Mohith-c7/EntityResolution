# Entity Resolution: 0.985 and 0.99 Experiment Plan

Date: 27 September 2026

## Decision

Use one separate GPU machine for a bounded multilingual matcher experiment. Keep the existing Azure submission and audit jobs unchanged. Prepare the data and executable experiment before renting the GPU. More compute increases experiment throughput; a score improvement must still be demonstrated.

## Verified starting point

| Measurement | Score | Interpretation |
| --- | ---: | --- |
| Submission 03 portal | 0.9399 | Reported leaderboard result |
| Current bridge pipeline, final decision rules | 0.979820 | 20,000-reference development selection result; not a fresh audit or portal score |
| Bridge candidate oracle | 0.993451 | Score with perfect decisions within the retained candidate set; not model accuracy |
| India candidate oracle | 0.988462 | Retrieval remains a material bottleneck on this slice |

Reaching 0.985 requires +0.005180, about 26% of the remaining development loss. Reaching 0.99 requires +0.010180, about 50%. These gains cannot be promised from hardware or architecture alone.

Evidence comes from `reports/experiments/round3/bridge_stacked_l255s42.json`, `bridge_decision_rules_exclusive_stacked_l255s42.json`, and the corresponding model reports. The existing loss breakdown describes the threshold-only result, 0.979077, rather than the final 0.979820 decisions. Recompute that breakdown before allocating effort by error category.

France has no supplied labels. Its exact contribution to the portal gap is unknown; estimates based on country proportions are conditional explanations, not measurements. Matching its predicted link-count distribution to India or the US does not establish correctness.

## Machine recommendation and budget

Budget: $50–$100, with at most six billed experiment hours.

Preferred short experiment: **one H100 PCIe, 80 GB**, if immediately available. A single A100 is an adequate fallback for this model size. No multi-GPU training is needed for the initial test.

| Public single-GPU option | Listed hourly compute price | Six-hour compute cost |
| --- | ---: | ---: |
| Lambda H100 PCIe, 80 GB | $3.29 | $19.74 |
| Lambda H100 SXM, 80 GB | $4.29 | $25.74 |
| Lambda A100, 40 GB | $1.99 | $11.94 |

These are public list prices checked on 27 September, before tax and any separately billed resources. They are not allocation guarantees. Verify the final account quote and capacity before provisioning. A working account and payment setup have not been verified. Source: [Lambda instances and pricing](https://lambda.ai/instances).

Azure checks found zero T4, A100 and H100 family quotas in Central India, West Europe and East US 2. Central India also has 64 of 65 regional vCPUs already allocated, and its NC24ads_A100_v4 SKU reports `NotAvailableForSubscription`. Azure is therefore not an immediately deployable GPU option under the checked quotas. The retail pricing request was rate-limited, so no Azure price is quoted. The [Azure A100 specification](https://learn.microsoft.com/en-us/azure/virtual-machines/sizes/gpu-accelerated/nca100v4-series) confirms that NC24ads_A100_v4 contains one 80 GB A100.

Reserve up to $40 for a first six-hour session, subject to the final quote; retain the remaining budget for a justified follow-up. Enforce the time limit, copy checkpoints and results to durable storage, then terminate the experimental resource. No new cloud resource has been provisioned.

## Experiment 1: complementary evidence for difficult pairs

Train a multilingual cross-encoder: one model reads both business records together and predicts whether they describe the same business. This is an established entity-matching approach, but published benchmark gains do not predict this challenge's macro F0.5. See [Ditto](https://arxiv.org/abs/2004.00584).

Candidate models:

- [XLM-R base](https://huggingface.co/FacebookAI/xlm-roberta-base): MIT; roughly 0.3 billion parameters. Primary quality experiment.
- [Multilingual DistilBERT](https://huggingface.co/distilbert/distilbert-base-multilingual-cased): Apache 2.0; 134 million parameters. Smaller comparison if the first benchmark leaves time.

Pin the exact model revision, tokenizer, license and installed dependencies. Run downloaded weights locally on the rented machine. Fine-tune only on the official challenge data. Do not use hosted inference APIs, external business records or identity lookups.

### Training design

1. Keep each S1 identity and all its labeled variants within one fold. Preserve the reserved audit set.
2. Start with 100,000–200,000 training pairs: rejected true links, convincing false matches and representative easy pairs. Mine difficulty using out-of-fold scores. Never mine from audit labels.
3. Serialize both records with explicit name, address and country fields. Preserve raw scripts, accents and numbers. Measure truncation before selecting a 192- or 256-token combined input limit.
4. Initial settings: PyTorch, binary classification head, learning rate 2e-5, effective batch 64, up to two epochs. Select the physical batch and mixed-precision mode after measuring memory and throughput. These are experiment settings, not established optima.
5. Evaluate on the complete development entity set, including singletons and links absent from retrieval. Pair accuracy is not the selection metric.
6. Combine the new score with tree evidence using held-out calibration or a fold-safe meta-model. Do not assume two probability scales are interchangeable.

### Route only pairs where the model can help

Measure several label-free routing rules: uncertainty bands, disputed ownership, and the top few candidates for empty predictions. Include suspicious accepted links as well as rejected links. Compute a perfect-correction bound on development data to determine how much error each route can reach.

At 40 candidates for each of 1,732,544 references, the upper estimate is 69.3 million pairs. Routing 1% means about 693,000 neural comparisons; 5% means about 3.47 million. Measure actual pairs per second, including tokenization and transfer, before accepting the design. Runtime is routed pair count divided by measured throughput, plus retrieval and decision overhead.

A routing diagnostic is in `research/score_targets_20260927/probe_routing.py`. It requires an export with pair IDs and both probabilities together. The older local cache does not align with the new probability arrays; the reproduction check rejected that combination. No routing gain has been measured yet. Export the correct development scores by pair ID before running it. Never align predictions using array length alone.

## Experiment 2: ownership and normalization

The current frozen decision code rejects strictly weaker high-threshold claims before applying the lower top-candidate threshold. Ties and rescued links can still produce multiple owners. Test deterministic ownership across all proposed links, including rescues, on complete competing cohorts with out-of-fold scores for every competitor. Keep this separate from the frozen submission.

A target may have one owner while an S1 may have many targets. Do not impose one-to-one matching. Keep a no-match option. Measure whether rival false claims steal correct matches.

Add alternate comparison features for dotted legal forms, accent-folded text and observed script variants. Preserve raw fields. Downweight common name words using training statistics rather than deleting words such as “services” universally. Protect genuine address numbers when experimenting with leetspeak handling.

## Experiment 3: retrieval ceiling and country transfer

Diagnose missed links by script, empty address, different name at the same address, number corruption and overly common keys. Test targeted additional keys and bridge queries, followed by measured ranking and adaptive candidate budgets. Aim for a development candidate oracle of at least 0.997 while reporting mean, p95 and maximum candidates per reference and runtime.

Use train-US/evaluate-India and the reverse as transfer stress tests, refitting supervised aliases and derived statistics within the training side. Also test common-building, city-in-name and missing-address slices. These tests cannot establish France accuracy, but they can reject brittle changes.

Keep candidate export faithful: it must include the exact candidates considered by the final matching system, including expanded candidates, and every final match must be a member. A small neural subset does not justify hiding candidates scored by the tree matcher.

## Six-hour experiment allocation

Prepare pair exports, splits, code and dependencies before the paid session.

| Paid time | Work and decision |
| --- | --- |
| 0–30 minutes | Environment smoke test; benchmark training and inference; verify GPU utilization and input alignment |
| 30–150 minutes | Primary matcher pilot, with checkpoints and early stopping |
| 150–240 minutes | Full development evaluation, calibration and routing/runtime analysis |
| 240–300 minutes | One justified ablation or smaller-model comparison if throughput permits |
| 300–360 minutes | Freeze the best candidate, export results and artifacts, terminate experiment resource |

This is a time budget, not a prediction that every experiment will finish. Scale the pilot according to the first benchmark. Stop unpromising training early rather than consume the entire allocation.

## Acceptance and reproducibility

- Record code/model hashes, data split IDs, seeds, model revision, feature schema and all thresholds.
- Compare against the exact final 0.979820 development pipeline on identical references.
- Report macro F0.5, paired entity-level confidence intervals, micro precision/recall, singleton errors, empty non-singletons and country slices.
- Treat a development gain of at least 0.002 with acceptable runtime as a useful pilot signal, not proof of the 0.985 target.
- Select on development data, then freeze and evaluate once on a newly reserved audit. The current submission audit must not become another tuning set.
- Verify full-pipeline candidate accounting and output parity before any future submission run.

Deliverables: measured experiment report, reproducible checkpoint, calibrated decision configuration, projected full-inference runtime based on measured throughput, and a clear retain/reject decision. A 0.99 claim requires measured evidence; leaderboard performance remains separate from local validation.
