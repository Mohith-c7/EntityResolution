# Sprint status

Updated: 2026-09-27 11:16 UTC

- Research deadline: 2026-09-27 13:07:34 UTC (18:37 IST).
- Model-search cutoff: 11:22:34 UTC; candidate freeze: 11:37:34 UTC.
- Submission 04: complete and preserved on the Mac and both VMs. Strict and official validators passed with ID checks, independently on the Mac and VM1. The VM2 copy matches all 13,938 sealed files.
- Baseline: 1,732,544 references; 69,301,760 scored pairs; 5,772,161 links; 98,839 empty predictions. User-reported portal score: 0.968.
- Best existing fresh audit: 0.9785469758. Best new selection20k result: 0.9800089319. Fresh confirmation and audit remain unopened.

| Work | Execution | State |
|---|---|---|
| Exact frozen replay | VM2, 250 references / 10,000 pairs | Pair order and all probabilities exactly reproduced |
| Current development workbench | VM2 | Complete and sealed: 50,000 references / 2 million keyed pairs |
| Training workbench | VM2 | Complete and sealed: 50,000 references with owner-excluded first-stage scores; retrieval ranker itself is not cross-fitted |
| Full heldout claimant universe | Mac and both VMs | Complete and sealed: 331,012 references / 13,240,480 pairs. Incumbent final predictions preserved; labels unopened |
| Gate | VM2 | All six V3 settings preserve development decisions. V4 rejected: 18 accepted true links removed and no accepted false links removed. French global claim diagnostics complete; no measured French accuracy |
| Learned sibling matcher | VM2 | Both families stopped. Initial grid regressed. Anchored pilot adds no links and only recalibrates downward; no promotion |
| Missing-address retrieval | VM1 | First 5k probe failed (+0.000201 candidate oracle). Diagnosed true links lost before fused shortlist. Corrected upstream quota probe added +0.001196 oracle score but only +0.000143 actual score, with lower precision and one extra singleton error; rejected |
| Fuzzy reverse Source 1 evidence | VM1 index; VM2 adapter | Complete, independently verified. Early10k pilot gain +0.000408; both countries improve, precision improves, recall decreases. Below predeclared +0.0005 gate; selected20k not inspected |
| Confirmation and final audit | Separate reserved 10k cohorts | Exclusion ledger completed across both VMs and Mac; labels unopened |
| Baseline/test inputs for score reuse | VM2 isolated experiment root | Full copy sealed and verified; original output remains unchanged |

The independent learned sibling comparison uses a complete 30,000-reference claimant graph, evaluating the same 20,000 selection references. Its control is 0.9794740787. Blends 0.50, 0.75 and 1.00 score 0.9792936386, 0.9792900391 and 0.9792541551. None is promoted. These are development results, not a new audit or portal score.

The later anchored pilot's early10k tuning gain is +0.000074, with paired 95% interval [-0.000228, +0.000386]. It adds zero true links and zero false links, removes 85 true and 33 false links, and lowers India performance. It is stopped; its selection20k result has not been inspected.

Complete heldout incumbent predictions contain 1,094,049 links and 19,692 empty references. Ownership v2 changes five references, all outside the reserved fresh confirmation and audit cohorts; this mechanical change does not establish a score improvement. Neither fresh cohort's labels have been opened.

User-reported current qualification cutoff: portal macro F0.5 0.9858. The user confirmed submission_04 portal score 0.968. A verified identical upload copy is available at `/Users/sriharsha/Downloads/submission_04/matching_results.tsv`.

The corrected ownership policy made no prediction changes in the completed 50,000-reference development comparison. The conservative gate has no measured development gain. Natural-negative stress tests, French output monitors and perfect-correction bounds do not establish accuracy improvements.

All changes use the supplied records only. Test pseudo-label training and external identity lookup are excluded. Full-test candidate lists must continue to contain exactly the pairs scored by the frozen matching pipeline.

All sub-agents are interrupted at user request. Parent continues directly. Both VM feature/diagnostic jobs are complete. A bounded multilingual neural pilot now trains on the Mac MPS GPU: 120,000 supplied outer-training pairs, 20,000 owners, one epoch, batch32, total token length128, MIT Microsoft MiniLM117.7M parameters. Runtime cap40minutes; no development/audit labels in neural training. No accuracy result yet.

2026-09-27T10:44:56.408619+00:00: Full encoder fine-tuning stopped at its40minute cap after43,392/120,000pairs, with no completed checkpoint or evaluation; rejected. The independently declared frozen-encoder alternative completed in116.85seconds: all120,000pair embeddings, then fiveepochs of a1536→256→64→2 head. Keyed residual7,389 and development11,144scores completed in7.90/11.27seconds. Fixed37input anchored calibration/early10k evaluation now runs onVM2. No sub-agents restarted. No fresh confirmation/audit labels opened.

Portal04 confirmed by user at 2026-09-27T10:59:58.458358+00:00:0.968, +0.0281 versus03. New neural calibration v2 selection20k:0.9800089319 versus0.9794740787, paired gain+0.0005348532 (95%CI+0.0002972 to+0.0007895); both known countries improve, precision improves and true links+4, singletonFP unchanged34. This is tuning evidence, not a fresh audit or portal score. Street-collision augmented head completed; keyed score/evaluation follows.

The same-name, different-street augmented frozen encoder scores 0.9799177549 on selection20k (gain +0.0004436762, paired 95% CI +0.0001611 to +0.0007326). It is below neural calibration v2. A bounded last-two-layer cross-encoder completed training in 223.05 seconds on 60,000 training pairs; its keyed inference completed in 6.88 and 10.18 seconds. Fixed calibration and independent early10k evaluation follow. No fresh labels opened and no new full submission started.

11:25 UTC: Model search is closed. Last-two-layer cross-encoder early10k gain +0.00041756 failed the declared +0.0005 pilot gate, so its selected20k labels remain uninspected. Neural v2 (selection20k 0.9800089319) is preselected for fresh confirmation, with model/head/source hashes recorded. Exact existing reverse feature generation runs across the complete 331,012-reference heldout graph with one global route and 16 worker processes. All agents remain stopped. The 82 validation, routing and sibling tests pass. No new full submission has started.

11:39 UTC: Neural v2 frozen at 11:39:00 UTC, 86 seconds after the planned operational freeze; no model search or fresh-label selection occurred during the delay. Neural inference on all123,600 routed heldout pairs completed in118.17seconds. VM1 lacked the parent frozen descriptor, so the first prediction-export attempt stopped before creating predictions or opening fresh labels. Its completed feature/score tables are preserved; the exact descriptor is being copied and prediction export retried with unchanged code/model/routing/thresholds. Six new sparse-scatter/anchor/rank tests and26 existing release tests pass.

Fresh confirmation10k: incumbent0.9789314125, neuralv2 0.9794958377, paired gain+0.0005644251, 95%CI [+0.0001409501,+0.0010283485]. Both countries improve; microprecision0.995753→0.995907 and recall0.954075→0.954860; true links+27, false links−5, singletonFP16→15, newlyempty non-singletons0. Confirmation quality improvement passes. Runtime/promotion is not yet passed; final audit10k remains unopened. Portal04 remains0.968; no new submission started.
