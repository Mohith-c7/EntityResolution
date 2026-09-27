---
name: Entity Resolution — Six-Hour Master Sprint
date: 2026-09-27
status: Ready for execution approval
research_budget: 6 hours total, shared across all workstreams
submission_budget: 3 additional hours
target: 0.99 macro F0.5
fallback: Best verified complete submission
---

# Entity Resolution — Six-Hour Master Sprint

## 1. Objective and deadline

Use six hours to find and validate the strongest improvement we can reproduce, then reserve three hours for the complete submission, validation, transfer and upload. The target is **0.99 macro F0.5**. That score is an objective, not a promised result or a reason to miss the deadline.

This combines three plans:

1. The original evidence-led plan: reliable caches, learned sibling compatibility, targeted retrieval, ownership and a fresh audit.
2. The France research plan: shared-name ambiguity, street evidence, natural hard negatives and country-transfer checks.
3. The France gate sprint: reuse submission_04's stored scores for a fast, independently shippable decision-layer experiment.

Two main experiments run in parallel: **a conservative ambiguity/street gate** and **a learned sibling matcher**. Validation, submission preservation and runtime engineering support both. The 12-feature integration and retrieval probe have strict time limits and cannot delay the main experiments.

**Clock rule:** record one sprint start, `T0`, and absolute deadlines in IST and UTC. Research ends at `T0 + 6h`; the submission must be complete at `T0 + 9h`. Waiting for VM 1, recovering VM 2 access, transfers and failed trials all consume this same budget. If the clock has already started, use the remaining time; do not restart it.

**Research cutoff:** stop adding model variants at `T0 + 4h15m`. Freeze the selected candidate by `T0 + 4h30m`. The remaining 90 minutes are for the fresh audit, runtime checks and packaging. No tuning after audit labels are opened.

This document creates no jobs, changes no running submission and deletes no resources. It is the execution specification for the next approved work.

## 2. Starting evidence

| Evidence | Result | Interpretation |
| --- | ---: | --- |
| Submission 03 portal | 0.9399 | Reported leaderboard result; preserve the submitted file and its hash |
| Current development pipeline | 0.979820 | Selected on 20,000 development references |
| Current fresh audit | **0.978547** | 10,000 US/India references; this audit is now consumed |
| Audit precision / recall | 0.996726 / 0.949690 | Pair-level micro metrics; the scored metric is entity-level macro F0.5 |
| Audit India / US | 0.972883 / 0.982320 | France has no supplied labels |
| Development candidate oracle | 0.993451 | Perfect decisions within the current retained candidates |
| India development candidate oracle | 0.988462 | Retrieval limits this particular India cohort below 0.99 |
| Recent 12-feature trials | About +0.0001 to +0.0008 | Older, pre-bridge candidates; not verified gains on the current pipeline |
| Direct-evidence expert | +0.000223; interval crosses zero | Failed promotion; do not repeat the same global blend |

The audit gap to 0.99 is **+0.011453**, requiring removal of about **53.4% of remaining audit loss**. On the separate development cohort, reaching 0.99 would consume about 74.7% of the gap between the current model and its candidate oracle. Do not mix audit and development cohorts in headroom calculations.

Reported partial-test diagnostics suggest an important shared-name failure: France has more competing claims, and some high-confidence claims agree on a common name while disagreeing on the street. The 6.1% French multiple-claim figure and 3.69 links per reference must be recomputed on the complete artifact, with pre-decision and final claims reported separately.

France's proposed 0.856 score and submission_04's proposed 0.959 portal score are **conditional back-calculations**, not measurements. The full test mix is approximately 46.75% India, 38.27% US and 14.98% France; the public leaderboard subset may differ. One aggregate portal score cannot identify three country scores. No numerical France improvement is assumed in this schedule.

## 3. Machines, ownership and parallelism

Read-only snapshot at approximately **12:27–12:30 IST, 27 September**:

- **VM 1:** `er-cpu-01`, resource group `er-temp-rg`, 64 vCPUs and about 251 GiB RAM. Submission_04 was still running; the latest inspected log showed 1,150,000 of 1,732,544 references scored. Five recent intervals imply about 200 references/second. Roughly 48 minutes of scoring remained at that rate, plus finalization; this is a snapshot, not a completion guarantee.
- **VM 2:** `er-expcpu-0927-e70007`, resource group `er-expcpu-0927-e70007-rg`, 32 vCPUs and about 125 GiB RAM. Azure reports it running, but the latest SSH connection timed out. Resolve access before assigning a compute-dependent deadline. Do not assume it is stopped or start a replacement automatically.
- **Mac:** coordination, code, small fixtures, reports and durable copies. Avoid large jobs that compete with transfers or consume the only local artifact copy.

| Workstream | Primary owner | Deliverable | Compute |
| --- | --- | --- | --- |
| A — Evidence and submission preservation | Mohith | Pair-keyed workbench, split ledger, baseline replay, audit and validator reports | VM 1 after submission_04; light checks earlier |
| B — Ambiguity/street gate | Sanhitha; Harsha integrates decision logic | Auxiliary text views, signals, conservative gate and ownership experiment | VM 2, bounded CPU |
| C — Learned sibling matcher | Harsha | Routed target-to-target scorer and fold-safe integration | VM 2; one heavy training job at a time |
| D — Runtime and release | Mohith and Harsha | Measured run mode, frozen package, exact candidate export | VM 1 after release; VM 2 supports checks |
| E — Reporting and packaging | Sahasra | Experiment table, methodology updates, output inventory and archive check | Mac / light CPU |

Parallel development does not mean unlimited concurrent training. On VM 2, start with **20 threads for the heavy job, up to 8 for signals/data work, and 4 reserved**. Set LightGBM, BLAS, Arrow and process-pool limits explicitly. On VM 1, do not add heavy work while submission_04 runs. After it finishes, cap concurrent worker/thread totals around 56 and retain capacity for I/O and monitoring. Adjust from measured load rather than RAM availability alone.

Use isolated experiment directories and one integrator for shared code. Preserve the current dirty worktree; do not reset, clean or overwrite existing changes. Leave submission_04's frozen code, models and configuration untouched.

## 4. Six-hour schedule

| Shared time | Evidence / release lane | Gate lane | Model lane | Required checkpoint |
| --- | --- | --- | --- | --- |
| 0:00–0:30 | Record deadlines; check VM 2 access; inventory artifacts and prior exposure; preserve sub04 | Implement fixtures and auxiliary-view interface | Inspect surviving OOF models; define sibling route and required cache | Exact inputs, CPU allocation and fallback selected |
| 0:30–1:30 | Recover pair-keyed development workbench; reserve fresh confirmation and audit IDs; VM 1 remains DND until sub04 completes | Build name/street signals and real labeled collision cases | Recover minimum fold-safe inputs; measure route's perfect-correction headroom | Baseline parity; model-data readiness by 1:30 |
| 1:30–2:30 | Score declared heldout competitor universe after VM 1 releases; copy validated sub04 | Run fixed small gate grid; inspect true links harmed | Fit bounded sibling scorer; generate OOF support | Gate and sibling first comparisons; no broad search |
| 2:30–3:30 | One targeted retrieval diagnostic if capacity is free; prepare runtime sample | Test global ownership after normal and rescue claims | Integrate sibling support; optional one stage-two 12-feature ablation | Drop branches with insufficient gain or excessive runtime |
| 3:30–4:15 | Run one combination and one disjoint confirmation check | Finish selected gate and interpretation report | Finish selected matcher and inference path | Pick one candidate, or explicit baseline fallback |
| 4:15–4:30 | Lock code, models, rules, feature order, indexes, aliases and candidate lineage | No more parameter search | No more model search | Candidate frozen before opening audit labels |
| 4:30–5:30 | Evaluate candidate and sub04 control once on fresh audit; run country-weighted runtime benchmark | Inspect mechanical/normalization failures without retuning | Fix only reproducible implementation defects; changes require a new valid evaluation | Audit pass/fail and measured full-run ETA |
| 5:30–6:00 | Select execution mode; verify baseline fallback on Mac; package reports | Save selected signal configuration and hashes | Save all weights and exact dependency versions | Final operational freeze; research ends |

**Capacity failures shrink scope, not the deadline.** If VM 2 access is not restored within the first 20 minutes, continue fixtures and code locally, then use VM 1 only after sub04 releases it. Drop the heavy sibling and optional feature/retrieval branches if they cannot meet the same checkpoints. Do not stop submission_04 to recover experiment time.

## 5. Shared workbench and validation contract

### Recover what exists before rebuilding

The current bridge candidate/context caches were absent from the previously checked locations. Saved first-stage, second-stage and five OOF boosters survive. Existing sub04 chunk files contain pair IDs and scores, **not the complete feature matrix**.

1. Inventory valid archives on both machines and locally. Recover the current development pair table first.
2. Verify 250 references against the incumbent: exact candidate IDs and ordering, no missing/duplicate pairs, score differences at most `1e-6`, and identical selected links when the same claimant universe is available.
3. Reproduce the 0.979820 final development result where the original competing claims survive. Otherwise reproduce 0.979077 for threshold-only matching and explicitly mark final ownership replay incomplete.
4. A fresh cohort needs a freshly scored incumbent on that same cohort; it is not expected to equal the historical score.
5. Reuse saved OOF models to reconstruct probabilities. Do not retrain all five first-stage models merely because a cache is missing.

The minimum shared table contains S1 ID, target ID, country, source, candidate order, first-stage score, second-stage score, proposed/accepted state and retrieval provenance. Record feature availability explicitly. Training feature tables additionally contain the exact versioned pair/context features and OOF fold/model identity. Store ground truth separately from audit inputs.

Never join a probability array to a different candidate table by equal length or assumed row order. Use pair keys and a verified order hash. Cache reuse requires matching input, normalization, alias, index, native-library, feature-schema and model hashes.

**1h30m cutoff:** if correctly aligned current training inputs are unavailable, stop the dependent retraining branch. The stored-score gate can continue independently once complete sub04 chunks and a correctly scored labeled control exist. Misaligned old pre-bridge caches are not a substitute.

### Three data roles

- **Development:** the existing labeled 20k selection cohort can be used for the bounded search. Its score remains a development score.
- **Confirmation:** reserve 10k previously unexposed outer-holdout references for a single check of the selected combination. If this set is inspected and changes are made, it becomes development data; it does not remain fresh.
- **Final audit:** reserve a separate 10k previously unexposed references at the start. Keep labels sealed until the full pipeline is frozen. No retry with another threshold after reading the result.

Build the exclusion ledger from sampled-reference manifests **and** prior experiment/audit/training ID files on both VMs and the Mac. `models/**/sampled_references.json` alone may miss inspected entities. Exclude all members of a known entity together. Do not use raw random pair splits.

The incumbent and candidate must score the same audit references and candidate universe, except for an explicitly evaluated retrieval change. Their paired difference is the comparison; do not subtract the historical 0.978547 from a score on a different sample.

### Ownership evaluation

Score every reference in a declared heldout pseudo-test universe, rather than retrieving only exact-key rivals to a few selected claims. Prefer the available outer-holdout universe if its measured scoring time fits. Every competing claimant must be out of the fitted models' supervised training population; fold-safe aliases alone do not make in-sample matcher scores out of sample.

Use the same scored claimant universe for control and experiment. Completeness within this universe does not reproduce the full official test's density; record that limitation. If it cannot be scored within the budget, retain the original ownership policy for the promoted variant. Report gate-only or matcher-only evidence without claiming a validated global arbitration gain.

Audit predictions may be computed before labels are opened. Do not inspect audit labels, tune to their errors or fit any transformation with their supervised information.

## 6. Workstream B — Conservative ambiguity and street gate

### Auxiliary signals, not an index rebuild

Create a versioned signal module, proposed path `scripts/france_gate.py`. The file name identifies the motivating problem; the logic runs on all countries with open string labels. Preserve the frozen pipeline's original names, addresses, cores and retrieval configuration.

Build auxiliary views from supplied records only:

- Accent-folded names using the existing Latin-aware `accent_fold`; retain original scripts and Indic marks.
- Candidate affixes from edge position, frequency and observed name-variant relationships. Frequency alone does not prove a token is a legal form. Verify recovery of known training patterns and count cases where informative names become empty.
- A reduced-name view that downweights likely affixes and address-overlap tokens. Keep the original view alongside it. Empty reduced names never become a shared exact-match key.
- Per-country S1 counts for original and reduced names, with the current S1 excluded from rival counts. Preserve missingness separately from a count of zero.
- IDF-weighted address content and a cautious street-content view. Do not delete every common token or every token repeated in the name: those may be the real street name.
- Separate house-number, suffix and postcode evidence only where parsing is credible. Keep letter/suffix variants as evidence; do not treat the first digit sequence as a guaranteed house number.
- Pair contradiction, missing-address flags, candidate rank, number of competing claims, rival margin and relative street support.

Compare raw and reduced views explicitly. An automatically discovered token list is a hypothesis, not proof of language-independent semantic normalization.

### Labeled tests before a gate

Primary negatives come from different known training owners with shared names, locality or address evidence. Preserve positives with dropped house digits, missing address components, genuine shared buildings and noisy street names. Synthetic same-name/different-street examples are an additional stress test, not the source of truth for all address disagreement.

The conservative gate proposes rejection only when name ambiguity is high, both records contain sufficient reliable address evidence, and the evidence supplies a strong contradiction. Missing tokens or uncertain number extraction are not contradictions.

Predeclare at most six gate settings, such as three ambiguity cutoffs and two street-overlap cutoffs. Choose exact values after a label-free scale inspection, before evaluating the grid. Include the unchanged control. Keep original probabilities intact; record a separate veto and reason instead of overwriting the only saved score.

### Final ownership, including rescues

Do not assume prepending a gate to unchanged `decide()` provides full ownership enforcement. The current function arbitrates high-threshold claims, then applies the rank-one rescue; rescues and equal-score ties can escape exclusivity.

Evaluate ownership as a separate composition:

1. Form eligible ordinary and rank-one claims using the selected policy, respecting vetoes.
2. Collect all relevant competing claims, including the lower rescue band.
3. Evaluate owner evidence using score, score margin, street/number support and name ambiguity. Street agreement alone does not automatically win.
4. Accept at most one owner per target, with a no-owner option. Abstain on indistinguishable evidence or insufficient margin; sorting by ID may make processing deterministic but must not pretend to establish the correct owner.
5. Do not repeatedly rescue a new candidate after losing ownership. That creates an untested cascade.

Keep the original `decide()` available as the unchanged control. Implement new behavior in a separate versioned function, proposed path `scripts/decision_policy_v2.py`. A target can have at most one S1 owner under the audited training assumption; an S1 can still own many targets from either source.

### Gate acceptance

On the same fresh confirmation/audit entities as the control:

- Paired macro F0.5 lower confidence bound must exceed `-0.0005` for a gate-only non-inferiority claim.
- Point precision must not fall; singleton false predictions must not rise.
- Report country deltas, true/false links removed, newly empty non-singletons, and ownership winners changed. Flag either known-country point loss below `-0.0005` as a failure.
- A claimed quality improvement additionally needs a paired interval above zero. Non-inferiority alone does not establish improvement.

Remove the proposed “fewer than 80% of dropped links are true” rule. A global dropped-link fraction does not determine entity-level F0.5: removing a sole true match and removing one of ten true matches have different effects.

France monitors include normalization failures, empty auxiliary views, changed-link counts, pre/post-arbitration collisions, near ties, rank-one rescues and retained high-agreement record pairs. These are smoke tests, not French labels. A lower double-claim rate is mechanically expected from exclusivity and does not prove better ownership.

## 7. Workstream C — Learned sibling evidence

This is the main attempt to improve labeled recall beyond the existing gate. The current context model already has ordinary fuzzy peer similarities; the new evidence must come from a learned target-to-target compatibility function.

### Route and train

1. Use the incumbent's scores, missingness, script mismatch, peer support and ambiguity to define a label-free route. Begin with at most 20% of references, up to four difficult candidates and two supporting peers per routed reference.
2. Measure the route's perfect-correction macro gain on development. Stop if the accessible gain is below `+0.0025`; this is an opportunity ceiling, never an achieved score.
3. Build positive target-target pairs from shared training owners and hard negatives from different known owners in overlapping candidate/name/address neighborhoods. Include cross-source and same-source pairs.
4. Split by owner before pair construction. Select support peers from OOF incumbent predictions, including their mistakes. A candidate cannot support itself.
5. Fit a bounded tree or logistic compatibility model using raw/auxiliary name and address alignment, number relations, missingness, script/source information and distinctive-token contradiction.
6. Produce owner-excluded sibling-model predictions for training the integration layer. Training a sibling scorer on an owner and then using its in-sample score to train that owner's context model is not acceptable stacking.
7. Integrate support and contradiction into the current stage-two design, or a small fold-safe residual layer, with the first-stage model and retrieval frozen. Retain the original score as an explicit input/control.

At the proposed route cap, eight comparisons per routed reference imply at most approximately **2.77 million additional target-peer comparisons** over the full test set. This excludes record lookup, feature construction, context reconstruction and global ownership; all must be included in the runtime benchmark.

### Stop conditions

- No aligned bridge/OOF training inputs by 1h30m: stop this branch.
- No development gain of at least `+0.0005` by 3h30m, or precision/singleton guardrails fail: stop this branch.
- Insufficient complete inference implementation by 4h15m: do not freeze an experimental model that has no runnable submission path.
- Extra comparisons or feature reconstruction exceed the final runtime budget: retain the gate-only candidate.

A pilot can use 50k–100k owners for speed. Any claimed gain is measured against the full incumbent on identical evaluation references, not against an artificially weakened control. Scale the new component only if the input assets are ready and the measured remaining time fits; do not rebuild 300k references as an automatic ritual.

## 8. Bounded optional work

### Twelve additional features

One stage-two-only experiment is allowed if the current feature/context cache is ready. Keep first-stage OOF predictions and retrieval unchanged. Preserve Indic marks when adapting experimental accent handling. Use one seed and the same control settings. Cap this branch at **45 minutes of otherwise available compute**; stop below `+0.0005` development gain or on precision/singleton harm. Do not add its historical gain to the sibling or gate gain.

### Retrieval diagnostic

Use the current missed-link ledger to separate never-generated links from shortlist/ranking losses. Spend at most **30 minutes** measuring one targeted name-only, script-aware or address-led retrieval proposal, chosen from the dominant remaining error class.

A retrieval change is eligible for this sprint only if, by 2h30m:

- It improves the retained-candidate macro oracle by at least `+0.001` on the same development entities, including any previously retrieved links it loses.
- Mean/p95/max candidate counts and runtime are measured. Start with a mean budget no greater than the existing approximately 40 candidates, unless a justified tradeoff is explicitly accepted.
- The affected training caches and downstream adaptation can finish before the freeze.
- The full retrieval run passes the three-hour submission feasibility check.

Otherwise save the diagnostic for later and keep current retrieval. Raising the oracle without retraining or checking actual macro F0.5 is not a promotion. Candidate compactness matters to the organizer's final review.

### Deferred for this deadline

- New GPU provisioning, broad neural fine-tuning, five new first-stage fits, large seed grids and a retrieval-engine rewrite.
- Automatically stripping French tokens from the frozen index; broad normalization changes would invalidate candidate reuse.
- Training a gate on French test pseudo-labels. The supplied rules do not establish permission for this; organizer clarification is required. A 20% holdout of self-generated labels measures consistency with the pseudo-labeler, not business identity accuracy.
- Optimizing country link counts, forcing a match for empties, transitive component merging, or searching business identities externally.

## 9. Model selection and audit

At most four full-pipeline candidates reach the final development comparison:

1. Unchanged submission_04 control.
2. Best passing gate/decision variant.
3. Best passing sibling variant, optionally including the single successful feature ablation.
4. Their combination, only if the measured combination improves on its components.

Choose one candidate before confirmation; freeze at most one challenger for the final audit. An audit failure does not authorize selecting a different challenger on the same audit. Preserve the control as the fallback.

**Learned-model promotion:** require a positive paired entity-bootstrap 95% lower bound against the control rescored on the same new audit; no point precision drop; no increase in singleton false predictions; neither known-country point delta below `-0.0005`; and a runtime pass. Report recall and newly empty non-singletons even when other checks pass. These are predeclared engineering thresholds, not guarantees about France.

**Gate-only portal experiment:** it may meet the non-inferiority policy in Section 6 without showing a labeled gain. Label it as a portal experiment with unmeasured France accuracy. Preserve sub04 and record exactly how many known-country decisions also change.

Ten thousand audit references may not resolve very small gains. If the interval is inconclusive, report it and retain the incumbent for a claimed improvement; do not silently enlarge the sample repeatedly until the result passes.

The old audit's 15 singleton mistakes are not a universal cap for another sample. Compare the candidate's count with the incumbent's count on that same new sample.

Leave-country-out training is useful but optional within this window. Run one direction only if prepared assets and spare compute exist; refit all supervised aliases/transformations within the training country. It is a transfer stress test, not a France score estimator.

## 10. Reuse boundary and runtime gate

### Three possible release modes

| Mode | Reuses | Must still compute | Eligibility |
| --- | --- | --- | --- |
| R1 — Decision-only gate | Exact sub04 pair IDs, stage scores and candidate export | Record-derived gate signals, global decisions, both TSVs and validators | Fast fallback; no classifier rerun |
| R2 — Sibling/context rescore | Frozen candidate IDs and usable stored first-stage scores | Missing pair/context features, sibling scores, selected stage two, global decisions and outputs | Must benchmark feature reconstruction; chunks are not full feature caches |
| R3 — Changed retrieval / first stage | Only unchanged verified assets | New affected retrieval/features/models/context and final decisions | Highest risk; allowed only after an end-to-end throughput pass |

Reusing candidate IDs is exact only while indexes, normalization, aliases, blocking ranker, postings budgets, bridge settings and final candidate truncation remain frozen. In the current runner, bridge seed selection depends on the blocking ranker; changing the downstream matcher alone need not rerun bridge retrieval.

Changing the first-stage score requires rebuilding probability-dependent context such as peers, ranks and best-other scores. Changing any final score or veto requires recomputing decisions across the complete final claim graph. Never patch only a few affected rows and leave old ownership winners elsewhere.

`candidate_pairs.tsv` must contain the exact final blocking candidates actually scored by the matching pipeline. A gate cannot shrink the reported candidate list to only accepted or rerouted pairs. Auxiliary target-target comparisons do not hide the S1-target pairs already scored. For R1/R2 with unchanged retrieval, preserve the verified full candidate set; for R3, export the new exact set.

### Benchmark before final freeze

Use a deterministic, country-stratified sample, including common-name collisions, long candidate lists and routed sibling cases. Time reads, lookups, normalization, features, prediction, sorting/ownership and export—not only `model.predict`. Measure at least two representative blocks and report the slower rate plus memory and tail behavior.

The preferred final allocation gives **two hours to scoring/rescoring**. For 1,732,544 references that requires at least **241 references/second before safety margin**. Aim for a measured rate of at least **270 references/second**, or a directly measured stored-score processing ETA that fits the same allocation. The inspected sub04 rate around 200 references/second would not pass this conservative full-rerun budget.

If the selected model cannot pass, use its validated R2 path if available; otherwise use the independently validated R1 variant or unchanged sub04. Do not discover the deadline problem after launching a full run. At 6h, a complete, validated baseline must already exist locally or its remaining completion time must be explicitly accounted for.

## 11. Three-hour submission window

| Time after research ends | Action |
| --- | --- |
| 0:00–0:10 | Verify frozen package, disk space, input hashes, exact run mode and fallback availability |
| 0:10–2:10 | Run selected inference/rescoring mode with resumable, pair-keyed checkpoints; monitor throughput without changing model/rules |
| 2:10–2:35 | Finish global ownership, write both TSVs, run strict and official validators with ID checking |
| 2:35–2:50 | Copy outputs to Mac, verify hashes, upload the chosen matching file, record portal receipt/status |
| 2:50–3:00 | Buffer for transfer or format issues; preserve complete fallback and final artifact map |

If ETA exceeds the deadline, stop assigning work to the nonviable new run and select a **previously complete, validated** candidate. Never combine a partial new model run with old outputs without a separately specified and evaluated routing policy.

Preserve `output/submission_03/` and `output/submission_04/`. Use `output/submission_05/` for the selected new version. If a separate subsequent approved portal trial is justified and fits the remaining time, give it a new directory; never overwrite an uploaded version.

Submission_04 may be uploaded as soon as it completes, validates and reaches the Mac, if portal access and submission allowance are available. Keep that upload separate from experiment compute. A portal change between versions is an aggregate comparison; it is not a France-only measurement when other countries' predictions change. Public-score selection also does not guarantee a private-score improvement.

No dependency on an unknown portal submission allowance or immediate scoring turnaround. If only one upload remains, preselect the file from the declared validation policy and record the uncertainty. Keep the exact upload hash and portal status. `SCORED` and a reported leaderboard number are separate from local validator success.

## 12. Files and interfaces to implement after approval

These are proposed paths, not claims that the scripts already exist.

| Path | Owner | Contract |
| --- | --- | --- |
| `scripts/export_sprint_workbench.py` | Mohith | Verified pair-keyed scores/features, exposure ledger and provenance |
| `scripts/france_gate.py` | Sanhitha | Deterministic auxiliary views, ambiguity/street signals and veto reasons |
| `scripts/decision_policy_v2.py` | Harsha | Explicit proposal, gate and complete-owner arbitration; original control retained |
| `scripts/train_sibling_matcher.py` | Harsha | Owner-disjoint training, OOF support, model/schema manifest |
| `scripts/evaluate_sprint.py` | Mohith / Harsha | Paired entity macro F0.5, country/error slices, bootstrap and runtime report |
| `scripts/build_gated_submission.py` | Harsha / Mohith | Verified sub04 chunk reuse, global decisions, exact candidate export and validators |
| `reports/sprint_6h/` | Sahasra | Append-only experiment registry, decisions, timings and acceptance report |
| `models/sprint_6h/<run_id>/` | Harsha | Immutable selected models, parameters, fold IDs and hashes |

Agree these interfaces during the first 30 minutes. Each component must run on small fixtures without waiting for the full cache. Only the integrator edits shared runner/configuration files.

Required tests cover actual failure modes: missing addresses; all-stripped/empty names; Indic combining marks; postcode mistaken for house number; same-name different-owner cases; true numeric deletions; rescue-vs-normal ownership; exact ties; gate/no-gate parity; chunk hash/order mismatch; one S1 row even with zero candidates; and exact candidate/match membership.

## 13. Final artifact and cleanup checklist

- [ ] Every one of the 1,732,544 test S1 IDs appears exactly once in both files, including all France rows.
- [ ] Empty predictions remain empty; no forced matches, repeated IDs or invalid S2/S3 references.
- [ ] Final matches are a subset of the exact scored candidates.
- [ ] Strict and official validators pass; official validation runs with `--check-ids`.
- [ ] Report includes per-country references, candidate mean/p95/max, links, empties, ownership collisions, runtime and peak memory.
- [ ] Store hashes of source data, pair order, native extension, indexes, aliases, normalization, features, models, decisions and output files.
- [ ] Keep the exact code snapshot, pinned requirements and reproduction commands. Preserve the repaired sub04 checkpoint metadata and audit provenance.
- [ ] Copy outputs, weights, reports and reproducibility assets to the Mac or other durable storage and verify destination hashes.
- [ ] Record the actual uploaded artifact and portal result separately from local development/audit scores.
- [ ] Complete the methodology document and final package; no dataset or credentials committed to Git.

Only after artifact verification and confirmed completion of all jobs, inventory and clean up the two dedicated resource groups: `er-temp-rg` and `er-expcpu-0927-e70007-rg`. Check their contents before deletion; do not use wildcard cleanup. VM 1's `/mnt` storage is temporary and can be lost on deallocation. Copy and verify it first. Do not add an automatic shutdown schedule or rely on deletion as part of this planning step.

## 14. Research basis and applicability

- [Learnable similarity functions — Bilenko and Mooney](https://www.cs.utexas.edu/~ai-lab/pub-view.php?PubID=51499): motivates learning field-specific noise and target-target compatibility. It does not predict a gain on this dataset.
- [Scalable collective entity matching — Rastogi et al.](https://www.vldb.org/pvldb/vol4/p208-rastogi.pdf): motivates bounded use of neighboring records instead of unconditional graph merging.
- [Sparkly](https://pages.cs.wisc.edu/~anhai/papers1/sparkly-vldb2023.pdf): supports measuring lexical retrieval quality against candidate size and runtime, not replacing the current engine blindly.
- [Sudowoodo](https://arxiv.org/html/2207.04122v2): supports the research value of contrastive learning and pseudo-labeling, while making label quality important. It does not establish competition permission or French accuracy.
- [Fine-tuning LLMs for entity matching — Steiner, Peeters and Bizer](https://arxiv.org/abs/2409.08185): reports cross-domain risks for the tested generative models. It is not proof that all neural matchers transfer poorly.

Web research supplies methods only. No external business lookup, geocoder, business registry, hosted matching API or external identity augmentation enters this pipeline. Any future pretrained model must independently meet the organizer's license, parameter and offline-use rules.

## 15. Definition of success at the deadline

At six hours: one reproducible selected pipeline or an explicit no-go, its honest development/confirmation/audit evidence, a measured submission ETA, and a complete baseline fallback.

At nine hours: a complete, validated, preserved submission with the exact upload file identified. If the new work does not clear its gates, submit the best already verified artifact. A lower verified score is reported honestly; a development peak, candidate oracle, pseudo-label agreement or inferred France score is never presented as portal 0.99.
