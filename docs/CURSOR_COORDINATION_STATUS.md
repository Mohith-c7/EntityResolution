# Cursor final sprint coordination

Updated 2026-09-27. Coordinator: parent session subagent `cursor_final_coordination`.

Review complete: all ownership variants have zero development gain on both original and neural-v2 probabilities over the complete existing 30k graph; zero collisions and zero changed links. Independent tests reproduce ineligible-rival suppression and empty-input defects. Full findings: `docs/CURSOR_POLICY_REVIEW.md` in the parent workspace.

Cursor's separate feature delivery is now pushed at `43e14c633f45193611c8b2ea96d094dac957a919` on `cursor/token-alignment`. The coordinator's corrected version is pushed at `ee1e0b5b4c5ffcabc5f3fe54993fdd7e9d3908e1` on `cursor/token-alignment-fix`, with 13 tests passing. It fixes accent-folded/phonetic DF views, statistics provenance and salient token selection. Artifact versions are v2; rebuild v1 statistics. Integration remains the parent's responsibility.

Cursor's implemented branch is `retrieval-improve-0927`, locally at `608749aa62af4c0f73f513b86e87baf82641cf86`, based on safety commit `85c615b`. Its retrieval commit is `917264e494aab82e11cd3795903382217d9139e5`. Neither branch was advertised by origin when inspected. The separate `cursor/token-alignment` worktree remains at `00fa19b` with no committed feature delivery.

## Cursor delivery request

Push the branch containing the actual ownership/retrieval implementation. Record branch, immutable SHA, entrypoints, exact run command and runtime in a result document. Do not rewrite this coordination branch, the parent worktree, or frozen artifacts. The parent will evaluate the implementation on existing development data; do not use confirmation or final audit labels, launch a submission, or acquire new labels.

## File ownership

- Cursor implementation: `code/business_entity_resolution/src/model/ownership.py`, `code/business_entity_resolution/src/blocking/disk_index.py`, `tests/test_ownership.py`, `tests/test_disk_index_retrieval.py` on its branch.
- Coordinator: `research/final_2h/cursor_evaluation.py`, `tests/test_cursor_policy_review.py`, `docs/CURSOR_POLICY_REVIEW.md`, this file.
- Parent: `research/final_2h/fused_evidence.py`, `research/final_2h/diagnose_core.py`, model experiments, release decisions and VM2 jobs.
- Immutable: `scripts/research/sprint_6h` and existing models, output and frozen artifacts. No workspace branch switches or resets.

## Evaluation contract

Replay on every claimant in the existing 30,000-reference development graph. Compare the original frozen probabilities and the current `neural_v2_scored30k` probabilities separately. Inspect previously exposed `early_stop` 10,000 labels first, then `selection` 20,000 labels. No consumed fresh confirmation, final audit or new labels. Report paired macro F0.5, precision/recall, corrected/harmed links and ownership collisions. Development quality cannot certify full official-test claimant density or a 0.99 portal score.

Check deterministic tie handling, at most one accepted owner per target, empty inputs, and whether a claimant that fails its own rank acceptance suppresses a viable owner. Cursor's current `cover_rank_one` pools every score above `t_first`, even when that row is not its source's accepted rank-one candidate; this requires an independent counterexample test.

Evaluation code and data copies will use a separate coordinator directory on VM1. VM2 reads are limited to the existing development graph and exposed truth. Root owns the running fused-model VM2 job.
