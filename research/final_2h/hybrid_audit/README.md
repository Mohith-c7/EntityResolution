# Hybrid extension audit and TSV-only release protocol

This lane is additive. Submission05, its audited freeze, TSVs, ZIP and consumed audit remain unchanged. The new hybrid combines the original-route NN8 adapter with disjoint fixed4 empty-owner corrections, each anchored once to ORIGINAL p2. Blue05 is the comparator.

The extension reservation is `research/final_2h/extension_audit_last8_v1/plan.json`, SHA `7e0ae6bd98c6023690935f5317b3ea09550fcb5b117a23058db17b61566e8f2a`:10,000 owners (6,040 US/3,960 India) within the complete331,012-owner/13,240,480-pair heldout graph. It explicitly has `fresh_extension_audit_only` status. Do not relabel it as the consumed confirmation-and-audit reservation. No extension labels have been opened by this lane.

## Prepared artifacts and ownership

- `preflight.py`: reservation metadata/ID/exposure/hash checks without truth; final readiness requires explicit root promotion, hybrid freeze, Blue05-bound complete-graph predictions and measured runtime bound to that hybrid. It has no truth argument and opens no labels.
- `seal_predictions.py`: after explicit root promotion and new freeze, independently verifies complete40-candidate groups, original pair order/p2 anchors, disjoint main/rescue keys and unchanged outside-union probabilities. Replays the original global decision FUNCTION on ALL331,012 owners, with separately frozen baseline Blue05 rules and candidate rules, then slices the reserved10k. Prediction manifest keeps separate original-p2 and comparator lineage.
- `score_graph.py`: verified complete331,012-owner/13,240,480-pair graph; native8 on123,600 original-route pairs and native4 on78,703 disjoint extra pairs, anchored once to originalp2.
- `build_freeze.py`: binds actual generic `neural_layers.py` inference source for both fine heads, actual old-head source, separate comparator config, complete universe and new inputs; root-directed promotion remains separate from OPEN.
- `benchmark_cached_release.py`: fresh actual block decisions/export/ID validators and full new-stage timings for cache-reuse release.
- `evaluate_extension.py`: root-only fresh extension opening and paired evaluation with new prospective criteria.
- `prepare_empty_route.py`: label-free Blue05 fullgraph decisions→empty-owner ORIGINAL p2 top4 (targetID ascending ties)→exclude original-route keys without fifth-candidate refill→actual train reverse33 plus frozenp1,p2/full40rank→provided raw training text for old+fixed4 neural heads. This custom36 manifest is explicit, not disguised as the immutable original builder.
- `tests/test_hybrid_audit_preflight.py`: seven passing guard tests for full-claim scope, overlap, single anchor, comparator, unopened labels and root promotion binding.

Preparation completed on VM1 with8 workers in565.492 seconds:19,701 Blue05-empty owners,78,804 original top4 pairs,101 original-route overlaps excluded after top4,78,703 disjoint extras. Output `research/final_2h/hybrid_audit/heldout_empty_extra_v1/`; log `/mnt/er/hybrid_audit/heldout_empty_extra_v1.log`. It does not read labels. Root runs neural scoring on Mac. Agent1 owns fullTEST preparation/export under separate names.

## Exact available inputs

Mac:

- `models/sprint_6h/neural/neural_inputs/heldout331k_v2/`: sealed123,600 pairs; inputSHA `f96a5de02cbf93650169eacda0af8f6802f2c9a6cdeb8f3e5593693c60a831d4`.
- Old head scores `models/sprint_6h/neural/heldout331k_v2/`, fixed4 scores `models/sprint_6h/neural/last4_heldout331k/`; both complete123,600 and same inputSHA.
- NN8 head `models/final_2h/neural_last8_continue_v1/`, manifestSHA `5626ab23f1dda7cab60192654cb6835c1d17fc2f87e64b31a3b8dbfef0ef549c`. Root completed123,600 NN8 heldout scores in75.6 seconds; transferred `models/sprint_6h/neural/last8_heldout331k/` to VM1.
- FullTEST05 raw input `models/sprint_6h/neural/neural_inputs/full_test05/`:801,380 pairs, inputSHA `5ad5fb0ed89c56481cb307fd18d4688e9f8bb665fd31c177d38d826c9bb96440`; old+fixed4 scores already complete. Root/Agent1 can reuse raw input and old scores, score NN8 once, and retain fixed4 for extra disjoint rescue keys.

VM1 existence verified16:28 UTC:

- Original `models/sprint_6h/heldout_cohort/{pairs.parquet,manifest.json}`:13,240,480 pairs, pairSHA `e555a1e2b28b6dff1afcf9e51d5e42e9cfaed9c3f82620a2ac7b71b1b6e2944b`.
- Blue05 `research/final_2h/neural_last4_fine_scored_heldout331k/{pairs.parquet,manifest.json}`: full graph, pairSHA `8116c13f677c3a7018354defbc3b4c1d19d7e4488f74e5a8be578a0efa208f65`.
- Old37 `research/sprint_6h/sibling/neural_v2_features_heldout331k/features.parquet`: original123,600 route keys.
- Raw heldout JSONL and train Source1 reverse index `research/sprint_6h/reverse_competition/train_s1.sqlite` complete2,206,821 source records; target indices `models/index_train_S2.sqlite`,`S3.sqlite` and provided raw train TSVs.
- FullTEST base `models/final_2h/test_full_base/{pairs.parquet,manifest.json}`:69,301,760 pairs; original04 freeze `68d6148b3aedd1473b95503a7734a93dc6ccf5ebdc1a2e51041939305a32c0a9`.

VM2 development artifacts (Agent1 owns; no transfer of labels required):

- Hybrid `research/final_2h/fixed_hybrid_v1/{pairs.parquet,protocol.json,report.json}`. ProtocolSHA `f399fb99216cefbb66f1ceeca2004f1a943a9a22e8ae99778b62d2613647d42f`; reportSHA `65b19406d834e1f52fbaae1ca533e97f7e3f4d66a66263f3bf3143c6a90282a6`.
- NN8 native `research/final_2h/neural_last8_fine_v1/adapter.txt`:SHA `b462edc1d2a6cae9efdaa5ccb2f9463a49d53f8377049fbaa87ee23ae3da57ef`.
- Fixed4 native `research/final_2h/neural_last4_fine_v1/adapter.txt`:SHA `625268050ec1774ec2eb1c0c8d38b89bf13c284d3e7474b564bb8264eec0dac1`; fixed4 head manifestSHA `60bca75bf41628f43eff487f8bdbb9b2f2f5ad54967af467807d7bc55417083c`.

## Critical path after root promotion

1. Complete heldout extra36 features/raw-input manifest; transfer only raw extra JSONL+manifest to Mac. Score OLD frozen head and FIXED4 continuation on these exact extra keys. Existing original-route old scores remain reusable. Do not score extras with NN8 or refill fifth candidates.
2. Additive CPU join: old+fixed4 probabilities on the extra36; native4 correction from originalp2 once. Independently native8 on the original37+NN8 probabilities. Scatter disjoint keys into a copy of the complete original13.24M graph; preserve candidate membership/order and every outside-union p2. Keep Blue05 separate.
3. Freeze new code/models/schema/route/index/native inputs and prospective development criteria (point macro gain≥.0003, paired CI lower>0,country floor−.0005; precision/singletons diagnostic), including explicit `hybrid_policy` from preflight. Record original04 anchor, Blue05 comparator releaseSHA39c008..., hybridSHA, extension planSHA and model-fitted owner IDs; verify extension disjointness and identical NN8/NN4 outer-training owners.
4. Root records `status=root_promoted_hybrid_to_extension_audit`, `authorized_by=root`, candidate+reservation SHA and `extension_labels_read=false`. No automatic promotion based only on a helper exit code.
5. New hybrid runtime explicitly covers R2 verified-cache reuse: actual full official-test extra preparation and both new neural branches, measured anchored graph join, and fresh decisions/export/strict+official validators on two actual US/India-stratified heldout blocks. Report the remaining post-neural ETA separately from completed stages, include15% margin and compare with the18:10 UTC operational deadline. Full-test neural timings include France; heldout blocks imply no French accuracy result. This is not a cold reproduction benchmark. The old fixed4 projection is historical evidence only.
6. Run new sealer on complete graph, then `preflight.py ready`. Freeze manifest must include `prediction_input_sha256` for original,original_manifest,baseline,candidate,main_route,rescue_route,universe; `baseline_release_freeze_sha256`; standard immutable hash groups; candidate decision_config AND original Blue05 baseline_decision_config; reservation hashes. Predicted arrays are sliced only after full global competition.
7. Only root may explicitly issue OPEN for extension truth after these gates; development promotion alone is not permission to read labels. `evaluate_extension.py` implements the separate OPEN token, an atomic once-only lock in the extension cohort directory BEFORE truth opens, readonly alias SQL restricted to these10k owners, official truth integrity check and generic paired evaluation against Blue05. The evaluator has not been executed. The old hardcoded final4 exporter/evaluator cannot be reused unchanged.

## TSV-only official-test release

Agent1 can reuse verified69.3M base probabilities/all40 candidates and current test reverse/old-head features. Compute fixed Blue05-empty top4 rescue keys globally first, exclude existing main keys, build actual TEST-corpus reverse features and raw test text only for extras, score old+fixed4 extras and NN8 original801,380 pairs. Scatter anchored corrections disjointly, replay all1,732,544 owners, export a NEW versioned matching TSV and both strict/official ID validations. Candidate TSV remains immutable04/05 bytes if candidate membership/order remain exact. No ZIP rebuild, overwrite or modification is requested. Promote/upload only the newly frozen validated TSV after root audit decision;05 remains the safe fallback.
