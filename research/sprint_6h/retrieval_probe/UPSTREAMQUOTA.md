# Final corrected retrieval diagnostic

The corrected `upstreamquota` variant meets the retrieval oracle target but is
**not promoted as a standalone sprint change**. Frozen matching improves only
slightly while precision and singleton errors worsen. Its artifacts remain for
one possible combination if the reverse-evidence lane later provides substantial
improvement; no additional candidate-rule search or scoring is authorized here.

The root explicitly authorized this final correction after the first probe
exposed early fused/field truncation. It uses the same fixed 5,000 already-exposed
development references, core token-set threshold 80, unchanged name-only learned
ranker, and two new candidates per source. It reconstructs the existing capped
name/core posting union **before fused128/field24 caps**, reads only empty-address
target metadata, and preserves all current candidates. The sample hash and
script hash were frozen in [result_upstreamquota/protocol.json](result_upstreamquota/protocol.json).

| Measure | Current | Corrected upstream quota |
|---|---:|---:|
| Candidate oracle macro F0.5 | 0.993665222 | 0.994860982 |
| Candidate recall | 0.983075323 | 0.986598891 |
| Mean / p95 / max candidates | 40 / 40 / 40 | 43.4096 / 44 / 44 |
| Pairs actually model-scored | 200,000 | 217,048 |
| Frozen decision macro F0.5 on fixed sample | 0.979575823 | 0.979719127 |
| Frozen decision precision | 0.995240104 | 0.995065888 |
| Singleton false positives | 8 | 9 |

Candidate oracle gain is +0.001195760, with 61 new true links, 17,048 added pairs,
and zero lost current pairs or true links. Baseline candidate sets exactly replay
the saved development scores. After the oracle gate passed, both exact candidate
lists were scored by the unchanged frozen first/second matching models, rebuilding
each list's context. Baseline first-stage, second-stage, and final probabilities
match all 200,000 saved sample scores bit-for-bit. Added pair sets match the scored
sets exactly. Nineteen of 61 new true links pass probability 0.83; among all new
pairs, 22 pass it (19 true, 3 false).

First-stage probabilities of retained pairs stay unchanged. Rebuilt contexts
change 45,588 retained second-stage probabilities, with mean absolute difference
0.0000475982 and maximum 0.169523850. The probability-threshold-only macro F0.5
gain is +0.000309970. With unchanged frozen rank-one and ownership decisions, the
gain is +0.000143303. The latter comparison uses **5k changed-reference candidate
lists and 45k unchanged incumbent rival lists**, including 148,772 strong rival
claims. It is not a uniformly recomputed 50k retrieval graph or a full Source1
cohort release-gate evaluation. No sampled pair loses ownership to those extra
45k rival lists. See [result_upstreamquota/model_score_analysis.json](result_upstreamquota/model_score_analysis.json).

Actual additive upstream retrieval work is 521.094 aggregate worker-wall seconds,
or **104.219 ms/reference**: 7.465 seconds to form the posting union, 441.119 seconds
for empty-address metadata filtering, and 71.592 seconds in the unchanged learned
rerank. The paired retrieval phase took 131.609 wall seconds with 16 workers.
Candidate count increases 8.524%. Matching-only worker-wall time is 178.110 seconds
baseline and 164.100 seconds for the warmed corrected pass; that negative difference
cannot support a claimed scoring speedup. No runtime engineering was performed.

Full raw outputs, cached candidate/target objects, and all model-score chunks are
preserved on VM1 at
`/mnt/er/entity/research/sprint_6h/retrieval_probe/result_upstreamquota`.
Local copies contain protocol, results, all per-reference timing rows, exact
new/lost/added pair tables, and model scores for every addition. The complete
original failed result remains in `result/`. No production, index, or model files
were modified, no model was fitted, and no heldout labels or fresh labels were
used. All jobs finished before the existing 09:10:21 UTC cutoff.
