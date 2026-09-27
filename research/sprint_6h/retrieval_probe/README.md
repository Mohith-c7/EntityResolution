# Exposed development retrieval diagnostic

The final corrected result is documented in [UPSTREAMQUOTA.md](UPSTREAMQUOTA.md).
It passes the retrieval oracle target, is fully scored by the frozen matcher with
rebuilt context, and is **not promoted** because matching gain is small and
precision/singleton errors worsen. The earlier failed result below is preserved.

The predeclared missing-address, name-only extension **failed** its +0.001
candidate-oracle gate. No matching-model scoring, fitting, production edits,
heldout-label reads, or further population variants were performed.

Read [result/report.json](result/report.json) for the frozen 5,000-reference
country-stratified paired result. Its baseline candidate sets exactly replay the
saved, model-scored development50k candidates. The fixed ID hash is
`43db565a3dfa04e83a707d3350c2a9619981af20fe517265e226be53e12e996c`.

| Measure | Current | Name-only extension |
|---|---:|---:|
| Candidate oracle macro F0.5 | 0.993665222 | 0.993865807 |
| Candidate recall | 0.983075323 | 0.983884011 |
| Mean / p95 / max candidates | 40 / 40 / 40 | 40.205 / 41 / 44 |
| Candidate pairs | 200,000 | 201,025 |

The extension adds 14 true links among 1,025 new pairs, with zero lost current
pairs or true links. Oracle gain is +0.000200586. The original specification
preserved all current pairs and added at most two new candidates per source from
the existing learned name-only top20, requiring an empty target address and core
token-set similarity at least 80.

Actual additional name-only query work is 130.223 aggregate worker-wall seconds,
or 26.045 ms/reference: 24.189 seconds in raw posting accumulation and 105.406
seconds in posting rerank. The 16-worker paired run took 101.820 wall seconds.
Total baseline retrieval was 581.299 worker-wall seconds; the extension pass
took 654.949. The latter difference is confounded by warmed caches, so the
directly instrumented 130.223 seconds is the useful incremental measurement.
These are aggregate wall durations within worker processes, not process CPU time.

[profile.json](profile.json) describes all 50,000 exposed references: 3,085 truth
links absent from the final scored candidate lists. The largest mutually
exclusive class is missing address (1,032), followed by script-changed names
(853), address-led changed names (710), shared-name shortlist/rank errors (489),
and no overlap (1). Absence from the final list does not mean absence from raw
postings.

The root-authorized follow-up mechanism inspection is limited to the fixed
sample's 115 missing-address true links across 114 affected references. See
[result/missing_address_quota_diagnostic.json](result/missing_address_quota_diagnostic.json).
All 115 are in the actual name/core posting-list union before fused128/field24
caps. Only 20 survive those caps under the name-only query; 14 reach learned
top20. Applying the same two-new/source quota before learned top20 would retain
18. Applying that same quota before fused/field caps, with the same similarity
rule and learned ranker, would retain 61 (49 US, 12 India); 98 pass the similarity
rule in the full posting union. This diagnoses the earlier fused/field shortlist
as the main bottleneck. These totals are conditioned on errors and provide no
population candidate-count, runtime, loss, or matching-quality estimate.
The 61 known recovered misses alone contribute +0.001195760 to the fixed sample's
oracle if added to all current pairs; the 18 before-top20 recoveries contribute
+0.000254758. These are conditioned contributions, not measurements of another
population variant; see [result/quota_known_miss_contribution.json](result/quota_known_miss_contribution.json).

All failed-probe candidate lists are diagnostic only. Added pairs have **not**
been scored by the frozen matching models because the oracle gate failed, and
they must not be included in a production candidate list. The VM1 full output
is `/mnt/er/entity/research/sprint_6h/retrieval_probe/result`, including both
candidate-set chunk directories and cached records. Local copies include all
reports, timing rows, and exact new/lost/addition tables. A first harness run
failed on a dataclass field name before extension retrieval; its partial outputs
were preserved separately on VM1. The corrected run used the unchanged fixed
sample and retrieval specification.
