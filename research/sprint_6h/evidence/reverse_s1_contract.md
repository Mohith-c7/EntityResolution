# Fixed reverse Source 1 competition contract

One new text-evidence axis for a bounded anchored adapter; no model fit was run in this feasibility review. Implementation was authorized by the coordinator after the read-only headroom and cost checks below. No retrieval hyperparameter search is proposed.

## Universes and role safety

Build separate text-only indexes for complete supplied Source 1 universes: **training 2,206,821 records** and **test 1,732,544 records**. Do not combine them. The training index supplies residual20k, early10k, selected20k, and later heldout features; the test index supplies official-test features. Text from potential rivals is allowed, but no rival identity labels, frozen first-stage rival probabilities, or learned cheap-ranker scores enter these features. The incumbent p1/p2 remain frozen per original forward candidate. This is not a whole-pipeline OOF claim.

All features retain exact `(source1_entity_id,candidate_entity_id)` keys and the existing pinned global route. Training labels remain residual20k only, stopping labels early10k only, selected20k is evaluated once after specification freeze, and the new audit remains sealed. Forward candidate IDs are unchanged; reverse retrieved S1 IDs are temporary text-rival evidence, never additional target candidates.

## Three fixed text fields

Use existing frozen normalization and approximate phonetic functions, not new aliases or external linguistic data:

- `name_key = accent_fold(name_core(normalize_name(raw_name)) or normalize_name(raw_name))`.
- `phonetic_key = phonetic(name_key)`.
- `address_key = accent_fold(normalize_address(raw_address))`.

Keep primary normalized name/address, country, and original S1 ID separately. Normalize country as an open string. A known target country queries that country; missing target country uses the global corpus. Country-local DF is computed over the same eligible country universe; global fallback uses global DF. Do not add a second any-country query path for known-country targets. Expose own-reference country conflict as a coverage flag rather than silently treating it as ownership evidence.

## Bounded retrieval and exact shortlist

For each of the three fields, choose at most **two distinct known terms** with `0 < DF <= min(20000,max(1,floor(0.02*N_country)))`, ordered by `(DF,term)`. Use global N/DF for the missing-country fallback. Skip unknown and oversized terms and record coverage. Posting lists must be complete: never take the first row IDs from an oversized list. At most six selected term lists are read per unique routed target.

For each candidate S1 and field f, compute `H_f = sum(IDF(term) for selected query terms whose posting contains S1) / sum(IDF(term) for selected query terms)`, with `IDF=1+ln((N+1)/(DF+1))`. An unqueried field has H=0. This is posting overlap, not raw fuzzy similarity.

Keep top 16 positive candidates per field, plus top 32 by `F=max(0.65*max(H_name,H_phonetic)+0.35*H_address,0.92*H_address)`. The union has at most 80 S1 rows. Break every score tie by ascending S1 entity ID. An integer array of lexicographic ID ranks can supply efficient stable ties. Fetch original normalized records only for this bounded union.

## Same-rival raw reranking

For each union member and the target, compute N=core/name token-sort similarity, P=phonetic token-sort similarity, and A=full-address token-sort similarity, each RapidFuzz score divided by 100 and zero if either field is empty. Let D be Jaccard over canonical ASCII **standalone decimal address tokens**, and C indicate both numeric sets present and disjoint. Use existing `extract_digits` on the joined address tokens satisfying `isdecimal()` so Unicode decimal digits canonicalize; embedded leetspeak/alphanumeric identifiers are not street numbers.

Use the fixed joint score `J=max(0.58*max(N,P)+0.35*A+0.07*D,0.92*A+0.08*D)-0.08*C`. Retain **top eight S1 rows** by `(-J,S1_ID)` for target-query reuse. For each original forward pair, exclude exactly its own Source 1 ID and retain the best three remaining joint rivals. Compute the own S1-target components using the identical function, even if own S1 was not retrieved.

Rival N/P/A/D/C always come from the **same best joint rival**. Do not combine a maximum name score from one owner with a maximum address score from another. Preserve rival IDs only in diagnostic/provenance columns, never as model inputs.

## Exact feature schema

Thirty-three reverse evidence features, followed by three stored-score features, produce **36 adapter inputs**:

```
reverse_own_name_sort
reverse_own_phonetic_sort
reverse_own_address_sort
reverse_own_number_jaccard
reverse_own_number_contradiction
reverse_own_joint
reverse_other_name_sort
reverse_other_phonetic_sort
reverse_other_address_sort
reverse_other_number_jaccard
reverse_other_number_contradiction
reverse_other_joint
reverse_margin_name
reverse_margin_phonetic
reverse_margin_address
reverse_margin_number
reverse_margin_joint
reverse_other_second_joint
reverse_other_third_joint
reverse_best_second_joint_gap
reverse_name_terms_used
reverse_phonetic_terms_used
reverse_address_terms_used
reverse_query_fields_used
reverse_high_df_terms_skipped
reverse_shortlist_saturated
reverse_own_in_top8
reverse_no_other_returned
reverse_target_name_missing
reverse_target_address_missing
reverse_target_numbers_missing
reverse_target_country_missing
reverse_own_country_conflict
first_stage
probability
candidate_rank
```

Margins are own minus best-other. If no other S1 returns, all six other components and five margins are -1, with `reverse_no_other_returned=1`. Missing second/third joint scores and unavailable best-second gap are -1. Coverage counts and missingness flags remain real values, including query_fields_used=0 for no searchable terms. `shortlist_saturated` means a positive field shortlist or joint shortlist actually dropped candidates at its cap. `own_in_top8` refers to the pre-exclusion retained list. A missing rival is an uncertain search result, not proof of uniqueness.

`reverse_best_second_joint_gap` is best-other J minus second-other J. Terms-used counts are 0–2 per field and query-fields-used is 0–3. `reverse_high_df_terms_skipped` counts distinct oversized `(field,term)` instances before choosing the two terms. Missing-name/address flags use the normalized text views; missing-numbers uses the standalone decimal-token set. Own-country-conflict is one only when both countries are present and differ. Unknown terms are not high-DF terms. These definitions are fixed before feature generation and are not retrieval-tuned.

`candidate_rank` is **frozen p2 rank over all retained forward candidates for that reference**, using `(-p2,target_ID)`, not rank among the four routed rows or source concatenation order. `first_stage` and `probability` preserve exact stored p1/p2. The anchored adapter uses external logit(p2) exactly once; unchanged thresholds and complete declared claimant replay remain required.

Manifests bind both source TSV hashes, normalization/feature code hashes, field/query policy, DF table, index completion, country record counts, pair order, route manifest, and original p1/p2 provenance. An incomplete index or missing source record is an error; do not silently make absent coverage a unique-target feature.

## Inputs for integration

On VM2, repository root `/home/azureuser/er-experiments/e70007/sprint_6h/entity`:

- Residual fit: `research/sprint_6h/sibling/workbenches50k/residual_train/{pairs.parquet,manifest.json,truth.json,reference_ids.json}`.
- Heldout scoring graph: `research/sprint_6h/sibling/learned30k/{pairs.parquet,manifest.json}`.
- Pinned heldout route: `research/sprint_6h/sibling/learned30k_route/{route.parquet,manifest.json}`.
- Early truth: `research/sprint_6h/sibling/workbenches50k/early_stop/truth.json`.
- Raw development references: `models/sprint_6h/development50k/references.json`.

The residual20k route is label-free over its own declared 20k universe, matching existing fit setup; the early10k/selected20k scoring route is the globally pinned 30k graph. No role-only rerouting or selection label access is allowed.

## Read-only headroom and runtime evidence

On **early10k only**, with unchanged control replayed over the same 30k claimant graph: 455 routed missed true links have no exact target core or phonetic rival; correcting those rows perfectly with other decisions fixed yields +0.0033517723 macro F0.5. Of these, 341 have p>=0.32 (+0.0025582965 fixed-row ceiling). Conversely, **51 of 63 routed false links have no exact core, phonetic, or address rival**, versus only 12 with an exact name rival. Removing those 51 has a +0.0010806505 fixed-row ceiling. These reveal headroom not resolved by exact-count features, but are **not actual fuzzy-rival gains** or additive attainable improvements.

The current early p2 band [0.70,0.83] has 128/185 true pairs under exact core/phonetic uniqueness (69.2%), versus 80/125 with an exact name rival (64%). The earlier supplied 83%/46% observation does not directly reproduce under this predicate/band and should not be treated as current confirmation.

Existing 5.0M-target SQLite FTS index metadata records approximately 344.5 seconds to build, including character indexing. A read-only 20k-record prefix normalization+phonetic probe runs at about 10.8–11k records/CPU-second, implying roughly 3.4 CPU-minutes for normalizing all 2.2M training S1 records before database work. An actual reverse-index build time has not been measured; **5–15 minutes per index is an estimate to gate with observed progress**, not a benchmark result.

A read-only query proxy over 500 targets against the existing 5M native S2 postings index, using two terms per field and no aliases/ranker, takes 2.423 s (**206 queries/CPU-second**); retrieval is 0.944 s and bounded raw reranking 1.472 s. It accumulates about 9,795 records/query and reranks about 38/query. At the maximum 1.386M unique routed targets this is approximately 112 CPU-minutes, or 14 ideal minutes with eight workers. Real full-test time adds phonetic-channel lookup, cold I/O, score scanning, global routing and ownership replay. Target deduplication can reduce queries, but its reduction is not assumed in this bound.

The ordinary DiskSourceIndex builder explicitly rejects S1, and the existing native postings wrapper does not support a phonetic field. Therefore this requires a small separate text-only builder/query wrapper; it is not configuration-only reuse. Reuse the native posting packing/reduction and frequency policy where practical, with the above fixed semantics. The first implementation gates should verify source/index completion and query throughput, then measure actual early-role own-versus-rival separation before another model fit is accepted as useful evidence. No promise of recall recovery or French accuracy follows from this feasibility review.
