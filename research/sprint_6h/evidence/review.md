# Read-only sprint evidence review

Reviewed the current frozen bridge model, current score/decision code, sibling and gate implementations, and already-exposed development evidence. No new audit labels, external identity data, model fit, or VM artifact writes were used. This review changes only this note.

## Current 20k keyed evidence, after development50k sealing

The coordinator subsequently authorized a bounded one-CPU read-only VM2 pass. The selected role now contains 800,000 keyed pairs, verified by `load_pairs` against its complete manifest. Its incumbent decisions are inherited from the supplied **50k** claimant cohort, rather than recalculated inside the 20k role. The resulting selected macro F0.5 is **0.9795046342386952**, with India 0.9741704552757853 and US 0.9830319536774963. This differs legitimately from historical full-ownership 0.9798201146 because outside-50k rivals are incomplete. Truth was limited to this already-exposed selected role; no new cohort, confirmation, or audit labels were accessed.

There are **2,051 rejected retrieved true links**, **268 accepted false links**, and **1,267 unretrieved true links**. The fixed route covers 1,509 rejected true links and 100 accepted false links. The following mutually exclusive error categories use the unchanged routing order; their macro correction gains are *not additive* because entity F0.5 is nonlinear.

| Route category | Missed true links | False links | Affected references | Perfect correction with other decisions fixed |
|---|---:|---:|---:|---:|
| Routed | 1,509 | 100 | 1,443 | +0.0067710640 |
| No first-stage seed >=0.90 | 62 | 17 | 71 | +0.0028705177 |
| Only qualifying seed is candidate itself | 7 | 18 | 25 | +0.0011083333 |
| Outside 20% reference cap | 229 | 115 | 331 | +0.0021842452 |
| Outside candidate scores [0.03, 0.995] | 231 | 18 | 242 | +0.0010525730 |
| Outside four-candidate cap | 13 | 0 | 12 | +0.0000600712 |

**Best bounded next lever:** continue the planned learned target-compatibility residual and explicitly inspect its missing-address slice. The four-candidate limit is a negligible current bottleneck, so widening that budget has little evidence behind it. Candidate address missingness accounts for **1,106/2,051 (54%)** of retrieved false negatives. Of these, **752** are routed, and **489** are routed with baseline p>=0.32, within the ordinary 0.75-blend positive crossing range. Perfectly correcting just those 489 rows with other decisions fixed would improve macro by **+0.0018699590**. This is concrete opportunity for name/sibling evidence under missing address information, not a learned-gain forecast. The component already has target-pair address-missingness features; retain and evaluate those rather than launching an extra model branch based only on these counts.

The no-independent-seed references are a high macro-loss blindspot despite their small link count. The current route cannot act on them. Likewise, it reaches only **6 of 40** accepted false links on singleton truth references. Do not expect the learned sibling lane to solve the full empty/singleton burden. Any further reference-priority change would be a new selection decision and must respect the existing predeclaration; this review does not recommend a post hoc route change.

Another **456 routed false negatives have p<0.32**. Their perfect fixed-row correction gain is +0.0017146754, but this is beyond ordinary 0.83 acceptance even with residual=1 and weight=0.75. Rank-one rescue and ownership exceptions require the sibling lane's actual feasible-score replay. Only **one** false negative already has p>=0.83, so inherited ownership rejection is not the primary retrieved-recall problem in this claimant cohort.

Current overlapping feature slices, read from keyed captured features, show:

| Slice | India FN / FP | US FN / FP | Routed FN / FP |
|---|---:|---:|---:|
| Candidate address missing | 382 / 18 | 724 / 48 | 752 / 36 |
| Strong phonetic name (sort >=0.90) | 308 / 37 | 625 / 76 | 688 / 41 |
| Strong name, weak address (sort >=0.90 / <0.50) | 204 / 10 | 357 / 34 | 412 / 23 |
| Strong address, weak name (sort >=0.90 / <0.50) | 52 / 8 | 41 / 6 | 77 / 3 |
| Numeric contradiction | 63 / 9 | 204 / 27 | 199 / 11 |
| Numeric containment | 10 / 0 | 31 / 11 | 36 / 6 |
| Numeric substitution | 19 / 7 | 78 / 11 | 74 / 2 |
| Distinctive address mismatch | 172 / 22 | 128 / 27 | 242 / 16 |

None of the error rows have a missing reference address or either missing name. Numeric contradiction appears on 267 missed true links versus 36 false links, reinforcing the existing decision to avoid a number-disagreement veto.

There are no Indic-script **reference** records in this selected role, but supplied target text contains **180 Indic-target false negatives and 34 Indic-target false positives**. The route reaches 145 and 7 respectively, and every one of those 145 routed false negatives has at least one known true routed seed. Therefore a reference-only script count would conceal a real transfer slice. The 145 routed missed targets yield 260 comparisons against known true seeds; 103 have different Indic/non-Indic name-script status. Only 20 of those comparisons (17 candidate rows) have current raw sibling name-sort <0.50 but the existing frozen `phonetic()` alternate >=0.80. That is modest evidence for a target-pair phonetic feature, insufficient to justify another model experiment during this sprint. The existing learned compatibility/residual experiment should report this target-script slice.

Paths on VM2: `research/sprint_6h/sibling/workbenches50k/selection/{pairs.parquet,manifest.json,truth.json,countries.json}` and `models/sprint_6h/development50k/feature_chunks/*.parquet`. Pair SHA256: `c13bd4732b8899be87e8097203564090f99288f8b46161bdd95eddbf13930a90`; manifest SHA256: `3dc0fd76c98313e5c8bac95209cebb039a56525697e9712715d8b0382d3453cf`. Numeric/name slices preserve exact feature meanings. Target script checks read only error and seed text from supplied-record indexes, never owner labels from those indexes. All analysis ran with Arrow/BLAS/OpenMP limited to one CPU, and no VM artifacts were written.

## Urgent records-only gate connector sanity check

Used the actual VM2 `research/sprint_6h/gate/test_full_stats_v2.json` (SHA256 `0f56d44385a720f6b59b373642db6c39f757243717622cfc0e1346b894e0be49`) and unchanged `address_view` / `_weighted_overlap` functions. The examples below are supplied-record-style address strings, not labeled French business pairs, and were not used to fit or select a rule. Both parses are reliable in every row.

| Address strings compared | Parsed street tokens | Shared tokens | Weighted overlap |
|---|---|---|---:|
| 112 Rue de Béthune / 61 Rue de Thumesnil | `{bethune,de}` / `{de,thumesnil}` | `de` | 0.131903 |
| 12 Rue du Moulin / 61 Rue du Stade | `{du,moulin}` / `{du,stade}` | `du` | 0.321108 |
| 12 Rue des Écoles / 61 Rue des Fleurs | `{des,ecoles}` / `{des,fleurs}` | `des` | 0.283067 |
| 12 Avenue de la République / 61 Avenue de la Libération | `{de,la,republique}` / `{de,la,liberation}` | `de,la` | 0.332066 |
| 12 Rue de l’Église / 61 Rue de la Mairie | `{de,eglise,l}` / `{de,la,mairie}` | `de` | 0.103733 |
| 12 Rue Béthune / 61 Rue Thumesnil | `{bethune}` / `{thumesnil}` | none | 0.000000 |

The motivating Béthune/Thumesnil example does **not** exceed both predeclared cutoffs: 0.15 can flag its street contradiction, while 0.0 cannot. The broader connector blindspot is real: the next three pairs exceed both cutoffs despite sharing only the displayed connector words. The final row is disjoint but has just one token per street, so it fails the separate minimum-two-street-token guard.

France has 259,452 Source 1 records. The actual address document frequencies and overlap weights are: `de` 201,962 / 1.250491; `du` 34,271 / 3.024247; `des` 31,426 / 3.110908; `la` 96,207 / 1.992063. For comparison, `bethune` 187 / 8.229889 and `thumesnil` 32 / 9.969823. The calculation is `1 + log((1+n)/(1+df))`, divided by the smaller total street weight. Consequently every shared token has weight at least one, and moderately frequent connectors retain a substantial fraction of a short street span's weight. The parser removes the learned street-type prefix but retains connectors; weighting uses **all address document frequency**, rather than a separate street-content or locality frequency view.

Conclusion: distinctive street information is present in the token sets, but the current gate does not explicitly distinguish shared connector evidence from shared street-name evidence. Weighted containment alone does not implement that distinction consistently. The v3 house-agreement guard does not change this street representation. Simply deleting connectors would also change the minimum-two-token guard's coverage, so this is a representation/coverage issue, not justification for an unreviewed token-strip rule or threshold change. No rule, model, grid, or label choice was changed. Coordinator and gate implementer were notified immediately.

## Earlier sibling feasibility finding

`scripts/train_sibling_matcher.py::opportunity_audit` replaces routed accepted states with their true decisions. The residual integration instead changes probabilities by a bounded linear blend. These are different feasible sets. The perfect-decision ceiling can pass the +0.0025 opportunity floor while an approved blend cannot reach that floor even with an oracle residual.

For an ordinary link with threshold 0.83, an oracle positive residual of 1 can recover a score only when `p >= (0.83 - weight)/(1 - weight)`. The predeclared weights 0.25, 0.50 and 0.75 therefore have lower score limits 0.773333, 0.66 and 0.32. A rank-one rescue at 0.70 has different limits; final ownership and rescue decisions must be replayed to assess the actual effect. Reference routing prioritizes uncertainty near 0.50, so the largest predeclared blend materially expands reachable recall.

Concrete current-model evidence comes from the existing frozen 250-reference replay, containing 10,000 keyed candidate rows. All 250 references are in the already-exposed 20,000 selection truth. Applying the unchanged fixed route and replaying `run_frozen_pipeline.decide` within this 250-reference cohort gives:

| Diagnostic | Macro F0.5 gain | Rejected true links recovered | Accepted false links removed |
|---|---:|---:|---:|
| Perfect routed decisions, other decisions fixed | +0.0052741180 | 16 | 1 |
| Oracle residual endpoints, weight 0.25, decision replay | +0.0016729692 | 2 | 1 |
| Oracle residual endpoints, weight 0.50, decision replay | +0.0025907181 | 5 | 1 |
| Oracle residual endpoints, weight 0.75, decision replay | +0.0047200055 | 14 | 1 |

The unchanged route selects 50 references, 100 candidate rows and 189 comparisons. Two routed false negatives have scores 0.1080668 and 0.2315124, below the ordinary recovery floor even at weight 0.75. Outside the route, two rejected true links fall below the 0.03 candidate minimum, two fall outside the reference cap at scores 0.0536120 and 0.0981790, and one lacks an independent seed at score 0.7491069. These counts describe current keyed errors, not historical pair errors.

This small cohort is nonrepresentative and its claimant universe is incomplete. Its baseline macro is 0.9893576081 and must not be represented as reproduction of the historical 20k baseline 0.9798201146. The table corroborates the need for the feasible-score diagnostic and supports proceeding with the already-declared 0.75 option; it does not select a production weight or establish learned performance.

Recommended immediate check: on the regenerated keyed selection workbench, retain the existing perfect-decision audit and add a labeled endpoint diagnostic for each of the three fixed weights. Compare each candidate with the incumbent replayed over the identical declared claimant universe. Report the reachable gain, corrected false/true links, country deltas, and low-score routed false negatives. This requires no model training and no new parameter grid. Complete external claimant coverage remains a separate promotion requirement. The sibling implementer and coordinator were notified.

## Cautions limiting achievable gain

The gate remains a narrow precision intervention. In `gate/dev_official_raw_natural_v2.json`, all six declared settings preserve all 69,137 true links in a stress cohort but remove only 38/21/7 of 5,247 cross-owner name-neighbor negatives for rival cutoffs 1/2/4. These are text-neighborhood negatives, not incumbent false predictions. India has zero reliable-street pairs in this diagnostic, and the full supplied test Source 1 coverage has only 30 Indian records with two reliable street tokens out of 809,986. Thus the current gate cannot plausibly repair the India error burden; no India gain should be inferred from its US stress results. Full training statistics may change US coverage and must be used for the actual scored-pair evaluation.

The current bridge threshold-stage loss report (`reports/experiments/round3/bridge_losses_stacked_l255s42.json`, macro 0.9790768845 before the later ownership/rescue policy) attributes 0.00704 macro loss to missed retrieved links and 0.0035 to empty predictions with true candidates, versus 0.00393 total false-link loss. This supports the sibling lane's recall opportunity but is not a final-policy decomposition. The retained-candidate oracle 0.9934507878 is also a threshold-stage development ceiling; it neither guarantees a reachable residual gain nor validates France.

Use the regenerated incumbent accepted states and exact pair keys for all comparisons. Saved `accepted` values become stale after residual or gate score changes; replay the complete declared claimant table once before evaluating. A role subset must inherit the full source-cohort incumbent decisions or replay the same source-cohort claims for both control and candidate. Exact-key competitor approximations and within-role claimants cannot establish full-test ownership parity.

## Reproduction inputs

- Frozen configuration: `research/sprint_6h/coordination/baseline_freeze/frozen.json`.
- Keyed current replay: `research/sprint_6h/coordination/baseline_freeze/replay/chunks/0000000.parquet`.
- Already-exposed truth: `research/sprint_6h/validation/development_labels/truth.json` (restricted in memory to the replay references).
- Route: unchanged `select_route` defaults (20%, four candidates, two first-stage seeds >=0.90).
- Oracle endpoint diagnostic: set residual score to the true pair label on routed rows only; apply each declared linear blend; invoke unchanged `decide(candidate, candidate, frozen_config)`; score all 250 reference truths, including unretrieved links.
