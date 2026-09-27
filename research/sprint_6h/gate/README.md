Gate lane implementation, 27 September 2026
========================================

`scripts/france_gate.py` is an auxiliary records-only module. It does not alter
retrieval, saved matcher probabilities, frozen normalization, or ownership.
Country labels are normalized as open strings; there are no country conditions
or manually entered country legal/street word lists.

Interface
---------

* `fit_signals(records_by_country)` accepts mappings or normalized record
  objects, supporting `business_name`/`business_address` and `name`/`address`.
  Input is the declared Source 1 possible-owner universe, not targets. State is
  JSON-compatible (`aux-name-street-v2`) and contains exact membership/name keys, counts,
  address document frequency and structural template diagnostics.
* `pair_signals(s1, target, stats)` excludes the current Source 1 from each
  matching-name count. Missing/empty keys are `-1`; present unique keys have
  zero rivals. It verifies reference membership and normalized name keys.
* `gate_veto(signals, config)` is pure and returns `gate_veto`, `gate_reason`,
  version, and shorter aliases `veto`/`reason`. The original score is untouched.
  Config keys are `min_name_rivals`, `max_street_overlap`, `min_street_tokens`.
  The fixture default `{2, 0.0, 2}` is not a promoted/frozen experimental choice.

Views and limits
----------------

The original name remains available. The conservative reduced view strips only
edge tokens from the existing frozen `LEGAL_SUFFIXES`, after existing name
normalization. Empty reduced names are never count keys. A separately versioned
learned core requires at least three different observed base-name relationships,
each base repeated in the supplied records, and at least 90% edge dominance.
That hypothesis is exported for ablation and does not drive the default gate.

Street templates are inferred from short-numeric-leading clauses with diverse
content, edge dominance and, for prefixes, at least eight different following
words. A template remains a structural hypothesis, not proof of a token's
linguistic meaning. No supervised labels enter discovery. Version 2 resolves
conflicting orientations only when one marker has at least ten times the supplied
record support of every conflicting marker. This ratio was declared before its
labelled comparison. Other conflicting spans, unsupported syntax, missing
addresses and uncertain numeric extraction abstain. Original raw/folded
addresses and all address tokens remain available;
name/address overlap produces a separate name-token weight view.

The gate requires equal nonempty original/reduced names, sufficient possible
owner ambiguity, credible street evidence on both sides and overlap at or below
the chosen cutoff. Number, suffix or postcode differences alone cannot veto.
Containment overlap tolerates lost street tokens. Reliability and veto coverage
must be measured on real records; fixture success does not establish accuracy.

Validation and next experiment
------------------------------

`/tmp/entity-review-venv/bin/python -m pytest tests/test_france_gate.py
tests/test_normalization.py tests/test_competition_features.py -q` passes 36
tests. Fixtures cover missing addresses, dropped numbers, differing house/suffix
evidence, shared street/name content, empty cores, Indic marks, accents, postcode
confusion, self exclusion, JSON reuse, label independence, orientation abstention
and raw-lookup completion/input-hash binding, streamed gate pair preservation
and checkpoint/order-hash failures. The gate-specific tests also pass on VM2.

`profile_gate_records.py` opens only the explicitly supplied Source 1 TSV and
reports normalized view coverage, original/reduced owner ambiguity, templates,
timing, source/code hashes and optional fitted state. Prefix scans are explicitly
nonrepresentative; full-universe runs are marked separately.

Next, use the coordinator's current scored development claimant universe and
its already-exposed owner truth to evaluate natural different-owner name/street
collisions. Preserve true owner links with numeric deletions/shared buildings.
Inspect label-free overlap/count scales, then predeclare at most six settings
plus unchanged control before opening those evaluation labels. Report true/false
links vetoed, country deltas and entity macro F0.5 through the coordinator's
paired evaluator. Fresh confirmation/audit labels remain sealed. Full test
records can supply text statistics but never pseudo-label training targets.

Current native diagnostic (not a promotion)
-------------------------------------------

The already-exposed 20k development owners supply 69,137 known true targets and
5,247 exact original/reduced-name cross-owner target negatives. Target IDs were
first restricted by those exposed owner labels. Their original supplied raw text
was then recovered by streaming the official training S2/S3 files, retaining
only those IDs; no further labels were read. All six declared configurations
retained every true link in this stress cohort. Version 1 removed 37/21/7 false
pairs for rival cutoffs 1/2/4; version 2 removed 38/21/7. Overlap cutoffs 0 and
0.15 gave identical results. The owner-statistics universe here is only 20k.
These results must be repeated with full-training statistics before promotion.
The claimant probabilities and final entity-level predictions are not yet
available, so these pair counts are not a macro F0.5 improvement estimate.

The raw version 2 cohort contains 705 qualified house-number disagreements and
199 zero-street-overlap true pairs in the US; missingness, name and ambiguity
guards retain them. For example, supplied true variants `148 Pembroke Lane`
and `48 PEMBROKE LN` show why numerical contradiction cannot independently veto.
India's structural template coverage is effectively zero in the pilot, an
explicit limitation rather than inferred precision.

`build_raw_lookup.py` streams supplied S2/S3 TSVs into a new SQLite table retaining
original ID, name, address and country text. It writes an isolated partial file,
checks source stability and hashes, then exposes a complete file and atomic
completion marker. Optional expected source hashes protect against incomplete
transfers. `open_raw_lookup(path)` refuses missing completion markers;
`lookup_raw_records(conn, ids)` performs batched ID lookups and refuses missing
candidates. The old signal code and state manifest survive under `v1/`.

Full supplied test S1 v2 statistics cover all 1,732,544 references and carry source
hash `3d4a32c54c2ca9c53fd7c2be105bf26f708f94c4d2f88eb370972a195665c2f5`.
The fitted state hash is
`0f56d44385a720f6b59b373642db6c39f757243717622cfc0e1346b894e0be49`.
Label-free credible/two-token street coverage is France 248,155/201,942 of
259,452; US 347,329/95,189 of 663,106; India 47/30 of 809,986. This is coverage,
not matching accuracy. The raw S1-only SQLite lookup took 14.82 seconds on VM2,
one process, with 257 MiB peak RSS.

Full supplied training S1 v2 statistics now cover all 2,206,821 references, with
state SHA256 `57b41c42f45035597d31e0985d96ddcc12819947189a848605bc65fd1de1568c`.
US credible/two-token street coverage is 703,137/190,378 of 1,323,633; India
59/42 of 883,188. Repeating the same raw 20k natural stress cohort with this
larger ambiguity universe vetoes 49 false and 3 true pairs at rivals 1, or 45
false and 2 true at rivals 2/4. Both declared overlap settings give the same
counts. The earlier zero true veto does not survive the full owner universe.
One real failed true pair is `Cramerton, 3031 Misty Harbor Circle, Unit E, NC`
versus `3031 MISTY HARBR CIR, NC, CRAMERTON`: punctuation, abbreviations and
structural marker hypotheses can erase useful common street content. This
pair-only stress result neither promotes a setting nor measures final macro.

`build_gate_table.py` reads saved scored parquet chunks in 10k-row batches. It
loads only batch record IDs from read-only SQLite and emits every original
pair key with a separate `gate_veto`/`gate_reason`. One config per run is bound
to state, source, code, normalization and chunk hashes. Completed checkpoints
are reused only on exact hash agreement. Final merge streams those completed
tables and validates the entire UTF-8 `(S1 + tab + target + newline)` pair-order
hash against the verified reuse manifest when supplied. No score, candidate
membership or retrieval input is changed. `--workers` defaults to one and is
capped at eight for the allocated signals lane. Optional `--reference-lookup`
reuses a separate S1 lookup with the S2/S3 target lookup.

For gate-only R1, `--execution-mode R1 --decision-config frozen.json` binds an
eligibility floor to the exact minimum of frozen `t_first` and `t_rest` and to
immutable original probabilities. Rows below that floor survive with false
veto and reason `ineligible_r1_score`; these skipped rows are not observed
street agreement. Tests compare actual original and v2 decision masks against
full-signal tables. Floor-pruned R1 tables cannot be reused for sibling-boosted
R2 probabilities: compute after updated scores or include every globally routed
pair along with baseline-eligible pairs.

`evaluate_gate_grid.py` consumes the sealed full 50k development claimant graph
and explicitly supplied development-only owner truth. Each of the six already
declared settings uses the same raw training lookup/state and produces a full
pair-keyed gate table. Original-policy evaluation removes vetoed proposals
before unchanged arbitration and rank-one rescue, retains input order and empty
prediction owners, and reports full50k plus the exposed selected20k slice using
the paired evaluator. Original-policy fallback differs from v2's fixed proposals;
their effects must not be conflated. Confirmation/audit labels remain unopened.

Actual complete development decision results
-------------------------------------------

The verified current graph has 50,000 references and 2,000,000 scored pairs,
pair-order SHA256 `836f3ec30717aa90acd2b76a7c3b28a40ac38d88ebba6c5a376dae8e613a54fb`.
Original-policy baseline macro F0.5 is 0.9790805309852658 over all 50k, or
0.9795046342386952 on the selected20k slice inheriting all 50k claims. Every
declared v2 setting harms accepted true links and removes no accepted false
link: rival cutoffs 1/2/4 remove 6/5/4 true links in full50k and 3/2/2 in
selected20k. Overlap 0 and 0.15 produce identical development predictions.
All six fail the point precision check; none is promoted. Full results are
preserved in `development50k_grid_v2_report.json` and immutable VM2 tables.

The coordinator authorized one separately versioned v3 decision correction:
qualified `house_number_match is True` forces abstention. It never accepts a
pair, and missing/different numbers do not independently reject it. The wrapper
`house_agreement_gate_v3.py` consumes the unchanged v2 signals/state and is
pinned separately in the builder manifest (`--gate-rule v3-house`). Original
v2 code/helpers and state hashes survive in `v2/manifest.json`. One fixed
conservative comparison, rivals4/overlap0, was declared before the corrected
evaluation; no new tuning grid was opened. It restores all four observed
accepted true links and yields exact unchanged predictions on full50k and
selected20k, with delta/CI all zero. The raw20k natural stress cohort retains
all 69,137 known true targets and still rejects 45 cross-owner false pairs.
This establishes prevention of the observed regression, not improvement.

The same fixed v3 correction was then checked on raw records for every one of
the 50k current development owners and all 173,025 known true targets, including
targets outside accepted predictions. All true targets survive (69,813 India,
103,212 US). Of 34,334 natural same-name cross-owner negatives, 290 US pairs are
vetoed; none is an accepted baseline false link in the actual scoring graph.
The report is `dev50k_raw_natural_fulltrain_stats_v3_house.json`, with full text,
truth, state, wrapper and signal-code provenance. No further labels were opened.

Label-free actual France test-state probes expose another limitation:
`Rue de Béthune` versus `Rue de Thumesnil` has weighted overlap about 0.132
through shared `de`; cutoff 0 misses it, while 0.15 reaches it. Shared `du`,
`des`, or `de la` examples can yield overlap 0.28–0.33, exceeding both declared
cutoffs despite different other street words. Tokens remain preserved and no
manual connector list is introduced. No final configuration or further rule
has been promoted on the basis of these probes.

Approved fixed distinctive-token ablation, rejected
--------------------------------------------------

`v4_predeclared_distinctive.json` pins the one approved generic ablation before
evaluation. `distinctive_street_gate_v4.py` retains all original street tokens
and adds an auxiliary distinctive view: length at least three, nonnumeric, and
supplied-country address DF / owner record count at most 1%. A contradiction
requires a token on each side with no exact or near support across sides.
Near support uses edit distance / maximum length at most 0.25 when both tokens
have at least four characters; shorter tokens require exact equality. Existing
name, ambiguity, address, country and reliable-parse guards remain, and known
qualified house equality always abstains. The one fixed setting is rivals4;
weighted overlap/minimum raw token count are provenance, while the distinctive
criterion supplies the separately versioned street decision. No manual
connector list, normalization change, additional grid or post-result exception
was introduced.

This rule is rejected for promotion. It removes 18 accepted true links and zero
accepted false links in the complete current development graph; one previously
nonempty non-singleton becomes empty. Macro delta is -0.0000461437, paired 95%
CI [-0.0000976628, -0.0000135805]. Selected20k loses four accepted true links.
The separate raw stress test vetoes 19 known true targets and 2,998 natural
cross-owner negatives; all India positives survive. All 19 harmed true targets
are preserved with raw text, parsed evidence and acceptance state in
`v4_all_harmed_links.json`; this evidence does not authorize more exceptions.

All six original v3 settings have now been rerun on development50k. Every one
produces exact unchanged predictions on full50k and selected20k. Those outcomes
are preserved in `development50k_v3_six_report.json`, separate from v4.

Unlabelled French saved-claim diagnostics
---------------------------------------

`diagnose_french_claims.py` reads the complete portable sealed saved scores and
byte-copy proof. It scanned all 69,301,760 pairs, keeping every score at least
the exact baseline floor 0.6999999999999998: 5,919,526 pairs (France 997,413,
India 2,678,928, US 2,243,185). The 63,382,234 omitted below-floor rows are
explicitly skipped, never observed street agreement. All original saved scores
and candidate files remain immutable. Seven separately pinned French tables
compare v3's original six settings and the rejected fixed v4 rule. The release
worker independently verifies complete global baseline arbitration and compares
changed claims/winners, including inspectable paired state and IDs. These are
unlabelled diagnostics, neither French accuracy nor a links-count target.

Read-only preserved-address support diagnosis
--------------------------------------------

`diagnose_preserved_support.py` checks the rejected v4 contradictions using the
unchanged 1% DF and 0.25 token-distance definitions. It asks whether each side's
distinctive parsed tokens have support anywhere in the other full preserved
address. This would protect 16 of the 19 harmed true targets; three remain
contradictory (one Federal/Fdeeral transposition, two highway/hwy abbreviations).
Of 2,998 natural-negative contradictions, 14 have preserved support and 2,984
remain contradictory. `preserved_support_diagnostic.json` preserves all 19 true
cases and the evidence, hash `1fb52df87a8c7fef4c143f39ca3fafc1f815d1e0c53c121fe04b52018c2a222a`.
This is a diagnosis of parser loss, not another gate variant. No v5, threshold,
per-case vocabulary or full-test variant was implemented or executed.
