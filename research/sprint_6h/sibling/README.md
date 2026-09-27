# Sibling lane: bounded target compatibility

The new component learns S2/S3 target-to-target compatibility. It reuses frozen
S1 candidate IDs and first-stage probabilities. Current ordinary fuzzy context
features are not a new experiment in this lane.

## Data and split contract

`scripts/train_sibling_matcher.py` accepts a pair-keyed parquet with:

- `source1_entity_id`, `candidate_entity_id`, `first_stage`, `probability`.
- `accepted` for development opportunity auditing: full incumbent policy,
  including ownership and rank-one rescue.
- `oof_excluded_fold` for component-training input. It must match the existing
  `inner_fold(S1 owner)` and identify the actual excluded first-stage model.

The separate input JSON manifest requires `status: complete`,
`verified_current_pipeline: true`, `split: train|development|confirmation|audit|cohort|test`,
`pairs_sha256`, `pair_order_sha256`, and `reference_ids` including references with
zero candidates. Training additionally requires
`first_stage_owner_excluded_oof: true`. The order hash is SHA256 of UTF-8
`S1_ID\tTARGET_ID\n` lines in parquet order. Development audit additionally
requires `expected_macro_f05`, recomputed with the identical claimant universe.
The loader refuses probability arrays and equal-length joins.

Truth is a separate JSON `{S1_ID: [TARGET_ID, ...]}`. Unretrieved links and empty
entities remain in the metric. Unknown target owners never become negatives.
Only training-reference owners supplied to the component table produce the
bounded shared-owner positive enumeration. Optional `--owner-db` queries just
the routed target IDs from `owners(id,owner,outer_fold)` and uses only rows with
`outer_fold='train'` to extend natural hard-negative coverage. Those additional
training owners all enter the supervised owner ledger. Tune/holdout target
owners and unknown owners remain excluded from target-to-target supervision.

The current `bridge_v1/stacked_l255s42/report.json` declares
`development_training_added=false` and 300,000 training references. Therefore
component training uses a bounded 50,000-owner subset of its outer-training set;
residual training uses the first 20,000 outer-tune IDs sorted by SHA256 of
`context-split-v1:` plus ID. The middle 10,000 remain early/diagnostic; the final
20,000 remain the development comparison. The three ID files in this directory
reproduce that existing split from `bridge_v1/first_stage/tune_references.json`
(SHA256 `4e29bb444f1f85088730ecd5c764983aae01e0d6556d0f6c15c884e7ddb07548`).

The compatibility model records every supervised target owner in its manifest.
Its fixed transform is used on residual-training owners that are entirely
disjoint from this ledger. The residual refuses overlapping owners and refuses
an inference sibling model different from the training transform. Baseline p2
is not called OOF on outer-training owners. No component or integration layer
uses fresh confirmation/audit labels or test pseudo-labels.

## Route and opportunity stop

The fixed label-free route takes at most 20% of a declared reference universe,
four candidate rows per reference and two first-stage seeds with p1 >= .90.
Candidate probabilities must be .03–.995; candidate ordering favors proximity
to .83, then the reference budget favors uncertainty. IDs break ties. A candidate
cannot support itself. Wrong OOF seeds are deliberately retained.
For outer-training negative mining, a local view uses OOF p1 for both candidate
and seed probabilities, avoiding in-sample p2 confidence. Stored p1/p2 are never
changed. Development/inference routing uses the approved final p2 policy.

The first command is `audit`. It changes only routed candidate rows to their
true decision and keeps other claims fixed. Its result is an opportunity ceiling,
not learned performance or a validated global ownership score. Stop dependent
training below +.0025 perfect-correction macro F0.5 gain.
Optional `--decision-config` also reports ideal residual endpoints at the three
historical diagnostic weights (.25, .50, .75) with the original policy replayed inside the supplied cohort.
That diagnostic includes score reach and rank-one rescue, but omits outside
claimants and is not learned performance.

## Commands

Set BLAS/OpenMP limits in the launching environment and use only the coordinator's
allocated `--threads` (default 1; maximum 20). Commands refuse existing output
directories. Replace the uppercase paths below with verified workbench assets.

```sh
python scripts/train_sibling_matcher.py audit \
  --pairs DEV_PAIRS --manifest DEV_MANIFEST --truth DEV_TRUTH \
  --reference-ids research/sprint_6h/sibling/selection_ids.json \
  --output research/sprint_6h/sibling/route_audit

python scripts/train_sibling_matcher.py prepare \
  --pairs TRAIN_OOF_PAIRS --manifest TRAIN_MANIFEST --truth TRAIN_TRUTH \
  --owner-db models/scale_v1_plan/aliases/counts.sqlite \
  --index-prefix models/index_train --output research/sprint_6h/sibling/prepared

python scripts/train_sibling_matcher.py fit \
  --examples research/sprint_6h/sibling/prepared/examples.parquet \
  --trees 250 --threads ALLOCATED_THREADS --output research/sprint_6h/sibling/compatibility

python scripts/train_sibling_matcher.py support \
  --pairs DEV_PAIRS --manifest DEV_MANIFEST \
  --reference-ids research/sprint_6h/sibling/residual_train_ids.json \
  --model-file research/sprint_6h/sibling/compatibility/model.txt \
  --output research/sprint_6h/sibling/residual_support

python scripts/train_sibling_matcher.py fit-residual \
  --pairs DEV_PAIRS --manifest DEV_MANIFEST --truth DEV_TRUTH \
  --reference-ids research/sprint_6h/sibling/residual_train_ids.json \
  --support research/sprint_6h/sibling/residual_support/support.parquet \
  --support-manifest research/sprint_6h/sibling/residual_support/manifest.json \
  --trees 100 --threads ALLOCATED_THREADS --output research/sprint_6h/sibling/residual
```

For learned development, run `support` on the complete middle10k + selection20k
claimant universe (`learned30k`), excluding residual-fit owners, with the SAME fixed
compatibility model, then run `rescore` with its support paths, `--model-dir` the
residual directory, and a predeclared `--weight`. The output retains original
probabilities and every input candidate key. It requires a complete global
decision replay before evaluation/export; saved `accepted` states become stale.
The coordinator initially predeclared weights (.25, .50, .75). Before any learned
fit/result, development error evidence showed 456 routed false negatives below
p=.32 that an ideal .75 blend cannot ordinarily promote. The sole bounded
amendment replaced .25 with 1.0: final learned weights are (.50, .75, 1.0), plus
the .0 incumbent control. Historical endpoint diagnostics retain their original
weights. No further weight additions or audit tuning.
Audit/test roles allow label-free inference only; passing `--truth` is rejected.

For confirmation/audit and test release, select the top 20% route over the entire
declared claimant universe first using `route --pairs GLOBAL_PAIRS --manifest
GLOBAL_MANIFEST --output ROUTE_PLAN`. This command forbids role slicing. Pass
`--route-plan ROUTE_PLAN` to subsequent support/inference operations. Subset or
chunk manifests must carry `routing_universe_pair_order_sha256` from the global
plan source; complete-source manifests already carry that hash. The plan carries
the exact candidate/peer edges, its content hash, global cap and frozen model/rule
lineage. Membership is verified once per process and reused for streaming;
chunks must contain all candidate rows of each included reference. Do not run
independent top-20% selection per confirmation/audit sample or test chunk.

Optional `fit --fold F` removes every TT example touching any owner in fold F;
validation metrics require BOTH owners held. `support --training-support` uses
owner-excluded models. This is an alternative to the faster disjoint outer-owner
path, not an additional automatic experiment.

## Verification and current status

Local fixtures cover route caps, ordering and label independence, no self
support, wrong-seed hard negatives, unknown-owner exclusion, owner splitting,
Unicode preservation, missingness, and strict pair-key alignment. Twenty local fixtures pass. The current learned30k claimant / selection20k
opportunity audit passed: unchanged control .9794740786831396, fixed-other
perfect-decision ceiling +.006771317921747447, and ideal .75 endpoint replay
+.005162872824787779. These are ceilings, not achieved learned gains.
The full30k route selects 6000 references and 20457 target comparisons; only
selection20k contributes to reported metrics. The authorized single fit is complete. TT uses 231476 examples (218916 positives,
12560 natural negatives); its 50296 supervised owners overlap zero residual-fit
owners. On the same30k claimant universe, selected20k actual macro results are
.9792936385533769 at.50, .9792900391436615 at.75, and.9792541550531403 at1.0,
all below the .9794740786831396 control. This fit is rejected; no fresh
confirmation/audit labels were consumed. Promotion and full-runtime checks remain the
coordinator's shared sprint gates.

`build_role_workbenches.py` turns a verified complete 50k tune workbench into
three role-specific pair tables, manifests, ID files, country files and separate
truth files. It retains accepted decisions from the full source 50k cohort;
rivals outside each role are not silently discarded or recalculated. These
decisions remain incomplete for rivals outside the source cohort. Each role gets
its own exact expected macro metric and pair order hash, rather than joining an
old probability array. Input certification must already set
`verified_current_pipeline=true` after the coordinator's exact replay check.
The builder's optional owner-DB truth extraction uses one read-only TEMP-table
join rather than 100 repeated full scans of `owners`.

The frozen TT schema is `sibling-tt-v2-36`: original target text similarities plus
standalone numeric edit relations, phonetic/romanized names and collapsed-domain
name views. Features are symmetric in target order; original text is preserved.
`build_learned_workbench.py` removes all20k residual-fit owners from competing
claims, recalculates incumbent decisions over the complete30k held development
universe, and records the selected20k evaluation hash. Its baseline differs in
four accepted rows from inherited50k decisions; those scopes are not mixed.


## Authorized anchored pilot

The one follow-up adapter uses113 features:36 direct S1-target views, mean and
max of the36 raw target-peer views, plus frozenp1,p2,candidate rank, seed count
and max seedp1. It adds a learned raw margin to the exact frozenp2 logit once;
no standalone TT scalar enters this model. Source1 text is normalized with
normalize_name/normalize_address, matching frozen normalized target records.
Support reads no labels. Both training and early-stopping datasets receive the
same per-row frozen-logit init_score; reload inference explicitly requests
raw_score=True. Twenty-two fixtures pass, including reload/base/key/chunk parity.

The model fits residual20k and stops on the separate early10k logloss; best
iteration58. Complete30k decisions give early10k macro .9790302854941904 vs
.9789562060224467 control (+.00007407947174376876). This is tuning evidence,
not a fresh audit. The change adds zero true/false links, removes85 true and33
false links; microrecall falls .9521699091→.9497237251. India falls
−.0003055862, while US improves+.0003355244. It supplies no newly recovered
links or missing-address recoveries, so this pilot is rejected. Its selected20k
metrics remain unread. No further fit or combination variant was launched.

The initial TT supervised ledger intersects3011 of the cheap ranker's20k fit
owners. Only its first-stage training probabilities are owner-excluded; retrieval
is not cross-fitted, and the whole pipeline is not called OOF. The standalone TT
model is rejected. The anchored adapter does not consume that learned TT scalar.


## Separate reverse Source1 pilot

The coordinator authorized one new evidence branch after stopping both earlier
families. `train_reverse_adapter.py` consumes the fixed33 features in
`research/sprint_6h/evidence/reverse_s1_contract.md`, plus exact storedp1,p2 and
all-forward frozenp2 rank (-p2,targetID):36 inputs. It consumes no priorTT or
113-feature adapter output, rivalIDs, labels, or trained rival scores.

A strict keyed join requires exactly the existing routed candidate keys. It
checks completed fullTRAIN/test competitor counts, source3TSV hashes, canonical
DF/query policy, binary index SHA, immutable build snapshot, normalization/code
hashes and complete self-excluded rival diagnostics. FullTRAIN competitor text
is label-free; officialTEST uses its own separate index. Both role route universes
remain unchanged: fitting20k ownroute, held30k pinned globalroute.

The same anchored parameters and unit strength1 are predeclared, with only
residual20k fitting and early10k stopping/pilot evaluation. Selection20k remains
withheld. Twenty-five fixtures pass; integration is ready, but no reverse model
fit has started while keyed reverse features are being generated.
