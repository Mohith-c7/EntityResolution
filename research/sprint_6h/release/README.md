# Stored-score R1 release contract

Owner: release worker. No model inference, new retrieval, checkpoint repair,
remote redeployment, or baseline mutation is performed by these helpers.
Full test build is reserved for the coordinator-frozen candidate.

## Files

- `scripts/build_gated_submission.py`: verified chunk loading, pair-key veto
  alignment, global original/v2 decisions, exact candidate export, both validators.
- `research/sprint_6h/release/seal_reuse.py`: read-only seal of a completed baseline.
- `tests/test_build_gated_submission.py`: small offline fixtures, including real
  strict and official validator executions with ID checking.

## Complete baseline seal

Run only after VM1's existing scoring job has completed and both validators pass.
Preserve `output/submission_04/`; write the seal outside it. First collect the
native extension and both test index paths with SHA-256 in `asset_hashes.json`.
These hashes describe the surviving immutable asset snapshot; without historical
hashes they are current file identity evidence, not a retroactive proof of the
original scoring run's asset identity. Preserve provenance with the sealed files.
The helper also checks every frozen model, alias, rule and original code hash.

```sh
python research/sprint_6h/release/seal_reuse.py \
  --baseline output/submission_04 \
  --test-dir dataset/test \
  --frozen models/frozen_v4_checkpoint_repair/frozen.json \
  --asset-hashes research/sprint_6h/release/asset_hashes.json \
  --output research/sprint_6h/release/sub04_reuse_manifest.json
```

The seal requires a complete baseline report and its exact frozen descriptor,
Source1 universe-count metadata, every contiguous original chunk marker and
parquet SHA-256, full input slices including references without candidates,
valid unique pair keys and probabilities, and exact candidate export parity.
It hashes all three test files. It never writes into the baseline directory.

Pair-order SHA-256 is the hash of concatenated UTF-8 records:
`source1_entity_id + '\t' + candidate_entity_id + '\n'`, in original chunk row
order. Per-reference candidate order is recovered with original stable row order.

## Coordinator-frozen release descriptor

The exporter uses these R1 fields in the shared candidate freeze descriptor:

```json
{
  "status": "candidate_frozen",
  "frozen_at": "2026-09-27T11:37:34Z",
  "mode": "R1",
  "parent_frozen_sha256": "BASELINE_DESCRIPTOR_HASH",
  "candidate_freeze": {"path": "models/sprint_6h/SELECTED_RUN/candidate_freeze.json", "sha256": "AUDITED_FREEZE_HASH"},
  "reuse_manifest": {"path": "research/sprint_6h/release/sub04_reuse_manifest.json", "sha256": "HASH"},
  "code_sha256": {
    "scripts/build_gated_submission.py": "HASH",
    "scripts/decision_policy_v2.py": "HASH",
    "code/business_entity_resolution/src/pipeline/export.py": "HASH",
    "code/business_entity_resolution/src/utils/organizer_validate_submission.py": "HASH"
  },
  "policy": "ownership_v2",
  "decision_config": {"threshold": 0.83, "t_first": 0.7, "t_rest": 0.83, "ownership_margin": 0.0},
  "gate_table": {"path": "PAIR_KEYED_GATE_PARQUET", "sha256": "HASH", "pair_order_sha256": "REUSE_PAIR_ORDER_HASH"}
}
```

Numbers in this example illustrate the schema; they do not select or promote an
experiment. The coordinator must write the experimentally selected config and
freeze evidence. Shared split/model/schema/index/native hashes remain part of
the freeze contract governed by validation; the exporter also verifies every
asset in the reuse manifest. Pin `scripts/decision_policy_v2.py` for v2. Pin the
exporter itself for every policy. For `policy: original`, supply
`original_runner: {path, sha256}` for the exact frozen baseline runner; the
exporter executes its `decide` AST alone and avoids model/scorer imports.
The runner hash must also equal the baseline descriptor's frozen runner hash.
Both validator implementations must be pinned in `code_sha256` for every policy.

Keep the audited candidate freeze immutable. The full-test gate table is a
derived execution artifact whose hash becomes known later; add it to a separate
release execution descriptor that references `candidate_freeze`. When supplied,
the exporter verifies the candidate descriptor hash, freeze status, inherited
policy/config/time/parent fields, every release implementation hash against the
audited code hash map, and all candidate model/schema/index/native hash evidence.
It never edits the audited freeze. This is separate from the baseline parent
descriptor, which identifies the original stored scorer.

`gate_table` is optional for a control. If present it must contain **all** pair
keys exactly once with boolean `gate_veto` and string `gate_reason`. It can be
physically reordered: pair keys govern alignment. Equal lengths are insufficient.
The table's candidate order/universe hash must equal the reuse manifest.
Gate generation uses `fit_signals`, `pair_signals`, `gate_veto` from the gate
worker; the release exporter only consumes the pinned result. Record signal
state, source hashes, gate config and gate-code hashes in the frozen package.

## Build

```sh
python scripts/build_gated_submission.py \
  --release models/sprint_6h/SELECTED_RUN/release.json \
  --test-dir dataset/test \
  --output output/submission_05
```

The destination must not exist, and cannot be submission_03/submission_04 or a
child of either. Every Source1 ID appears once in both TSVs, even with no
candidates. The candidate TSV is the exact original scored candidate list;
vetoes only affect accepted matches. Global decisions use all scored claimants,
including rescue-band claims for v2. Original scores remain in baseline chunks
and copied pair diagnostics; a transient original-policy eligibility view may
use -1 for vetoes but never changes stored scores. Original policy intentionally
retains original ownership/rescue semantics; v2 is a separately evaluated policy.

Outputs include `pair_decisions.parquet`, `strict_validation.json`,
`official_validation.log`, `validation.json`, sealed manifests and a report.
The report separates validator success from unknown portal/France quality.
Only a complete report with `ready_for_upload: true`, both validator PASS statuses
and `id_checking: true` is upload eligible. The official command always has
`--check-ids`. No file is published to common output names automatically.

The current implementation loads the full score/claim table for global decisions.
Benchmark memory and complete reads, gate generation, arbitration, diagnostics,
TSV exports and validators on representative blocks before final execution. Small
fixture timing is correctness evidence and is not a full-test runtime estimate.

## Verification

```sh
/tmp/entity-review-venv/bin/python -m pytest -q tests/test_build_gated_submission.py
```

Current result: 26 fixtures passed. They cover full/empty owners, all-empty pair
universes, exact global control behavior, shuffled veto keys, preserved scores,
rescue-vs-normal arbitration, file/hash/order/count/coverage corruption, duplicate
keys, wrong gate universes, new-directory and freeze guards, exact baseline seal
candidate parity, and non-mutating sealing. Gate/ownership modules have separate
focused tests owned by their workers. No full test release has been executed.
The execution descriptor fixture also rejects a threshold changed after the
audited freeze while leaving that freeze untouched.
Copy fixtures also require unchanged evidence in the portable seal and exact
copied file bytes; either evidence edits or altered parquet bytes are rejected.

The completed baseline is preserved locally under `output/submission_04`. Its
local strict and official `--check-ids` validators both passed; see
`output/submission_04/local_preservation_report.json`. Portal quality remains
unknown. The original VM1 seal is retained as `sub04_reuse_manifest.vm1.json`
(SHA-256 `f598cfba6c03a7a5a5f2f973924c82d08f20a0976c1b87b5a4e6338c7252b0c4`).
The separate portable seal is `sub04_reuse_manifest.portable.json` (SHA-256
`ab0f23667ccd5332a2bf102aa1742bdc7891d339ee99e3de827b5232717cb82d`).
It only rebases paths and binds the existing repaired descriptor at
`research/sprint_6h/coordination/baseline_freeze/frozen.json`; original proof bytes
remain unchanged.

`verify_copied_seal.py` checked the isolated VM2 copy against both seals. The
complete `vm2_copy_verification.json` records exact byte parity for 13,938 files,
6,931 chunks, and 69,301,760 pairs. Both copied TSV hashes match the original
double-validated baseline. The source pair-order SHA-256 is
`97c9aba60de88f0026b97f546eabca72393db5d5cbd604ac96afd1d4a2c44fc1`.
This copy check performs no score recomputation or baseline writes. It derives
validation from exact byte parity and explicitly records that validators were
not rerun on VM2.

`benchmark_fixture.py` records synthetic mechanics on two blocks in
`synthetic_runtime.json`. Initial 20k/30k-reference blocks completed in
0.69s/1.08s with both validators passing; peak RSS was about 278 MiB on the Mac.
These omit full input/cache reads, record signal generation and full diagnostics
writing, and explicitly do not support a full-test ETA.

`benchmark_actual_blocks.py` consumes two disjoint actual country-stratified gate
blocks after the coordinator freezes an R1 candidate. Each block supplies hashed
Source1 rows, unchanged scores, the complete pair-keyed gate table, exact frozen
candidate SHA, and measured read/lookup/normalization/veto stages. The consumer
adds global decisions, pair diagnostics, TSV exports, and both ID validators.
It measures complete target-ID validator loading separately, extrapolates worst
serial per-pair rates with sorting growth for global decisions, adds fixed
preflight costs, and inflates the total ETA by at least 15% before comparing it
with the processing budget. The fixture tests this stage/report contract; no
actual runtime block execution or full test release has occurred yet.

`transfer_heldout.py` copied the source-sealed heldout cohort directly from VM1 to
the isolated VM2 tree using the restricted transfer identity and pinned host key.
`heldout_transfer.json` records exact checksum parity for 11 files / 599,358,558
bytes, 331,012 references and 13,240,480 pairs. Root-level pair/schema/reference
files, capture/asset/seal proofs and unchanged code were copied; feature/score
chunk directories were excluded. Original VM1 files remain intact, and labels
were not used. The manifest SHA is
`c7bbd0547faf2c3d397b55fb71b86b396fa0547480259ee93d2d7e1680f42118`.

The coordinator separately authorized pre-freeze, unlabeled French saved-claim
diagnostics. `diagnose_france_claims.py` scans all verified scores, keeps every
claim at or above the exact baseline acceptance floor across all countries, and
uses the original runner's hash-pinned `decide` AST. The independent scan retained
5,919,526 of 69,301,760 pairs, matching the gate worker's retained count/order.
`baseline_parity.full_test.json` proves exact reconstruction of the complete
baseline matching-file SHA and all 5,772,161 accepted links, including empty
Source1 rows. The floor subset preserves all high-threshold rivals and candidate
relative order. An omitted below-floor score can neither pass an acceptance
threshold nor outrank a score that can pass one; vetoed/lost claims also cannot
make a below-floor claim acceptable. Randomized threshold/tie/veto fixtures check
this equivalence.

Preparation ran with one CPU: 47 seconds to scan/filter and 91 seconds for exact
original arbitration. Its source is preserved as
`diagnose_france_claims.prepare_v1.py`; the report pins its historical hash.
Baseline global collisions are 37,919 high-threshold targets and 1,198 final
multi-owner targets, with at most five final owners. These are incumbent
mechanics, not France quality measurements. Later comparisons consume exact
French eligible pair tables and preserve skipped rows as unobserved. They write
paired baseline/gated accepted and ownership masks, original scores and changed
winner IDs to `changed_claims.parquet`; no submission TSV or portal output is
created. V3 and V4 diagnostic outputs remain distinct.
`derive_changed_winners.py` starts with every cached baseline accepted owner for
each affected target and applies the accepted-pair deltas. Its new JSON/parquet
artifacts include complete baseline/gated owner lists and unchanged co-owners.
A separate derivation manifest binds these files to the unchanged comparisons;
decisions are not rerun. A tie fixture verifies that an unchanged co-owner stays
in both complete lists.
