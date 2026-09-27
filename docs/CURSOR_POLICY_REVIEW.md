# Independent Cursor ownership review

Reviewed Cursor ownership commit `608749aa62af4c0f73f513b86e87baf82641cf86` on local branch `retrieval-improve-0927`. Retrieval parent: `917264e494aab82e11cd3795903382217d9139e5`. Both remain local; the subsequently published token feature implementation is a separate branch and does not contain these policies.

Cursor's flags-off rule exactly matches frozen control on every row of the existing 1,200,000-pair, 30,000-reference development graph. Its tie-only, rank-one-coverage-only and combined policies, plus an independently specified admissible-proposal comparator, all change **zero links** on both original frozen probabilities and current neural-v2 probabilities. Every policy has zero collision targets on this graph. Paired macro F0.5 delta and 95% bootstrap interval are exactly `0` and `[0, 0]` on both exposed development cohorts.

| Probability model | Early 10k F0.5 | Selection 20k F0.5 | Ownership gain |
| --- | ---: | ---: | ---: |
| Frozen original | 0.9789562060 | 0.9794740787 | 0 |
| Neural v2 | 0.9797462385 | 0.9800089319 | 0 |

The graph contains all 40 candidates per reference for the declared existing 30k universe, including lower-score rows. Control accepted links: 99,342 original and 99,294 neural-v2. Scores reproduce the existing independent development reports. Labels were read only after each policy made predictions, in early10k then selection20k order. No fresh confirmation, final audit or new labels were used. This comparison cannot certify full official-test claimant density.

## Mechanism tests

`tests/test_cursor_policy_review.py` independently checks deterministic multiway ties, one accepted owner for a strong-versus-rescue target, empty inputs and ineligible rivals. Against the immutable Cursor snapshot, five tests pass and two defects are reproduced as expected failures:

- **An unaccepted claim suppresses a viable owner.** A has Y at 0.80 and X at 0.75; B has only X at 0.72. Cursor considers A-X a rival above `t_first` and discards B-X. It later discards A-X because A-Y is its rank-one rescue. X ends with no owner. A fixed proposal pool would retain A-Y and B-X.
- **Empty input crashes.** The rule indexes the first entry of an empty `order` array. The frozen evaluator's control and independent comparator handle empty input.

Cursor's existing two disk-retrieval tests pass. The new conjunction engages when an exact-core block reaches `path_top_k`, combines rare name terms with address digits and adds a `name_number` path. Its fixture verifies path membership and retrieval of the correct-number record. It does not prove additional shortlist recall versus existing numeric and lexical paths. The ordering test directly queries SQLite without `ORDER BY`, so it does not validate the new query implementation. No retrieval score gain is claimed from these tests.

## Artifacts and reproduction

Evaluator: `research/final_2h/cursor_evaluation.py`. Local reports: `research/final_2h/cursor_results/results/{frozen_original,neural_v2}/`; VM1 reports: `/mnt/er/cursor-review/results/`. Reports include exact pair/frozen/policy/evaluator hashes, per-policy runtime, paired metrics and changed-entity lists. Cursor policy snapshot is `/mnt/er/cursor-review/code/cursor_ownership_608749a.py`, exported read-only with `git show`.

```bash
CURSOR_POLICY_PATH=/tmp/cursor_ownership_608749a.py \
  python -m pytest tests/test_cursor_policy_review.py -q

python research/final_2h/cursor_evaluation.py \
  --repo /mnt/er/entity \
  --pairs /mnt/er/cursor-review/inputs/research/sprint_6h/sibling/learned30k/pairs.parquet \
  --policy /mnt/er/cursor-review/code/cursor_ownership_608749a.py \
  --frozen /mnt/er/cursor-review/code/frozen.json \
  --workbenches /mnt/er/cursor-review/inputs/research/sprint_6h/sibling/workbenches50k \
  --output /mnt/er/cursor-review/results/new_original_run --model frozen_original
```

For neural-v2 use `sibling/neural_v2_scored30k/pairs.parquet`, a new output directory and `--model neural_v2`. The run refuses overwritten outputs, incorrect 10k/20k reference sets, incomplete 30k claimant coverage, unsealed development manifests, changed truth/country hashes or a different frozen control. It deliberately has no confirmation/audit mode.

Token feature review was completed separately. Corrected branch `cursor/token-alignment-fix`, commit `ee1e0b5b4c5ffcabc5f3fe54993fdd7e9d3908e1`, fixes frequency-view consistency, provenance and fuzzy cap selection with 13 passing tests. See `docs/CURSOR_TOKEN_ALIGNMENT_V2_REVIEW.md` on that branch. No branch was merged into the parent's worktree.
