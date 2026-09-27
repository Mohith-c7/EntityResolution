# Bounded reverse S1 text evidence

`scripts/reverse_competition.py` implements the fixed contract in
`research/sprint_6h/evidence/reverse_s1_contract.md`. It retrieves temporary S1
text rivals for existing routed S2/S3 targets; it never creates forward target
candidates, consumes ownership labels, or scores rivals with a trained model.

Training and test indexes are separate. Known, nonempty country labels query
their own country; a nonempty label absent from the indexed corpus gives no
evidence. Only missing
country uses global DF/postings. Eligible term postings are complete, but the
field/joint shortlist and cached raw top eight are bounded approximate
retrieval. Their failure to return a rival does not prove uniqueness. Own S1 is
removed from every rival feature after cached top eight and before best three.

The output feature parquet has exact routed pair keys and only the 33 frozen
numeric features. Rival identities and query coverage are sealed in a separate
diagnostics JSONL. Integration appends frozen p1, p2, and rank over all original
forward candidates. The builder does not read truth, accepted decisions, or
pair labels.

```sh
python scripts/reverse_competition.py build \
  --source dataset/train/train_source1.tsv --corpus train \
  --expected-sha256 591af0e1dfeb65cab71ea6ee8cb69df00f92d6ba6fa79e05746c938775d14973 \
  --output research/sprint_6h/reverse_competition/train_s1.sqlite

python scripts/reverse_competition.py features \
  --pairs research/sprint_6h/sibling/workbenches50k/residual_train/pairs.parquet \
  --manifest research/sprint_6h/sibling/workbenches50k/residual_train/manifest.json \
  --reference-ids research/sprint_6h/sibling/workbenches50k/residual_train/reference_ids.json \
  --index research/sprint_6h/reverse_competition/train_s1.sqlite \
  --target-index-prefix models/index_train \
  --output research/sprint_6h/reverse_competition/residual20k

python scripts/reverse_competition.py features \
  --pairs research/sprint_6h/sibling/learned30k/pairs.parquet \
  --manifest research/sprint_6h/sibling/learned30k/manifest.json \
  --route-dir research/sprint_6h/sibling/learned30k_route \
  --index research/sprint_6h/reverse_competition/train_s1.sqlite \
  --target-index-prefix models/index_train \
  --output research/sprint_6h/reverse_competition/learned30k

python scripts/reverse_competition.py build \
  --source dataset/test/test_source1.tsv --corpus test \
  --expected-sha256 3d4a32c54c2ca9c53fd7c2be105bf26f708f94c4d2f88eb370972a195665c2f5 \
  --output research/sprint_6h/reverse_competition/test_s1.sqlite
```

The residual route is selected once over the declared complete 20k reference
universe and saved with its manifest. Learned early/selection queries use the
existing pinned global 30k route. Subsets/chunks must inherit that global route
and include complete reference candidate groups; local rerouting is forbidden.

Observed VM1 training build: 2,206,821 rows in 239.3 seconds; 1,323,633 US and
883,188 India rows. The supplied training source SHA256 was verified before and
after construction. The binary index is 681 MiB. A 500 unique routed-target
query benchmark took 18.18 seconds (27.50 queries/second), with mean complete
posting union 4,902.6 and maximum 23,095. These are observed single-process
costs, not a full-test runtime guarantee.

Eight tests cover provenance/count checks, train/test isolation, reload,
rank-one self exclusion, target-cache reuse across owners, open/unseen country
handling, Unicode posting/DF agreement, decimal number semantics, missing
evidence sentinels, same-rival components, exact routed export keys, inherited
pinned chunk parity, label nonuse, and missing peer rejection.
