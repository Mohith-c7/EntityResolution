# Token alignment v2 correction

Base implementation: `43e14c633f45193611c8b2ea96d094dac957a919` on `cursor/token-alignment`. Corrections: branch `cursor/token-alignment-fix`.

The original statistics counted unaccented primary strings while feature construction queried accent-folded and phonetic strings. Frequent accented streets and frequent transformed name tokens could therefore receive unseen-token weight. Version 2 counts primary document frequencies after accent folding and separate phonetic document frequencies after the existing `phonetic()` transformation. Each transformed token counts at most once per document, including when two original tokens transform to the same token.

The fuzzy matrix still has at most eight unmatched tokens per side. It now selects those tokens by descending actual view-specific IDF, breaking equal weights lexicographically. Exact matching still occurs before this cap. Reusing statistics now requires its recorded Source 1 content hash to match the supplied Source 1 file, regardless of path. The feature manifest records the statistics hash.

## API and compatibility

The entrypoint remains `build_alignment_features(left, right, statistics) -> dict[str, float]`; `FEATURE_NAMES` remains the same ordered 16 names. `FEATURE_VERSION` is `token-alignment-v2`; `STATISTICS_VERSION` is `token-alignment-stats-v2`. The existing primary fields and `df(term, country)`, `count(country)`, `idf(term, country)` behavior are retained. A third optional `view="phonetic"` argument selects the phonetic frequencies.

`AlignmentStatistics` adds `phonetic_global_df` and `phonetic_country_df`, with the same shapes as the primary fields. `scripts/build_token_alignment_features.py:build_statistics(source1_path)` returns the fully populated typed object and elapsed seconds. `to_dict()` / `from_dict()` preserve both views. Programmatic constructors used for pair features must supply the phonetic frequencies; a missing phonetic view raises a precise error instead of silently using raw-token frequencies.

Version 1 artifacts require rebuilding from supplied Source 1 records. Primary document frequencies cannot recover exact phonetic document unions when transformed tokens collide within a document; `from_dict()` therefore rejects the old version. This is a statistics artifact version change, not a feature registry or frozen model change.

## Commands

```bash
python scripts/build_token_alignment_features.py \
  --pairs pairs.parquet \
  --source1 dataset/train/train_source1.tsv \
  --source2 dataset/train/train_source2.tsv \
  --source3 dataset/train/train_source3.tsv \
  --statistics-out token_alignment_stats_v2.json \
  --output token_alignment_features_v2.parquet
```

For test inference supply all three official test TSVs and build a new statistics artifact from test Source 1. Reuse an artifact only with the identical Source 1 corpus:

```bash
python scripts/build_token_alignment_features.py \
  --pairs pairs.parquet --source1 source1.tsv --source2 source2.tsv --source3 source3.tsv \
  --statistics token_alignment_stats_v2.json --output token_alignment_features_v2.parquet

python -m pytest tests/test_token_alignment.py tests/test_token_alignment_statistics.py -q
```

The independent checks cover frequent accented streets, phonetic token collisions and document unions, global-country fallback, artifact roundtrip, rejection of a changed Source 1 corpus in the actual batch command, source relocation with identical content, distinctive-token cap ordering, and explicit rejection of a missing phonetic view. Existing six alignment behavior tests also pass. No labels are used in statistics or feature construction. No model score improvement is claimed by these correctness checks.

The batch builder still reads a whole TSV into a pandas frame and generates pair rows in one process. Large pair sets should use the parent session's bounded parallel record pipeline. This change does not add a new evaluation cohort or inspect confirmation/final audit truth.
