# Project Idea — Business Entity Resolution

## One-line summary

Match each business record in a reference source (Source 1) to the records that
describe the **same real-world business** in two other sources (Source 2 and
Source 3), at scale, under a single accuracy metric — using only the provided
data.

## The problem

Businesses appear in multiple databases with inconsistent, noisy, or incomplete
descriptions. The same shop may be listed with a misspelled name, a swapped
address field, an abbreviation, a different legal suffix, or no address at all.
Conversely, two **different** businesses can share a city and a generic word
("Lille Ecole SAS") and look deceptively alike.

The task is to decide, for every reference business, which records in the other
sources are the *same entity* — and which are not — and to say so even when the
correct answer is "no match anywhere."

This is an **entity resolution** (record linkage / deduplication-across-sources)
problem, not a search or recommendation problem.

## The goal

For **every** one of the **1,732,544** Source 1 test records, produce the set of
matching Source 2 / Source 3 record IDs (possibly empty). The objective is to
**maximize macro-averaged F₀.₅** across all Source 1 entities.

* **Macro** — the score is computed per Source 1 entity, then averaged. Every
  reference business counts equally, including those with no true match.
* **F₀.₅** — weights precision higher than recall:
  `F0.5 = 1.25·TP / (1.25·TP + FP + 0.25·FN)`.
  A false link costs more than a missed link, so the system must be **precise**.
* **Singletons** — an entity with no true match scores **1 only if the
  prediction is empty**. Predicting a spurious link for a singleton is heavily
  penalized, so knowing when *not* to match is as important as matching.

The practical target is a leaderboard (portal) score high enough to clear the
qualification cutoff.

## The data

Seven organizer TSVs, split into training and test:

| Split | Source 1 (reference) | Source 2 | Source 3 | Ground truth |
|---|---|---|---|---|
| Train | ~2.2M | ~5.0M | ~5.3M | ~7.6M links |
| Test  | 1,732,544 | large | large | **not provided** |

* Each record has an `entity_id`, `business_name`, `business_address`, and
  `country`.
* **Training covers only the US and India** and is labeled (~3.46 links per S1,
  ~5.6% of S1 have no match).
* **Test adds France** (~260k S1) which has **no labels at all** — French
  accuracy could not be measured directly from the supplied labels.
* A key structural fact: in training, **no target record ever has more than one
  owner** (0 / 7.6M). This supported testing one-owner-per-target as an inference policy;
  it was an observed training property, not access to unseen test truth.

### Why it is hard

Observed noise included leetspeak, field swaps,
abbreviations, missing addresses, and legal-form changes. France is especially
treacherous because names follow a "City + generic + legal form" pattern shared
by many *distinct* businesses, so a naive same-name-same-city match over-accepts
false links.

## The approach

A multi-stage pipeline that balances **recall** (find the candidates) with
**precision** (decide carefully):

1. **Normalization & preprocessing** — clean, accent-fold, and standardize
   name/address text; phonetic views for transliteration.
2. **Blocking / candidate generation** — bounded disk indexes (exact-name,
   core-token, address, digit, character, BM25-style postings) retrieve a small
   set of plausible candidates per reference, so we never compare all pairs.
3. **Pair features** — token-set and sorted-string similarities, plus
   **rare-token soft alignment** evidence that weights distinctive words above
   shared cities/generic terms and tolerates one-character typos.
4. **Learned matcher** — a gradient-boosted (LightGBM) model scores each
   candidate pair; a neural re-ranking head adds context.
5. **Decision & ownership** — a decision policy applies thresholds and enforces
   the one-owner-per-target constraint, resolving contested targets and
   protecting singletons.

Everything runs **offline on the provided data only** — no external identity
lookup, no geocoding, no hosted-model calls, no test pseudo-labels. Countries
remain unrestricted strings (no country-specific word lists or rules).

## Deliverables

* `matching_results.tsv` — the predicted match set for every test Source 1 ID.
* `candidate_pairs.tsv` — the candidate pairs the matcher actually scored.

Both passed the official and strict format/ID validators before release.
Separate diagnostics checked target ownership.

## Final status

* Best reported portal score: **0.970** (`submission_05`).
* Best verified local fresh-audit macro F0.5: **0.9811416633**, from a later hybrid that was not exported as a completed submission.
* Last supplied qualification cutoff: **0.989279**; the project did not demonstrate that score.
* All Azure experiment resources were deleted and model work stopped.

The [project retrospective](docs/PROJECT_RETROSPECTIVE.md) describes the implemented system and results. Local audit scores did not measure France, and no universal local-to-portal correction was established.

## Ground rules

* Use only the provided data; no external augmentation or LLM labeling.
* Never tune on a consumed audit set; France is never labeled.
* Preserve frozen model artifacts and existing submissions.
* Keep the pipeline reproducible: versioned statistics, hashed manifests, and
  validated outputs.
