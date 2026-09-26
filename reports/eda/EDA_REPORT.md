# Exploratory Data Analysis & Data Contracts Report
## Amazon ML Challenge 2026 — Business Entity Resolution

> **Author:** Mohith (Data, EDA & Data Contracts Lead)  
> **Date:** September 26, 2026  
> **Status:** Completed & Validated across ~26.4 Million Records  
> **Artifacts Produced:** `src/data/loader.py`, `src/data/schema.py`, `notebooks/01_eda.ipynb`, `reports/eda/EDA_REPORT.md`

---

## 1. Dataset Overview

The dataset contains seven Tab-Separated Value (`.tsv`) files totaling **~2.52 GB** and **26,435,714 records** across training and test splits.

| Dataset Name | Filename | File Size | Exact Rows | Delimiter | Primary Key / ID |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Train Source 1** | `train_source1.tsv` | 200.34 MB | 2,206,821 | Tab (`\t`) | `entity_id` (Prefix `S1-`) |
| **Train Source 2** | `train_source2.tsv` | 466.64 MB | 5,034,616 | Tab (`\t`) | `entity_id` (Prefix `S2-`) |
| **Train Source 3** | `train_source3.tsv` | 480.37 MB | 5,285,603 | Tab (`\t`) | `entity_id` (Prefix `S3-`) |
| **Train Ground Truth** | `train_ground_truth.tsv` | 121.13 MB | 2,206,821 | Tab (`\t`) | `source1_entity_id` |
| **Test Source 1** | `test_source1.tsv` | 166.91 MB | 1,732,544 | Tab (`\t`) | `entity_id` (Prefix `S1-`) |
| **Test Source 2** | `test_source2.tsv` | 485.86 MB | 4,887,273 | Tab (`\t`) | `entity_id` (Prefix `S2-`) |
| **Test Source 3** | `test_source3.tsv` | 482.56 MB | 5,082,316 | Tab (`\t`) | `entity_id` (Prefix `S3-`) |
| **TOTAL** | — | **2.52 GB** | **26,435,714** | — | — |

---

## 2. Schema Specification & Types

Every source file across both train and test strictly adheres to the identical 4-column schema:

| Column Name | Storage Type | Nullable | Example Value | Description |
| :--- | :--- | :--- | :--- | :--- |
| `entity_id` | `object` (string) | **No (0%)** | `S1-925783039` | Unique identifier prefixed by source (`S1-`, `S2-`, `S3-`) |
| `business_name` | `object` (string) | **No (0%)** | `Orelee's Barbershop` | Raw commercial / legal enterprise name |
| `business_address`| `object` (string) | **Yes (~3%)** | `1795 Westchester Drive, High Point, NC` | Raw street, city, state, postal code address |
| `country` | `object` (string) | **No (0%)** | `US`, `India`, `France` | Country name string |

The Ground Truth file schema:
| Column Name | Storage Type | Nullable | Example Value | Description |
| :--- | :--- | :--- | :--- | :--- |
| `source1_entity_id` | `object` (string) | **No (0%)** | `S1-965667` | Exact 1-to-1 match with `train_source1.tsv` entity IDs |
| `matched_entity_ids` | `object` (string) | **Yes (5.58%)** | `S2-681193310,S3-775321672` | Comma-delimited list of matching S2/S3 IDs; empty for singletons |

---

## 3. Missing Value Analysis

Full-corpus streaming scan measured missing and empty values across all columns:

| Dataset | Total Rows | Missing `entity_id` | Missing `business_name` | Missing `business_address` | Missing `country` |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Train S1** | 2,206,821 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| **Train S2** | 5,034,616 | 0 (0.00%) | 0 (0.00%) | **168,967 (3.36%)** | 0 (0.00%) |
| **Train S3** | 5,285,603 | 0 (0.00%) | 0 (0.00%) | **175,916 (3.33%)** | 0 (0.00%) |
| **Test S1** | 1,732,544 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| **Test S2** | 4,887,273 | 0 (0.00%) | 0 (0.00%) | **129,408 (2.65%)** | 0 (0.00%) |
| **Test S3** | 5,082,316 | 0 (0.00%) | 0 (0.00%) | **136,098 (2.68%)** | 0 (0.00%) |

> [!NOTE]
> `business_name` and `country` are 100% populated in all datasets.
> Only `business_address` contains missing values, consistently hovering around **2.6% to 3.4%** in Sources 2 and 3. Downstream address comparisons and blocking must implement null-safe guards.

---

## 4. ID Uniqueness and Data Integrity

- **Unique IDs**: Across all 7 datasets, the number of unique IDs equals the total row count exactly.
- **Duplicate IDs**: **0 duplicate IDs** found in any file.
- **Source ID Prefixes**:
  - `train_source1.tsv` & `test_source1.tsv`: 100% start with `S1-`
  - `train_source2.tsv` & `test_source2.tsv`: 100% start with `S2-`
  - `train_source3.tsv` & `test_source3.tsv`: 100% start with `S3-`
- **Ground Truth 1-to-1 Contract**:
  - `train_source1` entity IDs: 2,206,821
  - `train_ground_truth` source1 IDs: 2,206,821
  - Difference: **0** (perfect bijective correspondence).
- **Intra-row Ground Truth Duplicates**: **0**. No single S1 record has repeated candidate IDs in its match list.

---

## 5. Country Distribution & Critical Open-Set Finding

| Split | Dataset | US | India | France | Total Distinct Countries |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | S1 | 1,323,633 (59.98%) | 883,188 (40.02%) | **0 (0.00%)** | 2 |
| **Train** | S2 | 3,016,817 (59.92%) | 2,017,799 (40.08%) | **0 (0.00%)** | 2 |
| **Train** | S3 | 3,170,056 (59.97%) | 2,115,547 (40.03%) | **0 (0.00%)** | 2 |
| **Test** | S1 | 663,106 (38.27%) | 809,986 (46.75%) | **259,452 (14.98%)** | 3 |
| **Test** | S2 | 1,871,330 (38.29%) | 2,312,565 (47.32%) | **703,378 (14.39%)** | 3 |
| **Test** | S3 | 1,945,701 (38.28%) | 2,405,000 (47.32%) | **731,615 (14.40%)** | 3 |

> [!IMPORTANT]
> **EMPIRICALLY CONFIRMED ARCHITECTURAL CONSTRAINT:**
> **France** is completely **absent in train (0.0%)**, but comprises **14.4% to 15.0% (~1.69 million records)** of the test dataset!
> Furthermore, India becomes the majority country in test (47.3% vs 40.0% in train).
> **Rule:** Never one-hot encode country or hard-code country filters. Country matching must remain an open-set equality flag (`country_exact_match`).

---

## 6. Ground Truth Analysis & Matching Problem Structure

Detailed analysis of all 2,206,821 training labels:

### Overall Match Breakdown
- **Total S1 Entities**: 2,206,821
- **Singletons (0 Matches)**: **123,247 (5.58%)**
- **Single Match (1 Match)**: **119,157 (5.40%)**
- **Multi-Match (>1 Match)**: **1,964,417 (89.02%)**
- **Total Positive Pairs**: **7,638,365**
  - **S2 Positives**: 3,693,619 (48.36%)
  - **S3 Positives**: 3,944,746 (51.64%)
- **Mean Matches per S1**: **3.461**
- **Median Matches per S1**: **3.0**
- **Maximum Matches per S1**: **11**

### Complete Match Distribution
| Matches per S1 | Count | Percentage | Cumulative % |
| :--- | :--- | :--- | :--- |
| **0 (Singleton)** | 123,247 | 5.58% | 5.58% |
| **1** | 119,157 | 5.40% | 10.98% |
| **2** | 375,212 | 17.00% | 27.99% |
| **3** | **530,841** | **24.05% (Peak Mode)**| 52.04% |
| **4** | 484,115 | 21.94% | 73.98% |
| **5** | 321,957 | 14.59% | 88.57% |
| **6** | 164,868 | 7.47% | 96.04% |
| **7** | 63,968 | 2.90% | 98.94% |
| **8** | 18,680 | 0.85% | 99.79% |
| **9** | 4,205 | 0.19% | 99.98% |
| **10** | 534 | 0.02% | 100.00% |
| **11** | 37 | 0.002% | 100.00% |

> [!TIP]
> **Implications for Sahasra (Model & Decision):**
> 1. Over **89%** of S1 entities have multiple matches across S2 and S3, with the modal count being 3 to 4 matches.
> 2. **5.58%** are pure singletons (no matches in S2 or S3). Under the competition Macro $F_{0.5}$ metric, predicting empty for singletons earns an automatic score of **1.0**. Predicting any false positive on a singleton yields **0.0**. The threshold must be tuned high to protect singletons.

---

## 7. Duplicate Analysis (Raw Data)

Measured on 500,000 deterministic sample records per source:

| Source | Duplicate Names % | Duplicate Addresses % | Duplicate (Name + Address) % |
| :--- | :--- | :--- | :--- |
| **Train S1** | 19.42% | 1.28% | **0.00%** |
| **Train S2** | 4.10% | 4.65% | **0.05%** |
| **Train S3** | 3.71% | 4.37% | **0.03%** |

- In S1, **19.42%** of business names recur at different geographic locations (national retail, healthcare providers, banking branches).
- However, combining `business_name` + `business_address` produces a nearly **100% unique tuple**. Exact duplicate records in raw data are virtually non-existent (<0.05%).

---

## 8. Text & Length Distributions

Length statistics computed across systematic samples (50,000 records per source):

| Dataset | Name Char Mean (p95) | Name Word Mean (p95) | Address Char Mean (p95) | Address Word Mean (p95) |
| :--- | :--- | :--- | :--- | :--- |
| **Train S1** | 24.1 (37.0) | 3.6 (5.0) | 52.0 (102.0) | 8.0 (15.0) |
| **Train S2** | 25.1 (40.0) | 3.5 (5.0) | 46.1 (96.0) | 7.3 (14.0) |
| **Train S3** | 25.2 (42.0) | 3.5 (6.0) | 46.6 (90.0) | 7.1 (14.0) |
| **Test S1** | 23.9 (36.0) | 3.5 (5.0) | 57.2 (105.0) | 8.6 (16.0) |
| **Test S2** | 25.7 (42.0) | 3.6 (5.0) | 50.4 (99.0) | 7.8 (15.0) |
| **Test S3** | 25.7 (42.0) | 3.6 (6.0) | 48.8 (94.0) | 7.5 (15.0) |

- Names average **24 to 26 characters** (~3.5 words) with maximum observed length around 92 characters.
- Addresses average **46 to 57 characters** (~7 to 8.5 words) with maximum observed length up to 235 characters. Test addresses are slightly longer on average due to French and extended Indian postal addresses.

---

## 9. Real Ground Truth Match Noise Patterns

By inspecting positive pairs from `train_ground_truth.tsv`, we discovered the following real noise patterns:

| Pattern Category | Source 1 (Reference) | Matched Candidate (S2/S3) | Observed Noise Characteristics |
| :--- | :--- | :--- | :--- |
| **Legal Suffix Inversion** | `Unique Constro Pvt Ltd` | `PVT UNIQUE CONSTRO LTD` (S2) | Legal suffixes (`pvt`, `ltd`) prepended to the front instead of trailing. |
| **Punctuation & Compacting** | `Department of Aging` | `#departmentaging` (S2) | Leading `#`, whitespace collapsed, word boundaries lost. |
| **Leetspeak / OCR Noise** | `Unique Constro Pvt Ltd` | `Unique C0nstro [Pvt]` (S2) | Digit `0` substituted for letter `O`; square brackets around suffix. |
| **Typographical Errors** | `Precision Semiconductor Partners` | `Precision Semicosdutor Partners` (S2) | Letter transpositions (`c` $\rightarrow$ `s`, `n` $\rightarrow$ `d`). |
| **Multilingual Script in Address**| `...Bhubaneswar, Khordha, Orissa` | `...BHUBANESWAR, KHORDHA, ଓଡ଼ିଶା` (S2) | State name in Odia Indic script (`ଓଡ଼ିଶା`). |
| **Historical vs Modern Names** | `...Khordha, Orissa` | `...Odisha` (S2) | Historical state name `Orissa` vs modern official `Odisha`. |
| **Truncation & Dropped Words** | `Department of Aging` | `Department Of` (S3) | Severe truncation of enterprise name. |
| **Address Reordering** | `4089 Ethel Road, Bartlett, TN` | `4089-A ETHEL ROAD, TN, BARTLETT` (S2) | City (`Bartlett`) and State (`TN`) positions inverted. |
| **State Abbreviation Expansion** | `...Bristol City, VA` | `...Bristol City, Virginia` (S3) | 2-letter state code (`VA`) expanded to full name (`Virginia`). |
| **House Number Variations** | `233 Lavinia Street...` | `##233 Lavinia St...` (S3) | Punctuation prefixes (`##`) added to address numbers. |
| **Missing Address in Candidates**| `233 Lavinia Street...` | `NaN` (S3) | Address is null; matching depends entirely on name similarity. |

---

## 10. Team Handoff Specifications

### Handoff to Sanhitha (Normalization & Preprocessing)
```text
Normalization-relevant findings:
1. Name field: 100% populated. Average length 24-26 chars (3-4 words).
2. Address field: 100% populated in S1; ~3.3% missing in S2/S3. Average length 46-57 chars.
3. Country field: 100% populated. US and India in train; US, India, France in test.

Observed name variations:
- Legal suffix inversions (e.g. "PVT UNIQUE CONSTRO LTD" vs "Unique Constro Pvt Ltd").
- Leetspeak substitutions ("C0nstro" vs "Constro").
- Punctuation/hashtag concatenation ("#departmentaging" vs "Department of Aging").
- Accented vowels ("óf" vs "of").

Observed address variations:
- Indian house numbers ("H.NO.16-11-23/37/A", "KH NO. -570/13").
- Street/unit abbreviations ("RD", "ST", "AVE", "BLVD", "STE", "APT", "FL").
- Historical vs modern names ("Orissa" vs "Odisha").
- Non-Latin script suffixes (Odia, Devanagari) appearing in Indian addresses.
- Inverted city/state order.

Important character/language patterns:
- French addresses in test include accented characters (é, è, ç, à, ô).
- NFKC Unicode normalization is required.

Missing-value rates:
- Name: 0.00%
- Address: 0.00% (S1), 3.36% (S2), 3.33% (S3)
- Country: 0.00%
```

### Handoff to Harsha (Blocking & Candidate Generation)
```text
Blocking-relevant findings:
Train sizes: S1 = 2,206,821 | S2 = 5,034,616 | S3 = 5,285,603
Test sizes:  S1 = 1,732,544 | S2 = 4,887,273 | S3 = 5,082,316

Ground truth match volume:
- Total positive pairs: 7,638,365 (3,693,619 S2 + 3,944,746 S3)
- Average matches per S1: 3.461 (median: 3.0, max: 11)
- Zero-match S1 (singletons): 5.58% (123,247)
- Multi-match S1: 89.02% (1,964,417)

Important exact-match statistics (from raw sample):
- Duplicate names: ~19.4% in S1, ~4% in S2/S3
- Duplicate addresses: ~1.3% in S1, ~4.5% in S2/S3
- Duplicate (name + address): <0.05%

Crucial blocking recommendations:
1. Default TOP_K = 20 easily accommodates the maximum observed matches (11 matches).
2. Must include rare-token inverted index on normalized name and name_core to catch inverted suffixes.
3. Must include numeric/digit blocking path to catch street number/postcode matches when names are noisy.
4. Because ~3.3% of S2/S3 records have null addresses, address-only blocking paths must be complemented by name blocking paths.
```

### Handoff to Sahasra (Features, Model & Decision)
```text
Model-relevant findings:
- Positive relationships: 7,638,365
- Total S1 entities: 2,206,821
- S2 positives: 3,693,619 (48.36%)
- S3 positives: 3,944,746 (51.64%)

Label distributions:
- 0 matches (singletons): 5.58%
- 1 match: 5.40%
- 2 matches: 17.00%
- 3 matches: 24.05% (mode)
- 4 matches: 21.94%
- 5 matches: 14.59%
- 6-11 matches: 11.43%

Important class/distribution observations:
1. Target metric is Macro F0.5 (precision weighted 2x over recall).
2. Singletons score 1.0 when predicted empty, 0.0 if any false match is predicted.
3. Decision threshold must be swept higher than 0.5 (expected 0.60 - 0.75) to penalize false positives.
4. Country is an open-set string: test includes 14.5% France which is completely unseen in train.
   Feature must be boolean string equality (country_exact_match), never categorical/one-hot.
5. Address features must gracefully handle NaN (~3.3% in S2/S3) using null-safe defaults.
```

---

## 11. Data Quality Checks & Contract Verification

All data contract assertions executed and passed:
- [x] All 7 required dataset files exist with expected filenames and schemas.
- [x] 100% 1-to-1 match between `train_source1.tsv` and `train_ground_truth.tsv` S1 IDs.
- [x] Zero duplicate IDs across all 26.4 million records.
- [x] Zero intra-row duplicate candidate IDs in ground truth.
- [x] Candidate IDs strictly adhere to `S2-` and `S3-` naming convention.
- [x] No source files were modified or corrupted during inspection.
