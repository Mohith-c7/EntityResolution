"""Versioned feature order and definitions shared by training and prediction."""

FEATURE_VERSION = "pairwise-v1-36"
FEATURE_NAMES = (
    "name_jaro_winkler", "name_levenshtein", "name_token_sort", "name_token_set",
    "name_partial", "name_jaccard", "name_length_ratio", "name_character_cosine",
    "address_jaro_winkler", "address_levenshtein", "address_token_sort", "address_token_set",
    "address_partial", "address_jaccard", "address_length_ratio", "address_numeric_overlap",
    "digit_token_jaccard", "exact_numeric_match", "postcode_agreement", "street_number_match",
    "exact_name", "exact_name_core", "any_name_token_match", "name_token_in_address",
    "address_token_in_name", "country_exact_match",
    "candidate_address_missing", "postcode_missing", "country_missing",
    "source1_name_length", "candidate_name_length", "source1_address_length", "candidate_address_length",
    "blocking_score", "source_is_s2", "source_is_s3",
)
assert len(FEATURE_NAMES) == 36
FEATURE_VERSION_V2 = "pairwise-v2-50"
FEATURE_NAMES_V2 = FEATURE_NAMES + (
    "canonical_name_sort", "canonical_name_set", "canonical_core_exact", "canonical_name_cosine", "canonical_name_length_ratio",
    "query_family_log_frequency", "candidate_alias_confidence", "candidate_alias_log_support",
    "address_ascii_containment", "first_numeric_canonical_exact", "numeric_canonical_jaccard",
    "name_weighted_jaccard", "address_weighted_jaccard", "strongest_shared_address_idf",
)
assert len(FEATURE_NAMES_V2) == 50
FEATURE_VERSION_V3 = "pairwise-v3-65"
FEATURE_NAMES_V3_EXTRA = (
    "num_containment", "num_single_substitution", "num_disjoint_contradiction",
    "num_left_only", "num_right_only", "phonetic_name_set", "phonetic_name_sort",
    "phonetic_char_jaccard", "concat_ratio", "concat_partial",
    "cand_name_min_log_df", "ref_name_min_log_df", "addr_left_unmatched_idf",
    "addr_right_unmatched_idf", "addr_distinct_mismatch",
)
FEATURE_NAMES_V3 = FEATURE_NAMES_V2 + FEATURE_NAMES_V3_EXTRA
assert len(FEATURE_NAMES_V3) == 65


def names_for_version(version):
    if version == FEATURE_VERSION:
        return FEATURE_NAMES
    if version == FEATURE_VERSION_V2:
        return FEATURE_NAMES_V2
    if version == FEATURE_VERSION_V3:
        return FEATURE_NAMES_V3
    raise ValueError(f"Unsupported feature version: {version}")

DEFINITIONS = {
    "similarities": "All similarities are in [0,1]; zero when either field is empty. RapidFuzz implements edit and token metrics.",
    "name_character_cosine": "Binary cosine over the union of character 3- and 4-gram sets of accent-folded names; no fitted vectorizer.",
    "numeric": "Digit Jaccard distinguishes name and address tokens. Exact numeric agreement requires nonempty sets.",
    "postcode_agreement": "Overlap of uncertain address numeric tokens of length >=4, without geographic validation.",
    "street_number_match": "Agreement of the first ordered address numeric token. A proxy, not an identified street component.",
    "missingness": "Candidate address missing means empty normalized text; postcode and country flags indicate either side missing.",
    "lengths": "Lengths of primary normalized strings, before accent folding.",
    "country_exact_match": "Equality of nonempty normalized strings; no country vocabulary or categorical encoding.",
    "metadata": "Only fused blocking score and S2/S3 flags enter the model; IDs, folds and labels never do.",
    "flags": "Exact equality and token-overlap flags require nonempty evidence; cross-field flags use whitespace-token intersections.",
    "v3_numeric": "Zero-stripped standalone address numbers only in the additive view. Containment and one-position substitution compare unmatched tokens of length >=2. Missing numbers are not contradictions; original alphanumeric evidence remains in legacy columns.",
    "v3_phonetic": "Approximate alternate names from Python Unicode character names, consonant/vowel simplification and contextual leetspeak folding. Preserve original text. Compare token-set, token-sort and character trigrams; collapse spaces for ratio/partial ratio. Candidate domain tokens are omitted from its collapsed view.",
    "v3_rarity": "Log(1 + minimum target-source name-token document frequency), with Latin accent folding; label-free and zero for empty/absent tokens.",
    "v3_address": "Fraction of each side's nondecimal address-token IDF absent on the other side. Distinctive mismatch requires unmatched non-name tokens of length >=3 and IDF >=6 on both sides. Values are -1 if either side has no such address tokens.",
}
