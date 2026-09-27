# Submission04 ownership diagnostic

Label-free census of the preserved Submission04 accepted links. No submission
TSVs, release rules, models, or indices were modified.

The complete accepted output has 1,198 targets owned by more than one Source1:
966 France, 176 India, 56 US. There are 2,425 accepted claims on those targets,
with 1,227 excess claims and at most five owners per target. There are no
cross-country duplicate targets.

`comparison_v1/comparison.json` compares those exact target IDs with the existing
`cursor_evaluation.admissible_lex` mechanism. It proposes ordinary accepted links
and the original rank-one rescue, then selects one owner per target. It accepts
1,198 claims, removes 1,227, adds zero, and leaves zero duplicate targets in the
reported set. Removals are 995 France, 176 India, and 56 US; 1,221 are below .83
and six are exact high-score ties. France has 989 accepted low-band duplicate
claims, including 476 rescues promoted after a higher original candidate lost
competition. This is a mechanism diagnostic, not evidence of a quality gain.

The comparison reuses the earlier sealed complete Submission04 floor cache on
VM2, with exact byte reconstruction of the accepted TSV. Both the eligible-pair
file and original decision masks were verified against their SHA256 seals.
All 5,173 above-floor claims for the 1,198 reported target IDs are included. The
retained graph includes every above-floor pair for their 4,876 claimants (2,411
accepted duplicate owners plus 2,465 external claimants): 21,739 pairs total.
This makes original rank-one eligibility and competition exact for reported
targets. Other target IDs are deliberately excluded from comparison results.
The floor is the frozen .6999999999999998 threshold, preserving floating-point
border cases around .7. No truth files were read.

`diagnose.py` creates the complete census and accepted duplicate-pair table.
`compare.py` validates the floor cache and runs the bounded comparator. Local
`comparison_v1/duplicate_target_claims.parquet` contains scores, original and
comparator decisions, candidate order, and countries; `needed_owner_pairs.parquet`
contains the complete retained owner graphs. Census took 12.62 seconds and
comparison took 8.04 seconds, excluding transfer time. The work finished within
the requested 20-minute limit.

The separate research helper `../post_selection_exclusivity.py` calls unchanged
control decisions first and arbitrates only actually selected links, choosing
highest probability then ascending Source1 ID for exact ties. It cannot select
a new link or alter fallback eligibility. Three fixture tests confirm that an
unselected higher-probability claim cannot steal a selected target, unrelated
fallback rescues stay accepted, and ties are deterministic.

`post_selection_v1/report.json` applies it to the same sealed original selections.
It removes1,227 claims, adds0, and yields zero duplicate targets, while preserving
all5,769,736 accepted links outside the duplicate-target set. The hypothetical
total accepted count is5,770,934. On this particular subset its winners equal
the admissible comparator's winners; the mechanisms can differ on other targets
because admissible preselection changes rescue proposals. No policy is adopted
from this diagnostic, and no quality gain is claimed.
