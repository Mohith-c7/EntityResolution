"""Apply pure exclusivity to the verified original selections on duplicate targets."""
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    helper = ROOT/'research/final_2h/post_selection_exclusivity.py'
    spec = importlib.util.spec_from_file_location('pure_post_selection',helper)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    source = HERE/'comparison_v1/duplicate_target_claims.parquet'
    frame = pd.read_parquet(source)
    original = frame.original_accepted.to_numpy()
    final, removed = module.exclusive_selected(frame, original)
    admissible = frame.admissible_accepted.to_numpy()
    census = json.loads((HERE/'accepted_ownership_census.json').read_text())
    frame['post_selection_accepted'] = final
    frame['post_selection_removed'] = removed
    report = {'status':'complete', 'scope':'Label-free pure post-selection exclusivity on the verified original Submission04 selections',
      'unchanged_control_selection_source':'Sealed original decision masks with exact complete Submission04 accepted TSV reconstruction',
      'original_duplicate_targets':census['targets_owned_by_multiple_Source1'],
      'original_accepted_claims_on_duplicate_targets':int(original.sum()),
      'post_selection_accepted_claims_on_duplicate_targets':int(final.sum()),
      'post_selection_remaining_duplicate_targets':int((frame.loc[final].candidate_entity_id.value_counts()>1).sum()),
      'removed_claims':int(removed.sum()), 'added_claims':int((final & ~original).sum()),
      'removed_by_country':dict(Counter(frame.loc[removed,'country'])),
      'original_total_links':census['accepted_links'], 'post_selection_total_links':census['accepted_links']-int(removed.sum()),
      'unrelated_original_accepted_links_preserved':census['accepted_links']-int(original.sum()),
      'removed_original_fallback_rescues':int((removed & ~frame.original_rank_one.to_numpy() & (frame.probability.to_numpy()<.83)).sum()),
      'remaining_original_fallback_rescues_on_duplicate_targets':int((final & ~frame.original_rank_one.to_numpy() & (frame.probability.to_numpy()<.83)).sum()),
      'different_reported_target_decisions_vs_admissible':int((final!=admissible).sum()),
      'comparison_scope':'Pure post-selection never changes proposal eligibility or promotes fallback links. Admissible preselection changes rescue eligibility; equal decisions on this duplicate-target subset do not establish policy equivalence elsewhere.',
      'labels_read':False, 'output_TSVs_modified':False, 'release_policies_modified':False,
      'input_sha256':{'helper':sha(helper),'diagnostic_scores':sha(source),'census':sha(HERE/'accepted_ownership_census.json')},
      'baseline_matching_sha256':census['matching_results_sha256']}
    out=HERE/'post_selection_v1'; out.mkdir(exist_ok=False)
    frame.to_parquet(out/'duplicate_target_decisions.parquet',index=False)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)

if __name__=='__main__':main()
