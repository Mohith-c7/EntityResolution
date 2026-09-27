"""Unchanged frozen matcher parity and candidate-context effects for fixed5k."""
import json,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'code/business_entity_resolution'))
import run_frozen_pipeline as runner
from src.evaluation.metrics import score_matches
out=Path(__file__).with_name('result_upstreamquota')
data=json.loads(Path(__file__).with_name('sample.json').read_text());truth={e:set(v) for e,v in data['truth'].items()}
config=json.loads((ROOT/'models/frozen_v4_checkpoint_repair/frozen.json').read_text())
report=json.loads((out/'report.json').read_text())
assert report['target_met'] and report['all_reported_candidate_pairs_model_scored']
keys=['source1_entity_id','candidate_entity_id']
frames={m:pd.concat([pd.read_parquet(p) for p in sorted((out/f'scored_{m}').glob('*.parquet'))],ignore_index=True) for m in ['baseline','upstreamquota']}
saved=pd.read_parquet(Path(__file__).with_name('saved_sample_scores.parquet'))
parity=frames['baseline'].merge(saved,on=keys,suffixes=('_new','_saved'),validate='one_to_one')
assert len(parity)==len(saved)==len(frames['baseline'])
comparison={name:{'exact_equal_pairs':int((parity[name+'_new']==parity[name+'_saved']).sum()),'pairs':len(parity),'max_absolute_difference':float(np.max(np.abs(parity[name+'_new']-parity[name+'_saved'])))} for name in ['first_stage','second_stage','probability']}
retained=frames['baseline'].merge(frames['upstreamquota'],on=keys,suffixes=('_baseline','_upstreamquota'),validate='one_to_one')
context={'retained_pairs':len(retained),'first_stage_changed_pairs':int((retained.first_stage_baseline!=retained.first_stage_upstreamquota).sum()),'second_stage_changed_pairs':int((retained.second_stage_baseline!=retained.second_stage_upstreamquota).sum()),'second_stage_mean_absolute_difference':float(np.mean(np.abs(retained.second_stage_baseline-retained.second_stage_upstreamquota))),'second_stage_max_absolute_difference':float(np.max(np.abs(retained.second_stage_baseline-retained.second_stage_upstreamquota)))}
def metrics(frame,keep):
    pred=frame.loc[keep].groupby('source1_entity_id',sort=False).candidate_entity_id.agg(set).to_dict()
    return score_matches(truth,pred,data['countries'])
matching={}
other_claims=pd.read_parquet(Path(__file__).with_name('other45k_strong_claims.parquet'))
for m,frame in frames.items():
    chosen,lost=runner.decide(frame,frame,config)
    matching[m]={'probability_threshold_only':metrics(frame,frame.probability>=config['threshold']),'frozen_decision_with_sample_only_competitors':metrics(frame,chosen),'sample_ownership_rejected_pairs':int(lost.sum())}
    full_claims=pd.concat([other_claims,frame[['source1_entity_id','candidate_entity_id','probability']]],ignore_index=True)
    full_chosen,full_lost=runner.decide(frame,full_claims,config)
    matching[m]['frozen_decision_with_saved_dev50k_competitors']=metrics(frame,full_chosen)
    matching[m]['dev50k_ownership_rejected_pairs']=int(full_lost.sum())
new=pd.read_parquet(out/'added_pairs.parquet');new_scored=new.merge(frames['upstreamquota'],on=keys,validate='one_to_one')
new_scored['true_link']=[c in truth[e] for e,c in zip(new_scored.source1_entity_id,new_scored.candidate_entity_id)]
new_above=new_scored[new_scored.probability>=config['threshold']]
summary={'scope':'Fixed5k frozen-matcher characterization; no model fit or selection change','baseline_score_parity':comparison,'baseline_all_scores_exact':all(v['max_absolute_difference']==0 for v in comparison.values()),'candidate_context_effect':context,'matching':matching,'added_pairs_scored':len(new_scored),'new_true_links_scored':int(new_scored.true_link.sum()),'new_pairs_above_current_threshold':len(new_above),'new_true_links_above_current_threshold':int(new_above.true_link.sum()),'new_false_links_above_current_threshold':int((~new_above.true_link).sum()),'probability_threshold':config['threshold'],'limitation':'Sample-only decisions omit other reference claims. Saved-dev50k decisions include unchanged frozen claims from the other45k exposed references and changed claims only for this5k; these are not full Source1 cohort release-gate metrics. Pair contexts were rebuilt independently from each variant candidate list.'}
new_scored.to_parquet(out/'added_pair_model_scores.parquet',index=False)
(out/'model_score_analysis.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary))
