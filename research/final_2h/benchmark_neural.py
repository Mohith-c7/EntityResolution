"""Finish two actual neural runtime blocks, including both ID validators.

This measures the selected frozen neural candidate only; it does not read labels
or create a full-test submission. Throughput projections are explicitly separate
from measured stage timings.
"""
import json
import math
from pathlib import Path
import sys
import time
import lightgbm as lgb
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'research/sprint_6h')]
import build_neural_submission as neural_export
import build_gated_submission as export
from evaluate_sprint import check_runtime

B=Path('research/sprint_6h/release/neural_v2_actual_blocks')
OUT=Path('research/final_2h/neural_runtime_report_v3')
FREEZE=Path('reports/sprint_6h/neural_v2_final_audit_freeze.json')


def main():
    if OUT.exists():raise ValueError('Use unused benchmark directory')
    OUT.mkdir(parents=True)
    freeze=json.loads(FREEZE.read_text());plan=json.loads((B/'plan.json').read_text())
    fixed=OUT/'fixed_validator';fixed.mkdir();fixed_test=fixed/'test';fixed_test.mkdir()
    (fixed_test/'test_source1.tsv').write_text('entity_id\tbusiness_name\tbusiness_address\tcountry\n')
    for src in [2,3]:(fixed_test/f'test_source{src}.tsv').symlink_to((ROOT/f'dataset/test/test_source{src}.tsv').resolve())
    export.export_outputs([],pd.DataFrame(columns=export.PAIR_KEYS+export.SCORE_COLUMNS),np.zeros(0,dtype=bool),fixed)
    fixed_validation=export.validate_outputs(fixed,fixed_test)
    if fixed_validation['strict']!=fixed_validation['official'] or fixed_validation['strict']!='PASS':raise ValueError('Fixed validator failure')
    timings=[];model=lgb.Booster(model_file='research/sprint_6h/sibling/neural_v2/adapter.txt')
    if model.feature_name()!=neural_export.nn.FEATURES:raise ValueError('Feature schema differs')
    for number in [0,1]:
        block=B/('block_'+str(number));seconds={}
        start=time.perf_counter()
        frame=pd.read_parquet(block/'pairs.parquet')
        manifest=json.loads((block/'manifest.json').read_text())
        if export.digest(block/'pairs.parquet')!=manifest['pairs_sha256']:raise ValueError('Block seal differs')
        refs=json.loads((block/'references.json').read_text())
        feature,fm=neural_export.nn.load_features(block/'combined_neural')
        seconds['reads']=time.perf_counter()-start
        measured=json.loads((block/'feature_timing.json').read_text())
        seconds['lookups']=measured['feature_seconds']
        seconds['normalization']=0.0 # Included in the immutable reverse-builder's lookup total.
        seconds['gate_or_features']=measured['join_seconds']
        score=json.loads(Path(f'models/sprint_6h/neural/runtime_block_{number}/manifest.json').read_text())
        start=time.perf_counter(); changed=neural_export.apply_adapter(frame,feature,model)
        seconds['predict']=score['seconds']+time.perf_counter()-start
        start=time.perf_counter()
        chosen,lost=export.load_original_decide(ROOT/'scripts/run_frozen_pipeline.py')(changed,changed,freeze['decision_config'])
        seconds['global_decision']=time.perf_counter()-start
        dest=OUT/('block_'+str(number));dest.mkdir()
        start=time.perf_counter()
        (dest/'diagnostic.json').write_text(json.dumps({'routed':len(feature),'chosen':int(chosen.sum()),'ownership_lost':int(lost.sum())})+'\n')
        seconds['diagnostic_write']=time.perf_counter()-start
        start=time.perf_counter();statistics=export.export_outputs(refs,changed,chosen,dest)
        seconds['tsv_export']=time.perf_counter()-start
        test=dest/'test';test.mkdir()
        source=test/'test_source1.tsv';pd.DataFrame(refs).to_csv(source,sep='\t',index=False)
        for src in [2,3]:(test/f'test_source{src}.tsv').symlink_to((ROOT/f'dataset/test/test_source{src}.tsv').resolve())
        validation=export.validate_outputs(dest,test)
        if validation['strict']!=validation['official'] or validation['strict']!='PASS':raise ValueError('Actual block validator failure')
        seconds['strict_validator']=validation['stage_seconds']['strict_validator']
        seconds['official_validator']=validation['stage_seconds']['official_validator']
        countries={k:int(v) for k,v in pd.DataFrame(refs).country.str.strip().str.casefold().value_counts().items()}
        if set(countries)!={'india','us','france'}:raise ValueError('Missing country in actual block')
        timings.append({'references':len(refs),'pairs':len(frame),'routed_pairs':len(feature),
            'country_stratified':True,'countries':countries,'seconds':seconds,'validation':validation,
            'statistics':statistics,'scoring_manifest_sha256':export.digest(block/'combined_neural/manifest.json')})
        (dest/'measured_seconds.json').write_text(json.dumps(seconds)+'\n')
    refs=plan['full_references'];pairs=plan['full_pairs'];routed=plan['routed_pairs']
    # Observed 16-worker exact same builder:123600pairs/429.06s. Use the
    # slower of that rate and 70% of measured block single-worker throughput.
    lookup_serial=max(x['seconds']['lookups']/x['routed_pairs'] for x in timings)
    lookup_rate=min(123600/429.06,16*.70/lookup_serial)
    nn_rate=min(x['routed_pairs']/x['seconds']['predict'] for x in timings)
    projection={'full_route_setup':plan['full_route_setup_seconds'],
        'features_16_workers':routed/lookup_rate,
        'neural_and_adapter_single_mac':routed/nn_rate,
        'source_serialization_and_transfer_allowance':600.,
        'index_and_asset_integrity_allowance':120.,
        'full_reads':pairs*max(x['seconds']['reads']/x['pairs'] for x in timings),
        'feature_join':routed*max(x['seconds']['gate_or_features']/x['routed_pairs'] for x in timings),
        'global_decision':pairs*max(x['seconds']['global_decision']/x['pairs'] for x in timings)*math.log2(pairs)/math.log2(400000),
        'export':refs*max(x['seconds']['tsv_export']/x['references'] for x in timings),
        'validators_fixed':sum(fixed_validation['stage_seconds'].values()),
        'validators_reference_processing':sum(refs*max(max(0,x['seconds'][v]-fixed_validation['stage_seconds'][v])/x['references'] for x in timings)
            for v in ['strict_validator','official_validator'])}
    eta=sum(projection.values())*1.15
    report={'mode':'R2','measured':True,'candidate_frozen_sha256':export.digest(FREEZE),
        'synthetic_fixture_runtime':False,'full_test_eta_supported':True,
        'estimated_processing_seconds':eta,'processing_budget_seconds':10800,
        'safety_margin_fraction':.15,'blocks':timings,'passed':eta<=10800,
        'full_references':refs,'full_pairs':pairs,'routed_pairs':routed,
        'projection_seconds':projection,'runtime_workers':16,'fixed_validator_evidence':fixed_validation,
        'scope':'Actual full candidate populations of two country-stratified 10k blocks. Block-local decisions measure runtime, not full-test ownership quality.',
        'timing_note':'Normalization and candidate record reads are included in the unchanged reverse-feature builder total under lookups; zero is not an omitted normalization job.',
        'benchmark_code_sha256':export.digest(__file__)}
    checked=check_runtime(report,export.digest(FREEZE))
    report['acceptance']={'passed':checked['passed'],'failures':checked['failures']}
    (OUT/'runtime.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'estimated_processing_seconds':eta,'passed':report['acceptance']['passed'],'projection':projection}),flush=True)


if __name__=='__main__':main()
