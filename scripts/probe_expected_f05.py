"""Test calibrated per-reference F0.5 decisions on the context selection split."""
import json
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow as pa
from sklearn.isotonic import IsotonicRegression

from train_scale_model import predict, score, write_json, digest
from src.model.expected_f05 import choose_expected_f05


def main():
    pa.set_cpu_count(1)
    root=Path("models/next_round/context_model");out=Path("models/next_round/expected_f05")
    if out.exists():raise FileExistsError(out)
    out.mkdir()
    split=json.loads((root/"sampled_references.json").read_text())
    calibration_ids=[r["entity_id"] for r in split["stage2_earlystop"]]
    selection_ids=[r["entity_id"] for r in split["stage2_select"]]
    allowed=set(calibration_ids+selection_ids)
    if len(allowed)!=30000:raise ValueError("Calibration and selection overlap")
    names=json.loads((root/"protocol.json").read_text())["feature_names"]
    frames=[];refs={}
    for path in sorted(Path("models/next_round/context_cache").glob("*.parquet")):
        f=pd.read_parquet(path,columns=[*names,"source1_entity_id","label"])
        frames.append(f[f.source1_entity_id.isin(allowed)])
        metadata=json.loads((Path("models/scale_v1_run/cache_tune")/(path.stem+".json")).read_text())
        for row in metadata["entity_rows"]:
            if row["entity_id"] in allowed:refs[row["entity_id"]]=row
    frame=pd.concat(frames,ignore_index=True);del frames
    calibration=frame[frame.source1_entity_id.isin(calibration_ids)].copy()
    selection=frame[frame.source1_entity_id.isin(selection_ids)].copy();del frame
    model=lgb.Booster(model_file=str(root/"model.txt"))
    chosen=json.loads((root/"report.json").read_text())["selection"]
    weight=chosen["context_weight"]
    calibration_p=(1-weight)*calibration.ctx_probability.to_numpy()+weight*predict(model,calibration[names].to_numpy(dtype=np.float32),4)
    selection_p=(1-weight)*selection.ctx_probability.to_numpy()+weight*predict(model,selection[names].to_numpy(dtype=np.float32),4)
    calibrator=IsotonicRegression(out_of_bounds="clip").fit(calibration_p,calibration.label)
    selection_calibrated=calibrator.predict(selection_p)
    positions={e:i for i,e in enumerate(selection_ids)}
    group=selection.source1_entity_id.map(positions).to_numpy(dtype=np.int32)
    expected=np.array([len(refs[e]["true_ids"]) for e in selection_ids]);country=np.array([refs[e]["country"] for e in selection_ids])
    label=selection.label.to_numpy(dtype=np.uint8)
    baseline=score(selection_p,chosen["threshold"],label,group,expected,country)[0]
    missing_rate=(sum(len(refs[e]["true_ids"]) for e in calibration_ids)-int(calibration.label.sum()))/len(calibration_ids)
    # All candidate lists have 40 rows. Preserve their original row positions for scoring.
    if not (np.bincount(group,minlength=len(selection_ids))==40).all():raise ValueError("Expected the fixed 40-candidate development cache")
    positions_matrix=np.argsort(group,kind="stable").reshape(len(selection_ids),40)
    results=[]
    for calibrated,probabilities in ((False,selection_p),(True,selection_calibrated)):
        for rate in (0.,missing_rate):
            selected=np.zeros(len(selection),dtype=bool)
            for start in range(0,len(selection_ids),250):
                locations=positions_matrix[start:start+250]
                decisions,_=choose_expected_f05(probabilities[locations],rate)
                selected[locations.ravel()]=decisions.ravel()
            metrics=score(selected,.5,label,group,expected,country)[0]
            results.append({"calibrated":calibrated,"missing_rate":rate,**metrics})
    eligible=[r for r in results if r["micro_precision"]>=baseline["micro_precision"]-.002
              and r["singleton_false_positives"]<=baseline["singleton_false_positives"]]
    best=max(eligible,key=lambda r:r["macro_f05"]) if eligible else None
    report={"status":"exploratory_selection_only","baseline":baseline,"trials":results,"selection":best,
        "macro_gain":best["macro_f05"]-baseline["macro_f05"] if best else None,
        "calibration_entities":len(calibration_ids),"selection_entities":len(selection_ids),
        "assumptions":"Independent calibrated candidate labels; optional independent Poisson missed-link count. Real entity variants are dependent, so utility optimality is only under this model.",
        "calibration_caveat":"Uses the stage-two early-stopping set, not a pristine calibration set. Final acceptance requires a new audit.",
        "fresh_audit_evaluated":False,"submission_generated":False}
    write_json(out/"calibration.json",{"x":calibrator.X_thresholds_.tolist(),"y":calibrator.y_thresholds_.tolist(),"model_sha256":digest(root/"model.txt")})
    write_json(out/"report.json",report);write_json(Path("reports/experiments/round2/expected_f05.json"),report)
    print(json.dumps(report),flush=True)


if __name__=="__main__":main()
