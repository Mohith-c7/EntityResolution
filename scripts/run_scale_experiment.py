"""Build cross-fitted data, train a larger matcher and audit a frozen selection.

Run this file from a source snapshot, with the repository as working directory.
Existing submission assets and outputs are never altered.
"""
import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""): h.update(block)
    return h.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(value, indent=2) + "\n"); temporary.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, default=Path("models/scale_v1_plan"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--idle-workers", type=int, default=8)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--submission-progress", type=Path, default=Path("output/submission_03/progress.json"))
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / "run.lock").open("a")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    scripts = Path(__file__).resolve().parent
    source_root = scripts.parent
    snapshot = json.loads((source_root / "snapshot.json").read_text())
    def verify():
        for filename, expected in snapshot["files"].items():
            if digest(source_root / filename) != expected: raise ValueError("Frozen experiment source changed")
    verify()
    os.nice(5)
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1", PYTHONUNBUFFERED="1")
    state_path = args.output / "progress.json"
    started = time.time()
    def stage(name, command):
        verify()
        state = {"status": "running", "stage": name, "pid": os.getpid(), "started_unix": started, "command": command}
        write_json(state_path, state)
        print(json.dumps(state), flush=True)
        with (args.output / f"{name}.log").open("a") as log:
            result = subprocess.run([sys.executable, *command], env=env, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            write_json(state_path, {**state, "status": "failed", "exit_code": result.returncode})
            raise RuntimeError(f"{name} failed; see its log")
    def cache(split, selection=None):
        command = [str(scripts / "build_scale_cache.py"), "--plan", str(args.plan), "--split", split,
            "--output", str(args.output / f"cache_{split}"), "--workers", str(args.workers),
            "--idle-workers", str(args.idle_workers), "--submission-progress", str(args.submission_progress)]
        if selection: command += ["--audit-selection", str(selection)]
        stage(f"cache_{split}", command)
    cache("train")
    cache("tune")
    # Keep model fitting within the available memory while inference workers run.
    while args.submission_progress.exists():
        status = json.loads(args.submission_progress.read_text()).get("status")
        if status == "complete": break
        write_json(state_path, {"status": "waiting", "stage": "waiting_for_submission_before_model_fit", "pid": os.getpid()})
        time.sleep(20)
    model_dir = args.output / "model_300k"
    report_path = model_dir / "report.json"
    if not report_path.exists():
        stage("train_model", [str(scripts / "train_scale_model.py"), "--train-cache", str(args.output / "cache_train"),
            "--tune-cache", str(args.output / "cache_tune"), "--output", str(model_dir), "--threads", str(args.threads)])
    report = json.loads(report_path.read_text())
    if not report["chosen"] or report["macro_gain"] <= 0:
        write_json(state_path, {"status": "complete_no_promotion", "stage": "tuning_rejected", "model_report": str(report_path),
                               "fresh_audit_evaluated": False, "macro_gain": report["macro_gain"]})
        return
    baseline_path = Path("models/features_v3_audit01/model.txt").resolve()
    selection = {"status": "frozen_before_audit", "model_path": str((model_dir / "model.txt").resolve()),
        "model_sha256": report["model_sha256"], "threshold": report["chosen"]["threshold"],
        "baseline_model_path": str(baseline_path), "baseline_model_sha256": digest(baseline_path), "baseline_threshold": .75,
        "tuning_report_sha256": digest(report_path), "plan_sha256": digest(args.plan / "sampled_references.json")}
    selection_path = args.output / "frozen_selection.json"
    if selection_path.exists() and json.loads(selection_path.read_text()) != selection: raise ValueError("Audit selection changed")
    write_json(selection_path, selection)
    cache("holdout", selection_path)
    audit_dir = args.output / "audit"
    if not (audit_dir / "report.json").exists():
        stage("audit", [str(scripts / "audit_scale_model.py"), "--selection", str(selection_path),
            "--cache", str(args.output / "cache_holdout"), "--output", str(audit_dir), "--threads", str(args.threads)])
    report = json.loads((audit_dir / "report.json").read_text())
    write_json(state_path, {"status": "complete", "stage": "fresh_audit_complete", "accepted": report["accepted"],
        "macro_f05": report["model"]["macro_f05"], "paired_gain": report["paired_macro_gain"],
        "seconds": time.time()-started, "leaderboard_score": None})


if __name__ == "__main__": main()
