"""Freeze source for a reproducible larger matcher experiment, optionally launch it."""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan", type=Path, default=Path("models/scale_v1_plan"))
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    run = args.output.resolve(); snapshot = run / "source_snapshot"
    if snapshot.exists(): raise FileExistsError("Preserve the existing snapshot; use a fresh experiment directory")
    required = ["full.json", *[f"exclude_{i}.json" for i in range(5)], "counts.sqlite"]
    if any(not (args.plan / "aliases" / name).exists() for name in required):
        raise FileNotFoundError("Run build_crossfit_aliases.py first")
    snapshot.mkdir(parents=True)
    files = list((root / "code/business_entity_resolution/src").rglob("*.py"))
    files.append(root / "code/business_entity_resolution/src/blocking/pack_postings.c")
    files += [root / "scripts" / name for name in ("build_scale_cache.py", "train_scale_model.py",
        "audit_scale_model.py", "run_scale_experiment.py", "build_crossfit_aliases.py", "freeze_scale_experiment.py")]
    for path in files:
        dest = snapshot / path.relative_to(root)
        dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, dest)
    manifest = {"files": {str(path.relative_to(snapshot)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in snapshot.rglob("*") if path.is_file()},
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "experiment": "300k train / 50k tune / reserved 10k audit; five-fold aliases; bounded postings; 65 features"}
    (snapshot / "snapshot.json").write_text(json.dumps(manifest, indent=2) + "\n")
    command = [sys.executable, str(snapshot / "scripts/run_scale_experiment.py"), "--plan", str(args.plan.resolve()),
               "--output", str(run), "--workers", "2", "--idle-workers", "8", "--threads", "8"]
    if args.launch:
        env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1", PYTHONUNBUFFERED="1")
        with (run / "run.log").open("a") as log:
            process = subprocess.Popen(command, cwd=root, env=env, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        state = {"pid": process.pid, "command": command, "snapshot_files": len(manifest["files"])}
        if shutil.which("caffeinate"):
            monitor = subprocess.Popen(["caffeinate", "-i", "-w", str(process.pid)], stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
            state["caffeinate_pid"] = monitor.pid
        (run / "process.json").write_text(json.dumps(state, indent=2) + "\n")
        print(json.dumps(state))
    else:
        print(json.dumps({"snapshot": str(snapshot), "command": command}))


if __name__ == "__main__": main()
