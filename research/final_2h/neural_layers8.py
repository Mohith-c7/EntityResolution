"""Offline MPS continuation from a sealed checkpoint; final eight layers only.

Training reads only an explicitly sealed outer-training JSONL. Inference never
reads labels, selects thresholds, or writes a competition submission.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(2**20), b""):
            result.update(part)
    return result.hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def lr_multiplier(step, total_steps, warmup_steps):
    """Warm up the first updates, then decay across the complete epoch plan."""
    if not 0 < warmup_steps < total_steps:
        raise ValueError("Warmup must be positive and shorter than the training plan")
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    return max(0., (total_steps - step) / (total_steps - warmup_steps))


def epoch_order(row_count, seed, epoch):
    import numpy as np
    return np.random.default_rng(np.random.SeedSequence([seed, epoch])).permutation(row_count)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["fit", "score"])
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-minutes", type=float, default=18)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-origin", type=Path)
    args = parser.parse_args()
    if args.epochs < 1 or not math.isfinite(args.max_minutes) or args.max_minutes <= 0:
        raise ValueError("Require positive epochs and time cap")
    run_started = time.monotonic()
    import numpy as np
    import torch
    from transformers import BertForSequenceClassification, XLMRobertaTokenizer
    if not torch.backends.mps.is_available():
        raise RuntimeError("Requires MPS; CPU fallback is disabled")
    marker = json.loads(args.input_manifest.read_text())
    if marker.get("status") != "complete" or sha(args.input) != marker["input_sha256"]:
        raise ValueError("Input is not sealed")
    rows = [json.loads(line) for line in args.input.open()]
    keys = [(r["source1_entity_id"], r["candidate_entity_id"]) for r in rows]
    if len(keys) != len(set(keys)) or len(rows) != marker["rows"]:
        raise ValueError("Duplicate keys or incomplete input")
    if args.command == "fit":
        if marker.get("role") != "outer_train" or marker.get("owners_disjoint_development") is not True:
            raise ValueError("Training requires outer-training provenance")
        if any(marker.get(key) is not False for key in ("audit_labels_read", "development_labels_read", "test_records_read")):
            raise ValueError("Training provenance must exclude development/audit labels and test records")
        if args.base_origin is None:
            raise ValueError("Continuation requires the supplied base --base-origin provenance")
        origin_path = args.base_origin / "origin.json" if args.base_origin.is_dir() else args.base_origin
        origin = json.loads(origin_path.read_text())
        if origin["license"] != "mit":
            raise ValueError("This experiment requires the supplied MIT MiniLM checkpoint")
        if origin.get("repo") != "microsoft/Multilingual-MiniLM-L12-H384" or origin.get("revision") != marker.get("base_revision"):
            raise ValueError("Require the supplied multilingual MiniLM base model")
        for filename, descriptor in origin["files"].items():
            if sha(origin_path.parent / filename) != descriptor["sha256"]:
                raise ValueError("Base model or license changed")
        initializer_manifest_path = args.model.parent / "manifest.json"
        initializer = json.loads(initializer_manifest_path.read_text())
        if (initializer.get("status") != "complete" or initializer.get("training_only") is not True
                or initializer.get("epochs") != 3 or initializer.get("completed_epochs") != 3
                or initializer.get("training_input_sha256") != sha(args.input)
                or initializer.get("training_manifest_sha256") != sha(args.input_manifest)
                or initializer.get("base_origin_sha256") != sha(origin_path)
                or any(initializer.get(key) is not False for key in
                       ("audit_labels_read", "development_labels_read", "test_records_read"))):
            raise ValueError("Require complete sealed three-epoch initialization on identical outer-training input")
        expected_files = initializer["checkpoint_files"]
        if set(expected_files) != {p.name for p in args.model.iterdir() if p.is_file()}:
            raise ValueError("Initializer checkpoint file set changed")
        for filename, digest in expected_files.items():
            if sha(args.model / filename) != digest:
                raise ValueError("Initializer checkpoint or tokenizer changed")
        labels = np.array([r["label"] for r in rows], dtype=np.int64)
        if set(labels.tolist()) != {0, 1}:
            raise ValueError("Both supplied training classes are required")
        if int((labels == 1).sum()) != marker["positive_pairs"] or int((labels == 0).sum()) != marker["negative_pairs"]:
            raise ValueError("Training class counts differ from sealed manifest")
        if len({r["source1_entity_id"] for r in rows}) != marker["owners"]:
            raise ValueError("Training owner count differs from sealed manifest")
        if set(initializer["training_owners"]) != {r["source1_entity_id"] for r in rows}:
            raise ValueError("Continuation owner lineage differs")
    else:
        if marker.get("labels_read") is not False or any("label" in r for r in rows):
            raise ValueError("Inference input must not contain labels")
        origin = json.loads((args.model.parent / "manifest.json").read_text())
        for filename, digest in origin["checkpoint_files"].items():
            if sha(args.model / filename) != digest:
                raise ValueError("Fine-tuned checkpoint changed")
    args.output.mkdir(parents=True, exist_ok=False)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(8)
    tokenizer = XLMRobertaTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    model = BertForSequenceClassification.from_pretrained(args.model, num_labels=2,
        local_files_only=True, trust_remote_code=False).to("mps")
    parameters = sum(p.numel() for p in model.parameters())
    if parameters > 8_000_000_000:
        raise ValueError("Model exceeds parameter cap")
    started = time.monotonic()
    def encode(batch):
        tokens = tokenizer([r["text_left"] for r in batch], [r["text_right"] for r in batch],
            truncation="longest_first", max_length=128, padding="max_length", return_tensors="pt")
        return {name: values.to("mps") for name, values in tokens.items()}
    def emit(value):
        print(json.dumps(value), flush=True)
    def save_checkpoint(path):
        # Keep trainable parameters and Adam state on MPS between epochs. Save a
        # detached CPU state dict instead of moving the live model/optimizer.
        torch.mps.synchronize()
        state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
        model.save_pretrained(path, state_dict=state, safe_serialization=True)
        tokenizer.save_pretrained(path)
        return {p.name: sha(p) for p in sorted(path.iterdir()) if p.is_file()}
    if args.command == "score":
        model.half().eval()
        with (args.output / "scores.jsonl").open("x") as stream, torch.inference_mode():
            for start in range(0, len(rows), 32):
                batch = rows[start:start + 32]
                probabilities = model(**encode(batch)).logits.float().softmax(-1)[:, 1].cpu().numpy()
                if not np.isfinite(probabilities).all():
                    raise ValueError("Nonfinite neural probability")
                for row, probability in zip(batch, probabilities):
                    stream.write(json.dumps({"source1_entity_id": row["source1_entity_id"],
                        "candidate_entity_id": row["candidate_entity_id"],
                        "neural_probability": float(probability)}) + "\n")
        torch.mps.synchronize()
        write(args.output / "manifest.json", {"status": "complete", "rows": len(rows),
            "input_sha256": sha(args.input), "checkpoint_manifest_sha256": sha(args.model.parent / "manifest.json"),
            "scores_sha256": sha(args.output / "scores.jsonl"), "labels_read": False,
            "seconds": time.monotonic() - started, "code_sha256": sha(__file__)})
        return
    model.float().train()
    model.bert.requires_grad_(False)
    for layer in model.bert.encoder.layer[-8:]:
        layer.requires_grad_(True)
    model.bert.pooler.requires_grad_(True)
    model.classifier.requires_grad_(True)
    model.bert.embeddings.eval()
    for layer in model.bert.encoder.layer[:-8]:
        layer.eval()
    print(json.dumps({"trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}), flush=True)
    encoder_parameters = [p for name,p in model.named_parameters() if p.requires_grad and not name.startswith("classifier.")]
    classifier_parameters = list(model.classifier.parameters())
    optimizer = torch.optim.AdamW([{"params": encoder_parameters, "lr": 1e-5}, {"params": classifier_parameters, "lr": 1e-4}], weight_decay=.01)
    steps_per_epoch = math.ceil(len(rows) / 32)
    steps = steps_per_epoch * args.epochs
    warmup = min(steps - 1, max(1, math.ceil(.06 * steps)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
        lambda step: lr_multiplier(step, steps, warmup))
    processed = 0; loss_sum = 0.
    global_step = 0
    epoch_reports = []
    provenance = {"training_input_sha256": sha(args.input), "training_manifest_sha256": sha(args.input_manifest),
                  "base_origin_sha256": sha(origin_path), "code_sha256": sha(__file__),
                  "parent_script_sha256": sha(Path(__file__).resolve().parent / "neural_epochs.py"),
                  "initializer_manifest_sha256": sha(initializer_manifest_path),
                  "initializer_checkpoint_files": expected_files,
                  "initializer_epochs": initializer["epochs"], "lineage_epochs_planned": initializer["epochs"] + args.epochs,
                  "trainable_encoder_layers": 8, "trainable_pooler_and_classifier": True,
                  "seed": args.seed, "epochs_planned": args.epochs, "steps_per_epoch": steps_per_epoch,
                  "total_steps": steps, "warmup_steps": warmup, "batch_size": 32, "max_length": 128,
                  "device": "mps", "max_minutes": args.max_minutes,
                  "base_learning_rates": {"last_eight_layers_and_pooler": 1e-5, "classifier": 1e-4},
                  "audit_labels_read": False, "development_labels_read": False, "test_records_read": False}
    write(args.output / "training_plan.json", {"status": "verified_outer_training_plan", **provenance})
    emit({"stage": "training_started", "rows": len(rows), "parameters": parameters,
        "batch_size": 32, "epochs": args.epochs, "total_steps": steps, "warmup_steps": warmup,
        "max_total_tokens": 128, "device": "mps", "max_minutes": args.max_minutes})
    for epoch in range(1, args.epochs + 1):
        epoch_started = time.monotonic()
        order = epoch_order(len(rows), args.seed, initializer["epochs"] + epoch)
        order_sha256 = hashlib.sha256(order.astype("<i8").tobytes()).hexdigest()
        epoch_processed = 0; epoch_loss_sum = 0.
        emit({"stage": "epoch_started", "epoch": epoch, "order_sha256": order_sha256})
        for number, start in enumerate(range(0, len(rows), 32)):
            if time.monotonic() - run_started >= args.max_minutes * 60:
                checkpoint = args.output / "time_limit_checkpoint"
                checkpoint_files = save_checkpoint(checkpoint)
                write(args.output / "status.json", {"status": "time_limit_rejected", **provenance,
                    "pairs": processed, "global_step": global_step, "completed_epochs": epoch - 1,
                    "partial_epoch": epoch, "partial_epoch_rows": epoch_processed,
                    "partial_checkpoint_preserved": True, "checkpoint_files": checkpoint_files,
                    "epoch_reports": epoch_reports, "seconds": time.monotonic() - run_started})
                emit({"stage": "time_limit_rejected", "completed_epochs": epoch - 1,
                      "global_step": global_step, "seconds": time.monotonic() - run_started})
                return
            indexes = order[start:start + 32]
            batch = [rows[i] for i in indexes]
            tokens = encode(batch)
            tokens["labels"] = torch.tensor(labels[indexes], dtype=torch.long, device="mps")
            optimizer.zero_grad(set_to_none=True)
            loss = model(**tokens).loss
            if not torch.isfinite(loss).item():
                raise ValueError("Nonfinite training loss")
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step(); scheduler.step()
            loss_value = loss.detach().item() * len(batch)
            global_step += 1; processed += len(batch); loss_sum += loss_value
            epoch_processed += len(batch); epoch_loss_sum += loss_value
            if (number + 1) % 100 == 0:
                emit({"stage": "train", "epoch": epoch, "epoch_step": number + 1,
                    "global_step": global_step, "total_steps": steps, "pairs": processed,
                    "epoch_mean_loss": epoch_loss_sum / epoch_processed, "mean_loss": loss_sum / processed,
                    "learning_rates": scheduler.get_last_lr(),
                    "pairs_per_second": processed / (time.monotonic() - started),
                    "seconds": time.monotonic() - run_started})
        checkpoint = args.output / ("checkpoint" if epoch == args.epochs else f"epoch_{epoch:03d}_checkpoint")
        checkpoint_files = save_checkpoint(checkpoint)
        epoch_report = {"status": "epoch_complete", **provenance, "epoch": epoch,
                        "rows": epoch_processed, "global_step": global_step, "order_sha256": order_sha256,
                        "mean_loss": epoch_loss_sum / epoch_processed,
                        "seconds": time.monotonic() - epoch_started, "elapsed_seconds": time.monotonic() - run_started,
                        "checkpoint": checkpoint.name, "checkpoint_files": checkpoint_files,
                        "learning_rates": scheduler.get_last_lr()}
        write(args.output / f"epoch_{epoch:03d}.json", epoch_report)
        epoch_reports.append(epoch_report)
        emit({"stage": "epoch_complete", "epoch": epoch, "mean_loss": epoch_report["mean_loss"],
              "seconds": epoch_report["seconds"], "elapsed_seconds": epoch_report["elapsed_seconds"],
              "checkpoint": checkpoint.name})
    torch.mps.synchronize()
    write(args.output / "manifest.json", {"status": "complete", "training_only": True,
        "completed_at": datetime.now(timezone.utc).isoformat(), "parameters": parameters,
        "training_rows": len(rows), "training_owners": sorted({r["source1_entity_id"] for r in rows}),
        "training_input_sha256": sha(args.input), "training_manifest_sha256": sha(args.input_manifest),
        "base_origin_sha256": sha(origin_path), "checkpoint_files":
        {p.name: sha(p) for p in sorted(checkpoint.iterdir()) if p.is_file()}, "code_sha256": sha(__file__),
        **provenance, "epochs": args.epochs, "completed_epochs": len(epoch_reports),
        "epoch_reports": epoch_reports, "global_step": global_step, "processed_training_rows": processed,
        "mean_loss": loss_sum / processed, "seconds": time.monotonic() - run_started,
        "audit_labels_read": False, "development_labels_read": False, "test_records_read": False,
        "versions": {"torch": torch.__version__, "transformers": __import__("transformers").__version__}})
    emit({"stage": "training_complete", "seconds": time.monotonic() - run_started,
        "completed_epochs": len(epoch_reports), "mean_loss": loss_sum / processed,
        "accuracy_evaluated": False})


if __name__ == "__main__":
    main()
