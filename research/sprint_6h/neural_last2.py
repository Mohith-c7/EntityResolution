"""Offline MPS pilot: one fixed training epoch and keyed inference.

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
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["fit", "score"])
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-minutes", type=float, default=40)
    args = parser.parse_args()
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
        origin = json.loads((args.model / "origin.json").read_text())
        if origin["license"] not in ("mit", "apache-2.0"):
            raise ValueError("License is not permitted")
        for filename, descriptor in origin["files"].items():
            if sha(args.model / filename) != descriptor["sha256"]:
                raise ValueError("Base model or license changed")
        labels = np.array([r["label"] for r in rows], dtype=np.int64)
        if set(labels.tolist()) != {0, 1}:
            raise ValueError("Both supplied training classes are required")
    else:
        if marker.get("labels_read") is not False or any("label" in r for r in rows):
            raise ValueError("Inference input must not contain labels")
        origin = json.loads((args.model.parent / "manifest.json").read_text())
        for filename, digest in origin["checkpoint_files"].items():
            if sha(args.model / filename) != digest:
                raise ValueError("Fine-tuned checkpoint changed")
    args.output.mkdir(parents=True, exist_ok=False)
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
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
    for layer in model.bert.encoder.layer[-2:]:
        layer.requires_grad_(True)
    model.bert.pooler.requires_grad_(True)
    model.classifier.requires_grad_(True)
    model.bert.embeddings.eval()
    for layer in model.bert.encoder.layer[:-2]:
        layer.eval()
    print(json.dumps({"trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}), flush=True)
    order = np.random.default_rng(42).permutation(len(rows))
    encoder_parameters = [p for name,p in model.named_parameters() if p.requires_grad and not name.startswith("classifier.")]
    classifier_parameters = list(model.classifier.parameters())
    optimizer = torch.optim.AdamW([{"params": encoder_parameters, "lr": 2e-5}, {"params": classifier_parameters, "lr": .001}], weight_decay=.01)
    steps = math.ceil(len(rows) / 32)
    warmup = max(1, math.ceil(.06 * steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
        lambda step: min((step + 1) / warmup, max(0., (steps - step) / max(1, steps - warmup))))
    processed = 0; loss_sum = 0.
    emit({"stage": "training_started", "rows": len(rows), "parameters": parameters,
        "batch_size": 32, "epochs": 1, "max_total_tokens": 128, "device": "mps"})
    for number, start in enumerate(range(0, len(rows), 32)):
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
        processed += len(batch); loss_sum += loss.detach().item() * len(batch)
        if (number + 1) % 100 == 0:
            emit({"stage": "train", "step": number + 1, "steps": steps, "pairs": processed,
                "mean_loss": loss_sum / processed, "pairs_per_second": processed / (time.monotonic() - started)})
        if time.monotonic() - started > args.max_minutes * 60:
            model.to("cpu").save_pretrained(args.output / "time_limit_checkpoint", safe_serialization=True)
            tokenizer.save_pretrained(args.output / "time_limit_checkpoint")
            write(args.output / "status.json", {"status": "time_limit_rejected", "pairs": processed, "partial_checkpoint_preserved": True})
            return
    torch.mps.synchronize()
    checkpoint = args.output / "checkpoint"
    model.to("cpu").save_pretrained(checkpoint, safe_serialization=True)
    tokenizer.save_pretrained(checkpoint)
    write(args.output / "manifest.json", {"status": "complete", "training_only": True,
        "completed_at": datetime.now(timezone.utc).isoformat(), "parameters": parameters,
        "training_rows": len(rows), "training_owners": sorted({r["source1_entity_id"] for r in rows}),
        "training_input_sha256": sha(args.input), "training_manifest_sha256": sha(args.input_manifest),
        "base_origin_sha256": sha(args.model / "origin.json"), "checkpoint_files":
        {p.name: sha(p) for p in sorted(checkpoint.iterdir()) if p.is_file()}, "code_sha256": sha(__file__),
        "epochs": 1, "batch_size": 32, "seed": 42, "max_length": 128,
        "mean_loss": loss_sum / len(rows), "seconds": time.monotonic() - started,
        "audit_labels_read": False, "development_labels_read": False, "test_records_read": False,
        "versions": {"torch": torch.__version__, "transformers": __import__("transformers").__version__}})
    emit({"stage": "training_complete", "seconds": time.monotonic() - started,
        "accuracy_evaluated": False})


if __name__ == "__main__":
    main()
