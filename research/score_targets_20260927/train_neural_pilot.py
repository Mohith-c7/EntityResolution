"""Offline, single-GPU cross-encoder pilot; never generates a submission."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for part in iter(lambda: handle.read(2**20), b''):
            h.update(part)
    return h.hexdigest()


def metrics(probabilities, frame, truth, threshold):
    ids = sorted(truth)
    index = {e: i for i, e in enumerate(ids)}
    group = frame.source1_entity_id.map(index).to_numpy(dtype=np.int64)
    expected = np.array([len(truth[e]) for e in ids])
    accepted = probabilities >= threshold
    labels = frame.label.to_numpy(dtype=bool)
    predicted = np.bincount(group[accepted], minlength=len(ids))
    tp = np.bincount(group[accepted & labels], minlength=len(ids))
    denominator = predicted + .25 * expected
    score = np.divide(1.25 * tp, denominator, out=np.zeros(len(ids)), where=denominator > 0)
    score[(expected == 0) & (predicted == 0)] = 1
    return {'macro_f05': float(score.mean()), 'precision': float(tp.sum()/max(1,predicted.sum())),
            'recall': float(tp.sum()/max(1,expected.sum())), 'threshold': float(threshold),
            'singleton_false_positives': int(((expected == 0) & (predicted > 0)).sum()),
            'empty_non_singletons': int(((expected > 0) & (predicted == 0)).sum())}


class EncodedPairs:
    def __init__(self, frame, tokenizer, max_length):
        self.encodings = {}
        for start in range(0, len(frame), 4096):
            batch = frame.iloc[start:start+4096]
            encoded = tokenizer(batch.text_left.tolist(), batch.text_right.tolist(),
                                truncation='longest_first', max_length=max_length, padding=False)
            for key, values in encoded.items():
                self.encodings.setdefault(key, []).extend(values)
        self.labels = frame.label.to_numpy(dtype=np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {**{key: value[index] for key, value in self.encodings.items()},
                'labels': int(self.labels[index])}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--pretrained', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--accumulation', type=int, default=2)
    p.add_argument('--max-length', type=int, default=256)
    p.add_argument('--epochs', type=int, default=2)
    p.add_argument('--max-hours', type=float, default=4.5)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    if min(args.batch_size, args.accumulation, args.max_length, args.epochs) < 1:
        p.error('Training sizes must be positive')
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('This pilot requires a CUDA GPU supporting bfloat16; no CPU fallback')
    args.output.mkdir(parents=True, exist_ok=False)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = True
    manifest = json.loads((args.data/'manifest.json').read_text())
    origin = json.loads((args.pretrained/'origin.json').read_text())
    if origin['license'] not in ('mit', 'apache-2.0'):
        raise ValueError('Model license not permitted')
    frames = {}
    for split in ('train', 'validation'):
        path = args.data/f'{split}.parquet'
        if digest(path) != manifest['summaries'][split]['sha256']:
            raise ValueError(f'Corrupt {split} data')
        frames[split] = pd.read_parquet(path)
    if set(frames['train'].source1_entity_id) & set(frames['validation'].source1_entity_id):
        raise ValueError('Training/validation entity overlap')
    truth = json.loads((args.data/'validation_truth.json').read_text())
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained, local_files_only=True, trust_remote_code=False)
    model = AutoModelForSequenceClassification.from_pretrained(args.pretrained, num_labels=2,
                        local_files_only=True, trust_remote_code=False, use_safetensors=True).cuda()
    parameters = sum(x.numel() for x in model.parameters())
    if parameters > 8_000_000_000:
        raise ValueError('Model exceeds competition parameter limit')
    collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    started = time.monotonic()
    loaders = {}
    for split, frame in frames.items():
        encoded = EncodedPairs(frame, tokenizer, args.max_length)
        loaders[split] = DataLoader(encoded, batch_size=args.batch_size, shuffle=split == 'train',
                     collate_fn=collator, num_workers=0, pin_memory=True,
                     generator=torch.Generator().manual_seed(args.seed))
    config = {**vars(args), 'parameters': parameters, 'gpu': torch.cuda.get_device_name(),
              'pretrained_origin': origin, 'data_manifest_sha256': digest(args.data/'manifest.json'),
              'source_sha256': digest(Path(__file__)), 'status': 'pilot_development_only',
              'comparison_scope': 'pre-bridge retrieval; not comparable to 0.979820 final pipeline'}
    (args.output/'config.json').write_text(json.dumps(config, indent=2, default=str)+'\n')
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=.01)
    total_steps = math.ceil(len(loaders['train'])/args.accumulation)*args.epochs
    warmup = max(1, int(total_steps*.06))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step:
        min((step+1)/warmup, max(0., (total_steps-step)/max(1,total_steps-warmup))))
    best = -1.; step = 0
    print(json.dumps({'stage':'training_started','parameters':parameters,'gpu':config['gpu']}),flush=True)
    for epoch in range(args.epochs):
        model.train(); optimizer.zero_grad(set_to_none=True)
        processed = 0; loss_sum = 0.; epoch_start = time.monotonic()
        for number, batch in enumerate(loaders['train']):
            batch = {k:v.cuda(non_blocking=True) for k,v in batch.items()}
            # Correct the final, shorter accumulation group.
            group_start = (number//args.accumulation)*args.accumulation
            denominator = min(args.accumulation, len(loaders['train'])-group_start)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss = model(**batch).loss
            (loss/denominator).backward()
            processed += len(batch['labels']); loss_sum += float(loss.detach())*len(batch['labels'])
            update = (number+1)%args.accumulation == 0 or number+1 == len(loaders['train'])
            if update:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True); step += 1
                if step%50 == 0:
                    print(json.dumps({'stage':'train','epoch':epoch+1,'step':step,'pairs':processed,
                       'loss':loss_sum/processed,'pairs_per_second':processed/(time.monotonic()-epoch_start)}),flush=True)
                if time.monotonic()-started > args.max_hours*3600:
                    model.save_pretrained(args.output/'time_limit_checkpoint', safe_serialization=True)
                    tokenizer.save_pretrained(args.output/'time_limit_checkpoint')
                    (args.output/'status.json').write_text(json.dumps({'status':'time_limit_no_final_evaluation','step':step}))
                    return
        model.eval(); probabilities = []; inference_start = time.monotonic()
        with torch.inference_mode():
            for batch in loaders['validation']:
                batch.pop('labels')
                batch = {k:v.cuda(non_blocking=True) for k,v in batch.items()}
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    logits = model(**batch).logits
                probabilities.append(logits.float().softmax(dim=1)[:,1].cpu().numpy())
        probabilities = np.concatenate(probabilities)
        inference_seconds = time.monotonic()-inference_start
        trials = [metrics(probabilities,frames['validation'],truth,t) for t in np.arange(.30,.991,.01)]
        chosen = max(trials, key=lambda r:r['macro_f05'])
        report = {'status':'exploratory_development_only','epoch':epoch+1, 'best_threshold':chosen,
                  'threshold_trials':trials,'inference_seconds':inference_seconds,
                  'pairs_per_second':len(probabilities)/inference_seconds,
                  'peak_gpu_bytes':torch.cuda.max_memory_allocated()}
        (args.output/f'epoch_{epoch+1}.json').write_text(json.dumps(report,indent=2)+'\n')
        scored = frames['validation'][['source1_entity_id','candidate_entity_id','label','country']].copy()
        scored['neural_probability'] = probabilities
        scored.to_parquet(args.output/f'epoch_{epoch+1}_scores.parquet',index=False)
        if chosen['macro_f05'] > best:
            best = chosen['macro_f05']; model.save_pretrained(args.output/'best', safe_serialization=True)
            tokenizer.save_pretrained(args.output/'best')
            (args.output/'best_report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({'stage':'epoch_complete', 'epoch':epoch+1, **chosen}),flush=True)
    (args.output/'status.json').write_text(json.dumps({'status':'pilot_complete','best_development_macro_f05':best}))


if __name__ == '__main__':
    main()
