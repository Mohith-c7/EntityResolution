"""Fixed frozen-encoder pilot; keyed local inputs, no audit or test labels."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='0'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for x in iter(lambda:stream.read(2**20),b''):h.update(x)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['fit','score'])
    for name in ['input','manifest','base','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--head',type=Path)
    a=p.parse_args()
    import numpy as np
    import torch
    from transformers import BertModel,XLMRobertaTokenizer
    from transformers.utils import logging
    logging.set_verbosity_error()
    if not torch.backends.mps.is_available():raise RuntimeError('MPS required')
    marker=json.loads(a.manifest.read_text())
    if marker['status']!='complete' or sha(a.input)!=marker['input_sha256']:raise ValueError('Input seal differs')
    rows=[json.loads(x) for x in a.input.open()]
    keys=[(r['source1_entity_id'],r['candidate_entity_id']) for r in rows]
    if len(keys)!=len(set(keys)) or len(rows)!=marker['rows']:raise ValueError('Pair keys/coverage invalid')
    if a.command=='fit':
        if marker.get('role')!='outer_train' or marker.get('owners_disjoint_development') is not True:raise ValueError('Fit requires outer training')
    elif marker.get('labels_read') is not False or any('label' in r for r in rows):raise ValueError('Inference forbids labels')
    origin=json.loads((a.base/'origin.json').read_text())
    if origin['license'] not in ['mit','apache-2.0']:raise ValueError('License forbidden')
    for file,details in origin['files'].items():
        if sha(a.base/file)!=details['sha256']:raise ValueError('Pretrained file changed')
    torch.manual_seed(42);torch.set_num_threads(8)
    a.output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    tokenizer=XLMRobertaTokenizer.from_pretrained(a.base,local_files_only=True)
    encoder=BertModel.from_pretrained(a.base,local_files_only=True).half().to('mps').eval()
    encoder.requires_grad_(False)
    features=np.lib.format.open_memmap(a.output/'features.npy',mode='w+',dtype=np.float32,shape=(len(rows),1536))
    with torch.inference_mode():
        for start in range(0,len(rows),32):
            batch=rows[start:start+32]
            tokens=tokenizer([r['text_left'] for r in batch],[r['text_right'] for r in batch],
                max_length=128,truncation='longest_first',padding='max_length',return_tensors='pt')
            input_ids=tokens['input_ids'].numpy(); attention=tokens['attention_mask'].numpy().astype(bool)
            left=np.zeros_like(attention);right=np.zeros_like(attention)
            for i in range(len(batch)):
                seps=np.flatnonzero((input_ids[i]==tokenizer.sep_token_id)&attention[i])
                if len(seps)<3:raise ValueError('Pair token delimiters changed')
                left[i,1:seps[0]]=True;right[i,seps[1]+1:seps[-1]]=True
            output=encoder(**{k:v.to('mps') for k,v in tokens.items()}).last_hidden_state.float()
            lm=torch.tensor(left,device='mps').unsqueeze(-1);rm=torch.tensor(right,device='mps').unsqueeze(-1)
            lv=(output*lm).sum(1)/lm.sum(1).clamp(min=1);rv=(output*rm).sum(1)/rm.sum(1).clamp(min=1)
            f=torch.cat([lv,rv,(lv-rv).abs(),lv*rv],dim=1).cpu().numpy()
            if not np.isfinite(f).all():raise ValueError('Nonfinite embedding')
            features[start:start+len(batch)]=f
            if (start//32+1)%500==0:print(json.dumps({'stage':'frozen_encoder','pairs':start+len(batch),'seconds':time.monotonic()-started}),flush=True)
            if time.monotonic()-started>15*60:
                (a.output/'status.json').write_text(json.dumps({'status':'feature_time_limit_rejected','pairs':start+len(batch)}))
                return
    features.flush();del encoder;torch.mps.empty_cache()
    (a.output/'keys.json').write_text(json.dumps(keys)+'\n')
    head=torch.nn.Sequential(torch.nn.Linear(1536,256),torch.nn.ReLU(),torch.nn.Dropout(.1),torch.nn.Linear(256,64),torch.nn.ReLU(),torch.nn.Linear(64,2))
    if a.command=='fit':
        labels=torch.tensor([r['label'] for r in rows],dtype=torch.long)
        if set(labels.tolist())!={0,1}:raise ValueError('Both classes required')
        mean=features.mean(axis=0);scale=features.std(axis=0).clip(min=.01)
        data=torch.from_numpy((features-mean)/scale)
        optimizer=torch.optim.AdamW(head.parameters(),lr=1e-3,weight_decay=.01)
        generator=np.random.default_rng(42)
        for epoch in range(5):
            order=generator.permutation(len(rows));loss_total=0.
            head.train()
            for start in range(0,len(rows),512):
                idx=order[start:start+512];optimizer.zero_grad(set_to_none=True)
                loss=torch.nn.functional.cross_entropy(head(data[idx]),labels[idx]);loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(),1.);optimizer.step();loss_total+=loss.item()*len(idx)
            print(json.dumps({'stage':'head_epoch','epoch':epoch+1,'loss':loss_total/len(rows),'seconds':time.monotonic()-started}),flush=True)
        torch.save(head.state_dict(),a.output/'head.pt');np.savez(a.output/'scaling.npz',mean=mean,scale=scale)
        report={'status':'complete','training_only':True,'encoder_frozen':True,'head_epochs':5,'training_owners':sorted(set(r['source1_entity_id'] for r in rows)),
            'head_sha256':sha(a.output/'head.pt'),'scaling_sha256':sha(a.output/'scaling.npz'),'training_input_sha256':sha(a.input)}
    else:
        hm=json.loads((a.head/'manifest.json').read_text())
        if hm['head_sha256']!=sha(a.head/'head.pt') or hm['scaling_sha256']!=sha(a.head/'scaling.npz') or hm['base_origin_sha256']!=sha(a.base/'origin.json'):raise ValueError('Head/encoder lineage changed')
        head.load_state_dict(torch.load(a.head/'head.pt',map_location='cpu',weights_only=True));head.eval()
        scaling=np.load(a.head/'scaling.npz');probabilities=[]
        with torch.inference_mode():
            for start in range(0,len(rows),512):
                f=torch.from_numpy((features[start:start+512]-scaling['mean'])/scaling['scale'])
                probabilities.extend(head(f).softmax(-1)[:,1].tolist())
        with (a.output/'scores.jsonl').open('x') as stream:
            for key,probability in zip(keys,probabilities):stream.write(json.dumps({'source1_entity_id':key[0],'candidate_entity_id':key[1],'neural_probability':probability})+'\n')
        report={'status':'complete','labels_read':False,'scores_sha256':sha(a.output/'scores.jsonl'),'head_manifest_sha256':sha(a.head/'manifest.json')}
    report.update(rows=len(rows),seconds=time.monotonic()-started,input_sha256=sha(a.input),input_manifest_sha256=sha(a.manifest),
        base_origin_sha256=sha(a.base/'origin.json'),code_sha256=sha(__file__),features_sha256=sha(a.output/'features.npy'))
    (a.output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='training_owners'}),flush=True)


if __name__=='__main__':main()
