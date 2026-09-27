import importlib.util
from pathlib import Path
import pytest
P=Path(__file__).resolve().parents[1]/'research/final_2h/neural_layers12.py'
s=importlib.util.spec_from_file_location('layers12',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class Param:
    requires_grad=True
class Block:
    def __init__(self):self.p=Param();self.training=True
    def parameters(self):return [self.p]
    def requires_grad_(self,value):self.p.requires_grad=value
    def eval(self):self.training=False
class Bert:
    def __init__(self,n):
        self.embeddings=Block();self.pooler=Block();self.encoder=type('Encoder',(),{'layer':[Block() for _ in range(n)]})()
    def requires_grad_(self,value):
        for b in [self.embeddings,self.pooler,*self.encoder.layer]:b.requires_grad_(value)
class Model:
    def __init__(self,n):self.bert=Bert(n);self.classifier=Block()
def test_all_twelve_encoder_blocks_train_but_embeddings_stay_frozen():
    model=Model(12);m.configure_trainable(model)
    assert not model.bert.embeddings.p.requires_grad;assert not model.bert.embeddings.training
    assert all(b.p.requires_grad for b in model.bert.encoder.layer)
    assert model.bert.pooler.p.requires_grad and model.classifier.p.requires_grad
    with pytest.raises(ValueError,match='exactly twelve'):m.configure_trainable(Model(8))
