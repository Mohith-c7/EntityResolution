"""Training-only early-stop metric matching the fixed entity decision rule."""
import numpy as np
import pandas as pd


class EarlyMacroMetric:
    def __init__(self, full, routed, truth, t_first=.70, t_rest=.83):
        self.keys=['source1_entity_id','candidate_entity_id']
        if set(full.source1_entity_id)!=set(truth):
            raise ValueError('Early cohort must include every declared reference')
        self.positions=pd.MultiIndex.from_frame(full[self.keys]).get_indexer(pd.MultiIndex.from_frame(routed[self.keys]))
        if (self.positions<0).any() or len(set(self.positions))!=len(self.positions):
            raise ValueError('Routed keys must be unique members of the full cohort')
        self.base=full.probability.to_numpy().copy()
        self.codes,self.owners=pd.factorize(full.source1_entity_id,sort=False)
        self.labels=np.array([c in truth[e] for e,c in full[self.keys].itertuples(index=False,name=None)])
        self.true_counts=np.array([len(truth[e]) for e in self.owners])
        self.t_first=t_first;self.t_rest=t_rest

    def __call__(self, probabilities, dataset=None):
        probabilities=np.asarray(probabilities)
        if len(probabilities)!=len(self.positions) or not np.isfinite(probabilities).all():
            raise ValueError('Invalid early probabilities')
        all_probabilities=self.base.copy();all_probabilities[self.positions]=probabilities
        maxima=np.full(len(self.owners),-np.inf)
        np.maximum.at(maxima,self.codes,all_probabilities)
        # At most one rank-one rescue per reference, retaining stable tie order.
        peak=np.flatnonzero(all_probabilities==maxima[self.codes])
        _,first_positions=np.unique(self.codes[peak],return_index=True)
        first=np.zeros(len(self.base),dtype=bool);first[peak[first_positions]]=True
        chosen=(all_probabilities>=self.t_rest)|(first&(all_probabilities>=self.t_first))
        predicted=np.bincount(self.codes,weights=chosen,minlength=len(self.owners))
        correct=np.bincount(self.codes,weights=chosen&self.labels,minlength=len(self.owners))
        denominator=.25*self.true_counts+predicted
        scores=np.divide(1.25*correct,denominator,out=np.zeros(len(self.owners)),where=denominator>0)
        scores[(self.true_counts==0)&(predicted==0)]=1.
        return 'early_macro_f05',float(scores.mean()),True
