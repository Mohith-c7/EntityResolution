"""Experimental expected macro-F0.5 decisions under independent calibrated links."""
from functools import lru_cache
import numpy as np


@lru_cache(maxsize=8)
def quadrature(candidates):
    # Without missing links the integrand has degree at most 5*n-1.
    nodes,weights=np.polynomial.legendre.leggauss(max(16,(5*candidates+1)//2))
    return (nodes+1)/2,weights/2


def choose_expected_f05(probabilities, missing_rate=0.):
    """Return a boolean prediction matrix and expected utilities for k=0..n.

    Missing links are an optional independent Poisson count. This is a decision
    experiment, not a claim that real candidate labels are independent.
    """
    p=np.asarray(probabilities,dtype=np.float64)
    if p.ndim!=2 or not np.isfinite(p).all() or np.any((p<0)|(p>1)) or not np.isfinite(missing_rate) or missing_rate<0:
        raise ValueError("Expected a finite probability matrix and nonnegative missing rate")
    rows,n=p.shape
    if not n:return np.zeros_like(p,dtype=bool),np.ones((rows,1))*np.exp(-missing_rate)
    order=np.argsort(-p,axis=1,kind="stable");ranked=np.take_along_axis(p,order,axis=1)
    nodes,weights=quadrature(n)
    q=1-ranked[:,:,None]+ranked[:,:,None]*nodes[None,None,:]
    generating=np.prod(q,axis=1)*np.exp(missing_rate*(nodes-1))[None,:]
    prefix=np.cumsum(ranked[:,:,None]/q,axis=1)
    powers=nodes[None,:]**(4*np.arange(1,n+1)[:,None])
    utilities=5*np.sum(prefix*generating[:,None,:]*powers[None,:,:]*weights[None,None,:],axis=2)
    empty=np.prod(1-ranked,axis=1)*np.exp(-missing_rate)
    utilities=np.column_stack([empty,utilities]);chosen=np.argmax(utilities,axis=1)
    selected=np.zeros((rows,n),dtype=bool)
    np.put_along_axis(selected,order,np.arange(n)[None,:]<chosen[:,None],axis=1)
    return selected,utilities
