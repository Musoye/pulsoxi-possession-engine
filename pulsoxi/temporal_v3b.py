"""Controlled V3B decoder: fixed V1 D/X mask, reliability-aware live H/A Viterbi."""
from __future__ import annotations
import numpy as np
from .temporal import N_FRAMES, HALVES, TransitionCosts

def two_state_reliability_viterbi(probs,reliability,*,known=None,costs=TransitionCosts(),uncertainty_penalty=2.0):
    x=np.asarray(probs,float); rel=np.clip(np.asarray(reliability,float),0,1); n=len(x)
    if x.shape!=(n,3) or len(rel)!=n: raise ValueError("invalid shapes")
    if known is not None and np.asarray(known).shape!=(n,): raise ValueError("known mask shape mismatch")
    e=np.log(np.maximum(x[:,:2]/np.maximum(x[:,:2].sum(axis=1,keepdims=True),1e-12),1e-12))
    if known is not None: e[~np.asarray(known,dtype=bool)]=0.0
    if not n:return np.empty(0,dtype=np.uint8)
    back=np.zeros((n,2),dtype=np.uint8); score=e[0].copy()
    for t in range(1,n):
        switch=costs.opponent+uncertainty_penalty*(1-rel[t]); trans=np.array([[0.,-switch],[-switch,0.]])
        choices=score[:,None]+trans; back[t]=np.argmax(choices,axis=0).astype(np.uint8); score=e[t]+choices[back[t],np.arange(2)]
    out=np.empty(n,dtype=np.uint8); out[-1]=np.argmax(score)
    for t in range(n-1,0,-1):out[t-1]=back[t,out[t]]
    return out

def decode_full_match_v3b(probs,reliability,v1_labels,*,known=None,costs=TransitionCosts(),uncertainty_penalty=2.0):
    probs=np.asarray(probs); rel=np.asarray(reliability); mask=np.asarray(v1_labels)
    if probs.shape!=(N_FRAMES,3) or rel.shape!=(N_FRAMES,) or mask.shape!=(N_FRAMES,): raise ValueError("expected full-match inputs")
    out=mask.copy(); states=np.array(("H","A"))
    for first,last in HALVES:
        live=np.isin(out[first:last+1],["H","A"]); i=0
        while i<len(live):
            if not live[i]: i+=1; continue
            j=i
            while j<len(live) and live[j]: j+=1
            sl=slice(first+i,first+j); local_known=None if known is None else np.asarray(known)[sl]
            out[sl]=states[two_state_reliability_viterbi(probs[sl],rel[sl],known=local_known,costs=costs,uncertainty_penalty=uncertainty_penalty)]
            i=j
    return out
