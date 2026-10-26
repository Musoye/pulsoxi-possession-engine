"""Reliability-aware extensions to the joint temporal decoder."""
from __future__ import annotations
import numpy as np
from .temporal import N_FRAMES, HALVES, STATES, TransitionCosts

def reliability_aware_viterbi(probs, reliability, *, known=None, costs=TransitionCosts(), uncertainty_penalty=2.0):
    x=np.asarray(probs,dtype=float); rel=np.clip(np.asarray(reliability,dtype=float),0,1)
    if x.ndim!=2 or x.shape[1]!=3 or len(rel)!=len(x): raise ValueError("invalid probabilities/reliability shape")
    if np.any(~np.isfinite(x)) or np.any(x<0) or np.any(~np.isfinite(rel)): raise ValueError("finite nonnegative inputs required")
    n=len(x)
    if known is not None:
        known=np.asarray(known,dtype=bool)
        if known.shape!=(n,): raise ValueError("known mask shape mismatch")
    if not n: return np.empty(0,dtype=np.uint8)
    x=x/np.maximum(x.sum(axis=1,keepdims=True),1e-12)
    emission=np.log(np.maximum(x,1e-12))
    if known is not None: emission[~known]=0.0
    back=np.zeros((n,3),dtype=np.uint8); scores=emission[0].copy()
    for t in range(1,n):
        opponent=costs.opponent+uncertainty_penalty*(1-rel[t])
        trans=np.array([[0.,-opponent,-costs.enter_dead],[-opponent,0.,-costs.enter_dead],[-costs.leave_dead,-costs.leave_dead,0.]])
        choices=scores[:,None]+trans; back[t]=np.argmax(choices,axis=0).astype(np.uint8)
        scores=emission[t]+choices[back[t],np.arange(3)]
    out=np.empty(n,dtype=np.uint8); out[-1]=np.argmax(scores)
    for t in range(n-1,0,-1): out[t-1]=back[t,out[t]]
    return out

def decode_full_match_v3(probs,reliability,*,known=None,costs=TransitionCosts(),uncertainty_penalty=2.0):
    probs=np.asarray(probs); rel=np.asarray(reliability)
    if probs.shape!=(N_FRAMES,3) or rel.shape!=(N_FRAMES,): raise ValueError("expected full-match probabilities and reliability")
    if known is not None and np.asarray(known).shape!=(N_FRAMES,): raise ValueError("known mask shape mismatch")
    labels=np.full(N_FRAMES,"X",dtype="<U1"); states=np.array(STATES)
    for first,last in HALVES:
        loc=slice(first,last+1); mask=None if known is None else np.asarray(known)[loc]
        labels[loc]=states[reliability_aware_viterbi(probs[loc],rel[loc],known=mask,costs=costs,uncertainty_penalty=uncertainty_penalty)]
    return labels
