"""Independent H/A interval metrics with censored-reference handling."""
from __future__ import annotations
from itertools import permutations

def spells(labels, *, periods=None, unknown="X"):
    out=[]; start=None; current=None; n=len(labels)
    for i in range(n+1):
        value = labels[i] if i<n else unknown
        same = value==current and (periods is None or i==0 or i==n or periods[i]==periods[i-1])
        if not same:
            if current not in (unknown,None) and start is not None: out.append((current,start,i))
            start=i if value not in (unknown,None) else None; current=value
    return out

def interval_iou(a,b):
    if a[0]!=b[0]: return 0.0
    inter=max(0,min(a[2],b[2])-max(a[1],b[1])); union=max(a[2],b[2])-min(a[1],b[1])
    return inter/union if union else 0.0

def best_matches(reference,predicted,threshold):
    """Maximum-cardinality one-to-one matching using Hopcroft-Karp."""
    edges={u:[v for v,b in enumerate(predicted) if interval_iou(a,b)>=threshold] for u,a in enumerate(reference)}
    pair_u={}; pair_v={}; dist={}
    from collections import deque
    def bfs():
        q=deque()
        for u in range(len(reference)):
            if u not in pair_u: dist[u]=0; q.append(u)
            else: dist[u]=None
        found=False
        while q:
            u=q.popleft()
            for v in edges[u]:
                w=pair_v.get(v)
                if w is None: found=True
                elif dist.get(w) is None: dist[w]=dist[u]+1; q.append(w)
        return found
    def dfs(u):
        for v in edges[u]:
            w=pair_v.get(v)
            if w is None or (dist.get(w)==dist[u]+1 and dfs(w)):
                pair_u[u]=v; pair_v[v]=u; return True
        dist[u]=None; return False
    while bfs():
        for u in range(len(reference)):
            if u not in pair_u: dfs(u)
    return sorted(pair_u.items())
def overlap_scores(reference,predicted,thresholds=(.3,.4,.5,.6,.7)):
    result={}
    for t in thresholds:
        matches=best_matches(reference,predicted,t); tp=len(matches); p=tp/len(predicted) if predicted else 0.; r=tp/len(reference) if reference else 0.; f=2*p*r/(p+r) if p+r else 0.; result[t]={"tp":tp,"precision":p,"recall":r,"f1":f}
    return result

def boundary_scores(reference,predicted,tolerance_frames=5):
    rb=[a[2] for a,b in zip(reference,reference[1:]) if a[0]!=b[0]]
    pb=[b[1] for a,b in zip(predicted,predicted[1:]) if a[0]!=b[0]]
    used=set(); errors=[]
    for x in rb:
        choices=[(abs(x-y),j) for j,y in enumerate(pb) if j not in used and abs(x-y)<=tolerance_frames]
        if choices:
            e,j=min(choices); used.add(j); errors.append(e)
    return {"reference":len(rb),"predicted":len(pb),"matched":len(errors),"false_positive":len(pb)-len(used),"missed":len(rb)-len(errors),"precision":len(errors)/len(pb) if pb else 0.,"recall":len(errors)/len(rb) if rb else 0.,"median_error_seconds":(sorted(errors)[len(errors)//2]/10.0 if errors else None)}



