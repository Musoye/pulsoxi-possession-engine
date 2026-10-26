"""Local-only aggregate audit for Phase 3B E1/E2 evidence comparisons."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent.parent; sys.path.insert(0,str(ROOT))
from games import GAME
from pulsoxi.atd_adapter import load_tactical_features
from pulsoxi.inference import build_reliability
from pulsoxi.temporal import HALVES

def load_labels(path):
    rows=Path(path).read_text(encoding='utf-8').splitlines()[1:]
    return np.array([r.rsplit(',',1)[1].strip() for r in rows],dtype='<U1')

def agg(labels,features):
    vis=np.isfinite(features[:,0]) & (features[:,0]>.5)
    speed=np.asarray(features[:,13],float); motion=np.asarray(features[:,16],float)
    rel=build_reliability(features); transitions=np.flatnonzero(labels[1:]!=labels[:-1])+1
    live=labels!='D'; short={}
    starts=[]; i=0
    while i<len(labels):
        if labels[i] in ('H','A'):
            j=i+1
            while j<len(labels) and labels[j]==labels[i]: j+=1
            if i==0 or labels[i-1] not in ('H','A'): starts.append((i,j,labels[i]))
            i=j
        else:i+=1
    durations=[(b-a)/25 for a,b,_ in starts]
    reappear= np.flatnonzero(vis[1:] & ~vis[:-1])+1
    after=[int(t) for t in transitions if t in set(reappear) or t-1 in set(reappear)]
    gaps=[]; g=0
    for v in vis:
        if v:gaps.append(g);g=0
        else:g+=1
    def trans_stats():
        idx=[t for t in transitions if labels[t] in ('H','A') and labels[t-1] in ('H','A')]
        return {'count':len(idx),'reliable_motion':int(sum(motion[t]>.5 for t in idx)),'unreliable_motion':int(sum(motion[t]<=.5 for t in idx)),'mean_reliability':float(np.mean(rel[idx])) if idx else 0.0,'p10_reliability':float(np.percentile(rel[idx],10)) if idx else 0.0}
    short_idx=[k for k,d in enumerate(durations) if d<2]
    def conf(th):
        vals=[]
        for k in short_idx:
            a,b,_=starts[k]; vals.append(float(np.mean(rel[a:b])))
        return {'count':int(sum(v>=th for v in vals)),'threshold':th}
    return {'reappearances':len(reappear),'switches_immediately_after_reappearance':len(after),'reappearance_gap_frames':{'p50':float(np.percentile(gaps,50)) if gaps else 0,'p90':float(np.percentile(gaps,90)) if gaps else 0},'high_speed_frames_ge18':int(np.sum(speed>=18)),'high_speed_valid_motion':int(np.sum((speed>=18)&(motion>.5))),'transitions_at_high_speed':int(sum(speed[t]>=18 for t in transitions)),'transitions_at_discontinuity_ge25':int(sum(speed[t]>=25 for t in transitions)),'short_possessions_lt_0_5s':int(sum(d<.5 for d in durations)),'short_possessions_lt_1s':int(sum(d<1 for d in durations)),'short_possessions_lt_2s':int(sum(d<2 for d in durations)),'short_lt2_reliability_ge_0_5':conf(.5),'transition_reliability':trans_stats(),'mean_live_reliability':float(np.mean(rel[live])),'half_boundary_transitions':int(sum(t in {b[0],b[1]+1} for t in transitions for b in HALVES))}

def main():
 p=argparse.ArgumentParser();p.add_argument('--data-dir',default=None);p.add_argument('--out',required=True);a=p.parse_args()
 features,_,_=load_tactical_features(GAME,a.data_dir); result={}
 for name in ('v1','v3b','v1e2','v3be2'):
  result[name]=agg(load_labels(ROOT/'outputs/local_inference'/f'submission_{name}.csv'),features)
 result['fixed_mask_check']={'D_counts_equal':len(set(int(np.sum(load_labels(ROOT/'outputs/local_inference'/f'submission_{n}.csv')=='D')) for n in ('v1','v3b','v1e2','v3be2')))==1,'X_counts_equal':len(set(int(np.sum(load_labels(ROOT/'outputs/local_inference'/f'submission_{n}.csv')=='X')) for n in ('v1','v3b','v1e2','v3be2')))==1}
 Path(a.out).parent.mkdir(parents=True,exist_ok=True);Path(a.out).write_text(json.dumps(result,indent=2)+"\n",encoding='utf-8'); print('Wrote aggregate audit')
if __name__=='__main__':main()