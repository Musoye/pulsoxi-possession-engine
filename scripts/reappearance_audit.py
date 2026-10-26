import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent.parent;sys.path.insert(0,str(ROOT))
from games import GAME
from pulsoxi.atd_adapter import load_tactical_features
from pulsoxi.diagnostics import reappearance_events
from pulsoxi.inference import build_reliability
from scripts.diagnose_submission import load_labels

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--data-dir',default=None);a=ap.parse_args()
 features,names,meta=load_tactical_features(GAME,a.data_dir); visible=np.isfinite(features[:,0])&(features[:,0]>.5); frames=meta['video_frames']; events=reappearance_events(visible,frames); rel=build_reliability(features); speed=features[:,13]; motion=features[:,16]
 result={'true_reappearances':len(events),'gap_percentiles_frames':{str(q):float(np.percentile([g for _,g in events],q)) for q in (10,25,50,75,90)} if events else {},'gap_distribution_frames':{str(g):int(sum(x==g for _,x in events)) for g in sorted(set(x for _,x in events))},'candidates':{}}
 for name in ('v3b','v3be2'):
  y=load_labels(ROOT/'outputs/local_inference'/f'submission_{name}.csv'); transitions=set(np.flatnonzero((y[1:]!=y[:-1])&np.isin(y[1:],['H','A'])&np.isin(y[:-1],['H','A']))+1); unique={w:set() for w in (1,5,10,25)}; first=set()
  reliability=[]; gaprel=[]
  for t,g in events:
   for w in unique: unique[w].update(t+j for j in range(w) if t+j<len(y) and (t+j) in transitions)
   if t in transitions:first.add(t)
   reliability.append(float(rel[t])); gaprel.append((g,float(rel[t])))
  result['candidates'][name]={'first_visible_direct_switches':len(first),'unique_direct_switches_within_frames':{str(w):len(v) for w,v in unique.items()},'reliability_at_reappearance':{'mean':float(np.mean(reliability)) if reliability else 0,'median':float(np.median(reliability)) if reliability else 0},'motion_valid_at_reappearance':int(sum(motion[t]>.5 for t,_ in events)),'high_speed_reappearances':int(sum(speed[t]>=18 for t,_ in events))}
 Path(a.out).parent.mkdir(parents=True,exist_ok=True);Path(a.out).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
