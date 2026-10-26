"""Local-only decoder ablation: V1 dead mask plus V2 H/A decisions."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent.parent; sys.path.insert(0,str(ROOT))
from games import GAME,half_video_blocks
from pulsoxi.atd_adapter import load_tactical_features
from pulsoxi.inference import infer_labels,infer_labels_v2
from scripts.diagnose_submission import segments

def summary(labels):
    spells=[]; dead=[]; transitions=[]
    for _,first,last in half_video_blocks(GAME):
        spells += segments(labels,first,last,{"H","A"}); dead += segments(labels,first,last,{"D"})
        transitions += [(f,labels[f-1],labels[f]) for f in range(first+1,last+1) if labels[f]!=labels[f-1]]
    d=np.asarray([(b-a)/25 for a,b,_ in spells]); return {"possessions":len(spells),"H":sum(v=="H" for _,_,v in spells),"A":sum(v=="A" for _,_,v in spells),"mean_s":float(np.mean(d)) if len(d) else None,"median_s":float(np.median(d)) if len(d) else None,"short_lt_1s":int(np.sum(d<1)),"short_lt_2s":int(np.sum(d<2)),"dead_segments":len(dead),"dead_frames":int(sum(b-a for a,b,_ in dead)),"direct_HA":sum(x[1:]==("H","A") for x in transitions),"direct_AH":sum(x[1:]==("A","H") for x in transitions),"D_transitions":sum("D" in x[1:] for x in transitions),"transitions":len(transitions),"graded_X":int(sum(np.sum(labels[a:b+1]=="X") for _,a,b in half_video_blocks(GAME)))}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--data-dir",default=None); ap.add_argument("--out",default="outputs/local_inference/decoder_forensics.json"); a=ap.parse_args()
    features,_,_=load_tactical_features(GAME,a.data_dir)
    from starter.load_smart_tagging import load as load_tags
    events=load_tags(GAME,a.data_dir)
    v1,_,_=infer_labels(features,smart_tagging_events=events); v2,_,_=infer_labels_v2(features,smart_tagging_events=events)
    hybrid=v1.copy()
    for _,first,last in half_video_blocks(GAME):
        live=v1[first:last+1]!="D"; replacement=v2[first:last+1]; hybrid[first:last+1][live & np.isin(replacement,["H","A"])]=replacement[live & np.isin(replacement,["H","A"])]
    report={"v1":summary(v1),"v2":summary(v2),"hybrid_v1_dead_v2_live":summary(hybrid)}
    Path(a.out).parent.mkdir(parents=True,exist_ok=True); Path(a.out).write_text(json.dumps(report,indent=2)+chr(10),encoding="utf-8"); print(json.dumps(report,sort_keys=True))
if __name__=="__main__": main()
