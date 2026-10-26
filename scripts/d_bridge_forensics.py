"""Aggregate-only V1/V3 dead-ball bridge diagnostics."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import numpy as np
from scripts.diagnose_submission import load_labels,fill_half,segments
from games import half_video_blocks

def report(labels):
    blocks=[(a,b) for _,a,b in half_video_blocks("BAR-ATH")]; filled=labels.copy();
    for a,b in blocks: filled=fill_half(filled,a,b)
    d=segments(filled,*blocks[0],{"D"})+segments(filled,*blocks[1],{"D"}); spells=segments(filled,*blocks[0],{"H","A"})+segments(filled,*blocks[1],{"H","A"})
    bridges={k:0 for k in ("H-D-A","A-D-H","H-D-H","A-D-A")}
    for a,b,v in d:
        before=filled[a-1] if a>0 else "X"; after=filled[b] if b<len(filled) else "X"; key=f"{before}-D-{after}"
        if key in bridges: bridges[key]+=1
    lengths=np.asarray([b-a for a,b,_ in d]); return {"dead_segments":len(d),"dead_frames":int(lengths.sum()),"d_1_frame":int(np.sum(lengths==1)),"d_2_5_frames":int(np.sum((lengths>=2)&(lengths<=5))),"d_lt_1s":int(np.sum(lengths<25)),"d_lt_2s":int(np.sum(lengths<50)),"bridges":bridges,"possessions":len(spells)}

def main():
    p=argparse.ArgumentParser(); p.add_argument("v1"); p.add_argument("v3"); p.add_argument("--out",default=None); a=p.parse_args(); result={"v1":report(load_labels(a.v1)),"v3":report(load_labels(a.v3))};
    if a.out: Path(a.out).parent.mkdir(parents=True,exist_ok=True); Path(a.out).write_text(json.dumps(result,indent=2)+chr(10),encoding="utf-8")
    print(json.dumps(result,sort_keys=True))
if __name__=="__main__": tmain()
