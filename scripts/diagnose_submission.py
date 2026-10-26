"""Local-only aggregate diagnostics for a PulsoXI submission.

Reads a local submission CSV and optionally a local feature NPZ. Prints aggregate statistics only;
keep both inputs and detailed JSON reports outside the public repository.
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
import numpy as np
from games import half_video_blocks
from pulsoxi.temporal import N_FRAMES


def load_labels(path):
    with Path(path).open(newline="", encoding="utf-8") as fh:
        rows=list(csv.DictReader(fh))
    if len(rows)!=N_FRAMES or [int(r["frame"]) for r in rows]!=list(range(N_FRAMES)):
        raise ValueError("submission must contain frames 0..199318 in order")
    labels=np.asarray([r["label"] for r in rows],dtype="U1")
    if not np.all(np.isin(labels,["H","A","D","X"])): raise ValueError("invalid label")
    return labels


def fill_half(labels, first, last):
    out=labels.copy(); part=out[first:last+1]; known=np.flatnonzero(part!="X")
    if len(known):
        first_known=int(known[0]); part[:first_known]=part[first_known]
        for i in range(first_known+1,len(part)):
            if part[i]=="X": part[i]=part[i-1]
        out[first:last+1]=part
    return out


def segments(labels, first, last, allowed):
    out=[]; start=None; value=None
    for frame in range(first,last+2):
        current=labels[frame] if frame<=last else None
        if current not in allowed:
            if start is not None: out.append((start,frame,value))
            start=None; value=None
        elif start is None: start=frame; value=current
        elif current!=value:
            out.append((start,frame,value)); start=frame; value=current
    return out


def main():
    ap=argparse.ArgumentParser(description="Aggregate local PulsoXI possession diagnostics")
    ap.add_argument("submission"); ap.add_argument("--features-npz",default=None); ap.add_argument("--out",default=None); args=ap.parse_args()
    raw=load_labels(args.submission); blocks=[(a,b) for _,a,b in half_video_blocks("BAR-ATH")]
    filled=raw.copy()
    for first,last in blocks: filled=fill_half(filled,first,last)
    spells=[]; dead=[]
    for first,last in blocks:
        spells += segments(filled,first,last,{"H","A"})
        dead += segments(filled,first,last,{"D"})
    durations=np.asarray([b-a for a,b,_ in spells],dtype=float)/25.0
    direct=sum(1 for a,b,v in zip(spells,spells[1:],spells[1:]) if a[1]==b[0] and a[2]!=b[2]) if False else 0
    transitions=[]
    for first,last in blocks:
        for frame in range(first+1,last+1):
            if filled[frame]!=filled[frame-1]: transitions.append((frame,filled[frame-1],filled[frame]))
    result={"raw_label_counts":{k:int(np.sum(raw==k)) for k in "HADX"},"graded_frames":sum(b-a+1 for a,b in blocks),"graded_x":int(sum(np.sum(raw[a:b+1]=="X") for a,b in blocks)),"spells_total":len(spells),"spells_by_team":{k:int(sum(v==k for _,_,v in spells)) for k in "HA"},"spells_by_half":{str(h+1):int(sum(a<=s<=b for s,_,_ in spells)) for h,(a,b) in enumerate(blocks)},"duration_seconds":({"mean":float(np.mean(durations)) if len(durations) else None,"median":float(np.median(durations)) if len(durations) else None,"p10":float(np.percentile(durations,10)) if len(durations) else None,"p25":float(np.percentile(durations,25)) if len(durations) else None,"p75":float(np.percentile(durations,75)) if len(durations) else None,"p90":float(np.percentile(durations,90)) if len(durations) else None,"short_lt_0_5s":int(np.sum(durations<0.5)),"short_lt_1s":int(np.sum(durations<1.0)),"short_lt_2s":int(np.sum(durations<2.0))}),"dead_segments":len(dead),"dead_duration_seconds":{"mean":float(np.mean([b-a for a,b,_ in dead])/25) if dead else None,"median":float(np.median([b-a for a,b,_ in dead])/25) if dead else None},"transitions_total":len(transitions),"direct_H_to_A":sum(x[1:]==("H","A") for x in transitions),"direct_A_to_H":sum(x[1:]==("A","H") for x in transitions),"transitions_involving_D":sum("D" in x[1:] for x in transitions)}
    if args.features_npz:
        data=np.load(args.features_npz,allow_pickle=False); visible=data["ball_visible"].astype(bool); result["transitions_during_missing_ball"]=int(sum(not visible[f] for f,_,_ in transitions)); result["transitions_after_ball_reappears"]=int(sum(f>0 and visible[f] and not visible[f-1] for f,_,_ in transitions))
    if args.out: Path(args.out).parent.mkdir(parents=True,exist_ok=True); Path(args.out).write_text(json.dumps(result,indent=2)+chr(10),encoding="utf-8")
    print(json.dumps(result,sort_keys=True))

if __name__=="__main__": main()