import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from pathlib import Path
import numpy as np
from pulsoxi.external_skillcorner import iter_frames,frame_inputs,possession_group,contiguous_runs
from pulsoxi.inference import generate_emissions,generate_emissions_e2, _decode_hysteresis, HysteresisConfig, E2Config
from pulsoxi.external_interval_metrics import spells, overlap_scores, boundary_scores
from pulsoxi.turnover_stabilizer import stabilize_turnovers

def load_map(path):
 m=json.loads(Path(path).read_text(encoding="utf8")); out={}
 for p in m["players"]: out[p["id"]]="home" if p["team_id"]==m["home_team"]["id"] else "away"
 return out

def run(root,mid):
 d=Path(root)/str(mid); mp=load_map(d/f"{mid}_match.json"); frames=[]; refs=[]; periods=[]; frame_ids=[]
 for raw in iter_frames(d/f"{mid}_tracking_extrapolated.jsonl"):
  if raw.get("period") not in (1,2): continue
  o=frame_inputs(raw,mp); frames.append(o); refs.append(possession_group(raw)); periods.append(raw.get("period")); frame_ids.append(raw.get("frame"))
 n=len(frames); ids=sorted({p for o in frames for p in o["player_ids"] if p is not None}); idx={p:i for i,p in enumerate(ids)}; xy=np.full((n,len(ids),2),np.nan); teams=np.empty(len(ids),dtype=int); teams[:]=1
 for p,i in idx.items(): teams[i]=0 if mp.get(p)=="home" else 1
 ball=np.full((n,2),np.nan); visible=np.zeros(n,bool); detected=np.zeros(n,bool)
 for j,o in enumerate(frames):
  ball[j]=o["ball_xy"]; detected[j]=o["ball_detected"]; visible[j]=np.all(np.isfinite(ball[j]))
  for p,v in zip(o["player_ids"],o["player_xy"]):
   if p in idx: xy[j,idx[p]]=v
 f=np.full((n,17),np.nan); f[:,0]=visible; f[:,1:3]=ball
 ok=np.all(np.isfinite(xy),axis=2)
 for t,c,cc,dc in ((0,3,(5,6),9),(1,4,(7,8),10)):
  sel=teams==t; present=ok[:,sel]; pts=xy[:,sel]; f[:,c]=present.sum(1)
  for a,col in enumerate(cc): f[:,col]=np.divide(np.where(present,pts[:,:,a],0).sum(1),f[:,c],out=np.full(n,np.nan),where=f[:,c]>0)
  dist=np.linalg.norm(pts-ball[:,None,:],axis=2); dist[~present|~visible[:,None]]=np.inf; m=np.min(dist,1); f[:,dc]=np.where(np.isfinite(m),m,np.nan)
 f[:,11]=f[:,9]-f[:,10]; f[:,15]=np.cumsum(~visible); f[visible,15]=0
 for i in range(1,n):
  if visible[i] and visible[i-1] and frames[i]["period"]==frames[i-1]["period"]:
   f[i,12]=np.linalg.norm(ball[i]-ball[i-1]); f[i,13]=f[i,12]*10
 f[:,16]=0
 e1,k1=generate_emissions(f); e2,k2=generate_emissions_e2(f,e2=E2Config(persistence_frames=2))
 valid=np.array([r in {"home","away"} for r in refs]); pred=np.where(np.argmax(e1[:,:2],1)==0,"home","away")
 def interval_eval(labels):
  mapped=np.where(labels==0,"home",np.where(labels==1,"away","X")); ref_intervals=[]; pred_intervals=[]; eval_frames=0; i=0
  while i<n:
   if refs[i] not in {"home","away"}: i+=1; continue
   j=i+1
   while j<n and refs[j]==refs[i] and periods[j]==periods[i]: j+=1
   left=i>0 and refs[i-1] in {"home","away"} and periods[i-1]==periods[i]
   right=j<n and refs[j] in {"home","away"} and periods[j-1]==periods[j]
   if left and right:
    ref_intervals.append((refs[i],i,j)); eval_frames+=j-i; pred_intervals.extend((x,i+u,i+v) for x,u,v in spells(mapped[i:j]))
   i=j
  ov=overlap_scores(ref_intervals,pred_intervals); bnd=boundary_scores(ref_intervals,pred_intervals,5); return {"evaluable_reference_possessions":len(ref_intervals),"evaluable_frames":eval_frames,"predicted_intervals":len(pred_intervals),"overlap":ov,"boundary_pm_0_5s":bnd,"boundary_pm_1_0s":boundary_scores(ref_intervals,pred_intervals,10)}

 def stats(p,k):
  mask=valid&k; return {"agreement":int(np.sum(p[mask]==np.array(refs)[mask])),"compared":int(mask.sum()),"pred_boundaries":int(np.sum(p[1:]!=p[:-1])),"reference_runs":len(contiguous_runs(refs))}
 d1=_decode_hysteresis(e1,k1,config=HysteresisConfig()); d2=_decode_hysteresis(e2,k2,config=HysteresisConfig())
 def run_diagnostics(labels):
  runs=[]; i=0
  while i<n:
   j=i+1
   while j<n and labels[j]==labels[i] and periods[j]==periods[i] and frame_ids[j]==frame_ids[j-1]+1: j+=1
   runs.append((int(labels[i]),i,j,periods[i])); i=j
  patterns={"all":0,"by_threshold":{str(t):0 for t in (.3,.5,.8,1.0,1.5)},"middle_d":0,"middle_x":0,"period_or_gap":0}
  for a,b,c in zip(runs,runs[1:],runs[2:]):
   if a[0] in (0,1) and c[0]==a[0] and b[0] in (0,1) and b[0]!=a[0]:
    patterns["all"]+=1; duration=(b[2]-b[1])/10.0
    for t in (.3,.5,.8,1.0,1.5):
     if duration<t: patterns["by_threshold"][str(t)]+=1
   elif a[0] in (0,1) and c[0]==a[0] and b[0]==2: patterns["middle_d"]+=1
   elif a[0] in (0,1) and c[0]==a[0] and b[0] not in (0,1,2): patterns["middle_x"]+=1
  return {"decoded_runs":len(runs),"decoded_boundaries":len(runs)-1,"reversals":patterns}
 decoded_diagnostics=run_diagnostics(d2)
 stabilized_metrics={str(t):dict(interval_eval((z:=stabilize_turnovers(d2,periods=periods,frame_ids=frame_ids,fps=10,max_reversal_seconds=t))),changed_frames=int(np.sum(z!=d2)),merged_reversals=int(np.sum(z!=d2))) for t in (.3,.5,.8,1.0,1.5)}
 print(json.dumps({"match":mid,"frames":n,"valid_reference_frames":int(valid.sum()),"unknown_frames":int((~valid).sum()),"complete_reference_runs":len(contiguous_runs(refs)),"e1":stats(np.where(np.argmax(e1[:,:2],1)==0,"home","away"),k1),"e2":stats(np.where(np.argmax(e2[:,:2],1)==0,"home","away"),k2),"nearest":stats(np.where(np.nan_to_num(f[:,9],nan=np.inf)<np.nan_to_num(f[:,10],nan=np.inf),"home","away"),np.isfinite(f[:,9])&np.isfinite(f[:,10])),"temporal_e1_boundaries":int(np.sum(d1[1:]!=d1[:-1])),"temporal_e2_boundaries":int(np.sum(d2[1:]!=d2[:-1])),"decoded_diagnostics":decoded_diagnostics,"intervals_nearest":interval_eval(np.where(np.nan_to_num(f[:,9],nan=np.inf)<np.nan_to_num(f[:,10],nan=np.inf),0,1)),"intervals_e1":interval_eval(d1),"intervals_e2":interval_eval(d2),"stabilized":stabilized_metrics},indent=2))
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--root",default="external_data/skillcorner");p.add_argument("--match",type=int,required=True);a=p.parse_args();run(a.root,a.match)













