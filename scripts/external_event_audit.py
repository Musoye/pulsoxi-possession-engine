from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent.parent;sys.path.insert(0,str(ROOT))
from pulsoxi.external_metrica import read_tracking,read_events,event_frame,nearest_team
from pulsoxi.tracking_features import extract_tracking_features
from pulsoxi.inference import generate_emissions,generate_emissions_e2

def anchors_for(events,home,away):
 lost=[e for e in events if e.get('Type')=='BALL LOST' and e.get('Subtype')=='INTERCEPTION']; rec=[e for e in events if e.get('Type')=='RECOVERY' and e.get('Subtype')=='INTERCEPTION']; anchors=[]; excluded=0
 for r in rec:
  rf=event_frame(r); candidates=[l for l in lost if l.get('Team') and r.get('Team') and l.get('Team')!=r.get('Team') and l.get('Period')==r.get('Period') and event_frame(l) is not None and rf is not None and 0<=rf-event_frame(l)<=25]
  if len(candidates)!=1 or rf is None: excluded+=1; continue
  obs=[nearest_team(f,home,away) for f in range(rf-5,rf+6)]; obs=[x for x in obs if x]
  if not obs: excluded+=1; continue
  predicted=max(set(x[0] for x in obs),key=lambda k:sum(x[0]==k for x in obs)); anchors.append((r,rf,predicted))
 return rec,anchors,excluded

def local_features(frame,home,away):
 if frame not in home or frame not in away: return None
 ball=np.asarray(home[frame][1]); players=np.concatenate((home[frame][2],away[frame][2]))
 if not np.all(np.isfinite(ball)): return None
 x,_=extract_tracking_features(ball[None,:],players[None,:,:],np.r_[np.zeros(len(home[frame][2]),dtype=int),np.ones(len(away[frame][2]),dtype=int)])
 return x

def score_anchor(rf,expected,home,away):
 frames=list(range(rf-5,rf+6)); rows=[]; actual=[]
 for f in frames:
  x=local_features(f,home,away)
  if x is not None: rows.append(x[0]); actual.append(f)
 if not rows:return None
 x=np.asarray(rows); e1,_=generate_emissions(x); e2,_=generate_emissions_e2(x)
 def score(e):
  teams=np.where(np.argmax(e[:,:2],axis=1)==0,'Home','Away'); pred=max(set(teams),key=lambda k:sum(teams==k)); return pred
 return {'expected':expected,'nearest':max(set(x[0] for x in [nearest_team(f,home,away) for f in actual] if x),key=lambda k:sum(nearest_team(f,home,away)[0]==k for f in actual if nearest_team(f,home,away))),'e1':score(e1),'e2':score(e2),'support_frames':len(actual)}

def audit(game,root):
 base=Path(root)/'data'/f'Sample_Game_{game}'; home=read_tracking(base/f'Sample_Game_{game}_RawTrackingData_Home_Team.csv'); away=read_tracking(base/f'Sample_Game_{game}_RawTrackingData_Away_Team.csv'); events=read_events(base/f'Sample_Game_{game}_RawEventsData.csv'); rec,anchors,excluded=anchors_for(events,home,away); scored=[score_anchor(rf,r.get('Team'),home,away) for r,rf,_ in anchors]; scored=[x for x in scored if x]; out={'game':game,'tracking_frames_home':len(home),'tracking_frames_away':len(away),'events_total':len(events),'recovery_interception_events':len(rec),'accepted_anchor_conditions':len(anchors),'excluded_anchor_conditions':excluded,'unscored_after_acceptance':len(anchors)-len(scored),'silver_label_note':'Event-derived anchors, not continuous ground truth or official spellF1.'}
 for key in ('nearest','e1','e2'):
  out[key]={'correct':sum(x[key]==x['expected'] for x in scored),'total':len(scored),'accuracy':sum(x[key]==x['expected'] for x in scored)/len(scored) if scored else None}
 return out

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',default='external_data/metrica');p.add_argument('--out',required=True);a=p.parse_args(); result={'sample_game_1':audit(1,a.root),'sample_game_2':audit(2,a.root)}; Path(a.out).parent.mkdir(parents=True,exist_ok=True);Path(a.out).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
