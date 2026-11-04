import argparse,csv,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
import numpy as np
from games import half_video_blocks
from pulsoxi.turnover_stabilizer import stabilize_turnovers

def main():
 p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--seconds',type=float,default=1.5);a=p.parse_args()
 rows=list(csv.DictReader(Path(a.input).open(newline='',encoding='utf8'))); labels=np.array([r['label'] for r in rows]); frames=np.array([int(r['frame']) for r in rows]); periods=np.zeros(len(labels),dtype=int)
 for half,first,last in half_video_blocks('BAR-ATH'): periods[first:last+1]=half
 numeric=np.where(labels=='H',0,np.where(labels=='A',1,2)); out=stabilize_turnovers(numeric,periods=periods,frame_ids=frames,fps=25.0,max_reversal_seconds=a.seconds); new=np.where(out==0,'H',np.where(out==1,'A',labels))
 Path(a.output).parent.mkdir(parents=True,exist_ok=True)
 with Path(a.output).open('w',newline='',encoding='utf8') as h:
  w=csv.writer(h);w.writerow(['frame','label']);w.writerows(zip(frames,new))
 print({'rows':len(new),'changed_frames':int(np.sum(new!=labels)),'d_x_changed':int(np.sum(np.logical_or(labels=='D',labels=='X') != np.logical_or(new=='D',new=='X')))})
if __name__=='__main__': main()
