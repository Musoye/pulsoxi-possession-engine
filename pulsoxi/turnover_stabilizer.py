from __future__ import annotations
import numpy as np

def _runs(labels, periods, frames):
 out=[]; i=0
 while i<len(labels):
  j=i+1
  while j<len(labels) and labels[j]==labels[i] and periods[j]==periods[i] and frames[j]==frames[j-1]+1: j+=1
  out.append([labels[i],i,j,periods[i]]); i=j
 return out

def stabilize_turnovers(labels, *, periods, frame_ids, fps, max_reversal_seconds):
 x=np.asarray(labels).copy(); p=np.asarray(periods); f=np.asarray(frame_ids)
 if len(x)!=len(p) or len(x)!=len(f) or fps<=0 or max_reversal_seconds<0: raise ValueError("aligned labels, periods and positive fps required")
 runs=_runs(x,p,f); limit=max_reversal_seconds*fps; k=1
 while k<len(runs)-1:
  l,m,r=runs[k-1:k+2]
  continuous=l[3]==m[3]==r[3] and f[l[2]-1]+1==f[m[1]] and f[m[2]-1]+1==f[r[1]]
  if l[0] in (0,1) and m[0] in (0,1) and r[0]==l[0] and m[0]!=l[0] and continuous and (m[2]-m[1])<=limit:
   l[2]=r[2]; runs[k-1:k+2]=[l]
   k=max(1,k-1)
  else: k+=1
 for label,a,b,_ in runs: x[a:b]=label
 return x
