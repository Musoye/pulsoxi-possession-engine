"""Minimal adapter for public Metrica Sample Games 1-2 CSV exports.

This adapter produces event-local nearest-team observations only; it is not a
replacement for the licensed BAR-ATH ATD adapter and does not create possession labels.
"""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np

PITCH_LENGTH_M=105.0
PITCH_WIDTH_M=68.0
FPS=25.0

def read_tracking(path):
    """Read one public team CSV into frame -> (period, ball_xy, player_xy)."""
    rows={}
    with Path(path).open(newline='',encoding='utf-8') as f:
        reader=csv.reader(f); next(reader); next(reader); next(reader)
        for row in reader:
            if not row or len(row)<5: continue
            try: period=int(row[0]); frame=int(row[1])
            except (ValueError,TypeError): continue
            vals=[]
            for i in range(3,len(row)-1,2):
                try: vals.append((float(row[i]),float(row[i+1])))
                except (ValueError,TypeError): vals.append((np.nan,np.nan))
            ball=vals[-1] if vals else (np.nan,np.nan)
            rows[frame]=(period,np.asarray(ball,float),np.asarray(vals[:-1],float))
    return rows

def read_events(path):
    with Path(path).open(newline='',encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))

def event_frame(event):
    try: return int(round(float(event['Start Frame'])))-1
    except (ValueError,TypeError,KeyError): return None

def nearest_team(frame, home, away):
    if frame not in home or frame not in away: return None
    ball=home[frame][1]
    if not np.all(np.isfinite(ball)): return None
    distances=[]
    for team,rows in (('Home',home),('Away',away)):
        players=rows[frame][2]; ok=np.all(np.isfinite(players),axis=1)
        distances.append(float(np.min(np.linalg.norm((players[ok]-ball)*[PITCH_LENGTH_M,PITCH_WIDTH_M],axis=1))) if np.any(ok) else np.inf)
    if not np.isfinite(distances).any() or distances[0]==distances[1]: return None
    return ('Home' if distances[0]<distances[1] else 'Away', abs(distances[0]-distances[1]))
