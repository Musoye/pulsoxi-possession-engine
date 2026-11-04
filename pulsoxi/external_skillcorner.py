"""Memory-bounded adapter for public SkillCorner football tracking JSONL."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
FPS = 10.0
def iter_frames(path):
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip(): yield json.loads(line)
def possession_group(frame):
    value=frame.get("possession"); group=value.get("group") if isinstance(value,dict) else None
    return {"home team":"home","away team":"away"}.get(group, group if group in {"home","away"} else None)
def frame_inputs(frame, team_by_player):
    ball=frame.get("ball_data") or {}; players=frame.get("player_data") or []
    return {"frame":frame.get("frame"),"period":frame.get("period"),"timestamp":frame.get("timestamp"),"ball_xy":np.asarray([ball.get("x",np.nan),ball.get("y",np.nan)],float),"ball_detected":bool(ball.get("is_detected",False)),"player_xy":np.asarray([[p.get("x",np.nan),p.get("y",np.nan)] for p in players],float),"player_detected":np.asarray([bool(p.get("is_detected",False)) for p in players]),"player_ids":tuple(p.get("player_id") for p in players),"player_teams":tuple(team_by_player.get(p.get("player_id")) for p in players)}
def contiguous_runs(groups):
    runs=[]; start=None; current=None
    for i,group in enumerate(list(groups)+[None]):
        if group!=current:
            if current in {"home","away"} and start is not None: runs.append((current,start,i-1))
            start=i if group in {"home","away"} else None; current=group
    return runs

