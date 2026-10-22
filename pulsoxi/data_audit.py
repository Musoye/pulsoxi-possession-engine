"""Licence-safe tracking data quality summaries."""
from __future__ import annotations
import numpy as np
from .tracking_features import FEATURE_NAMES


def _summary(values):
    values=np.asarray(values,dtype=float); finite=values[np.isfinite(values)]
    return {"count": int(len(values)), "missing": int(len(values)-len(finite)), "min": float(np.min(finite)) if len(finite) else None, "max": float(np.max(finite)) if len(finite) else None, "mean": float(np.mean(finite)) if len(finite) else None}


def audit_arrays(frame_ids, ball_xy, player_xy, team_of_slot, features=None, fps=25.0, max_jump_m=15.0):
    """Return a concise audit dict; flags are reported, never silently repaired."""
    ids=np.asarray(frame_ids,dtype=np.int64); ball=np.asarray(ball_xy,float); players=np.asarray(player_xy,float)
    visible=np.all(np.isfinite(ball),axis=1)
    gaps=[]; gap=0
    for value in visible:
        if value: gap=0
        else: gap+=1
        if gap: gaps.append(gap)
    player_ok=np.all(np.isfinite(players),axis=2)
    jumps=np.linalg.norm(np.diff(players * np.array([105.3,68.0]),axis=0),axis=2)
    valid_jump=player_ok[1:] & player_ok[:-1]
    suspicious=int(np.sum((jumps > max_jump_m) & valid_jump))
    duplicate_frames=int(np.sum(ids[1:] == ids[:-1])) if len(ids)>1 else 0
    report={"frame_count":int(len(ids)),"frame_id_first":int(ids[0]) if len(ids) else None,"frame_id_last":int(ids[-1]) if len(ids) else None,"frame_ids_contiguous":bool(len(ids) and np.array_equal(ids,np.arange(ids[0],ids[0]+len(ids)))),"ball_visibility_fraction":float(np.mean(visible)) if len(visible) else None,"ball_missing_gap_frames":_summary(gaps),"player_count_distribution":_summary(player_ok.sum(axis=1)),"slot_coverage":{str(i+1):float(np.mean(player_ok[:,i])) for i in range(player_ok.shape[1])},"duplicate_frame_ids":duplicate_frames,"coordinate_out_of_range":int(np.sum((ball < 0) | (ball > 1)) + np.sum((players < 0) | (players > 1))),"suspicious_position_jumps":suspicious}
    if features is not None:
        f=np.asarray(features); report["feature_missingness"]={name:float(np.mean(~np.isfinite(f[:,i]))) for i,name in enumerate(FEATURE_NAMES)}
    return report

