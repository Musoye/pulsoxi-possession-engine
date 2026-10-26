"""Canonical, aggregate-only diagnostics for PulsoXI predictions."""
from __future__ import annotations
import numpy as np
from games import half_video_blocks, GAME

def half_mask(n, video_frames=None):
    if video_frames is None: return np.ones(n,bool)
    frames=np.asarray(video_frames)
    mask=np.zeros(n,bool)
    for _,first,last in half_video_blocks(GAME): mask |= (frames>=first)&(frames<=last)
    return mask

def possession_spans(labels, video_frames=None):
    """Return every contiguous H/A span, bounded by official halves and X/D."""
    y=np.asarray(labels); n=len(y); frames=np.arange(n) if video_frames is None else np.asarray(video_frames)
    if len(frames)!=n: raise ValueError('video_frames length mismatch')
    out=[]; valid=half_mask(n,video_frames); i=0
    while i<n:
        if not valid[i] or y[i] not in ('H','A'): i+=1; continue
        j=i+1
        while j<n and valid[j] and y[j]==y[i] and frames[j]==frames[j-1]+1: j+=1
        out.append((i,j,y[i])); i=j
    return out

def duration_summary(labels, fps=25.0, video_frames=None):
    spans=possession_spans(labels,video_frames); d=np.asarray([(b-a)/fps for a,b,_ in spans],float)
    return {'population':'all contiguous H/A spans within official graded halves','denominator':int(len(d)),'durations_seconds':d,'mean':float(np.mean(d)) if len(d) else 0.0,'median':float(np.median(d)) if len(d) else 0.0,'p10':float(np.percentile(d,10)) if len(d) else 0.0,'p25':float(np.percentile(d,25)) if len(d) else 0.0,'p75':float(np.percentile(d,75)) if len(d) else 0.0,'p90':float(np.percentile(d,90)) if len(d) else 0.0,'lt_0_5':int(np.sum(d<.5)),'lt_1':int(np.sum(d<1)),'lt_2':int(np.sum(d<2))}

def reappearance_events(ball_visible, video_frames=None):
    """Return (frame, missing_gap_frames) only for true within-half reappearances."""
    v=np.asarray(ball_visible,bool); n=len(v); frames=np.arange(n) if video_frames is None else np.asarray(video_frames)
    valid=half_mask(n,video_frames); events=[]; gap=0
    for i in range(n):
        continuous=i>0 and frames[i]==frames[i-1]+1 and valid[i] and valid[i-1]
        if not valid[i] or not continuous: gap=0; continue
        if v[i]:
            if gap>0: events.append((i,gap))
            gap=0
        else: gap+=1
    return events

def assert_duration_invariants(summary):
    assert 0 <= summary['lt_0_5'] <= summary['lt_1'] <= summary['lt_2'] <= summary['denominator']