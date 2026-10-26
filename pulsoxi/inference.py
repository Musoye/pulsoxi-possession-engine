"""Explainable first-pass possession inference from engineered tracking features.

This is a deterministic baseline candidate, not a trained model. It consumes only feature arrays and
public Smart Tagging-shaped event dictionaries; the local runner is responsible for loading licensed data.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .temporal import N_FRAMES, HALVES, decode_full_match, TransitionCosts
from .temporal_v3 import decode_full_match_v3
from .temporal_v3b import decode_full_match_v3b

@dataclass(frozen=True)
class InferenceConfig:
    distance_scale_m: float = 2.0
    motion_bonus: float = 0.25
    missing_decay: float = 0.97
    dead_event_window_frames: int = 25
    dead_evidence: float = 6.0
    empty_team_dead_evidence: float = 1.0
    transition_costs: TransitionCosts = TransitionCosts()


def _event_dead_frames(events, n, config):
    score = np.zeros(n, dtype=float)
    team_bias = np.zeros(n, dtype=float)
    for event in events or ():
        code = str(event.get("code") or "").strip().lower()
        start = int(event.get("start_frame", -1)); end = int(event.get("end_frame", start))
        if start < 0 or start >= n or end < start:
            continue
        if code == "set pieces":
            lo=max(0, start-config.dead_event_window_frames); hi=min(n, end+config.dead_event_window_frames+1)
            score[lo:hi] = np.maximum(score[lo:hi], config.dead_evidence)
        team=str((event.get("tags") or {}).get("Team") or "").strip().lower()
        if team in {"barcelona", "barça"}:
            team_bias[max(0,start):min(n,end+1)] += 0.35
        elif team in {"atletic club", "athletic club", "athletic"}:
            team_bias[max(0,start):min(n,end+1)] -= 0.35
    return score, team_bias


def generate_emissions(features, *, smart_tagging_events=None, config=InferenceConfig()):
    """Return nonnegative H/A/D evidence and a reliability mask.

    Distance evidence is based on the two minimum player-ball distances. Missing ball frames
    inherit the prior team evidence with exponential decay; they are not automatically dead.
    """
    x=np.asarray(features,dtype=float)
    if x.ndim != 2 or x.shape[1] < 17:
        raise ValueError("features must have at least 17 columns from tracking_features")
    n=len(x); out=np.ones((n,3),dtype=float); known=np.zeros(n,dtype=bool)
    home, away = x[:,9], x[:,10]
    visible=np.isfinite(x[:,0]) & (x[:,0] > 0.5)
    diff=(away-home)/config.distance_scale_m
    score=np.zeros(n,dtype=float)
    for i in range(n):
        if np.isfinite(diff[i]):
            score[i]=diff[i]; known[i]=True
        elif i:
            score[i]=score[i-1]*config.missing_decay
        if i and np.isfinite(x[i,14]) and x[i,16] > 0.5:
            score[i] += config.motion_bonus if score[i] >= 0 else -config.motion_bonus
    dead, bias=_event_dead_frames(smart_tagging_events,n,config)
    out[:,0]=np.exp(score+bias); out[:,1]=np.exp(-score-bias); out[:,2]=np.exp(dead)
    empty=(x[:,3] + x[:,4]) <= 0
    out[empty,2] *= np.exp(config.empty_team_dead_evidence)
    out[~visible & (dead <= 0),2] = 1.0
    out /= out.sum(axis=1,keepdims=True)
    return out, known


def infer_labels(features, *, video_frames=None, smart_tagging_events=None, config=InferenceConfig()):
    """Generate full-match labels; outside official halves the decoder emits X."""
    x=np.asarray(features)
    if video_frames is not None and len(video_frames) != len(x):
        raise ValueError("video_frames must match feature rows")
    if len(x) != N_FRAMES:
        raise ValueError(f"expected {N_FRAMES} feature rows, received {len(x)}")
    emissions, known=generate_emissions(x, smart_tagging_events=smart_tagging_events, config=config)
    labels=decode_full_match(emissions, known=known, costs=config.transition_costs)
    return labels, emissions, known


def diagnostics(labels, known):
    labels=np.asarray(labels); known=np.asarray(known,dtype=bool)
    changes=int(np.sum(labels[1:] != labels[:-1]))
    return {"label_counts": {str(k): int(v) for k,v in zip(*np.unique(labels,return_counts=True))}, "transitions": changes, "known_fraction": float(np.mean(known)), "abstention_fraction": float(np.mean(labels=="X"))}


@dataclass(frozen=True)
class HysteresisConfig:
    challenger_margin: float = 0.35
    switch_evidence: float = 1.25
    strong_switch_margin: float = 1.50
    evidence_decay: float = 0.90
    dead_margin: float = 1.20


def _decode_hysteresis(probs, known, *, config):
    """Decode one half with accumulated opposing-team evidence.

    A challenger accumulates positive log(p_challenger / p_current) evidence. Weak proximity
    differences decay before reaching the switch threshold; a large, reliable margin can switch
    immediately. Missing-ball frames hold the current state and decay challenger evidence.
    """
    probs=np.asarray(probs,float); known=np.asarray(known,bool); result=np.zeros(len(probs),dtype=np.uint8)
    state=0; accumulated=0.0
    for i in range(len(probs)):
        live=probs[i,:2]; dead=probs[i,2]
        if dead > max(live) * config.dead_margin and known[i]:
            result[i]=2; accumulated=0.0; continue
        if state==2:
            state=int(np.argmax(live)); accumulated=0.0
        if not known[i]:
            accumulated *= config.evidence_decay; result[i]=state; continue
        challenger=1-state
        margin=float(np.log(max(live[challenger],1e-12))-np.log(max(live[state],1e-12)))
        if margin > config.challenger_margin:
            accumulated = accumulated * config.evidence_decay + margin
        else:
            accumulated *= config.evidence_decay
        if margin >= config.strong_switch_margin or accumulated >= config.switch_evidence:
            state=challenger; accumulated=0.0
        result[i]=state
    return result


def infer_labels_v2(features, *, video_frames=None, smart_tagging_events=None,
                     config=InferenceConfig(), hysteresis=HysteresisConfig()):
    """Candidate V2: V1 emissions with evidence-accumulated H/A hysteresis."""
    x=np.asarray(features)
    if len(x)!=N_FRAMES: raise ValueError(f"expected {N_FRAMES} feature rows, received {len(x)}")
    emissions, known=generate_emissions(x, smart_tagging_events=smart_tagging_events, config=config)
    labels=np.full(N_FRAMES,"X",dtype="<U1"); states=np.array(("H","A","D"))
    for first,last in HALVES:
        loc=slice(first,last+1); decoded=_decode_hysteresis(emissions[loc],known[loc],config=hysteresis)
        labels[loc]=states[decoded]
    return labels, emissions, known

@dataclass(frozen=True)
class ReliabilityV3Config:
    uncertainty_penalty: float = 2.0
    distance_margin_scale_m: float = 4.0


def build_reliability(features, *, config=ReliabilityV3Config()):
    """Build bounded transition confidence without ground-truth-derived calibration."""
    x=np.asarray(features,dtype=float)
    visible=(np.isfinite(x[:,0]) & (x[:,0]>0.5)).astype(float)
    margin=np.abs(x[:,10]-x[:,9]); margin=np.nan_to_num(margin,nan=0.0,posinf=0.0,neginf=0.0); margin_conf=np.clip(margin/config.distance_margin_scale_m,0,1)
    motion=np.where(np.isfinite(x[:,16]),np.clip(x[:,16],0,1),0.0); motion=np.nan_to_num(motion,nan=0.0,posinf=0.0,neginf=0.0)
    return visible*(0.65*margin_conf+0.35*motion)


def infer_labels_v3(features, *, video_frames=None, smart_tagging_events=None,
                     config=InferenceConfig(), reliability=ReliabilityV3Config()):
    """Candidate V3: V1 joint decoder with reliability-aware H/A costs."""
    x=np.asarray(features)
    if len(x)!=N_FRAMES: raise ValueError(f"expected {N_FRAMES} feature rows, received {len(x)}")
    emissions,known=generate_emissions(x,smart_tagging_events=smart_tagging_events,config=config)
    confidence=build_reliability(x,config=reliability)
    labels=decode_full_match_v3(emissions,confidence,known=known,costs=config.transition_costs,uncertainty_penalty=reliability.uncertainty_penalty)
    return labels, emissions, confidence





def infer_labels_v3b(features, *, video_frames=None, smart_tagging_events=None,
                      config=InferenceConfig(), reliability=ReliabilityV3Config()):
    """Controlled ablation: V1 D/X decisions plus V3 reliability-aware live H/A decoding."""
    x=np.asarray(features)
    if len(x)!=N_FRAMES: raise ValueError(f"expected {N_FRAMES} feature rows, received {len(x)}")
    emissions,known=generate_emissions(x,smart_tagging_events=smart_tagging_events,config=config)
    v1=decode_full_match(emissions,known=known,costs=config.transition_costs)
    confidence=build_reliability(x,config=reliability)
    labels=decode_full_match_v3b(emissions,confidence,v1,known=known,costs=config.transition_costs,uncertainty_penalty=reliability.uncertainty_penalty)
    return labels,emissions,confidence


@dataclass(frozen=True)
class E2Config:
    """Untuned, interpretable football-aware emission coefficients."""
    distance_scale_m: float = 2.0
    persistence_frames: int = 5
    persistence_weight: float = 0.35
    motion_weight: float = 0.20
    ball_speed_scale_mps: float = 18.0
    discontinuity_speed_mps: float = 25.0
    missing_decay: float = 0.97


def generate_emissions_e2(features, *, smart_tagging_events=None, config=InferenceConfig(), e2=E2Config()):
    """E2: causal proximity persistence plus valid relative-motion evidence."""
    x=np.asarray(features,dtype=float)
    if x.ndim != 2 or x.shape[1] < 17: raise ValueError("features must have at least 17 columns")
    n=len(x); out=np.ones((n,3),float); known=np.zeros(n,bool)
    visible=np.isfinite(x[:,0]) & (x[:,0] > .5)
    raw=(x[:,10]-x[:,9])/e2.distance_scale_m
    raw=np.nan_to_num(raw,nan=0.0,posinf=0.0,neginf=0.0); raw=np.clip(raw,-4.0,4.0)
    score=np.zeros(n,float)
    for i in range(n):
        if visible[i] and np.isfinite(x[i,9]) and np.isfinite(x[i,10]):
            lo=max(0,i-e2.persistence_frames+1)
            score[i]=(1-e2.persistence_weight)*raw[i]+e2.persistence_weight*float(np.mean(raw[lo:i+1]))
            known[i]=True
            if x[i,16] > .5 and np.isfinite(x[i,14]):
                score[i] += np.sign(score[i] if score[i] else raw[i])*e2.motion_weight*np.clip(x[i,14]/10.0,0,1)
            if np.isfinite(x[i,13]):
                speed=float(x[i,13])
                if speed >= e2.discontinuity_speed_mps: score[i] *= .25
                else: score[i] *= 1.0-.25*np.clip(speed/e2.ball_speed_scale_mps,0,1)
        elif i: score[i]=score[i-1]*e2.missing_decay
    dead,bias=_event_dead_frames(smart_tagging_events,n,config)
    out[:,0]=np.exp(score+bias); out[:,1]=np.exp(-score-bias); out[:,2]=np.exp(dead)
    empty=(x[:,3]+x[:,4])<=0; out[empty,2]*=np.exp(config.empty_team_dead_evidence)
    out[~visible & (dead<=0),2]=1.0
    out/=out.sum(axis=1,keepdims=True)
    return out,known


def infer_labels_v1_e2(features, *, video_frames=None, smart_tagging_events=None, config=InferenceConfig(), e2=E2Config()):
    """Controlled V1+E2 comparison: V1 supplies the fixed D/X mask."""
    x=np.asarray(features)
    if len(x)!=N_FRAMES: raise ValueError(f"expected {N_FRAMES} feature rows, received {len(x)}")
    e1,known=generate_emissions(x,smart_tagging_events=smart_tagging_events,config=config)
    v1=decode_full_match(e1,known=known,costs=config.transition_costs)
    emissions,e2known=generate_emissions_e2(x,smart_tagging_events=smart_tagging_events,config=config,e2=e2)
    labels=decode_full_match_v3b(emissions,np.ones(N_FRAMES),v1,known=e2known,costs=config.transition_costs,uncertainty_penalty=0.0)
    return labels,emissions,e2known


def infer_labels_v3b_e2(features, *, video_frames=None, smart_tagging_events=None, config=InferenceConfig(), reliability=ReliabilityV3Config(), e2=E2Config()):
    """V3B+E2: fixed V1 D/X mask and reliability-aware live H/A decoding."""
    x=np.asarray(features)
    if len(x)!=N_FRAMES: raise ValueError(f"expected {N_FRAMES} feature rows, received {len(x)}")
    e1,known=generate_emissions(x,smart_tagging_events=smart_tagging_events,config=config)
    v1=decode_full_match(e1,known=known,costs=config.transition_costs)
    emissions,e2known=generate_emissions_e2(x,smart_tagging_events=smart_tagging_events,config=config,e2=e2)
    confidence=build_reliability(x,config=reliability)
    labels=decode_full_match_v3b(emissions,confidence,v1,known=e2known,costs=config.transition_costs,uncertainty_penalty=reliability.uncertainty_penalty)
    return labels,emissions,confidence
