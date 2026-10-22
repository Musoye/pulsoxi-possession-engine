"""Time-aware decoding of per-frame team-possession likelihoods.

States are H, A, D in that order. This module is independent of the
organiser's licensed data; a separate feature/model stage must estimate the
per-frame class probabilities. Cost values are engineering placeholders,
NOT measured or tuned competition hyperparameters.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import numpy as np

STATES = ("H", "A", "D")
# Video-frame inclusive start/end pairs, from the official DATA.md.
HALVES = ((11185, 90651), (110808, 190197))
N_FRAMES = 199319


@dataclass(frozen=True)
class TransitionCosts:
    opponent: float = 3.0      # H <-> A
    enter_dead: float = 2.0    # H/A -> D
    leave_dead: float = 2.0    # D -> H/A


def transition_log_scores(config: TransitionCosts = TransitionCosts()) -> np.ndarray:
    if min(config.opponent, config.enter_dead, config.leave_dead) < 0:
        raise ValueError("transition costs must be nonnegative")
    costs = np.array([
        [0., config.opponent, config.enter_dead],
        [config.opponent, 0., config.enter_dead],
        [config.leave_dead, config.leave_dead, 0.],
    ], dtype=np.float64)
    return -costs


def viterbi(probs: np.ndarray, *, known: np.ndarray | None = None,
            costs: TransitionCosts = TransitionCosts()) -> np.ndarray:
    """Decode one contiguous period. Returns integer codes 0=H,1=A,2=D.

    probs: (frames, 3) uncalibrated or calibrated nonnegative class evidence.
    known: optional boolean mask; False = no reliable evidence at this frame.
           Missing observations get neutral emissions and do not force switches.
    A separate call for each half prevents spurious transitions across halftime.
    """
    x = np.asarray(probs, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 3:
        raise ValueError("expected probability array of shape (n_frames, 3)")
    n = len(x)
    if not n:
        return np.empty(0, dtype=np.uint8)
    if np.any(~np.isfinite(x)) or np.any(x < 0):
        raise ValueError("probabilities must be finite and nonnegative")
    sums = x.sum(axis=1, keepdims=True)
    if np.any(sums <= 0):
        raise ValueError("at least one class must have positive evidence per frame")
    x = x / sums
    emission = np.log(np.maximum(x, 1e-12))
    if known is not None:
        mask = np.asarray(known, dtype=bool)
        if mask.shape != (n,):
            raise ValueError("known mask must have shape (n_frames,)")
        emission[~mask] = 0.0
    transition = transition_log_scores(costs)
    backptr = np.zeros((n, 3), dtype=np.uint8)
    scores = emission[0].copy()
    for t in range(1, n):
        choices = scores[:, None] + transition
        backptr[t] = np.argmax(choices, axis=0).astype(np.uint8)
        scores = emission[t] + choices[backptr[t], np.arange(3)]
    result = np.empty(n, dtype=np.uint8)
    result[-1] = np.argmax(scores)
    for t in range(n - 1, 0, -1):
        result[t - 1] = backptr[t, result[t]]
    return result


def decode_full_match(probs: np.ndarray, *, known: np.ndarray | None = None,
                      costs: TransitionCosts = TransitionCosts()) -> np.ndarray:
    """Return one string label per tactical frame, X outside graded half periods."""
    probs = np.asarray(probs)
    if probs.shape != (N_FRAMES, 3):
        raise ValueError(f"expected ({N_FRAMES}, 3) probabilities, got {probs.shape}")
    mask = None if known is None else np.asarray(known)
    if mask is not None and mask.shape != (N_FRAMES,):
        raise ValueError("known must be a frame-length boolean array")
    labels = np.full(N_FRAMES, "X", dtype="<U1")
    state_arr = np.array(STATES)
    for first, last in HALVES:
        loc = slice(first, last + 1)
        decoded = viterbi(probs[loc], known=None if mask is None else mask[loc], costs=costs)
        labels[loc] = state_arr[decoded]
    return labels
