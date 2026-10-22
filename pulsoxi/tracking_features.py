"""Audited, licence-safe football tracking features."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping, Sequence
import numpy as np
from games import GAME, half_video_blocks

FEATURE_NAMES = (
    "ball_visible", "ball_x_m", "ball_y_m", "home_player_count", "away_player_count",
    "home_centroid_x_m", "home_centroid_y_m", "away_centroid_x_m", "away_centroid_y_m",
    "home_min_ball_distance_m", "away_min_ball_distance_m", "nearest_distance_diff_m",
    "ball_displacement_m", "ball_speed_mps", "nearest_player_relative_speed_mps",
    "missing_ball_gap_frames", "nearest_relative_motion_reliable",
)

@dataclass(frozen=True)
class TrackingFeatureConfig:
    pitch_length_m: float = 105.3
    pitch_width_m: float = 68.0
    fps: float = 25.0
    max_track_speed_mps: float = 12.5

def _gap_lengths(visible: np.ndarray) -> np.ndarray:
    out = np.zeros(len(visible), dtype=np.int32)
    gap = 0
    for i, present in enumerate(visible):
        gap = 0 if present else gap + 1
        out[i] = gap
    return out

def extract_tracking_features(ball_xy: np.ndarray, player_xy: np.ndarray, team_of_slot: Sequence[int], *, config: TrackingFeatureConfig = TrackingFeatureConfig(), video_frames: Sequence[int] | None = None) -> tuple[np.ndarray, tuple[str, ...]]:
    """Extract deterministic features from normalized ATD coordinates.

    ball_xy is (frames, 2), player_xy is (frames, slots, 2), and missing values are NaN.
    team_of_slot comes from ATD metadata: 0 home, 1 away; slots are not player identities.
    """
    ball = np.asarray(ball_xy, dtype=np.float64)
    players = np.asarray(player_xy, dtype=np.float64)
    teams = np.asarray(team_of_slot)
    if ball.ndim != 2 or ball.shape[1] != 2:
        raise ValueError("ball_xy must have shape (frames, 2)")
    if players.ndim != 3 or players.shape[0] != len(ball) or players.shape[2] != 2:
        raise ValueError("player_xy must have shape (frames, slots, 2)")
    if players.shape[1] != len(teams) or not np.all(np.isin(teams, (0, 1))):
        raise ValueError("team_of_slot must contain one 0/1 entry per slot")
    if video_frames is not None and len(video_frames) != len(ball):
        raise ValueError("video_frames must have one value per frame")
    if config.pitch_length_m <= 0 or config.pitch_width_m <= 0 or config.fps <= 0:
        raise ValueError("pitch dimensions and fps must be positive")
    n = len(ball)
    scale = np.array([config.pitch_length_m, config.pitch_width_m])
    ball_m, players_m = ball * scale, players * scale
    ball_ok = np.all(np.isfinite(ball), axis=1)
    player_ok = np.all(np.isfinite(players), axis=2)
    out = np.full((n, len(FEATURE_NAMES)), np.nan, dtype=np.float64)
    out[:, 0] = ball_ok
    out[ball_ok, 1:3] = ball_m[ball_ok]
    for team, count_col, centroid_cols, distance_col in ((0, 3, (5, 6), 9), (1, 4, (7, 8), 10)):
        present = player_ok[:, teams == team]
        points = players_m[:, teams == team]
        counts = present.sum(axis=1)
        out[:, count_col] = counts
        for axis, col in enumerate(centroid_cols):
            sums = np.where(present, points[:, :, axis], 0.0).sum(axis=1)
            out[:, col] = np.divide(sums, counts, out=np.full(n, np.nan), where=counts > 0)
        if points.shape[1]:
            distances = np.linalg.norm(points - ball_m[:, None, :], axis=2)
            distances[~present | ~ball_ok[:, None]] = np.inf
            nearest_distances = np.min(distances, axis=1)
            out[:, distance_col] = np.where(np.isfinite(nearest_distances), nearest_distances, np.nan)
    out[:, 11] = out[:, 9] - out[:, 10]
    out[:, 15] = _gap_lengths(ball_ok)
    out[:, 16] = 0.0
    same_period = np.ones(n, dtype=bool)
    if video_frames is not None:
        frames = np.asarray(video_frames, dtype=np.int64)
        same_period[1:] = frames[1:] == frames[:-1] + 1
        blocks = [(first, last) for _, first, last in half_video_blocks(GAME)]
        same_period &= np.array([any(first <= f <= last for first, last in blocks) for f in frames])
    consecutive_ball = ball_ok & np.r_[False, ball_ok[:-1]] & same_period
    if n:
        delta = np.linalg.norm(ball_m - np.roll(ball_m, 1, axis=0), axis=1)
        out[consecutive_ball, 12] = delta[consecutive_ball]
        out[consecutive_ball, 13] = delta[consecutive_ball] * config.fps
        for i in np.flatnonzero(consecutive_ball):
            valid = player_ok[i] & player_ok[i - 1]
            if not np.any(valid):
                continue
            distances = np.linalg.norm(players_m[i] - ball_m[i], axis=1)
            distances[~valid] = np.nan
            nearest = int(np.nanargmin(distances))
            relative = ((players_m[i, nearest] - ball_m[i]) - (players_m[i - 1, nearest] - ball_m[i - 1]))
            player_displacement_m = np.linalg.norm(players_m[i, nearest] - players_m[i - 1, nearest])
            observed_speed_mps = player_displacement_m * config.fps
            if observed_speed_mps > config.max_track_speed_mps:
                continue
            out[i, 14] = np.linalg.norm(relative) * config.fps
            out[i, 16] = 1.0
    return out, FEATURE_NAMES

def video_frame_from_atd(frame_id: Sequence[int]) -> np.ndarray:
    """Convert official one-based ATD frame IDs to zero-based tactical frames."""
    return np.asarray(frame_id, dtype=np.int64) - 1

def metadata_team_map(team_ids: Mapping[int, str], home_id: str) -> np.ndarray:
    """Convert metadata slot-team IDs to 0/1 without treating slots as player IDs."""
    if not team_ids:
        raise ValueError("team metadata is empty")
    return np.asarray([0 if team_id == home_id else 1 for _, team_id in sorted(team_ids.items())], dtype=np.int8)









