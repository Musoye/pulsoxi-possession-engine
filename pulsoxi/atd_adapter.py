"""Official kloppy ATD adapter for PulsoXI features.

This module only reads authorised local assets. It does not copy, serialize, or upload tracking data.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from games import GAME, GAMES, require, source_frames
from starter.load_atd import load as load_atd
from starter.baseline import team_of_each_track
from .tracking_features import extract_tracking_features, TrackingFeatureConfig, video_frame_from_atd


def _slot_columns(df, slots):
    columns = []
    for slot in range(1, slots + 1):
        pair = (f"{slot}_x", f"{slot}_y")
        if not all(c in df.columns for c in pair):
            raise ValueError(f"ATD dataframe is missing slot columns {pair}")
        columns.append(pair)
    return columns


def dataframe_arrays(df, team_of_slot):
    """Convert official ``TrackingDataset.to_df()`` output to extractor arrays."""
    required = {"frame_id", "ball_x", "ball_y"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"ATD dataframe missing required columns: {missing}")
    slots = len(team_of_slot)
    slot_columns = _slot_columns(df, slots)
    ball = df[["ball_x", "ball_y"]].to_numpy(dtype=float)
    players = np.stack([df[[x, y]].to_numpy(dtype=float) for x, y in slot_columns], axis=1)
    frame_ids = df["frame_id"].to_numpy(dtype=np.int64)
    if len(frame_ids) != len(df) or not np.array_equal(frame_ids, np.arange(1, len(df) + 1)):
        raise ValueError("ATD frame_id must be contiguous and start at 1")
    return ball, players, frame_ids


def load_tactical_features(game=GAME, data_dir=None, config=None):
    """Load local licensed tactical ATD through kloppy and return features plus safe metadata."""
    meta_path, _ = require(["atd_meta", "atd_raw"], game, data_dir, "tactical")
    dataset = load_atd(game, data_dir, "tactical")
    frame_ids = np.asarray([int(record.frame_id) for record in dataset.records], dtype=np.int64)
    if len(frame_ids) != source_frames("tactical", game) or not np.array_equal(frame_ids, np.arange(1, len(frame_ids) + 1)):
        raise ValueError("kloppy dataset does not match the official tactical timeline")
    team_of_slot = team_of_each_track(meta_path)
    ball, players, frame_ids = dataframe_arrays(dataset.to_df(), team_of_slot)
    cfg = config or TrackingFeatureConfig(*GAMES[game]["pitch"], GAMES[game]["fps"])
    video_frames = video_frame_from_atd(frame_ids)
    features, names = extract_tracking_features(ball, players, team_of_slot, config=cfg, video_frames=video_frames)
    return features, names, {"frame_ids": frame_ids, "video_frames": video_frames, "team_of_slot": team_of_slot}

