"""Local-only command for the authorised tactical ATD audit."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from games import GAME, require
from pulsoxi.atd_adapter import dataframe_arrays, load_tactical_features
from pulsoxi.data_audit import audit_arrays
from starter.load_atd import load

def main():
    parser = argparse.ArgumentParser(description="Audit local licensed PulsoXI tactical ATD")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--out", default="outputs/local_audit/atd_audit.json")
    args = parser.parse_args()
    paths = require(["atd_meta", "atd_raw"], GAME, args.data_dir, "tactical")
    features, _, metadata = load_tactical_features(GAME, args.data_dir)
    dataframe = load(GAME, args.data_dir, "tactical").to_df()
    ball, players, frame_ids = dataframe_arrays(dataframe, metadata["team_of_slot"])
    report = audit_arrays(frame_ids, ball, players, metadata["team_of_slot"], features)
    report["required_files"] = paths
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + chr(10), encoding="utf-8")
    print(f"Audit complete: {target}")
    print(f"frames={report['frame_count']} ball_visibility={report['ball_visibility_fraction']}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
