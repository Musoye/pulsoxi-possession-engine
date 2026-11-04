"""Cross-platform PulsoXI verification and local reproduction interface."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ROWS = 199_319
FINAL_SHA256 = "83347409855d6d89c8ef30d500cf43de107c440187b25c227d0c23ca40e5115f"
V3_SHA256 = "46e9dc60762d4812d5ae6d7b23d71e54b48fb73c514b0e98349c80f9c1511bfe"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def csv_audit(path: Path) -> dict:
    result = {"path": str(path), "exists": path.is_file(), "valid": False}
    if not path.is_file():
        return result
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        frames = [int(row["frame"]) for row in rows]
        labels = [row["label"].strip().upper() for row in rows]
        result.update({
            "rows": len(rows),
            "header": list(rows[0]) if rows else [],
            "frame_range": [frames[0], frames[-1]] if frames else None,
            "labels": sorted(set(labels)),
            "valid": (len(rows) == ROWS and frames == list(range(ROWS))
                      and set(labels) <= {"H", "A", "D", "X"}
                      and all(labels)),
            "sha256": sha256(path),
        })
    except (OSError, ValueError, KeyError, UnicodeError) as exc:
        result["error"] = str(exc)
    return result


def validator(path: Path) -> bool:
    command = [sys.executable, str(ROOT / "evaluation" / "validate_submission.py"), str(path)]
    completed = subprocess.run(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (ROOT / "outputs" / "local_reproduction").mkdir(parents=True, exist_ok=True)
    if completed.stdout:
        print(completed.stdout.rstrip())
    return completed.returncode == 0


def verify(paths: dict[str, Path], run_validators: bool = True, compare_refs: bool = True) -> tuple[dict, bool]:
    report = {"mode": "verify", "artifacts": {}, "checks": {}}
    for name, path in paths.items():
        report["artifacts"][name] = csv_audit(path)
    report["checks"]["v3_hash"] = not compare_refs or report["artifacts"]["preserved_v3"].get("sha256") == V3_SHA256
    report["checks"]["final_hash"] = report["artifacts"]["final"].get("sha256") == FINAL_SHA256 if compare_refs else report["artifacts"]["final"].get("valid", False)
    report["checks"]["v3_reproduced_equal"] = (not compare_refs or (paths["preserved_v3"].is_file() and paths["preserved_v3"].read_bytes() == paths["regenerated_v3"].read_bytes()))
    report["checks"]["v4_reproduced_equal"] = (not compare_refs or (paths["preserved_v4"].is_file() and paths["preserved_v4"].read_bytes() == paths["regenerated_v4"].read_bytes()))
    required = ("preserved_v3", "regenerated_v3", "preserved_v4", "regenerated_v4", "final") if compare_refs else ("regenerated_v3", "regenerated_v4", "final")
    report["checks"]["all_structurally_valid"] = all(report["artifacts"][name].get("valid", False) for name in required)
    v3_name = "preserved_v3" if compare_refs else "regenerated_v3"
    v4_name = "preserved_v4" if compare_refs else "regenerated_v4"
    v3 = report["artifacts"][v3_name]
    v4 = report["artifacts"][v4_name]
    report["checks"]["dx_masks_equal"] = False
    if v3.get("valid") and v4.get("valid"):
        a = list(csv.DictReader(paths[v3_name].open(newline="", encoding="utf-8")))
        b = list(csv.DictReader(paths[v4_name].open(newline="", encoding="utf-8")))
        report["checks"]["dx_masks_equal"] = all(x["label"] == y["label"] or (x["label"] not in "DX" and y["label"] not in "DX") for x, y in zip(a, b))
    if run_validators:
        report["checks"]["final_validator"] = validator(paths["final"])
    ok = all(report["checks"].values())
    return report, ok


def full_reproduction(data_dir: Path, output_root: Path) -> tuple[dict, bool]:
    if not data_dir.is_dir():
        raise FileNotFoundError(f"data directory does not exist: {data_dir}")
    if not any(data_dir.rglob("*atd*.xml")):
        raise FileNotFoundError("no local ATD metadata/XML asset found under data directory")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_root / stamp
    run_dir.mkdir(parents=True, exist_ok=False)
    v3 = run_dir / "submission_v3be2.csv"
    v4 = run_dir / "submission_v4_15s.csv"
    logs = run_dir / "inference.log"
    with logs.open("w", encoding="utf-8") as log:
        command = [sys.executable, str(ROOT / "scripts" / "run_inference.py"), "--data-dir", str(data_dir), "--candidate", "v3be2", "--out", str(v3), "--diagnostics", str(run_dir / "diagnostics_v3be2.json")]
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        return_code = process.wait()
    if return_code != 0 or not v3.is_file():
        return {"mode": "reproduce", "run_dir": str(run_dir), "exit_code": return_code, "status": "FAIL", "log": str(logs)}, False
    command = [sys.executable, str(ROOT / "scripts" / "apply_turnover_stabilization.py"), "--input", str(v3), "--output", str(v4), "--seconds", "1.5"]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    (run_dir / "stabilization.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode != 0 or not v4.is_file():
        return {"mode": "reproduce", "run_dir": str(run_dir), "exit_code": result.returncode, "status": "FAIL"}, False
    paths = {"preserved_v3": ROOT / "outputs/local_inference/submission_v3be2.csv", "regenerated_v3": v3, "preserved_v4": ROOT / "outputs/local_inference/submission_v4_15s.csv", "regenerated_v4": v4, "final": v4}
    report, ok = verify(paths, compare_refs=all(paths[name].is_file() for name in ("preserved_v3", "preserved_v4")))
    report["mode"] = "reproduce"
    report["run_dir"] = str(run_dir)
    return report, ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify or reproduce PulsoXI frozen candidates")
    parser.add_argument("--reproduce", action="store_true", help="run fresh V3B+E2 inference and V4 postprocessing")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/local_reproduction")
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()
    if args.reproduce:
        report, ok = full_reproduction(args.data_dir.resolve(), args.output_root.resolve())
    else:
        base = ROOT / "outputs"
        run = base / "local_reproduction/20261102_171145"
        paths = {"preserved_v3": base / "local_inference/submission_v3be2.csv", "regenerated_v3": run / "submission_v3be2.csv", "preserved_v4": base / "local_inference/submission_v4_15s.csv", "regenerated_v4": run / "submission_v4_15s.csv", "final": base / "local_inference/PulsoXI.csv"}
        report, ok = verify(paths)
    destination = args.report or (ROOT / "outputs/local_reproduction/verification_report.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("PASS" if ok else "FAIL", "verification report:", destination)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
