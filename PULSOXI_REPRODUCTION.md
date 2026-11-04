# PulsoXI Reproduction Guide

PulsoXI is a possession-control submission for the FC Barcelona More Than a Hack 2027 challenge. The selected frozen candidate is V3B+E2 with deterministic 1.5-second turnover stabilization; unstabilized V3B+E2 is the fallback.

## Install

Clone the PulsoXI repository, use Python 3.12, create `.venv`, and install the pinned runtime:

```bash
git clone https://github.com/musoye/pulsoxi-possession-engine.git
cd pulsoxi-possession-engine
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-pulsoxi.txt
```

On Windows PowerShell use `py -3.12 -m venv .venv` and `.\.venv\Scripts\python.exe`.

## Licensed data

Place the authorised, non-redistributable bundle in a private local directory containing the tactical ATD metadata/raw files, Smart Tagging input, and homography. Do not commit or upload these files. The pipeline reads them through `--data-dir` and does not require a private prediction cache.

## Reproduce and validate

The primary interface is Python. Verify the existing local artifacts without running raw inference:

```bash
python reproduce_pulsoxi.py
```

For a fresh reproduction from authorised local assets:

```bash
python reproduce_pulsoxi.py --reproduce --data-dir data
```

The verifier writes `outputs/local_reproduction/verification_report.json`, checks actual hashes and byte equality, validates CSV structure, runs the official validator, and confirms the D/X mask is unchanged. It never silently creates replacement predictions in default mode.

From a persistent PowerShell terminal:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\reproduce_pulsoxi.ps1 -DataDir D:\PulsoXI\licensed_data
```

Replace the path with the actual authorised data directory. The runner creates timestamped ignored outputs, prevents duplicate Python workers, validates V3 and V4, and compares SHA-256 values. The selected output must be `frame,label`, exactly 199,319 rows, frames `0..199318`, and labels only `H`, `A`, `D`, or `X`.

Direct commands are:

```bash
python scripts/run_inference.py --data-dir /path/to/licensed_data --candidate v3be2 --out outputs/local_inference/submission_v3be2.csv
python scripts/apply_turnover_stabilization.py --input outputs/local_inference/submission_v3be2.csv --output outputs/local_inference/PulsoXI.csv --seconds 1.5
python evaluation/validate_submission.py outputs/local_inference/PulsoXI.csv
```

The verified selected-candidate SHA-256 is `83347409855d6d89c8ef30d500cf43de107c440187b25c227d0c23ca40e5115f`. Generated CSVs, diagnostics, external datasets, credentials, and private ground truth are ignored by Git.

## Architecture and limitations

Official metadata supplies team membership. Tracking features convert normalized coordinates into metres and derive ball visibility, player availability, team centroids, nearest-player distances, motion, and missing-ball gaps. E2 combines distance-based control, causal proximity persistence, reliable relative motion, ball-speed attenuation, and missing-evidence decay. V3B performs reliability-aware temporal decoding independently by half. Stabilization merges only complete short H-A-H/A-H-A reversals within a half and never changes D/X.

The public implementation explicitly protects half boundaries and nonconsecutive frame IDs. It does not expose a separate ATD tracking-validity mask to the optional stabilizer; this remains a documented limitation. See `PulsoXI_TECHNICAL_REPORT.md` and the official submission specification for deliverables and restrictions.
