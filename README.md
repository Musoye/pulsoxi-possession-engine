# PulsoXI Possession Engine

PulsoXI is our reproducible possession-control submission for FC Barcelona More Than a Hack 2027. The frozen selected candidate is V3B+E2 with deterministic 1.5-second turnover stabilization; unstabilized V3B+E2 is the fallback.

Team: PulsoXI

Team members: AbdulShakur Bashir, Abraham Folorunso, Abdulrasheed Abdulsalam

Use Python as the primary interface:

```bash
python reproduce_pulsoxi.py
python reproduce_pulsoxi.py --reproduce --data-dir /path/to/licensed_data
```

The first command verifies existing local artifacts without rerunning inference. The second loads the authorised local ATD/Smart Tagging assets, generates V3B+E2, applies the existing stabilizer, validates the output, and writes a machine-readable report. On Windows, use `.venv\Scripts\python.exe`; on Linux/macOS, use `.venv/bin/python`.

Install pinned dependencies with `python -m pip install -r requirements-pulsoxi.txt`. Validate a submission with `python evaluation/validate_submission.py <submission.csv>`.

For a complete local reproduction, place the authorised ATD and Smart Tagging bundle in a private directory and run `python reproduce_pulsoxi.py --reproduce --data-dir <licensed-data-directory>`. This generates fresh V3B+E2 and V4 outputs and writes a JSON verification report. The default `python reproduce_pulsoxi.py` only verifies existing local artifacts.

The licensed match data is not included and must remain private. See [PULSOXI_REPRODUCTION.md](PULSOXI_REPRODUCTION.md) for data layout, security rules, and exact reproduction details, and [PulsoXI.md](PulsoXI.md) for the technical report.
