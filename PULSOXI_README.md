# PulsoXI Possession Engine Ã¢â‚¬â€ Current Frozen Checkpoint

This repository contains the current PulsoXI source checkpoint for the possession engine. It preserves the official organiser framework while adding the frozen PulsoXI inference pipeline, V3B+E2 preferred candidate, V1 fallback, tests, diagnostics, and reproducibility documentation. Licensed match assets are not included.

## Start here Ã¢â‚¬â€ current PulsoXI repository

```bash
git clone https://github.com/musoye/pulsoxi-possession-engine.git
cd pulsoxi-possession-engine
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-pulsoxi.txt
python -m pip check
python -m unittest discover -s tests -q
python scripts/preflight.py
```

Do not clone the organiser repository for the current PulsoXI workflow. Use the PulsoXI repository above. The organiser files are already preserved in this repository.

The authorised dataset should be placed only in the local ignored `data/tactical/` directory described below. Do not replace the official scorer or upload the dataset.

## What this checkpoint implements now

- Standalone Viterbi temporal decoder for **H / A / D** emission probabilities, with missing-evidence handling and separate inference per half.
- Deterministic, 199,319-frame submission writer with **X** outside published graded half boundaries.
- A preflight helper calling the official public evaluator and checking dataset file presence.
- Audited tracking feature extraction using official team metadata.\n- E1 and football-aware E2 emissions, with V3B reliability-aware live H/A decoding and fixed V1 D/X masking.\n- Preferred `v3be2` inference command, V1 fallback, synthetic tests, diagnostics, and a technical-report outline.

## Current limitations

- No private full-match ground truth or competition accuracy claim is included.
- The full licensed dataset is required to run local inference.
- V3B+E2 is the preferred frozen candidate; V1 remains the fallback.

## Competition constraints

The submitted solution must be algorithmically generated and exactly reproducible. The entire public repository must be safe from restricted match assets and unshareable derivatives. The competition's official `RULES.md`, `DATA.md`, `EVALUATION.md`, `SUBMISSION.md`, and scoring source take precedence over this kit.

Official repository: https://github.com/metrica-sports/more-than-a-hack-2027

Team: **PulsoXI**. Submission deadline: **2026-11-17 23:59 CET** (23:59 WAT).

## Phase 1A tracking features

`pulsoxi.tracking_features.extract_tracking_features` consumes normalized ATD-like arrays and an explicit metadata-derived slot/team map. It returns a deterministic `(frames, features)` NumPy matrix. Features include ball visibility and metric coordinates, team counts and centroids, team-wise minimum ball distances, their difference, reliable ball displacement/speed, same-slot player-ball relative speed, and consecutive missing-ball gap length.

ATD frame IDs are converted with `video_frame = frame_id - 1`. Derivatives require consecutive reliable observations and are reset across gaps and the official blocks returned by `games.half_video_blocks("BAR-ATH")`. Missing coordinates remain NaN in geometric features; counts, visibility, and gap length remain explicit numeric indicators. Track slots are never interpreted as persistent player identities.

The real licensed ATD integration remains untested until the gated files are available locally.

### Local licensed-data audit

After placing authorised files under `data/tactical/` (or setting `MTAH_DATA_DIR`), run locally:

```powershell
python scripts/audit_atd.py --out outputs/local_audit/atd_audit.json
```

The command fails if the gated files are absent. It writes only a concise JSON audit under the ignored `outputs/` directory and does not print raw tracking rows or coordinates.

## Phase 2B first inference candidate

`pulsoxi.inference` converts tracking features into interpretable H/A/D evidence. Relative team evidence is based on the difference between away and home minimum player-ball distances; reliable relative motion adds a small directional bonus. Missing-ball frames carry forward prior team evidence with decay and are not classified as dead solely because the ball is missing. Smart Tagging `SET PIECES` events provide bounded dead-ball support around event boundaries, while team tags provide only a small auxiliary bias. The existing configurable Viterbi decoder processes each official half independently and emits `X` outside graded periods.

Run only on an authorized local machine containing the gated dataset:

```powershell
python scripts/run_inference.py --out outputs/local_inference/submission.csv --diagnostics outputs/local_inference/diagnostics.json
python evaluation/validate_submission.py outputs/local_inference/submission.csv
```

This is an explainable initial candidate, not a trained model or a competition score. Parameters are engineering defaults and must not be tuned against the five-minute public ground truth.

## Candidate V2: evidence-based transitions

Candidate V1 remains the established fallback and default CLI candidate. The preferred experimental candidate is V3B+E2 (`v3be2`); it is not claimed to be more accurate without private ground truth. Candidate V2 reuses V1 H/A/D evidence but decodes H/A changes with accumulated log-likelihood evidence. A challenger must exceed a margin and accumulate evidence across frames; a sufficiently strong reliable margin may switch immediately. Missing-ball frames decay challenger evidence and hold the current state. Smart Tagging dead-ball evidence remains separate from H/A switching.

Generate candidates locally into separate ignored paths:

```powershell
python scripts/run_inference.py --candidate v1 --out outputs/local_inference/submission_v1.csv --diagnostics outputs/local_inference/diagnostics_v1.json
python scripts/run_inference.py --candidate v2 --out outputs/local_inference/submission_v2.csv --diagnostics outputs/local_inference/diagnostics_v2.json
python evaluation/validate_submission.py outputs/local_inference/submission_v1.csv
python evaluation/validate_submission.py outputs/local_inference/submission_v2.csv
```

V2 parameters are engineering defaults, not tuned on the public ground truth. Fewer predicted possessions would not by itself establish improvement.

## Candidate V3: reliability-aware joint decoding

Candidate V3 preserves the joint H/A/D Viterbi structure and V1 dead-ball transition costs. It varies only the H-to-A/A-to-H transition cost by bounded confidence: `opponent_cost(t) = base_opponent_cost + uncertainty_penalty * (1 - reliability_t)`. Reliability combines ball visibility, clipped team-distance margin, and the existing reliable relative-motion indicator. Missing or non-finite confidence becomes zero and never creates evidence. V3 is selectable with `--candidate v3`; V1 remains the default.

## Phase 3A E2 emissions

E2 is an optional football-aware emission model and does not alter E1. Its signed control score combines normalized nearest-player distance difference with a causal five-frame proximity mean (35% persistence weight). Valid nearest-player relative speed contributes up to 0.20, while ball speed attenuates evidence and speeds at or above 25 m/s are treated as discontinuities. Missing or invalid observations carry only a 0.97-decayed prior score and never create new evidence. E2 experiments use the V1-derived D/X mask so E1/E2 comparisons do not silently change dead-ball decisions.

Run locally with `--candidate v1e2` or `--candidate v3be2`; generated outputs belong under ignored `outputs/local_inference/`.

## Reproducing the frozen candidates

Use Python 3.11 where available and the existing virtual environment. From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-starter.txt
```

Place the authorised competition files under the local `data/tactical/` directory; they are intentionally excluded from Git. Generate the preferred candidate and validate it:

```powershell
python scripts/run_inference.py --candidate v3be2 --out outputs/local_inference/submission_v3be2.csv --diagnostics outputs/local_inference/diagnostics_v3be2.json
python evaluation/validate_submission.py outputs/local_inference/submission_v3be2.csv
```

The fallback is `--candidate v1`. A successful run must produce 199,319 rows with frames 0 through 199,318. Generated submissions and diagnostics remain local under ignored `outputs/`. No private full-match ground truth or competition accuracy claim is included in this repository.

The frozen runtime dependency set is recorded in `requirements-pulsoxi.txt`. `requirements-starter.txt` remains available for the organiser starter notebooks, but the PulsoXI inference command should use the frozen file.

## Complete setup and run guide

This section describes the current frozen PulsoXI workflow from a fresh clone to a validated submission. It assumes that the user already has the authorised competition data on their local drive.

### 1. Clone the repository

```powershell
git clone https://github.com/musoye/pulsoxi-possession-engine.git
cd pulsoxi-possession-engine
git checkout main
```

Do not place licensed tracking files in the Git repository history. The repository contains the code and public organiser files only.

### 2. Connect an existing local dataset

The inference loader looks for the competition assets through the repository's `data` directory. If the authorised dataset already exists elsewhere on the same computer, copy or link only the required files into the local, ignored data directory.

Expected local layout:

```text
pulsoxi-possession-engine/
ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚ÂÃƒÆ’Ã¢â‚¬Â¦ÃƒÂ¢Ã¢â€šÂ¬Ã…â€œÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ data/
ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡   ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚ÂÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ tactical/
ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡       ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚ÂÃƒÆ’Ã¢â‚¬Â¦ÃƒÂ¢Ã¢â€šÂ¬Ã…â€œÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ BAR-ATH_tactical_atd.xml
ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡       ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚ÂÃƒÆ’Ã¢â‚¬Â¦ÃƒÂ¢Ã¢â€šÂ¬Ã…â€œÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ BAR-ATH_tactical_atd.txt
ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡       ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚ÂÃƒÆ’Ã¢â‚¬Â¦ÃƒÂ¢Ã¢â€šÂ¬Ã…â€œÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ BAR-ATH_tactical_smart_tagging.xml
ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡       ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚ÂÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ BAR-ATH_tactical_homography.csv
```

For example, if the authorised files are already in `D:\PulsoXI\licensed_data\tactical`, use PowerShell:

```powershell
New-Item -ItemType Directory -Force data\tactical
Copy-Item D:\PulsoXI\licensed_data\tactical\BAR-ATH_tactical_atd.xml data\tactical\
Copy-Item D:\PulsoXI\licensed_data\tactical\BAR-ATH_tactical_atd.txt data\tactical\
Copy-Item D:\PulsoXI\licensed_data\tactical\BAR-ATH_tactical_smart_tagging.xml data\tactical\
Copy-Item D:\PulsoXI\licensed_data\tactical\BAR-ATH_tactical_homography.csv data\tactical\
```

Alternatively, when permitted by the local operating system and dataset licence, create a directory junction or symbolic link to the authorised folder. Do not use a public cloud folder, public link, or third-party upload service.

The files remain local because `.gitignore` excludes `data/*`. Never run `git add data`, and never send these files to hosted services.

Check that the files exist without printing their contents:

```powershell
Get-ChildItem data\tactical -File | Select-Object Name,Length
```

The official file requirements are defined by `DATA.md`, `games.py`, and the organiser's current instructions. Do not rename files or invent substitute data files.

### 3. Create and install the environment

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-pulsoxi.txt
python -m pip check
```

The validated environment used Python 3.12.1. Use the existing `.venv` when it is already configured.

### 4. Run the current preferred method

The frozen preferred candidate is V3B+E2. It reads the local ATD and Smart Tagging files, extracts tracking features, creates E2 emissions, applies the V3B decoder, and writes a submission CSV under the ignored `outputs` directory.

```powershell
python scripts/run_inference.py `
  --candidate v3be2 `
  --out outputs\local_inference\submission_v3be2.csv `
  --diagnostics outputs\local_inference\diagnostics_v3be2.json
```

The command does not modify the licensed source files. It creates only local generated outputs.

### 5. Validate the generated submission

```powershell
python evaluation\validate_submission.py `
  outputs\local_inference\submission_v3be2.csv
```

A valid result must contain:

- 199,319 rows
- Frame indices `0` through `199318`
- Only `H`, `A`, `D`, and `X` labels
- No `X` predictions inside graded halves

### 6. Generate and validate the fallback

V1 is the established fallback and can be generated independently:

```powershell
python scripts/run_inference.py `
  --candidate v1 `
  --out outputs\local_inference\submission_v1.csv `
  --diagnostics outputs\local_inference\diagnostics_v1.json

python evaluation\validate_submission.py `
  outputs\local_inference\submission_v1.csv
```

Use V1 only as a fallback or comparison candidate. Do not select a candidate based only on possession count or smoother-looking output.

### 7. Safe handoff for submission

Before uploading through the official competition portal:

```powershell
Get-FileHash outputs\local_inference\submission_v3be2.csv -Algorithm SHA256
```

Record the candidate name, commit hash, Python version, dependency file, validation result, and submission SHA-256. Upload only the validated submission CSV requested by the organiser. Do not upload the tracking files, Smart Tagging files, videos, diagnostics, or source data.

### 8. Troubleshooting

- If the loader reports missing files, check `data\tactical` and compare filenames with `DATA.md`.
- If `kloppy` or another dependency is missing, activate `.venv` and reinstall `requirements-pulsoxi.txt`.
- If validation reports an incorrect row count or frame range, stop and investigate; do not manually edit the CSV.
- If licensed data cannot be copied into the repository workspace, use the supported `--data-dir` option with an authorised local data root:

```powershell
python scripts/run_inference.py `
  --data-dir D:\PulsoXI\licensed_data `
  --candidate v3be2 `
  --out outputs\local_inference\submission_v3be2.csv
```

The data directory must remain local and must not be a public or shared upload location.

## Linux/macOS terminal commands

Use these Bash commands instead of the PowerShell examples above:

```bash
git clone https://github.com/musoye/pulsoxi-possession-engine.git
cd pulsoxi-possession-engine

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-pulsoxi.txt
python -m pip check
```

If the authorised data is already stored at `/mnt/pulsoxi/licensed_data`:

```bash
mkdir -p data/tactical
cp /mnt/pulsoxi/licensed_data/tactical/BAR-ATH_tactical_atd.xml data/tactical/
cp /mnt/pulsoxi/licensed_data/tactical/BAR-ATH_tactical_atd.txt data/tactical/
cp /mnt/pulsoxi/licensed_data/tactical/BAR-ATH_tactical_smart_tagging.xml data/tactical/
cp /mnt/pulsoxi/licensed_data/tactical/BAR-ATH_tactical_homography.csv data/tactical/
```

Or keep the dataset where it is and pass its path directly:

```bash
python scripts/run_inference.py \
  --data-dir /mnt/pulsoxi/licensed_data \
  --candidate v3be2 \
  --out outputs/local_inference/submission_v3be2.csv \
  --diagnostics outputs/local_inference/diagnostics_v3be2.json
```

Validate the generated submission:

```bash
python evaluation/validate_submission.py \
  outputs/local_inference/submission_v3be2.csv
```

Expected output includes:

```text
199319 rows, frame 0..199318
Valid
```

Do not commit or upload the licensed data, videos, generated diagnostics, or local prediction files.
