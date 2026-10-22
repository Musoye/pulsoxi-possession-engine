# PulsoXI Possession Engine — Phase 0 Starter Kit

This is a **public-code-only add-on** for the official Metrica Sports More Than a Hack 2027 repository. It does not contain licensed match assets, a trained model, or a scored PulsoXI submission.

## Start here (Windows PowerShell)

```powershell
git clone https://github.com/metrica-sports/more-than-a-hack-2027.git
cd more-than-a-hack-2027
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-starter.txt
python evaluation/run_evaluation.py
```

Extract **the contents of this kit** into your local `more-than-a-hack-2027/` folder. It creates `pulsoxi/`, `scripts/`, `tests/`, and a few documentation files. Do not replace or modify the official scorer. Keep the official README and `.gitignore` unchanged; **append** the contents of `PULSOXI_GITIGNORE_ADDITIONS.txt` to the official `.gitignore` before committing anything.

```powershell
python -m unittest discover -s tests -v
python scripts/preflight.py
```

The first command exercises **synthetic tests only**. The second executes the organiser's scorer on their published public samples and checks optional licensed data availability. Preflight results are not PulsoXI benchmark results.

After you receive the organiser's separate licensed dataset email:

1. Extract it locally to `data/` (do not upload it into GitHub, ChatGPT or third-party hosted LLM services).
2. Run `python scripts/preflight.py` again.
3. Run `python starter/baseline.py` to reconstruct the official reference predictions, then `python evaluation/run_evaluation.py`.
4. Develop a dedicated feature extraction stage with audited team/track-slot mapping; feed derived H/A/D probabilities into `pulsoxi.temporal.decode_full_match`.
5. Generate a compliant submission only after these probabilities are derived from an actual algorithm. See `pulsoxi/submission.py`.
6. Run `python evaluation/validate_submission.py your_output.csv` before submitting.

## What this kit implements now

- Standalone Viterbi temporal decoder for **H / A / D** emission probabilities, with missing-evidence handling and separate inference per half.
- Deterministic, 199,319-frame submission writer with **X** outside published graded half boundaries.
- A preflight helper calling the official public evaluator and checking dataset file presence.
- Synthetic unit tests and a report outline.

## What it does NOT implement yet

- Reading your licensed ATD files; feature extraction; mapping each track slot to a team; live/dead probability estimator; ball recovery; model training; quantitative comparison of PulsoXI vs published baselines.
- The transition penalties in `temporal.py` are **untuned engineering defaults**, not competition-optimal values. The 5-minute public labels cannot justify tuning them.

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
