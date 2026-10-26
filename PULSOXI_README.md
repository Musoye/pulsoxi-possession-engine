# PulsoXI Possession Engine â€” Phase 0 Starter Kit

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
