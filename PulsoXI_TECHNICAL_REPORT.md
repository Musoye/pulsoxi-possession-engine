# PulsoXI: Reliability-Aware Football Possession Segmentation Using Tracking Evidence and Temporal Stabilization

## Team and objective

PulsoXI is our possession-control submission for FC Barcelona More Than a Hack 2027.

Team: PulsoXI

Team members: AbdulShakur Bashir, Abraham Folorunso, Abdulrasheed Abdulsalam

PulsoXI predicts H, A, D, or X for every tactical-video frame. The selected candidate is V3B+E2 with deterministic 1.5-second turnover stabilization. Unstabilized V3B+E2 is retained as the fallback.

## Data and responsible use

Production inference uses the organiser-provided tactical ATD data, official ATD metadata, Smart Tagging input, and tactical homography. Licensed files remain local and are not included in this public repository.

Independent checks used SkillCorner Open Data matches 1886347 and 1996435. The provider possession references were not manually annotated by our team. They were used only for external evaluation and selection of the stabilizer. These are reduced external H/A possession-overlap metrics, not official BAR-ATH private spellF1. No BAR-ATH private ground truth, human-produced BAR-ATH statistics, or published match statistics were used for tuning.

## Tracking preprocessing

ATD frame IDs are one-based and align to video frames as:

`video_frame = frame_id - 1`

Normalized coordinates are converted to metres using the official 105.3 m by 68.0 m pitch dimensions. The extractor computes ball visibility, ball coordinates, team-wise player counts and centroids, minimum player-ball distances, nearest-distance difference, ball displacement and speed, nearest-player relative speed, missing-ball gap length, and a relative-motion reliability flag.

Team membership comes from official metadata; track slots are not treated as persistent player identities. NaNs and absent players are excluded from distances and centroids. Motion features require consecutive frames within the official half blocks.

## E2 football-control evidence

E2 starts from the signed nearest-player distance difference:

`control = away_min_ball_distance - home_min_ball_distance`

It combines normalized proximity evidence with a causal five-frame proximity mean using a 35 percent persistence component. Reliable nearest-player relative speed contributes up to 0.20. Ball speed attenuates evidence, and speeds at or above 25 m/s are treated as discontinuities. Missing or invalid observations contribute only a 0.97-decayed prior and never create new evidence.

The V1-derived D/X mask is retained, so Smart Tagging dead-ball evidence remains separate from H/A control switching.

## V3B temporal decoding

V3B consumes E2 emissions and reliability signals. It accumulates opposing-team evidence, applies reliability-aware switching, decays evidence during missing observations, and resets independently at the two official match-half boundaries. D and X are not converted into possession labels by the stabilizer.

## Turnover stabilization

The postprocessor detects complete contiguous H-A-H or A-H-A runs within one half. A middle run no longer than 1.5 seconds is merged into its surrounding team. At 25 FPS this is compared as:

`middle_frame_count <= 1.5 * 25`

The implementation uses frame IDs and does not merge across half changes, frame gaps, D, or X. No reference labels are needed at inference time.

## Independent external evidence

On SkillCorner match 1886347, reduced original E2 mean overlap F1 was 0.1692. E2 plus 1.5-second stabilization produced 0.4366.

On SkillCorner match 1996435, reduced original E2 mean overlap F1 was 0.1960. E2 plus 1.5-second stabilization produced 0.4472.

These figures are not official competition scores. Stabilization materially reduced boundary recall, and only two provider matches were available. Provider definitions, coverage, and timing differ from the organiser's full task, so these results do not establish private BAR-ATH performance or guaranteed qualification.

## Reproduction and verification

Use Python 3.12 and install the pinned dependencies:

```text
python -m venv .venv
python -m pip install -r requirements-pulsoxi.txt
```

To verify existing local artifacts without rerunning raw inference:

```text
python reproduce_pulsoxi.py
```

To reproduce from authorised local data:

```text
python reproduce_pulsoxi.py --reproduce --data-dir /path/to/licensed_data
```

To validate the final CSV:

```text
python evaluation/validate_submission.py outputs/local_inference/PulsoXI.csv
```

The final submission must contain the exact frame,label header, 199,319 rows, frame indices 0 through 199318, and only H, A, D, or X labels.

Fresh local raw regeneration produced V3B+E2 SHA-256:

`46e9dc60762d4812d5ae6d7b23d71e54b48fb73c514b0e98349c80f9c1511bfe`

Applying stabilization to that regenerated file produced V4 SHA-256:

`83347409855d6d89c8ef30d500cf43de107c440187b25c227d0c23ca40e5115f`

Both files contain 199,319 rows and pass the official validator. The synthetic test suite has 62 passing tests. The Python verifier writes a machine-readable report under outputs/local_reproduction/.

## Limitations

The stabilizer explicitly protects official half boundaries and nonconsecutive frame IDs. The current public feature API does not expose a separate per-frame ATD tracking-validity mask to the stabilizer. Hash equality therefore verifies the current implementation but does not prove that every possible tracking discontinuity is independently represented. This is documented as a limitation rather than hidden.

## References

- Official submission specification: https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/SUBMISSION.md
- Official organiser repository: https://github.com/metrica-sports/more-than-a-hack-2027
- SkillCorner Open Data: https://skillcorner.com/skillcorner-open-data/
