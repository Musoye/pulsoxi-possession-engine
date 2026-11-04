# PulsoXI: Reliability-Aware Football Possession Segmentation Using Tracking Evidence and Temporal Stabilization

## Objective and data

PulsoXI predicts `H`, `A`, `D`, or `X` for each tactical-video frame in the More Than a Hack 2027 possession task. The selected candidate is V3B+E2 with deterministic 1.5-second turnover stabilization; unstabilized V3B+E2 is the fallback. Production inference uses the organiser-provided tactical ATD, official metadata, Smart Tagging, and homography assets. Licensed files remain local and are not part of this repository.

The independent checks used SkillCorner Open Data matches 1886347 and 1996435. These provider references were not manually annotated by our team. They are reduced external H/A possession-overlap metrics, not official BAR-ATH private spellF1, and were not derived from the competition ground truth.

## Tracking preprocessing

ATD frame IDs are one-based and align to video frames as `video_frame = frame_id - 1`. Normalized coordinates are converted to metres using the official 105.3 m by 68.0 m pitch dimensions. Features include ball visibility, ball coordinates, team-wise player counts and centroids, minimum player-ball distances, nearest-distance difference, ball displacement and speed, nearest-player relative speed, missing-ball gap length, and a relative-motion reliability flag.

Team membership comes from official metadata; track slots are not treated as persistent player identities. NaNs and absent players are excluded from distances and centroids. Derivatives require consecutive frames within the official half blocks.

## E2 evidence and V3B decoding

E2 starts from the signed distance difference: `control = away_min_ball_distance - home_min_ball_distance`. It combines normalized proximity evidence with a causal five-frame proximity mean using a 35% persistence component. Reliable nearest-player relative speed contributes up to 0.20. Ball speed attenuates evidence, and speeds at or above 25 m/s are treated as discontinuities. Missing or invalid observations contribute only a 0.97-decayed prior and never create new evidence.

V3B consumes E2 emissions and reliability, accumulates opposing-team evidence, and resets independently at the two official half boundaries. The V1-derived D/X mask is retained, so Smart Tagging dead-ball evidence remains separate from H/A switching.

## Turnover stabilization

The postprocessor detects complete contiguous H-A-H or A-H-A runs within one half. A middle run no longer than 1.5 seconds is merged into its surrounding team. At 25 FPS this is compared as `middle_frame_count <= 1.5 * 25`; the implementation uses frame IDs and does not merge across half changes, frame gaps, D, or X. No reference labels are required at inference time.

## External evidence

On SkillCorner match 1886347, reduced original E2 mean overlap F1 was 0.1692 and E2 plus 1.5-second stabilization was 0.4366. On match 1996435, the corresponding values were 0.1960 and 0.4472. These figures are not official competition scores. Stabilization materially reduced boundary recall, and only two provider matches were available; provider definitions and coverage differ from the organiser’s full task.

## Verification and reproducibility

Fresh local raw regeneration produced a V3B+E2 file matching the preserved candidate byte-for-byte: SHA-256 `46e9dc60762d4812d5ae6d7b23d71e54fb73c514b0e98349c80f9c1511bfe`. Applying stabilization to that regenerated file produced V4 SHA-256 `83347409855d6d89c8ef30d500cf43de107c440187b25c227d0c23ca40e5115f`. Both contain 199,319 rows and pass the official validator. The synthetic test suite has 62 passing tests.

The exact reproduction command is documented in the README; `reproduce_pulsoxi.ps1` runs persistently, writes timestamped logs, validates outputs, and compares hashes. The public submission specification requires the exact `frame,label` header, frames `0..199318`, and labels H/A/D/X.

## Limitations

The stabilizer explicitly protects official half boundaries and nonconsecutive frame IDs. The current public feature API does not expose a separate per-frame ATD tracking-validity mask to the stabilizer, so hash equality verifies the current implementation but does not prove every possible tracking discontinuity is independently represented. No private full-match labels, human-produced BAR-ATH statistics, or leaderboard results are claimed.

## References

- Official submission specification: https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/SUBMISSION.md
- Official organiser repository: https://github.com/metrica-sports/more-than-a-hack-2027
- SkillCorner Open Data: https://skillcorner.com/skillcorner-open-data/
