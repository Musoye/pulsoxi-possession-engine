# PulsoXI Technical Report Evidence Outline

## Objective
Possession-label inference from tactical tracking and Smart Tagging evidence for the More Than a Hack 2027 submission format.

## Data and assets
- Public organiser framework and evaluation files used.
- Authorised local ATD, homography, and Smart Tagging files required for reproduction.
- Licensed tracking files, match video, full-match labels, predictions, and detailed diagnostics intentionally omitted.

## Feature extraction
- Normalised coordinates converted to metres using the documented pitch dimensions.
- Team membership comes from ATD metadata; track slots are not persistent player identities.
- Ball visibility, team counts/centroids, nearest-player distances, distance difference, displacement/speed, valid relative motion, and missing-ball gaps.

## E2 emissions
- Signed proximity score from away-minus-home minimum distance.
- Causal five-frame persistence contribution.
- Valid relative-motion contribution.
- Ball-speed attenuation and discontinuity handling.
- Missing observations decay prior evidence without fabricating turnovers.
- E1 remains unchanged.

## V3B decoder
- V1 supplies the fixed D/X mask.
- Live H/A intervals are decoded independently within official halves.
- Reliability-aware H/A transition costs are applied without changing dead-ball thresholds.

## Dead-ball and Smart Tagging treatment
- SET PIECES events provide bounded D evidence.
- Team tags provide only the documented auxiliary bias.
- D and X are never allowed to bridge official half boundaries.

## Verification evidence
- 36 synthetic/unit tests passed.
- Four candidate submissions passed format validation.
- Public evaluator passed on supplied public examples only.
- Structural diagnostics are reported without claiming accuracy or spellF1.

## Limitations
- No private full-match ground truth was available.
- No competition accuracy or spellF1 improvement is claimed.
- Full ATD reappearance audit is deferred because of local memory/runtime limits.
- E2 coefficients are engineering defaults, not learned parameters.

## Reproduction
1. Install `requirements-starter.txt` in an isolated environment.
2. Place authorised files under ignored `data/tactical/`.
3. Run `scripts/run_inference.py --candidate v3be2`.
4. Validate with `evaluation/validate_submission.py`.
5. Keep all generated outputs local and ignored.
