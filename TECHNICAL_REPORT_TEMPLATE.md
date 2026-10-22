# PulsoXI — technical report (draft template; NOT YET A SUBMISSION)

## Objective
Infer H / A / D / X labels for all 199,319 tactical-video frames using an entirely reproducible algorithm.

## Assets used
*Complete after confirming access and terms.*

## Data audit and alignment
ATD frame ID begins at 1; submission frame begins at 0. Tactical video is the reference. List missingness and tracking issues after running on the licensed dataset locally.

## Approach
- Spatial feature extraction: **not implemented yet**
- Live/dead likelihoods: **not implemented yet**
- Team-control likelihoods: **not implemented yet**
- Temporal Viterbi decoder: prototype implemented; transition penalties are not tuned

## Experiments
**No PulsoXI competition scores yet.** Log each model variation, data/label availability, validation split, spellF1 and macroF1 only after actual controlled experiments.

## Reproduction
List exact frozen package versions, data placement, one command that regenerates the submitted CSV, and the official validation command.

## Limitations
Include unsynchronised broadcast feed, missing ball tracking, identity switches, limited public ground truth, dead-ball boundary uncertainties.
