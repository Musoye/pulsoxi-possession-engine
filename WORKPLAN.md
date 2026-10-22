# PulsoXI Workplan — updated 2026-10-22

## Already confirmed
- [x] Team name: PulsoXI
- [x] Team lead's registration confirmed by organiser's welcome email
- [x] Submission deadline: November 17 at 23:59 CET
- [x] Data access link promised in a separate email within 72 hours of registration
- [x] Submission instructions promised by November 14
- [x] Scoring objective: spellF1; macroF1 first tie-break
- [x] Organiser's public source and evaluation scripts identified
- [x] PulsoXI Phase 0 code skeleton and synthetic decoder tests prepared

## Still to do
- [ ] At least 2 other teammates register with the exact name PulsoXI by November 10
- [ ] Clone official repository and run `python evaluation/run_evaluation.py` locally
- [ ] Receive licensed assets and unpack safely under `data/`
- [ ] Verify data frame alignment and slot/team mapping
- [ ] Regenerate the organiser's reference baseline from assets
- [ ] Build team-controlled and dead-ball emissions from football-specific features
- [ ] Evaluate temporal decoding and report honest limitations
- [ ] Generate a reproducible CSV via source code, without manual corrections
- [ ] Validate against official `validate_submission.py`
- [ ] Finish public reproducibility repository and concise technical report
- [ ] Submit all three deliverables by November 17, 23:59 CET

## Guardrails
Do not share video, tactical tracking, frame-level exports, or licensed derivatives outside the signed data agreement. Public code must run against the organiser's assets downloaded separately by authorised participants. Public five-minute truth is a small sanity-check sample, **not** a robust hyperparameter-selection benchmark.
