# Independent CPU audit of initial execution

Updated 2026-10-21T13:04:56.381706+00:00. Status: **pass**.

No neural model was loaded or executed; CUDA remained uninitialized. The audit imports no experimental implementation. It reconstructs saved R2 mixtures using NumPy and reads tensors on CPU. Saved GPU receipts are distinguished from independent CPU calculations.

- Preserved source/config snapshot: 41 files; frozen plan inputs: 34 files. Current originals and snapshots match their bindings.
- Baseline preflight: all 150 saved update receipts and three final-audio equality flags agree with their frozen reference receipts; neural computation was not replayed.
- Frozen reserve: 200 utterances / 33 speakers; saved selection replay and exclusion checks pass. No saved inference-bearing record uses this reserve or confirmation48.

- final-reserve-200: 200 recordings, 33 speakers, 1320.930 seconds; independent selection replay = True.
- train-1h: 286 recordings, 251 speakers, 3609.175 seconds; independent selection replay = True.
- train-10h: 2807 recordings, 251 speakers, 36013.160 seconds; independent selection replay = True.

## Completed source evaluation

| Run / variant | Clips | WER % | DNSMOS | NISQA | UTMOS |
|---|---:|---:|---:|---:|---:|
| r2-canonical-zero-uniform-dev24 / adversarial | 24 | 3.368421 | 3.332028 | 4.214219 | 4.166126 |
| r2-canonical-zero-uniform-dev24 / baseline_attenuated | 24 | 4.842105 | 3.301993 | 4.192638 | 4.083689 |
| r2-canonical-zero-uniform-dev24 / baseline_frozen | 24 | 74.736842 | 3.164039 | 4.034993 | 3.318612 |
| r2-canonical-zero-uniform-dev24 / clean | 24 | 3.578947 | 3.332344 | 4.083788 | 4.188579 |
| r2-canonical-zero-uniform-dev24 / reconstruction | 24 | 3.578947 | 3.331732 | 4.212429 | 4.166712 |

These are Whisper-small source results on reused development clips. They are not held-out-target transfer or human-content results. A completed implementation or generated cohort is not a positive scientific result.

## R1 calibration and execution status

r1-calibration: independently recomputed saved-scalar median scale = 355.822872887595; three finite positive norm ratios. Underlying gradient vectors were not independently recomputed.
r1-deletion-calibration: independently recomputed saved-scalar median scale = 376.686067149993; three finite positive norm ratios. Underlying gradient vectors were not independently recomputed.
r1-calibrated-development24: saved status negative; completed 0 of 24.
r1-calibrated-pilot3: saved status completed; completed 3 of 3.
r1-calibration: saved status passed; completed not a generation stage of not applicable.
r1-deletion-calibration: saved status passed; completed not a generation stage of not applicable.
r1-deletion-development24: saved status running; completed not a generation stage of not applicable.
r1-deletion-pilot3: saved status completed; completed 3 of 3.

## Source-only screening interpretation

r2-canonical-zero-uniform-dev24: compare against the registered quality-admissible B2 LR=.02 control (WER 45.684211%, UTMOS 3.676064). Candidate delta WER -42.315789 points; delta UTMOS +0.490062. Automatic quality admissible: True; registered source/quality performance screen: False. The near-clean attenuated diagnostic is not a substitute promotion control. No human or held-out-transfer claim follows.

## Independent R2 geometry

r2-canonical-zero-uniform-dev24: 24 records, 1051 actual updates; status counts {'no_first_order_feasible_descent': 1, 'completed': 21, 'degenerate_bank': 2}. Maximum independently reconstructed-delta absolute discrepancy 4.08e-06; largest float32 norm-matching residual 3.29e-06. All final mixtures satisfy registered amplitude, bank, per-frame cosine and relative-radius constraints; unchanged clean/reconstruction audio samples match the frozen control.
r2-canonical-zero-uniform-smoke3: 3 records, 100 actual updates; status counts {'completed': 2, 'degenerate_bank': 1}. Maximum independently reconstructed-delta absolute discrepancy 3.03e-06; largest float32 norm-matching residual 6.44e-07. All final mixtures satisfy registered amplitude, bank, per-frame cosine and relative-radius constraints; unchanged clean/reconstruction audio samples match the frozen control.

Norm matching is deterministic baseline attenuation on the same clean features. Floating-point output norms need not be literally equal; residuals and elementwise checks are reported. No vocoder resynthesis was performed by the auditor.

## Limits and issues

- No discrepancy found in the checked snapshot.

- No neural outputs or complete optimizer trajectories were independently reexecuted.
- R1 calibration gradient norms are saved producer scalars; underlying gradient vectors were not saved for independent recomputation.
- Hash integrity does not establish human intelligibility, black-box transfer or superiority.
- This is a snapshot of concurrently finishing execution; rerun after later phases complete.

Re-run `/opt/venv/bin/python /workspace/improvement/execution/audit_initial_execution.py` after new stages finish. Detailed bindings, results and record-level geometry are in `audit_initial_execution.json`.
