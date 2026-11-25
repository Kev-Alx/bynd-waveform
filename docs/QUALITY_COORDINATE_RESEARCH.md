# Clean-statistic Adam coordinates

The layer-5/7 reconstruction study completed successfully, but neither layer passed its unchanged quality screen. Retain the matched raw WavLM-Large layer-6/regular HiFi-GAN pair. The completion audit at `/workspace/reproduction-runs/feature-layer-completion-review-20261020` binds 1,470 frozen inputs, 48 features, 96 scores/WAVs, and 24 exact layer-6 controls. Both attacks were skipped.

The strong repeatable full-objective control reaches WER **74.7368%**, DNSMOS **3.1640**, NISQA **4.0350**, UTMOS **3.3186**, versus paper targets **75.43%, 2.954, 3.526, 3.590**. No joint match exists. Removing perceptual losses did not resolve the gap. Saved-output diagnostics show median perturbation-energy share **21.95%** in the clean waveform's quietest frame quartile, versus **0.0265%** of clean waveform energy there; output RMS there is **1.300 times** reconstruction RMS. The lowest temporal-variance channel quartile contains **24.28%** of perturbation energy versus **11.80%** of clean feature variance. These observations motivate experiments; they do not establish the cause of poor quality. No listening or new neural inference was used to derive them.

## Fixed experiments

Run exactly two cases, in this order, on the existing 24 development clips:

1. `time_energy`: clean waveform RMS for each 400-sample receptive field at hop 320. Reject any mismatch with the frozen feature frame count; no padding, interpolation or frame dropping.
2. `channel_std`: per-channel temporal population standard deviation of original clean q.

For each nonnegative statistic s, compute `raw=clamp(s/max(s), .1, 1)` and `W=raw/sqrt(mean(raw**2))` on CPU float64, then cast to q dtype. A zero statistic uses ones. All entries remain positive, with ratio at most ten. RMS normalization approximately preserves the first physical step's overall scale under Adam; it does not guarantee identical norms because gradients, Adam epsilon and projection matter. No alternate floors, inverse weights, combined time/channel case or grid is queued.

Parameterize the physical perturbation as `delta=W*theta`, initialize theta to zero and optimize theta with the same Adam LR=.1. Evaluate the original full objective at `q+delta`. After every update, compute the physical delta norm and, when needed, multiply theta by `(.1*(norm(q)+1e-8))/norm(delta)`. This implements the original radial physical projection up to float32 rounding. Positive W makes the mapping invertible: the physical feasible set and objective are unchanged. The finite 50-step optimizer path changes. Simply scaling gradients by W would mostly cancel through Adam's normalization and is not the experiment.

Keep rho=.1, lambda=1, alpha=1, cutoff=6000 Hz, original clean denominators, sum lexical NLL, original uppercase references, no-timestamps prompting, fifty updates, constant LR, zero initialization and terminal update 50. Models stay frozen. This is an independent update choice left unspecified by the paper, not a recovered author algorithm. Production attack sources remain unchanged. No kNN matching or waveform postprocessing.

## Execution and evidence

`/workspace/scripts/run_coordinate_research.py` first runs a uniform-coordinate control on the fixed first three IDs. Its graph bypasses identity multiplication; all 150 projected updates, gradient/loss/moment hashes and final audio/Adam states must match the frozen full-objective control bitwise. A failure stops the queue.

For each weighted trajectory retain all 50 theta and physical delta tensors, frozen W, losses, theta-gradient hashes, full moment receipts, final Adam state and final audio. Independently reconstruct W from original inputs, require exact `delta=theta*W`, validate the original radius (<=.100001), and bind terminal resynthesis to WAVs. Reuse the frozen scalar/score/reference/count/backend audit. Generation uses strict math, explicit reflection and fresh-process `CUBLAS_WORKSPACE_CONFIG=:4096:8`. Native saved-audio scoring runs with the original auto policy and workspace unset. Existing cached q uses the independently validated original encoder policy; no new encoder extraction is needed.

The descriptive direction signal is UTMOS gain >=.05, WER loss <=5 percentage points, each MOS >2.5 against the full control. Report published joint thresholds separately without relaxing them. No automatic promotion, fresh48, new999, corpus selection or scoring changes. The pilot has been reused extensively; any positive result needs an independently frozen follow-up and cannot establish generalization or human quality.

Output: `/workspace/reproduction-runs/paper-quality-coordinate-research`. `job.json`, `progress.json`, `research.log`, phase logs, `control-preflight.json`, `results.json` and `report.md` expose actual status. Global `JOB_STATUS.md` and JSON track the queue. Detached logging/supervision persists after this chat. Discord remains disabled.
