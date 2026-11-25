# Initial implementation and development experiments

Started 21 October 2026 (Asia/Taipei), after the user authorized continuation. **The bounded source-only cycle has finished; see [final measured results](06_initial_results.md).** All new source, tests, data, models, waveforms and receipts remain under `/workspace/improvement`. Original source files, configuration and historical results are preserved, with an additional [baseline snapshot manifest](baseline_snapshot_manifest.json).

## Registered scope

This first execution cycle is limited to source-only G0/G1 development on the fixed 24 previously exposed clips. The initial GPU envelope is capped at six hours; R3's first training seed is capped at one hour. The four held-out recognizers, the existing 48-clip confirmation pool, and the new final reserve are not used for generation, selection or initial evaluation. Human intelligibility has not been measured.

The [execution plan](execution/plan.json) was frozen before new optimization. A fresh final reserve was frozen before method selection: **200 LibriSpeech test-other utterances from 33 speakers**, with no speaker, source-file hash or normalized-text overlap with the historical development/test manifest union. It is a harder-speech evaluation shift, not a matched substitute for test-clean, and remains unscored. Exhaustive perceptual duplicate detection and absence from every pretrained model's data are not claimed.

Official LibriSpeech archives passed the publisher's MD5 checks and have SHA256 receipts. Nested training manifests contain **286 utterances / 251 speakers / 3,609.18 seconds** and **2,807 utterances / 251 speakers / 36,013.16 seconds**. Selection used speaker round robin and a fixed seeded ordering with duration-only stopping, excluding historical/final speakers and duplicate audio/text. The larger pool is prepared, not trained.

## Baseline and numerical controls

The actual pretrained-model baseline preflight repeated all **150 updates** on the shortest, median-upper and longest development clips. Loss, feature gradient, projected delta, Adam state and final audio hashes matched the frozen control exactly. The measured optimization portion took **44.72 seconds**, excluding model loading/checks. See [preflight result](execution/baseline-preflight/result.json).

Generation uses the previously verified strict numerical path. Formal source scoring uses [source_evaluate.py](source_evaluate.py) in a separate process, with the complete historical native evaluation backend asserted before predictions and again after scoring. This prevents strict generation flags from changing the evaluation protocol. The evaluator only accepts registered development IDs and the public Whisper-small checkpoint. Direct source-transcription observations inside a strict generation run are diagnostics, not the formal matched scores below.

## R1: decoded word-risk objective

The canonical implementation uses one real K=4 beam search, lexical-plus-EOS path probabilities, detached word-edit costs, five-update candidate refresh and a fixed source-gradient scale. Its three duration-selected calibration clips passed, fixing c=355.8228729. This matches the median gradient scale, not every recording: the long clip's initial scaled risk gradient is only about 1/89 of its sequence-sum-NLL gradient.

The three-clip pilot completed 150 updates. On the first clip of the subsequent 24-clip run, all four beam candidates differed only in punctuation/case and had zero normalized word-edit cost. The risk objective was flat; the canonical condition stopped **before any full-cohort optimization completed**. This is a method applicability failure, not a failed GPU job or a 24-clip performance result. Candidate and zero-delta evidence are preserved in [the failure receipt](execution/r1-calibrated-development24/failures/1673-143397-0019.json).

One repair was registered before its outputs: add exactly one synthetic candidate by deleting the middle word from the source's highest-ranked nonempty hypothesis. Native hypotheses and the reference remain in the candidate set, and no additional fallback is permitted. This counterfactual is not a posterior beam hypothesis; its prior-art and content-preservation limitations remain. Repair calibration passed on the same three clips, fixing c=376.6860671. Both its three-clip pilot and all **24 development clips / 1,200 updates** completed. Full-cohort generation took **939.64 seconds**, with **10.132 GB** peak allocated GPU memory. Formal saved-audio scoring took **106.03 seconds** and returned **58.5263% source WER, 3.1778 DNSMOS, 3.9957 NISQA, 3.3497 UTMOS**. In this run the strict-generation WER diagnostic happened to match the formal native score; numerical-policy equivalence is still not presumed in general.

**The repaired R1 fails the registered UTMOS floor of 3.5.** Against quality-admissible B2 it gains 12.8421 WER points but loses 0.3263 UTMOS. Against the stronger LR 0.1 control it loses 16.2105 WER points for only +0.0311 UTMOS. It therefore supplies no qualifying attack/quality improvement. The c=1 ablation remains unexecuted: a scale ablation is not needed to promote a condition that already fails the quality gate, and the one scientific candidate-set repair has been spent. See [formal source summary](execution/r1-deletion-development24/source-summary.json), [independent resume audit](execution/resume-audit.md), [canonical report](execution/r1-report.md), [repair registration](execution/r1-native-plus-deletion-registration.json), and [completed repair generation](execution/r1-deletion-development24/result.json).

## R2: local-neighbor feature geometry

All 24 clips were retained. Twenty-one completed 50 updates, two had no admissible neighbor bank, and one had no first-order feasible descent. The latter still performed one optimizer step before projection returned it to zero, giving **1,051 actual objective calls/optimizer steps**, not 1,200. Mean admissible-frame coverage was 6.27%; mean radius was 0.02046 and maximum radius 0.04920 against the allowed 0.1. All finite, cosine, physical-radius and zero-frame constraints passed.

Fresh native source evaluation scored all five waveform variants on the same 24 clips / 475 reference words:

| Condition | Source WER % | DNSMOS | NISQA | UTMOS |
|---|---:|---:|---:|---:|
| Clean | 3.5789 | 3.3323 | 4.0838 | 4.1886 |
| Zero-edit reconstruction | 3.5789 | 3.3317 | 4.2124 | 4.1667 |
| R2 local-neighbor attack | 3.3684 | 3.3320 | 4.2142 | 4.1661 |
| Norm-matched attenuated original attack | 4.8421 | 3.3020 | 4.1926 | 4.0837 |
| Frozen strong original attack | 74.7368 | 3.1640 | 4.0350 | 3.3186 |

**R2 did not increase source recognition errors.** Its only changed hypothesis relative to reconstruction corrected one substitution. It preserved automatic quality while restricting the attack almost entirely away. The norm-matched comparison passes the literal small quality-screen arithmetic (+0.0824 UTMOS, −1.4737 WER points), but both conditions are near clean error rates; that diagnostic is not the primary promotion control. Against registered quality-admissible B2 (original sum-NLL, LR 0.02, **45.6842% WER / 3.6761 UTMOS** on the same 24 clips), R2 loses **42.3158 WER points**, failing G1's ≤2-point loss guard. No confirmation/final promotion occurred.

Generation took 298.04 seconds, with 8.304 GB peak allocated GPU memory; native source/quality evaluation took 193.80 seconds. Fresh replay reproduced the frozen control's WER, DNSMOS and UTMOS exactly; NISQA differed by about 1e-7. See [source summary](execution/r2-canonical-zero-uniform-dev24/source-summary.json), [geometry diagnostics](execution/r2-canonical-zero-uniform-dev24/generation-diagnostics.json), and [paired development analysis](execution/r2-canonical-zero-uniform-dev24/paired-development-analysis.json).

## R3: amortized waveform generator

The independent waveform U-Net has **1,590,833 trainable parameters**. It uses the registered spectral and frozen-WavLM content losses, sequential backward passes, bounded waveform residuals, and source-only checkpoint selection. Its CPU tests include exact duration/bounds, gradient flow, calibration failures and native interpolation execution checks.

The real preflight found two execution problems before any training. First, the long calibration clip exceeded the cached-feature maximum-absolute tolerance (0.00011206 versus 0.0001), although its relative Frobenius error was only 1.075e-6. Native and gradient-adapter outputs matched bitwise, and strict recomputation repeated exactly. The named strict-reference repair therefore regenerates both clean and adversarial content features with the same frozen transform, retaining the original [cache-conformance failure](execution/r3-waveform/preflight.json).

Second, the actual content-gradient calibration failed because the pinned upstream encoder computes its positional convolution from a transposed view, then changes that saved input using `x += x_conv`. The observed [autograd failure](execution/r3-waveform-reference-repair/run-failure.json) is preserved. A separate [execution adapter](r3_encoder_repair.py) clones only the positional-convolution input in differentiable calls and removes the temporary hook after every forward. The encoder's operations, values, weights, layer, normalization and scientific objective remain unchanged; the [versioned runner](r3_real_runner_v2.py) binds both the original source and the repair. This is an implementation repair, not a negative learned-method result or a new hyperparameter trial. Five [actual-upstream CPU regressions](execution/r3-encoder-repair-tests.log) reproduced the original error, confirmed bitwise-unchanged forward values and matched the gradient from an independently constructed out-of-place upstream reference.

The repaired **real GPU preflight passed** in **4.14 seconds**, with **4.798 GB** peak allocated memory. Fixed waveform-gradient calibration produced λ_STFT=65.083679 and λ_content=186.61234. The longest training clip (17.615 seconds) completed four accumulated microbatches and one AdamW update with finite nonzero parameter gradients and valid duration/amplitude bounds. That fit model was discarded; a freshly initialized seed-11 generator began its registered one-hour training run at **10:10 on 22 October, Asia/Taipei**. Its current completion state and final measured results are maintained in [the live status](JOB_STATUS.md) and [derived results table](06_initial_results.md). No trained result is inferred from this preflight. See [preflight](execution/r3-waveform-autograd-repair/preflight.json), [calibration](execution/r3-waveform-autograd-repair/penalty-calibration.json), and [repair registration](execution/r3-autograd-execution-repair-registration.json).

## Detached continuation and verification

The previous chat stopped at a usage limit. Its already-running R1 optimization subsequently finished successfully; no experiment was repeated to reconstruct lost chat state. On **22 October 2026, Asia/Taipei**, [continue_development.py](continue_development.py) was launched as its own operating-system session with no controlling terminal. It sequences native R1 scoring, the R3 cache-reference preflight, and the one-hour R3 trial. Only the previously declared named strict-reference repair is allowed, and only after the exact cache-mismatch receipt with native/adapter equivalence exists. Other preflight failures stop the R3 queue for diagnosis.

[Job status](JOB_STATUS.md) and [JSON status](JOB_STATUS.json) refresh every 30 seconds. Per-stage logs, PIDs and explicit exit codes live under `execution/jobs/`; experimental receipts retain their original paths. Notifications are disabled. The current CPU suite passed **92 tests** in 4.04 seconds, including the five actual-upstream encoder-gradient regressions; these tests complement the actual frozen-baseline replay and do not establish an improved method. A second detached CPU process waits for the trial's explicit exit record, then refreshes the independent audits, [measured-results table](06_initial_results.md) and package validation without requiring another chat turn.

## Interpretation

These are exposed-development measurements and implementation checks. They do not establish improved black-box transfer, human intelligibility, defense robustness or a publishable contribution. The completed negative outcomes test the proposed mechanisms: top-K hypotheses can offer no word-risk gradient, a counterfactual candidate can supply gradients without meeting the audio-quality floor, and high-similarity local feature geometry can be too sparse to support a useful attack. The learned waveform method is a one-seed, one-hour feasibility trial; the detached postprocessor records its actual outcome and audit status in [the results table](06_initial_results.md) after execution, including any failure. The final/confirmation reserves remain unused.
