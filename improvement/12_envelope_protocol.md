# Fixed clean-energy waveform experiment

Registered22 October2026, Asia/Taipei, after the completed refinement failed its advancement gate. This protocol is frozen before optimization outputs. [The explanation](11_changes_and_goals.md) gives the broader goals and negative results; [the primary-source note](execution_envelope/prior-work.md) defines the prior-art boundary.

## Hypothesis and intervention

A global waveform-amplitude bound may allow disruptive noise in quiet speech regions. A clean-energy-dependent feasible set may preserve automatic speech quality while retaining useful recognition errors. This explanation is unproven: the experiment tests it. The earlier hard STFT/content-cap continuation is closed; its1,789 no-op updates and unchanged WER do not justify another repair or a looser repeat.

Let clean native16kHz mono speech be x. Make400-sample frames at160-sample hops, adding the frame ending at the final sample if necessary. Compute each frame's clean RMS in CPU float64. For frame j, `c_j=min(0.003, RMS(x_j)*10^(-30/20))`. For sample t, set `b_t=min(c_j for every frame j containing t)`. No floor, trimming, resampling, loudness normalization or transcript change is permitted. An exactly silent frame forces zero perturbation in its samples.

The primary candidate, **envelope_nll50**, starts from zero residual and takes50 updates:

`delta <- clip(delta + (2*b/50)*sign(gradient_x source_sum_NLL), -b, b)`, followed by the valid-audio intersection `-1 <= x+delta <= 1`.

Choose the final iterate. There is no decoding/MOS stopping, iterate selection, random restart, threshold tuning or coefficient sweep. The coordinate step and feasible set change together; their individual causal effects are not isolated. Every registered frame has residual energy at most1/1000 of its clean energy before float32 roundoff. The guarantee does not cover arbitrary windows, frequency masking, perceived noisiness or retained words. Silent0/0 frames have zero perturbation rather than a defined SNR.

**envelope_full50** is a prespecified objective ablation using the same envelope/steps plus the frozen R3 STFT/content weights65.08367870148994 and186.6123439723859. The primary candidate remains NLL-only regardless of this ablation's outcome. STFT/content are measured without penalty backwards in the primary candidate, retaining comparable diagnostics.

## Comparisons and data

Report seven conditions: clean, originalFV50 (LR0.02/rho0.1/full originalTV+HF), global_full50, global_nll50, envelope_nll50, envelope_full50, and attenuated_global_nll50. The global conditions use the same frozen50-step algorithms as the previous48-point study. Their saved pilot outputs and the originalFV50 output are reused after hash checks; they are not regenerated for cosmetic consistency.

The attenuation diagnostic uniformly scales the frozen global_nll50 residual to the final envelope_nll50 residual RMS, without any decoded/MOS feedback. Amplification is prohibited; inability to attenuate to the requested RMS stops generation and is recorded, rather than changing the recipe. This is a postprocessing diagnostic, not a claim that the global optimizer is optimal at the smaller norm. It helps distinguish energy allocation from merely reducing total perturbation energy.

Pilot: the exact existing48 utterances /30 speakers /360.010s, manifest SHA256 `9a5f56f5cca6c073dbe6e2b6d804971735045ea80c4456bea0d577b7e292748a`. Conditional larger validation: the prepared76 utterances /34 speakers /524.905s, SHA256 `3747ca5b726ef899c24bc8f92e83ba44f086a537bffcf98b9bcd52cba71df30a`. The larger set has historical study exposure and shared speakers. Its utterance/hash/text separation is established in [the receipt](execution_refine/validation-76.receipt.json); it is not an untouched final set or an independent-speaker claim.

All methods keep frozen Whisper-small, the strict generation path, original transcripts, whole audio and the same formal native evaluation backend. Validation recomputes all seven registered conditions from the same source-only recipes. No additional surrogate, new training data, learned generator or quality-score gradient is introduced.

## Advance once, or stop

All seven conditions must contain the complete cohort, and the independent audit must pass. Any failed/partial generation prevents a scientific aggregate/promotion claim. The quality floors remain mean DNSMOS≥3, NISQA≥3.75 and UTMOS≥3.5.

The primary candidate is fixed in advance. The primaryG1 screen requires candidate andFV50 quality admissibility, then either candidate WER at least3 points higher thanFV50, or candidate UTMOS at least0.05 higher with WER loss no greater than2 points.

A separate **exploratory shape signal** can justify validation if the candidate passes all quality floors and all four conditions: WER≥global_full50−2 points, NISQA≥global_full50+0.10, UTMOS≥global_full50−0.05, and WER≥the RMS-matched attenuation diagnostic+3 points. This does not relax or rename primary superiority. It requires a meaningful quality benefit and an energy-allocation signal, rather than automatic promotion of an ineffective clean-like attack.

If either rule passes, freeze the candidate, control, pilot scores, audit and validation-manifest hashes in read-only receipts before larger validation. Evaluate76 automatically, retaining the same primary candidate and all controls. Do not choose the weighted ablation after seeing validation. If neither rule passes, stop this configuration; do not run another envelope setting. Report descriptive paired10,000-replicate speaker-bootstrap intervals, not confirmatory p-values or human/transfer confidence.

## Execution, accounting and reproduction

The continuation runs in a detached operating-system session with no controlling terminal, a single-GPU file lock,30-second heartbeat status, per-stage logs and explicit exit records. Generation uses `CUBLAS_WORKSPACE_CONFIG=:4096:8`; native source/MOS scoring runs in a fresh process with that variable unset. The existing scorer is reused with only the new frozen cohort/variant registration. An independent CPU auditor reconstructs the clean envelopes from raw audio, checks every saved final waveform and frame energy, verifies source-word counts and score aggregates, and checks objective/gradient-call receipts through the earlier independent helper.

The recorded scheduled subtotal is approximately3.391 hours, including the completed refinement's4,547.135 seconds. [The ledger](execution_envelope/compute-ledger.json) records exact input receipts and early timing limitations. This queue has a hard **7,200-second/two-hour wall limit**, including scoring and CPU stage waits, leaving about36 minutes against the initial six-hour envelope for previously disclosed gaps. Actual forwards/backwards and wall time remain visible; the NLL-only and full-loss conditions have different content-backward costs. No inference-only efficiency or equal-total-compute claim is made.

Implementation: [envelope primitive](envelope_pgd.py), [registration/generation](envelope_experiment.py), [native evaluator adapter](source_evaluate_envelope.py), [independent audit](execution_envelope/audit_envelope.py), [selection/report](postprocess_envelope.py), and [detached supervisor](supervise_envelope.py). The supervisor runs registration → pilot generation → native scoring → audit → fixed-candidate decision → conditional validation generation/scoring/audit/report → package check. Existing or partial outputs cause a stop, not a silent overwrite/retry.

All artifacts remain under `/workspace/improvement`. [Live status](JOB_STATUS.md), [results](13_envelope_results.md). Confirmation48, final200, held-out recognizers, human recruitment, external messages and notifications remain unused. No novelty or publishability claim is made.
