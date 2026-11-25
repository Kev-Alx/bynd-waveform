# Saved R3 failure replay

Verdict: **FAIL_NO_ATTACK_SIGNAL**. This replays the existing 24 examples; future pilots must contain 48.

All 24 normalized adversarial transcripts equal clean transcripts. Source WER changes by 0.00000000 percentage points.

The median residual RMS uses 0.041297% of the fixed 0.003 RMS budget.

Run `PYTHONDONTWRITEBYTECODE=1 /opt/venv/bin/python /workspace/improvement/execution48/diagnostics_saved_r3.py --assert-attack-signal`.

Exit code 1 records the failure symptom. It does not mean that a dependency or the diagnostic program failed.

## Measured failure

| Saved condition | Median RMS | RMS budget used | Mean DC power fraction | Mean edge power fraction, first/last 256 samples |
| --- | ---: | ---: | ---: | ---: |
| initial-generator | 0.00018407346 | 6.135782% | 99.9979% | 0.6171% |
| adversarial | 1.2389224e-06 | 0.041297% | 10.0951% | 60.4733% |

Training completed 1779 optimizer updates / 7116 microbatches in 3541.89 seconds. 99.83% of logged update gradients exceeded the fixed norm-1 clipping threshold.

The initial generator mainly emits a constant offset. The trained generator suppresses almost all of its RMS energy. The remaining residual is too small to support a meaningful test of the allowed waveform budget.

The checkpoint source WER remains exactly 3.578947% at updates 500, 1000 and 1500. Mean STFT loss falls from 0.00053914753 to 0.00017185896; the selected checkpoint is update 1500 because WER ties and the registered selector then favors smaller STFT loss. That selection rule rewards preservation when every checkpoint fails to attack.

Same-clip first/last training observations:

| Quantity | Mean change | Median change | Range |
| --- | ---: | ---: | ---: |
| source_sum_nll | -0.00081978978 | 9.4413757e-05 | -0.17394257 to 0.37466431 |
| stft_loss | -0.0010571095 | -0.00031415241 | -0.012936205 to 0.00065007445 |
| content_loss | -3.1809766e-05 | -4.9894209e-07 | -0.0016382552 to 1.537504e-07 |

These comparisons control clip identity but compare changing generator parameters over training. They do **not** measure clean-to-adversarial NLL gain. The saved trace has no paired clean NLL and no source/penalty component gradient vectors; neither objective dominance nor gradient cancellation can be established from it.

The fixed weights are STFT=65.0836787015, content=186.612343972. Calibration equalized each separate waveform-gradient norm to source NLL at the median of three random probes. The sum of two penalties and their parameter-space alignment can behave differently near identity. The calibration is a starting convention, not evidence of balanced optimization.

## Ranked hypotheses and tests

1. **Quality penalties favor identity and overwhelm useful attack directions near the learned solution.** The generator removes most initial residual energy; STFT/content penalties shrink. Calibration balanced separate waveform-gradient norms at one random 1e-4 probe, which does not balance their joint parameter-space gradients throughout training. Test: At fixed per-example budget, 50-step NLL-only PGD succeeds on the same 48 clips but the identical optimizer with the registered STFT/content penalties fails. If both succeed comparably, penalty dominance alone is insufficient.

2. **Amortization across utterances cancels source attack directions or prevents the shared generator learning them.** The trained generator is nearly identity despite finite updates and repeated training exposure. No source/penalty component gradient cosines or cross-utterance cosines were saved, so cancellation remains unproven. Test: Same-objective per-example PGD succeeds on 48 fixed clips while the frozen generator fails under the same budget. If the per-example objective also fails, amortization alone is not the explanation.

3. **The generator initialization and interpolation architecture favor a DC or boundary response over input-specific adversarial structure.** Initial saved residuals have almost all power at DC; final residual spectra and boundary energy are measured below. Existing weights and finite gradients do not establish expressive attack directions. Test: After the PGD control, fit the unchanged generator to the fixed successful per-example residuals without ASR or quality losses. Failure to fit them under the same budget supports a representation or optimization bottleneck; successful fitting weakens it. This is a separate later experiment, not a retune of current coefficients.

4. **Teacher-forced lexical NLL can change without altering native beam decoding.** Every saved decoded transcript is unchanged. Paired clean/adversarial teacher-forced NLL was not saved; training NLL comparisons across examples cannot demonstrate this mismatch. Test: On the 48 fixed clips, record paired clean/adversarial sum NLL and native beam-5 WER at initialization and final PGD. Large paired NLL gains with unchanged decoding support the mismatch; decoded WER growth refutes it as the sole cause.

5. **The fixed 0.003 waveform box is too small for this source/model/domain at 50 updates.** The generator uses far less than the allowed budget, so its failure cannot establish that the budget is intrinsically insufficient. Test: NLL-only per-example PGD at Linf/RMS 0.003 reaches large budget fractions yet fails to raise paired source WER. Successful PGD under this exact budget rejects budget insufficiency as the sole cause. Do not increase the budget in this diagnostic comparison.

## Recommended controlled comparison

Use exactly 48 fixed pilot examples for every arm: clean; the frozen selected R3 generator; per-example waveform PGD with the identical R3 objective; the same PGD with NLL alone; and the frozen original B2 feature-delta baseline. Use the existing coefficients without retuning, unchanged 0.003 Linf/RMS waveform constraints, identical initialization and 50 updates for both PGD arms, and the same frozen source/evaluation policy. The runner must freeze the step size and all remaining optimizer choices before outcomes. B2 is an attack positive control; its feature-space ball is not the same threat model as the waveform box.

Record paired clean/initial/final teacher-forced NLL, objective components, component waveform-gradient norms/cosines, residual budget use, native beam decoding, exact edits, and the three existing quality scores. If the full objective works per example, focus on amortization. If only NLL-only PGD works, focus on penalty constraints. If neither works despite budget use, examine the source NLL/decoding mismatch and fixed budget before training another generator. Successful NLL-only attack audio still needs its quality judged; it is a diagnostic result, not a promotion.

No held-out model, 48-point pilot inference, new source inference, or final-reserve audio was read by this diagnostic. The JSON contains waveform hashes, full per-clip residual/spectral summaries, same-clip trace comparisons, and checkpoint parameter norms.