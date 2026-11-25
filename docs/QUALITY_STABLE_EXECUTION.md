# WER and sound-quality anchors under validated numerical execution

19 October 2026. The optimizer mechanism job completed and its 630 frozen inputs, 12 saved captures and 48 branches passed the completion review. Perceptual-gradient removal changes the first projected Adam step by 20.6–42.2%; later local differences are smaller. These are moment-matched local counterfactuals, not full loss ablations.

The isolated explicit-reflection path passed native/explicit bitwise forward checks at all six fixed states. Its repeated full gradients were identical across three repeats at each state. Gradient differences from the old, nondeterministic native-math path were small (relative L2 4.13e-7–7.21e-7). Two separate fresh-process runs also matched all 50 projected deltas, loss/gradient/moment hashes and final waveforms for each of three fixed development clips. Forty CPU equivalence/integration tests passed. This establishes numerical repeatability for those clips and this environment, not author parity or WER/MOS improvement.

This next batch evaluates exactly three previously studied settings on the existing frozen 24-clip development pilot:

| Anchor | Lexical NLL | LR | Prefix |
|---|---|---:|---|
| strong-sum-lr-0.1 | Sum | 0.1 | No timestamps |
| quality-sum-lr-0.02 | Sum | 0.02 | No timestamps |
| timestamp-sum-lr-0.05 | Sum | 0.05 | Initial 0.00 timestamp |

All use the existing raw-input WavLM-Large layer-6/regular HiFi-GAN cached clean features, original transcripts, Whisper-small checkpoint, full vocabulary, original-clean perceptual denominators, alpha=1, rho=0.1, lambda=1, zero initialization, constant Adam and 50 updates with the projected terminal delta returned. Keep the original 6000 Hz cutoff and STFT conventions. The sole new execution policy is explicit reflected padding for differentiable Whisper/HF plus strict deterministic math CUDA. This is not a new hyperparameter grid.

Freeze the existing 24 IDs/audio hashes, feature cache, model artifacts, tokenizer, scorers and all relevant sources before execution. Save all iterates and terminal optimizer evidence. For the strong anchor, require its first three final delta/audio hashes to match the validated trajectories. Independently resynthesize every final waveform from saved q/delta, check every saved projected iterate, reconstruct WER counts and aggregate all 72 score rows per setting. No peak/loudness normalization, clipping, reference substitution or sample filtering is added.

Evaluate the saved FLOAT WAVs using the unchanged native beam-5/temperature-0 Whisper decoder, corpus WER normalizer and fixed DNSMOS OVRL/NISQA overall/UTMOS22 strong scorers. Generation controls do not silently change the evaluation execution policy. Record both policies explicitly. Compare with the published WER=75.43%, DNSMOS=2.954, NISQA=3.526 and UTMOS=3.590 descriptively. Earlier native-execution scores are separate-run context because numerical sensitivity has been demonstrated; differences alone do not prove an algorithm gain. A joint descriptive target requires all four values, without the earlier research's two-point WER tolerance.

No automatic candidate selection, fresh-48 confirmation, validation expansion or 999-clip test follows this batch. Those data remain untouched. Completion status, per-stage logs and the cumulative report persist. Discord remains disabled.
