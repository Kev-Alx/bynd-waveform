# 48-point pilot results

Updated **2026-10-22T13:39:31+08:00 (Asia/Taipei)**. Execution and independent saved-artifact audit completed.

Every condition contains the same48 utterances / 994 reference words / 30 speakers. Higher attacked source WER means more recognition errors. These are exposed-development source-only results.

| Condition | Source WER % | DNSMOS | NISQA | UTMOS |
|---|---:|---:|---:|---:|
| Clean | 4.0241 | 3.2721 | 4.0533 | 4.0259 |
| Reconstruction | 4.0241 | 3.3002 | 4.1716 | 4.0131 |
| Original FV quality control | 49.7988 | 3.1739 | 4.0698 | 3.5789 |
| Untrained waveform generator | 4.0241 | 3.2700 | 4.0557 | 4.0259 |
| Frozen trained waveform generator | 4.0241 | 3.2717 | 4.0533 | 4.0259 |
| Per-example waveform PGD, full R3 loss | 43.4608 | 3.1656 | 3.7776 | 3.8873 |
| Per-example waveform PGD, NLL only | 58.7525 | 3.0025 | 3.4367 | 3.5678 |

## Registered development screen

- Per-example waveform PGD, full R3 loss: -6.3380 WER points and +0.3084 UTMOS versus the FV quality control; screen **fails**. Descriptive paired speaker interval for source-WER difference: [-14.0171, 1.0763].
- Per-example waveform PGD, NLL only: +8.9537 WER points and -0.0111 UTMOS versus the FV quality control; screen **fails**. Descriptive paired speaker interval for source-WER difference: [1.0416, 17.0167].
- Frozen trained waveform generator: -45.7746 WER points and +0.4470 UTMOS versus the FV quality control; screen **fails**. Descriptive paired speaker interval for source-WER difference: [-51.6878, -41.0924].

## Mechanism evidence

Full-objective per-example PGD changes source WER by +39.4366 points relative to clean; NLL-only PGD changes it by +54.7284 points. The generator checkpoint and coefficients were frozen before this pilot. The per-example conditions separate a shared generator from direct waveform optimization; the loss pair isolates the penalty terms under this optimizer. No result proves a globally optimal attack or identifies all causes of amortized-training failure.

Generation took 1562.82 seconds, native source/MOS evaluation 466.82 seconds. All additional model passes are disclosed in saved traces. No lifetime-efficiency superiority is inferred from these timings.

No confirmation/final audio, sealed recognizers or human raters were used. A screen pass is exploratory and still needs mechanism/content validation and untouched evaluation. The bootstrap intervals describe only this exposed source cohort; the single learned seed does not estimate training variance.

[Protocol](07_pilot48_protocol.md) · [Machine-readable results](execution48/comparison.json) · [Independent audit](execution48/audit-pilot48.md) · [Live job status](JOB_STATUS.md)
