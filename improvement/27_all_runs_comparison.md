# Every completed research condition

Compiled 2026-10-28 T13: 18: 08+08: 00 (Asia/Taipei). Covers the earlier audited starting point and all completed improvement experiments through the latest model run.

[Short overview](/workspace/improvement/26_research_summary.md) · [Data, runtime and evidence](/workspace/improvement/28_data_runtime_and_evidence.md)

## Read the tables

WER is source Whisper-small corpus word error rate; higher is the attack objective. DNSMOS, NISQA and UTMOS are automatic speech-quality estimates; higher is better. The fixed joint requirement is **WER>75%, DNSMOS≥3, NISQA≥3.75, UTMOS≥3.5**. “Missing gate” names every unmet requirement. A quality-passing clean recording is a control, not an attack success.

**Coverage: 78 scored conditions / 3192 scored recording-condition rows.** Reused controls and checkpoints remain visible; these are not 78 independent experiments or 3192 unique recordings. Main catalogue cohorts are D24(24 recordings/475 words) and D48(48 recordings/994 words). All later performance pilots use D48. Training pools T1/T10 and exposure limits are defined in the companion data file.

## Earlier baseline audit: ten complete runs

These runs predate the improvement experiments and were rechecked from saved evidence in the earlier chat. Three 24-point original settings also appear in the 78-condition catalogue below; do not add them as independent replications. The 999-point runs are historical test-clean results, not a fresh test of an improved method. Paper values are separate claims under an unmatched protocol.

| Audited run | N | Original setting / change | Attack WER% | DNSMOS | NISQA | UTMOS | Clean WER% | Reconstruction WER% |
|---|---|---|---|---|---|---|---|---|
| [historical-selected-999](/workspace/reproduction-runs/english-feature-test-999/surrogate-eval) | 999 | rho 0.5; lambda 0.1; mean NLL; LR 0.05 | 74.1064 | 2.9638 | 3.8711 | 2.7924 | 3.9636 | 3.7272 |
| [paper-budget-mean-999](/workspace/reproduction-runs/paper-quality-research/paper-setting-test-999/evaluation) | 999 | rho 0.1; lambda 1.0; mean NLL; LR 0.05 | 64.7415 | 3.1262 | 3.9542 | 3.3582 | 3.9636 | 3.7272 |
| [strong-sum-lr-0.1](/workspace/reproduction-runs/paper-quality-stable-execution/pilot/strong-sum-lr-0.1/evaluation) | 24 | rho 0.1; lambda 1.0; sum NLL; LR 0.1 | 74.7368 | 3.1640 | 4.0350 | 3.3186 | 3.5789 | 3.5789 |
| [quality-sum-lr-0.02](/workspace/reproduction-runs/paper-quality-stable-execution/pilot/quality-sum-lr-0.02/evaluation) | 24 | rho 0.1; lambda 1.0; sum NLL; LR 0.02 | 45.6842 | 3.2075 | 4.1245 | 3.6761 | 3.5789 | 3.5789 |
| [timestamp-sum-lr-0.05](/workspace/reproduction-runs/paper-quality-stable-execution/pilot/timestamp-sum-lr-0.05/evaluation) | 24 | rho 0.1; lambda 1.0; sum NLL; LR 0.05 | 71.7895 | 3.1273 | 4.0778 | 3.4034 | 3.5789 | 3.5789 |
| [without-tv](/workspace/reproduction-runs/paper-quality-stable-ablations/pilot/without-tv/evaluation) | 24 | rho 0.1; lambda 1.0; sum NLL; LR 0.1; without-tv | 76.0000 | 3.1530 | 4.0398 | 3.3226 | 3.5789 | 3.5789 |
| [without-hf](/workspace/reproduction-runs/paper-quality-stable-ablations/pilot/without-hf/evaluation) | 24 | rho 0.1; lambda 1.0; sum NLL; LR 0.1; without-hf | 74.7368 | 3.1351 | 4.0028 | 3.3058 | 3.5789 | 3.5789 |
| [without-perceptual](/workspace/reproduction-runs/paper-quality-stable-ablations/pilot/without-perceptual/evaluation) | 24 | rho 0.1; lambda 1.0; sum NLL; LR 0.1; without-perceptual | 78.7368 | 3.1564 | 4.0863 | 3.3504 | 3.5789 | 3.5789 |
| [time_energy](/workspace/reproduction-runs/paper-quality-coordinate-research/pilot/time_energy/evaluation) | 24 | rho 0.1; lambda 1.0; sum NLL; LR 0.1; time_energy optimizer coordinates | 71.7895 | 3.1436 | 4.0871 | 2.9045 | 3.5789 | 3.5789 |
| [channel_std](/workspace/reproduction-runs/paper-quality-coordinate-research/pilot/channel_std/evaluation) | 24 | rho 0.1; lambda 1.0; sum NLL; LR 0.1; channel_std optimizer coordinates | 74.1053 | 3.1864 | 4.0641 | 3.3989 | 3.5789 | 3.5789 |

Paper headline, for context only: WER 75.4300%, DNSMOS 2.9540, NISQA 3.5260, UTMOS 3.5900. The author subset/artifacts and several implementation choices were not recovered; those numbers cannot be treated as a matched local baseline. [Original-method audit](/workspace/improvement/01_original_method_and_baseline.md).

## Complete 78-condition development catalogue

### 1. Historical original control · LR 0.02

**Data:** D24: 24 recordings/475 reference words. Frozen original feature method; no attack-model training.

[Saved full-cohort scores](/workspace/reproduction-runs/paper-quality-stable-execution/pilot/quality-sum-lr-0.02/evaluation/summary.json).

| Condition | Change or control | WER% | DNSMOS | NISQA | UTMOS | Missing gate |
|---|---|---|---|---|---|---|
| clean | Unmodified speech; control | 3.5789 | 3.3323 | 4.0838 | 4.1886 | WER |
| reconstruction | Frozen WavLM6 → HiFi-GAN, no edit; control | 3.5789 | 3.3317 | 4.2124 | 4.1667 | WER |
| adversarial | Original FV50, sum NLL; LR 0.02 | 45.6842 | 3.2075 | 4.1245 | 3.6761 | WER |

### 2. Historical original control · LR 0.05

**Data:** D24: 24 recordings/475 reference words. Frozen original feature method; no attack-model training.

[Saved full-cohort scores](/workspace/reproduction-runs/paper-quality-stable-execution/pilot/timestamp-sum-lr-0.05/evaluation/summary.json).

| Condition | Change or control | WER% | DNSMOS | NISQA | UTMOS | Missing gate |
|---|---|---|---|---|---|---|
| clean | Unmodified speech; control | 3.5789 | 3.3323 | 4.0838 | 4.1886 | WER |
| reconstruction | Frozen WavLM6 → HiFi-GAN, no edit; control | 3.5789 | 3.3317 | 4.2124 | 4.1667 | WER |
| adversarial | Original FV50, sum NLL; LR 0.05 | 71.7895 | 3.1273 | 4.0778 | 3.4034 | WER, UTMOS |

### 3. Historical original control · LR 0.1

**Data:** D24: 24 recordings/475 reference words. Frozen original feature method; no attack-model training.

[Saved full-cohort scores](/workspace/reproduction-runs/paper-quality-stable-execution/pilot/strong-sum-lr-0.1/evaluation/summary.json).

| Condition | Change or control | WER% | DNSMOS | NISQA | UTMOS | Missing gate |
|---|---|---|---|---|---|---|
| clean | Unmodified speech; control | 3.5789 | 3.3323 | 4.0838 | 4.1886 | WER |
| reconstruction | Frozen WavLM6 → HiFi-GAN, no edit; control | 3.5789 | 3.3317 | 4.2124 | 4.1667 | WER |
| adversarial | Original FV50, sum NLL; LR 0.1 | 74.7368 | 3.1640 | 4.0350 | 3.3186 | WER, UTMOS |

### 4. Word-risk objective

**Data:** D24: 24 recordings/475 reference words. Feature search, 50 updates; same frozen encoder/decoder as original. No learned attack-model training.

[Saved full-cohort scores](/workspace/improvement/execution/r1-deletion-development24/source-summary.json).

| Condition | Change or control | WER% | DNSMOS | NISQA | UTMOS | Missing gate |
|---|---|---|---|---|---|---|
| adversarial | N-best word risk + one deletion candidate; feature search | 58.5263 | 3.1778 | 3.9957 | 3.3497 | WER, UTMOS |
| clean | Unmodified speech; control | 3.5789 | 3.3323 | 4.0838 | 4.1886 | WER |
| reconstruction | Frozen WavLM6 → HiFi-GAN, no edit; control | 3.5789 | 3.3317 | 4.2124 | 4.1667 | WER |

### 5. Local feature mixtures

**Data:** D24: 24 recordings/475 reference words. Feature search, 50 updates; neighboring frames within ±2, cosine floor0.95, global radius 0.1. No model training.

[Saved full-cohort scores](/workspace/improvement/execution/r2-canonical-zero-uniform-dev24/source-summary.json).

| Condition | Change or control | WER% | DNSMOS | NISQA | UTMOS | Missing gate |
|---|---|---|---|---|---|---|
| adversarial | Convex local feature mixtures with cosine≥0.95 | 3.3684 | 3.3320 | 4.2142 | 4.1661 | WER |
| baseline_attenuated | Original attack scaled to R2 residual norm; diagnostic | 4.8421 | 3.3020 | 4.1926 | 4.0837 | WER |
| baseline_frozen | Original FV50, LR 0.1; attack-strength control | 74.7368 | 3.1640 | 4.0350 | 3.3186 | WER, UTMOS |
| clean | Unmodified speech; control | 3.5789 | 3.3323 | 4.0838 | 4.1886 | WER |
| reconstruction | Frozen WavLM6 → HiFi-GAN, no edit; control | 3.5789 | 3.3317 | 4.2124 | 4.1667 | WER |

### 6. Learned waveform generator

**Data:** D24: 24 recordings/475 reference words. R3: 1.59 M waveform U-Net trained on T1; one seed, 1,779 updates / 7,116 microbatches. New audio uses its frozen predicted residual.

[Saved full-cohort scores](/workspace/improvement/execution/r3-waveform-autograd-repair/source-summary.json).

| Condition | Change or control | WER% | DNSMOS | NISQA | UTMOS | Missing gate |
|---|---|---|---|---|---|---|
| adversarial | Selected trained 1.59 M waveform U-Net | 3.5789 | 3.3318 | 4.0838 | 4.1886 | WER |
| clean | Unmodified speech; control | 3.5789 | 3.3323 | 4.0838 | 4.1886 | WER |
| initial-generator | Untrained waveform U-Net; control | 3.5789 | 3.3383 | 4.0808 | 4.1886 | WER |

### 7. Generator checkpoint · 500 updates

**Data:** D24: 24 recordings/475 reference words. Saved R3 checkpoint from the same T1 training run; no independent retraining.

[Saved full-cohort scores](/workspace/improvement/execution/r3-waveform-autograd-repair/checkpoints/update-00500-evaluation/native-source/source-summary.json).

| Condition | Change or control | WER% | DNSMOS | NISQA | UTMOS | Missing gate |
|---|---|---|---|---|---|---|
| adversarial | R3 waveform model at 500 updates | 3.5789 | 3.3322 | 4.0838 | 4.1885 | WER |

### 8. Generator checkpoint · 1000 updates

**Data:** D24: 24 recordings/475 reference words. Saved R3 checkpoint from the same T1 training run; no independent retraining.

[Saved full-cohort scores](/workspace/improvement/execution/r3-waveform-autograd-repair/checkpoints/update-01000-evaluation/native-source/source-summary.json).

| Condition | Change or control | WER% | DNSMOS | NISQA | UTMOS | Missing gate |
|---|---|---|---|---|---|---|
| adversarial | R3 waveform model at 1000 updates | 3.5789 | 3.3326 | 4.0838 | 4.1885 | WER |

### 9. Generator checkpoint · 1500 updates

**Data:** D24: 24 recordings/475 reference words. Saved R3 checkpoint from the same T1 training run; no independent retraining.

[Saved full-cohort scores](/workspace/improvement/execution/r3-waveform-autograd-repair/checkpoints/update-01500-evaluation/native-source/source-summary.json).

| Condition | Change or control | WER% | DNSMOS | NISQA | UTMOS | Missing gate |
|---|---|---|---|---|---|---|
| adversarial | R3 waveform model at 1500 updates | 3.5789 | 3.3318 | 4.0838 | 4.1886 | WER |

### 10. Waveform mechanism pilot

**Data:** D48: 48 recordings/994 reference words. Controlled per-recording waveform search versus the frozen R3 model and original FV. Waveform Linf/RMS≤0.003, 50 sign updates; no new model training.

[Saved full-cohort scores](/workspace/improvement/execution48/mechanism-pilot48/source-summary.json).

| Condition | Change or control | WER% | DNSMOS | NISQA | UTMOS | Missing gate |
|---|---|---|---|---|---|---|
| clean | Unmodified speech; control | 4.0241 | 3.2721 | 4.0533 | 4.0259 | WER |
| fv_quality | Original FV50, LR 0.02; quality control | 49.7988 | 3.1739 | 4.0698 | 3.5789 | WER |
| pgd_full | Waveform sign-PGD50, STFT/content penalties | 43.4608 | 3.1656 | 3.7776 | 3.8873 | WER |
| pgd_nll | Same waveform PGD50 with recognition loss only | 58.7525 | 3.0025 | 3.4367 | 3.5678 | WER, NISQA |
| r3_initial | Frozen untrained R3 waveform model | 4.0241 | 3.2700 | 4.0557 | 4.0259 | WER |
| r3_trained | Frozen trained R3 waveform model | 4.0241 | 3.2717 | 4.0533 | 4.0259 | WER |
| reconstruction | Frozen WavLM6 → HiFi-GAN, no edit; control | 4.0241 | 3.3002 | 4.1716 | 4.0131 | WER |

### 11. Waveform refinement

**Data:** D48: 48 recordings/994 reference words. Continue the same waveform initializer to100 updates; weighted loss versus hard proxy constraints. Original FV50/FV100 use LR 0.02. No attack-model training.

[Saved full-cohort scores](/workspace/improvement/execution_refine/pilot/source-summary.json).

| Condition | Change or control | WER% | DNSMOS | NISQA | UTMOS | Missing gate |
|---|---|---|---|---|---|---|
| clean | Unmodified speech; control | 4.0241 | 3.2721 | 4.0533 | 4.0259 | WER |
| constrained100 | 50 more projected/backtracked updates under distortion caps | 43.4608 | 3.1655 | 3.7771 | 3.8871 | WER |
| full50 | Waveform sign-PGD50, STFT/content penalties | 43.4608 | 3.1656 | 3.7776 | 3.8873 | WER |
| fv100 | Same original LR 0.02 trajectory, 100 updates | 65.4930 | 3.0888 | 4.0057 | 3.3105 | WER, UTMOS |
| fv50 | Original FV50, LR 0.02; quality control | 49.7988 | 3.1739 | 4.0698 | 3.5789 | WER |
| weighted100 | 50 more weighted waveform updates after full50 | 53.4205 | 3.1158 | 3.5937 | 3.8390 | WER, NISQA |

### 12. Quiet-region protection

**Data:** D48: 48 recordings/994 reference words. Per-recording waveform search with clean25 ms / 10 ms frame-energy bounds; primary NLL-only, weighted ablation and RMS-matched control. No model training.

[Saved full-cohort scores](/workspace/improvement/execution_envelope/pilot/source-summary.json).

| Condition | Change or control | WER% | DNSMOS | NISQA | UTMOS | Missing gate |
|---|---|---|---|---|---|---|
| attenuated_global_nll50 | Global NLL residual scaled to envelope residual RMS | 39.9396 | 3.1112 | 3.7017 | 3.7636 | WER, NISQA |
| clean | Unmodified speech; control | 4.0241 | 3.2721 | 4.0533 | 4.0259 | WER |
| envelope_full50 | Same local-energy budgets plus STFT/content penalties | 19.4165 | 3.2710 | 4.0650 | 4.0045 | WER |
| envelope_nll50 | Waveform PGD50 with clean-local-energy budgets | 29.1751 | 3.2752 | 4.0294 | 3.9140 | WER |
| fv50 | Original FV50, LR 0.02; quality control | 49.7988 | 3.1739 | 4.0698 | 3.5789 | WER |
| global_full50 | Reused weighted waveform PGD50 control | 43.4608 | 3.1656 | 3.7776 | 3.8873 | WER |
| global_nll50 | Reused NLL-only waveform PGD50 control | 58.7525 | 3.0025 | 3.4367 | 3.5678 | WER, NISQA |

### 13. Decoder qualification

**Data:** D48: 48 recordings/994 reference words. Reconstruction-only qualification before BigVGAN attacks; original controls retained. No newly trained decoder.

[Saved full-cohort scores](/workspace/improvement/execution_scaled/qualification/source-summary.json).

| Condition | Change or control | WER% | DNSMOS | NISQA | UTMOS | Missing gate |
|---|---|---|---|---|---|---|
| bigvgan_reconstruction | Qualified 22.05 kHz BigVGAN →16 kHz; no edit | 3.9235 | 3.2981 | 4.0521 | 3.9868 | WER |
| clean | Unmodified speech; control | 4.0241 | 3.2721 | 4.0533 | 4.0259 | WER |
| fv_quality50 | Reused original FV50, LR 0.02; quality control | 49.7988 | 3.1739 | 4.0698 | 3.5789 | WER |
| fv_reconstruction | Frozen original feature decoder; no edit | 4.0241 | 3.3002 | 4.1716 | 4.0131 | WER |

### 14. Larger models and longer feature runs

**Data:** D48: 48 recordings/994 reference words. Frozen WavLM6/HiFi-GAN; original small-only LR 0.1 or three-source ensemble. Ensemble weights 0.5/0.25/0.25, per-recording clean gradient calibration. All rho 0.1; full original TV/HF. No attack-model training.

[Saved full-cohort scores](/workspace/improvement/execution_scaled/fvpilot/source-summary.json).

| Condition | Change or control | WER% | DNSMOS | NISQA | UTMOS | Missing gate |
|---|---|---|---|---|---|---|
| clean | Unmodified speech; control | 4.0241 | 3.2721 | 4.0533 | 4.0259 | WER |
| ensemble_eot_200 | Small/large-v2/Data2 Vec FV; sampled transforms; 200 updates | 81.7907 | 3.1708 | 4.0672 | 3.2345 | UTMOS |
| ensemble_eot_50 | Small/large-v2/Data2 Vec FV; sampled transforms; 50 updates | 64.6881 | 3.1748 | 4.0921 | 3.3771 | WER, UTMOS |
| ensemble_eot_500 | Small/large-v2/Data2 Vec FV; sampled transforms; 500 updates | 89.5372 | 3.1476 | 4.0211 | 3.0850 | UTMOS |
| ensemble_raw_200 | Small/large-v2/Data2 Vec FV; identity only; 200 updates | 81.9920 | 3.0532 | 3.9517 | 3.1636 | UTMOS |
| ensemble_raw_50 | Small/large-v2/Data2 Vec FV; identity only; 50 updates | 68.3099 | 3.1270 | 4.0665 | 3.3671 | WER, UTMOS |
| ensemble_raw_500 | Small/large-v2/Data2 Vec FV; identity only; 500 updates | 90.4427 | 3.0997 | 3.9295 | 3.0557 | UTMOS |
| fv_quality50 | Reused original FV50, LR 0.02; quality control | 49.7988 | 3.1739 | 4.0698 | 3.5789 | WER |
| fv_reconstruction | Frozen original feature decoder; no edit | 4.0241 | 3.3002 | 4.1716 | 4.0131 | WER |
| single_150 | Original small-only FV, LR 0.1; 150 cumulative updates | 97.1831 | 3.0512 | 3.9500 | 3.0866 | UTMOS |
| single_1500 | Original small-only FV, LR 0.1; 1500 cumulative updates | 101.4085 | 3.0196 | 3.8260 | 2.9437 | UTMOS |
| single_200 | Original small-only FV, LR 0.1; 200 cumulative updates | 95.9759 | 3.0641 | 3.9187 | 3.0560 | UTMOS |
| single_50 | Original small-only FV, LR 0.1; 50 cumulative updates | 83.9034 | 3.1056 | 3.9766 | 3.1981 | UTMOS |
| single_500 | Original small-only FV, LR 0.1; 500 cumulative updates | 100.5030 | 3.0540 | 3.7788 | 2.9846 | UTMOS |
| single_600 | Original small-only FV, LR 0.1; 600 cumulative updates | 93.3602 | 3.0770 | 3.8614 | 2.9859 | UTMOS |

### 15. BigVGAN mel-space pilot

**Data:** D48: 48 recordings/994 reference words. Frozen BigVGAN log-mel decoder, three sources and transforms; LR 0.01, relative radius 0.1 plus log-mel box≤log2. Different norm units from FV, not a matched perceptual budget.

[Saved full-cohort scores](/workspace/improvement/execution_scaled/melpilot/source-summary.json).

| Condition | Change or control | WER% | DNSMOS | NISQA | UTMOS | Missing gate |
|---|---|---|---|---|---|---|
| bigvgan_reconstruction | Qualified 22.05 kHz BigVGAN →16 kHz; no edit | 3.9235 | 3.2981 | 4.0521 | 3.9868 | WER |
| clean | Unmodified speech; control | 4.0241 | 3.2721 | 4.0533 | 4.0259 | WER |
| mel_eot_200 | BigVGAN log-mel edits; three sources + transforms; 200 updates | 80.1811 | 2.8827 | 3.3091 | 2.2951 | DNSMOS, NISQA, UTMOS |
| mel_eot_50 | BigVGAN log-mel edits; three sources + transforms; 50 updates | 33.9034 | 3.1236 | 3.7549 | 3.2508 | WER, UTMOS |
| mel_eot_500 | BigVGAN log-mel edits; three sources + transforms; 500 updates | 78.7726 | 2.6038 | 2.7068 | 1.6528 | DNSMOS, NISQA, UTMOS |

### 16. Quality and efficiency pilot

**Data:** D48: 48 recordings/994 reference words. Matched decoder-graph implementation and one fixed 25% spectral repair. The model weights stay frozen; no training. Exact raw FV50 WAV parity was demonstrated.

[Saved full-cohort scores](/workspace/improvement/execution_tradeoff/pilot/source-summary.json).

| Condition | Change or control | WER% | DNSMOS | NISQA | UTMOS | Missing gate |
|---|---|---|---|---|---|---|
| raw50 | Original strong FV50 with retained decoder graph | 83.9034 | 3.1056 | 3.9766 | 3.1981 | UTMOS |
| stft_identity | Raw FV50 through STFT/inverse-STFT only | 83.9034 | 3.1056 | 3.9766 | 3.1981 | UTMOS |
| anchor25 | Raw FV50, 25% log-magnitude blend to reconstruction | 72.8370 | 3.1580 | 4.0272 | 3.3604 | WER, UTMOS |

### 17. Learned speech-feature model

**Data:** D48: 48 recordings/994 reference words. 5.07 M conditioned temporal feature policy trained directly on T10. Common300-update warmup, source 3,791 updates, quality 3,680 updates. No optimized attack-teacher targets; fresh WavLM encoding and one policy/decoder forward per new clip.

[Saved full-cohort scores](/workspace/improvement/execution_model_v1/pilot/source-summary.json).

| Condition | Change or control | WER% | DNSMOS | NISQA | UTMOS | Missing gate |
|---|---|---|---|---|---|---|
| initial_r050 | Untrained feature model, relative radius 0.05 | 4.3260 | 3.3025 | 4.1827 | 4.0177 | WER |
| source_r025 | Recognition-only feature model; radius 0.025 | 3.9235 | 3.2932 | 4.1106 | 3.8986 | WER |
| source_r050 | Recognition-only feature model; radius 0.050 | 4.0241 | 3.2665 | 4.0387 | 3.6630 | WER |
| source_r100 | Recognition-only feature model; radius 0.100 | 8.3501 | 3.2005 | 3.8915 | 2.7783 | WER, UTMOS |
| quality_r025 | Violation-only quality correction feature model; radius 0.025 | 3.9235 | 3.2890 | 4.2117 | 4.0072 | WER |
| quality_r050 | Violation-only quality correction feature model; radius 0.050 | 3.8229 | 3.2908 | 4.2198 | 3.9104 | WER |
| quality_r100 | Violation-only quality correction feature model; radius 0.100 | 4.1247 | 3.2308 | 4.2051 | 3.6825 | WER |
| quality_shuffled_r050 | Quality model conditioned on next clip, radius 0.05; diagnostic | 4.0241 | 3.2896 | 4.2158 | 3.9658 | WER |

## Attempts without a completed performance table

| Attempt / check | Data and execution | What happened | How to interpret it |
|---|---|---|---|
| Frozen-original replay | 3 duration-selected D24 recordings; 150 total updates | All saved losses, gradients, deltas, Adam states and final audio matched exactly | Execution equivalence on these controls; not a new attack result |
| R1 canonical word-risk | 3 calibration clips; 3 generation probes; then intended D24 | Full D24 s topped at first clip: all 4 beam hypotheses had correct normalized words, so every risk cost was zero | No completed 24-point performance/MOS result; 3-clip generation observations use another backend and are not a fair WER comparison |
| R1 logging-only wrapper | Same failed attempt, scientific code unchanged | Preserved clearer failure receipts | Not another scientific method or scored run |
| R2 geometry smoke | 3 duration-selected recordings | Execution checks completed before D24 | Engineering check; not48-point performance evidence |
| R3 cached-reference preflight | 3 duration-selected recordings | Strict clean features marginally exceeded the old cached absolute tolerance | Stopped before training; named strict-reference variant followed |
| R3 s trict-reference attempt | Gradient calibration before training | Upstream WavLM in-place operation broke autograd | Execution failure; no trained-model result |
| R3 value-preserving autograd repair | CPU upstream regressions and real GPU preflight | Cloned saved convolution input; forward values preserved; then completed R3 training | Repair lineage for the one completed R3 experiment |
| R3 s aved failure replay | Existing24 s aved outputs; no new model calls | Nearly zero residual; all normalized transcripts unchanged; diagnostic exit 1 | A recorded negative attack-signal finding, not a crashed benchmark |
| Latest model automatic selection | After all 384 pilot outputs were scored | Manifest reader rejected repeated recording IDs across8 variant timing rows; selection exited1 | Training/scoring remain complete; no frozen selection written; independent CPU review findszero eligible candidates |
| Conditional validation76 and new transfer/defenses | Registered only | No improved condition earned the WER/quality gate; latest selection also had the error above | Not run; absence must not be reported as a defense/transfer result |
| Untouched confirmation48, final200, human andphysical tests | Reserved/planned only | No improvement-cycle evaluation executed | No generalization, human-content, OTA or full-defense claim |

[Latest independent gate review](/workspace/improvement/research_summary/model_gate_review.json) · [Canonical R1 s top](/workspace/improvement/execution/r1-report.md) · [R3 execution lineage](/workspace/improvement/05_execution_report.md).

Rows and numerical gates were checked against saved summaries and per-recording edit/quality rows for this consolidation. No recognition, quality model or GPU experiment was rerun. Full precision remains in [all_results.csv](/workspace/improvement/all_results.csv) and the linked receipts.
