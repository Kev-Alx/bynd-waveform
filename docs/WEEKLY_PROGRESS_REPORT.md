# BeyondWaveform — weekly progress update

Prepared **21 October 2026**. Results through **20 October 2026, 23:04 Asia/Taipei**. Designed as a three-page update; the PDF is the page-count-controlled version.

## Page 1 — Defense tests

Available defense tests are complete. The 6 kHz filter and 8 kHz resampling only slightly reduced attack errors. Stronger 2 kHz resampling also broke recognition of clean speech.

WER is the percentage of transcription errors. For a defense, lower attacked WER is better, while clean WER should also stay low. “Reconstructed” means audio passed through the speech generator without an attack.

| Defense | Clean WER % | Reconstructed WER % | Attacked WER % | What it means |
| --- | --- | --- | --- | --- |
| No defense | 3.96 | 3.73 | 74.11 | Reference attack before any defense. |
| Low-pass filter, 6 kHz | 3.84 | 4.08 | 73.54 | Attack errors barely fell; clean speech stayed readable. |
| WaveGuard resampling, 8 kHz | 4.08 | 4.17 | 71.73 | Small reduction in attack errors; clean speech mostly preserved. |
| WaveGuard resampling, 2 kHz | 32.37 | 34.80 | 101.60 | Also damaged clean speech badly; unsuitable as a useful defense. |

All four rows use the same 999 utterances. WER can exceed 100% when the recognizer inserts extra words. The 2 kHz result therefore does not indicate a successful useful defense.

Paper attack WER: raw 75.43%, LPF 68.39%, WaveGuard 57.03%. These are reference values, not matched comparisons: our filtering settings and the authors’ full protocol differ.

### Defense coverage still missing

| Defense | What is missing |
| --- | --- |
| CE-AT, DT/DE-AT, DMW, TI | Missing defended model weights and/or a clear ASR training recipe. |
| AudioPure | Released setup targets one-second audio; the paper’s sentence-level setup is unknown. |
| PVP Vote | Method identity, model settings and voting rules are not resolved. |

These defense tests used the historical attack, with a larger change limit than the paper (rho 0.5 versus 0.1). WaveGuard here tests its resampling transformation, not the complete consistency detector. Post-defense sound quality was not measured.

Source: [defense suite](/workspace/reproduction-runs/english-defenses-999/report.md).

<div style="page-break-before: always;"></div>

## Page 2 — All experiment batches

Actual study names are shown below. The nine scored groups contain 80 unique attack/output result sets; reused controls are counted once. Other rows are diagnostics. Output sets are not counts of independent source audios or necessarily independent optimizations.

| Study (actual name) | Output sets / scope | Audios tested per condition | What we tested or learned |
| --- | --- | --- | --- |
| [paper-quality-research](/workspace/reproduction-runs/paper-quality-research/report.md) | 9 scored | 12 / 100 / 999 | Average/sum loss and step sizes. Fixed 999-clip control: 64.74% WER / 3.358 UTMOS. |
| [paper-quality-refinement](/workspace/reproduction-runs/paper-quality-refinement/report.md) | 16 scored | 12 / 100 | Text case, optimizer, frequency penalty and random starts. No joint target match. |
| [paper-quality-contracts](/workspace/reproduction-runs/paper-quality-contracts/report.md) | 9 scored | 12 / 100 | Vocoder and normalization changes hurt quality. Kept the original compatible pair. |
| [paper-quality-overnight](/workspace/reproduction-runs/paper-quality-overnight/report.md) | 23 scored | 24 / 100 | Timestamps, step schedules and normalized gradients. No joint validation match. |
| [feature-delta-diagnostics](/workspace/reproduction-runs/feature-delta-diagnostics/report.md) | 9 outputs | 24 | Smoothing saved changes weakened attacks without enough quality gain. Stopped. |
| [decoder-alignment-diagnostics](/workspace/reproduction-runs/decoder-alignment-diagnostics/report.md) | 2 prompts | 24 | Token filtering explained only 0.24% of the loss increase. No loss change justified. |
| [paper-quality-finalization](/workspace/reproduction-runs/paper-quality-finalization/report.md) | 6 outputs | 24 | Averaging final updates missed the required +0.05 UTMOS gain. Stopped. |
| [optimizer-mechanism-diagnostics-v2](/workspace/reproduction-runs/optimizer-mechanism-diagnostics-v2/report.md) | Derivative probes | 3 | Old GPU gradients varied. Quality penalties still changed Adam updates by 21–42%. |
| [deterministic-execution-validation](/workspace/reproduction-runs/deterministic-execution-validation/report.md) | 2 repeats | 3 | Strict execution matched all 150 paired updates and final audio exactly. |
| [paper-quality-stable-execution](/workspace/reproduction-runs/paper-quality-stable-execution/report.md) | 3 scored | 24 | Repeatable full control: 74.74% WER / 3.319 UTMOS. Smaller steps weakened attacks. |
| [hf-pooling-saved-audio-20261020](/workspace/reproduction-runs/hf-pooling-saved-audio-20261020/report.md) | 216 WAV records | 24 | Both high-frequency energy measures mostly fell. No demonstrated pooling fix. |
| [paper-quality-stable-ablations](/workspace/reproduction-runs/paper-quality-stable-ablations/report.md) | 3 scored | 24 | Removing both penalties: 78.74% WER / 3.350 UTMOS. Diagnostic; changes the method. |
| [encoder-execution-diagnostics](/workspace/reproduction-runs/encoder-execution-diagnostics/report.md) | 3 modes | 24 | Original mode matched cached features. Other modes changed them; kept original. |
| [paper-quality-feature-layers-v2](/workspace/reproduction-runs/paper-quality-feature-layers-v2/report.md) | 2 recon. sets | 24 | Layers 5 and 7 failed reconstruction quality checks. No attacks were run. |
| [paper-quality-coordinate-research](/workspace/reproduction-runs/paper-quality-coordinate-research/report.md) | 2 scored | 24 | Channel weights gained 0.080 UTMOS; time weights lost 0.414. Neither met all targets. |

Audios tested = source clips per condition, not all WAV variants or repeats added together. Multiple counts mean pilot / validation / fixed test: 12 or 24 / 100 / 999. For example, the HF study measured 216 WAV records from 24 source clips. Cohorts overlap across studies; do not sum this column. Each study name links to its detailed report in the Markdown version.

<div style="page-break-before: always;"></div>

## Page 3 — Full-scope tests and latest pilot

Attack strength and sound quality have not matched the paper together. Higher attacked WER means a stronger attack; higher DNSMOS, NISQA and UTMOS mean better estimated quality.

### Full-scope test — 999 audios per condition

| Run (actual name) | Audio type | Audios tested | WER % | DNSMOS | NISQA | UTMOS |
| --- | --- | --- | --- | --- | --- | --- |
| Paper reference | Attacked | 999 | 75.43 | 2.954 | 3.526 | 3.590 |
| english-feature-test-999 | Clean | 999 | 3.96 | 3.264 | 4.029 | 4.085 |
| english-feature-test-999 | Reconstructed | 999 | 3.73 | 3.272 | 4.142 | 4.028 |
| english-feature-test-999 | Attacked | 999 | 74.11 | 2.964 | 3.871 | 2.792 |
| paper-setting-test-999 | Clean | 999 | 3.96 | 3.264 | 4.029 | 4.085 |
| paper-setting-test-999 | Reconstructed | 999 | 3.73 | 3.272 | 4.142 | 4.028 |
| paper-setting-test-999 | Attacked | 999 | 64.74 | 3.126 | 3.954 | 3.358 |

Both runs use the same 999 source clips (20,310 reference words), with no defense. english-feature-test-999 used rho=0.5 / lambda=0.1; paper-setting-test-999 used the paper budget, rho=0.1 / lambda=1, mean loss and LR=0.05. The latter is inside paper-quality-research. The authors’ exact subset/scorer recipe remains unresolved.

### Latest pilot — actual configuration names

Latest pilot: all eight settings reuse the same 24 source clips. These are development results, not final 999-clip test results.

| Configuration | Audios tested | WER % | DNSMOS | NISQA | UTMOS |
| --- | --- | --- | --- | --- | --- |
| strong-sum-lr-0.1 | 24 | 74.74 | 3.164 | 4.035 | 3.319 |
| quality-sum-lr-0.02 | 24 | 45.68 | 3.207 | 4.125 | 3.676 |
| timestamp-sum-lr-0.05 | 24 | 71.79 | 3.127 | 4.078 | 3.403 |
| without-tv | 24 | 76.00 | 3.153 | 4.040 | 3.323 |
| without-hf | 24 | 74.74 | 3.135 | 4.003 | 3.306 |
| without-perceptual | 24 | 78.74 | 3.156 | 4.086 | 3.350 |
| time_energy | 24 | 71.79 | 3.144 | 4.087 | 2.904 |
| channel_std | 24 | 74.11 | 3.186 | 4.064 | 3.399 |

without-tv removes smoothness, without-hf removes the frequency penalty, and without-perceptual removes both. These diagnostic ablations change the required method.

channel_std gained 0.080 UTMOS with 0.63 points less WER than strong-sum-lr-0.1. It still misses the paper’s 3.590 UTMOS. time_energy worsened quality.

### Next steps

Proposed next steps; no new experiment is launched by this report:

1. Independently audit channel_std and paired changes for each clip.
2. If supported, freeze a 48-clip confirmation protocol before considering another 999-clip test.
3. Resolve author model/scorer details and the six missing defense recipes.

Quality scores are automatic estimates, not human listening results. The fresh 48-clip confirmation cohort remains unscored; numeric matches alone would not establish author-code parity.

Sources: [historical 999-clip test](/workspace/reproduction-runs/english-feature-test-999/report.md), [paper-budget 999-clip summary](/workspace/reproduction-runs/paper-quality-research/paper-setting-test-999/evaluation/summary.json), [repeatable controls](/workspace/reproduction-runs/paper-quality-stable-execution/report.md), [loss ablations](/workspace/reproduction-runs/paper-quality-stable-ablations/report.md), [weighted coordinates](/workspace/reproduction-runs/paper-quality-coordinate-research/report.md). Full context: [thread summary](/workspace/docs/THREAD_SUMMARY_AND_PLAN.md).
