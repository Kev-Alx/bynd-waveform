# Feature/vocoder quality study

**Completed:** the raw prematched generator did not improve the joint frontier, and both normalized-input paths failed reconstruction-quality screening. The regular/raw sum-NLL validation frontier remains 51.9857%/3.5874, 69.3483%/3.3084 and 75.5601%/3.2457 (WER/UTMOS for LR 0.02/0.05/0.1). No new test was launched. See [QUALITY_OVERNIGHT.md](/workspace/docs/QUALITY_OVERNIGHT.md) for the analysis and three-study overnight continuation.

18 October 2026, Asia/Taipei. Current job directory: `/workspace/reproduction-runs/paper-quality-contracts`. Live overview: [JOB_STATUS.md](/workspace/reproduction-runs/JOB_STATUS.md). Discord notifications remain disabled.

## What the completed batch showed

The preceding refinement batch completed successfully at 14:40 Asia/Taipei. It produced 14 new pilots and two new validation evaluations; the third validation measurement reused an audited control. No candidate qualified for confirmation, and no additional 999-clip test was launched.

| Validation candidate | WER % | DNSMOS | NISQA | UTMOS |
|---|---:|---:|---:|---:|
| Previous quality control, sum NLL/LR 0.02 | 51.9857 | 3.1811 | 4.1038 | 3.5874 |
| Lowercase mean NLL/LR 0.05 | 66.3442 | 3.1174 | 4.0462 | 3.3306 |
| HF alpha 10, sum NLL/LR 0.1 | 76.0692 | 3.1088 | 4.0240 | 3.2454 |
| Paper anchors, unmatched protocol | 75.43 | 2.954 | 3.526 | 3.590 |

The lowercase pilot's 109.55% WER did not persist on validation. A short pilot can rank candidates poorly; the larger validation result governs the conclusion.

For original-case sum-NLL/LR 0.1 pilots, raising alpha from 1 to 10 reduced the mean last-update HF ratio from 0.3682 to 0.1101. Pilot UTMOS changed only from 3.2603 to 3.2836; validation UTMOS was 3.2457 versus 3.2454. This suggests that simply increasing the measured >6 kHz penalty does not resolve the observed quality gap in these runs. It does not isolate the causal source of audible degradation.

The paper explicitly fixes the HF cutoff at **6000 Hz** (section 5), so this batch retains it. The next hypothesis concerns the encoder preprocessing and generator, which the strict parity audit identified as consequential author-unspecified choices.

## Predeclared comparisons

Both available upstream generators accept WavLM layer-6 features and share the existing differentiable interface. The pinned [upstream README](/workspace/artifacts/sources/knn-vc-c616845c4e309e24d5927f15adbdf277a3d65358/README.md) describes regular and prematched training variants and provides the alternative checkpoint. A local download receipt records URL, byte count and SHA256; no published checksum is claimed. Neither generator is identified as the BeyondWaveform authors' checkpoint.

First audited result: prematched/raw sum-NLL/LR=0.02 produced **36.6834% attack WER**, DNSMOS/NISQA/UTMOS **2.9854 / 4.3044 / 3.3311** on the 12-clip pilot. Reconstruction was **4.0201% WER / 3.6987 UTMOS**, passing the predefined screen. Attack performance and UTMOS are worse than the cached regular/raw sum/LR=0.02 control, so this setting does not close the gap. The queue advanced to other settings. Its saved-output audit verified all 36 score rows, checkpoint identity and actual feature deltas (max relative radius 0.0781410); all 199 frozen inputs were rechecked. These pilot results do not establish validation or test performance.

Three new paths are compared with six cached regular/raw controls:

- Prematched generator, original waveform preprocessing.
- Regular generator, waveform layer normalization before WavLM.
- Prematched generator, waveform layer normalization before WavLM.

Normalization follows a configurable behavior already supported by the encoder wrapper. The WavLM checkpoint advertises normalization, while the selected kNN-VC extraction path uses raw waveforms. A normalized input is an experimental preprocessing choice; compatibility with generator training is not assumed. Strict state-dict loading checks the model interface, and reconstruction measurements check the resulting speech.

For each new path, six optimizer settings are frozen: mean/sum lexical NLL crossed with Adam LR 0.02/0.05/0.1. Thus the maximum is **18 new 12-clip pilots**, plus six cached controls. Each path first runs sum-NLL/LR 0.02. Its reconstruction must have WER at most five percentage points above clean WER and DNSMOS/NISQA/UTMOS at least 2.954/3.526/3.590. If it fails, its remaining five settings are skipped and the failed probe remains visible. This is an exploratory screen, not an author rule or proof of general compatibility.

All candidates retain the paper's rho=0.1, lambda=1, 50 updates and 6000 Hz cutoff. Alpha=1, zero initialization, original uppercase attack targets, feature layer, STFT, decoder, text normalization, scorer versions/checkpoints, input manifests and output encoding remain fixed. The only feature-path changes are the recorded generator variant and external waveform normalization. There is no nearest-neighbor feature replacement: the generator directly receives this utterance's `E(x)+delta`.

## Selection and evidence

The same pilot rule promotes quality-first, attack-first and balanced candidates to 100-clip validation. Validation qualifiers require WER >=73.43% and all three paper quality anchors. They repeat with seed 2027 before any one paired 999-clip test. The WER floor remains an exploratory target, not a statistical equivalence margin. No test outcome selects a replacement. Repeated historical test IDs are a paired benchmark rather than a fresh blind holdout.

Before freezing the study, the previous batch's 126 frozen inputs are checked. Its 20 reported score sets are reviewed by recomputing lexical edits, corpus WER, MOS means, coverage, trace finiteness and saved WAV hashes, and checking the existing independent delta-audit receipts. `prior-review.json` records this review, with old plan/results snapshots under `history/`. Completed source, configs, plans and evaluation outputs remain unchanged.

The new plan freezes the additional checkpoint/receipt, source, configs, previous evidence and inherited scorer/data artifacts. Each path has explicit feature metadata, so independent run verification compares against that path's actual generator and normalization. Every attack retains saved deltas, which are checked against freshly extracted features from the corresponding encoder preprocessing. Default attack and scoring implementations are unchanged. Cached controls receive separate `contracts-verification.json` audits.

All stages use `scripts/run_logged.py` with their own logs and job records. The persistent research job has `research.log`, `job.json`, `progress.json`, cumulative `results.json` and `report.md`. The live updater tracks it with `--research paper-quality-contracts`, refreshing every 30 seconds. Inspect job records/PIDs before launching another copy. The environment retains `PYTHONPATH=/workspace/artifacts/defense-runtime:/workspace`, four OMP/MKL threads, one OpenBLAS thread, UTF-8 settings, and `BEYONDWAVEFORM_NOTIFY_DISCORD=0`.
