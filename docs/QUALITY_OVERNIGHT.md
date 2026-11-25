# Three overnight WER/quality experiments

**Completed at 01:46 Asia/Taipei on 19 October:** all three studies finished without runtime failures; no joint WER/quality qualifier advanced to a new test. Strongest new validation: 71.0285% WER / 3.3713 UTMOS. The [independent review](/workspace/reproduction-runs/overnight-review-20261019/report.md) passed all 24 score sets and explains a major cosine-pilot insertion outlier. The next study is [feature-delta diagnostics](/workspace/docs/QUALITY_DELTA_DIAGNOSTICS.md); details below describe the completed overnight plan.

18 October 2026, Asia/Taipei. The user requested two or three experiments queued for overnight execution. `scripts/run_overnight_research.py` runs three independent studies sequentially on the TITAN RTX, then shared validation and conditional confirmation/testing. Discord remains disabled.

Outputs: `/workspace/reproduction-runs/paper-quality-overnight/{job.json,progress.json,research.log,plan.json,results.json,report.md}`. The [live status](/workspace/reproduction-runs/JOB_STATUS.md) links to the queue. The report includes each batch's audited/failed/planned counts and every WER/MOS result.

## Analysis of the completed feature/vocoder batch

The previous batch completed successfully at **18:00 Asia/Taipei**, with eight new pilots, ten further settings skipped after reconstruction failures, and one new validation measurement alongside cached controls. All **17 reported score sets** were reviewed: coverage, waveform hashes, lexical errors, MOS means, finite traces and existing independent delta-audit receipts. All **199 frozen inputs** matched before changing source.

| Pilot path | Reconstruction WER % | Reconstruction UTMOS |
|---|---:|---:|
| Regular/raw | 6.0302 | 4.0548 |
| Prematched/raw | 4.0201 | 3.6987 |
| Regular/normalized | 6.0302 | 3.2117 |
| Prematched/normalized | 6.0302 | 2.8149 |

Normalized-input reconstruction lost substantial quality despite retaining readable content, and both normalized paths failed the predeclared screen. Of six raw-prematched attack settings, five had both lower WER and lower UTMOS than the corresponding regular/raw setting. The remaining setting improved WER at lower UTMOS. None closed the joint gap.

The regular/raw sum-NLL validation frontier remains **51.9857% WER / 3.5874 UTMOS** at LR 0.02, **69.3483% / 3.3084** at LR 0.05, and **75.5601% / 3.2457** at LR 0.1. Paper anchors are 75.43% WER and DNSMOS/NISQA/UTMOS 2.954/3.526/3.590 under an unmatched author protocol. No new 999-clip test was launched.

Pilot traces do not support a broad claim of optimizer instability: the median number of NLL-decreasing updates was zero for mean/LR 0.05 and sum/LR 0.02, and two out of 49 for sum/LR 0.1. The next update rules are exploratory alternatives, not a demonstrated correction for oscillation.

## Frozen overnight queue

All cases keep **rho=0.1, lambda=1, 50 updates and 6000 Hz cutoff**; original regular/raw layer-6 features, generator, spectral settings, decoder and scoring references/channels/checkpoints remain fixed. No nearest-neighbor matching, extra loss, quality-score optimization, or test-based selection is introduced.

| Study | Configurations | What changes |
|---|---:|---|
| Timestamp conditioning | 8 | A fixed 0.00-second timestamp conditions attack lexical NLL; mean/sum × LR 0.02/0.05/0.1, plus lowercase mean at LR 0.02/0.05 |
| Learning-rate schedules | 4 | Original sum-NLL Adam at initial LR 0.05/0.1, with cosine decay or a midpoint step to 10% of initial LR |
| Normalized feature-gradient updates | 6 | Mean/sum NLL with L2 projected descent step fractions 0.005/0.01/0.02 of the clean feature norm |

Two freshly generated sum-NLL constant-Adam controls at LR 0.02/0.1 bring the total to **20 pilot cases**. Each study is predeclared independently; results from earlier overnight groups do not redesign later groups. Cases run sequentially to avoid GPU contention.

The timestamp hypothesis comes from the strict audit: attack NLL used a no-timestamps prefix, while the installed Whisper decoder permits timestamps and initially requires a timestamp token. The installed `whisper/decoding.py` (including its timestamp rules), `transcribe.py` and `tokenizer.py` are frozen as primary implementation evidence. The new option inserts `timestamp_begin` after SOT/language/task and before lexical targets. The initial timestamp, prefix and EOT are unscored. A fixed 0.00 timestamp is not an oracle alignment; actual decoder start times can differ. Recognition decoding and supervised fine-tuning retain their prior behavior. Absent the new optional key, default attack behavior and metadata remain compatible.

Cosine LR varies from initial eta to 0.1 eta over updates 1–50. The two-stage rule uses initial eta for updates 1–25 and 0.1 eta for 26–50; Adam momentum buffers continue across the transition. L2 PGD minimizes the same complete negative-NLL-plus-perceptual objective using `delta -= eta*(||q||+eps)*g/(||g||+eps)`, followed by the same relative-ball projection. For that optimizer only, LR denotes the relative feature-step fraction. Zero gradients remain no-ops. Constant Adam/SGD retain their previous path. New schedules/updates record their effective LR at every step, checked during verification.

## Shared pilot, selection and execution

A deterministic sample (seed **20261018**) selects **24 fresh dev-clean clips**, excluding all prior pilot, validation and test IDs and audio hashes. No ASR or MOS result participates in sampling. The clips total 170.515 seconds, with mean 7.105 and range 1.94–20.47 seconds. All 20 settings share these IDs; old 12-clip scores are not mixed into the new promotion ranking. This larger fresh pilot reduces dependence on the repeatedly used 12 clips; it does not eliminate tuning uncertainty.

Up to three new candidates are selected by the existing quality-first, attack-first and balanced pilot rules, alongside one control. Candidates also need reconstruction WER <=clean+5 percentage points and all three reconstruction quality anchors. The selected settings receive the unchanged 100-clip validation pool. Joint targets remain WER >=73.43% and MOS >=2.954/3.526/3.590, plus the reconstruction gate. This is an exploratory target, not statistical equivalence or author parity. Qualifiers repeat with seed 2027 before one paired 999-clip test; no test outcome selects a replacement. The historical test pool is not a fresh blind holdout.

Every generation, scoring and verification phase has a supervised log/status file. Independent case failures are recorded while other cases continue. Any failures leave an overall failed status after processing and prevent automatic final testing. Frozen-input/runtime changes stop the queue rather than silently changing the experiment. Scientific negative results complete normally. Existing records resume under the standard integrity checks; immutable promotion receipts prevent silently selecting a different test configuration after a decision.

The previous plan/results/source snapshots and review are under `history/`. Whisper, attack and config sources were archived before adding optional policies. Older plans stay unchanged; the new plan maps their original hashes to preserved snapshots and freezes the current implementations separately. It contains **257 input hashes**, actual serialized generation/confirmation configs, all three disjoint pools and inherited scorer/model artifacts.

All **133 tests passed**, including token masking/alignment, real native-Whisper input gradients for both prefixes, unchanged decoding/default metadata, schedule endpoints and applied LRs, normalized-step radius/zero-gradient behavior, split isolation, artifact identity and continuation after an independent failure. Additional real pretrained one-clip checks exercise generation, recognition, saved WAVs, all 50 finite updates, schedule traces and independent delta radii for one case from each study; their scores do not select candidates.

Launch uses detached `run_logged.py`, UTF-8, `PYTHONPATH=/workspace/artifacts/defense-runtime:/workspace`, four OMP/MKL threads, one OpenBLAS thread and `BEYONDWAVEFORM_NOTIFY_DISCORD=0`. The status updater tracks `--research paper-quality-overnight` every 30 seconds. Inspect live job records and PIDs before starting another copy. The queue takes several hours; confirmation and a conditional 999-clip test can extend it.
