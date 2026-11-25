# Data, computation and evidence limits

[Overview](/workspace/improvement/26_research_summary.md) · [Every scored condition](/workspace/improvement/27_all_runs_comparison.md)

## 1. Data used and data kept untouched

All improvement experiments use English LibriSpeech speech. They do not establish Mandarin, physical-playback or human-listening performance. Dataset rows below are pools, not independent experiments. Training audio was acquired and prepared during the earlier research; the opening baseline audit predates that acquisition.

| Pool / manifest | Recordings | Speakers | Audio seconds | Use and exposure |
|---|---|---|---|---|
| [D24: original development](/workspace/reproduction-runs/paper-quality-overnight/data/pilot-24.jsonl) | 24 | 18 | 170.515 | Early R1/R2/R3 and original controls; repeatedly exposed |
| [Historical test999](/workspace/data/manifests/english-test-999.jsonl) | 999 | 40 | 7402.945 | Earlier baseline/transfer/defense results; already exposed, not a fresh test |
| [Historical validation100](/workspace/data/manifests/english-validation.jsonl) | 100 | 36 | 714.400 | Earlier development; 24 recordings later added to D48 |
| [Legacy pilot20](/workspace/data/manifests/pilot.jsonl) | 20 | 18 | 80.765 | Earlier exploration; two recordings overlap validation100 |
| [Legacy reconstruction12](/workspace/data/manifests/reconstruction-pilot.jsonl) | 12 | 12 | 62.625 | Earlier encoder/decoder checks |
| [Untouched confirmation48](/workspace/reproduction-runs/paper-quality-finalization/data/confirmation/confirmation-48.jsonl) | 48 | 28 | 319.725 | Reserved; no improvement-cycle inference; speakers overlap development |
| [D48: main pilot](/workspace/improvement/execution48/development-48.jsonl) | 48 | 30 | 360.010 | 24 old D24 +24 added historical-validation recordings; every later performance pilot |
| [Exposed validation76](/workspace/improvement/execution_refine/validation-76.jsonl) | 76 | 34 | 524.905 | Remaining validation100 recordings; prepared, never evaluated for these methods |
| [T1: train-clean-100](/workspace/improvement/execution/data/train-1h.jsonl) | 286 | 251 | 3609.175 | 286 recordings; R3 waveform-model training |
| [T10: train-clean-100](/workspace/improvement/execution/data/train-10h.jsonl) | 2807 | 251 | 36013.160 | 2,807 recordings; new feature-model training; T1 is nested inside |
| [Untouched final200: test-other](/workspace/improvement/execution/data/final-reserve-200.jsonl) | 200 | 33 | 1320.930 | Prospectively frozen harder-speech reserve; no improvement-cycle inference |

D24 has **475 reference words**; D48 has **994**; historical test999 has **20,310**. New-model timing uses the same D48 total of **360.010 audio seconds**. T1/T10 contain about **1.003/10.004 hours** of speech, not 1/10 hours of training compute.

| Boundary | What the evidence permits |
|---|---|
| D48 versus validation76 | No recording/audio-hash/normalized-text overlap; 26 shared speakers. Validation76 is exposed development, not a fresh or speaker-disjoint test. |
| Training versus evaluation | Prepared training excludes the historical/final speakers and duplicate audio/text; the new model registration checked training IDs and speakers against D48/validation76. |
| Confirmation48 | Unscored new utterances; all 28 speakers appear in historical development. It cannot establish unseen-speaker generalization. |
| Final200 | 200 test-other recordings / 33 speakers, selected to exclude historical/training speakers and duplicate audio/text. Its distribution is harder than test-clean. |
| Metadata versus outcomes | Pool sizes/overlap use saved manifests and receipts. This consolidation did not open protected audio or infer protected outcomes. |
| Repeated development | All candidate development scores are exposed. Repeated tuning and one-seed model training limit generalization/variance claims. |

[Data and evaluation rules](/workspace/improvement/data_and_evaluation_constraints.md) · [Validation76 construction](/workspace/improvement/execution_refine/validation76-data-note.md).

## 2. Original method versus changed methods

| Method family | Original behavior | Change tested | Training data / new-recording cost |
|---|---|---|---|
| Original FV | WavLM6 → per-recording feature edit → HiFi-GAN; frozen Whisper-small gradients | Quality/strong learning rates and longer trajectories | No attack-model training; repeat optimization for every new recording |
| Earlier loss/optimizer studies | Full original TV/HF, uniform feature coordinates | Remove TV, HF or both; reweight optimizer coordinates by frame energy or channel standard deviation | D24; no training; physical feature-ball projection retained; no joint pass |
| Decoded word risk | Teacher-forced transcript NLL | Expected bounded word error over 4 decoded paths; append a deletion candidate in one declared repair | No training; beam refresh/path scoring adds search cost |
| Local feature geometry | Free feature tensor within a global ball | Convex neighboring-frame mixtures, ±2 frames/cosine≥0.95 | No training; useful feature directions became scarce |
| Waveform U-Net | Per-recording feature optimization | 1,590,833 trainable parameters predicting a waveform residual; STFT+WavLM content losses | T1; 1,779 updates/7,116 microbatches; fixed waveform Linf/RMS 0.003 |
| Waveform PGD / continuation | Feature-space Adam with original TV/HF | Waveform sign updates, step 0.00012, 50 updates; full penalties versus NLL-only; later 100-step/capped variants | No attack-model training; weighted penalties 65.0837(STFT)/186.6123(content) |
| Clean-energy budgets | Uniform global waveform bound | 25 ms frames / 10 ms hops; perturbation RMS≤clean-frame RMS×10^(-30/20), capped at 0.003 | No training; protects declared frame energy, not human words |
| Larger-source ensemble | Whisper-small only | Whisper-small/large-v2/Data2 Vec weights 0.5/0.25/0.25; clean-gradient norm calibration | No attack-model training; 3 source gradients/update; optional identity/FIR 6 k/resample 8 k draws |
| BigVGAN log-mel | Feature-conditioned HiFi-GAN | Separately qualified frozen BigVGAN; mel LR 0.01, relative radius 0.1, log-mel box≤log2 | No newly trained vocoder; representation/budget units differ from original FV |
| Graph reuse / spectrum repair | Two decoder renders/update in the expanded implementation | Retain one decoder graph; then test a separate fixed 25% log-magnitude blend | No learned model; new recordings still pay the FV50 s earch |
| Direct-trained feature policy | Individually optimized feature residual | 5,070,464 parameters; width 384, eight temporal blocks, global context; fixed radii 0.025/0.05/0.1 | T10; no optimized attack-teacher targets; fresh encoder  + one policy/HiFi-GAN forward at inference |
| Feature-model quality arm | Source-only policy gradient | Activate normalized spectral/envelope violation correction above fixed proxy caps; project conflicting gradient component | Same warmup/data/RNG rules; 4 microbatches/update; optimizer AdamW 2 e-4, clip 5; completed counts differ |

Fixed quality-proxy caps for the feature policy: spectral **0.5987145876**, envelope **0.06727352284**, calibrated from all 48 original quality-control recordings against neutral reconstruction. These are proxy budgets, not MOS guarantees. Training transformations are differentiable approximations; they are not exact evaluation LPF/WaveGuard implementations.

## 3. Measured per-recording generation cost

Use timing scopes carefully. Historical rows below measure optimizer regions, excluding loading, feature preparation and scoring. The graph-reuse comparison used identical resident models and cached features in one session. Learned-model rows include input/output I/O and fresh WavLM encoding, but exclude one-time loading, hashing and shared diagnostic/reference work. These are timings of different-quality outputs; they are not an equivalent-success speed contest.

| Condition | Timing scope | Mean sec/clip | Sec/audio sec | 48-clip hours | WER% | UTMOS |
|---|---|---|---|---|---|---|
| ensemble_eot_50 | Cumulative search | 71.990 | 9.598 | 0.960 | 64.69 | 3.377 |
| ensemble_eot_200 | Cumulative search | 287.942 | 38.391 | 3.839 | 81.79 | 3.235 |
| ensemble_eot_500 | Cumulative search | 719.749 | 95.964 | 9.597 | 89.54 | 3.085 |
| ensemble_raw_50 | Cumulative search | 71.944 | 9.592 | 0.959 | 68.31 | 3.367 |
| ensemble_raw_200 | Cumulative search | 287.656 | 38.353 | 3.835 | 81.99 | 3.164 |
| ensemble_raw_500 | Cumulative search | 718.995 | 95.863 | 9.587 | 90.44 | 3.056 |
| single_50 | Cumulative search | 19.606 | 2.614 | 0.261 | 83.90 | 3.198 |
| single_150 | Cumulative search | 58.920 | 7.856 | 0.786 | 97.18 | 3.087 |
| single_200 | Cumulative search | 78.641 | 10.485 | 1.049 | 95.98 | 3.056 |
| single_500 | Cumulative search | 196.046 | 26.139 | 2.614 | 100.50 | 2.985 |
| single_600 | Cumulative search | 235.371 | 31.382 | 3.138 | 93.36 | 2.986 |
| single_1500 | Cumulative search | 587.729 | 78.362 | 7.836 | 101.41 | 2.944 |
| mel_eot_50 | Cumulative search | 137.035 | 18.271 | 1.827 | 33.90 | 3.251 |
| mel_eot_200 | Cumulative search | 547.108 | 72.946 | 7.295 | 80.18 | 2.295 |
| mel_eot_500 | Cumulative search | 1367.350 | 182.308 | 18.231 | 78.77 | 1.653 |
| Matched FV50: twice-render | Search, same session | 19.649 | 2.620 | 0.262 | 83.90 | 3.198 |
| Matched FV50: graph reuse | Search, same session | 16.774 | 2.236 | 0.224 | 83.90 | 3.198 |
| initial_r050 | Fresh encoding +generation +I/O | 0.142 | 0.0190 | 0.00190 | 4.33 | 4.018 |
| source_r025 | Fresh encoding +generation +I/O | 0.148 | 0.0197 | 0.00197 | 3.92 | 3.899 |
| source_r050 | Fresh encoding +generation +I/O | 0.141 | 0.0188 | 0.00188 | 4.02 | 3.663 |
| source_r100 | Fresh encoding +generation +I/O | 0.141 | 0.0187 | 0.00187 | 8.35 | 2.778 |
| quality_r025 | Fresh encoding +generation +I/O | 0.139 | 0.0186 | 0.00186 | 3.92 | 4.007 |
| quality_r050 | Fresh encoding +generation +I/O | 0.139 | 0.0186 | 0.00186 | 3.82 | 3.910 |
| quality_r100 | Fresh encoding +generation +I/O | 0.139 | 0.0185 | 0.00185 | 4.12 | 3.682 |
| quality_shuffled_r050 | Fresh encoding +generation +I/O | 0.140 | 0.0186 | 0.00186 | 4.02 | 3.966 |

Expanded checkpoints are cumulative snapshots of four underlying trajectories. **Do not add the 50/200/500 checkpoint times** as separate experiments. The independent original 1500, raw-ensemble 500, transformed-ensemble 500 and mel 500 trajectories account for the expanded optimization workload. Learned per-condition48-clip totals are attributed inference sums; the actual eight-condition process shares encoding/diagnostics and has its own wall time below.

**Exact-output implementation result:** graph reuse reduced measured FV50 latency **19.649 → 16.774 seconds/recording**, a **14.63%** reduction (paired recording-bootstrap 95% interval 12.88–16.19%). Peak allocated memory increased **6.885 →6.928 GiB**. All48 raw WAVs matched the frozen outputs exactly. The fixed spectral repair then added **0.563 seconds total**, but changed WER **83.9034 → 72.8370%** and UTMOS **3.1981 → 3.3604**. Neither output meets the joint requirement.

## 4. Experiment-level generation, scoring and memory

“Generation” includes whatever setup/diagnostics its runner timed; it is not necessarily the optimizer-loop sum above. Evaluation includes both recognition and CPU quality scoring. Memory uses binaryGiB; “—” means unavailable in this table, not zero. No new attack-model training occurred in the non-learned rows.

| Experiment | Generation min | Source/MOS min | Peak gen GiB | Peak eval GiB | Scope |
|---|---|---|---|---|---|
| Waveform mechanism48 | 26.05 | 7.78 | 9.789 | 1.128 | Saved runner wall timers |
| Waveform refinement48 | 68.90 | 6.80 | 9.775 | 1.098 | Saved runner wall timers |
| Quiet-region48 | 14.97 | 7.87 | 4.340 | 1.128 | Saved runner wall timers |
| BigVGAN reconstruction qualification48 | 0.54 | 4.45 | 1.285 | 1.098 | Saved runner wall timers |
| Expanded FV, all checkpoints | 1623.94 | 17.52 | 13.121 | 1.131 | Saved runner wall timers |
| BigVGAN mel, all checkpoints | 1095.92 | 5.70 | 11.117 | 1.129 | Saved runner wall timers |
| Graph/repair paired pilot | 29.54 | 12.74 | 6.928 | 1.129 | 42.40 min whole paired cycle; two generation paths |
| Learned feature model, eight conditions | 0.98 | 13.23 | 2.113 | 1.098 | Training/preprocessing excluded here; listed next |

The older R3 training region was **3541.89 seconds**; its entire detached training/checkpoint-evaluation process was **3661.90 seconds**. The initial development/R3 s cheduled subtotal was **1.563 hours**; with the first48-point mechanism generation/evaluation it was **2.128 hours**. The early ledger explicitly omits some startup/failed-log timings, so it is not an exact grand total. [Early compute ledger](/workspace/improvement/execution_refine/compute-ledger.md).

The expanded queue finished normally in **45 h51 m**, under its then-declared48-hour ceiling. It performed **144,000 per-recording optimization updates /288,000 source forward-backward pairs**, not large-model training. Its feature and mel generation took **27.07/18.27 hours**. Adding model diversity/steps did not resolve quality. The later 13-hour instruction applied to the new learned-model queue; it does not describe the earlier expanded run.

## 5. Latest learned model: actual phase accounting

| Phase | Wall seconds | CUDA-stream seconds | Allocated GiB | Reserved GiB | Coverage |
|---|---|---|---|---|---|
| Preprocessing | 728.94 | 350.08 | 1.861 | 3.869 | 2,807 clean feature/neutral-reference caches; not attack teachers |
| Common warmup | 489.33 | 474.67 | 5.827 | 7.279 | 300 updates /1,200 microbatches |
| Source-only arm | 7013.78 | 6856.09 | 5.826 | 7.986 | 3,791 updates /15,164 microbatches; update cap |
| Quality-corrected arm | 7172.69 | 7016.17 | 5.826 | 7.986 | 3,680 updates /14,720 microbatches; wall cap |
| Pilot generation, 384 outputs | 58.98 | Not separately totaled | 2.113 | 3.096 | 5.26 s loading; actualprocess shares encoding/diagnostics |
| Recognition inside pilot evaluation | 341.75 | 341.74 | — | — | Native beam 5, temperature 0 |
| Quality inside pilot evaluation | 443.83 | 443.82 | — | — | CPU DNSMOS/NISQA/UTMOS; stream events include host gaps |
| Full pilot evaluation | 793.73 | Nested times above | 1.098 | 1.490 | Includes 4.48 s loading and overhead; do not add nested rows again |

| Overall accounting | Measured result |
|---|---|
| Optimized teacher-target generation | 0 seconds; this method trained directly through the source |
| Preprocessing + all training arms | 4.279 hours; includes comparison-arm cost |
| Queue wait | 1.947 hours; not active GPU allocation |
| GPU-phase allocation wall | 4.529 hours; includes loading/I/O/CPU intervals while this queue owned its GPU phase |
| Whole detached queue | 6.498 hours including wait and delivery; 13-hour maximum |
| Execution/delivery | Training and scoring complete; selection exit 1; report/build/browser delivery complete |
| Exact kernel-active GPU time /energy | Not measured. Allocation wall and CUDA-stream elapsed intervals are not these quantities. |

**Selection error:** the timing JSONL has 384 valid `(recording, variant)` rows but only 48 distinct recording IDs. A reader meant for unique-ID utterance manifests rejected it. We preserved the failed exit and independently checked all saved WER/MOS/timing rows in [model_gate_review.json](/workspace/improvement/research_summary/model_gate_review.json). Zero candidates qualify, so fixing that reader alone would not change the scientific conclusion.

**Comparison limit:** both arms start from the same warmup and use the same data/radius/augmentation schedule, but the quality arm completed 111 fewer updates. At radius 0.1, quality correction raises UTMOS **2.7783 → 3.6825**, while WER falls **8.3501 → 4.1247%**. This is a measured tradeoff in two time-bounded runs, not a clean equal-update estimate or proof of the failure mechanism. The untrained and shuffled controls also have near-clean WER.

## 6. Forecasts and efficiency targets

| Forecast /target | Basis and value | What it does not establish |
|---|---|---|
| Graph-reuse target | Proposed ≥10% lower matched loop time and <20 GiB allocated; met | It was a research target, not a user latency limit; quality remained failed |
| Learned-model target | Proposed ≥10× generation reduction versus 16.774 s FV50, and <20 GiB allocated; measured output-generation timings/memory meet it | The model fails attack WER; this is not a faster effective attack |
| 1,000 new FV50+repair recordings | 4.715 hours using measured generation/preprocessing/repair/I/O assumptions | Conditional same-distribution forecast; includes per-recording search, excludes formal evaluation |
| Teacher corpus, prepared T1/T10 | Length-aware earlier forecast 1.578/15.577 hours of original-search target generation | Not performed for this new model; count-only/duration-only sensitivities differed |
| 1,000 new learned quality_r050 outputs | 0.0387 inference hours; 2.370 hours including warmup+quality-arm training/preprocessing | Illustrative failed-model output cost; no model was selected; excludes full research/evaluation overhead |
| 10,000 new learned quality_r050 outputs | 0.3874 inference hours; 2.718 hours including that training/preprocessing | Same hardware/audio-duration distribution; not completed larger-audio inference |
| Arithmetic break-even | 504.4 recordings versus 16.774 s reference, using displayed quality model | Both reference/model fail the joint requirements in different ways; not a valid deployment crossover |

The learned pipeline first measured **0.365 seconds/training microbatch** on short/median/long inputs and forecast costs before admitting the long stages. The profile forecast preprocessing 647.29 s; actual 728.94 s. Warmup forecast 438.27 s; actual 489.33 s. Forecast two-arm training 11076.63 s before margins; actual 14186.47 s. Margins/time caps mattered; the quality arm stopped short of the common update limit. No convergence or success benefit follows from the larger-data arithmetic below.

| Hypothetical training data | Preprocessing forecast, hours | One full training pass compute forecast, hours | Evidence status |
|---|---|---|---|
| 10 audio hours | 0.180 | 0.285 | Three-input engineering-profile extrapolation; 50/100-hour corpora not run |
| 50 audio hours | 0.899 | 1.423 | Three-input engineering-profile extrapolation; 50/100-hour corpora not run |
| 100 audio hours | 1.797 | 2.847 | Three-input engineering-profile extrapolation; 50/100-hour corpora not run |

These profile-based larger-data forecasts exclude training-loading/optimizer overhead and assume the same duration distribution; actual repeated training demonstrates that the profile estimate was optimistic. Preserve that error rather than presenting the forecasts as measured corpus throughput. [Profile/admission receipts](/workspace/improvement/execution_model_v1/profile.json) · [Previous prepared-data forecast](/workspace/improvement/execution_tradeoff/prepared_data_forecast.json).

## 7. Transfer and defense results: historical versus new

Historical evidence below belongs to the earlier **999-recording, rho 0.5/lambda 0.1/LR 0.05/50-step** attack. Its source quality was **DNSMOS 2.9638 / NISQA 3.8711 / UTMOS 2.7924**, failing current requirements. These saved results were carried into the research audit; they were not produced by the new methods and do not establish their transfer.

| Historical target, N=999 | Clean WER% | Reconstruction WER% | Attack WER% |
|---|---|---|---|
| Whisper-tiny | 7.6514 | 7.5234 | 67.8976 |
| Whisper-base | 5.5687 | 5.6573 | 68.7100 |
| Whisper-medium | 3.0182 | 3.0773 | 62.4274 |
| Whisper-large-v2 | 4.1408 | 3.4318 | 65.5391 |
| Wav2 Vec2-large CTC | 1.9301 | 1.9941 | 44.5446 |
| HuBERT-large CTC | 2.1664 | 2.2550 | 48.2078 |

[Historical Whisper transfer](/workspace/reproduction-runs/english-transfer-999/report.md) · [Historical CTC transfer](/workspace/reproduction-runs/english-ctc-transfer-999/report.md).

| Historical preprocessing, N=999 | Clean WER% | Reconstruction WER% | Attack WER% | Limit |
|---|---|---|---|---|
| Raw | 3.963565 | 3.727228 | 74.106352 | Source recognizer only |
| LPF6 kHz | 3.835549 | 4.076809 | 73.540128 | Independent named low-pass implementation |
| WaveGuard down/up8 kHz | 4.081733 | 4.165436 | 71.733136 | Resampling/PCM16; not full detector/AUC |
| WaveGuard down/up2 kHz | 32.373215 | 34.800591 | 101.600197 | Strongly damages clean speech; high attack WER needs that context |

[Historical defense evidence](/workspace/reproduction-runs/english-defenses-999/report.md). Post-defense MOS was not measured in that historical table.

| New-method endpoint | Actual status |
|---|---|
| Validation76 | Prepared protocol; no improvement method advanced to it |
| New four-target transfer | Not run: Whisper-base/medium, Wav2 Vec2-large, HuBERT-large were reserved for gated evaluation |
| New exact defenses | Not run: raw, LPF6 k, WaveGuard8 k, withheldLPF4 k/WaveGuard12 k |
| CE-AT, DT/DE-AT, DMW, TI | Matched defended weights/recipes unavailable |
| AudioPure/PVP sentence setting | Original comparable sentence-ASR configurations/artifacts not recovered |
| Human intelligibility /physical playback | Not measured |
| Novelty /publication | Literature informed experiments; publication planning deferred; no contribution or acceptance claim |

## 8. Literature and other deliverables

| Work | What it contributed | Saved source |
|---|---|---|
| Original-paper/source audit | Verified supplied paper identity, independent implementation contracts and missing author details | [Original audit](/workspace/improvement/01_original_method_and_baseline.md) |
| Earlier primary-literature review | Compared latent/diffusion, semantic/prosodic, transfer/psychoacoustic and codec precedents; motivated R1/R2/R3 | [Literature review](/workspace/improvement/03_literature_review.md) |
| Quality/runtime source audit | Duplicate decoding, NLL/WER divergence, prior R3 collapse and costs motivated the paired test | [Source audit](/workspace/improvement/19_source_audit.md) |
| Consensus+author-code review | Separated conditioned FAPG/SpeechAdvGAN from universal UAPG/AudioShield; DDSP and codec ideas remained alternatives | [Model directions](/workspace/improvement/23_model_directions_and_sources.md) |
| Publication-positioning note | An intermediate document; later user direction explicitly deferred publication planning | [Historical note](/workspace/improvement/22_publication_positioning.md) |
| Audio comparison | 78 conditions, 3,192 score rows, 48 sentences, 2,082 deduplicated original-WAV assets; listen/results/runtime/model views | http://localhost: 8888/ |
| Interface verification | Latest model report/build and browser checks completed; existing server retained | [Browser receipt](/workspace/improvement/execution_model_v1/browser_receipt.json) |
| Reproducibility | Frozen plans, raw scores/WAVs, hashes, detached logs/heartbeats/exit receipts; no original experiment overwritten | [Experiment handoff](/workspace/improvement/HANDOFF.md) |

Playback uses the saved 16 kHz float WAVs without loudness normalization, trimming or lossy conversion. Identical PCM files can share one served asset; 3,192 score rows therefore do not imply 3,192 distinct WAVs. The interface was checked as responding on port 8888 during this summary; no restart or UI changes were needed.

## 9. How this consolidation was checked

| Check | Result |
|---|---|
| Current development catalogue | 78 conditions; 3192 per-recording score rows; counts, unique IDs per condition, pooled S/D/I, reference totals, WER and mean MOS reconciled |
| Historical starting-point evidence | Ten complete runs retained from the earlier independent 6,570-row audit; the consolidation reused that evidence rather than rerunning models |
| Latest timing table | 384 unique(recording, variant) pairs; 48 complete recordings for each variant |
| Latest gate arithmetic | No variant passes the fixed WER/MOS gate; all source results preserved despite selection error |
| Protected data | Only frozen metadata/receipts used; no confirmation/final audio or outcomes opened |
| Cost scope | Nested timers/trajectory checkpoints explicitly identified; missing timing/memory is not treated as zero |
| New work in this summary | CPU evidence reconciliation and Markdown writing only; no GPU run, model scoring or parameter change |

[Reviewed data snapshot](/workspace/improvement/research_summary/reviewed_results.json) · [Input hashes](/workspace/improvement/research_summary/source_manifest.json) · [Consolidation verification](/workspace/improvement/research_summary/verification.json).

No exact all-history kernel-active GPU total is available: early timer coverage is incomplete and later wall/stream timers measure different scopes. Per-phase measured costs above are more defensible than a fabricated grand total.
