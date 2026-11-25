# BeyondWaveform: thread summary and future research plan

Updated **20 October 2026**, Asia/Taipei, after inspecting the completed loss-ablation job and its per-case verification records. This document consolidates the work performed in this thread, measured results, implementation changes, failures and repairs, operating instructions, unresolved gaps, and future plans. Earlier experiments are identified as existing baselines rather than work newly performed here. This summary update modifies this document only; it does not launch research jobs.

## 1. Current outcome

**The available defense tests, successive WER/quality studies, numerical diagnostics and latest complete loss ablations have finished. We have not simultaneously matched the paper's WER and sound-quality results.** UTMOS remains the main gap at strong attack settings. Numerical repeatability was established on the checked development clips through an isolated execution path; it did not close the quality gap or establish author-code parity.

The latest queue, `paper-quality-stable-ablations`, finished on **20 October at 14:53:53**, exit 0. Its full-objective control preflight and all three generation/evaluation/verification sequences passed. Job/status records report complete, and no research runner or status updater was active when checked. The fresh 48-clip confirmation cohort remains unscored. No additional selected-candidate 999-clip test was launched after the separately predeclared paper-budget control.

| Reference or measurement | Pool and settings | WER % | DNSMOS | NISQA | UTMOS |
|---|---|---:|---:|---:|---:|
| Paper anchors | Author protocol/subset incompletely specified | 75.4300 | 2.9540 | 3.5260 | 3.5900 |
| Existing headline baseline | 999 test clips; rho=0.5, lambda=0.1 | 74.1064 | 2.9638 | 3.8711 | 2.7924 |
| New paper-budget test control | Same 999 IDs; mean NLL, Adam LR=0.05; rho=0.1, lambda=1 | 64.7415 | 3.1262 | 3.9542 | 3.3582 |
| Strong attack control | 100 validation clips; sum NLL, Adam LR=0.1 | 75.5601 | 3.0978 | 4.0254 | 3.2457 |
| Higher-quality control | 100 validation clips; sum NLL, Adam LR=0.02 | 51.9857 | 3.1811 | 4.1038 | 3.5874 |
| Repeatable strong control | Existing 24-clip development pilot; sum NLL, LR=0.1 | 74.7368 | 3.1640 | 4.0350 | 3.3186 |
| Repeatable quality control | Same 24 clips; sum NLL, LR=0.02 | 45.6842 | 3.2075 | 4.1245 | 3.6761 |
| Repeatable timestamp setting | Same 24 clips; sum NLL, LR=0.05; initial timestamp | 71.7895 | 3.1273 | 4.0778 | 3.4034 |

The validation and repeatable pilot controls show the central tradeoff: attack strength can approach the paper's WER, while quality-preserving settings produce much lower WER. Pools differ across rows; pilot/validation observations are not final test measurements. The latest ablations also failed to close the UTMOS gap and deliberately remove parts of the full objective, so they cannot qualify as full-method reproductions. Their complete results are in Section 5.12. All quality figures are automatic estimator scores, not human listening assessments.

## 2. User objective and working preferences

The requested order was to review [STRICT_PARITY_AUDIT.md](/workspace/docs/STRICT_PARITY_AUDIT.md), evaluate defenses on the existing implementation first, and then research WER and quality improvements, particularly choices unspecified in the paper. Changing the 999 samples was allowed, but the test subset was retained for controlled comparisons. Later, the user requested a job-status file and two or three experiments queued overnight.

Current working preferences:

- Keep logs, job records, progress and result files available for long runs.
- Check actual job/process state before launching another copy.
- Preserve completed experiments and distinguish failed infrastructure from negative scientific results.
- **Discord notifications are disabled.** The user revoked the earlier completion-notification request. Set `BEYONDWAVEFORM_NOTIFY_DISCORD=0`, omit `--notify-discord`, and do not retry old receipts or request webhook access.
- Keep author-unspecified assumptions explicit; do not represent an independent implementation as recovered author code.

## 3. Audit review and experimental foundations

The supplied [paper](/workspace/BeyondWaveform.pdf), [reproduction specification](/workspace/BeyondWaveform_Reproduction_Spec.md), and existing strict audit provided the starting point.

The audit identified several distinct issues:

1. **Budget mismatch:** the existing 999-clip headline used rho=0.5/lambda=0.1, while the paper states rho=0.1/lambda=1, 50 updates and a 6000 Hz HF cutoff. Its logged final radii exceeded 0.1; old deltas were not saved, so those historical radius findings rely on traces.
2. **Unspecified objective/update choices:** lexical NLL reduction, optimizer, learning rate, initialization, HF alpha, epsilon and final-iterate selection are not fully identified by the paper. Mean versus sum NLL changes the relative perceptual weighting at equal numeric lambda.
3. **Feature/vocoder uncertainty:** layer 6, waveform normalization and the particular feature-conditioned HiFi-GAN are independent choices. A different feature layer or preprocessing cannot be assumed compatible with arbitrary generator weights.
4. **ASR protocol uncertainty:** transcript casing, text normalization, complete decoder defaults and timestamp conditioning affect results. The attack's no-timestamps prompt differs from timestamp-enabled recognition.
5. **Quality comparability:** the implemented channels are DNSMOS OVRL, NISQA overall MOS and UTMOS22 strong. No channel-index bug was demonstrated. Author checkpoint/calibration/preprocessing/subset parity remains unknown.
6. **Sample-rate validation gap:** the general spectral config can accept a rate inconsistent with actual 16 kHz audio. Research preflights reject that mismatch; a universal pipeline-level guard remains a possible hardening task.
7. **Incomplete coverage:** several defended checkpoints/recipes and other full-paper experiments are unavailable.

The existing core attack already used same-shaped feature perturbations, a frozen WavLM encoder, a frozen differentiable feature-conditioned vocoder, lexical teacher-forcing NLL, clean-referenced temporal/HF penalties, and global relative-Frobenius projection. We preserved that structure. No kNN feature replacement or waveform residual was added.

For clean features `q=E(x)`, the objective is `-NLL(V(q+delta), y) + lambda*(TV(q+delta)/(TV(q)+eps) + alpha*HF(V(q+delta))/(HF(x)+eps))`, with a global delta radius bounded by `rho*(||q||+eps)`. The temporal denominator uses original clean features, and the HF denominator uses original clean audio rather than vocoder reconstruction. Only delta is optimized.

Important artifact/interface anchors are WavLM-Large layer 6 (1024 channels), feature-conditioned regular `g_02500000.pt` or alternative `prematch_g_02500000.pt`, pinned kNN-VC source `c616845c4e309e24d5927f15adbdf277a3d65358`, and native 16 kHz output of 320 samples per feature frame. The English surrogate is Whisper-small with beam 5, temperature 0 and no previous-text conditioning. The UTMOS SpeechMOS source is pinned at `ed25eacbfa42b99156c36ebec67a733b5dbb9b79`. Full checkpoint hashes and effective settings live in the frozen plans and run metadata; these anchors do not identify the authors' artifacts.

Two later clarifications matter. Appendix D explicitly identifies 999 test-clean utterances for final English evaluation despite the broader train-clean-100 description in Section 5; changing to a training pool was not justified. Equation 13 explicitly specifies an **STFT**, but omits its time index/pooling and numerical STFT parameters. Global versus equal-frame pooling is unresolved; replacing STFT with a whole-clip FFT would change the stated definition.

A bounded [author-artifact search on 19 October](/workspace/verification/author-artifact-search-20261019.md) found only the supplied v1 record and did not locate an identifiable author repository/checkpoint. This does not prove that artifacts do not exist. No authors were contacted.

## 4. Defense tests completed before tuning

The suite evaluated the existing headline attack WAVs on the unchanged 999 IDs, with **clean, reconstruction and adversarial** controls. Every condition has 20,310 reference words and 2,997 score rows. Raw scores were reaudited; three transformed conditions were newly evaluated.

| Condition | Clean WER % | Reconstruction WER % | Attack WER % |
|---|---:|---:|---:|
| Raw historical control | 3.963565 | 3.727228 | 74.106352 |
| Independent LPF, 6000 Hz | 3.835549 | 4.076809 | 73.540128 |
| WaveGuard down/up, 8000 Hz | 4.081733 | 4.165436 | 71.733136 |
| WaveGuard down/up, 2000 Hz | 32.373215 | 34.800591 | 101.600197 |

The 2 kHz condition substantially damages benign recognition. Its attack WER above 100% is mathematically valid because insertions count toward WER. It is not evidence of a quality-preserving defense or identification of the paper's WaveGuard setting.

WaveGuard preprocessing follows pinned upstream down/up resampling and PCM16 behavior, with an explicit historical `kaiser_best` filter. `resampy==0.4.3` was installed into a workspace-local runtime. Tests compare the adapter against the pinned upstream functions and PCM16 I/O. These measurements cover preprocessing WER, not WaveGuard's complete consistency detector or AUC. LPF design and WaveGuard settings remain independent choices.

**Full paper defense parity remains incomplete.** Missing prerequisites:

| Defense | Remaining gap |
|---|---|
| CE-AT | Defended ASR checkpoint or complete adversarial-training recipe |
| DT-AT / DE-AT | Resolve the inconsistent name; obtain objective, training recipe and weights |
| DMW | ASR adaptation, hyperparameters and trained weights |
| TI | Identify the claimed defense and weights; the cited work describes an attack |
| AudioPure | Sentence-compatible checkpoint and declared segmentation/diffusion/merging recipe; released SC09 setup is one second |
| PVP Vote | Identify intended method, checkpoint, precision/voting/tie policy; citation mismatch remains |

No substitute measurements were fabricated. Post-defense MOS, detector AUC, human listening, physical playback and full transfer/defense coverage were not produced by these runs.

Evidence: [defense results](/workspace/reproduction-runs/english-defenses-999/report.md), [source review](/workspace/docs/DEFENSE_RESEARCH.md), and `/workspace/verification/defense-research`.

## 5. Research batches completed

Full-objective attack studies retained rho=0.1, lambda=1, 50 updates and the stated 6000 Hz cutoff. Later loss ablations explicitly disable named components while retaining those recorded numeric settings; they are diagnostic exceptions to the full objective. Attack studies score clean/reconstruction/adversarial audio and retain delta evidence. Saved-audio, derivative and execution diagnostics are identified separately and do not automatically generate new WER/MOS measurements.

The first four batches were:

| Batch | Finished, Asia/Taipei | Newly generated pilot runs | New 100-clip validation runs | New 999-clip runs |
|---|---|---:|---:|---:|
| Paper-budget NLL/LR study | 18 Oct 02:20:40 | 6 | 2 | 1 |
| Casing/optimizer/HF/initialization refinement | 18 Oct 14:40:42 | 14 | 2 | 0 |
| Generator/normalization study | 18 Oct 18:00:45 | 8 | 1 | 0 |
| Three overnight studies | 19 Oct 01:46:42 | 20 | 3 | 0 |
| **Subtotal for these first four batches, excluding reused results** | | **48** | **8** | **1** |

Reports also display cached controls, which are labeled and are not counted as new runs above. Smoke fixtures are separate integration checks, not selection evidence. This is not a total for the entire thread: later studies include transformed saved deltas, shared trajectories with multiple outputs, derivative probes and fresh attack trajectories, which must not be counted as interchangeable independent runs.

### 5.1 Paper-budget NLL and learning-rate study

Six original 12-clip pilot configurations crossed mean/sum lexical NLL with Adam LR 0.02/0.05/0.1. Two candidates advanced to 100-clip validation. A separately predefined mean-NLL/LR=0.05 control ran on the original 999 test IDs to close the missing paper-budget comparison; it was fixed before new pilot/validation outcomes.

The test produced **64.7415% WER**, with DNSMOS/NISQA/UTMOS **3.1262 / 3.9542 / 3.3582**. This improved quality over the historical larger-budget attack but did not reproduce the paper's WER and quality jointly.

Sum/LR=0.1 validation reached **75.5601% WER / 3.2457 UTMOS**. Sum/LR=0.02 reached **51.9857% / 3.5874**. The latter is close to the quality anchor but far from the WER target.

Evidence: [results](/workspace/reproduction-runs/paper-quality-research/report.md), [protocol](/workspace/docs/QUALITY_RESEARCH.md).

### 5.2 Casing, SGD, HF weight and initialization

Fourteen new 12-clip pilots tested:

- Lowercase/sentence-case attack targets, crossed with mean/sum NLL and Adam LR 0.05/0.1: eight cases.
- Sum-NLL SGD at LR 1/5/20: three cases.
- HF alpha 0.1/10: two cases.
- Gaussian feature initialization: one case.

Three cached controls and one cached validation result were also shown. The lowercase mean/LR=0.05 pilot reached **109.5477% WER / 3.4548 UTMOS**, but its validation result was **66.3442% / 3.3306**. The pilot's many insertions did not translate into a comparable validation result.

Alpha=10 sum/LR=0.1 validation reached **76.0692% WER / 3.2454 UTMOS**. Raising alpha reduced the pilot's measured last-update HF ratio from approximately 0.3682 to 0.1101, while UTMOS changed little. More suppression of the measured >6 kHz fraction did not resolve the quality gap in those runs.

No joint qualifier advanced to confirmation or a new test. Evidence: [results](/workspace/reproduction-runs/paper-quality-refinement/report.md), [protocol](/workspace/docs/QUALITY_REFINEMENT.md).

### 5.3 Generator and normalization study

The alternative upstream prematched generator was downloaded with a content-hash receipt. Three new paths were considered: prematched/raw, regular/normalized, prematched/normalized. Each path started with sum/LR=0.02; a failed reconstruction screen skipped its remaining five optimizer settings. The raw-prematched path completed six settings; both normalized paths stopped after their probes, avoiding ten further runs.

| Pilot path | Reconstruction WER % | Reconstruction UTMOS |
|---|---:|---:|
| Existing regular/raw | 6.0302 | 4.0548 |
| Prematched/raw | 4.0201 | 3.6987 |
| Regular/normalized | 6.0302 | 3.2117 |
| Prematched/normalized | 6.0302 | 2.8149 |

Five of six raw-prematched attacks had both lower WER and lower UTMOS than the corresponding regular/raw settings. The sixth gained WER at lower quality. Both normalized paths failed the reconstruction-quality gate. These results support retaining regular/raw among the tested choices, not a claim that every other feature/vocoder architecture is inferior.

The additional regular/raw sum/LR=0.05 validation result was **69.3483% WER / 3.3084 UTMOS**. No new test ran. Evidence: [results](/workspace/reproduction-runs/paper-quality-contracts/report.md), [protocol](/workspace/docs/QUALITY_CONTRACTS.md).

### 5.4 Overnight queue: all three studies finished

The three sequential studies were:

| Study | New configurations | Procedure |
|---|---:|---|
| Timestamp conditioning | 8 | Unscored fixed 0.00-second timestamp before lexical targets; reduction/LR and limited lowercase comparisons |
| Learning-rate schedules | 4 | Sum-NLL Adam LR 0.05/0.1, cosine or midpoint decay to 10% |
| Normalized feature gradients | 6 | Mean/sum NLL; L2 PGD step fractions 0.005/0.01/0.02 of the clean feature norm |

Two freshly generated controls brought the total to 20 pilot runs. All used a **fresh 24-clip dev-clean sample**, seed **20261018**, excluding previous pilot/validation/test IDs and audio hashes. Its clips total 170.515 seconds, averaging 7.105 seconds. No ASR/MOS scores participated in sampling. Old 12-clip scores were not mixed into this promotion ranking.

All three families passed real pretrained one-clip generation/recognition/delta checks before launch. The full queue then ran from **18 Oct 21:51:21 to 19 Oct 01:46:42**, approximately 3 hours 55 minutes, with no failed cases.

Pilot-selected validation results:

| Candidate | WER % | DNSMOS | NISQA | UTMOS |
|---|---:|---:|---:|---:|
| Timestamp, sum NLL, LR=0.02 | 51.5275 | 3.1981 | 4.1315 | 3.5891 |
| Cosine Adam, sum NLL, initial LR=0.1 | 68.5845 | 3.1074 | 4.1090 | 3.3277 |
| Timestamp, sum NLL, LR=0.05 | 71.0285 | 3.1537 | 4.0233 | 3.3713 |
| Cached sum-NLL/LR=0.02 control | 51.9857 | 3.1811 | 4.1038 | 3.5874 |

Timestamp/LR=0.05 improved over the corresponding earlier no-timestamps LR=0.05 validation measurement (**69.3483% / 3.3084**) by about **1.6802 WER percentage points and 0.0629 UTMOS**. This is a possible lead, not established significance or a joint match. Timestamp/LR=0.02 differs from its control by only about 0.0017 UTMOS, which should not be treated as meaningful without uncertainty analysis.

Cosine/LR=0.1 achieved **91.7895% WER** on the fresh pilot but **68.5845%** on validation. The normalized-gradient cases did not win promotion; no claim of their validation performance is made. No overnight case passed the joint validation targets, so confirmation and an additional 999-clip test were correctly skipped.

Evidence: [full results](/workspace/reproduction-runs/paper-quality-overnight/report.md), [protocol](/workspace/docs/QUALITY_OVERNIGHT.md), [job record](/workspace/reproduction-runs/paper-quality-overnight/job.json).

### 5.5 Saved-delta smoothing and component-gradient diagnostics

On the fresh 24-clip pilot, ten conditions compared the saved strong attack with Gaussian smoothing over 3/5/9 feature frames, smoothing restored to the original norm, and unsmoothed attenuation matched to each smoothed norm. These transformations used **zero new optimization updates**. Returned deltas were independently checked and re-vocoded; no waveform normalization was added.

| Direction at original delta norm | WER % | UTMOS |
|---|---:|---:|
| Identity control | 72.6316 | 3.3035 |
| Smoothing, 3 frames | 70.9474 | 3.2679 |
| Smoothing, 5 frames | 54.7368 | 3.2564 |
| Smoothing, 9 frames | 30.9474 | 3.2754 |

No smoothing direction achieved the predefined 0.05 UTMOS gain with at most five WER-point loss against its norm-matched control. Apparent gains at lower norms also weakened the attack. **This smoothing family was stopped; it did not justify a smoothed optimizer.** The completion review checked 397 frozen inputs, 216 new deltas and all 240 adversarial score rows.

A separate first-three-clip probe measured original-objective component gradients at zero/saved delta. Saved-delta unpreconditioned ASR norms were 3.14–5.72, TV norms 0.00137–0.00341 and HF norms 0.0232–0.0347. These raw norms do not measure contributions after Adam's moment scaling; that limitation motivated Section 5.8.

Evidence: [results](/workspace/reproduction-runs/feature-delta-diagnostics/report.md), [completion review](/workspace/reproduction-runs/feature-delta-review-20261019/report.md), [protocol](/workspace/docs/QUALITY_DELTA_DIAGNOSTICS.md).

### 5.6 Teacher-forcing versus native-decoder alignment

Measured saved clean/reconstruction/adversarial audio under the no-timestamps and fixed initial-timestamp teacher-forcing prefixes. The analysis retained the full-vocabulary lexical loss and separated native static token suppression from dynamic timestamp grammar. No attack, free-decoding recognition, MOS scoring or new dataset use occurred in this diagnostic.

All **942 lexical targets per view** were statically allowed. Static suppression explained only **0.241647%** of the adversarial-minus-reconstruction NLL rise, with median adversarial suppressed mass **0.0004703**. This missed the predeclared screen requiring at least 10% explanatory share, median mass at least 0.10 and 95% target coverage. Dynamic filtering can make forced-reference histories unreachable; those masked losses are not interchangeable with corpus WER.

**No filtered-softmax intervention was justified or activated.** The completion review checked 510 frozen inputs, all 24 IDs, both prefixes and measured-asset/token aggregation bindings. Evidence: [results](/workspace/reproduction-runs/decoder-alignment-diagnostics/report.md), [protocol](/workspace/docs/QUALITY_DECODER_ALIGNMENT.md).

### 5.7 Last-ten-iterate averaging

Two existing settings each ran one fresh 50-update trajectory per clip, retaining projected deltas 41–50. Each trajectory returned terminal delta 50, the mean of deltas 41–50, and the terminal direction scaled to the mean's norm. Thus **48 actual pilot trajectories produced six output conditions**, not six independently optimized attacks per clip.

| Setting | Returned delta | WER % | UTMOS |
|---|---|---:|---:|
| Sum/LR=0.1 | Terminal | 78.5263 | 3.3142 |
| Sum/LR=0.1 | Mean 41–50 | 79.1579 | 3.3303 |
| Sum/LR=0.1 | Norm-matched terminal | 78.7368 | 3.3269 |
| Timestamp/LR=0.05 | Terminal | 70.7368 | 3.3975 |
| Timestamp/LR=0.05 | Mean 41–50 | 68.4211 | 3.4314 |
| Timestamp/LR=0.05 | Norm-matched terminal | 70.3158 | 3.4049 |

UTMOS gains over the proper norm-matched controls were **+0.003445** and **+0.026580**, below the predefined +0.05 gate. Neither expanded to confirmation. **The averaging-window family was stopped; no further window search was launched.** The independent review checked 599 frozen inputs, 48 trajectories, 480 snapshots, 144 returned deltas and 240 score/WAV bindings.

Before measuring these outputs, a **fresh 48-clip development confirmation pool** was frozen with seed 20261019, excluding prior IDs or audio hashes. It contains 28 speakers and 319.725 seconds, with manifest SHA256 `377a51cacce705ee4e6d5777aad4d53c042cb473e4b0a4dd001273a8d640a713`. All 28 speakers overlap earlier development pools: it is utterance/audio-disjoint, not speaker-disjoint or a final test set. It remains **unscored** because the pilot gates failed.

The new terminal reruns differed from their earlier same-configuration parents by +5.895/−3.789 WER points. Clean/reconstruction samples matched exactly while adversarial samples differed, and trace differences began at update 1. One 13-word clip supplied 20 of the 28 extra strong-control errors, mainly insertions. These findings prompted numerical diagnostics rather than treating small score changes as algorithm gains.

Evidence: [results](/workspace/reproduction-runs/paper-quality-finalization/report.md), [completion review](/workspace/reproduction-runs/finalization-completion-review-20261019/report.md), [protocol](/workspace/docs/QUALITY_FINALIZATION.md).

### 5.8 Numerical repeatability and actual Adam-step contributions

On the fixed first three pilot IDs, repeated the unchanged full-objective derivative three times at zero and saved parent delta. Both default and math-only attention produced nonidentical gradients at fixed inputs. Profiling identified efficient attention in the default path, rather than assuming a flash kernel. Strict math initially rejected native CUDA reflection-padding backward.

The original strict attempt failed because the intended known-operation classifier omitted the actual `reflection_pad1d_backward_out_cuda` alias. Its source, tests, plan and failure records were archived before adding the narrow alias regression fix. The corrected v2 job completed with strict native mode accurately labeled **unsupported**, preserving strict algorithms, `warn_only=False` and the cuBLAS workspace requirement. Unknown exceptions were not relabeled as scientific results.

Three fresh full-objective 50-update trajectories then captured updates 1/10/25/50. At each capture, local branches used the **same current delta and actual Adam moments** with full, ASR-only, TV-removed or HF-removed gradients. Only the full branch continued. All full-gradient clones exactly matched actual projected updates.

| Capture | ASR-only versus full projected-step difference | Without TV | Without HF |
|---|---:|---:|---:|
| Update 1 | 20.601–42.224% | 4.320–10.217% | 20.455–41.860% |
| Update 50 | 0.708–1.154% | 0.032–0.233% | 0.707–1.127% |

The independent review checked **630 frozen inputs, 12 captures and 48 branches**, including saved moments and their recurrences. Small raw perceptual-gradient norms therefore did not establish an inactive regularizer. Conversely, these local alternatives are not complete loss ablations or quality estimates. Actual post-update moment tensors were not separately saved in this particular probe; actual/cloned delta equality and cloned moment recurrence were verified, with that limit retained in the report.

Evidence: [v2 results](/workspace/reproduction-runs/optimizer-mechanism-diagnostics-v2/report.md), [completion review](/workspace/reproduction-runs/optimizer-completion-review-20261019/report.md), [protocol](/workspace/docs/QUALITY_OPTIMIZER_DIAGNOSTICS.md).

### 5.9 Equivalent frontend and full-trajectory repeatability

Added an isolated explicit-reflection frontend: reflected slices/concatenation supply the same STFT input values, followed by `center=False`, avoiding the native reflection-padding backward operator. The Whisper mel calibration, Hann windows, STFT conventions, global HF arithmetic and original-clean denominators remain unchanged. It is an execution-path change, not a new loss definition. Production attack/frontend sources were not silently replaced.

Fresh CUDA processes used **strict deterministic algorithms, `warn_only=False`, math SDPA only, cuDNN deterministic, benchmarking/TF32 disabled and `CUBLAS_WORKSPACE_CONFIG=:4096:8` before initialization**. Native/new mel, HF and full-objective forward values matched bitwise at all six fixed states. Three repeated full gradients matched exactly at each state. Relative L2 differences from the old nondeterministic native-math gradients were 4.13e-7–7.21e-7.

Two separate fresh-process runs also matched **all 50 deltas, loss/gradient/moment hashes and final audio/optimizer values on each of the three clips**: 150 paired update comparisons. The plan bound 664 inputs, and all gates passed. This establishes repeatability on those clips/runtime only; it neither identifies author code nor guarantees other environments. The old unstable scores remain preserved rather than rewritten.

Evidence: [results](/workspace/reproduction-runs/deterministic-execution-validation/report.md), [protocol](/workspace/docs/DETERMINISTIC_EXECUTION_VALIDATION.md), [isolated frontend](/workspace/beyondwaveform/deterministic_frontend.py).

### 5.10 WER/quality anchors under repeatable execution

Ran the existing strong sum/LR=0.1, quality sum/LR=0.02 and timestamp sum/LR=0.05 settings on the same 24 development clips using strict explicit-reflection generation. Native saved-audio decoding and MOS evaluation retained their previous execution policy. This produced **72 new attack trajectories**; the three results appear in Section 1. None met all four descriptive paper targets jointly.

The independent completion audit checked **1,008 frozen inputs, 3,600 projected iterates, 72 final audio/Adam states and 216 score/WAV rows**. All 96 paired cross-setting clean/reconstruction comparisons had identical samples and scores. All bound GPU resynthesis receipts reported zero error; CPU review did not rerun neural inference. Strong-control first-three trajectories matched the numerical validation at every update. Maximum independently calculated radius was 0.1000000203, within the stated float32 tolerance.

Clean and reconstruction WER were both **3.5789%**, with UTMOS **4.1886** and **4.1667**. Quality degradation therefore appears during attack optimization on this pilot, despite a strong reconstruction baseline. Comparing newer stable scores with earlier native runs does not isolate a causal frontend benefit; historical generation was numerically sensitive. The smaller timestamp/strong differences also remain uncertain on this reused, small pilot.

Evidence: [results](/workspace/reproduction-runs/paper-quality-stable-execution/report.md), [completion review](/workspace/reproduction-runs/stable-execution-completion-review-20261020/report.md), [protocol](/workspace/docs/QUALITY_STABLE_EXECUTION.md).

### 5.11 Saved-audio HF time-pooling analysis

Added a separate differentiable diagnostic exposing per-frame power, the existing global HF fraction, its energy-weighted reconstruction, and an equal-positive-energy-frame mean. The latter excludes exactly silent frames without adding an epsilon inside HF; this is an independent interpretation of the omitted time pooling, not established author behavior.

The CPU saved-audio study bound **1,245 inputs**, included all **216 saved WAV records**, and retained all 144 reconstruction/adversarial versus original-clean paired comparisons. The global helper matched the existing frontend bitwise on every waveform. The largest energy-weighted reconstruction error was 4.47e-8 from reduction rounding; there were no exactly silent frames.

| Adversarial setting | Median global/clean ratio | Median equal-frame/clean ratio | Global <1 but equal-frame >1 |
|---|---:|---:|---:|
| Strong | 0.372686 | 0.755771 | 1/24 |
| Quality | 0.269086 | 0.666535 | 1/24 |
| Timestamp | 0.312934 | 0.696287 | 3/24 |

Both HF summaries decreased on most attacked clips despite the remaining UTMOS gap. This did not establish a time-pooling bug or justify promoting an equal-frame attack. No objective, decoder or MOS result was changed by the analysis. Evidence: [report](/workspace/reproduction-runs/hf-pooling-saved-audio-20261020/report.md), [module](/workspace/beyondwaveform/hf_pooling_diagnostics.py).

### 5.12 Latest completed full-trajectory loss ablations

The latest study addressed what the local Adam probes could not: complete 50-update trajectories with TV removed, HF removed, or both removed. All three used the same 24 development IDs and strong-control settings. Numeric lambda remained 1; disabled terms contributed exact zero. **These are deliberate ablations, not full-objective candidates.**

The new flag-aware collector first reproduced **all 150 control updates and final audio/optimizer values**. Its plan froze **1,347 inputs**. The queue ran from **20 October 13:50:48 to 14:53:53**, completing all generation, native evaluation and verification stages with exit 0.

| Objective | WER % | DNSMOS | NISQA | UTMOS | Delta WER vs full | Delta UTMOS vs full |
|---|---:|---:|---:|---:|---:|---:|
| Frozen full control | 74.7368 | 3.1640 | 4.0350 | 3.3186 | — | — |
| Without TV | 76.0000 | 3.1530 | 4.0398 | 3.3226 | +1.2632 pp | +0.0040 |
| Without HF | 74.7368 | 3.1351 | 4.0028 | 3.3058 | +0.0000 pp | −0.0128 |
| Without both perceptual terms | 78.7368 | 3.1564 | 4.0863 | 3.3504 | +4.0000 pp | +0.0318 |

On this pilot, removing HF slightly worsened all three MOS means without changing aggregate WER. Removing both raised WER and UTMOS but lowered DNSMOS slightly; UTMOS still missed 3.590 by about 0.240. The full objective therefore did not uniformly dominate the ablations in the manner suggested by the paper's qualitative loss-guidance discussion. These small-pilot observations require paired uncertainty analysis and do not prove an implementation defect or justify removing required losses from a claimed reproduction.

Each per-case verification record reports **1,200 checked projected iterates, 24 zero-error final GPU resyntheses, exact disabled-term-zero checks, terminal audio/moment bindings and matching evaluation protocol**. Across the three ablations this is 3,600 iterates and 72 resyntheses. Those job-generated checks were inspected for this summary; a separate post-completion independent audit and paired statistical analysis of this newest batch remain future work. No new job was launched by this document update.

Evidence: [results](/workspace/reproduction-runs/paper-quality-stable-ablations/report.md), [job](/workspace/reproduction-runs/paper-quality-stable-ablations/job.json), [protocol](/workspace/docs/QUALITY_STABLE_ABLATIONS.md), [runner](/workspace/scripts/run_stable_ablation_research.py).

## 6. Implementation and operational work

| Area | Work performed |
|---|---|
| Defense adapter | Added source-backed WaveGuard resampling/PCM16 behavior and regression/integration tests |
| Experiment drivers | Added defense, NLL/LR, refinement, feature/vocoder, overnight, saved-delta, decoder, finalization, optimizer, deterministic-execution, stable-anchor and ablation runners with frozen plans and stage supervision |
| Whisper targets | Added optional transcript casing and fixed initial-timestamp conditioning; lexical targets remain the only scored tokens |
| Update rules | Added optional cosine/two-stage LR schedules and normalized L2 PGD; scheduled cases record and audit effective LRs |
| Default compatibility | Preserved default attack behavior and metadata when optional keys are absent; recognition decoding and supervised fine-tuning were kept fixed |
| Data isolation | Audited existing 12/100/999 pools; created a disjoint 24-clip pilot and predeclared unused 48-clip confirmation cohort; kept original 999 test IDs |
| Scoring provenance | Reused hash-verified scorer checkpoints; recorded runtime, channels, preprocessing and aggregation; historical fixture scores reproduced within 1e-5 |
| Delta evidence | Saved feature deltas; independently re-extracted matching clean features and checked actual relative radii against 0.1 with float32 tolerance |
| Historical evidence | Archived original source/config/plan/failed-attempt records before amendments; preserved completed plans/results and used historical-source mappings |
| Gradient/optimizer evidence | Added component derivatives, same-moment cloned Adam branches, raw tensor hashes, per-update receipts and complete flag-aware ablation trajectories |
| Numerical execution | Added isolated explicit-reflection STFT/HF/Whisper helpers; validated strict CUDA gradients and paired full trajectories before WER/MOS scoring |
| Finalization and spectral diagnostics | Added saved-delta transforms, projected-iterate collection, norm-matched comparisons and HF frame-pooling statistics; retained negative results and stop decisions |
| Monitoring | Added Markdown/JSON status files, 30-second heartbeats, per-stage logs, liveness checks and duplicate-monitor locking |
| Overnight execution | Used detached supervised processes; independent case failures allow remaining cases to run, retain a failed overall status and prevent automatic final testing |

Important repaired failures:

1. **ASCII export failure:** a decoded Unicode character caused the first LPF attempt to fail after inference. Explicit UTF-8 JSON/JSONL/report I/O and UTF-8 launch settings fixed it. The failed attempt was archived and the affected inference rerun.
2. **JSON numeric parsing:** PyYAML interpreted valid JSON `1e-08` as a string. `.json` configs now use the JSON parser, and preflight validates the actual serialized generation configs before model loading.
3. **False `needs_attention`:** the monitor could observe an exited child before its supervisor wrote the terminal record. It now rereads final status and reports a live supervisor as finishing.
4. **Evaluation-reader mismatch:** new review code initially used an ID-unique manifest reader for three-variant scores. The corrected reader enforces `(ID, variant)` uniqueness; the failed preparation was archived before any new attack ran.
5. **Test fixtures:** initialized synthetic Whisper positional embeddings and corrected a mocked continuation-test record. These were fixture corrections, not changes to pretrained weights or research settings.
6. **Strict-operation alias:** the original strict diagnostic did not recognize `reflection_pad1d_backward_out_cuda`. Preserved the failure/source, added only the missing known-operation alias and regression test, then ran v2 without weakening deterministic settings.
7. **Strict CUDA frontend limitation:** native reflection-padding backward lacked the required deterministic implementation. The isolated equivalent frontend was tested explicitly rather than silently disabling strict mode or substituting a new spectral definition.
8. **Detached stdin preparation:** a preparation command using `python -` under `run_logged.py` performed no work because the wrapper deliberately gives children `DEVNULL` stdin. Retained the evidence and used an explicit script file; later launches execute file-backed commands.
9. **Ablation evidence and status:** strengthened disabled-term proofs to bind actual saved trajectory scalars, final audio/moments and decoder/scorer/backend identity. Reserved `control-trajectories/` separately from preparation logs, and emitted `N/24 [id]` for the existing clip-progress parser.

The early `waiting-for-defense-baseline` log was intentional: tuning waited for successful available-defense completion and audits. Later failures were diagnosed separately from that dependency. The final overnight job did not wait on unfinished defenses.

Discord completion hooks were initially added as requested. Delivery could not read the private root-owned configuration, so no successful delivery was claimed; failures were tracked separately from compute. The user subsequently disabled notifications. Old failed delivery receipts can remain visible in status history without indicating a current compute failure.

Unrelated workspace changes were preserved. This thread did not create a commit, PR or deployment.

## 7. Verification and interpretation rules

Test coverage expanded with each study. An early regression run passed 133 tests; subsequent diagnostics added numerical, optimizer, tensor-integrity and frontend coverage. The **latest targeted combined run passed 82 tests**, covering frontend equivalence, full-trajectory integration, protocol/cache guards, HF analytical/gradient checks and raw disabled-term receipt tampering. These are counts from different runs, not additive totals or a claim that every repository test was rerun at the latest revision.

Other coverage includes Unicode/config regressions, source-backed defense comparisons, token alignment/masking, native Whisper input gradients, frozen weights, default decoder/metadata compatibility, projection bounds, schedules, normalized steps, split isolation, promotion rules, archived-source integrity, status races and independent-failure continuation.

Scientific audits check full ID/variant coverage; original references; lexical edit counts and corpus denominators; arithmetic utterance-MOS means; model/config/runtime identities; WAV/index hashes; 50 finite updates; and actual saved-delta radii using the corresponding encoder preprocessing. Small float32 values such as 0.100000016 are within the declared 0.100001 tolerance, not silently expanded attack budgets.

Separate completion reviews exist for the overnight, saved-delta, decoder, averaging, optimizer and stable-anchor studies. A CPU audit can verify tensor values, receipts, edit counts and aggregation without replaying every CUDA gradient or MOS model; the reports preserve that distinction. In particular, the stable-anchor review verified all 1,008 inputs and 3,600 iterates, while the newest ablation batch currently has completed runner-generated per-case verification and awaits its independent completion review.

For this documentation-only update, the actual latest job/progress/results, all three ablation verification records, current consolidated status and the linked prior reports were inspected. They report terminal success, 1,200 checked iterates and 24 exact final resyntheses per ablation, with removed terms verified zero. Tests, full frozen-input audits and model inference were **not rerun** by this summary task.

Selection conventions:

- The paper's broad mean-MOS gate (>2.5 for all three metrics) is distinguished from our stricter research quality anchors.
- Early refinement/overnight studies used a predeclared **WER >=73.43%** exploratory floor, two points below the paper's 75.43%, plus DNSMOS/NISQA/UTMOS >=2.954/3.526/3.590. The later stable-anchor study required the exact descriptive **75.43%** WER target. Neither convention is an author selection rule or a statistical equivalence margin; old gates are not rewritten after results.
- Smoothing/averaging used separate +0.05 UTMOS/at-most-five-WER-point-loss directional screens. Passing such a screen would not, by itself, match the paper's joint targets.
- Later studies also require acceptable reconstruction WER and quality, so failed reconstruction paths do not advance automatically.
- Pilot evidence selects validation settings; qualifying validation results repeat before one final test. Test/defense outcomes do not select replacement configurations.
- Original same-seed native CUDA runs varied. Bitwise repeatability was subsequently established for the isolated strict explicit-reflection path on the tested first three clips, including two complete fresh-process 50-update runs. Generality beyond that evidence is not assumed.
- Diagnostics with removed losses cannot qualify as full-objective reproductions, even when WER exceeds the paper anchor.
- Absolute scorer values and the author subset remain unmatched. A threshold crossing alone cannot establish full reproduction.

## 8. Proposed future research

**These are proposed next steps, not jobs launched by this summary task. All existing queues are complete.** The older proposals for late-iterate averaging, raw component-gradient measurement and new fixed-anchor repeatability tests have already been carried out; they are not pending experiments.

### Priority 1: independently review the completed ablations

Audit the 1,347 frozen bindings, all new projected iterates, disabled-term receipts, terminal audio/moments, unchanged control WAVs, evaluation settings and score aggregation against the completed full-objective baseline. Preserve the distinction between checked saved GPU receipts and fresh neural re-inference.

Then calculate paired per-utterance WER/edit and MOS differences, with uncertainty that accounts for repeated speakers where feasible. Determine whether the +0.0040 UTMOS without TV, −0.0128 without HF, and +0.0318 without both are consistent across clips or concentrated in a few cases. Equal aggregate WER can hide different substitutions, deletions and insertions. Evaluate the paper's Figure 8 claim on WER/DNSMOS separately from the UTMOS objective.

The current descriptive conclusion is mixed loss-guidance behavior and a persistent quality gap. Do not promote an ablated method or declare significance from the small mean changes alone.

### Priority 2: explain the residual quality gap and recover missing author details

Prioritize the encoder/vocoder training contract, optimizer and lexical NLL recipe, Whisper checkpoint/decoder, original subset and quality-scorer variants. The bounded author-artifact search did not resolve these. No external messages have been authorized by this documentation request; any later author contact needs explicit user authorization.

Use the existing saved audio/features to investigate where quality loss occurs: voiced speech versus quieter regions, duration, token length, insertion-heavy failures, feature displacement and spectral/loudness changes. Choose representative listening clips by declared rules rather than attractive results. Current evidence shows high-quality reconstructions and reduced measured HF power after attack, so suppression of the existing HF scalar alone is not a sufficient explanation.

Do not infer zero regularization from raw gradients; the moment-matched first-step effects were substantial. Do not assume optimizer oscillation or resume window searches without new evidence. Keep any alternative scorer/normalizer/decoder interpretation separately labeled rather than replacing official historical results.

### Priority 3: one bounded full-objective experiment only when supported

Choose one paper-unspecified detail with a concrete mechanism, declare its exact mathematical/computational change and stop rule, freeze its control/cohort/artifacts before scoring, and preserve rho=0.1/lambda=1/50 updates/fc=6000 plus both perceptual terms. A different feature/vocoder pair must first pass reconstruction checks; arbitrary layer or normalization changes are not interchangeable.

Equal-frame HF pooling remains a possible declared sensitivity interpretation, but the saved-audio evidence did not establish it as the missing mechanism. If later justified, first specify its original-clean denominator and zero-frame policy and verify gradient stability and strict CUDA behavior. It is not an established bug fix. No replacement coefficient, extra LR grid or new averaging/smoothing window is currently selected.

Directions already stopped or unsupported by their predefined screens:

| Direction | Current decision |
|---|---|
| Gaussian delta smoothing | Stopped; no norm-matched directional gain |
| Last-ten-iterate averaging | Stopped; both UTMOS gains below +0.05 |
| Static-suppression filtered softmax | Not justified by the mechanism screen |
| Blind waveform normalization/prematched substitution | Tested paths worsened reconstruction or attack-quality tradeoffs |
| Extra HF suppression alone | Earlier alpha=10 and later spectral evidence did not close UTMOS gap |
| Removing perceptual terms as the final method | Diagnostic only; changes the required full objective |

### Priority 4: confirm and test a qualified configuration

The already frozen, unscored **48-clip cohort** can support a newly declared prospective confirmation protocol once a valid full-objective pilot candidate is justified. Do not imply that the old averaging gate passed or silently change its old plan. Disclose its utterance/audio disjointness and speaker overlap; avoid turning confirmation into another tuning set.

Require paired improvements, acceptable reconstruction controls and all four explicitly declared WER/MOS targets before freezing a final configuration. Confirm the result under the same numerical policy. Only then run a fixed final 999-clip test, reporting exact values, uncertainty, assumptions and any miss. No current setting has earned this promotion.

If creating a new 999 subset, predeclare sampling criteria/seed without using attack or MOS results. Retain complete IDs/hashes and clean controls; keep the original subset for paired comparisons. Treat a fresh final pool as held out until selection is finished.

### Priority 5: broader reproduction coverage

After selecting a defensible final attack, rerun available defenses on its frozen WAVs, then extend transfer/defense evaluation as artifacts become available. Resolve the six missing defense recipes rather than substituting unrelated procedures. AudioPure sentence adaptation, PVP identity/precision behavior, trained defenses, human intelligibility and physical playback each require their own declared protocol and evidence.

### Priority 6: limited engineering hardening

Add the remaining pipeline-wide sample-rate contract if needed, preserve explicit numerical-execution metadata, and continue fixing genuine provenance/status defects with regression checks. Archive bound source bytes and hashes before changing files referenced by old plans. These engineering changes must not silently change the scientific objective or retroactively rewrite completed results.

## 9. Files and continuation instructions

| Purpose | Location |
|---|---|
| Current consolidated status | [JOB_STATUS.md](/workspace/reproduction-runs/JOB_STATUS.md), [JOB_STATUS.json](/workspace/reproduction-runs/JOB_STATUS.json) |
| Latest full result table | [stable ablations report](/workspace/reproduction-runs/paper-quality-stable-ablations/report.md) |
| Latest frozen inputs/configs | [ablation plan](/workspace/reproduction-runs/paper-quality-stable-ablations/plan.json) |
| Latest completion/progress | [job.json](/workspace/reproduction-runs/paper-quality-stable-ablations/job.json), [progress.json](/workspace/reproduction-runs/paper-quality-stable-ablations/progress.json) |
| Latest execution log | [research.log](/workspace/reproduction-runs/paper-quality-stable-ablations/research.log) |
| Repeatable full-objective anchor results | [stable execution report](/workspace/reproduction-runs/paper-quality-stable-execution/report.md) |
| Independent anchor completion audit | [review report](/workspace/reproduction-runs/stable-execution-completion-review-20261020/report.md), [review.json](/workspace/reproduction-runs/stable-execution-completion-review-20261020/review.json) |
| Exact-trajectory numerical validation | [report](/workspace/reproduction-runs/deterministic-execution-validation/report.md) |
| Saved-audio HF interpretation audit | [report](/workspace/reproduction-runs/hf-pooling-saved-audio-20261020/report.md) |
| Unscored prospective confirmation cohort | [48-clip manifest](/workspace/reproduction-runs/paper-quality-finalization/data/confirmation/confirmation-48.jsonl) |
| Detailed chronological handoff | [HANDOFF.md](/workspace/docs/HANDOFF.md) |
| Defense implementation/runner | [defenses.py](/workspace/beyondwaveform/defenses.py), [run_defense_suite.py](/workspace/scripts/run_defense_suite.py) |
| Initial research drivers | [NLL/LR](/workspace/scripts/run_quality_research.py), [refinement](/workspace/scripts/run_quality_refinement.py), [generator study](/workspace/scripts/run_quality_contracts.py), [overnight queue](/workspace/scripts/run_overnight_research.py) |
| Diagnostic drivers | [saved deltas](/workspace/scripts/run_feature_diagnostics.py), [decoder alignment](/workspace/scripts/run_decoder_diagnostics.py), [finalization](/workspace/scripts/run_finalization_research.py), [optimizer mechanisms](/workspace/scripts/run_optimizer_diagnostics.py) |
| Repeatable-generation drivers | [execution validation](/workspace/scripts/run_deterministic_execution.py), [stable anchors](/workspace/scripts/run_stable_quality_research.py), [loss ablations](/workspace/scripts/run_stable_ablation_research.py) |
| Attack/config/ASR extensions | [attacks.py](/workspace/beyondwaveform/attacks.py), [config.py](/workspace/beyondwaveform/config.py), [whisper.py](/workspace/beyondwaveform/models/whisper.py) |
| Isolated numerical/research helpers | [deterministic frontend](/workspace/beyondwaveform/deterministic_frontend.py), [Adam diagnostics](/workspace/beyondwaveform/adam_diagnostics.py), [trajectory diagnostics](/workspace/beyondwaveform/trajectory_diagnostics.py), [HF pooling](/workspace/beyondwaveform/hf_pooling_diagnostics.py) |
| Supervision/status | [run_logged.py](/workspace/scripts/run_logged.py), [parallel_jobs.py](/workspace/scripts/parallel_jobs.py), [update_job_status.py](/workspace/scripts/update_job_status.py) |

The environment uses `/opt/venv/bin/python` and a TITAN RTX with 24 GB. `/opt/venv` is read-only; new outputs/dependencies belong under `/workspace/reproduction-runs` and `/workspace/artifacts`. The current container lacks tmux; persistent jobs used `subprocess.Popen(..., start_new_session=True)` with detached standard streams and `run_logged.py` supervision.

The installed numerical stack used for the recent evidence includes Torch 2.4.1+cu121. Frozen plans record the remaining package and artifact versions. No commit, PR or deployment was created by this thread, and unrelated workspace edits were preserved.

Standard launch environment:

```text
PYTHONPATH=/workspace/artifacts/defense-runtime:/workspace
OMP_NUM_THREADS=4
MKL_NUM_THREADS=4
OPENBLAS_NUM_THREADS=1
PYTHONUTF8=1
PYTHONIOENCODING=utf-8
BEYONDWAVEFORM_NOTIFY_DISCORD=0
```

For strict generation subprocesses additionally set `CUBLAS_WORKSPACE_CONFIG=:4096:8` **before CUDA initialization** and enforce the declared strict/math controls. The recent saved-audio evaluation subprocesses intentionally retain the frozen native/default evaluation policy, with their backend metadata recorded separately. Do not impose generation flags on evaluation silently.

Before new work, inspect job records and actual process liveness. Preserve the relevant current source/plan/evidence before changing frozen code. Older plans may intentionally reference archived source; use their historical-source mappings rather than rewriting completed plans. Use a new frozen plan/output directory for new scientific choices and update the status monitor to the correct research directory. Run file-backed scripts under `run_logged.py`; stdin-fed Python is not a valid child launch with its `DEVNULL` policy.

The consolidated status last recorded **20 October 14:53:53 Asia/Taipei**, with both available defenses and latest ablations complete. The updater stopped normally; its timestamp is a completion snapshot, not a stalled heartbeat. The old eight failed Discord delivery receipts remain historical notification records and do not mean the latest compute failed. Do not restart completed directories as though they were new experiments: inspect partial/completed outputs and the runner's overwrite guards first.

The next research continuation should start with **Priority 1: independently review the completed loss ablations**, then choose a supported full-objective direction. This documentation update launched no experiments; the joint paper-performance goal remains unresolved.
