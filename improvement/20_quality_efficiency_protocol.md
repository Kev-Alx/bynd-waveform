# Registered quality and efficiency pilot

26 October 2026, Asia/Taipei. This cycle extends the completed research; the old queue and its selection remain frozen.

## Diagnosis before selection

`execution_tradeoff/diagnose_saved.py --signals --require-goal` reproduces the saved joint failure in under four seconds: 0/67 conditions pass (expected assertion exit 1). The 48-point strong single-source FV50 already reaches 83.9034% WER but UTMOS 3.1981. Its S/D/I are 520/82/232 over 994 words: insertions contribute, but do not dominate. FV1500 costs 30 times as many updates, reaches 101.4085% WER and UTMOS 2.9437. Saved loop throughput is 19.61 versus 587.73 seconds/recording, respectively. These are cumulative loop measurements, excluding original loading/calibration/preprocessing; checkpoint costs are not independent trajectories.

Strong FV50 has mean relative STFT magnitude error .634 versus .330 for neutral reconstruction, and frame-energy-envelope error .276 versus .133. These measures use clean speech as reference; vocoder reconstruction itself changes phase and spectrum. The within-FV50 UTMOS/log-magnitude-distance Spearman correlation is weak (-.284), and is not causal evidence or a human listening result. The existing no-edit decoder passes quality. These observations justify testing repair of attack-added spectral distortion, but do not establish its cause.

Ranked hypotheses: (1) redundant decoder forwards add avoidable compute; retaining its graph removes that work with identical math at a memory cost; (2) a modest fixed pull of attacked magnitudes toward the same decoder's neutral reconstruction repairs some quality at a risk to WER, since Whisper consumes magnitudes; (3) extended unbounded NLL ascent amplifies distortion without monotonic corpus WER gains. This cycle tests the first two with separate controls, using the cheaper fixed FV50 teacher. It does not claim a new learned generator or original algorithm.

## Fixed experiment

Use exactly the frozen 48 recordings / 360.010 seconds / 994 reference words in `execution48/development-48.jsonl`. No validation, transfer, confirmation or final data is accessed. Generate from the canonical saved clean WavLM6 features, original HiFi-GAN, Whisper-small, strict math, float32, batch one, Adam LR .1, relative feature radius .1 and original TV/HF penalties, 50 updates.

For every recording, run the original two-decoder-forward loop and the exact retained-graph one-forward loop. Alternate order by recording index; verify exact final delta and waveform equality, and compare with the frozen single50 output. This is a matched 48-point efficiency comparison with the same resident models, hardware, instrumentation, lengths and precision. Keep Whisper activation checkpointing unchanged. No larger source models are loaded: historical-queue throughput is context, not the matched speedup denominator. Explicitly charge model loading, feature preprocessing and warmup outside timed optimization.

The engineering check on the shortest clip (1.94 s; 5 updates after warmup) passed exact equality; two-forward/one-forward wall ratio 1.08045; peak allocated memory 1.65/2.11 GB. This is correctness/memory evidence only, not a performance pilot.

Quality intervention: Hann STFT 512 samples /160 hop, zero-padded boundary; one iSTFT; no iterations. Set target log magnitude to .75 attacked + .25 neutral-HiFi-GAN-reconstruction, floor magnitudes 1e-7, and retain attacked phase. Preserve exact attack output length. No clipping or loudness normalization. Score three complete 48-point conditions: unprocessed one-forward FV50, STFT/iSTFT identity control (fraction 0), and fixed 25% reconstruction-magnitude anchor. No fraction sweep or MOS-guided per-recording choice. Recompute achieved distortions after synthesis; projected magnitudes need not be STFT-consistent.

Reuse clean/reconstruction/quality-control scores and assets as historical controls where appropriate, without treating them as new replications. Formal scoring uses the historical native Whisper-small beam5/temperature0 backend and all three frozen MOS scorers, in a separate process with CUBLAS_WORKSPACE_CONFIG unset. Preserve full cohort means and denominators.

## Outcomes and proposed efficiency targets

The research gate remains WER strictly >75%, DNSMOS OVRL >=3, NISQA MOS >=3.75 and UTMOS >=3.5. There is no user-specified latency cutoff. Proposed engineering target: >=10% reduction in aggregate matched generation-loop wall time with exact raw output equality and <20 GiB peak allocated GPU memory (headroom on the 24 GiB card). This is a worthwhile engineering step, not a claim of practical deployment. Quality mechanism signal: >=.15 mean UTMOS improvement with <=5 percentage-point WER loss versus raw; the unchanged full gate determines eligibility to advance, regardless of that mechanism signal. Report every gain alongside the other priority and any memory tradeoff.

Freeze candidate only after a joint pass and a measured feasible next-stage budget. This pilot does not automatically launch validation. If no joint pass, stop scientifically, retain results and identify the next hypothesis. No teacher/student training is authorized by this plan alone, and none is needed to test this hypothesis; broader user authorization remains in force.

## Accounting and ceiling

Separate preprocessing/teacher data, training (zero here), generation, incremental spectral repair, model loading/warmup and evaluation. Measure per-recording wall seconds, seconds per second of original audio, paired p50/p95 and aggregate throughput, CUDA-event elapsed time for each phase (GPU stream elapsed time, not kernel-active utilization or energy), peak allocated/reserved memory, process GPU allocation time and sampled device utilization/memory. Trace source/decoder counts explicitly. Record missing historical memory/active-GPU evidence as unavailable.

Prelaunch forecast uses the relevant measured FV50 cohort throughput: 941.2 s per 48 original loops; one-forward forecast 871.1 s if the engineering ratio generalized (uncertain). Both paired paths total about 30.2 minutes; allow 5 minutes scoring, 3 minutes initialization/preprocessing, and overhead. New detached cycle ceiling: **45 minutes (2,700 s)**, 30-second heartbeats, persistent stage logs, explicit child and overall exit receipts, shared research lock, no external notifications. Do not restart over partial outputs. The 45-minute ceiling is a bounded experiment budget, not a deployment latency requirement.

Before the launch, save conditional forecasts for 1h/10h teacher corpora and 1,000 new clips using measured FV50 throughput (about 2.61 GPU-allocation hours per audio hour, and 5.45 hours/1,000 similar clips, loop only). Update these from the paired 48-point run. Spectral repair's cheap marginal inference does not eliminate FV50 optimization; any future student must account separately for teacher creation, actual training and measured retained attack/quality/transfer.
