# Joint quality and efficiency pilot results

Updated 26 October 2026, 17:59 Asia/Taipei. [Protocol](20_quality_efficiency_protocol.md) · [source audit](19_source_audit.md) · [complete measurements](execution_tradeoff/results.json) · [per-recording latency](execution_tradeoff/latency_per_recording.csv).

The pilot completed normally, but no condition passed the unchanged joint gate. Selection is scientific stop; candidate is null. Validation and transfer/defense tests remain gated off.

## What was tested and why

The cheapest completed strong original setting already exceeded 75% WER, but failed UTMOS. Longer runs were much slower and did not repair quality. We tested two separable ideas: reuse the decoder graph to remove duplicate rendering while retaining the exact attack; then apply one fixed spectral repair to that 50-step attack. This is an engineering improvement and mechanism experiment, not a new generator or a novelty claim.

The earlier strong FV50 loop averaged 19.61 seconds/recording; FV1500 averaged 587.73. Historical S/D/I and spectrum diagnostics are in [saved_diagnosis.json](execution_tradeoff/saved_diagnosis.json). Insertions were important but not dominant. Higher distortion correlated only weakly with UTMOS, so spectral damage was a testable explanation, not an established cause. Absolute smoothness/high-frequency penalties do not directly preserve phonetic content.

## Recognition and quality on all 48 recordings

The cohort is unchanged: 48 exposed development recordings, 30 speakers, 360.010 seconds, 994 reference words. The fixed gate remains WER >75%, DNSMOS OVRL >=3, NISQA MOS >=3.75 and UTMOS >=3.5.

| Condition | N | WER % | DNSMOS | NISQA | UTMOS | Joint gate |
|---|---:|---:|---:|---:|---:|---|
| FV50, reused decoder graph | 48 | 83.9034 | 3.1056 | 3.9766 | 3.1981 | fail |
| STFT round-trip control | 48 | 83.9034 | 3.1056 | 3.9766 | 3.1981 | fail |
| 25% reconstruction-spectrum anchor | 48 | 72.8370 | 3.1580 | 4.0272 | 3.3604 | fail |

The repair changed pooled WER by **-11.0664 percentage points** and mean UTMOS by **+0.1624**. DNSMOS changed +0.0525; NISQA changed +0.0506. The predeclared quality mechanism signal (+.15 UTMOS with no more than 5 WER-point loss) was not met. This secondary signal never replaces the joint gate.

The anchor moves log magnitudes 25% toward the same HiFi-GAN neutral reconstruction, retaining attacked STFT phase, with one inverse transform and no phase iterations, normalization or clipping. The STFT round-trip control isolates transform/reconstruction effects. Achieved spectrum differences and spectrogram inconsistency are recorded in [achieved_spectra.json](execution_tradeoff/achieved_spectra.json); a target complex spectrum is not automatically realizable. A WER decrease is consistent with removing magnitude information used by Whisper, but does not uniquely establish mechanism. Automatic MOS and ASR do not establish what humans understand. No listening or blinded-human scores are invented.

## Matched efficiency measurements

Every clip ran both paths in alternating order on the same TITAN RTX, float32, batch one, identical cached features, source, objective, 50 Adam updates, projection and checkpointing. The same phase timing/synchronization and allocator-cache policy applied to both. Each pair and every generated raw waveform matched the frozen single50 output exactly. New fresh feature extraction was profiled separately; 48/48 fresh tensors were bit-identical to the canonical cached tensor, and all passed the frozen numerical tolerance. The canonical cache was used for the paired comparison.

| Path | Mean sec/recording | p50 sec | p95 sec | Sec/audio sec | Loop wall min | CUDA-event min | Peak allocated GiB | Peak reserved GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Original twice-render loop | 19.649 | 18.908 | 27.426 | 2.620 | 15.719 | 15.675 | 6.885 | 7.426 |
| Retained decoder graph | 16.774 | 16.191 | 21.583 | 2.236 | 13.419 | 13.373 | 6.928 | 7.459 |

Aggregate loop wall time fell **14.63%**; paired-recording bootstrap 95% interval 12.88–16.19%. This interval describes variation over these 48 recordings in one system session, not repeat-system uncertainty. The proposed >=10% reduction/<20 GiB allocated-memory engineering target was met. Peak memory changed 6.885 → 6.928 GiB. Report that memory tradeoff alongside speed. This removes exactly one decoder forward per step, not recognizer backpropagation or Whisper activation recomputation. Raw quality and attack strength remain identical.

| Phase, all 48 | Original seconds | Retained-graph seconds |
|---|---:|---:|
| decoder no grad | 147.469 | 0.000 |
| decoder graph | 149.096 | 148.883 |
| source forward backward | 448.139 | 452.938 |
| decoder vjp penalties | 192.168 | 196.884 |
| optimizer projection | 0.837 | 0.907 |
| final render | 2.962 | 2.964 |

Source forward/backward pairs remained 2,400 per path. Decoder forwards fell 4,848 → 2,448; backwards stayed 2,400. Internal checkpoint recomputation is additional work. Full per-recording wall, seconds/audio-second, GPU-stream elapsed time and memory are in the linked CSV; per-phase traces are in `execution_tradeoff/pilot/timings/`.

## Complete costs and conditional scale forecast

| Cost category | Measured seconds | Interpretation |
|---|---:|---|
| Feature preprocessing, 48 clips | 1.105 | Fresh WavLM extraction, profiled separately |
| Teacher/generation, reused graph | 805.148 | Necessary 50-step per-new-recording search |
| Spectrum repair, 48 clips | 0.563 | Additional CPU computation; search cost is still charged |
| Shared paired-record IO/setup/comparison/output overhead | 7.587 | Conservatively charged in full to candidate forecasts; includes extra comparison work |
| Model loading | 6.288 | Encoder plus source/decoder cold loads |
| Generation warmup | 0.933 | Excluded from matched loop speed estimate |
| New learned training | 0 | No student trained |
| Formal evaluation, 144 scored outputs | 764.500 | Includes source/scorer loading and overhead |
| Recognition within evaluation | 609.258 | Measured separately |
| Quality within evaluation | 149.711 | Measured separately |
| GPU-job process wall (allocation upper bound) | 2537.125 | Generation + evaluation process durations; includes CPU/setup/idle; not kernel-active time |
| Entire paired experiment | 2543.791 | Runs both generation paths; under 2,700-second ceiling |

The generation process wall duration was 0.4924 hours. Together with evaluation the GPU-job process wall upper bound is 0.7048 hours, including setup/idle/I/O. Summed recorded CUDA-event intervals were 0.6953 hours across the experiment, excluding some loading/warmup transfers. CUDA-event intervals may include host dispatch gaps and CPU work between events; neither this sum nor allocated wall hours is an exact kernel-active-hours or energy measurement. [Thirty-second sampled GPU utilization/memory](execution_tradeoff/gpu_samples.json) and peak allocated/reserved memory are retained; historical per-method peaks were unavailable.

Including fresh feature extraction, repair and all measured shared paired-record I/O/setup/comparison overhead conservatively charged to one path, the forecast for the selected implementation is 16.967 seconds/recording and 2.262 seconds/audio-second, before amortized loading/warmup. This is an accounting forecast, not a separate direct single-path deployment latency measurement. Under the same duration/text distribution, hardware, precision and batch size, **1,000 new clips are forecast at 4.715 hours**, and **ten audio hours of teacher data at 22.622 allocated GPU-hours**. A new 48-point generation job is forecast at 13.694 minutes including measured loading/warmup, before evaluation. These are conditional forecasts, not completed larger runs or a user-mandated latency limit.

Whisper still uses its fixed 30-second input context; decoder and transcript costs vary with length. Longer training clips or changed batching need their own throughput check. Teacher-corpus scoring, training a reusable student, and actual student inference/retention are additional, unmeasured costs. A future break-even calculation must divide total teacher-data plus training cost by measured per-input time saved, after both teacher and student satisfy the unchanged gates. Cheap repair does not create a fast reusable attack.

## Decision and evidence limits

The pilot completed normally, but no condition passed the unchanged joint gate. Selection is scientific stop; candidate is null. Validation and transfer/defense tests remain gated off. The raw implementation speed improvement is separately useful, but does not solve the quality failure or prove deployment scalability. Do not repeat this fixed repair fraction as a hidden sweep. The next experiment should address the measured quality/attack coupling or a better bounded recognition objective, with a matched cost forecast before launch. The data prepared for larger training remain unused in this cycle.

No 76-point validation, unseen-model transfer, original-defense persistence, untouched confirmation/final evaluation or human study ran. Existing frozen plans/code and exposed/untouched cohort boundaries are preserved. [Explicit exit receipt](execution_tradeoff/exit_receipt.json); [selection](execution_tradeoff/selection.json); [audio comparison](http://localhost:8888/).

## Prepared-data forecast with clip overhead

The prepared training recordings average about 12.8 seconds, versus 7.5 in the pilot. A fixed-context recognizer makes a duration-only forecast incomplete. Fit measured per-clip generation (including preprocessing/repair) as intercept + audio duration; this is a conditional workload model, not actual corpus throughput.

Fitted intercept 12.729 s/clip plus 0.565 s/audio-second. Leave-one-recording-out mean absolute prediction error 0.709 s (4.2% MAPE).

| Prepared corpus | Clips | Audio hours | Length-aware teacher generation hours | Count-only sensitivity hours | Duration-only sensitivity hours |
|---|---:|---:|---:|---:|---:|
| train-1h.jsonl | 286 | 1.003 | 1.578 | 1.348 | 2.268 |
| train-10h.jsonl | 2807 | 10.004 | 15.577 | 13.229 | 22.630 |

These scenarios exclude one-time loading, corpus evaluation and all later student training; count/duration sensitivities are not uncertainty bounds. Model-parameter bootstrap intervals, duration-range coverage and limits are in [prepared_data_forecast.json](execution_tradeoff/prepared_data_forecast.json). Text/duration distribution and system shifts remain unmeasured: run a representative workload check before a long launch. No prepared training job was launched.

## Publication objective

The user now explicitly asks for a contribution suitable for peer-reviewed speech/NLP research in the ACL theme. Decoder graph reuse and fixed spectrum blending are engineering/diagnostic steps, not sufficient evidence of method novelty. The next method must be positioned against the nearest primary prior work, test a falsifiable mechanism, and show quality/attack/compute gains under matched conditions; transfer, defense persistence and human intelligibility remain required evidence. See [publication positioning and contribution plan](22_publication_positioning.md). Acceptance cannot be inferred from a development pilot.


## Forecast calibration and decoded-feedback cost

The prelaunch cycle estimate including its allowance was 38.20 minutes; actual wall time was 42.40 minutes (+11.0%). Formal evaluation took 12.74 minutes versus the five-minute allowance. In particular, native beam5 source decoding averaged **4.298 s per raw50 recording**, versus 16.774 s for its full optimized generation loop. Ten full decoded-feedback refreshes would conditionally add about 43.0 s per recording, overwhelming the current search cost. The proposed lexical-coverage method must therefore profile and sharply budget feedback, test no-feedback capped-loss controls, and retain native formal scoring. A greedy or partial-decoding approximation would require its own measured validity/cost study. [Calibration receipt](execution_tradeoff/forecast_calibration.json).
