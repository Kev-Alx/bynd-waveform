# Why 48 recordings took so long, and where the audio is saved

Updated 26 October 2026, Asia/Taipei. This note answers the follow-up after the handoff; no new model run was launched.

## Saved audio

Yes. Every completed 48-point performance pilot saved its audio. The authoritative outputs are **16 kHz, 32-bit floating-point WAV**, verified from the saved files. The interface plays original WAV samples, without MP3 conversion or loudness normalization.

| Study | Audio directory under `/workspace/improvement/` |
|---|---|
| Waveform mechanism pilot | `execution48/mechanism-pilot48/audio/` |
| Waveform refinement | `execution_refine/pilot/audio/` |
| Quiet-region constraints | `execution_envelope/pilot/audio/` |
| Expanded feature methods and ensembles | `execution_scaled/fvpilot/audio/` |
| BigVGAN mel methods | `execution_scaled/melpilot/audio/` |

Each directory groups WAVs by condition, then recording ID. Some clean/reconstruction/control audio is reused from an earlier study rather than copied into every later folder. Each study's **`generated.jsonl`** lists the exact audio path, hash, sample rate and length for every recording/condition; use it to find reused controls. Its `source-scores.jsonl` contains the corresponding reference, recognized transcript and quality scores.

For example, `execution_scaled/fvpilot/audio/ensemble_eot_500/` contains all 48 saved attacked WAVs for that condition, and `execution_scaled/melpilot/audio/mel_eot_200/` contains its 48 BigVGAN counterparts. The interface at **http://localhost:8888/** also offers WAV download links. Its deduplicated copies are served from `audio_browser/dist/audio/`.

## This cycle optimized recordings; it did not train a large model

For a normal prediction, a model processes a recording once. In these experiments the optimizer repeatedly changed the recording's features or mel spectrogram, rendered new speech, measured how the recognizers responded, and propagated gradients back through the recognizers and decoder. The recognizer/decoder weights stayed frozen, but computing how their outputs change with the input still costs forward and backward computation.

Each of the 48 recordings had four separate optimization trajectories:

| Trajectory | Updates per recording | Recognizers per update | Measured optimization-loop time, all 48 |
|---|---:|---:|---:|
| Strong original feature control | 1,500 | 1 | 7.84 h |
| Raw feature ensemble | 500 | 3 | 9.59 h |
| Feature ensemble with transformations | 500 | 3 | 9.60 h |
| BigVGAN mel ensemble with transformations | 500 | 3 | 18.23 h |

These times sum the saved final-checkpoint `wall_seconds_cumulative` values. They exclude calibration, loading and cohort overhead. Full feature generation took 27.07 h; full mel generation took 18.27 h. Their formal scoring took about 17.7 and 5.9 minutes respectively. The whole cycle took 45h 51m.

That is **48 × (1,500 + 500 + 500 + 500) = 144,000 optimizer updates**, and **288,000 recognizer forward/backward pairs**, plus 432 calibration backward calls. Top-level decoder counters report 288,912 forwards and 144,000 backwards. Internal activation-checkpoint recomputation adds more work beyond these top-level counters. The 50/200/500 checkpoint results are saved from their ongoing trajectories, not separate restarts for every endpoint.

The ensemble includes Whisper-large-v2 (~1.55B parameters). The implementation processes source gradients sequentially and recomputes decoder/model activations to limit memory on one 24 GB TITAN RTX. This trades runtime for memory. BigVGAN also makes the decoder part more expensive. The long duration comes overwhelmingly from repeated gradient optimization, rather than WAV saving, browser serving, or the final WER/MOS scoring.

## What successful scaling would mean

**If the final method keeps this per-recording optimization, it remains slow on new recordings.** It has no one-time training stage that removes the search. On the same GPU, settings and similar recording lengths, the measured BigVGAN branch averages about 22.8 minutes per recording; a simple 1,000-recording extrapolation is approximately 380 hours / 16 days, before additional overhead. This is a conditional arithmetic forecast, not a benchmark of a future implementation or training job.

**A learned generator is a different design.** It could train once and predict edits for new recordings in one/few forward passes. Training might still be expensive, especially if every training update differentiates through the large frozen recognizers/decoder. A teacher/student approach could instead generate expensive targets once and fit a smaller student; creating the teacher data would still have the per-recording cost. Neither training time nor student attack/quality/transfer retention has been demonstrated here.

The earlier one-hour learned-generator feasibility run produced almost-zero edits and failed to learn a useful attack. Also, the expanded per-recording methods have not yet passed the joint WER/quality gate. Therefore a fast, successful learned model is **a research hypothesis**, not a completed speedup. Establish a good quality-preserving teacher before spending a large budget distilling its outputs.

Sources: saved stage timings in `JOB_STATUS.json`; final trajectory receipts under `execution_scaled/{fvpilot,melpilot}/traces/`; loop/counter implementation in `scaled_attack.py`; original sample files and each study's `generated.jsonl`.
