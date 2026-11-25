# Execution and reproduction guide

All new code, data, logs, status files, checkpoints, and reports belong under `/workspace/improvement`. The original repository, model assets, and historical runs are read-only inputs for this research cycle. This guide records the current implementation; results belong in the [execution report](/workspace/improvement/05_execution_report.md).

## Current detached queue

The [first continuation supervisor](/workspace/improvement/continue_development.py) was launched detached as PID **121842**. It completed R1 formal scoring, preserved a cached-reference mismatch, then stopped when R3's content-gradient calibration exposed an upstream inplace-autograd error. The separately registered [autograd continuation](/workspace/improvement/run_autograd_repair.py) uses the same lock, detached stage wrapper, and status machinery, with R3 name `r3-waveform-autograd-repair`. Consult the live [JOB_STATUS.md](/workspace/improvement/JOB_STATUS.md), [JOB_STATUS.json](/workspace/improvement/JOB_STATUS.json), and [progress.json](/workspace/improvement/execution/progress.json) for the active PID and phase. PIDs recorded here are launch-time receipts, not proof that a process is still alive. Status timestamps use Asia/Taipei; the supervisor refreshes them every30 seconds while running.

The queue owns a nonblocking file lock at `/workspace/improvement/execution/continuation.lock` and executes one GPU stage at a time:

1. Native source/MOS evaluation of the already completed R1 deletion development24 generation.
2. R3 default cached-reference preflight, preserving its outcome.
3. The bounded R3 run: default reference if preflight passes, otherwise the specifically guarded named strict-reference repair described below.

Every stage uses the existing [run_logged.py](/workspace/scripts/run_logged.py) wrapper. Its records are `/workspace/improvement/execution/jobs/<stage>/job.json`, `stage.log`, and `supervisor-output.log`. The outer detached output is [continuation-supervisor.log](/workspace/improvement/execution/continuation-supervisor.log). Notifications are disabled with `BEYONDWAVEFORM_NOTIFY_DISCORD=0`.

Read-only status checks:

```bash
cat /workspace/improvement/JOB_STATUS.md
cat /workspace/improvement/JOB_STATUS.json
tail -n 40 /workspace/improvement/execution/continuation-supervisor.log
```

Do not launch a second queue or repeat completed calibration, optimization, training, or evaluation commands. The supervisor rejects an already locked queue and completed job receipts; individual runners also preserve existing evidence. A stale heartbeat requires checking the recorded process/job state before deciding that a stage stopped. A chat limit alone does not establish process failure.

## Frozen implementation and environment

Use `/opt/venv/bin/python` from `/workspace`. The frozen [plan.json](/workspace/improvement/execution/plan.json) records source/model/input hashes, source-only scope, package versions, hardware, scorer contracts, and the initial6 GPU-hour ceiling. R3 training is separately capped at3600 wall seconds, seed11, one-hour training corpus, and12000 updates. Registered hashes must continue to match; changing scientific code requires an explicitly named new registration before outputs.

| Component | Current implementation / evidence |
|---|---|
| Shared frozen inputs and baseline | [execution_runtime.py](/workspace/improvement/execution_runtime.py), [baseline preflight](/workspace/improvement/execution/baseline-preflight/result.json) |
| R1 native N-best risk | [r1_sequence.py](/workspace/improvement/r1_sequence.py), [original runner](/workspace/improvement/r1_real_runner.py) |
| One declared R1 deletion repair | [r1_deletion_repair.py](/workspace/improvement/r1_deletion_repair.py), [r1_deletion_runner.py](/workspace/improvement/r1_deletion_runner.py), [registration](/workspace/improvement/execution/r1-native-plus-deletion-registration.json) |
| R2 constrained feature geometry | [r2_geometry.py](/workspace/improvement/r2_geometry.py), [r2_real_runner.py](/workspace/improvement/r2_real_runner.py), [full24 registration](/workspace/improvement/execution/r2-canonical-zero-uniform-dev24/registration.json) |
| R3 bounded waveform generator | [r3_waveform.py](/workspace/improvement/r3_waveform.py), [r3_real_runner.py](/workspace/improvement/r3_real_runner.py), [CPU static review](/workspace/improvement/execution/r3-static-review.md) |
| R3 value-preserving autograd execution repair | [r3_encoder_repair.py](/workspace/improvement/r3_encoder_repair.py), [separate v2 runner](/workspace/improvement/r3_real_runner_v2.py), [actual-upstream regression](/workspace/improvement/test_r3_encoder_repair.py), [5-test receipt](/workspace/improvement/execution/r3-encoder-repair-tests.log) |
| Formal source/MOS evaluation | [source_evaluate.py](/workspace/improvement/source_evaluate.py) |

Recorded dependencies: PyTorch/torchaudio2.4.1+cu121, CUDA12.1, NumPy1.26.4, openai-whisper20240930, transformers4.46.3, torchmetrics1.6.1, jiwer3.0.5, librosa0.10.2.post1, onnxruntime1.20.1, resampy0.4.3, scipy1.14.1, and soundfile0.12.1. The observed GPU is one TITAN RTX with24GB VRAM. The [repository package definition](/workspace/pyproject.toml) supplies dependency ranges; exact observed versions and model/scorer hashes are in the plan. Reuse this environment rather than upgrading packages during a registered trial.

Whisper-small is the sole ASR used here. R1/R2 use the pinned WavLM6 cache and regular HiFiGAN vocoder. R3 loads the same pinned WavLM checkpoint/layer for content loss and loads no vocoder. Quality uses the frozen DNSMOS OVRL, NISQA MOS, and UTMOS strong checkpoints; scores are arithmetic utterance means from saved16kHz float32 WAVs.

## Generation and evaluation are separate numerical contracts

Generation/training runs in a fresh strict process with:

```bash
PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 BEYONDWAVEFORM_NOTIFY_DISCORD=0 /opt/venv/bin/python -u /workspace/improvement/r3_real_runner.py preflight
```

This is a command reference for the existing queued stage, **not an instruction to run it again**. Strict generation uses deterministic algorithms and mathematical attention. R3 uses explicit reflection for STFT and native CPU linear-interpolation derivatives with device copies where strict CUDA backward is unavailable; their costs are measured.

Formal source/MOS scoring runs in a fresh process with `CUBLAS_WORKSPACE_CONFIG` absent. For example, the current queued R1 evaluation is equivalent to:

```bash
env -u CUBLAS_WORKSPACE_CONFIG PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 BEYONDWAVEFORM_NOTIFY_DISCORD=0 /opt/venv/bin/python -u /workspace/improvement/source_evaluate.py --name r1-deletion-development24
```

The helper establishes and verifies the historical native backend before and after predictions. Generation-time strict-backend transcripts are diagnostics and must not replace these formal scores. Do not use `execution_runtime.evaluate_records` for formal evaluation. The helper refuses to overwrite a completed `source-summary.json`; an authorized repeat requires a new output directory and preserved previous results.

## R3 reference guard, training, and interruption

Default preflight checks three calibration clips for native/adapter bitwise equality, strict repeatability, and compatibility with cached clean features. A cached-reference mismatch must stop before penalty calibration or training and preserve `r3-waveform/preflight.json`, conformance details, and failure receipts.

The supervisor permits exactly the named `/workspace/improvement/execution/r3-waveform-reference-repair` variant only when all three clips pass native/adapter and repeatability checks, the recorded failure mechanism is `cached content representation mismatch`, and both calibration/training are recorded as unexecuted. It then executes:

```bash
PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 BEYONDWAVEFORM_NOTIFY_DISCORD=0 /opt/venv/bin/python -u /workspace/improvement/r3_real_runner.py run --name r3-waveform-reference-repair --allow-new-content-reference
```

This is also queue provenance; do not start it manually while the queue exists. The repair keeps checkpoint, layer, and audio fixed, regenerates clean references under the same strict transform used for adversarial features, and makes no old-cache numerical-parity claim. Other preflight failures require inspection; the supervisor stops instead of inventing a repair.

That first reference-repair attempt then failed before training because the pinned encoder's positional-convolution input aliases a tensor later modified with `x += x_conv`. The failure remains at [run-failure.json](/workspace/improvement/execution/r3-waveform-reference-repair/run-failure.json). The separately registered v2 adapter clones only that positional-convolution input in differentiable calls and removes its temporary hook in `finally`. Five CPU tests on actual pinned upstream WavLM reproduce the old error and verify native bitwise forward values, finite gradients, agreement with an independent out-of-place reference, and no leaked hooks. The new detached helper runs the v2 command below after binding its code, test receipt, and prior failure; it does not change loss, optimizer, model weights/layer, or budget:

```bash
PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 BEYONDWAVEFORM_NOTIFY_DISCORD=0 /opt/venv/bin/python -u /workspace/improvement/r3_real_runner_v2.py run --name r3-waveform-autograd-repair --allow-new-content-reference
```

This command is provenance for the queued execution. Do not launch it a second time. The initial static review was not a GPU gradient proof; both failed preflights remain evidence, and the repaired full-checkpoint preflight still must pass before training.

The active R3 folder, selected in JOB_STATUS.json as `r3_name`, contains `registration.json`, `preflight.json`, `penalty-calibration.json`, `status.json`, `training-trace.jsonl`, `training.json`, `checkpoints/`, `generated.jsonl`, and final `source-summary.json` as each stage completes. Longest-training-clip fit is recorded before the separate training-hour clock; checkpoint source/MOS evaluation is charged to that hour. Checkpoints are source-selected every500 updates; otherwise terminal output is explicitly a feasibility checkpoint. The initial untrained generator and clean identity are controls.

Completed preflight/training receipts can be reused. **Interrupted incomplete training cannot resume in place.** Its trace and checkpoints must be preserved; the runner rejects that folder, and a restart requires a named registered variant. Do not erase a partial trace to force a fresh run. Existing completed summaries are stronger evidence than a stale outer status, so inspect both before restarting anything.

## Inputs, outputs, and verification

- Development: [development24](/workspace/improvement/execution/development-24.jsonl), [calibration3](/workspace/improvement/execution/calibration-3.jsonl).
- Training: [train1h](/workspace/improvement/execution/data/train-1h.jsonl), [selection receipt](/workspace/improvement/execution/data/train-1h.receipt.json). All286 utterances are untruncated; maximum17.615 seconds. The nested10-hour manifest is acquired but is not trained in this initial run.
- Sealed final reserve: [200-clip manifest](/workspace/improvement/execution/data/final-reserve-200.jsonl), [receipt](/workspace/improvement/execution/data/final-reserve-200.receipt.json). Confirmation48, final200 audio inference, held-out ASRs, and human recruitment remain outside this cycle.
- R1 canonical failure: [result](/workspace/improvement/execution/r1-calibrated-development24/result.json); preserve the original all-correct candidate-set failure. R1 repair: [calibration](/workspace/improvement/execution/r1-deletion-calibration/result.json), [full24 generation](/workspace/improvement/execution/r1-deletion-development24/result.json), and its generated/audio/score files in that same folder.
- R2: [formal full24 summary](/workspace/improvement/execution/r2-canonical-zero-uniform-dev24/source-summary.json), plus generated WAVs, final geometry state, diagnostics, and score records in that folder.
- Verification: [original87-test receipt](/workspace/improvement/execution/implementation-tests-current.log), [new5-test upstream-autograd regression](/workspace/improvement/execution/r3-encoder-repair-tests.log), [initial independent audit](/workspace/improvement/execution/audit_initial_execution.md), [resumption audit](/workspace/improvement/execution/resume-audit.md), [package validation](/workspace/improvement/artifact_validation.json).

For an authorized CPU verification refresh, use `PYTHONDONTWRITEBYTECODE=1 /opt/venv/bin/python /workspace/improvement/validate_package.py` and `PYTHONDONTWRITEBYTECODE=1 /opt/venv/bin/python /workspace/improvement/execution/resume_audit.py`. These refresh their own verification reports inside improvement; they do not execute neural inference. The original suite passed87 tests and the new autograd regression passed5; do not repeat unchanged tests without a new change or unresolved concern. If needed after such a change, target the applicable `/workspace/improvement/test_*.py` modules with pytest `-p no:cacheprovider` and bytecode disabled.
