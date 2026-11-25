# Defense evaluation before WER and quality research

Reviewed 17 October 2026 (Asia/Taipei), against the supplied PDF and `STRICT_PARITY_AUDIT.md`. The priority is to measure the existing implementation under defenses before changing the attack or dataset. Historical attack artifacts, configurations, and scores remain the baseline.

The audit's main conclusions hold: the existing 999-clip run uses rho=0.5/lambda=0.1 rather than the paper's 0.1/1; token reduction, preprocessing, optimizer, decoder, and scorer choices remain unresolved. The largest quality discrepancy is UTMOS: 2.792377 versus the paper's 3.590. Changing the quality output channel is unsupported by the evidence.

## Executable conditions

The frozen suite in `/workspace/reproduction-runs/english-defenses-999/plan.json` applies these predeclared conditions to all **999 existing test IDs**, with **clean, reconstruction, and adversarial** controls. No waveform generation or attack selection occurs.

| Condition | Actual procedure | Relationship to the paper |
|---|---|---|
| Raw | Reaudit historical hypotheses, edit counts, and source WAVs | Existing independent baseline, not a new inference run |
| LPF 6000 | Existing TorchAudio two-pole low-pass, 6000 Hz, Q=0.70710678 | Independent setting; Table 1 does not specify cutoff, order, or filter design |
| WaveGuard down/up 8000 | Official down/up transformation, historical `kaiser_best` policy, upstream PCM16 clipping/truncation | Upstream default rate; the BeyondWaveform WaveGuard choice is unspecified |
| WaveGuard down/up 2000 | Same implementation at a fixed stronger rate | Predeclared sensitivity condition; no selection using test results |

The WaveGuard adapter follows `down_up_sample_np` and `save_as_wav` from the [official source](https://github.com/shehzeen/waveguard_defense) at commit **4320cfe12d0452e7f34f2903a4e302780662f5e1**. Original requirements specify librosa 0.7.2; this environment uses 0.10.2.post1, so keyword sample rates and `res_type="kaiser_best"` are explicit. `resampy==0.4.3` is separately pinned. The old and new complete dependency stacks have not been shown numerically identical. Tests execute the pinned upstream functions with a modern librosa call shim and compare the complete output, including actual upstream PCM16 file I/O, exactly against the adapter. They also check attenuation of a 6 kHz tone and preservation of a 500 Hz tone.

WaveGuard originally compares original/transformed transcriptions for **detection**, reporting ROC/AUC. These runs measure **transformed-audio WER**, consistent with the broad preprocessing category in BeyondWaveform. They do not reproduce WaveGuard's full detector, a detection threshold, its LPC/mel/filter/quantization variants, or the unidentified Table 1 configuration. The [official README](https://github.com/shehzeen/waveguard_defense#running-the-defense) explicitly offers multiple transformations and user-selected parameters.

## Conditions that cannot yet be identified with Table 1

| Defense | What primary-source inspection established | Missing prerequisite |
|---|---|---|
| CE-AT | The paper names cross-entropy adversarial training and cites general adversarial-training work | Defended ASR weights or complete training data, attack budget, objective, optimizer, epochs, and selection recipe |
| DT-AT / DE-AT | Different abbreviation in table/prose; no decoding-trajectory objective is given | Actual objective, training recipe, and defended checkpoints |
| DMW | Citation [20] describes probabilistic-margin instance reweighting | ASR adaptation, hyperparameters, and trained weights |
| TI | Citation [9] is *Evading Defenses to Transferable Adversarial Examples by Translation-Invariant Attacks* | Explanation of the claimed training defense and its weights; an input smoothing filter is not an identified substitute |
| AudioPure | [Official implementation](https://github.com/cychomatica/AudioPure), commit **903dd8554473b4fd461919876de2c5ba39747246**, uses an SC09 configuration with 16000-sample segments; `RevDiffWave` fixes `audio_shape=(1,16000)` | Sentence-compatible model/checkpoint and the paper's segmentation, diffusion time, reverse sampler, repetitions, and merge policy |
| PVP Vote | A separately recovered [Precision-Varying Prediction paper](https://arxiv.org/abs/2603.22590) varies FP32/FP16/BF16 and describes stochastic precision sampling and diversity detection; its [repository](https://github.com/blindconf/multi_precision_fusion), commit **e8e6f0d8f0b51d288edc1de9da0292ddebb5c8a7**, uses SpeechBrain | BeyondWaveform cites Acoustic-decoy [17] instead. The intended PVP identity, majority/voting unit, precisions, tie policy, and checkpoints are absent. TITAN RTX also lacks native BF16 support |

AudioPure source availability improves on the audit's general “external implementation needed” statement, but does not supply the missing sentence-ASR experiment. Feeding entire sentences to its fixed one-second SDE or silently truncating them would change the task. It needs a declared adaptation and validation before a full test run. Likewise, varying waveform quantization is not a recovered substitute for PVP's model precision.

Pinned repository trees, selected original source files, their URLs/SHA256 hashes, and the resampy installation receipt are stored in `/workspace/verification/defense-research`. No unavailable condition is given a measured result.

## Running, verification, and outputs

The first LPF attempt completed all 999 inference iterations but failed at JSONL export because a decoded em dash could not be encoded under an ASCII locale. No complete hypotheses file survived. The failure logs, original frozen plan, and original source were archived under `english-defenses-999/attempts`. The repair explicitly uses UTF-8 for JSON/JSONL I/O and defense reports, with an ASCII-locale end-to-end regression. A traceable plan amendment updates only serialization/supervision source hashes, adds Discord completion hooks, and leaves models, attack/defense settings, source WAVs, and test IDs unchanged. Relaunched processes also set `PYTHONUTF8=1` and `PYTHONIOENCODING=utf-8`.

All restarted jobs and their stages enable `BEYONDWAVEFORM_NOTIFY_DISCORD=1`, using the user's existing `scripts/discord_gpu_report.py`. A notification permission/delivery error is recorded separately and does not alter compute success. Discord notices require the job user to read the private root-owned webhook config; no webhook is printed or copied into experiment artifacts.

The shared `/opt/venv` is read-only. The optional resampler was installed without replacing shared dependencies:

```bash
/opt/venv/bin/python -m pip install --no-deps --target /workspace/artifacts/defense-runtime resampy==0.4.3
PYTHONPATH=/workspace/artifacts/defense-runtime:/workspace /opt/venv/bin/python scripts/run_defense_suite.py --prepare
PYTHONPATH=/workspace/artifacts/defense-runtime:/workspace /opt/venv/bin/python scripts/run_defense_suite.py --check-only
```

Preparation refuses to overwrite an existing plan. The runner freezes source/config/checkpoint/evaluation hashes, original IDs, package versions, and the conditions above. It rehashes all 2,997 historical WAVs and recomputes the saved raw scores before execution. Every evaluation rechecks WAV hashes. Completed conditions are independently audited for complete unique `(ID, variant)` coverage, unchanged references/audio/target/decoder, every lexical edit count, the corpus denominator of 20,310 words, and summary totals. A tampered completed result fails rather than being overwritten or silently reused.

The initial LPF evaluation is already supervised by `scripts/run_logged.py`. The suite adopts it only after checking its exact command and successful exit; it then runs the two WaveGuard conditions sequentially. Progress includes per-clip tqdm output, 30-second phase heartbeats, `progress.json`, `job.json`, and cumulative `results.json`/`report.md`. `available-conditions-complete` means the declared executable conditions finished; `complete_paper_defense_suite` stays false.

This container lacks tmux, so the persistent supervisor is launched with `subprocess.Popen(..., start_new_session=True)` and redirected file descriptors. Existing PID/job/log state must be inspected before starting another copy. Resume the frozen suite using the same `PYTHONPATH` and `/opt/venv/bin/python -u scripts/run_defense_suite.py`; do not repeat the independent LPF launcher.

Quality scoring is defined on exported audio before defense. The unchanged historical WAVs retain their existing measured quality. This job does not measure post-defense MOS, Mandarin, other target models, waveform-baseline defenses, detector AUC, human intelligibility, or physical playback.

## Research after the defense baseline

1. Run rho=0.1/lambda=1 on the existing pool as a paper-setting comparison, retaining deltas for independent radius verification. Preserve the historical selected run as a separate result.
2. Use disjoint pilot/validation data for optimizer LR/schedule/initialization, lexical sum-versus-mean NLL, transcript case/format, and unspecified STFT/alpha sensitivity. Interpret lambda consistently with NLL reduction. Freeze decoding and scorer manifests.
3. Evaluate encoder normalization/layer and vocoder as a joint compatibility contract. A raw WavLM flag change can damage the pretrained vocoder's input distribution; reconstruction WER and all three quality scores are necessary controls.
4. Select against a declared WER/quality objective on validation data. For paper-quality research, track the UTMOS gap explicitly alongside DNSMOS/NISQA and clean/reconstruction quality. Freeze the selection before test evaluation.
5. If replacing the 999 samples, define duration/speaker/seed criteria before scoring, retain all chosen IDs, audit disjointness, report clean baselines, and keep the old pool for paired comparison. Do not select clips or seeds for closeness to the published WER/MOS values.

These are next-stage experiments, not launched by the defense job. Current defense results must not become feedback for tuning a claimed black-box attack.
