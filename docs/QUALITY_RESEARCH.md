# Paper-budget WER and sound-quality research

**Completed historical batch.** The next stage is documented in [QUALITY_REFINEMENT.md](/workspace/docs/QUALITY_REFINEMENT.md). As of 18 October 2026 the user disabled Discord job notifications; historical launch/notification details below describe the earlier run only.

Prepared 17 October 2026 (Asia/Taipei), following the strict parity audit and the current-implementation defense tests. This stage studies unspecified optimization choices while retaining the paper's stated rho=0.1, lambda=1, and 50 updates.

The historical independent 999-clip attack achieved WER 74.106352% and DNSMOS/NISQA/UTMOS 2.963822 / 3.871132 / 2.792377. The paper reports 75.43% and 2.954 / 3.526 / 3.590. UTMOS is the largest quality gap. On the existing 100-clip validation pool, the paper-budget mean-NLL/LR=0.05 candidate reached 72.556% WER with UTMOS 3.3636. These are unmatched-protocol comparisons, not author parity or statistical equivalence.

## Frozen experiments

The first queued research attempt stopped before generating any clip: PyYAML treated valid JSON scientific notation `1e-08` as a string. The loader now parses `.json` files with the JSON parser and preserves the existing YAML behavior. Preflight loads every serialized candidate and instantiates the attack/spectral schemas before model loading. The failed attempt and zero-output run metadata are archived under `paper-quality-research/attempts`; `config-parsing-repair.json` records the source-only amendment. Candidate file bytes, numerical settings, manifests, model/scorer weights, promotion rules, and the completed defense plan are unchanged. The original defense config-loader source is preserved by hash for historical auditing, and completed defense configs must parse identically under both loaders.

`scripts/run_quality_research.py` writes a read-only plan at `/workspace/reproduction-runs/paper-quality-research/plan.json`. The queue waits for the full **available** defense suite to finish successfully, then verifies each defense result before starting attack experiments. This does not turn the six still-unidentified Table 1 defenses into completed measurements.

All six candidates share the existing compatible layer-6 WavLM/regular HiFi-GAN contract, waveform normalization policy, seed, uppercase transcript policy, spectral settings, raw Whisper-small decoder, Adam, zero initialization, alpha=1, and final-iterate selection. They retain actual delta tensors for independent constraint checks.

| Variable | Predeclared values |
|---|---|
| Lexical NLL reduction | mean, sum |
| Adam learning rate | 0.02, 0.05, 0.1 |
| Feature radius / perceptual weight / updates | 0.1 / 1 / 50 |

Mean and sum at the same numeric lambda have different relative perceptual weighting. For K lexical tokens, mean NLL with lambda=1 has the same minimizers as sum NLL with perceptual coefficient K; finite-step Adam trajectories need not agree. The paper does not specify the reduction or learning rate. This grid examines that ambiguity and does not claim to recover the hidden author objective.

Execution order:

1. Generate, recognize, quality-score, and audit all six candidates on the disjoint **12-clip pilot**.
2. Promote a quality-oriented candidate and an attack-strength comparator to the **100-clip validation** pool.
3. Generate and evaluate one separately predefined **999-clip test control**: mean NLL, LR=0.05, rho=0.1, lambda=1, 50 updates. This closes the audit's missing matched-pool paper-budget comparison. Its identity is frozen before new pilot/validation outcomes; it is not selected from target or test feedback.

The original test IDs remain the paired comparison pool. Split audit and original audio hashes are checked during preparation. Pilot/validation/test counts are 12/100/999. All conditions include clean, reconstruction, and adversarial controls. No new validation-selected attack is automatically evaluated on the test set in this stage.

## Promotion rule and interpretation

Pilot candidates must first have all three mean quality scores strictly above 2.5, matching the paper's broad gate. The quality-oriented promotion then maximizes WER among candidates also reaching **DNSMOS >=2.954, NISQA >=3.526, UTMOS >=3.590**. If none reaches all three targets, it promotes the candidate maximizing its weakest quality-to-target ratio and explicitly reports the target miss. The second promotion is the highest-WER remaining candidate passing the broad gate. Ties are deterministic.

These additional absolute MOS targets express the user's current quality goal; they are not the paper's original selection rule. The report also exposes all six results so a stronger-WER/lower-quality result cannot be represented as a simultaneous match. Pilot outcomes remain exploratory until validated on the larger disjoint pool.

## Scoring and audit controls

DNSMOS and NISQA checkpoints were copied into `/workspace/artifacts/quality-research-scorers` from the audit cache after matching their historical SHA256 hashes. UTMOS uses the existing hash-verified checkpoint. No home-directory variable or shared dependency was changed. Before freezing the plan, actual clean/attack fixture inference reproduced all three historical quality scores within 1e-5.

Each evaluation records the relocated scorer manifest: model hashes, runtime/dependency versions, selected channels, DNSMOS non-personalized mode, native short-audio repetition/windowing, checkpoint segmentation, 16 kHz input, amplitude policy, and arithmetic pooling. All scorers measure the exported waveform before defenses. Decoder and model artifacts are fixed across candidates; NLL reduction changes the optimization objective, not recognition decoding.

The independent run audit checks full unique `(ID, variant)` coverage, hypotheses/edit counts, corpus denominators, all three MOS means, model/config/runtime identities, and every saved WAV hash. It reloads each delta and re-extracts the original clean WavLM features to check its actual relative Frobenius radius against 0.1 and the logged final radius. Optimization traces must contain 50 finite updates. Tests include rejection of an out-of-budget saved delta.

The frozen plan hashes the relevant source, manifests, model/scorer weights, copied configs, and pinned upstream Python/JSON/YAML sources. The existing attack and defense implementations are unchanged by this new driver.

## Launch, progress, and resume

The persistent queue uses `run_logged.py` with redirected file descriptors and `start_new_session=True`. Inspect the current job/PIDs before launching another copy. The environment includes:

```text
PYTHONPATH=/workspace/artifacts/defense-runtime:/workspace
PYTHONUTF8=1
PYTHONIOENCODING=utf-8
BEYONDWAVEFORM_NOTIFY_DISCORD=1
```

Primary outputs are `/workspace/reproduction-runs/paper-quality-research/{job.json,progress.json,research.log,plan.json,results.json,report.md}`. Candidate runs live under `pilot/`, `validation/`, and `paper-setting-test-999/`. Each includes generation records, float WAVs, saved deltas, evaluation rows/summaries, and `verification.json`. Per-stage logs/statuses are in `phases/`; `pilot-promotion.json` preserves the deterministic promotion decision. Existing generation records are reused only under the existing input/artifact integrity checks, and completed evaluations are reaudited before reuse.

For a single live overview, open `/workspace/reproduction-runs/JOB_STATUS.md` or `JOB_STATUS.json`. The separate `scripts/update_job_status.py` monitor updates both every 30 seconds while the defense/research jobs are active, and stops when both are terminal. Its own status/log are under `reproduction-runs/job-status-updater`. This monitor does not change frozen experiment inputs.

Every generation, evaluation, verification, and overall completion enables the user's Discord notification hook. Delivery errors are separate from compute status and leave retryable `job.discord.json` receipts. The webhook remains root-owned/unreadable to this job user until access is provided; no successful Discord delivery is claimed while that persists.

Once started, this queue takes several hours, including the fixed 999-clip generation. Early pilot/validation results are written as each audited case finishes. Further changes to transcript formatting, update schedules, feature/vocoder contracts, or dataset sampling should be designed after inspecting these results, with fresh validation-only decisions and an explicit new freeze.
