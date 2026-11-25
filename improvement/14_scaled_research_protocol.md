# Larger-model study: exceed75% WER and measure retained transfer and defense persistence

Prepared22 October2026, Asia/Taipei. The user explicitly authorized larger models, more data and longer runs. That supersedes the earlier six-hour exploratory limit. This new registered cycle is capped at **48 measured wall hours on the existing single TITAN RTX**, including qualification, generation, scoring, audits and conditional validation. It adds no paid compute or external service.

The objective is **source corpus WER strictly above75%, with the existing automatic quality floors, followed by independently measured transfer and preprocessing persistence**. A high source loss or source WER alone cannot establish the latter goals. Report each target's absolute WER, family-balanced held-out WER and candidate-minus-control gain; do not present the75% source milestone as75% on every recognizer. Speech-content preservation and broad defense robustness remain claims requiring additional evidence.

## Why the direction changes

The latest energy-envelope experiment finished normally and failed scientifically: its primary attack reached29.1751% WER while the quality-admissible original control reached49.7988%. The full-loss envelope ablation reached19.4165%. Earlier exact distortion-cap refinement left WER unchanged. Those directions are closed. Increasing their budgets or repeating small settings is not the next study.

The new study retains a speech-generative search space and tests two substantially larger interventions: **cross-model/transformation optimization** and **an independently qualified112.4M-parameter pretrained decoder with a different conditioning representation**. It includes stronger long-run original controls. These are literature-backed research experiments, not a claim that ensembling, EOT, mel optimization or diffusion is novel. [Generative prior-work review](execution_scaled/generative-prior-work.md) and [transfer/training review](execution_scaled/transfer-defense-prior-work.md) give primary sources and known collisions. In particular, DAA already explores DiffVC latent attacks; merely swapping in diffusion would not establish a new contribution.

## Models, data and access tracks

Generation uses frozen public **Whisper-small, Whisper-large-v2, and Data2Vec-audio-base-960h**. Large-v2 is the already-local1.55B-parameter checkpoint; its official SHA256 is bound. Data2Vec is pinned to `32331f3123e703528918aa688a9a38232d58c872`. The mixed autoregressive/CTC objective introduces genuinely different recognition training/model structure, but the final CTC models still share architectural and pretraining characteristics. Do not claim every evaluated model family is unseen during generation.

The new BigVGAN-v2 decoder has **112,414,512 parameters**, accepts100-band24kHz log-mel features and was pretrained on a broader audio collection. Its official snapshot is pinned to `c329ede9e9bbc100ddf5c91e2330a61921262370`; use the pure PyTorch path, because its fused inference CUDA activation has no supported backward. [The asset receipt](execution_scaled/assets.json) binds source, configuration, license and weight files. Data2Vec and BigVGAN downloads total about823MB. These are **additional-surrogate/pretraining-resource tracks**, not the paper's single-source access setting.

The four evaluation checkpoints remain Whisper-base, multilingual Whisper-medium, wav2vec2-large-960h-lv60-self and HuBERT-large-ls960-ft. Their configurations and every target weight/preprocessor file are bound before selection. **None supplies training gradients, stopping feedback or checkpoint-selection scores.** Large-v2 and Data2Vec are different checkpoints from those four. Their future evaluation is allowed only after a single candidate has passed source-only validation.

All pilot conditions retain exactly the existing48 utterances/30 speakers. The conditional larger cohort has76 further utterances/34 speakers. It is separated in utterance/hash/text but shares speakers and has historical study exposure. The untouched confirmation48 and final200 remain unused in this cycle. Fixed pilot manifest SHA256: `9a5f56f5cca6c073dbe6e2b6d804971735045ea80c4456bea0d577b7e292748a`; validation SHA256: `3747ca5b726ef899c24bc8f92e83ba44f086a537bffcf98b9bcd52cba71df30a`.

More attack-training data is prepared: the earlier disjoint one-hour and ten-hour speech pools contain286 and2,807 recordings respectively. They are **not newly trained in this cycle**. The next learned-method branch should first obtain successful nonzero per-example feature teachers, distill a temporal feature-residual policy, then fine-tune it on the ten-hour pool with the same source ensemble/transforms. Scaling the previously collapsed waveform U-Net with its unchanged training signal is not justified. Teacher collection, architecture, seeds and training budget require their own frozen registration after a useful teacher exists.

## Executed engineering qualification

Before this long-run registration, duration-selected1.94s,6.295s and24.03s cases tested actual gradients through every new source, all three training transforms, the frozen feature vocoder and BigVGAN. These are **engineering cases, not a three-point performance pilot**: no decoding, MOS or target prediction was performed. All passed. Peak allocated memory reached14,528,583,680 bytes for the source/FV path and11,996,304,896 bytes for the BigVGAN gradient check with sources resident. Model parameters remained frozen with no accumulated parameter gradients. [Probe registration](execution_scaled/probe-registration.json), [measured result](execution_scaled/probe-result.json).

Memory control uses sequential detached-audio ASR derivatives followed by one decoder vector-Jacobian product per optimization update. No source activation graph is retained while differentiating the decoder. Frozen Whisper residual blocks and BigVGAN residual blocks use value-preserving non-reentrant activation checkpointing. BigVGAN replication padding is expressed by endpoint concatenation, preserving forward values while allowing strict deterministic differentiation. Summed CTC is evaluated on CPU with the true autograd derivative; transfers and checkpoint recomputation contribute to measured runtime.

## Decoder qualification before mel attacks

Generate48 BigVGAN reconstructions with no attack. Compare clean, original FV reconstruction, original qualityFV50 and BigVGAN reconstruction under the same native source decoder and scorers. BigVGAN qualifies only if mean quality floors pass, reconstruction WER≤clean WER+2points, and UTMOS≥original FV reconstruction UTMOS−0.15. A negative result skips all mel attacks and remains visible; it does not stop the FV ensemble comparison.

BigVGAN's adaptation resamples16→24kHz, right-pads to the next256-sample hop, applies its exact pinned log-mel analysis, synthesizes with its frozen decoder, removes only declared right padding and resamples24→16kHz to the exact original sample count. A final valid-audio clamp is explicit and clipped samples are reported; there is no peak/loudness normalization. Original FV uses its existing native `320*feature_frames` output contract, which can be slightly shorter than the source at frame boundaries; that legacy difference is retained and disclosed.

## Optimization and causal controls

All searches start from zero latent residual, keep the original transcript for evaluation, and use the same clean-referenced TV/HF terms and relative radius0.1. Adam LR is0.1 for the original feature space and0.01 for log-mel; no learning-rate sweep is scheduled. Mel residuals additionally satisfy `abs(delta_logmel)≤log(2)`, and the radius/box intersection is enforced by box clipping then radial scaling toward zero. These are different representations with noncommensurate norm units, not a claim of identical perceptual budgets.

For each non-single-source method/utterance, calibrate one fixed multiplier per source at its unmodified reconstruction: multiply each source by `norm(clean_small_audio_gradient)/norm(its_clean_audio_gradient)`. Weights remain0.5 small,0.25 large-v2 and0.25 Data2Vec throughout that utterance. Raw component losses and norms remain logged. Unknown CTC reference symbols fail rather than silently altering text; all124 development/validation references passed the pinned vocabulary check. Data2Vec uses its pinned utterance mean/variance normalization and an explicit uppercase/apostrophe loss-reference policy; evaluation references stay unchanged.

At each robust update draw one transform uniformly from identity,129-tap Hann FIR low-pass6kHz, and TorchAudio16→8→16kHz resampling. The seeded schedule is fixed by utterance ID and recorded. All sources share that update's draw. This is a stochastic transformation-gradient estimate, not nine source calls per update. The training FIR and resampling paths are explicit approximations to evaluation preprocessing, not exact reproductions of the two-pole LPF or Librosa/PCM16 WaveGuard path.

Run all48 utterances under these conditions:

| Condition | Search space and sources | Updates/checkpoints | Purpose |
|---|---|---|---|
| Original qualityFV50 | Existing frozen FV, small only, LR0.02 | Reuse saved50 | Quality-admissible reference |
| Long original | Original FV, small only | Up to1500; save50/150/200/500/600/1500 | Native-step and source-backward-matched strong controls |
| Ensemble raw | Original FV, three sources, identity |500; save50/200/500| Isolate model diversity from transforms |
| Ensemble EOT | Original FV, same three sources and sampled transforms |500; save50/200/500| Test preprocessing-persistent transfer objective |
| BigVGAN ensemble EOT | Qualified larger decoder/log-mel, same sources/transforms |500; save50/200/500| Test a different trained generative search representation |

An ensemble checkpoint at50/200/500 uses150/600/1500 primary source backward calls; the long-original checkpoints supply those budgets. Calibration adds three source forward/backward calls per ensemble method/utterance. The extra decoder forward for detached gradients, activation recomputation, CPU CTC, model size and wall time are reported separately. Matched backward count does not mean matched total compute.

## Selection and automatic larger validation

Score every fixed checkpoint on the complete48-point cohort in fresh native-evaluation processes. Source WER is summed edits/summed reference words. Floors remain **DNSMOS≥3, NISQA≥3.75, UTMOS≥3.5**. No quality scorer is differentiated. No bad recordings are removed or replaced; failed/partial conditions cannot receive aggregate promotion claims.

Choose the highest source-WER quality-admissible point strictly above75%; ties use UTMOS, then fewer updates, then the fixed name order. A longer original baseline is allowed to be the best measured method, but must be labelled as a **known baseline**, not a discovered contribution. Select the strongest quality-admissible original control using only source results; if the candidate is that same original point, retain the frozen qualityFV50 comparator and explicitly identify the known-baseline outcome. Publish every stronger original checkpoint instead of concealing it.

Freeze candidate, control, checkpoint, method, source/quality evidence and validation-manifest hashes before generating76 new paired outputs. Recompute candidate/control plus clean and each method's reconstruction. Keep the exact same candidate through validation. It must again exceed75% sourceWER and meet every quality floor before any new evaluation-target prediction. If no point qualifies, finish with a negative scientific result rather than relax thresholds or silently start a setting sweep.

## Transfer and defense endpoint

After76-point qualification, run the frozen candidate/control/clean/reconstructions on all four held-out checkpoints. Score raw audio, exact two-pole LPF6kHz, pinned WaveGuard8kHz down/up+PCM16, and withheld-parameter LPF4kHz and WaveGuard12kHz. WaveGuard2kHz remains a diagnostic because it can seriously damage clean speech. Score post-defense quality, including clean controls, separately. WaveGuard resampling WER is not evasion of its full detection system.

The raw and per-defense primary transfer comparison is the original equal-family weighting:0.5 mean of the two Whisper models plus0.5 mean of the two CTC models. Report candidate/control WER and clean-subtracted attack effects per target as well as the macro result. Standard WER can exceed100%; therefore retain substitution/deletion/insertion counts, empty-output rates and a capped corpus diagnostic instead of disguising insertion effects.

For each of the five main conditions, the descriptive preservation screen requires macro candidate−control WER≥0 and the paired10,000-replicate speaker-bootstrap95% lower bound≥−3points. Report all conditions separately; no mean can hide a failed defense condition. The2kHz diagnostic does not enter this screen. These exposed-validation intervals are not untouched-final proof, a guarantee against unseen defenses, or evidence of human semantic preservation. Target outputs cannot reselect the candidate. If they motivate later redesign, this comparison becomes development evidence.

The original proposal's stronger transfer-improvement threshold remains separately visible: raw macro gain≥3points with a bootstrap lower bound>0 is reported as a transfer-superiority **signal**. Merely passing the preservation screen does not establish that stronger result or a publishable claim.

CE-AT, DE/DT-AT, DMW and TI defended checkpoints/recipes remain unavailable. The paper's sentence-ASR AudioPure/PVP settings remain unidentified. **This run cannot establish preservation of those missing paper claims.** It will report the actual named, executable transfer/preprocessing evidence. A blinded human transcription/listening study and untouched final200 still stand between an automatic-score result and a content-preserving publishable claim.

## Operation and reproducibility

All new artifacts remain under `/workspace/improvement/execution_scaled` or linked root documents. [Supervisor](supervise_scaled.py), [components](scaled_components.py), [optimizer](scaled_attack.py), [registration](scaled_scope.py), [generation](scaled_run.py), [native scorer adapter](scaled_source_evaluate.py), [independent audit](execution_scaled/audit_scaled.py), [source-only selection](scaled_postprocess.py), and [held-out evaluator](scaled_transfer_evaluate.py) are hash-bound before the long queue.

The detached process has no controlling terminal, holds the shared single-GPU lock, updates status every30seconds and wraps every stage with persistent logs and explicit exits. Strict generation sets `CUBLAS_WORKSPACE_CONFIG=:4096:8`; formal scoring uses a fresh process with it unset. Caches, downloads, generated WAVs, tensors and reports stay under the common artifact directory. Notifications are disabled. Existing/partial outputs cause a stop for inspection, not overwrite or duplicate launch.

Order: registration → decoder reconstruction48/scoring/audit/gate → long FV comparisons48/scoring/audit → qualified mel comparison48/scoring/audit → source-only freeze → conditional76 feature cache/generation/scoring/audit/gate → conditional four-target/defense evaluation and independent audit → report/package checks. The hard48-hour limit preserves partial receipts if reached. Completion and scientific success are separate statuses. [Results](15_scaled_results.md) · [Live status](JOB_STATUS.md).
