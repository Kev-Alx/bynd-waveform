# BeyondWaveform: strict implementation parity audit

**Verdict: complete mathematical, architectural, and evaluation parity is not established.** The implementation correctly realizes the paper's main feature-vocoder objective structure. The saved headline experiment uses a different feasible set and perceptual weight, while several consequential details are unspecified in the paper. These are distinct from implementation defects. The repository already labels its work as an independent reproduction; that qualification is justified.

Audit date: 15 October 2026, Asia/Taipei. Repository baseline: `d7f70e8a717db7feba35b69dacabade7de196b25`; working tree was clean before this audit. Source of record: [supplied PDF](/workspace/BeyondWaveform.pdf), 11 pages, SHA256 `fa1754d1f985d2791c1119fa906fe3baafa340db8a5f9a85ed97be83abf17eee`. Equations and tables were checked against a fresh PDF extraction, with the repository's paper transcription used for navigation. Figure 8's extracted image was also inspected.

No attack implementation, model configuration, checkpoint, or historical result was changed. New files contain only this report and audit probes/evidence. The audit used CPU for pretrained-model probes.

## Findings, in order of consequence

| ID | Classification | Finding | Consequence |
|---|---|---|---|
| A1 | **Confirmed experimental mismatch; high** | The 999-utterance run uses `rho=0.5`, `lambda_perc=0.1`; the paper uses `0.1`, `1`. | The reported 74.106% WER does not establish reproduction of the paper's 75.43% at its stated budget. |
| A2 | **Unresolved mathematical parity; high** | Transcript NLL is a token mean; the paper does not specify its reduction. | Relative ASR/perceptual weighting depends on the reduction and transcript length. |
| A3 | **Unresolved architecture/preprocessing parity; high** | WavLM checkpoint has `normalize=True`, effective wrapper uses `False`; layer 6 and the kNN-VC generator are independent choices. | This is a verified compatible path, but it cannot be identified with the authors' encoder/vocoder path. |
| A4 | **Unresolved ASR protocol parity; high** | Custom text normalization and partially implicit decoding settings differ from other reasonable evaluation recipes. | The same saved hypotheses produce different WER values under different normalizers. |
| A5 | **Unresolved quality parity; medium** | Correct score channels are selected, but author versions, calibration, preprocessing, and exact evaluation pool are unavailable. | Similar metric names and thresholds do not establish numerical comparability. |
| A6 | **Confirmed configuration defect; medium** | Spectral sample rate can disagree with the actual fixed 16 kHz audio pipeline without rejection. | Valid-looking configurations can penalize the wrong physical frequency band. Current saved configurations use the correct rate. |
| A7 | **Incomplete evaluation coverage; high for full-paper parity** | Several defenses and baselines require external implementations/artifacts; raw transfer results do not cover them. | The paper's defended robustness and aggregate improvement claims are not reproduced by this code alone. |

### A1. The saved experiment is outside the paper's stated feature budget

Paper §5 (p.4), Appendix A (p.10), and Table 2 (p.11) specify 50 updates, `rho=0.1`, and `lambda_perc=1`. [The frozen configuration](/workspace/configs/english_selected.yaml:18) and [actual generation metadata](/workspace/reproduction-runs/english-feature-test-999/run.json) instead use 50 updates, `rho=0.5`, `lambda_perc=0.1`, Adam LR 0.05.

All **999 recorded final relative radii** exceed 0.1: minimum **0.124354**, median **0.166802**, maximum **0.191633**. Thus the larger configured radius is not merely unused metadata: the traces report outputs outside the paper's default feasible ball. These are verified trace values, not an independent reconstruction of the perturbations; `save_delta=false` and no delta tensors were retained.

The independent validation selector follows the stated selection principle and legitimately can select different settings on different data/models. That does not make its selected point identical to the paper's point. The paper-setting candidate exists in the 100-clip validation grid and reports 72.556% WER; that is a validation observation, not a matched 999-clip paper-setting test result.

**Resolution:** retain the existing independent result, and separately evaluate a frozen paper-setting configuration on the same test pool. Do not select new settings using test results. Retain final deltas, or equivalent verifiable perturbation evidence, if independently checking Eq.6 on saved outputs is required.

### A2. Mean NLL changes the effective perceptual weight

[Loss reduction](/workspace/beyondwaveform/losses.py:34) supports sum and mean, but [the selected configuration](/workspace/configs/english_selected.yaml:14) uses mean. For K lexical tokens, let `S = sum_j -log p(y_j | y_<j, x)` and let R denote the complete perceptual regularizer. The implemented scalar objective is

`L_code = -S/K + lambda_perc * R`.

Multiplication by positive K gives the same constrained minimizers as

`-S + (K * lambda_perc) * R`.

This is not the same relative weighting as sequence-sum NLL with the same numeric lambda. It also does not imply identical finite-step optimizer trajectories after rescaling. On the saved test references, K ranges from **2 to 163**, with median **32** and mean **40.354**. At lambda=1, the mean-NLL objective has a perceptual coefficient K times larger than a sum-NLL objective with lambda=1.

The paper calls the quantity transcript NLL and specifies lexical-token exclusion, but does not give a reduction formula. Therefore this is a **parity blocker, not proof that mean reduction is an author-code mismatch**. Alpha=1, epsilon=1e-8, Adam, learning rate, zero initialization, and final-iterate selection are also independent assumptions rather than recovered author settings.

**Resolution:** obtain the authors' reduction and update recipe. Until then, record the explicit objective and assess sum/mean sensitivity on validation data with consistently interpreted weights.

### A3. Encoder normalization and vocoder compatibility must be resolved together

[The wrapper](/workspace/beyondwaveform/models/feature.py:14) applies utterance layer normalization only when its own flag is true. [The loader](/workspace/beyondwaveform/models/feature.py:61) reads the checkpoint config but independently defaults that flag to false. The actual checkpoint has `normalize=True`, `extractor_mode='layer_norm'`, and `layer_norm_first=True`.

The [official WavLM usage example](https://github.com/microsoft/unilm/blob/master/wavlm/README.md?plain=1) explicitly applies waveform layer normalization when `cfg.normalize` is true. Internal convolution/transformer normalization does not itself implement that waveform preprocessing. However, the selected [local kNN-VC extraction path](/workspace/artifacts/sources/knn-vc-c616845c4e309e24d5927f15adbdf277a3d65358/matcher.py:80) also passes the waveform directly to WavLM without this external normalization. The implementation intentionally follows that path for its WavLM-conditioned vocoder.

A real pretrained probe on pilot utterance `1919-142785-0048` found a **0.4570 relative L2 change in features** when toggling waveform normalization. This is a materially different representation. It is not, by itself, a quality comparison or evidence that either choice is better.

Layer 6 is one-based (`output_layer=6` becomes transformer layer index 5 upstream), with 1024 channels. The selected HiFi-GAN accepts `[B,T,1024]`, applies its native linear projection, and produces 320 samples per feature frame at 16 kHz. Strict state-dict loading succeeds. It is a feature-conditioned generator, not a mel-conditioned HiFi-GAN mistakenly fed SSL vectors. The paper names WavLM-Large and HiFi-GAN but does not identify layer, normalization, generator variant, training data, or checkpoint hashes.

**Resolution:** treat encoder preprocessing, feature layer, and vocoder weights as a joint contract. Do not blindly enable normalization or substitute a generic HiFi-GAN. Obtain the author artifacts, or label and validate the chosen pair as an independent implementation. The saved reconstruction control is reassuring: reconstruction WER **3.727%**, versus clean **3.964%**, on the selected 999 clips.

### A4. ASR normalization, teacher forcing, and decoding are not a recovered paper protocol

[Evaluation normalization](/workspace/beyondwaveform/evaluation.py:16) performs NFKC, lowercasing, punctuation-to-space, and whitespace folding. It does not canonicalize numbers, expand contractions, or handle abbreviations like Whisper's English normalizer. For example, `DON'T` versus `do not` scores 100% WER with this normalizer and 0% after Whisper English normalization. Neither recipe is specified by the attack paper.

Recomputing **the same 2,997 saved reference/hypothesis pairs**, without rerunning recognition:

| Variant | Repository corpus WER | Whisper English normalization, corpus WER | Repository normalization, mean utterance WER |
|---|---:|---:|---:|
| Clean | 3.963565% | 3.407634% | 4.910914% |
| Reconstruction | 3.727228% | 3.190650% | 4.779716% |
| Adversarial | 74.106352% | 73.690699% | 77.032156% |

The alternative columns are sensitivity diagnostics, not replacement official scores. The paper does not specify its normalizer or micro-versus-macro aggregation.

[Teacher forcing](/workspace/beyondwaveform/models/whisper.py:12) is correctly next-token aligned: `[SOT, language, transcribe, no_timestamps]` conditions the first lexical target; labels are shifted by one; prompt/language/task/no-timestamps/EOT targets are excluded. Only lexical positions enter cross-entropy, while the full vocabulary remains in the softmax denominator. EOT exclusion is consistent with the paper's exclusion of special tokens. Native log-Mel preprocessing remains differentiable.

However, training targets are `text.strip()` only, while evaluation targets are normalized. Every saved English reference is uppercase. Raw uppercase targets average **40.354 Whisper tokens**, versus **22.686** after lowercasing. For the pretrained pilot reconstruction, raw uppercase / lowercase / sentence-case mean NLL was **2.433855 / 3.314165 / 2.653299**. This shows sensitivity to orthographic conditioning; it does not establish which form the paper used or which has better attack transfer.

[Recognition](/workspace/beyondwaveform/models/whisper.py:54) uses native autoregressive `whisper.transcribe`, forced language, transcription task, float32, temperature 0, beam 5, and no previous-text conditioning. It does not mistake teacher-forced argmax for decoded ASR output. The installed Whisper version additionally defaults to timestamp generation, token suppression, `no_speech_threshold=0.6`, `logprob_threshold=-1`, and `compression_ratio_threshold=2.4`. Scalar temperature 0 disables the usual multi-temperature retry sequence. The teacher-forced objective uses a no-timestamps prefix, while inference leaves timestamps enabled. This is a surrogate/decoder conditioning difference, not automatically an error: the author decoder is unknown.

CTC targets use each checkpoint's processor and greedy collapse decoding without an external language model. The chosen Whisper large checkpoint is explicitly **large-v2**; the paper says only “large.” The CTC IDs/revisions are explicit independent choices.

**Resolution:** freeze the complete effective decoder options, transcript policy, tokenizer, and aggregation recipe, rather than recording only beam and temperature. Author parity requires the missing scoring protocol. Treat Mandarin whitespace removal and the handling of digits, punctuation, and script variants as equally unresolved; the CER edit-count implementation itself is conventional.

### A5. Quality channels are correct, but author checkpoint parity remains unknown

[Quality scoring](/workspace/beyondwaveform/evaluation.py:42) selects:

| Metric | Exact implementation here | Selected output |
|---|---|---|
| DNSMOS | TorchMetrics 1.6.1, non-personalized `sig_bak_ovr.onnx`, polynomial calibration; P.808 model also loaded | Channel 3: **OVRL**, from `[P808,SIG,BAK,OVRL]` |
| NISQA | TorchMetrics 1.6.1, `nisqa.tar` | Channel 0: **overall MOS** |
| UTMOS | SpeechMOS revision `ed25eacbfa42b99156c36ebec67a733b5dbb9b79`, `utmos22_strong_step7459_v1.pt` | UTMOS22 strong learner; native temporal averaging and `*2+3` mapping |

There is **no demonstrated channel-index bug**. Figure 8 explicitly labels DNSMOS as OVRL, supporting the selected channel for that figure. Table 4 does not separately identify its checkpoint/calibration. UTMOS strong learner is not an identification of every possible UTMOS system/ensemble; the paper does not specify its variant. The local installed metric implementations were inspected directly, including output ordering and calibration.

All scorers receive saved, mono float32 16 kHz audio before any ASR defense. The wrapper does not peak-normalize, loudness-normalize, clip, or silently equalize duration. DNSMOS internally repeats short signals until at least 9.01 seconds, scores 9.01-second windows at one-second hops, then averages; **722/999** clean and attacked clips are below 9.01 seconds. NISQA uses its checkpoint's mel/segmentation parameters at the supplied rate. UTMOS resamples to 16 kHz internally and returns its calibrated scalar. Corpus quality is an arithmetic mean of utterance scores.

Historical DNSMOS/NISQA files were absent from their recorded home-cache paths in this session. Temporary downloads matched the recorded SHA256 hashes exactly before inference. UTMOS's local checkpoint hash also matched. Re-running all three scorers on the first saved clean and attacked clip reproduced historical scores exactly for DNSMOS and UTMOS, and within **9.54e-7 MOS** for NISQA. This supports internal scoring reproducibility, not identity with the paper's scorers.

A remaining reproducibility defect is that [quality provenance](/workspace/beyondwaveform/evaluation.py:75) records model hashes/channels but no scorer runtime versions or preprocessing recipe, and [evaluation metadata](/workspace/beyondwaveform/pipeline.py:141) does not record its own environment. Generation metadata records some package versions, but evaluation can run later elsewhere; `librosa` and `onnxruntime` are not pinned in `pyproject.toml`. DNSMOS/NISQA cold-cache downloads use mutable upstream branches. Frozen-run scripts compare recorded model hashes, which is useful, but do not establish full evaluation runtime identity.

**Resolution:** record and validate a scorer manifest including exact weight hashes, implementation/dependency versions, channel, personalized/calibration mode, windowing/repetition, resampling, amplitude policy, and aggregation. Preserve the current results; there is no evidence here that choosing another channel would be a correction.

### A6. The spectral sample rate is not bound to the actual waveform rate

[Audio loading](/workspace/beyondwaveform/io.py:68), Whisper, and the chosen vocoder use 16 kHz, but [SpectralConfig](/workspace/beyondwaveform/config.py:7) accepts other valid rates. [HF bin frequencies](/workspace/beyondwaveform/losses.py:30) are then computed from that independent value. [Generation](/workspace/beyondwaveform/pipeline.py:75) passes the spectral config through without a consistency check.

A 5 kHz tone sampled at the actual 16 kHz rate yields HF fractions **0.0000989** with `sample_rate=16000`, versus **0.9998853** with `sample_rate=22050`, despite a nominal 6 kHz cutoff in both configurations. The latter configuration silently treats approximately 4.354 kHz of real audio as the 6 kHz boundary.

**Resolution:** reject rate mismatches against a shared audio/model sample-rate contract, or propagate a supported rate with actual resampling and compatible model frontends. All inspected saved English runs use 16000, so this defect does not explain their reported mismatch with the paper.

### A7. Full defended/model/evaluation parity is unavailable

The code supports raw Whisper-family and CTC evaluation on the same exported waveforms, plus independent waveform baselines and loss ablations. [The defense dispatcher](/workspace/beyondwaveform/evaluation.py:86) only directly implements no defense and an explicitly independent two-pole LPF. CE-AT, DT/DE-AT, DMW, TI, WaveGuard, AudioPure, and PVP Vote require external implementations/checkpoints; Muting and Sloth also require supplied extensions. Extension hooks do not constitute reproduced algorithms.

Mandarin surrogate training uses AISHELL development data with explicit train/validation separation, consistent with the paper's broad split rule, but its training recipe and weights are independent. A completed training job is not evidence of completed Mandarin attack/defense evaluation. Human intelligibility and physical playback claims require measurements beyond digital MOS/WER.

The saved ablations also do not reproduce the paper's qualitative pattern uniformly: removing all perceptual penalties gives **75.372% WER** and slightly higher average quality than the full independent run's **74.106%**, while the paper says that removal reduces both attack effectiveness and quality. Because configurations/artifacts differ, this is a failure to reproduce that empirical conclusion, not proof of incorrect loss arithmetic.

## Mathematical and pipeline checks that passed

| Paper requirement | Code / verified behavior | Verdict |
|---|---|---|
| Eqs.2,4,5,7,15: `V(E(x)+delta)` | Detached unpadded `[T,D]` clean features; same-shaped delta; no kNN replacement or additive waveform residual | Pass |
| Eq.6: relative Frobenius ball | Global L2/Frobenius projection onto `rho*(norm(q)+eps)` after each update; no coordinatewise feature clipping | Pass for configured rho; saved rho differs |
| Eqs.9–10: maximize correct-transcript NLL | Minimizes `-nll`; lexical targets only | Sign/mask pass; reduction unresolved |
| Eq.11: detached clean denominators | `TV(q)` and `HF(x)` cached from original clean features/audio, not `HF(V(q))` | Pass |
| Eq.11 weights | `lambda_perc*(tv_ratio + alpha_hf*hf_ratio)` | Placement pass; alpha value unspecified |
| Eq.12: temporal squared variation | Sum squared adjacent differences over feature dimensions, divide by `T-1`; computes TV of `q+delta`, not delta alone | Pass |
| Eq.13: HF energy fraction | Squared complex magnitude; sum numerator and denominator across frequency/time; strict `f>6000`; no dB/magnitude substitution | Pass under declared STFT convention |
| Eq.13 frequency details | 16 kHz, FFT=1024 gives 15.625 Hz spacing; bins 385–512 contribute, beginning at 6015.625 Hz | Correct for inspected configuration |
| Eq.13 numerical conventions | Hann, hop=256, win=1024, centered reflect padding, one-sided STFT; no epsilon inside HF ratio, silence rejected | Consistent implementation; author settings unknown |
| Eq.14 gradients | Only delta optimized; encoder detached; vocoder and ASR frozen but differentiable with respect to inputs | Pass with actual pretrained models |
| Eq.15 export | Native vocoder output, tanh amplitude bound, float WAV, no post-export quantization/clipping | Pass for selected generator |
| Sample rates/shapes | Mono 16 kHz input; 1024-channel layer-6 features; generator outputs `320*T` samples at 16 kHz; native 80-bin Whisper-small frontend | Pass for selected artifacts |
| Corpus WER/CER | Sum S,D,I across utterances, divide by total reference units, multiply by 100; no 100% cap | Pass |
| Appendix A selection | Same validation manifest/surrogate/scorers; all three mean quality scores >2.5; maximize surrogate corpus error | Pass for declared mean-MOS interpretation |

One pretrained pilot produced `[91,1024]` features and **29,120** output samples from **29,440** input samples. Combined-objective gradient norm into the features was **0.312933**, finite and nonzero; all encoder/vocoder/ASR weights remained frozen with no parameter gradients. The loss components were NLL **2.433855**, TV ratio **1.0**, and HF ratio **1.427824**. The non-unit reconstruction HF ratio is expected because Eq.11 references original clean audio.

The native vocoder's small duration loss is a consequence of encoder framing, not sample-rate conversion. Inputs longer than 30 seconds are rejected; no silent Whisper truncation is used. Exact author duration filtering/output alignment remain unknown.

## Measured results versus paper anchors

These values are descriptive comparisons across unmatched experimental protocols. They are not equivalence tests.

| Metric | Paper clean | Saved clean | Paper feature attack | Saved feature attack |
|---|---:|---:|---:|---:|
| WER, % | 4.750 | 3.964 | 75.430 | 74.106 |
| DNSMOS | 3.075 | 3.264 | 2.954 | 2.964 |
| NISQA | 3.714 | 4.029 | 3.526 | 3.871 |
| UTMOS | 3.755 | 4.085 | 3.590 | 2.792 |

The attack's close DNSMOS value does not establish perceptual parity: its UTMOS is about **0.798 MOS lower**, and all three clean baselines differ. Source: PDF Tables 1 and 4 and [saved evaluation summary](/workspace/reproduction-runs/english-feature-test-999/surrogate-eval/summary.json). All saved corpus counts and utterance quality averages in this table were independently recomputed.

## Paper ambiguities that must not be turned into false code findings

- §5 initially names LibriSpeech train-clean-100, but Appendix D explicitly places evaluation on a held-out 999-utterance test-clean subset. The implementation's use of test-clean follows that more specific statement. Author subset IDs and selection criteria remain unavailable.
- Appendix A prose says to reject quality scores below 2.5, while Table 2 uses strict `>2.5`. The implementation follows the table. This is not a demonstrated threshold bug.
- Figure 8 labels DNSMOS OVRL, but its full-loss bar is visually above 3.0, whereas Table 4 reports 2.954. Their evaluation pools/settings are not fully identified. Do not require one run to match both as if they were the same fully specified experiment.
- Optimizer, LR, initialization, NLL reduction, alpha, epsilon, STFT parameters, feature extraction layer/preprocessing, precise vocoder/ASR/quality checkpoints, decoder options, normalizers, and metric pooling are incompletely specified. An independent reasonable choice cannot be certified as the hidden author implementation.

## Validation evidence and limits

- Existing suite: `/opt/venv/bin/python -m pytest -q` — **59 passed**. Includes arithmetic/finite-difference, token alignment, synthetic gradients, native Whisper frontend, bounds, selection, and saved-audio integrity tests.
- Saved-score recomputation: all **2,997** main-run reference/hypothesis pairs, corpus edit totals, and per-utterance quality means; score-file hash checked against the saved summary.
- Actual pretrained probe: one local pilot clip through WavLM-Large, feature-conditioned HiFi-GAN, Whisper-small, and the complete scalar loss; normalization and transcript-case sensitivity included.
- Scorer probe: one saved clean/attack pair, verified audio and model hashes, actual DNSMOS/NISQA/UTMOS inference, direct channel comparison.
- No full 999-clip ASR or quality rerun, attack sweep, defense training, human study, or OTA experiment was performed. Radius conclusions use saved traces because deltas are absent. Statistical equivalence to paper results was not tested.

Evidence files: [score recomputation and source hashes](/workspace/verification/strict-parity-audit/results.json), [pretrained-model probes](/workspace/verification/strict-parity-audit/model-probes.json), [quality probes](/workspace/verification/strict-parity-audit/quality-probes.json). Reproducible probe sources are in the same directory: [score probe](/workspace/verification/strict-parity-audit/recompute_scores.py), [model probe](/workspace/verification/strict-parity-audit/probe_models.py), [quality probe](/workspace/verification/strict-parity-audit/probe_quality.py). Run from `/workspace` with `PYTHONPATH=/workspace /opt/venv/bin/python <probe-path>`. Quality probes use a temporary cache and refuse downloaded weights whose hashes differ from the historical record.

**Acceptance condition:** complete parity requires author artifacts/protocols plus matched-setting, matched-pool measurements. The present evidence supports a functioning, internally auditable independent implementation, with a confirmed sample-rate configuration hole and explicitly unresolved reproduction choices.
