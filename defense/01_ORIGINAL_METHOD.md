# What BeyondWaveform attacks, and what a defense must preserve

Research phase, 22 October 2026 (Asia/Taipei). Evidence labels: **paper** means stated in the supplied document; **code** means inspected local implementation; **inference** means our interpretation; **hypothesis** means a proposed explanation requiring an experiment.

Source of record: [supplied PDF](/workspace/BeyondWaveform.pdf), SHA256 `fa1754d1f985d2791c1119fa906fe3baafa340db8a5f9a85ed97be83abf17eee`. Its hash was checked in [local audit](evidence/local_audit.json). The repository [transcription](/workspace/paper/BeyondWaveform.md) was read against the publicly accessible [v1 full text](https://arxiv.org/html/2606.05678v1) and [11-page PDF](https://arxiv.org/pdf/2606.05678v1). Public metadata identifies one version, submitted 4 June 2026; the local file is not asserted to be byte-identical to the online PDF. These notes analyze the supplied version, not an assumed future revision.

## Problem and access assumptions

The paper studies **untargeted, per-utterance transcription failure** while retaining what humans hear as the original speech. English uses WER; Mandarin uses CER. Higher machine error means a stronger attack. A defense succeeds by reducing that error while retaining accurate benign recognition. There is no original proposed defense whose numerical score we should maximize.

The adversary knows a clean waveform `x` and its reference transcript `y`, has white-box access to public surrogate `f_s`, and generates one audio file without using the target `f_t`. The exported file is then reused on target recognizers and defended systems. The raw Whisper-small result uses the generation surrogate itself; transfer to other recognizers is the black-box part. **Paper:** §§3–5 and Appendix A.

The deployed defender receives only incoming audio `u`. It has neither the original `x`, its clean SSL features, the attack perturbation, nor the correct transcript. Paired clean/attacked data and labels are legitimate during defense training, but any proposed inference rule needing them is invalid for this setting.

## Mechanism and objective

A frozen SSL encoder produces `q = E(x) ∈ R^(T×D)`. A same-shaped trainable tensor `δ` changes its frame trajectory. A frozen vocoder synthesizes

`u(δ) = V(q + δ)`, subject to `||δ||F / (||q||F + ε) ≤ ρ`.

Only `δ` is optimized. In paper notation, the minimized loss is

`−NLL_lexical(y | V(q+δ); f_s) + λ [TV(q+δ)/(sg(TV(q))+ε) + α HF(V(q+δ))/(sg(HF(x))+ε)]`.

`TV(z)` is the mean over adjacent frames of squared Euclidean feature differences, summing feature channels. `HF(u)` is the fraction of STFT power above 6 kHz. The denominators come from the **original** clean signal/features and are detached. They are not computed from the vocoder reconstruction. NLL excludes Whisper prompt, language, task and other special-token targets; excluding targets does not remove those vocabulary entries from the softmax. **Paper:** Eqs. 2–15, §§4–5.

WavLM-Large is the stated encoder, HiFi-GAN the stated vocoder, and Whisper-small the English surrogate. For Mandarin, the surrogate is separately fine-tuned on AISHELL-1 development data; defended targets use training data and final evaluation uses held-out test. The encoder, vocoder and surrogate remain frozen during each attack. There is no learned universal generator in the core method. Default attack generation uses 50 updates, `ρ=0.1` and `λ=1`. **Paper:** §5, Appendices A/D, Table 2.

The local [feature wrapper](/workspace/beyondwaveform/models/feature.py) uses a compatible WavLM layer-6 / kNN-VC HiFi-GAN pair, 1024 feature channels, and native 16 kHz generation at 320 samples per feature frame. It does not perform nearest-neighbor voice conversion. [Attack code](/workspace/beyondwaveform/attacks.py) detaches `q`, differentiates through vocoder and surrogate into `δ`, and projects the **global** Frobenius norm after each update. Frozen weights do not mean detached input gradients. [Loss code](/workspace/beyondwaveform/losses.py) implements the clean normalizers and lexical mask.

## Training, selection and evaluation

The core attack does not train model weights. Preparation/fine-tuning for Mandarin and trained defense baselines are separate procedures. The paper does not identify their full recipes or defended weights.

Appendix A selects each attack's strength using a held-out validation split, the common surrogate, and DNSMOS/NISQA/UTMOS quality screens around 2.5; it freezes settings before using target or defense outcomes. The prose uses both “below 2.5” and `>2.5`, leaving equality handling unclear. That boundary should be explicit in our independent protocol. Feature sweeps are `ρ ∈ {0.05,0.1,0.5,1}` and `λ ∈ {0.1,0.5,1,2}`. Iterative baselines have 50 updates, but equal updates need not mean equal model passes or compute.

Section 5 initially names LibriSpeech train-clean-100; Appendix D clarifies a held-out 999-utterance test-clean subset for attack evaluation, with train-clean-100 used for training preparation. Original utterance IDs, validation IDs, sampling seed and much preprocessing are missing. AISHELL-1 is the Mandarin corpus. Targets include Whisper-family variants and HuBERT/Wav2Vec2 CTC. A checkpoint family name does not identify a particular model revision or decoder.

Selected **reported anchors**, not measurements from this task:

| Setting | Reported result | Interpretation |
|---|---:|---|
| English clean / feature attack on raw Whisper-small | 4.75 / 75.43 WER % | Surrogate-model result |
| English feature attack with LPF / WaveGuard | 68.39 / 57.03 WER % | Configurations not fully identified |
| English feature attack with CE-AT / DT-AT | 71.26 / 67.23 WER % | Defended weights/recipes unavailable locally |
| Mandarin raw clean / feature attack | 6.29 / 72.25 CER % | Different language/surrogate preparation |
| Feature attack DNSMOS / NISQA / UTMOS | 2.954 / 3.526 / 3.590 | Automatic quality estimators, not word preservation |
| Human transcription WER | 5.47% | Reported human intelligibility study |
| Clean / attacked over-the-air WER | 7.45 / 78.23% | Limited 100-utterance physical setup |

The paper also reports +26.6 WER transfer improvement and +36.2 WER against English training defenses relative to attack baselines. These are claims about attack performance across incompletely recoverable experiments, not certified robustness statements. The supplied specification already transcribes the complete tables; repeating every row here would not resolve the missing configurations.

## Why it can work

**Inference:** around a feature trajectory, `Δu ≈ J_V(q) δ`. The attack searches directions shaped by the vocoder, then chooses them through `J_(f_s∘V)` using transcript loss. Such directions can alter speech-relevant evidence while remaining compatible with a generator's learned output distribution. Additive-noise suppression need not remove them.

**Hypothesis:** shared acoustic/phonetic sensitivities across recognizers explain part of transfer. **Alternative explanations:** generator coloration, language-model hallucination, deletions/no-speech decisions, timestamp/prompt mismatch, or ordinary intelligibility damage could also contribute. Cross-architecture transfer alone does not distinguish these mechanisms. We need phonetic, decoding and human-content controls.

The TV/HF terms may guide optimization toward stable vocoder outputs instead of merely limiting attack strength. That explains why the paper's ablation could show both worse quality and weaker errors after removing them. It does not imply this must happen for every implementation; local ablations do not reproduce the same pattern consistently.

Every generated waveform can still be written algebraically as `x + (u−x)`. The distinction is the **feasible set and generation procedure**, not whether waveform differences exist. Neither the paper's results nor this algebra establish that feature-vocoder attacks are intrinsically stronger than every possible waveform attack.

## Essential idea versus implementation choices

| Component | Status for the central idea | Defense implication |
|---|---|---|
| Search over SSL features and differentiable synthesis | Essential | Defend against generated speech variation, not only additive small-norm noise |
| Surrogate-only generation before transfer evaluation | Essential to the stated access model | Separate original-transfer and defense-aware experiments |
| Bounded relative feature perturbation | Essential to the stated feasible set | Verify actual delta norms; do not equate radius with perceptual fidelity |
| Correct-transcript lexical loss | Central optimization mechanism | Adaptive tests must attack the actual defended recognizer |
| Clean-normalized TV/HF regularization | Central claimed objective | Include its full setting and separate ablations |
| WavLM-Large and HiFi-GAN | Stated instantiation | A defense generalizing only to these weights may overfit artifacts |
| Feature layer, normalization, vocoder weights | Unspecified independent choices | Test matched alternate generator contracts |
| Adam, LR, initialization, scheduling | Not specified by paper | Local optimizer choices are not recovered author settings |
| Sum versus mean lexical NLL, α, STFT details | Unspecified numerical choices | Relative regularization and optimization can change materially |
| Decoder, transcript normalization, checkpoint variants | Evaluation choices incompletely specified | Freeze and report; do not silently optimize the scorer |

## Limitations and bottlenecks that matter for defense research

1. **Content is not a hard constraint.** TV encourages smooth trajectories, HF limits a spectral fraction, and global feature distance bounds an embedding norm. None guarantees unchanged phonemes or words. MOS can be high for fluent but different speech. Human transcription is needed for the intended threat.
2. **Generator range is not a proven linguistic manifold.** A speech-like waveform can be phonetically ambiguous or wrong. Reconstructing with the same vocoder is not necessarily purification: it may preserve the attack.
3. **The global radius is not locally uniform.** It permits concentration in a short word or subset of channels; feature scale, anisotropy, duration and silence can change its interpretation. It is not invariant to arbitrary feature-coordinate reparameterization.
4. **Teacher-forced loss is only a proxy for decoding.** Increasing ground-truth NLL need not maximize free-running WER. Special-token behavior, no-speech decisions and insertions may dominate. WER can exceed 100%.
5. **Component compatibility is a major architectural dependency.** Changing SSL normalization/layer without matching vocoder weights can damage reconstruction before any adversarial optimization.
6. **Loss scales can be unstable.** Near-zero clean HF energy magnifies the normalized penalty; token sum/mean changes effective λ with transcript length. These are implementation sensitivity questions, not demonstrated flaws in every run.
7. **Defense evaluation is mostly non-adaptive transfer.** The stated attack does not optimize through target defenses. A new defense cannot claim broad robustness from this condition alone.
8. **Missing artifacts limit comparison.** CE-AT, DT/DE-AT, DMW, TI, AudioPure sentence settings and PVP identity are not all recoverable. Reference mismatches are detailed in the repository audit; we must not fabricate those baselines.
9. **Reported compute does not transfer to this machine.** The paper's ~10 s/utterance and ~4 GPU-hours use an L20 46 GB. Local hardware is a busy TITAN RTX 24 GB. Long sentences, vocoder gradients, multiple attack restarts and ensemble passes can dominate cost.
10. **Scope is finite.** Existing human/physical evidence does not establish multilingual, accent, conversational, commercial-API, streaming, or unseen-vocoder robustness.

The resulting research question is: **Can a recognizer retain sufficient independent phonetic evidence to recover the original words under generated feature perturbations, without sacrificing benign recognition or relying on blocked attack gradients?** The proposed alternatives and falsification tests are in [defense designs](04_DEFENSE_DESIGNS.md).
