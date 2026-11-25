# One bounded next experiment: clean-energy envelope projection

Primary-source check: 22 October 2026 (Asia/Taipei). This note recommends one finite experiment after the completed waveform-refinement cycle. It reports no new model measurements, supplies no novelty claim, and does not treat an RMS envelope as auditory masking.

## Why change the constraint

The local [48-utterance refinement results](../10_refinement_results.md) show that extending the weighted objective raised source WER from 43.4608% to 53.4205%, but lowered NISQA from 3.7776 to 3.5937, below the registered 3.75 floor. Keeping both proxy distortions below their initialization values yielded essentially unchanged WER. The earlier source-NLL-only attack reached 58.7525% WER but NISQA 3.4367. These observations motivate testing whether a clean-speech-dependent amplitude limit protects quiet regions more effectively than a uniform amplitude limit. They do not establish that quiet-region noise caused the NISQA failure.

## What the primary sources actually establish

**Gao et al., SAGO.** Section II-B1, Eq. 2 applies a fixed VAD mask to waveform gradients before a signed update; Eq. 3 projects to the perturbation set. Section III-B specifies 50 iterations, L-infinity limits 0.002/0.0035, and cepstral-power detection to trim leading/trailing nonspeech. It does not give a local-RMS envelope formula or a complete frequency-masking model. There is an apparent sign inconsistency: Eq. 1 minimizes negative cross entropy, while Eq. 2 adds that loss's gradient. Our implementation should explicitly maximize reference-transcript NLL and test its direction, rather than copy the notation. [Paper, arXiv v1](https://arxiv.org/html/2411.09220v1).

**Qin et al., frequency masking.** Section 4 uses an STFT with N=2048 and hop 512. For each analysis frame, its Eqs. 1–3 define

\[
p_x(k)=10\log_{10}|s_x(k)/N|^2,
\quad \bar p_\delta(k)=96-\max_k p_x(k)+p_\delta(k).
\]

The clean-speech masking threshold uses spectral maxima, quiet thresholds, 0.5-Bark masker separation and spreading functions. Eq. 5 penalizes positive excess, averaged over frequency bins:

\[
\ell_\theta=\frac1{\lfloor N/2\rfloor+1}
\sum_k\max(\bar p_\delta(k)-\theta_x(k),0).
\]

Their two stages first find an attack and then optimize recognition plus masking loss. This is distinct from limiting sample amplitudes according to local speech energy. Their listening results cannot establish imperceptibility here. [ICML 2019 paper, Sections 4.1–4.2](https://proceedings.mlr.press/v97/qin19a/qin19a.pdf).

**Implementation qualification.** The authors' released threshold code sums individual masking contributions and the quiet threshold in **linear power**, `sum(10**(T/10)) + 10**(ATH/10)`. Its attack code applies a ReLU difference against transformed perturbation power. An implementation must keep threshold and perturbation units consistent; directly subtracting a dB threshold from linear power is invalid. [Author threshold code](https://raw.githubusercontent.com/yaq007/cleverhans/master/examples/adversarial_asr/generate_masking_threshold.py), [author attack code](https://github.com/yaq007/cleverhans/blob/master/examples/adversarial_asr/generate_imperceptible_adv.py).

## Actual preregistered engineering change

The root experiment implements **50-step PGD with a fixed clean-energy envelope**. Its primary condition maximizes source NLL alone; a full-R3-objective ablation uses the same envelope. Archived uniform-bound NLL50, full50 and FV50 conditions provide controls. An additional control uniformly attenuates the archived global-bound NLL output to match the candidate's whole-utterance residual RMS; it must fail rather than amplify if matching would require a factor above one. This control tests whether lower overall noise energy alone explains any quality gain. Preserve the frozen 48 utterances, source checkpoint, transcript normalization, native decoder, zero initialization, 50 source backward calls and final-iterate selection. Do not tune against MOS or search envelope ratios.

The actual rule below is an engineering adaptation, **not an equation attributed to SAGO or Qin, and not a reproduction of either method**. For native 16-kHz clean waveform x, use 400-sample (exactly 25-ms) frames with hop 160 samples, and include a final end-aligned frame so every sample is covered. Let F_m denote each such frame and define

\[
r_m=\sqrt{\frac{\sum_{t\in F_m}x_t^2}{|F_m|}},
\qquad c_m=\min(0.003,10^{-30/20}r_m),
\qquad b_t=\min_{m:t\in F_m}c_m.
\]

Freeze b from clean x once. There is no nonzero noise floor: a completely silent frame forces b=0 throughout that frame. Combine valid-audio and residual bounds into fixed coordinate intervals

\[
l_t=\max(-b_t,-1-x_t),\qquad u_t=\min(b_t,1-x_t),
\]

and set coordinate step size alpha_t=2*b_t/50. For the primary NLL-only condition, update

\[
\delta_{k+1,t}=\operatorname{clip}_{[l_t,u_t]}
\{\delta_{k,t}+\alpha_t\operatorname{sign}(\partial_t\mathrm{NLL}(x+\delta_k,y))\}.
\]

The projection and coordinate-dependent step rule together constitute the fixed intervention; attributing a result solely to projection would require another ablation. Clipping is the exact Euclidean projection onto this fixed box. It guarantees the saved sample amplitudes satisfy the envelope and the 0.003 global L-infinity bound, within declared arithmetic/serialization tolerance. RMS <=0.003 follows from L-infinity.

Unlike a centered per-sample RMS proxy, taking the minimum over all covering frames gives a frame-energy guarantee. For every registered frame F_m,

\[
\sum_{t\in F_m}\delta_t^2
\leq |F_m|c_m^2
\leq 10^{-3}\sum_{t\in F_m}x_t^2.
\]

Thus positive-energy frames have signal-to-perturbation energy ratio at least 1000, corresponding to at least 30-dB SNR, before numerical roundoff. If a clean frame has zero energy, its perturbation is also zero; do not assign a finite SNR to the undefined zero/zero ratio. The guarantee concerns the specified frames, not every arbitrary time interval. It does not certify spectral masking, inaudibility, phoneme preservation or improvement on any quality predictor. Audit the serialized waveform and report the declared tolerance rather than silently relaxing the bound.

An earlier research suggestion used a centered 401-sample RMS at each sample, b_t=min(.003,10^(-30/20)*RMS_centered401(x)_t), with constant step size .00012. That was a preliminary suggestion and is **not the preregistered implementation**. It did not establish the frame-energy inequality above. The actual covering-frame minimum and relative step sizes supersede that suggestion before new outputs are examined.

The hypothesis is that removing perturbation capacity from quiet regions can retain enough source attack strength while recovering NISQA. If the envelope destroys the attack or fails the unchanged quality floor, reject this change. Preserve the original primary advancement rule; freeze any qualifying candidate before validation76. Do not keep testing alternative envelope ratios after seeing this result.

## Resource decision and limits

The envelope uses fixed-size frame-energy calculations and one vector clamp per step, with O(T) storage and linear work for fixed frame/hop settings. The primary source-NLL-only condition needs no extra pretrained model, WavLM backward pass, masking-threshold backpropagation or line search. The full-R3-objective ablation retains the existing encoder/distortion cost; count it separately. This is a defensible next test inside the parent's roughly 2.5-hour remaining scheduled-compute envelope; exact runtime and peak memory still require the actual run receipt. Reuse archived controls without relabeling them as new independent trials. Charge native scoring and any validation run to the remaining budget and stop at the registered deadline.

Full Qin-style masking is technically plausible on 24 GB: the threshold is clean-derived and can be cached on CPU, and its differentiable residual STFT is small relative to the ASR graph. The reason to defer it here is implementation and validation scope, not a proven GPU-memory barrier. A faithful adaptation needs pinned window/scaling conventions, masker and threshold-in-quiet tests, consistent dB/linear units, gradient checks and a fixed optimization schedule. The [author implementation](https://github.com/yaq007/cleverhans/tree/master/examples/adversarial_asr) targets Python 2.7/TensorFlow 1.13/Lingvo; using a new PyTorch/Whisper frontend would be an adaptation, not an exact reproduction.

Even a passing source/quality screen would support only an empirical result under this protocol. Transfer and human-perception claims need separate evidence. SAGO already establishes speech-aware gradient restrictions; Qin already establishes masking-oriented ASR optimization. This experiment should make no first-use or new-masking claim.
