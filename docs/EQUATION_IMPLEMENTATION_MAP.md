# BeyondWaveform: every numbered equation and its implementation

Code snapshot inspected: 2026-10-15 13:31 Asia/Taipei. Source paper: [BeyondWaveform.pdf](/workspace/BeyondWaveform.pdf).

The supplied paper contains **16 numbered/displayed equations**, all mapped below. Each code block is copied from the current source, with only common leading indentation removed. Blocks are excerpts from their surrounding functions, not standalone scripts. Some equations are definitions of the same computation, so they correctly share code. Equation (8) is a set definition implemented implicitly; Equation (14) is solved by a finite optimization procedure rather than an exact mathematical argmin.

## Symbols used in the mapping

| Paper symbol | Code name / meaning |
| --- | --- |
| `x`, `y` | `x` / `audio`: original waveform; `transcript` / `text`: ground-truth transcript |
| `E`, `q` | `encoder` / `WavLMEncoder`; `q` is the clean `(T, D)` feature tensor |
| `delta`, `z` | `delta`: the optimized feature perturbation; `z = q + delta` |
| `V`, `x_adv` | `vocoder` / `FeatureVocoder`; `audio` inside optimization and `final` afterward |
| `f_s`, `f_t` | `surrogate` for gradients; separately loaded `target` for evaluation |
| `rho`, `epsilon` | `config.rho`, `config.eps` |
| `alpha`, `lambda_perc` | `config.alpha_hf`, `config.lambda_perc` |
| `sg(...)` | `.detach()` / computation without gradients |
| `theta` | A candidate attack configuration selected on validation data |

## Quick index

| Equation | Operation | Implementation |
| --- | --- | --- |
| (1) | Decode speech with surrogate/target | [WhisperASR.transcribe](/workspace/beyondwaveform/models/whisper.py:55) |
| (2) | Feature perturbation → waveform | [feature_attack](/workspace/beyondwaveform/attacks.py:39) |
| (3) | Measure target WER/CER | [evaluate + error_counts + aggregate_errors](/workspace/beyondwaveform/pipeline.py:108) |
| (4) | Extract clean speech features | [WavLMEncoder.forward](/workspace/beyondwaveform/models/feature.py:21) |
| (5) | Add the feature perturbation | [z = q + delta](/workspace/beyondwaveform/attacks.py:60) |
| (6) | Enforce relative Frobenius budget | [project_relative_ball_ + relative_radius](/workspace/beyondwaveform/attacks.py:21) |
| (7) | Synthesize perturbed features | [FeatureVocoder.forward](/workspace/beyondwaveform/models/feature.py:35) |
| (8) | Define feasible output waveforms | [Projection plus vocoder parameterization; implicit set](/workspace/beyondwaveform/attacks.py:77) |
| (9) | Maximize lexical transcript NLL | [WhisperASR.nll + lexical_nll](/workspace/beyondwaveform/models/whisper.py:40) |
| (10) | Negate NLL for minimization | [Negative nll in the scalar loss](/workspace/beyondwaveform/attacks.py:66) |
| (11) | Clean-referenced perceptual loss | [CleanReference + alpha_hf weighting](/workspace/beyondwaveform/losses.py:49) |
| (12) | Temporal variation | [temporal_variation](/workspace/beyondwaveform/losses.py:8) |
| (13) | High-frequency energy fraction | [high_frequency_fraction](/workspace/beyondwaveform/losses.py:15) |
| (14) | Optimize the complete objective | [feature_attack; finite projected optimization](/workspace/beyondwaveform/attacks.py:39) |
| (15) | Generate the final attacked waveform | [Final vocoder call and AttackResult](/workspace/beyondwaveform/attacks.py:84) |
| (16) | Choose a validation configuration | [select_configuration](/workspace/beyondwaveform/selection.py:8) |

## Equation (1): ASR predictions

Paper: [Equation (1)](/workspace/paper/BeyondWaveform.md:52)

$$\hat{y}_s = f_s(x), \qquad \hat{y}_t = f_t(x). \tag{1}$$

The decoded transcript comes from `transcribe`. The same Whisper adapter represents the surrogate or a Whisper-family target depending on which checkpoint is loaded. A separate CTC adapter handles Wav2Vec2/HuBERT targets.

Whisper decoding:

Source: [beyondwaveform/models/whisper.py:54–57](/workspace/beyondwaveform/models/whisper.py:54)

```python
@torch.no_grad()
def transcribe(self, audio):
    return whisper.transcribe(self.model, audio, language=self.language, task="transcribe",
                              fp16=False, condition_on_previous_text=False, **self.decoding)["text"].strip()
```

CTC target decoding:

Source: [beyondwaveform/models/targets.py:25–30](/workspace/beyondwaveform/models/targets.py:25)

```python
@torch.no_grad()
def transcribe(self, audio):
    batch = self.processor(audio.detach().cpu().numpy(), sampling_rate=16000, return_tensors="pt")
    batch = {key: value.to(self.device) for key, value in batch.items()}
    ids = self.model(**batch).logits.argmax(-1)
    return self.processor.batch_decode(ids)[0]
```

**Implementation note:** Equation (1) depicts decoded outputs. Attack gradients use the likelihood function in Equation (9), not the discrete decoded text. The clean control passes original `x` to these decoders; adversarial evaluation passes the generated waveform. Decoder settings are independent implementation choices.

## Equation (2): Generate audio from perturbed features

Paper: [Equation (2)](/workspace/paper/BeyondWaveform.md:60)

$$x_{\mathrm{adv}}(\delta) = V(E(x) + \delta), \tag{2}$$

`feature_attack` computes the clean features once, then adds `delta` and sends the perturbed features through the frozen vocoder. These are two excerpts from the same function.

Load the clean features and zero-perturbation reconstruction with model weights frozen:

Source: [beyondwaveform/attacks.py:39–46](/workspace/beyondwaveform/attacks.py:39)

```python
def feature_attack(x, transcript, encoder, vocoder, surrogate,
                   config=AttackConfig(), spectral=SpectralConfig()):
    for module in (encoder, vocoder, surrogate):
        freeze(module)
    start = time.perf_counter()
    with torch.no_grad():
        q = encoder(x).detach()
        reconstruction = vocoder(q).detach()
```

Inside the optimization loop:

Source: [beyondwaveform/attacks.py:58–61](/workspace/beyondwaveform/attacks.py:58)

```python
for step in range(config.steps):
    optimizer.zero_grad(set_to_none=True)
    z = q + delta
    audio = vocoder(z)  # frozen weights, live input gradients
```

**Implementation note:** This is a feature-space change followed by synthesis. There is no additional waveform residual and no kNN feature replacement in this generation path. The returned final waveform is covered by Equation (15).

## Equation (3): Evaluate transfer error

Paper: [Equation (3)](/workspace/paper/BeyondWaveform.md:66)

$$\mathrm{Err}\big(f_t(x_{\mathrm{adv}}), y\big) \tag{3}$$

The evaluator loads the already-saved waveform, transcribes it with the selected target, computes lexical edit counts, then aggregates them into corpus WER/CER.

Target prediction and per-utterance error counts:

Source: [beyondwaveform/pipeline.py:125–129](/workspace/beyondwaveform/pipeline.py:125)

```python
audio = load_audio(asset["path"]).to(device)
transcript = target.transcribe(defend(audio, defense))
row = {"id": record["id"], "variant": variant, "audio_sha256": asset["sha256"],
       "reference": record["text"], "hypothesis": transcript,
       **error_counts(record["text"], transcript, record["language"])}
```

English uses words; Mandarin uses characters:

Source: [beyondwaveform/evaluation.py:24–31](/workspace/beyondwaveform/evaluation.py:24)

```python
def error_counts(reference, hypothesis, language="en"):
    reference, hypothesis = normalize_text(reference, language), normalize_text(hypothesis, language)
    if not reference:
        raise ValueError("Empty reference after normalization is not scoreable")
    result = jiwer.process_characters(reference, hypothesis) if language == "zh" else jiwer.process_words(reference, hypothesis)
    return {"substitutions": result.substitutions, "deletions": result.deletions,
            "insertions": result.insertions, "reference_units": result.hits + result.substitutions + result.deletions,
            "reference_normalized": reference, "hypothesis_normalized": hypothesis}
```

Corpus-level percentage:

Source: [beyondwaveform/evaluation.py:34–39](/workspace/beyondwaveform/evaluation.py:34)

```python
def aggregate_errors(records):
    counts = {k: sum(row[k] for row in records) for k in ["substitutions", "deletions", "insertions", "reference_units"]}
    if counts["reference_units"] <= 0:
        raise ValueError("No reference units")
    counts["error_percent"] = 100 * (counts["substitutions"] + counts["deletions"] + counts["insertions"]) / counts["reference_units"]
    return counts
```

**Implementation note:** `aggregate_errors` sums substitutions, deletions and insertions before dividing by total reference words/characters; it does not average utterance WER/CER. Text normalization is defined in [beyondwaveform/evaluation.py:16–21](/workspace/beyondwaveform/evaluation.py:16). `defend` is the identity for raw evaluation. Exact author defense recipes are not supplied by this general evaluator; the LPF adapter is explicitly an independent example. Targets are loaded by `evaluate` after generation, not used by `feature_attack`.

## Equation (4): Extract the clean feature trajectory

Paper: [Equation (4)](/workspace/paper/BeyondWaveform.md:80)

$$q = E(x), \qquad q = (q_1, \dots, q_T). \tag{4}$$

`WavLMEncoder.forward` is the implementation of `E(x)`. It returns one frame-by-feature tensor, and `feature_attack` detaches that tensor so clean features are fixed during perturbation optimization.

Source: [beyondwaveform/models/feature.py:21–26](/workspace/beyondwaveform/models/feature.py:21)

```python
def forward(self, x):
    if x.ndim != 1:
        raise ValueError("One unpadded utterance required")
    if self.normalize:
        x = torch.nn.functional.layer_norm(x, x.shape)
    return self.model.extract_features(x[None], output_layer=self.layer, ret_layer_results=False)[0][0]
```

Source: [beyondwaveform/attacks.py:44–48](/workspace/beyondwaveform/attacks.py:44)

```python
with torch.no_grad():
    q = encoder(x).detach()
    reconstruction = vocoder(q).detach()
if q.ndim != 2 or q.shape[0] < 2 or not torch.isfinite(q).all():
    raise ValueError("Encoder must produce finite (T,D), T>=2 features")
```

**Implementation note:** The selected implementation uses WavLM-Large layer 6, 1024-dimensional features, and `normalize: false`. The loader requires a compatible layer-6 vocoder; these artifact/layer/normalization choices are not fully identified in the paper. See [feature-model loader](/workspace/beyondwaveform/models/feature.py:42).

## Equation (5): Construct perturbed features

Paper: [Equation (5)](/workspace/paper/BeyondWaveform.md:86)

$$z(\delta) = q + \delta. \tag{5}$$

The perturbation has exactly the shape of `q`. It is the trainable input variable; adding it to `q` produces `z`.

Initialize the variable and give only that variable to the optimizer:

Source: [beyondwaveform/attacks.py:50–56](/workspace/beyondwaveform/attacks.py:50)

```python
delta = torch.zeros_like(q)
if config.initialization == "normal" and config.rho > 0:
    delta.normal_()
    delta *= config.init_fraction * config.rho * (q.norm() + config.eps) / delta.norm()
delta.requires_grad_()
optimizer_class = torch.optim.Adam if config.optimizer == "adam" else torch.optim.SGD
optimizer = optimizer_class([delta], lr=config.learning_rate)
```

Apply Equation (5):

Source: [beyondwaveform/attacks.py:60–61](/workspace/beyondwaveform/attacks.py:60)

```python
z = q + delta
audio = vocoder(z)  # frozen weights, live input gradients
```

**Implementation note:** The reported English configuration uses zero initialization. The optional normal initialization branch shown above is inactive for that configuration. Model weights are not optimizer parameters.

## Equation (6): Enforce the relative feature budget

Paper: [Equation (6)](/workspace/paper/BeyondWaveform.md:90)

$$\mathcal{U}_\rho(x) = \left\{ \delta : \frac{\|\delta\|_F}{\|q\|_F + \varepsilon} \le \rho \right\}, \tag{6}$$

For a `(T, D)` tensor, `torch.linalg.vector_norm` over all entries is its Frobenius norm. Projection rescales the entire perturbation if its norm exceeds `rho * (||q|| + eps)`.

Source: [beyondwaveform/attacks.py:16–27](/workspace/beyondwaveform/attacks.py:16)

```python
def relative_radius(delta, q, eps=1e-8):
    return torch.linalg.vector_norm(delta) / (torch.linalg.vector_norm(q.detach()) + eps)


@torch.no_grad()
def project_relative_ball_(delta, q, rho, eps=1e-8):
    """Euclidean projection onto Eq.6, not elementwise clipping."""
    radius = rho * (torch.linalg.vector_norm(q.detach()) + eps)
    norm = torch.linalg.vector_norm(delta)
    if norm > radius:
        delta.mul_(radius / norm)
    return delta
```

Apply the projection after every optimizer update and check the ratio:

Source: [beyondwaveform/attacks.py:76–80](/workspace/beyondwaveform/attacks.py:76)

```python
optimizer.step()
project_relative_ball_(delta, q, config.rho, config.eps)
radius = relative_radius(delta, q, config.eps).item()
if radius > config.rho + 1e-6:
    raise RuntimeError("Feature radius violated")
```

**Implementation note:** The radius is global across time and feature dimensions, not elementwise clipping or a per-frame budget. The equation specifies a feasible set; this implementation chooses Euclidean radial projection to enforce it. `eps` is outside the clean-feature norm, matching the written denominator.

## Equation (7): Vocode the perturbed trajectory

Paper: [Equation (7)](/workspace/paper/BeyondWaveform.md:94)

$$x_{\mathrm{adv}}(\delta) = V(z(\delta)) = V(E(x) + \delta). \tag{7}$$

Equation (7) restates Equation (2) after naming the perturbed features `z`. The vocoder adapter calls the underlying HiFi-GAN generator directly.

Source: [beyondwaveform/attacks.py:60–61](/workspace/beyondwaveform/attacks.py:60)

```python
z = q + delta
audio = vocoder(z)  # frozen weights, live input gradients
```

Source: [beyondwaveform/models/feature.py:35–39](/workspace/beyondwaveform/models/feature.py:35)

```python
def forward(self, z):
    if z.ndim != 2 or z.shape[1] != self.input_dim:
        raise ValueError(f"Vocoder expects (T,{self.input_dim}), got {tuple(z.shape)}")
    # Do NOT call upstream matcher.vocode(): it uses inference_mode.
    return self.generator(z[None])[0, 0]
```

**Implementation note:** Vocoder weights are frozen, but input gradients remain live during optimization. Calling an inference-only upstream helper would sever those gradients, so the adapter invokes `self.generator` directly. The optimizer can therefore backpropagate from the ASR objective into `delta`.

## Equation (8): Define the feasible waveform set

Paper: [Equation (8)](/workspace/paper/BeyondWaveform.md:98)

$$\mathcal{M}_{\mathrm{FV}}(x;\rho) = \left\{ V(E(x) + \delta) : \delta \in \mathcal{U}_\rho(x) \right\}. \tag{8}$$

This is a mathematical set definition, not a separate function that enumerates waveforms. It is realized by combining the constrained perturbation from Equation (6) with `V(q + delta)` from Equations (2)/(7).

Every completed update is projected, and the final waveform is synthesized from the feasible final perturbation:

Source: [beyondwaveform/attacks.py:76–85](/workspace/beyondwaveform/attacks.py:76)

```python
    optimizer.step()
    project_relative_ball_(delta, q, config.rho, config.eps)
    radius = relative_radius(delta, q, config.eps).item()
    if radius > config.rho + 1e-6:
        raise RuntimeError("Feature radius violated")
    trace.append({"step": step + 1, "loss_before_update": loss.item(), "nll_before_update": nll.item(),
                  "tv_ratio": tv.item(), "hf_ratio": hf.item(), "gradient_norm": gradient.norm().item(),
                  "radius_after_update": radius})
with torch.no_grad():
    final = vocoder(q + delta).detach()
```

**Implementation note:** There is no distinct `M_FV` object in the program. The construction represents membership in this set; it does not compute the whole set or guarantee a perceptual-quality threshold. With `rho=0`, the output is `V(E(x))`, which generally differs from original `x`.

## Equation (9): Maximize the correct transcript’s negative log-likelihood

Paper: [Equation (9)](/workspace/paper/BeyondWaveform.md:104)

$$\max_{\delta \in \mathcal{U}_\rho(x)} \ \mathrm{NLL}_{\mathrm{text}}\big(y \mid x_{\mathrm{adv}}(\delta); f_s\big). \tag{9}$$

`surrogate.nll(audio, transcript)` computes the negative log-probability of the ground-truth lexical tokens under teacher forcing. Making that value larger makes the correct transcript less probable.

Build aligned decoder inputs/labels and mask out special tokens:

Source: [beyondwaveform/models/whisper.py:12–21](/workspace/beyondwaveform/models/whisper.py:12)

```python
def teacher_forcing_tokens(tokenizer, text, device):
    prefix = list(tokenizer.sot_sequence_including_notimestamps)
    lexical = tokenizer.encode(text.strip())
    if not lexical or any(token >= tokenizer.eot for token in lexical):
        raise ValueError("Transcript must contain ordinary lexical tokens only")
    sequence = torch.tensor(prefix + lexical + [tokenizer.eot], device=device, dtype=torch.long)
    inputs, labels = sequence[:-1], sequence[1:]
    # Prefix is conditioning only. EOT and all other specials are unscored.
    mask = labels < tokenizer.eot
    return inputs[None], labels[None], mask[None]
```

Evaluate transcript likelihood through the differentiable audio frontend:

Source: [beyondwaveform/models/whisper.py:40–44](/workspace/beyondwaveform/models/whisper.py:40)

```python
def nll(self, audio, text):
    inputs, labels, mask = teacher_forcing_tokens(self.tokenizer, text, audio.device)
    if inputs.shape[-1] > self.model.dims.n_text_ctx:
        raise ValueError("Transcript exceeds Whisper context")
    return lexical_nll(self.model(self.mel(audio), inputs), labels, mask, self.reduction)
```

Compute cross-entropy only at lexical positions:

Source: [beyondwaveform/losses.py:34–46](/workspace/beyondwaveform/losses.py:34)

```python
def lexical_nll(logits, labels, lexical_mask, reduction="mean"):
    """Already next-token-aligned logits/labels; special positions score zero."""
    if logits.shape[:-1] != labels.shape or labels.shape != lexical_mask.shape:
        raise ValueError("Token shapes do not align")
    mask = lexical_mask.bool()
    if not mask.any():
        raise ValueError("Transcript has no lexical tokens")
    loss = F.cross_entropy(logits[mask].float(), labels[mask], reduction="none")
    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        return loss.mean()
    raise ValueError("NLL reduction must be sum or mean")
```

**Implementation note:** The prefix conditions the decoder, while prompt/language/task/no-timestamps tokens and EOT do not contribute to this attack loss. The selected configuration uses a mean over lexical tokens; the implementation supports sum as well. The paper does not specify sum versus mean, which affects the balance with perceptual regularization. Maximization is implemented through the negative sign in Equation (10).

## Equation (10): Use negative NLL as the attack loss

Paper: [Equation (10)](/workspace/paper/BeyondWaveform.md:108)

$$\mathcal{L}_{\mathrm{attack}}(\delta) = -\mathrm{NLL}_{\mathrm{text}}\big(y \mid x_{\mathrm{adv}}(\delta); f_s\big). \tag{10}$$

The minimization objective includes `-nll`, exactly the sign required to increase NLL when minimizing. The attack term is written inline rather than stored in a separate `attack_loss` variable.

Source: [beyondwaveform/attacks.py:62–66](/workspace/beyondwaveform/attacks.py:62)

```python
nll = surrogate.nll(audio, transcript) if config.attack else z.new_zeros(())
tv, hf = reference.terms(z, audio,
                         temporal=config.temporal and config.lambda_perc > 0,
                         high_frequency=config.high_frequency and config.lambda_perc > 0 and config.alpha_hf > 0)
loss = -nll + config.lambda_perc * (tv + config.alpha_hf * hf)
```

**Implementation note:** With the attack enabled, minimizing the first term pushes the surrogate away from the correct transcript. In the no-attack ablation, `config.attack=False` replaces NLL with a zero placeholder; that zero is not a measured likelihood.

## Equation (11): Build the clean-referenced perceptual loss

Paper: [Equation (11)](/workspace/paper/BeyondWaveform.md:114)

$$\mathcal{L}_{\mathrm{perc}}(\delta) = \frac{\mathrm{TV}(z(\delta))}{\mathrm{sg}(\mathrm{TV}(q)) + \varepsilon} + \alpha \, \frac{\mathrm{HF}(x_{\mathrm{adv}}(\delta))}{\mathrm{sg}(\mathrm{HF}(x)) + \varepsilon}, \tag{11}$$

The denominator statistics are computed once from original clean `q` and original clean waveform `x`, with gradients stopped. The numerator statistics use current perturbed features and generated audio.

Cache clean denominators and calculate the two normalized ratios:

Source: [beyondwaveform/losses.py:49–61](/workspace/beyondwaveform/losses.py:49)

```python
class CleanReference:
    """Detached ORIGINAL-clean denominators, cached once (Eq.11)."""
    def __init__(self, q, x, spectral, eps):
        self.spectral = spectral
        self.eps = eps
        self.tv = temporal_variation(q.detach()).detach()
        self.hf = high_frequency_fraction(x.detach(), spectral).detach()

    def terms(self, z, audio, temporal=True, high_frequency=True):
        zero = z.new_zeros(())
        tv = temporal_variation(z) / (self.tv + self.eps) if temporal else zero
        hf = high_frequency_fraction(audio, self.spectral) / (self.hf + self.eps) if high_frequency else zero
        return tv, hf
```

Combine the ratios with the paper’s two weights:

Source: [beyondwaveform/attacks.py:63–66](/workspace/beyondwaveform/attacks.py:63)

```python
tv, hf = reference.terms(z, audio,
                         temporal=config.temporal and config.lambda_perc > 0,
                         high_frequency=config.high_frequency and config.lambda_perc > 0 and config.alpha_hf > 0)
loss = -nll + config.lambda_perc * (tv + config.alpha_hf * hf)
```

**Implementation note:** `tv + config.alpha_hf * hf` is the perceptual loss; `config.lambda_perc` then weights that complete perceptual loss. The paper’s `alpha` is `alpha_hf`, not a waveform-baseline step size. The HF denominator uses original `x`, not the vocoder reconstruction. `alpha_hf=1` and `eps=1e-8` are declared independent choices. Flags allow the stated loss removals.

## Equation (12): Compute temporal variation

Paper: [Equation (12)](/workspace/paper/BeyondWaveform.md:120)

$$\mathrm{TV}(z) = \frac{1}{T-1} \sum_{t=2}^{T} \|z_t - z_{t-1}\|_2^2. \tag{12}$$

The code subtracts adjacent feature frames, squares the differences, sums over feature coordinates, and averages over the `T−1` adjacent-frame pairs.

Source: [beyondwaveform/losses.py:8–12](/workspace/beyondwaveform/losses.py:8)

```python
def temporal_variation(z: torch.Tensor) -> torch.Tensor:
    """Eq.12: sum feature dimensions, MEAN adjacent time differences."""
    if z.ndim != 2 or z.shape[0] < 2:
        raise ValueError("TV requires an unpadded (T,D) tensor with T >= 2")
    return z.diff(dim=0).square().sum(dim=-1).mean()
```

**Implementation note:** `sum(dim=-1)` implements the squared L2 norm within a frame pair. The final `.mean()` implements division by `T−1`. There is no extra division by the feature dimension. Inputs with fewer than two frames are rejected.

## Equation (13): Compute the high-frequency energy fraction

Paper: [Equation (13)](/workspace/paper/BeyondWaveform.md:124)

$$\mathrm{HF}(x) = \frac{\sum_{f > f_c} |S_f(x)|^2}{\sum_f |S_f(x)|^2}, \tag{13}$$

The function computes STFT power, sums the power strictly above the cutoff frequency, and divides by total STFT power.

Source: [beyondwaveform/losses.py:15–31](/workspace/beyondwaveform/losses.py:15)

```python
def high_frequency_fraction(x: torch.Tensor, config: SpectralConfig) -> torch.Tensor:
    """Eq.13, selected interpretation: sum power over both frequency and time.

    Strictly f > fc. No extra epsilon is added to Eq.13. Zero-energy signals
    are rejected; epsilon belongs to the clean normalizers in Eq.11.
    """
    if x.ndim != 1 or x.numel() <= config.n_fft // 2:
        raise ValueError("HF expects a nonempty waveform long enough for reflection padding")
    window = torch.hann_window(config.win_length, device=x.device, dtype=x.dtype)
    power = torch.stft(x, n_fft=config.n_fft, hop_length=config.hop_length,
                       win_length=config.win_length, window=window, center=config.center,
                       pad_mode="reflect", return_complex=True).abs().square()
    energy = power.sum()
    if not torch.isfinite(energy) or energy.detach() <= 0:
        raise ValueError("HF undefined for silence or nonfinite energy")
    frequencies = torch.fft.rfftfreq(config.n_fft, d=1 / config.sample_rate).to(x.device)
    return power[frequencies > config.cutoff_hz].sum() / energy
```

**Implementation note:** The paper writes a frequency-only sum although STFT coefficients also have a time index. This implementation pools power over both time and frequency, then takes the ratio; it does not average frame-wise ratios. Cutoff `f_c=6000 Hz` comes from the paper. The selected 16 kHz rate, FFT/window size 1024, hop 256, Hann window, centered reflection padding and one-sided spectrum convention are explicit implementation choices. No epsilon is added inside this energy fraction; silent/nonfinite total energy is rejected. Epsilon is added to the clean reference denominators in Equation (11). Configuration fields: [SpectralConfig](/workspace/beyondwaveform/config.py:6).

## Equation (14): Optimize the full constrained objective

Paper: [Equation (14)](/workspace/paper/BeyondWaveform.md:130)

$$\delta^\star = \arg\min_{\delta \in \mathcal{U}_\rho(x)} \Big[ \mathcal{L}_{\mathrm{attack}}(\delta) + \lambda_{\mathrm{perc}} \mathcal{L}_{\mathrm{perc}}(\delta) \Big]. \tag{14}$$

The executable scalar is `-nll + lambda_perc * (tv + alpha_hf * hf)`. Only `delta` receives an optimizer update. The encoder, vocoder and surrogate weights are frozen.

Freeze model parameters:

Source: [beyondwaveform/attacks.py:39–42](/workspace/beyondwaveform/attacks.py:39)

```python
def feature_attack(x, transcript, encoder, vocoder, surrogate,
                   config=AttackConfig(), spectral=SpectralConfig()):
    for module in (encoder, vocoder, surrogate):
        freeze(module)
```

Construct the optimizer and run the constrained updates:

Source: [beyondwaveform/attacks.py:54–77](/workspace/beyondwaveform/attacks.py:54)

```python
delta.requires_grad_()
optimizer_class = torch.optim.Adam if config.optimizer == "adam" else torch.optim.SGD
optimizer = optimizer_class([delta], lr=config.learning_rate)
trace = []
for step in range(config.steps):
    optimizer.zero_grad(set_to_none=True)
    z = q + delta
    audio = vocoder(z)  # frozen weights, live input gradients
    nll = surrogate.nll(audio, transcript) if config.attack else z.new_zeros(())
    tv, hf = reference.terms(z, audio,
                             temporal=config.temporal and config.lambda_perc > 0,
                             high_frequency=config.high_frequency and config.lambda_perc > 0 and config.alpha_hf > 0)
    loss = -nll + config.lambda_perc * (tv + config.alpha_hf * hf)
    if not torch.isfinite(loss):
        raise FloatingPointError(f"Nonfinite objective at step {step}")
    if loss.requires_grad:
        gradient, = torch.autograd.grad(loss, delta)
    else:
        gradient = torch.zeros_like(delta)
    if not torch.isfinite(gradient).all():
        raise FloatingPointError(f"Nonfinite gradient at step {step}")
    delta.grad = gradient
    optimizer.step()
    project_relative_ball_(delta, q, config.rho, config.eps)
```

**Implementation note:** The paper writes an argmin. This program approximates it with the configured finite number of projected steps; it does not prove a global minimum. The reported English run uses 50 Adam steps, learning rate 0.05, zero initialization, rho=0.5 and lambda_perc=0.1. The paper states 50 steps, rho=0.1 and lambda_perc=1; optimizer/LR and some loss constants are unspecified. The code supports different defaults, so the saved run configuration—not just `AttackConfig()` defaults—determines a run. See [frozen English configuration](/workspace/configs/english_selected.yaml:1).

## Equation (15): Produce the final attacked waveform

Paper: [Equation (15)](/workspace/paper/BeyondWaveform.md:134)

$$x_{\mathrm{adv}} = x_{\mathrm{adv}}(\delta^\star) = V(E(x) + \delta^\star). \tag{15}$$

After the optimization loop, the code synthesizes the waveform from the final perturbation and returns it along with the reconstruction control, perturbation, trace and timing.

Source: [beyondwaveform/attacks.py:84–88](/workspace/beyondwaveform/attacks.py:84)

```python
with torch.no_grad():
    final = vocoder(q + delta).detach()
if x.is_cuda:
    torch.cuda.synchronize(x.device)
return AttackResult(final, reconstruction, delta.detach(), trace, time.perf_counter() - start)
```

The generation pipeline exports the resulting waveform and controls:

Source: [beyondwaveform/pipeline.py:92–95](/workspace/beyondwaveform/pipeline.py:92)

```python
assets = {"clean": save_audio(output / "audio" / "clean" / f"{uid}.wav", x),
          "adversarial": save_audio(output / "audio" / "adversarial" / f"{uid}.wav", adversarial)}
if reconstruction is not None:
    assets["reconstruction"] = save_audio(output / "audio" / "reconstruction" / f"{uid}.wav", reconstruction)
```

**Implementation note:** `final` is the implementation’s finite-optimization estimate of `V(E(x) + delta_star)`. The final iterate is used; there is no search over earlier iterates for the best output. Disabling gradients here is intentional because all perturbation updates have finished. No model training happens in this final synthesis step.

## Equation (16): Select attack parameters on validation data

Paper: [Equation (16)](/workspace/paper/BeyondWaveform.md:332)

$$\theta_a^\star = \arg\max_{\theta \in \Theta_a} \mathrm{Err}\big(f_s(x_{\mathrm{adv}}^{a,\theta}), y\big) \tag{16}$$

The selector first restricts candidates to comparable raw-surrogate validation evaluations, then applies all three quality gates and maximizes validation WER/CER among eligible candidates.

Source: [beyondwaveform/selection.py:8–45](/workspace/beyondwaveform/selection.py:8)

```python
def select_configuration(summaries, output, threshold=2.5, strict=True):
    candidates = []
    manifest = None
    comparison_protocol = None
    for path in summaries:
        result = json.loads(Path(path).read_text())
        if result["evaluation_scope"] != "surrogate" or result["roles"] != ["validation"]:
            raise ValueError("Selection requires exclusively validation data and the raw generation surrogate")
        if manifest is None:
            manifest = result["manifest_sha256"]
        elif manifest != result["manifest_sha256"]:
            raise ValueError("Candidates must use the identical validation manifest")
        protocol = {"ids": result["evaluation_ids"], "target": result["target_artifacts"],
                    "quality": result["quality_provenance"], "metric": result["metric"],
                    "normalization": result["normalization"]}
        if comparison_protocol is None:
            comparison_protocol = protocol
        elif protocol != comparison_protocol:
            raise ValueError("Candidate subsets, surrogate, decoding, and scoring protocols must match")
        attack = result["summary"]["adversarial"]
        metrics = attack.get("quality", {})
        if set(metrics) != {"dnsmos", "nisqa", "utmos"} or not all(math.isfinite(v) for v in metrics.values()):
            raise ValueError("Selection needs all three finite quality metrics; no silent gate bypass")
        error = attack["error_percent"]
        if not math.isfinite(error):
            raise ValueError("Nonfinite error rate")
        eligible = all(value > threshold if strict else value >= threshold for value in metrics.values())
        candidates.append({"summary": str(Path(path).resolve()), "generation_run": result["generation_run"],
                           "generation_fingerprint": result["generation_fingerprint"], "error_percent": error,
                           "quality": metrics, "eligible": eligible})
    eligible = [r for r in candidates if r["eligible"]]
    winner = max(eligible, key=lambda r: r["error_percent"]) if eligible else None
    selection = {"threshold": threshold, "strict": strict, "aggregation": "corpus error; mean utterance MOS",
                 "manifest_sha256": manifest, "candidates": candidates, "selected": winner}
    atomic_json(output, selection)
    if winner is None:
        raise ValueError("No configuration passed all three quality gates; audit saved")
    return selection
```

The evaluator supplies corpus error and mean quality scores to the selector:

Source: [beyondwaveform/pipeline.py:133–139](/workspace/beyondwaveform/pipeline.py:133)

```python
summary = {}
for variant in variants:
    pool = [r for r in rows if r["variant"] == variant]
    summary[variant] = {"count": len(pool), **aggregate_errors(pool)}
    if scorer:
        summary[variant]["quality"] = {metric: sum(r["quality"][metric] for r in pool) / len(pool)
                                         for metric in scorer.metrics}
```

**Implementation note:** For the reported runs, `strict=True`: mean DNSMOS, NISQA and UTMOS must each be strictly greater than 2.5, following Table 2. Appendix A’s prose uses “below 2.5,” so equality is not perfectly consistent across the paper; the strict interpretation is documented. The gate applies to mean clip scores, not a requirement that every clip passes. Candidate IDs, manifest, surrogate, decoder, scoring provenance and language metric must agree; target/test-only evaluations are rejected. `max` returns the first candidate in input order on exact ties. No eligible candidate produces an explicit failure, not a weaker fallback. This selector chooses a configuration; separate run-freezing scripts persist it for later test/transfer runs.

## Model training versus these equations

Equations (1)–(16) describe ASR inference, perturbation optimization, evaluation and attack-configuration selection. The Mandarin model fine-tuning recipe appears in prose, not as another numbered equation. Its separate implementation is [training.py](/workspace/beyondwaveform/training.py:1). In particular, [WhisperASR.supervised_loss](/workspace/beyondwaveform/models/whisper.py:46) includes EOT in its fine-tuning labels, while the attack `nll` in Equations (9)/(10) scores lexical tokens only. Fine-tuning updates model parameters; `feature_attack` updates only `delta`.

## Source snapshot and verification

All 16 displayed equations were extracted from the supplied paper transcription. Every quoted snippet and line link was checked against the current local files when this document was generated. This mapping describes implementation correspondence and declared choices; it is not a new numerical equivalence claim. No experiments, model training or model-scoring runs were launched for this mapping.

The encoder and vocoder are adapters around loaded upstream pretrained networks, rather than architecture definitions supplied by Equations (1)–(16). Their concrete loading and artifact provenance are in [load_feature_models](/workspace/beyondwaveform/models/feature.py:42). Additional independent choices are documented in [the reproduction guide](/workspace/docs/REPRODUCTION.md:35) and [the equation specification](/workspace/BeyondWaveform_Reproduction_Spec.md:153).

<details>
<summary>SHA-256 of the paper transcription and quoted source files</summary>

| File | SHA-256 |
| --- | --- |
| `beyondwaveform/attacks.py` | `ced4edf7d4ecaa71c87f480e88b74a7e15dc19ed9858fad1e6b1d37979c195d9` |
| `beyondwaveform/evaluation.py` | `47ffe359ae65d8c1539aac6d5cf2d09999ba7b55413432101134011808ea5655` |
| `beyondwaveform/losses.py` | `065535e8797f4fa61f05a7fbd7b4e570f5fbd2cff25aed952bdeb3401481b534` |
| `beyondwaveform/models/feature.py` | `5c68f45cf897f232effbae19472403eca2f9fd20aea785f80a4659842b628a61` |
| `beyondwaveform/models/targets.py` | `de8333c2d67c39e38d456c6ed7fcaf2c11be789580cab371e9c664a019c7337f` |
| `beyondwaveform/models/whisper.py` | `54576d77f99c8fad530cbf3f1096a46c259e21c4e549588614e2654b301cabd8` |
| `beyondwaveform/pipeline.py` | `08e96f4f59d5acc41e82ece347709e5248088110750266d1c3bd7d02fc8e342a` |
| `beyondwaveform/selection.py` | `c12f4a5bde8a6116b0f7b5088760e273630dd4f1da9f8fed7e2fbbce6f57945b` |
| `paper/BeyondWaveform.md` | `ca16b30b9a09d0d504065cbb4bfcf65d36f6449bc9c0485c29f8d2ff1d6de0cd` |

</details>
