# Constrained waveform refinement: prior work and correctness notes

Research check: 2026-10-22. Primary sources only. This note assesses a proposed experiment; it does not report a completed run or establish novelty. No model inference, GPU work, held-out evaluation, or external communication was performed for this note.

The proposed experiment starts from the existing 50-step R3 waveform attack, fixes each utterance's resulting STFT and clean-content distortion as caps, and spends 50 further source-NLL backward passes on constrained refinement. A 100-step weighted-loss control and a 100-step waveform attack provide the relevant optimization-budget comparisons. A promising result would demonstrate an empirical tradeoff for this setup; the ingredients already have substantial precedent.

## Closest methodological precedents

| Primary source | Verified overlap and difference |
|---|---|
| [Qin et al., ICML 2019, §4.2, Eqs. 4–8](https://proceedings.mlr.press/v97/qin19a/qin19a.pdf) | Explicit two-stage ASR attack: first optimize recognition under an amplitude bound, then initialize a second optimization from that attack and balance recognition against a psychoacoustic masking loss. Their second stage uses an adaptive weighted penalty and an unbounded max-norm, rather than our proposed fixed initial-distortion caps. Two-stage attack refinement and recognition–perception tradeoffs are established prior work. They separately conduct listening studies; their evidence cannot establish perceptual quality for this implementation. |
| [Szurley & Kolter, 2019, §3.4, Eqs. 6–10](https://arxiv.org/abs/1906.06355) | Combine ASR loss and a fixed psychoacoustically weighted quadratic waveform distortion, then use normalized-gradient PGD under an L2 bound with valid-audio clipping. Their masking-derived weighting differs from log-magnitude STFT and WavLM-content distances. Waveform PGD with a perceptual penalty is an existing baseline, not a new mechanism. |
| [Schönherr et al., NDSS 2019](https://www.ndss-symposium.org/ndss-paper/adversarial-attacks-against-automatic-speech-recognition-systems-via-psychoacoustic-hiding/) | Use psychoacoustic thresholds to restrict perturbation opportunities when attacking ASR, and evaluate human transcription. Explicitly controlling perceptibility during ASR optimization predates the proposed experiment. Their perceptual thresholds are not interchangeable with learned representation or STFT distances. |
| [Yu et al., NeurIPS 2020, PCGrad, Algorithm 1](https://arxiv.org/pdf/2001.06782) | For minimizing losses, conflicting task gradients have negative inner product; the algorithm subtracts the conflicting projection and aggregates modified task gradients. This is gradient surgery for multitask optimization, not an enforcement mechanism for nonlinear hard constraints. Its pairwise operation inspires a direction adjustment but supplies no hard-cap or listening guarantee for this attack. [Author code](https://github.com/tianheyu927/PCGrad). |
| [Liu et al., NeurIPS 2021, CAGrad, Eq. 3 and Theorem 3.2](https://arxiv.org/html/2110.14048v2) | For updates `θ − αd`, CAGrad maximizes the worst individual gradient improvement subject to `||d − g₀||₂ ≤ c||g₀||₂`. Its main objective is an average loss; the paper also permits another specified main objective. With its smoothness, lower-bound, step-size and `c < 1` assumptions, convergence concerns stationarity of the main objective. Those results do not certify this nonconvex attack, a finite-step cap, or human intelligibility. [Official code](https://github.com/Cranial-XIX/CAGrad). |
| [Rosen, 1960, linear constraints](https://epubs.siam.org/doi/10.1137/0108011), [Rosen, 1961, nonlinear constraints](https://epubs.siam.org/doi/10.1137/0109044), [Zoutendijk, 1959](https://academic.oup.com/jrsssb/article/21/2/338/7027990) | Gradient projection and feasible-direction methods are classical constrained optimization. A source-gradient direction adjusted to respect local distortion constraints belongs to this family. This experiment should not claim to invent the optimizer. |

## Direction and feasibility calculation

The following equations are this note's derivation for checking the proposed implementation, not equations attributed to the preceding papers. Let `δ` be the waveform residual, `A(δ)` the source model's reference-transcript NLL to **maximize**, and `D₁,D₂` the STFT and clean-content distortions to **upper-bound**. Keep the clean waveform, clean encoder reference, normalization, and caps fixed throughout refinement:

\[
\tau_i=D_i(\delta_{50}),\qquad
g=\nabla_\delta A,\qquad a_i=\nabla_\delta D_i.
\]

An ascent update is `δ + ηd`. At an active cap, the first-order requirement is `aᵢᵀd ≤ 0`, whereas ascent requires `gᵀd > 0`. The sign is opposite to treating all three raw gradients as minimization objectives. A simultaneous local projection can be written

\[
d^*=\arg\min_d\tfrac12\|d-g\|_2^2
\quad\text{s.t.}\quad Bd\le0,
\qquad B=[a_1^\top;a_2^\top].
\]

Its KKT conditions are `d = g − Bᵀλ`, `λ ≥ 0`, `Bd ≤ 0`, and `λᵢ(Bd)ᵢ = 0`. With only two constraints, test the empty, singleton and joint active sets; verify residuals and handle dependent or near-zero gradients. Projecting against every cap, even when inactive, is conservative. A slack-aware local alternative uses `aᵢᵀd ≤ (τᵢ − Dᵢ)/η`; this remains only a linearization.

Critical implementation checks:

1. **Simultaneous constraints matter.** One pass of pairwise projections can undo an earlier constraint: for `a₁=(1,0)`, `a₂=(−1,1)`, `g=(1,1)`, the successive half-space projections give `(0,1)` then `(0.5,0.5)`, violating the first constraint. Verify all constraints after the final operation.
2. **Do not change the direction after certifying it.** Applying `sign`, Adam preconditioning, or coordinate clipping can break a dot-product condition. For `a=(2,−1)`, `d=(1,2)` is tangent, but `aᵀsign(d)=1>0`. Positive scalar normalization preserves the homogeneous conditions; arbitrary coordinate transformations do not.
3. **A tangent direction is not finite-step feasibility.** At `δ=(1,0)` with `D=||δ||₂²`, direction `(0,1)` has zero directional derivative but increases distortion by `η²` for every nonzero step. The actual nonconvex STFT and encoder distances require fresh nonlinear checks after amplitude/RMS/valid-audio projection. Backtrack or retain the previous feasible waveform. Exhausted backtracking and zero projected directions are valid recorded no-progress outcomes. A strictly inward margin can help but has no universal nonzero solution.
4. **Keep cap tolerances explicit.** Register numeric tolerances before outcomes; log every initial cap, accepted distortion, objective, step length, backtrack and rejection. Never silently relax caps after failure. Recheck the saved waveform if serialization changes its values.
5. **Name waveform projections accurately.** Clamping and radial scaling can produce a feasible waveform without being the exact Euclidean projection onto the entire constraint intersection. If both `||δ||∞ ≤ .003` and `RMS(δ) ≤ .003` apply to the same samples, the RMS constraint is redundant because `RMS(δ) ≤ ||δ||∞`. Padding must not dilute RMS or content denominators.

## What an experiment could support

The cap formulation guarantees at most that **measured** distortions on accepted waveforms remain below the specified per-utterance baseline levels, within registered tolerance. It does not guarantee equal UTMOS/DNSMOS/NISQA, unchanged human transcription, imperceptibility, source-decoded WER improvement, or transfer to another ASR. NLL and decoded WER are different objectives. The preceding psychoacoustic ASR papers demonstrate why listening evidence must be collected separately; our proposed caps are not their masking model.

Use the identical manifest, transcript policy, source decoder, saved-waveform scoring, starting attack, and source-backward budget for comparisons. Report extra distortion backpropagation, backtracking forwards, elapsed time and memory separately: equal source backward counts are not equal total compute. Distinguish 50+50 refinement from an independently initialized 100-step control; include ordinary continuation from the same 50-step state where feasible to isolate the effect of constraints. Never equate a waveform `.003` budget to the paper's feature-space Frobenius radius `.1`; the decoder changes the geometry and units.

Freeze the refinement and promotion criteria before validation. A better source-model tradeoff supports a source-model claim; full tests, transfer evaluation and human assessment must support broader paper claims. This is a targeted prior-work check, not an exhaustive novelty search.
