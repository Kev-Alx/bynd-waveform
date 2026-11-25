# Prospective experimental plan and promotion protocol

Prepared 21 October 2026 (Asia/Taipei). **Steps 1–4 only. All new experiments in this plan are proposed and unexecuted.** The user has not specified a compute allowance; the 24-hour and 120-hour scenarios below are planning alternatives, not approved reservations or launched jobs. No benchmark results are predicted. The machine audit finds one TITAN RTX with 24 GB, not two GPUs.

## 1. Experimental question and tracks

Does a new method increase zero-target-query ASR transfer failure at a fixed standard of human semantic content, perceived quality, and resources? The original surrogate, raw Whisper-small, is an optimization component; its WER is **not** a black-box-transfer measurement.

- **T0, core:** same current utterance/transcript, raw Whisper-small generation surrogate, common frozen WavLM/HiFi-GAN when applicable, no additional clean training corpus or generation ASR. Feature and waveform methods use distinct physical constraints, with quality/resource-matched comparison.
- **T1, added clean data:** generator training or donor retrieval uses an explicitly disjoint training manifest. Training, retrieval, storage, and preprocessing count as resources. T1 results cannot silently be ranked as a T0 improvement.
- **T2, additional surrogate/labels:** a separate generation aligner or surrogate, never one of the final targets. Report extra model/data access; compare T2 to T2 controls.
- **T3, altered objective/threat model:** universal prefixes, muting, output-length/latency attacks, or target-query methods. Report their intended endpoint separately; low human intelligibility or different access precludes the primary content-preserving transfer claim.

All conditions seeking the primary claim must preserve the original sentence to human listeners. Pretrained resource access is not equal across representations: FV has WavLM and HiFi-GAN in addition to Whisper, while simple waveform PGD may use Whisper alone. Report model parameters, training provenance, checkpoint access, and encoder/vocoder runtime; the reconstruction-centered waveform control shares the same initial FV waveform and exposes this difference. An identical feature radius cannot be imposed meaningfully on waveform, codec, and generative families. Accordingly, publish within-family constraints and resource/quality Pareto curves, rather than declaring unlike norms equivalent.

## 2. Data registration before any new scoring

Existing repeatedly used pilot/validation cohorts and both exposed 999-clip runs are **development history**. The audit also found two identical ID/audio rows shared by the original pilot20 and validation100; these are not independent validation evidence. Existing 999 performance can be replayed for continuity but cannot become a fresh final test by renaming it.

The unscored 48-clip manifest has SHA256 `377a51cacce705ee4e6d5777aad4d53c042cb473e4b0a4dd001273a8d640a713`, 319.725 seconds and 28 speakers. All 28 overlap prior development speakers. It is audio/utterance-disjoint confirmation only, **not speaker-disjoint final evaluation**. The saved-score audit found none of these 48 in the checked score history. Preserve it unscored during development.

Before method selection, a future implementation pass must freeze these manifests under this same `/workspace/improvement` directory:

1. A source-only development set, with inherited exposure and all overlaps labeled. Use it for bounded debugging, operating points, and checkpoint selection.
2. A training manifest for T1 (nested speaker-balanced 1-hour / 10-hour LibriSpeech train-clean-100 subsets, selected duration-first without outcomes), filtering speakers and audio/text duplicates against every development/confirmation/final pool. train-clean-100 is not currently local; downloading and inspecting it is future work. Freeze corpus license, hashes, speaker IDs, duration/language filters, and any crop-to-transcript alignments.
3. The existing unscored48 manifest, kept as a one-use, no-retuning confirmation.
4. A **new, untouched final reserve**, proposed initially as 200 English utterances from at least 30 verified speakers absent from all training/development/confirmation data, preferably external natural speech. Freeze its selection seed, corpus version/license, audio hashes, speaker/source identities, transcript references, exclusions, and duration strata **before selecting methods**. Sample without attack, ASR, or MOS outcomes. Inspect FLEURS metadata rather than presuming speaker IDs or disjointness; if adequate identity/provenance cannot be established, use another licensed corpus. MLS shares LibriVox origins, so check source-level duplication. Do not claim that any final reserve has already been acquired or frozen.

Exact waveform hashes are insufficient: check speaker/source identities, resampled/cropped audio similarity, and repeated passages. Disclose overlap with public model pretraining wherever identifiable; a fresh attack-study reserve is not automatically absent from ASR or vocoder pretraining. If final provenance cannot be verified, stop at development conclusions. All sample eligibility rules are fixed before outcomes; no post-hoc dropping of difficult clips or changing the common intersection.

## 3. Frozen generation, target isolation, and baseline integrity

Keep the current compatible WavLM layer-6 / regular feature-conditioned HiFi-GAN contract, native 16 kHz audio, clean denominators, and exact transcript/scorer settings for FV controls. Preserve original source/config/checkpoint bytes and SHA256 bindings. Strict deterministic generation uses the existing verified path and records CUDA/cuBLAS/attention/padding policy; saved-audio evaluation keeps its separately frozen native policy. Exact-repeat tests address execution reliability, not cross-seed uncertainty.

Generation and source-only model selection must not invoke the following targets, inspect their new predictions, or tune to their historical idiosyncrasies:

| Frozen target role | Existing target definition to bind before execution |
|---|---|
| Whisper-family 1 | Raw multilingual Whisper-base; existing local checkpoint; beam_size=5, temperature=0 |
| Whisper-family 2 | Raw multilingual Whisper-medium; existing local checkpoint; beam_size=5, temperature=0 |
| CTC-family 1 | facebook/wav2vec2-large-960h-lv60-self, existing pinned snapshot `54074b1c16f4de6a5ad59affb4caa8f2ea03a119` |
| CTC-family 2 | facebook/hubert-large-ls960-ft, existing pinned snapshot `ece5fabbf034c1073acae96d5401b25be96709d8` |

The frozen execution plan must bind full Whisper checkpoint hashes, CTC preprocessing/decoder settings, text normalization, and all quality-scorer versions. These are existing independent target choices; author checkpoint parity is not established. Verify all four are executable before registration. If one cannot be evaluated, amend the endpoint **before any new target scoring**; otherwise label the primary endpoint incomplete and report available components without silently changing weights.

Baseline portfolio:

- **B0 clean/reconstruction/neutral controls:** original x, V(E(x)), zero perturbation, matched no-attack neighbor resynthesis, and norm-matched random perturbations. Report reconstruction-only WER changes; poor resynthesis is not evidence of a stronger content-preserving attack.
- **B1 original independent FV reproduction:** paper rho=.1, lambda=1, alpha=1, 50 updates, fc=6000, frozen mean-NLL/LR=.05 control; retain the historical rho=.5/lambda=.1 row only as a separately labeled budget mismatch.
- **B2 stronger local FV control:** full-objective sum-NLL, canonical LR=.1 and higher-quality LR=.02 plus one fixed middle point .05. Source-only quality selection chooses an admissible control before targets are opened. channel_std is a sensitivity row, not the proposed discovery.
- **B3 independently implemented simple waveform PGD:** lexical source NLL, no reused FV optimizer code, strict projection/gradient checks, 50 backpropagation calls. Original-waveform and reconstructed-waveform starts isolate a vocoder-start effect.
- **B4 established transfer/quality baselines:** MI-FGSM; source-compatible psychoacoustic/RIR attack motivated by Qin et al. (adaptation explicitly labeled). VMI-FGSM is conditional on calibrated cost. DAA/DiffVC is a conditional relevant generative baseline: first inspect implementation/checkpoints/licenses, pass reconstruction and 24 GB gates, then report its native budget and matched-resource version separately. Missing relevant strong baselines prevent a broad SOTA claim.
- **B5 different-endpoint checks:** AudioShield universal latent perturbation, Muting Whisper, and SlothSpeech are informative for universality, muting, or latency. Their native transcript/length/access constraints differ. Reproduce an intended endpoint only if artifacts are available; do not call their raw WER a fair ranking of content-preserving per-example attacks.

Paper anchors (75.43 WER / 2.954 DNSMOS / 3.526 NISQA / 3.590 UTMOS) remain historical author numbers under an incompletely identified protocol. Reaching them locally would not establish author-code parity.

## 4. Pre-registered primary endpoint and quality guardrails

Let WER_m,t,s be corpus word error rate (percentage points) for method m, target t, and seed s on the same frozen utterances. Compute it from **summed edit counts / summed reference words**, not the mean of utterance WERs. The primary transfer endpoint is

`T_m = mean_s { 0.5 * mean_Whisper_targets(WER_m - WER_clean) + 0.5 * mean_CTC_targets(WER_m - WER_clean) }`.

There are two targets in each family, so each target initially has weight 0.25. The primary comparison is `DeltaT = T_candidate - T_frozen_quality_admissible_control`. The control is selected using source-only development risk/quality from B1–B4, **not** by ranking newly observed target WER. Publish every target separately as well as family means; a positive macro mean can conceal a failed architecture. Clean WER is a common reference; reconstruction-relative effects are an additional decomposition.

Standard WER can exceed 100% through insertions. Retain it as the primary conventional measure, and report substitution/deletion/insertion counts, insertion contribution to DeltaT, empty output rate, output-length ratio, and a secondary per-utterance WER capped at 100%. A gain driven only by pathological insertion on one target is described narrowly. Inference latency or EOS repetition is a separate endpoint.

Use this **new local benchmark's** predeclared quality gate, not a claim that these numbers reproduce the paper's scorer calibration:

- Mean UTMOS ≥3.50, DNSMOS OVRL ≥3.00, NISQA overall ≥3.75 on the common cohort; exact scorers are frozen and never optimized directly in the core experiments.
- Blinded human transcription WER increase versus original clean ≤2 percentage points, with a one-sided paired 95% upper bound ≤2 points; additionally attacked human WER ≤10%. This is a noninferiority criterion, not evidence supplied by SSL similarity.
- Paired human quality/naturalness decrease versus clean ≤0.5 on a declared 1–5 scale, with the one-sided paired 95% upper bound ≤0.5. Report separate intelligibility, speaker identity, audible artifacts, and failure examples.
- Original sentence intent, duration, channel/rate contract, valid finite audio, no clipping beyond the frozen permitted bound, no intentional text edits, inserted prefix, or silence substitution. Assess semantic preservation through the registered aggregate human noninferiority test and report phonetic failures; one listener transcription error alone is not an automatic failure. A deliberate semantic edit is ineligible even if MOS predictors improve.

Apply gates to complete predeclared cohorts; do not remove bad clips or replace failures. A method failing a gate is ineligible for the primary stronger-content-preserving-attack claim, though its raw results remain visible. Paper-quality anchors are secondary references. Compare all methods' admissible operating points and **quality–transfer–resource Pareto frontiers**, with UTMOS/DNSMOS/NISQA and human results on separate axes; do not compress unrelated scales into an arbitrary weighted quality score.

Each family gets at most three source-only operating points. FV uses the fixed LR {.02,.05,.1} with rho=.1; waveform PGD/generator uses maximum residual L-infinity {.001,.003,.01} on [-1,1] audio, with RMS and psychoacoustic constraints disclosed. These ladders are different physical constraints; fairness comes from common content/quality/resource limits, not claiming an equivalent latent norm. For R1/R2, if optimizer steps differ in cost, run both 50-update and matched-gradient/FLOP-time comparisons, clearly identified. Inference/quality costs count too.

## 4a. Exact initial objectives and source-only calibration

All quantities below are proposed fixed choices; no calibration has yet run. Select three development clips by predeclared short/median/long duration strata and archive their IDs before derivative probes; never use confirmation/final clips.

**R1 units:** let g_N be the gradient of lexical **sum** NLL and g_R the gradient of bounded sequence risk, both with respect to the same physical feature perturbation delta at the canonical starting point. Set `c = median_i(||g_N,i||_F / ||g_R,i||_F)` once, then freeze objective `-c*R + original_TV/HF` with lambda=1. Require every ratio finite, risk gradient norm >1e-8, and c in [1e-4,1e4]; otherwise the initial condition fails as degenerate, with at most the already declared candidate-set fallback. Do not replace zero gradients with a huge coefficient. Report both gradient norms, c, and one c=1 ablation. This prevents a bounded-risk/sum-NLL scale change from masquerading as an objective improvement.

**R2 geometry/content:** optimize only local mixture weights and mixing amplitudes, using the B2 sum-NLL/full-TV-HF objective. Impose `cos(z_t,q_t)>=0.95` at each nonzero frame and global `||z-q||_F/(||q||_F+eps)<=0.1` as hard constraints; zero clean frames remain unchanged. Self-neighbor inclusion guarantees the clean feasible point. Original TV supplies temporal regularity; there is no added unspecified continuity penalty or learned content coefficient. Bank retrieval and similarity are computed only on clean features. Convex interpolation toward q enforces the global radius while preserving the mixture set; verify cosine constraints numerically.

**R3 waveform objective:** define `L_STFT=(1/3)*sum_n mean(abs(log(abs(STFT_n(xadv))+1e-5)-log(abs(STFT_n(x))+1e-5)))` for FFT/window sizes n={256,512,1024}, hop=n/4, periodic Hann windows, center=True, reflection padding, at 16 kHz. Define `L_content=||E6(xadv)-E6(x)||_F^2/(||E6(x)||_F^2+1e-8)`, with the same frozen WavLM layer-6 encoder, native alignment and cached clean features. Optimize `-NLL_sum + lambda_STFT*L_STFT + lambda_content*L_content`. This log-STFT term is a spectral distortion measure, **not** a psychoacoustic masking model. True masking is a separate Qin-style condition.

Set each lambda once using the median ratio of waveform-gradient norm of sum NLL to that penalty's gradient norm on the three declared development clips at a fixed seeded Rademacher residual of amplitude 1e-4, projected to the declared bounds. This nonzero probe avoids the zero derivative of a clean-reference squared distance. Require finite penalty norms >1e-12; otherwise fail the calibration rather than inflate weights. Freeze the resulting coefficients, probe seed 11, and all loss formulas before pilot scoring. Report unweighted and weighted norms. Bound waveform residual by the registered Linf point and RMS<=0.003 of full scale; all competing waveform controls report both limits. A same-objective per-example PGD comparator with the same content encoder separates amortization from added pretrained resources/losses.

A proposed fully specified small architecture is six strided Conv1d encoder blocks with kernel15, stride2, channels16/32/64/96/128/128, GELU, symmetric linear-upsample+Conv1d decoder with skip connections, and a one-channel tanh residual head. No batch normalization or stochastic dropout. Pad to a multiple of64 and crop exactly to original length. Confirm parameter count <=3M before implementation; exceeding it changes the registered model. Train with AdamW LR=1e-4, betas(.9,.999), weight_decay=0, clip parameter-gradient norm1, batch1, accumulation4, at most12,000 optimizer updates or8 measured GPU-hours/seed. Evaluate source-only development checkpoints every500 updates and choose the admissible checkpoint with highest source WER, ties by lower log-STFT loss then earliest checkpoint. The lean training test stops at one hour, one seed, and supplies no generalization claim. Sequential backpropagation/checkpointing for source and content terms counts all passes. R3 uses WavLM as an anchor although it removes the vocoder; report that resource honestly.

## 5. Randomness, statistics, humans, and multiplicity

Use seeds 11, 29, 47 for learned generators and declared stochastic starts/augmentations. Canonical deterministic zero-start attacks are run once; changing a seed that the algorithm does not use is not independent evidence. For deterministic R1/R2, report a predeclared tiny-initialization robustness condition alongside the canonical zero-start result, paired to the same control initializations. This is distinct from exact repeatability. Never select the best seed.

Use 10,000 **paired speaker-cluster bootstrap** replicates: resample speaker identities, retain every utterance and every paired method/target within each selected speaker, then recompute corpus edit-count ratios and DeltaT. Report percentile 95% intervals, target/family intervals, per-seed estimates and range. For learned methods also report hierarchical seed-plus-speaker resampling and distinguish training-seed variability from speaker sampling; three seeds do not justify a highly precise population-of-training-runs claim. If metadata lacks trustworthy speakers, the planned inference is not available. Human uncertainty clusters by utterance/speaker and accounts for raters; a mixed-effects sensitivity analysis is optional, not a substitute for more listeners.

On confirmation, load **Whisper-small only**, automatic quality scorers, and the registered human-screen audio. Do not load any of the four sealed final targets or compute DeltaT on the 48. Promotion uses source/quality plus a descriptive semantic screen only, never transfer outputs. A failure aborts this registered candidate rather than inviting reserve-driven retuning. On final reserve, support the primary hypothesis only if the paired 95% lower bound for DeltaT is >0, mean improvement is at least **3 WER points**, mean cross-seed direction is consistent (and at least two of three learned seeds improve), neither target family worsens by more than 2 points, and all content/quality gates pass. A final interval spanning zero or too wide is **inconclusive**, not a negative proof or permission to harvest another favorable seed. Negative DeltaT or failed content gates reject the claimed mechanism under this budget. Keep selection uncertainty and limited power explicit.

The initial final analysis contains one selected core candidate and one selected control; T1/T2 candidates form separate declared comparisons. If more than one confirmatory hypothesis is retained, register their count and use Holm correction or report exploratory status before scores are opened. Do not select the highest final result and then call its unadjusted CI confirmatory.

Human study is necessary before a stronger-attack claim: propose the same preselected 100 final utterances per condition, stratified by speaker/duration and chosen without outcomes, three blinded independent transcribers per utterance, randomized condition order, no same-listener clean/attack transcript leakage, and separate quality-rating sessions. Include clean and no-attack reconstructions. Report all listener exclusions in advance, compensation/consent and study protocol, and actual precision. The number is a starting design; if its interval cannot meet the semantic gate, the result is inconclusive. Human time/payment is a separate resource (approximately 20–50 person-hours depending on conditions and utterance length), not included in GPU hours. No participants were recruited in this planning task.

## 6. Runtime estimates, cost normalization, and budgets

**Measured history, not new throughput:** the historical 999-clip generation record totals 13.606 s/clip (3.776 computation-hours, 4.971 job-hours). The strict stable-execution workflow lasted 26.28 wall-minutes; the strict three-by-24 loss-ablation workflow lasted 63.07 wall-minutes including preflight, evaluation and verification. These differ in cached controls, strict backend and overhead; none is a calibrated runtime for a proposed architecture.

Use `h24 = 0.30 GPU-hours` for an illustrative 24-clip, 50-step generation condition, with **0.15–0.45 hours** as a provisional planning range. GPU memory estimates are proposals, not measured fits. At execution, calibrate on three predeclared short/medium/long development clips and one fixed training microbatch, never reserve audio. Record model load/warmup, GPU active time, wall time, peak allocated/reserved VRAM, duration, target decode and MOS costs, and numerical policy. Then replace estimates before launching longer jobs:

`H_gen = H_setup + (N/24)*(steps/50)*h24*duration_ratio*measured_method_factor`.

For feedforward generators and no-attack reconstructions, replace the iterative step formula with `H_forward = H_setup + N*measured_forward_seconds/3600`; a zero-step method does not have zero cost.

`H_train = H_load + updates*measured_update_seconds/3600`; `H_eval = N_outputs*sum_t decode_seconds_t/3600 + N_outputs*MOS_seconds/3600`.

`H_total = sum_seeds(H_train + H_gen + H_eval) + retrieval/preprocessing + verification`.

Use sequential model loading, batch-one gradients, accumulation/checkpointing and cached frozen clean features on the single GPU. Training crop length changes are declared, transcript-aligned methodological changes. A fit failure triggers one reduced-width/batch-one attempt, then stops; do not quietly borrow unavailable hardware. Always report 50-step native-method and matched-generation-time/gradient-budget views, and a **lifetime** cost view for learned methods. Generator break-even N satisfies `H_train + N*c_generator < N*c_per_example`; report if no finite evaluated break-even exists. Retrieval/data acquisition/storage and failed runs are included.

Proposed planning scenarios:

| Allocation | Lean first stage, GPU-hours | Full gated envelope, GPU-hours |
|---|---:|---:|
| Baseline/contract checks | 2 | 8 |
| R1 independent mechanism | 6 | 14 |
| R2 independent mechanism | 6 | 14 |
| R3 waveform policy | 6 (tiny feasibility training only) | 28 (≤8 hours/seed plus inference allowance) |
| Data and mechanism ablations | 2 | 16 |
| Verification / confirmation / final | 2 (verification only) | 32 |
| Contingency | 0 | 8 |
| **Total** | **24** | **120** |
| **Fundamental R1–R3 share** | **75%** | **46.7%** |

These are maximum scenario envelopes. Run a cheaper stage first, and stop rather than using budget because it exists. The lean scenario cannot support final superiority or publishability claims. Optional DAA/codec, extra surrogates, or human/final evaluation that exceed their allocation are deferred; no silent expansion. Calibration may make the full final design infeasible; reduce the number of exploratory conditions, not the semantic gate or statistical standard. The R3 28-hour family cap covers at most 24 hours of three-seed primary training plus four hours for generation/verification. Universal/FV-head/data ablations are conditional finalists paid from the separate 8-hour data and 8-hour mechanism allocations; do not schedule every CSV row as mandatory. At most 10% of either scenario goes to numeric hyperparameter tuning. The three-point operating ladders and one mechanism repair count against this cap; if they cannot fit, use the initial point only.

## 7. Stage gates and exact stopping behavior

**G0, contracts and leakage:** freeze data/provenance, target files, source-only selection, resource measurements, loss signs, reference normalization, 16 kHz contract and strict-repeat control. Reconstruct three duration-stratified development clips. For a new matching encoder/vocoder, require reconstruction WER no more than 2 points above original clean and mean UTMOS no more than 0.15 below the existing matched reconstruction on development data; inspect DNSMOS/NISQA and a declared listening sample. Failure stops attack optimization for that pair. Final and confirmation remain unscored.

**G1, independent mechanisms:** run B controls and R1/R2/R3 initial designs on the same development clips. A candidate has a development signal if it meets automatic/content proxies and improves source WER by ≥3 points at a common quality point, or improves UTMOS by ≥0.05 with source WER loss ≤2 points, or reduces lifetime generation cost by ≥2× with source WER loss ≤2 points at matched quality. These are **screening rules only**; source improvement is not proof of transfer. R3 under the lean scenario is only a one-hour-training feasibility test. Stop a family after one failed initial design plus one mechanism-supported repair, or after its allocated GPU-hours. Preserve negative and infrastructure-failure records separately.

**G2, source-only selection and ablations:** execute the rows in `experiment_matrix.csv` needed to distinguish the claimed mechanism. Freeze one candidate per retained track, its operating point, all seeds/checkpoints, expected comparisons and human/final protocol. Select by source-only quality/content admissibility and predeclared risk/cost rule. If more than one admissible candidate remains tied, prefer lower measured lifetime cost, then a deterministic preregistered order R1/R2/R3. Do not open target predictions to break ties. Stop if no candidate qualifies.

**G3, one-use48:** evaluate the frozen pair using Whisper-small source WER and automatic quality only; **do not load or score any of the four final targets**. Preselect 24 of these 48 by ascending SHA256 of utterance ID, before outcomes, with three independent blinded transcribers per condition and matched clean audio in separate listener groups. Require all automatic mean quality floors, source WER no more than 2 points below its fixed control, and descriptive human WER increase versus clean at most 5 points on the 24. This is a fixed semantic screen, not the formal G4 confidence-bound noninferiority test. Disclose all known speaker overlap; no speaker-generalization or transfer claim follows. Abort on gate failure; no candidate ranking or tuning from confirmation outcomes. A later redesign is a new study requiring new confirmation/final reserves.

**G4, untouched final:** evaluate the frozen comparison and all three genuine learned seeds on the common reserve, with sequential targets, human study, speaker bootstrap and full failure accounting. Controls use the same source audios. A proposed 200 utterances / ≥30 speakers is not a power guarantee; estimate precision from development speaker-cluster variability before freezing N, without looking at final scores. If full evaluation/human evidence cannot fit, report the limitation and stop at exploratory evidence. Do not present a reused 999 run or utterance-only48 as final validation.

**G5, optional robustness:** only after an eligible frozen result, evaluate the same WAVs under LPF6k and WaveGuard8k with matched clean/reconstructed controls; report post-defense quality. WaveGuard2k previously damaged clean speech severely and is a diagnostic. Missing CE-AT/DE-AT/DMW/TI/AudioPure/PVP recipes remain missing until resolved, never replaced silently. Physical playback and Mandarin are separate later protocols. G5 is outside the guaranteed lean scenario.

## 8. Reporting and artifact contract

Each matrix row is a proposed experiment with hypothesis, change, architecture/paradigm, data/preprocess, loss/training, metrics, cost estimate, and support/reject criteria. Execution must record status `proposed`, `running`, `completed`, `failed_infrastructure`, `negative`, or `inconclusive`, with seeds, version hashes, cohort IDs, waveform bindings, resource use, and all deviations. Current rows all remain **proposed**.

Report raw WER, source-versus-transfer decomposition, clean-versus-reconstruction-versus-attack quality changes, S/D/I, human semantics, per-speaker/duration strata, uncertainty, training and inference cost, and omitted baselines. Do not invent a “best method” or numerical gains before execution. A contribution claim is conditional on this evidence and the close prior-art checks in `03_literature_review.md`.
