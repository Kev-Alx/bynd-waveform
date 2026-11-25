# What is changing, why, and what would count as an improvement

Updated 22 October 2026, Asia/Taipei. This is the plain-language explanation of the research work. All new code, data manifests, generated audio, logs and reports live under `/workspace/improvement`.

**Expanded direction:** the user subsequently authorized larger models, more data and longer runs, with an explicit goal of **WER above75% while retaining transfer and defense persistence**. The small-envelope test below completed at29.1751% WER and is closed. The [new48-hour protocol](14_scaled_research_protocol.md) adds Whisper-large-v2 and a distinct Data2Vec CTC source, up to500 ensemble updates, a1500-step source-call control, and a separately qualified112.4M BigVGAN decoder. [Current results](15_scaled_results.md) and [live status](JOB_STATUS.md) supersede the earlier running-state references below. Larger teacher-based learning on the prepared ten-hour corpus is a gated next branch, not an already-trained result.

The current clean-energy experiment started detached at **22:20 on22 October2026 (Asia/Taipei)**. Read [live status](JOB_STATUS.md) for its present state and [the results document](13_envelope_results.md) for completed measurements.

The goal is to make speech harder for an automatic recognizer to transcribe while keeping the original speech understandable and natural to people. A useful improvement must increase recognition errors at acceptable speech quality, or preserve attack strength while improving quality. Raising the attack model's training loss alone does not establish either result. We have **not yet demonstrated an improved method**, human intelligibility, or transfer to unseen recognizers.

## The original method

The reproduction encodes speech with frozen WavLM layer6, changes those features for each utterance, and passes them through a matching HiFi-GAN vocoder to produce audio. Whisper-small supplies the attack gradient. The optimizer tries to increase the correct transcript's negative log likelihood (NLL), with feature/spectral penalties and a relative feature-radius limit. The models stay frozen; the per-recording feature perturbation is optimized.

The central question is whether feature-space editing is necessary for the attack–quality tradeoff, or whether a different objective, search space, or learned policy can do better. Comparisons keep the source recognizer, original transcript, audio duration and quality scorers fixed. Feature-radius and waveform-amplitude budgets have different units and are disclosed separately.

## Changes already tested

| Change | What was modified | Intended benefit | Observed result and decision |
|---|---|---|---|
| R1: decoded word-risk | Replaced correct-transcript token loss with a probability-weighted word-error objective over candidate transcripts | Make optimization track recognition mistakes more directly | Native candidate sets sometimes had no incorrect words and therefore no useful gradient. One registered deletion-candidate repair produced attacks, but missed the quality floor. Stop this version. |
| R2: local feature mixtures | Replaced an arbitrary feature perturbation with mixtures of nearby, similar clean feature frames | Keep edited features closer to plausible speech | Too few usable directions; recognition stayed near clean performance. Stop this version. |
| R3: learned waveform generator | Added a1.59M-parameter waveform U-Net, trained on a disjoint1-hour speech subset instead of optimizing features separately for every recording | Learn reusable attack directions and reduce generation cost | The one-seed/one-hour feasibility run emitted almost zero perturbation and changed no transcripts. A per-recording optimization control was needed to separate a failed learned policy from a failed waveform objective. |
| Waveform PGD50 | Optimized a small additive waveform residual directly; compared NLL alone with the same STFT/WavLM penalties used for R3 | Test whether the waveform space and objective can attack at all | Both attacked successfully. NLL alone damaged quality; weighted losses preserved more quality but reduced attack strength. This localizes the learned-generator failure and exposes a quality bottleneck. |
| Weighted100 versus constrained100 | Continued the same50-step waveform attack for50 additional steps. One condition kept weighted losses; another required actual STFT/content distortions to stay within their initial values | Recover attack strength without increasing the two distortion measures; control for simply running longer | Weighted continuation gained recognition errors but failed NISQA. Hard constraints rejected1,789/2,400 attempted updates and left corpus WER unchanged. Neither earned validation. Stop this constrained refinement. |

The R1/R2 and initial generator results used the earlier24-point study. Subsequent pilots use48 paired recordings as requested. Results from different cohorts are kept separate.

## Latest completed comparison:48 recordings /994 reference words

Higher WER means a stronger attack. Higher quality scores are better. These are automatic scores on the public source recognizer, not human or held-out transfer measurements.

| Method | Source WER % | DNSMOS | NISQA | UTMOS |
|---|---:|---:|---:|---:|
| Clean speech |4.0241|3.2721|4.0533|4.0259|
| Original feature method,50 steps |49.7988|3.1739|4.0698|3.5789|
| Original feature method,100 steps |65.4930|3.0888|4.0057|3.3105|
| Weighted waveform attack,50 steps |43.4608|3.1656|3.7776|3.8873|
| Weighted waveform attack,100 steps |53.4205|3.1158|3.5937|3.8390|
| Constrained waveform continuation,100 total steps |43.4608|3.1655|3.7771|3.8871|

The fixed mean-quality floors are DNSMOS≥3, NISQA≥3.75, UTMOS≥3.5. The weighted100 result missed NISQA by0.1563. The original100-step control missed UTMOS. The constrained result raised source loss slightly but did not raise decoded word errors. Its first successful recording therefore did **not** predict a successful completed experiment.

The previous detached queue completed normally at **17:21 on22 October2026, Asia/Taipei**. Every stage returned exit0 and the independent artifact audit passed. Its frozen scientific gate then selected **stop**, so the76-recording validation did not start. An idle GPU after that point reflected a finished negative experiment. [Completed results](10_refinement_results.md) and [frozen decision](execution_refine/selection.json) retain the evidence.

## Current modification: a clean-energy waveform budget

The next hypothesis is that a fixed absolute waveform budget permits too much perturbation in quiet regions. This is a testable explanation, not an established cause of the NISQA failures. Simply increasing optimization time or freezing both old distortion measures has already failed to resolve the tradeoff.

For each recording, compute the clean RMS energy in overlapping25ms windows with10ms hops, including an end-aligned window. Each window supplies a perturbation bound equal to the smaller of0.003 and its clean RMS multiplied by `10^(-30/20)`. Each sample receives the **smallest** bound of every window covering it. Silent windows allow zero perturbation. These limits are computed once from clean audio; they do not depend on decoded output or MOS scores.

The primary candidate runs50 zero-start NLL-ascent updates inside these time-varying limits. Its coordinate step is twice that coordinate's bound divided by50. A full-weighted-loss version is an objective ablation. The original feature attack and both previous global-budget waveform attacks remain visible controls. A uniformly attenuated global NLL attack is matched to the candidate's final residual RMS, to test whether any benefit comes merely from reducing total perturbation energy.

Before numerical roundoff, the construction bounds residual energy relative to clean energy in every registered window by30dB. It does **not** model frequency masking, prove that noise is inaudible, or guarantee preserved words. It is an engineering hypothesis informed by speech-aware attack constraints. [The primary-source note](execution_envelope/prior-work.md) distinguishes it from SAGO and Qin's psychoacoustic attack; no new masking principle is claimed.

Success requires a stronger attack–quality tradeoff on all48 pilot recordings under rules frozen before inference. A useful single fixed candidate will advance automatically to76 further exposed-development utterances. If it fails, stop this configuration; do not relax the quality floors or repeatedly search envelope settings. The exact gate and resource limit are specified in [the new protocol](12_envelope_protocol.md).

## Implementation and evidence boundaries

The original reproduction files and earlier registered experiments remain preserved. New experiment modules sit beside them under `improvement/`; generated results are grouped by execution directory. Generation and formal native scoring run in separate processes so generation's strict numerical settings do not silently change decoding. Long stages use a detached supervisor, a shared GPU lock,30-second status updates, stage logs and explicit exit receipts. [Live status](JOB_STATUS.md) is the place to distinguish running, completed, skipped and failed stages.

There is also an implementation repair in the frozen WavLM adapter: cloning an input needed for autograd avoids an upstream in-place mutation. Tests verified unchanged forward values and the intended gradients. That repair enables experiments; it is not itself a scientific improvement.

The earlier recorded scheduled compute plus the completed refinement is about3.39 hours of the initial six-hour envelope, with early timing gaps disclosed. The next queue is capped at two hours, leaving allowance for those gaps. No new training corpus or ASR model is added for this test. The untouched confirmation48, final200, held-out recognizers and human study remain unused. Larger exposed-development validation would support a next study, not a publishable superiority claim by itself.

Evidence entry points: [original method](01_original_method_and_baseline.md), [research hypotheses](02_research_directions.md), [literature review](03_literature_review.md), [initial experiments](06_initial_results.md), [48-point mechanism study](08_pilot48_results.md), and [latest completed refinement](10_refinement_results.md).
