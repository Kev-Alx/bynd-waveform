# Proposed defense evaluation contract

Research proposal, 22 October 2026 (Asia/Taipei). **No new split, training run, attack or result is registered by this document.** Before implementation, materialize the selected design's configuration, data roles and budgets in a frozen receipt under this directory. Existing exposed benchmarks remain diagnostics.

## Task, deployment access and primary question

Primary task: English sentence transcription, with corpus WER computed as `100 × Σ(S+D+I)/Σ reference words`. Lower is better. Use the same utterances, references, audio export policy, base checkpoint and effective decoder for each paired comparison. Preserve unclipped WER above 100%.

The defender sees the test waveform only. It cannot consult the clean reference, true transcript, attack trace or a teacher that consumed held-out labels. Fix language/task behavior, timestamp/no-speech options, beam/temperature, text normalization and precision explicitly. Retain raw hypotheses and edit counts. Keep a secondary standard-normalizer sensitivity report separate from the primary frozen scorer.

The primary comparison is new defense versus the strongest implemented and fully specified baseline at matched clean accuracy and resource track. It is not numerical proximity to the original paper's WER. A defense with worse attacked WER than undefended recognition has failed on that condition regardless of a detector score.

## Data roles and independence

| Role | Proposed data | Rules |
|---|---|---|
| Train | Audited natural train-clean-100; initially the available nested 1 h/10 h subsets for adaptation | Pair with clean/reconstruction/attacked variants; never train on evaluation speakers |
| Development and early diagnosis | Legacy dev-clean pools, explicitly exposed | Freeze screen IDs and comparison grid before experiments; no claim of untouched confirmation |
| Confirmation | Newly allocated unused speakers after exposure audit | One candidate, one decision rule; no repeated hyperparameter selection |
| Final primary test | A new speaker-disjoint reserve, not the historical999 or another task's reserve200 | Remains unopened until method, attack suite and baseline settings are frozen |
| Historical regression | Original 999 test-clean | Report exact paired results as exposed benchmark |
| External generalization | A separately pinned corpus such as FLEURS English | Audit source/speaker metadata and pretraining overlap; do not assume speaker IDs exist |

A practical candidate for fresh confirmation/final allocation is **LibriSpeech dev-other**, whose official publisher lists a separate harder-speech distribution. Its name does not prevent assigning a newly frozen research-test role. First audit that neither its audio nor speakers have been used in historical manifests/caches. If eligible, split speakers into disjoint confirmation and final groups using a fixed hash/seed, then sample without looking at ASR or quality outcomes. Planning sizes are roughly 100 confirmation and 400–500 final clips, subject to actual speaker/duration availability and power analysis. These are proposed sizes, not existing reserves. If eligibility fails, choose a different source before scoring anything. Source: [OpenSLR LibriSpeech](https://www.openslr.org/12/).

The existing protected 48 and test-other reserve200 are allocated to previous experiments and must not be consumed by this task without an explicit later allocation. The scored improvement48 is a third distinct cohort. Use absolute path/hash in every protocol. Audit ID, decoded PCM hash, speaker, chapter/source-recording and near-duplicate text across manifests, caches, donor banks and teacher outputs. Preserve cross-task reservations. Local non-exposure does not imply that Whisper/SSL pretraining never saw the corpus.

FLEURS is a possible external utterance-level check, with CC BY 4.0 shown by its [official card](https://huggingface.co/datasets/google/fleurs). Its accessible speaker metadata must be inspected before promising speaker-cluster confidence intervals. AISHELL-1 is a later Mandarin extension with its own phone/tone inventory, training splits, fine-tuned surrogate and CER scorer; [OpenSLR](https://www.openslr.org/33/) is the primary data source. No downloads are needed for this planning phase.

## Three access models, reported separately

1. **Original non-adaptive transfer.** Generate through the frozen public surrogate without the defense or defended target. Use the full supplied FV objective at `ρ=0.1, λ=1, 50 updates` under recorded independent optimizer choices. This is the closest scope to the paper. The historical `ρ=0.5, λ=0.1` attack is a separate stress condition.
2. **Defense-aware surrogate transfer.** Attacker knows the defense design and trains/uses an approximation or ensemble; final target parameters remain unavailable. Count surrogate-training and target-query access explicitly.
3. **Direct adaptive white-box.** Attacker optimizes through the actual complete defended model, including auxiliary phone path, canonicalizer, random views, fusion, detector or abstention logic. This is deliberately stronger access than the paper's transfer model and supports the main empirical robustness stress test.

For each model/family report whether attacks were generated against raw or defended recognition. Cached raw-model examples cannot replace current-defense adaptive attacks. If a black-box query attack is used, distinguish access to text only from access to scores/logits, and impose the declared query ceiling.

## Threat families, budgets and validity

Core family is `V(E(x)+δ)` with the verified global relative-Frobenius constraint. Retain perturbations, projection receipts, generator hashes and final waveform hashes. Attack the new defense through the vocoder. Use paper-default radius first and larger-radius sensitivity separately. Fixed λ is an optimization setting, not a hard content guarantee.

Add waveform `L_inf` PGD and a stronger transfer/momentum condition under explicit sample-amplitude bounds. Include a muting/prefix family only with a correctly implemented, separately documented budget and duration alignment. Do not silently substitute it for FV. Hold out at least one matched encoder/vocoder or distinct generative family from all training and selection. A new checkpoint with the same architecture is a weak holdout; a different representation/generation family is stronger. DiffVC-based attacks and GATAS are candidates after source/implementation review, not required available baselines today.

Report both the unchanged full FV objective and adaptive alternatives: lexical sum/mean NLL with explicit scaling; wrong-prefix/sequence or CTC-aligned loss where defined; phone-path/fusion-aware loss; and a margin/targeted-loss probe if NLL saturates. If optimizing special-token/no-speech behavior, label that objective separately from the supplied lexical loss. Different objectives can share a feasible set, but must not silently become the claimed paper implementation.

Every generator has a zero-perturbation reconstruction control. Reject a broken generator contract before claiming it provides a harder attack. Preserve sample rate, duration, loudness/clipping policy and export format; never improve WER by truncating attacked words. Count all model passes and retain all attempts, failures and invalidity reasons.

Validity has separate axes: (a) algorithmic budget/export validity, (b) automatic quality screen, and (c) human word preservation. MOS/SSL/ASR agreement is only a screen. Do not use the deployed phone verifier as the sole content validator: that would make the attack set favor the proposed defense. Quality thresholds should follow the frozen independent scorer and include the paper-like 2.5 screen for comparability; report continuous distributions, not just pass counts. Match attack-strength/quality fronts where attack spaces differ.

On a predeclared stratified subset, obtain blinded human transcriptions of attacked speech and corresponding clean/reconstruction controls. Sample before inspecting favorable defense outcomes; include all selected clips. Report human WER, perceived distortion and uncertainty with annotation/participant counts. Human-label disagreement is a limitation, not something automatically resolved in favor of the defense. Label spoken-word changes as a different threat-validity category. Report whole-set machine WER and validity strata with denominators; do not discard inconvenient clips to improve the headline.

## Adaptive attack adequacy

Start with 50 steps and retain the strongest valid result seen; increase to 200, then up to 1,000 steps on a fixed audit subset if errors or objectives still improve. Use at least three distinct restarts for promoted candidates, then increase restarts if they continue improving. These are proposed stress levels, not commitments that current hardware can complete them. Any lower ceiling must be declared with observed attack-strength curves and a budget-limited conclusion.

Use true input gradients through differentiable branches. For nondifferentiable transformations try more than identity BPDA: a matched smooth transform, codebook/phone relaxation where meaningful, hard-forward/surrogate-backward evaluation, and transfer or bounded query checks. Evaluate adversarial waveforms through the **real hard pipeline**. For random defenses use EOT and test sample counts such as 4/16/64, with fresh held-out inference randomness. Increasing seeds is not equivalent to independent training runs.

For candidate fusion, jointly target both recognizers and the actual score combination. Also try branch-balanced losses to avoid a large branch's gradient dominating. Where hard beam/lattice selection blocks gradients, attack differentiable sequence scores and multiple candidate targets, then evaluate the actual decoder. Wrong but mutually consistent words are successful failures. A broken gradient or failed attack surrogate is not a robustness result.

Check whether stronger budgets, more restarts or alternate losses increase error; whether black-box transfer beats white-box attacks; whether finite-difference directional derivatives match smooth-path autograd; and whether attack loss improves while real WER fails to move. These are diagnostics, not a theorem that the strongest attack has been found. An out-of-memory/timeout must be recorded as an attack failure/coverage gap, never a successful defense outcome.

Report per-family results and a worst-of-suite envelope. For a per-utterance envelope, retain the highest-error **valid** attempt for each original utterance under the predeclared constraint set; the original reference remains the denominator. Report validity coverage, and explicitly mark utterances with no feasible generated attack rather than dropping them. If scoring a full-coverage envelope uses clean fallback for those utterances, disclose that policy and show no-feasible-attack counts beside it. A corpus-level maximum across whole runs is a different summary and should be labeled separately.

## Required controls and comparisons

| Control group | Minimum comparison |
|---|---|
| Recognition controls | Clean, zero-edit reconstruction and attacked audio for every generator/defense |
| Preprocessing | Existing independent LPF and WaveGuard transformation; fixed codec only after sentence compatibility and adaptive attack support |
| Training | Equal-budget clean fine-tune, natural augmentation, waveform AT, FV AT, naive union AT, proposed objective; TransferBreaker adaptation where reproducible |
| Architecture | Equal-resource second recognizer/ensemble, ordinary CTC/attention score fusion, applicable LCAR grounding adaptation, shared versus separate encoder, phone-free versus phone-supervised auxiliary, fusion/expansion ablations |
| Inference | Whole utterance, fixed chunks, overlapping fusion, randomized offsets at matched calls |
| Generalization | Held-out speakers, at least one held-out generator family, external corpus, later Mandarin separately |

Do not supply invented CE-AT/DT-AT/DMW/TI checkpoints or call a sentence adaptation of one-second AudioPure an exact paper baseline. Independently implemented baselines must be named and scoped as such. Resource-matched adaptations and faithful full reproductions are distinct comparisons.

Primary recovery scoring requires a transcript for every clip. If an optional detector abstains, report risk/coverage and treat total abstention as deletions for an all-input WER accounting. Accepted-only WER is secondary and must include accepted/reference-word counts. Detection ROC/AUC and clean false-positive rate are separate endpoints. Never compare a reject-all detector to a full-coverage transcription system as if it restored speech.

## Statistics and promotion rules

Use corpus WER, clean WER, reconstruction WER, attacked WER by family, edit-type counts, per-speaker results, runtime/latency, peak memory, trainable/total parameters and total extra model/data cost. For the phone branch add PER, lattice-oracle WER, rescue/harm counts and error correlation. For resynthesis add post-defense quality and human intelligibility; transcript-only defenses do not need synthetic output-MOS numbers.

Bootstrap paired **speakers**, recomputing corpus numerator/denominator inside each replicate. Use 10,000 resamples when the speaker count supports it; disclose small-cluster limitations and unclustered external-data limits. For learned methods report at least three independent training seeds on the promoted configuration, separately from attack/random-inference seeds. A single seed does not estimate training variance. Predeclare one primary comparison, expose the exploration count and keep secondary subgroup claims descriptive unless multiplicity is addressed.

Proposed development promotion criteria are:

- Clean WER increase no more than 1 absolute point versus the matched undefended base on the same cohort. This is a design target and may reject an otherwise interesting tradeoff; preserve the whole frontier.
- At least 10 absolute points lower directly attacked FV WER versus undefended recognition on the fixed development screen, plus improvement versus the strongest equal-resource baseline. Confirmation must show a paired interval excluding no improvement against that baseline before final testing.
- No collapse under stronger adaptive losses/budgets; known-generator gains must persist in the held-out generator condition. Failure is reported per family rather than averaged away.
- Capacity, data, inference and training cost are within the declared track; the mechanism ablation explains more than adding a second model.

These are prospective decision rules, not a formal statistical power calculation or an assurance of publishability. Use speaker-level pilot variation to size the final sample before opening it. Once the final reserve is inspected, redesigns require a new untouched reserve or a clearly exploratory claim.

## Reproducibility and containment

Every new config, split, source snapshot, downloaded weight/data artifact, cache, log, checkpoint, generated WAV, hypothesis, report and handoff belongs beneath `/workspace/defense`. Existing assets elsewhere can be referenced read-only. New runtime/home caches, temporary directories and external library downloads must be explicitly routed here in the implementation phase; merely putting the final report here is insufficient.

Store provenance for source revision, working-tree hashes, dependencies, exact checkpoint digests, frontends, effective decoder defaults, scorer channels/calibration, randomness and numerical policy. Freeze selection receipts before confirmation/final access. Test changes appropriate to the implemented mechanism, not a large copied baseline suite. No external notifications, model training, or reserve evaluation are launched by this plan.
