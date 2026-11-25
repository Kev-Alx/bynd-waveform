# Data and evaluation constraints for steps 1–4

Prepared 21 October 2026 (Asia/Taipei). This is a planning supplement. It creates no new data split, downloads no dataset, and reports no new model inference.

## What the existing cohorts can establish

The historical 999 test-clean utterances have already been evaluated and their outcomes have informed subsequent development. Preserve their scores as a reproducibility benchmark, but treat them as exposed development history for the present research program. A new algorithm selected after seeing that history cannot use those same 999 utterances as its sole independent evidence of improvement.

The old 12/24-clip pilots and 100-clip validation sample are also development data. Repeated reuse makes apparent pilot wins vulnerable to selection bias. Resampling these same clips or changing their random seed does not create a new test population.

The saved-manifest audit also found **two identical utterances/audio hashes shared by the original 20-clip pilot and 100-clip validation set**. This is a legacy development overlap, not a discovered overlap with the historical 999-clip test. Retain the old results with that limitation and require explicit ID/audio/speaker exclusions for future cohorts. See [baseline evidence](baseline_evidence.json), `cohort_overlaps`.

The frozen, unscored 48-clip pool is a prospective **utterance-level confirmation** set. The repository documents 28 speakers, all overlapping previous development cohorts; it is not speaker-disjoint. Its manifest hash is `377a51cacce705ee4e6d5777aad4d53c042cb473e4b0a4dd001273a8d640a713`. The saved baseline audit independently checks this claim. Its outcomes must remain unopened until a single candidate and decision rule are frozen. Source: [thread summary](/workspace/docs/THREAD_SUMMARY_AND_PLAN.md:215).

Use a new speaker-disjoint reserve before claiming generalization to unseen speakers. Before selecting it, enumerate speaker/chapter IDs, decoded-audio hashes and normalized transcript hashes across **every** historical manifest, feature cache, teacher output and training pool. Also inspect near-duplicates, shared source recordings, and donor-bank provenance. Corpus names and distinct file names alone do not prove independence. Save the actual reserve manifest and exclusion receipt only in a later, authorized experimental phase; this planning package does not claim one has been frozen.

## Public data candidates and what each would test

| Candidate | Verified availability/licensing | Intended role | Required checks before use |
|---|---|---|---|
| LibriSpeech train-clean-100 | Official OpenSLR distribution, CC BY 4.0; compressed archive listed as 6.3 GB | Small 1-hour and 10-hour attack-training subsets for amortization, or a phonetic neighbor bank | Not present locally according to resource audit. Deduplicate and hold speakers apart from attack-development/evaluation data; preserve attribution. Fix transcript-only sampling without quality/ASR outcome filtering. |
| LibriSpeech test-other | Official OpenSLR distribution, CC BY 4.0; archive listed as 328 MB | Candidate untouched, harder-speech evaluation pool | Audit historical usage and actual speaker overlap before reserving any clips. Report as a difficulty/domain shift rather than a matched substitute for test-clean. |
| FLEURS English | Official Google dataset card, CC BY 4.0 | Candidate external-corpus confirmation of transfer and content preservation | Pin a dataset revision and English configuration; inspect native audio rate, split construction, transcript/reference rules, and speaker metadata. The retrieved card does not establish accessible speaker IDs, so do not promise speaker-cluster inference until metadata supports it. |
| AISHELL-1 | Official OpenSLR SLR33 lists Apache License v2.0 and a 15 GB speech archive | Later Mandarin replication with CER | Gate on an independently frozen Mandarin surrogate and explicit train/dev/test roles. Keep the paper's special surrogate fine-tuning setup separate from zero-additional-data English comparisons. |
| Multilingual LibriSpeech | Official OpenSLR SLR94, CC BY 4.0 | Optional cross-language data expansion | LibriVox origin can overlap LibriSpeech source speakers/books/audio. It is not automatically an independent English reserve. Avoid downloading the entire multi-terabyte English release for an initial pilot. |

Sources: [LibriSpeech publisher](https://www.openslr.org/12/), [FLEURS official dataset card](https://huggingface.co/datasets/google/fleurs), [AISHELL publisher](https://www.openslr.org/33/), [MLS publisher](https://us.openslr.org/94/). URLs and license labels were checked on 21 October 2026; local availability is a separate question from public availability.

## Keep resource and data advantages visible

Use separate comparison tracks: per-example attacks using the same frozen surrogate and no extra attack-training examples; methods using additional public attack-training data; and methods using extra white-box surrogate models. Report model pretraining provenance separately from additional attack-specific training. An amortized model's training cost, teacher-generation cost and donor-bank construction cost belong in its total budget, even if its final inference is fast.

Known pretrained ASR/SSL exposure cannot generally be eliminated by downloading a different split. Disclose available pretraining corpora and unknown exposure; distinguish *unseen by this attack-development process* from *unseen by every pretrained model*. A new public dataset is not proof of the latter.

For learned methods compare 1-hour versus 10-hour speaker-balanced training with the same architecture and explicit example/update accounting. Prefer a smaller, auditable natural-data subset before synthetic augmentation. A synthetic-data study must preserve the same references, disclose the synthesizer and its training provenance, and include a matched natural-data control. Any pseudo-label teacher is an added model and must never be a sealed evaluation target.

## Distinguish recognition failure from changed speech

Every attack condition needs its own clean and zero-edit reconstruction controls. Measure original-to-reconstruction and reconstruction-to-attack recognition changes separately. Also score original-to-attack directly; reconstruction-relative differences alone can hide a poor generator. A fixed feature-space radius is not a universal perceptual budget across a waveform generator, diffusion model and codec.

Automatic MOS, feature distances and phonetic-unit agreement are screening measurements. None establishes unchanged words or human intelligibility. Use a blinded human transcription and listening protocol on a predeclared sample before a content-preserving superiority claim. Report human WER and perceived quality alongside machine WER; a method that changes spoken words has changed the task.

Keep quality scorers and content guards outside the adversarial objective for the main evaluation. If a scorer or guard is optimized directly in an exploratory track, label it, withhold another independent measurement for evaluation, and report the risk of metric exploitation. A guard ASR used for training cannot simultaneously count as a black-box held-out target.

Final target predictions are an evaluation oracle after selection, never a reward, early-stopping signal, candidate ranker or transformation selector. If target results are inspected to redesign the method, retire that target/cohort combination to development and establish another final reserve before a new claim. The experiment plan defines the primary endpoint and statistical controls.

## Defense and language claims stay conditional

The existing LPF and WaveGuard resampling results are explicitly specified transformations; they do not identify every defense in the original paper. Do not average unavailable conditions into a denominator or describe missing defended weights as measured robustness. A newly trained defense, codec front end or extra surrogate would be a separate research condition.

The English primary study does not establish Mandarin, physical playback or human-listening performance. Those extensions need their own frozen protocols and resources. No such experiment was executed in this steps 1–4 package.
