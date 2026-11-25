# Constrained refinement with automatic larger validation

Registered22 October2026, Asia/Taipei. The user authorized larger validation/full runs when justified and asked that the work focus on improving the paper rather than repeatedly screening small pilots. The next queue therefore includes a frozen, conditional validation stage, without waiting for another chat turn.

## Evidence and proposed change

The completed48-point comparison found that full-objective waveform PGD reaches43.4608% source WER /3.8873 UTMOS /3.7776 NISQA, while NLL-only PGD reaches58.7525% WER but fails NISQA at3.4367. The frozen learned waveform generator produces no increased recognition error. This establishes a viable per-example attack direction and a remaining strength/distortion tradeoff; it does not establish transfer or human semantics.

The new method starts from the full-objective50-step waveform attack and performs50 additional source-NLL ascent steps while imposing **fixed per-recording STFT and WavLM6 distortion caps equal to the initializer's values**. This is a constrained optimization experiment, not a coefficient sweep. The absolute waveform limits remain Linf≤0.003, RMS≤0.003 and peak≤1. RMS is redundant given the identical Linf bound but remains reported.

At each update, differentiate source NLL and both distortions separately. Project the box-free preferred source-gradient sign simultaneously against the two normalized quality gradients by an active-set two-plane Euclidean solve. A fixed1% inward margin relative to each conflicting preferred-direction component supplies first-order room at curved boundaries; if that margin is infeasible, record a zero-margin fallback. Apply only positive Linf normalization afterward. Box masking is conservative and can suppress some feasible moves.

Try step sizes0.00012×{1,1/2,…,1/512}, in that fixed order. Check actual nonlinear STFT/content caps, physical bounds and source-NLL nondecrease **after** waveform projection. Reject an invalid proposal, backtrack, or retain the current feasible waveform if all trials fail. Caps allow only numerical tolerance max(1e-12,32×dtype-epsilon×abs(cap)); source nondecrease allows max(1e-7,8×epsilon×abs(NLL)) and also enforces the original initializer's floor. No decoding or MOS score chooses an iterate.

Projection methods, two-stage ASR attacks and perceptual constraints have substantial precedent. [The primary-source review](execution_refine/prior-work.md) documents those collisions and mathematical pitfalls. No standalone novelty claim is made. A cap on two proxy losses guarantees neither automatic MOS nor human intelligibility.

## Controls and compute

All conditions share the same inputs, source model, original transcripts, native decoding and quality scorers. The six reported variants are clean, original FV50, original FV100, full-objective waveform50, weighted waveform100, and constrained waveform100. Weighted100 extends the identical initializer with another50 ordinary full-loss sign updates. FV100 uses the same LR0.02/relative-radius0.1/full-TV-HF Adam trajectory as FV50 and saves both endpoints; pilot step50 must exactly match the frozen prior delta tensor.

The proposed method and weighted continuation each use100 total source backward calls, counting their shared50-step initializer. FV100 supplies an original-method control at100 source backward calls. The constrained method also uses line-search forward calls, all explicitly counted; this is not a claim of equal runtime or equal threat-model norms. Frozen50-step controls remain visible.

The recorded prior scheduled process wall time is2.1278 hours, including source/MOS CPU wall time and counting the R3 job only once. [The ledger](execution_refine/compute-ledger.md) discloses early timing gaps. The continuation is capped at3.5 scheduled hours, leaving about22 minutes against the original six-hour envelope for those gaps. One GPU stage runs at a time, in a detached process with30-second status updates.

## Freeze and advance to validation

The method first runs the common48-point cohort to establish a valid new optimization path and compare controls. Select the automatic-quality-admissible original control from FV50/FV100 by highest source WER, ties by UTMOS then FV50. The primary development signal retains the existing rule: all mean quality floors (DNSMOS≥3, NISQA≥3.75, UTMOS≥3.5), then either≥3 source-WER points over that control or≥0.05 UTMOS improvement with source-WER loss≤2 points.

The larger **exploratory** validation is also justified by a distinct mechanism signal: an admissible, failure-free continuation gains≥3 WER points over its full50 initializer with UTMOS loss≤0.05, or gains≥0.05 UTMOS with WER loss≤2. That rule can advance an improved waveform frontier even if it has not yet beaten the original FV control. It does not relax or rename the primary superiority criterion.

Among qualifying constrained100/weighted100 conditions, freeze the one with highest source WER, then UTMOS, then constrained100. Save the candidate/control configuration, pilot scores, audit, comparison and validation-manifest hashes in a read-only selection receipt **before** larger validation. The validation reports both continuations as controls but cannot reselect the primary candidate. If neither earns continuation, stop this method rather than repeating pilots.

The larger cohort contains **76 further utterances /34 speakers**, formed from historical validation100 minus the24 additions to pilot48. It has no utterance/hash/text overlap with the48-point screen or protected reserves. It shares26 speakers with the pilot and confirmation pools and was exposed to historical study; it is not an untouched final reserve or speaker-generalization test. Native16kHz mono whole-waveform metadata and source hashes are verified in [the data receipt](execution_refine/validation-76.receipt.json).

## Reporting boundary

The queue generates and scores every registered recording, runs an independent CPU artifact audit, freezes selection, and conditionally executes the larger validation automatically. Unexpected model/contract errors stop the job; mathematical refinement failures retain the last attempted feasible waveform and disqualify that condition from promotion. Failed initialization stops before an invalid partial-cohort performance claim.

Report source WER from summed edits/reference words, each MOS separately, every failed/no-op update, source and quality gradients/counters, backtracking costs, physical/cap checks and descriptive paired speaker intervals. The selected candidate remains fixed through validation. Confirmation48, final200, sealed recognizers, human recruitment and external notifications remain unused. Positive source validation would support the next frozen transfer/content study; it would not establish final superiority, novelty or the paper's full claims.
