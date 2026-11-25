# Continued mechanism experiments with48-point pilots

Registered22 October2026, Asia/Taipei, after the user requested continuation and48 data points per pilot. This document supplements the preserved24-point initial study. All future pilot evaluations in this work use exactly48 paired utterances. CPU unit checks and replay of historical results are not new pilot cohorts.

The previous run finished:1779 generator updates,7116 microbatches and3541.89 training seconds, with passing artifact audits. Its source WER equalled clean WER,3.5789%. The earlier failed cache-reference and autograd preflights remain visible as historical execution failures; both were resolved before training. The scientific outcome remains negative. The saved failure replay additionally finds median residual RMS1.239e-6, only0.0413% of the0.003 limit.

## Cohort and protected data

The [development manifest](execution48/development-48.jsonl) retains the original24 and adds24 selected from the previously exposed validation100 by fixed SHA256 ordering with seed20261022, without reading model outcomes. It contains48 whole native16kHz mono utterances,30 speakers and360.010 seconds, with durations1.94–24.03 seconds. All source hashes are verified. The [selection receipt](execution48/development-48.receipt.json) binds inputs and exclusions.

Confirmation48 and final200 remain separate and unscored. There is no shared utterance, exact source hash or normalized transcript with either reserve, and no final/training speaker overlap. The pilot shares21 speakers with confirmation, which already had known development-speaker exposure. This is exposed development evidence, not a fresh test or a speaker-generalization study.

## Fixed comparison

| Condition | Purpose | Fixed procedure |
|---|---|---|
| Clean | Input control | Original samples/reference, no normalization |
| Reconstruction | Feature-vocoder control | Strict frozen WavLM6 followed by the matching regular vocoder |
| Original quality FV control | Primary paired quality-admissible comparator |50 Adam updates, LR0.02, relative radius0.1, original sum-NLL plus full TV/HF |
| Initial waveform generator | Architecture/initialization control | Original seed11 untrained generator |
| Frozen trained generator | Replay prior learned method on48 | Prior selected update1500, no new checkpoint selection or training |
| Full-objective waveform PGD | Separate amortization from objective/budget |50 projected sign-gradient updates on the waveform; original R3 losses and coefficients |
| NLL-only waveform PGD | Isolate suppression by soft penalties |Same50 updates/initialization/projection; optimize only negative sum-NLL |

Both PGD conditions start at zero residual, use step size`2*0.003/50=0.00012`, Linf≤0.003, RMS≤0.003 and peak≤1, and save the final iterate. They use the unchanged reference transcript. Full-objective coefficients remain λ_STFT65.08367870148994 and λ_content186.6123439723859 from the original calibration; there is no coefficient/step/amplitude search. The same strict frozen WavLM6 transforms both clean and attacked audio. NLL-only PGD measures the penalties without their backward passes; all actual forward/backward calls and runtime are counted.

All48 clean feature references are regenerated with one strict execution path before their use. This avoids mixing old cached and newly extracted features within the new cohort; no bitwise parity with historical q caches or replayed historical FV outputs is claimed. The fresh FV control shares the exact same registered48 inputs, weights, source decoding and evaluation protocol. FV radius and waveform amplitude constraints remain distinct; comparisons use measured quality and resources rather than an assumed norm equivalence.

## Falsifiable interpretation

If full-objective PGD creates recognition errors while the frozen generator does not, the fixed objective/budget admits an effective per-example direction that amortized training failed to learn. This does not identify gradient cancellation versus architecture limitations by itself. If only NLL-only PGD works, the soft quality penalties suppress this optimizer under the registered settings. If neither works, the present budget/optimizer/objective remains insufficient; no claim that all waveform optimization is impossible follows. Paired clean/attacked teacher-forced NLL is recorded to check whether loss gains actually change decoding.

The automatic development screen requires DNSMOS≥3, NISQA≥3.75, UTMOS≥3.5, then either source WER≥3 percentage points above the paired FV quality control or UTMOS≥0.05 higher with source WER loss≤2 points. A screen pass warrants further mechanism/content testing only. No transfer or human-intelligibility claim follows. Source-only speaker-cluster bootstrap intervals, if reported, are descriptive for this exposed cohort; one learned seed supplies no training-seed variance estimate.

## Execution and failures

The new machine-readable registration is`execution48/plan.json`, frozen before optimization. Native source scoring runs in a fresh process with the historical decoding/backend policy and all336 saved variant WAVs. Its scope guard rejects partial cohorts, missing variants, changed references and unregistered IDs before model inference. No sealed recognizer is loaded.

The detached supervisor runs one GPU stage at a time, writes per-stage logs/PIDs/exit records and refreshes root job status every30 seconds. The new cycle has a two-hour wall-time ceiling inside the original six-hour initial compute envelope. Unexpected infrastructure errors stop execution. An explicitly detected mathematical PGD failure retains its last feasible waveform, saved trace and failure flag in the complete48-point accounting. No clip is discarded or replaced by another method.

There is no new generator training or further R1 candidate repair in this cycle. Postprocessing recomputes edit counts/MOS means, physical constraints, cohort/model bindings and the paired screen before writing the actual results.
