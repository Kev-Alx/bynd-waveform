# BeyondWaveform improvement research — steps 1–4

**New agent:** read [HANDOFF.md](HANDOFF.md). The ready-to-paste continuation prompt is [NEXT_AGENT_PROMPT.md](NEXT_AGENT_PROMPT.md).

## Audio comparisons and complete results (26 October)

- [Plain-language summary and proposed next steps](17_summary_and_next_steps.md)
- [Complete WER / quality tables: all 67 conditions](16_all_results.md)
- [CSV](all_results.csv)
- [Audio browser on Docker port 8888](http://localhost:8888/)

The expanded run completed normally in 45h 51m. No tested condition met both WER >75% and all fixed quality floors; validation, transfer and defense evaluation were not triggered.

**Start here:** [what is changing, why, and the goal](11_changes_and_goals.md). The constrained refinement completed with no qualifying improvement; its [results](10_refinement_results.md) remain preserved.

**Expanded objective:** [larger-model/long-run protocol](14_scaled_research_protocol.md), [measured results](15_scaled_results.md), and [live status](JOB_STATUS.md). The user authorized expanding the earlier compute envelope to pursue WER>75% together with measured transfer and defense persistence. The old envelope trial is complete and negative; its artifacts remain preserved.

**Previous completed work:** [clean-energy waveform constraints](12_envelope_protocol.md), with [results](13_envelope_results.md). It did not earn larger validation.

**Latest continuation:** all new pilot evaluations use **48 paired utterances**, as requested. See the [48-point protocol](07_pilot48_protocol.md), [48-point results](08_pilot48_results.md), and [live status](JOB_STATUS.md). The original24-point study and its resolved execution-failure records remain preserved below.

**Continuation:** the user's later “Continue” requests authorized isolated implementation and initial development experiments. See [execution report](05_execution_report.md), [live job status](JOB_STATUS.md), and [machine-readable status](JOB_STATUS.json). The 22 October continuation uses a detached supervisor with 30-second heartbeats and per-stage exit records under `execution/jobs/`. The planning snapshot below remains a record of the completed steps 1–4; current experimental status is reported separately.

The [measured-results table](06_initial_results.md) is refreshed by a detached CPU postprocessor after the waveform trial. That postprocessor also runs the saved-artifact audits and package validation, so reporting continues if the chat is interrupted again. See the [reproduction guide](execution/reproduction-guide.md) for frozen commands and restart limits.

Prepared **21 October 2026, Asia/Taipei**. All artifacts created for this request are in `/workspace/improvement`, one directory at the repository root. The original implementation and existing runs are preserved.

## Decision

The current independent implementation is sufficiently documented and internally consistent to support a new research program. Its saved results do **not** establish reproduction of the paper's joint attack-strength and audio-quality claims, black-box transfer superiority, or human intelligibility. Prioritize different objectives, search spaces and learning paradigms over another learning-rate/regularizer sweep.

The proposed directions are:

1. **Sequence-level, decoding-aware optimization:** replace ground-truth teacher-forced token NLL with a source-only decoded-hypothesis risk formulation; test transfer rather than assume it follows from larger surrogate loss.
2. **Content-anchored local speech-manifold search:** replace arbitrary feature-ball movement with constrained local speech exemplars and explicit content checks; test whether the restricted geometry improves the quality/transfer frontier.
3. **Amortized waveform generation:** learn a small input-conditioned residual model instead of optimizing every feature trajectory through a vocoder. Compare total training-plus-inference cost and quality against per-example search, with additional training data disclosed.

The documents specify controls, near-neighbor prior work, failure cases and alternatives. These are hypotheses, not validated improvements or novelty claims. In particular, earlier diffusion-latent ASR attacks, latent universal attacks and recent phoneme-level generative testing prevent claiming that generative search or amortization alone is new.

## Read the package

| Requested step | Artifact | Contents |
|---|---|---|
| 1. Understand and verify | [Original method and baseline](01_original_method_and_baseline.md) | Equations, implementation, assumptions, essential versus incidental choices, saved-evidence audit, failure modes and resources |
| 2. Explore alternatives | [Research directions](02_research_directions.md) | Ranked departures, mechanisms, falsifiable hypotheses, feasibility and novelty limits |
| 3. Research prior work | [Literature review](03_literature_review.md) | Primary sources, competing methods, implementations and unresolved questions |
| 4. Design experiments | [Experimental plan](04_experimental_plan.md) | Staged experiment matrix, losses, data, metrics, budgets, decision rules and ablations |
| 4. Machine-readable matrix | [Experiment matrix](experiment_matrix.csv) | One row per planned experiment, including support/rejection criteria and compute estimates |
| Evaluation controls | [Data and evaluation constraints](data_and_evaluation_constraints.md) | Dataset candidates/licenses, cohort exposure, target isolation, fair comparisons and content preservation |
| Audit evidence | [Baseline evidence](baseline_evidence.json) | Recomputed saved scores, file hashes, cohort overlaps and resource inventory |
| Repeat the audit | [Saved-baseline audit](audit_saved_baselines.py) | CPU/standard-library verification; does not execute a neural model |
| Source registry | [Literature sources](literature_sources.json) | Source metadata and evidence/availability limits |
| Original instructions | [Request](request.txt) | The user's full pasted research brief; this pass stops after step 4 |

## Evidence established in this pass

The saved-baseline audit independently recomputed WER edit distances and quality-score aggregates for **10 saved runs / 6,570 score rows**, and checked hashes for **6,330 saved waveform files** and **1,189 unique source audio files**. Its scope is saved prediction/score arithmetic and artifact integrity; it did not re-run recognition or MOS networks. All audited aggregates match their saved summaries. The trace-radius checks are not a new tensor-level reproduction of every historical perturbation.

| Existing result | Source clips | WER % | UTMOS | Interpretation |
|---|---:|---:|---:|---|
| Paper reference | 999 | 75.4300 | 3.5900 | Published reference under incompletely recovered author protocol |
| Historical selected attack | 999 | 74.1064 | 2.7924 | Larger feature radius, rho=0.5; lambda=0.1 |
| Paper-budget independent control | 999 | 64.7415 | 3.3582 | rho=0.1, lambda=1, mean NLL; misses the joint paper anchors |
| Repeatable full-objective control | 24 | 74.7368 | 3.3186 | Reused development pilot, sum NLL |
| Channel-weighted Adam coordinates | 24 | 74.1053 | 3.3989 | Descriptive +0.0803 UTMOS at −0.6316 WER points on the same development clips |

These rows are different cohorts/settings, not a fair head-to-head ranking. Higher attacked machine WER indicates more recognition errors; it is not enough to show an improvement unless speech content and quality are preserved. DNSMOS and NISQA, clean/reconstruction controls, complete precision and provenance are in the baseline report and JSON.

Two design constraints are material: the original 20-clip pilot overlaps the 100-clip validation set by two utterances, and all 28 speakers in the unscored 48-clip confirmation cohort appeared in previous development pools. The old 999-clip outcomes are also already exposed to this research process. Future claims need prospective selection and a new final reserve, with overlap checks.

Resources observed here are **one TITAN RTX with 24 GB VRAM**. The plan offers a bounded initial screen and a larger gated research scenario; neither is a user-approved compute allocation or an active job. Runtime estimates must be calibrated on the actual method and clip lengths.

## Work boundary and reproducibility

Completed now: paper/implementation analysis, saved-baseline verification, resource inventory, literature research, ranked falsifiable directions, and an experimental plan. No new attack, training run, target-model evaluation, listening study, defense training, dataset download or improved-method result was produced. The small audit script is verification tooling for step 1.

To repeat the saved-evidence audit from this environment:

```bash
/opt/venv/bin/python /workspace/improvement/audit_saved_baselines.py
```

It refreshes `baseline_evidence.json` inside this directory. See [artifact validation](artifact_validation.json) for the package's structural/link checks; repeat those with `/opt/venv/bin/python /workspace/improvement/validate_package.py`. Steps 5–7 remain future work under the user's requested stopping point.
