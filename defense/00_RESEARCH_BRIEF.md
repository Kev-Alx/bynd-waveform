# Research brief: defending against BeyondWaveform

22 October 2026 (Asia/Taipei). **The understanding, planning and research phase is complete. No new defense has been implemented, trained or evaluated.** All new artifacts are in `/workspace/defense`, the substitute location explicitly approved by the user when literal `/defense` proved unwritable. The persistent continuation record is [HANDOFF.md](HANDOFF.md).

## Research position

The attack optimizes speech features through a vocoder and a surrogate recognizer. It targets speech-relevant decision failures that simple additive-noise defenses may leave intact. A credible defense must recover the original words from the incoming audio and survive attacks optimized against its full deployed pipeline. A clean reference is available for research/training, not deployment.

The most informative lead is a **separately supervised acoustic phone path with phonetic-evidence-constrained decoding**. Test whether its soft phone lattice retains correct words that the main recognizer loses, then test actual recovery against an adversary attacking both branches. Different weights/frontends do not imply independent errors. A matched ordinary ensemble and conventional score fusion are essential controls.

Two distinct alternatives are **joint waveform/feature-vocoder adversarial training**, changing the learning problem, and **local-context transcript recovery**, changing inference over time. The first supplies a strong learned baseline; the second is an inexpensive way to test whether errors arise from accumulated decoding context. Neither should be added to the lead before its mechanism is understood.

This is a hypothesis ranking, not proof of novelty. Prior work already covers phoneme-space adversarial training, generic latent consistency, joint latent/waveform training, codec bottlenecks, acoustic reranking and CTC/attention fusion. Recent TransferBreaker and the revised neural-codec study make generic latent fine-tuning or quantization claims particularly weak. A contribution must show an additional mechanism and generator-family generalization under stronger adaptive evaluation.

## What the local evidence establishes

Existing historical999 defense results leave attack WER at **73.54% with LPF** and **71.73% with 8 kHz down/up sampling**, versus **74.11% raw**. The stronger 2 kHz condition damages clean speech severely. These are saved results for the older independent attack, not new measurements or exact paper-defense reproductions.

A later paper-budget independent control already exists: **64.74% WER / 3.358 UTMOS** on those 999 clips. The repository has not jointly reproduced the paper's **75.43% / 3.590** anchors. Exact encoder/vocoder/optimizer/scorer and several defense configurations remain unknown; reproducing them is not a prerequisite for studying the clearly specified local threat.

The 999 historical clips are exposed development history for a newly selected defense. Existing protected48 and reserve200 belong to previous experiments and are not allocated to this task. Current filesystem checks confirm available 1 h/10 h training subsets; older absence notes are stale. The GPU was busy with a separate attack queue at the audit snapshot, and this task did not interfere with it.

## Read in this order

| Document | Purpose |
|---|---|
| [Original method](01_ORIGINAL_METHOD.md) | Task, equations, architecture, evaluation, essential components, assumptions and failure modes |
| [Repository audit](02_REPOSITORY_AUDIT.md) | Actual implementation, saved evidence, data availability/exposure and resource constraints |
| [Primary literature](03_LITERATURE_REVIEW.md) | Verified precedents, version corrections, novelty limits and source-backed defense hypotheses |
| [Defense designs](04_DEFENSE_DESIGNS.md) | Three interventions, concrete lead architecture, falsification experiments and alternatives |
| [Experimental protocol](05_EXPERIMENTAL_PROTOCOL.md) | Fresh data roles, full adaptive attacks, content validity, controls, metrics and decision gates |
| [Phase plan](06_PHASE_PLAN.md) | Bounded diagnostic first, conditional implementation/training, confirmation and final-test sequence |

Audit evidence and provenance: [local inventory](evidence/local_audit.json), [primary-source registry](literature_sources.json), [package validation](evidence/package_validation.json), [artifact manifest](evidence/artifact_manifest.json).

The first later experiment should answer whether independent phone evidence or local contexts retain recoverable words under the attack. A failure at that step is a reason to revise the hypothesis before spending a large training budget. There is presently no evidence that a new defense outperforms the existing baselines.
