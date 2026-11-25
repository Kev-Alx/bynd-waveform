# What the current repository can support

22 October 2026 (Asia/Taipei). New work here is a read-only audit of code, manifests and saved statistics. It is **not** new ASR/MOS inference, a fresh full waveform-integrity check, or a trained-defense result. Reproduce the inventory with:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 /workspace/defense/tools/audit_local_evidence.py
```

Output: [local_audit.json](evidence/local_audit.json). It records SHA256 hashes for 39 source/evidence/manifest files, cohort metadata/overlaps, saved count arithmetic, Git state and point-in-time resources. Running it refreshes this research inventory only; its source does not launch any model.

## Starting state and scope

Git HEAD is `d7f70e8a717db7feba35b69dacabade7de196b25`, but the working tree already contains substantial modified/untracked reproduction and improvement work. HEAD alone does not identify the code analyzed here; use the per-file hashes. This task changes files only beneath `/workspace/defense`.

The [strict parity audit](/workspace/docs/STRICT_PARITY_AUDIT.md) is a historical assessment, not the latest results ledger. In particular, its missing paper-budget test was subsequently completed. The [20 October summary](/workspace/docs/THREAD_SUMMARY_AND_PLAN.md) contains the update. The separate [improvement work](/workspace/improvement/README.md) is attack research; its variants are not defenses and must not be relabeled as such.

## Existing measured defense baseline

The saved [defense results](/workspace/reproduction-runs/english-defenses-999/results.json) use the original 999 test-clean utterances, Whisper-small, and 20,310 reference words. Every condition scores clean, zero-edit vocoder reconstruction and adversarial audio. This audit recomputed WER from the saved substitution/deletion/insertion counts and checked agreement with the saved summary, without rerunning recognition or recomputing word alignments.

| Existing condition | Clean WER % | Reconstruction WER % | Attack WER % |
|---|---:|---:|---:|
| Raw | 3.9636 | 3.7272 | 74.1064 |
| Independent LPF, 6 kHz | 3.8355 | 4.0768 | 73.5401 |
| WaveGuard down/up, 8 kHz | 4.0817 | 4.1654 | 71.7331 |
| WaveGuard down/up, 2 kHz | 32.3732 | 34.8006 | 101.6002 |

These transformations leave substantial attack error. The stronger resampling condition badly damages clean recognition. Error above 100% is valid when insertions exceed the reference length. These facts motivate a different defense mechanism, but do not establish that every possible filter or full WaveGuard detector fails.

The suite applies the historical independent attack at `ρ=0.5, λ=0.1`, not the paper-default attack. WaveGuard here is its pinned resampling/PCM16 transformation, not its complete consistency detector or an identified Table 1 configuration. LPF is a specified independent filter. Its results cannot be compared as exact reproduction of the paper's defense columns. Post-transformation MOS was not measured.

## More recent paper-budget control

The later [paper-setting test summary](/workspace/reproduction-runs/paper-quality-research/paper-setting-test-999/evaluation/summary.json) scores the same 999 IDs with `ρ=0.1, λ=1`, 50 updates, mean lexical NLL and Adam LR 0.05:

| Saved attack setting | WER % | DNSMOS | NISQA | UTMOS |
|---|---:|---:|---:|---:|
| Historical selected independent setting | 74.1064 | 2.9638 | 3.8711 | 2.7924 |
| Paper-budget independent control | 64.7415 | 3.1262 | 3.9542 | 3.3582 |
| Paper reported anchor | 75.4300 | 2.9540 | 3.5260 | 3.5900 |

The later control closes the missing-budget experiment, not the joint WER/quality or author-artifact gap. Its saved clean scores are 3.2642/4.0291/4.0845 and reconstruction scores 3.2723/4.1416/4.0284. Local controls must accompany local attack quality; the paper's clean quality differs. The new control has not been shown here to have the same defense evaluations as the older attack.

## Reusable components and missing interfaces

| Existing component | Useful for a later defense study | Boundary |
|---|---|---|
| `feature_attack`, clean-reference losses | Generate paired training/development FV examples | Calls `freeze()` on passed models; do not pass a live trainable defender without a separate inner-loop design |
| Relative-ball projection and saved-delta diagnostics | Verify feasible perturbations | Old headline lacks saved deltas; traces are weaker evidence than tensors |
| Feature/vocoder loaders | Reuse a known compatible attack generator | This WavLM view is not an independent phonetic verifier |
| Whisper lexical NLL and native decoding | Differentiable attack objective and real free-running evaluation | `.transcribe()` itself is not the differentiable inner-loop objective |
| CTC target loader | Existing alternative-model comparisons | Previously inspected target results are exposed; model identity alone does not create a sealed target |
| Defense extension interface | Wrap an inference defense later | WaveGuard NumPy/resampling detaches gradients; adaptive evaluation needs an explicit approximation/alternative attack |
| UTF-8 evaluation and saved edit counts | Stable reporting and corpus-WER aggregation | Freeze effective decoder, tokenizer, normalizer and runtime |
| Job wrappers and integrity plans | Durable logging and audit patterns | They default to paths outside this task; later use must route every new file into `/workspace/defense` |
| Mandarin fine-tuning code | Possible later language extension | Not a ready feature-vocoder adversarial-training implementation |

Relevant source: [attacks](/workspace/beyondwaveform/attacks.py), [losses](/workspace/beyondwaveform/losses.py), [feature models](/workspace/beyondwaveform/models/feature.py), [Whisper](/workspace/beyondwaveform/models/whisper.py), [defenses](/workspace/beyondwaveform/defenses.py), [training](/workspace/beyondwaveform/training.py). No new defense implementation is included in this phase.

## Data available now, and exposure

Metadata was checked directly; earlier notes describing unavailable training data are outdated for the current filesystem.

| Existing cohort | Clips / speakers | Availability and role for this task |
|---|---:|---|
| Legacy pilot | 20 / 18 | Exposed development; 2 utterances overlap legacy validation |
| Legacy validation | 100 / 36 | Repeatedly exposed development |
| Historical test-clean | 999 / 40 | Reproduction/regression benchmark; no longer sole independent evidence for a newly designed defense |
| Reused development pilot | 24 / 18 | Exposed |
| Prior prospective confirmation | 48 / 28 | Separate older protected pool; speaker overlap with development, not allocated to this task |
| Improvement development | 48 / 30 | Scored attack-research pool; distinct from the protected 48 above |
| Improvement larger validation | 76 / 34 | Exposed, currently part of another queue |
| Existing train-clean-100 subset, 1 h | 286 / 251 | All referenced audio files present; potentially reusable training data after an exclusion audit |
| Existing train-clean-100 subset, 10 h | 2,807 / 251 | All files present; includes the 1 h subset, so they are nested learning-curve conditions |
| Another experiment's test-other reserve | 200 / 33 | Audio present; protected by that experiment, not ours to consume implicitly |

The 1 h and 10 h subsets are located through manifests under `/workspace/improvement/execution/data/`, pointing into `/workspace/improvement/data/LibriSpeech/train-clean-100`. Their durations total approximately 3,609 and 36,013 seconds. They span the same 251 speakers; an hour change alone does not increase speaker diversity. Every listed source audio path exists. This is not a fresh decoded-audio hash audit.

The audit confirms zero ID/recorded-audio-hash overlap between the legacy test999 and legacy development pools. It also confirms the old pilot/validation overlap and the designed 24+24 construction of the exposed improvement48. The protected 48 and exposed improvement48 are different cohorts. Neither should be called “the 48” without a path/hash.

The inventory covers named manifests, not every historical cache, teacher output or model-pretraining dataset. Before new splits, build a complete exposure register with IDs, speakers, source recording/chapter, decoded PCM hashes and text duplicates across all tasks. Do not repurpose the other experiment's reserve. Any new final set needs its own provenance, role and speaker exclusions. Public pretrained-model exposure remains a separate limitation even after local splits are disjoint.

## Resources and concurrent work

The audit observes one NVIDIA TITAN RTX with 24,576 MiB. At the recorded snapshot it used about 10,327 MiB and 92% GPU utilization. Existing attack-refinement progress reported `running`, with its supervisor PID present. These are point-in-time observations, not a health certificate or prediction of availability. This task has not interrupted, adopted or duplicated that job.

Do not size a future training run from the apparently free memory while that queue runs. Sequential scheduling, gradient accumulation and measured peak-memory preflight are appropriate initial assumptions. Native BF16 support should not be assumed on this GPU. The paper's L20 timing cannot be used as the local runtime estimate.

## Limits of this evidence

We have established a credible local attack and weak simple defenses, not exact author-code parity or a trained new defense. We have inspected saved quality estimates, not measured human word preservation. No benchmark has been newly scored and no reserved recognizer outputs have been opened by this audit. The next scientific step is to freeze the threat/evaluation contract and test whether independent phone evidence actually survives the attack before investing in a new architecture.
