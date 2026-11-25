# BeyondWaveform research: what we tried and what we learned

Updated 28 October 2026, Asia/Taipei. Covers the investigation before the handoff and the experiments continued in this chat, including the completed overnight model run.

**We have not found a method that meets all the requirements together.** Strong attacks usually lose speech quality; quality-preserving methods usually lose attack strength. We demonstrated one exact-output runtime improvement. The newest learned model generates audio quickly, but does not produce a useful attack.

| Read | Contents |
|---|---|
| **This file** | Original method, main changes, results and conclusions |
| [Every run and condition](/workspace/improvement/27_all_runs_comparison.md) | Ten audited historical runs, all **78 scored conditions**, controls, checkpoints and failed attempts |
| [Data, runtime and evidence](/workspace/improvement/28_data_runtime_and_evidence.md) | Dataset sizes/exposure, costs, latency, memory, transfer/defense evidence and limitations |

## 1. Requirements we kept

| Requirement | Meaning and current status |
|---|---|
| Recognition disruption | Source WER **strictly above 75%**; higher means more machine transcription errors |
| Speech quality | Mean **DNSMOS ≥3, NISQA ≥3.75, UTMOS ≥3.5**, all on the same complete cohort |
| Transferability | A qualifying fixed candidate must work on recognizers unused for its training/search; not demonstrated for the new methods |
| Defense resistance | Measure the actual named defenses after qualification; cannot infer it from source WER |
| Practical computation | Quality and runtime have equal priority; include preprocessing, teacher data, training, generation and evaluation |
| Evidence discipline | Later pilots use 48 recordings; preserve untouched reserves, full denominators, original experiments and negative results |
| Research direction | Remains open. Publication planning is deferred until there is a definitive measured improvement |

Automatic quality scores do not prove that humans hear the original words. WER can exceed 100% because insertions count.

## 2. How the original method works

1. Encode a clean recording using frozen **WavLM-Large layer 6**.
2. Optimize a separate feature edit for that recording, using gradients from frozen **Whisper-small**.
3. Decode the edited features using a compatible frozen **HiFi-GAN** vocoder.
4. Repeat the search, then export and evaluate the final audio.

| Component | Original/local implementation | Why we investigated alternatives |
|---|---|---|
| Attack object | A new feature tensor edit for each recording | Every new recording pays for another optimization |
| Objective | Increase transcript negative log-likelihood; penalize temporal variation and high-frequency energy | This loss is a proxy for decoded WER and perceived quality |
| Constraints | Relative feature radius; paper-stated setting is 0.1 | A feature bound does not guarantee retained words or naturalness |
| Search | Paper states 50 updates; our local controls use explicit Adam/LR settings | More updates and larger sources add substantial cost |
| Learned weights | Encoder, decoder and recognizer stay frozen; no trained attack generator | A reusable trained model might amortize generation cost |
| Qualification | Clean and zero-edit reconstruction controls, then attack WER/MOS | A failed reconstruction must not be mistaken for attack strength |

This is an **independent implementation**, not verified author-code parity. Author checkpoints/subset details and several numerical choices were unavailable. The paper's headline numbers and local measurements must remain separate. [Original-method audit](/workspace/improvement/01_original_method_and_baseline.md).

## 3. Main experiments and what changed

D24 = 24 exposed development recordings; D48 = the later fixed 48-recording pilot. These are different cohorts. WER/UTMOS below are representative operating points; all three quality scores and every condition are in the complete run tables.

| Experiment | Change from original | Data | WER% / UTMOS | Result |
|---|---|---|---|---|
| Original controls | Same feature method; quality LR 0.02 versus strong LR 0.1, 50 updates | D48 | **49.80 / 3.579** versus **83.90 / 3.198** | Shows the starting strength/quality tradeoff; neither jointly passes |
| R1: word-risk objective | Optimize expected errors over decoded candidate transcripts; one declared deletion-candidate repair | D24 | **58.53 / 3.350** | Canonical candidates sometimes gave zero useful risk; repaired version still failed UTMOS |
| R2: local feature mixtures | Restrict edits to mixtures of similar neighboring speech frames | D24 | **3.37 / 4.166** | Quality remained high; attack nearly disappeared |
| R3: waveform generator | Train a 1.59M-parameter U-Net to predict a bounded waveform residual | 1-hour training; D24 and D48 | D48: **4.02 / 4.026** | Learned an almost-zero edit; failed to attack |
| Waveform PGD | Per-recording waveform search, with versus without R3's quality penalties | D48 | Weighted: **43.46 / 3.887**; NLL-only: **58.75 / 3.568** | Direct waveform search can attack; NLL-only fails NISQA, weighted version misses WER goal |
| Longer/constrained waveform search | Continue to 100 updates; weighted loss versus hard distortion caps | D48 | Weighted: **53.42 / 3.839**; constrained: **43.46 / 3.887** | Weighted version fails NISQA; hard constraints add no WER gain |
| Quiet-region protection | Limit residual energy using clean local speech energy | D48 | **29.18 / 3.914** | Preserved quality but weakened the attack |
| More steps and larger-source ensembles | Up to 1,500 original updates; combine small/large-v2/Data2Vec, optionally with transforms | D48 | Single 1,500: **101.41 / 2.944**; transformed ensemble 500: **89.54 / 3.085** | High WER, failed UTMOS, very expensive |
| BigVGAN mel search | Use a separately qualified decoder and optimize log-mel features with the ensemble | D48 | 200 updates: **80.18 / 2.295** | Reconstruction was good; attacks failed all three quality floors |
| Decoder-graph reuse | Remove a duplicate decoder render per update | D48 | **83.90 / 3.198**, unchanged | **14.63% faster**, with exact saved WAV parity; quality problem remains |
| Fixed spectrum repair | Blend 25% toward neutral-reconstruction log magnitudes | D48 | **72.84 / 3.360** | UTMOS improved, but WER fell below 75% and quality still failed |
| Learned feature model | Train a 5.07M-parameter conditioned feature policy; recognition-only versus quality-corrected training | 10-hour training; D48 | Radius 0.1: **8.35 / 2.778** versus **4.12 / 3.682** | About **0.14 s/recording** generation, but no useful attack; quality correction trades away the small WER increase |

The longer-run checkpoints come from shared trajectories, not independent retraining. Reused controls are retained rather than counted as new independent evidence.

## 4. What this chat added after the handoff

| Work | What we established |
|---|---|
| Reused the completed investigation | Audited saved results and code instead of restarting the research |
| Diagnosed quality and cost together | Longer searches can raise loss while worsening quality; duplicate decoding and repeated recognizer gradients account for substantial compute |
| Tested graph reuse and spectrum repair | A real implementation speed gain, plus a negative quality-repair result |
| Reviewed alternative model directions | Used primary sources and Consensus; distinguished conditioned generators, universal perturbations, codecs and signal processors |
| Implemented and ran a different learned model | Direct feature-policy training; no optimized teacher-target corpus; measured preprocessing/training/inference/scoring separately |
| Kept the audio comparison usable | Updated the interface with all 78 conditions, saved audio and runtime/model views; port 8888 remains available |
| Preserved evidence | Frozen registrations, original code/results, detached logs, heartbeats, exits, hashes and untouched evaluation sets |

## 5. Latest overnight run: complete measurements, selection error

| Item | Verified outcome |
|---|---|
| Training | Common 300-update warmup; source arm 3,791 updates; quality arm 3,680 updates before its time cap |
| Pilot | All **8 × 48 = 384** outputs generated and scored |
| Performance | WER **3.82–8.35%**; every condition fails the WER goal |
| Cost | Preprocessing **12.15 min**; warmup plus both arms **4.077 h**; mean fresh-audio generation **0.139–0.148 s** for trained conditions |
| Overall timing | **1.947 h** queue wait, **4.529 h** GPU-phase allocation, **6.498 h** total including delivery; within the 13-hour ceiling |
| Execution error | The selection step used a reader requiring unique recording IDs on a timing table that correctly repeats each ID across eight variants; it exited 1 |
| Scientific conclusion | An independent CPU review of the complete saved rows finds **zero qualifying candidates**. The execution error does not erase the measurements or turn them into a success |
| Delivery | Report, interface build and browser checks completed; no validation or transfer/defense evaluation ran |

The training arms share initialization and sampling rules, but their completed update counts differ. This limits a strict equal-update comparison. The original failure receipts are preserved; this summary did not repair or relaunch the experiment. [Gate review](/workspace/improvement/research_summary/model_gate_review.json).

## 6. What we can conclude

| Established | Still unresolved |
|---|---|
| Exact decoder-graph reuse reduces runtime while preserving the same outputs | A method retaining WER >75% and every quality floor |
| Quality penalties/constraints often reduce attack strength | How to preserve natural speech while causing strong decoded errors |
| A small shared model can generate audio quickly on this hardware | A shared model that generates **effective** attacks quickly |
| Longer optimization and larger models did not solve this cohort's joint problem | Transfer and defense preservation for any improved method |
| Completed results, audio, costs and negative outcomes are inspectable | Untouched evaluation, blinded human word recovery and physical playback evidence |

The evidence supports continued open-direction research. It does not yet support a successful replacement for BeyondWaveform or a publication-readiness claim.
