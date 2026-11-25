# Original method and verified starting point

Prepared 21 October 2026, Asia/Taipei. Scope: steps 1–4 only. This audit reads the supplied paper, current implementation, frozen plans, saved audio and score records. It does not generate attacks, run neural models, train anything, change existing source files, or select a new method from test outcomes. The auditable arithmetic and hashes are in [baseline_evidence.json](/workspace/improvement/baseline_evidence.json); [audit_saved_baselines.py](/workspace/improvement/audit_saved_baselines.py) reruns the saved-evidence checks using Python's standard library only.

## What the method is trying to establish

The supplied [paper text](/workspace/paper/BeyondWaveform.md:48) studies an untargeted, per-utterance ASR robustness attack. An attacker knows the clean recording and transcript and has gradients through a public surrogate ASR, but generates audio without consulting the target ASR or its defenses. Its scientific claim is that searching a representation-to-waveform space produces stronger transfer to unseen recognizers and defenses while preserving intelligibility and perceived quality.

The encoder and vocoder are frozen pretrained components. There is no attack-model training in the original method: each recording receives a separate optimization of a feature tensor. WavLM-Large extracts 1024-dimensional features at approximately 50 frames/second; the selected compatible HiFi-GAN emits 320 waveform samples per frame at 16 kHz. The independent implementation uses WavLM layer 6 and the kNN-VC project's **regular** feature-conditioned generator. It does not perform kNN replacement or add a separately optimized waveform residual. See [feature adapter and loader](/workspace/beyondwaveform/models/feature.py:14).

For clean audio \(x\), transcript \(y\), encoder \(E\), vocoder \(V\), and surrogate \(f_s\),

\[
q=E(x),\qquad z=q+\delta,\qquad x_{adv}=V(z),\qquad
\|\delta\|_F\leq\rho(\|q\|_F+\epsilon).
\]

The projected optimization minimizes

\[
-\operatorname{NLL}_{lex}(y\mid V(q+\delta);f_s)+
\lambda\left[\frac{TV(q+\delta)}{\operatorname{sg}(TV(q))+\epsilon}
+\alpha\frac{HF(V(q+\delta))}{\operatorname{sg}(HF(x))+\epsilon}\right],
\]

where

\[
TV(z)=\frac{1}{T-1}\sum_{t=2}^T\|z_t-z_{t-1}\|_2^2,
\qquad HF(a)=\frac{\sum_{t,f>f_c}|S(a)_{tf}|^2}{\sum_{t,f}|S(a)_{tf}|^2}.
\]

The implementation caches **original-clean** TV and HF denominators, excludes prompt/language/task/EOT tokens from lexical NLL, differentiates through the frozen vocoder and surrogate to \(\delta\), and applies Euclidean projection onto the global feature ball after each update. It synthesizes the final projected iterate separately. See [loss definitions](/workspace/beyondwaveform/losses.py:8), [projection](/workspace/beyondwaveform/attacks.py:22), [optimization](/workspace/beyondwaveform/attacks.py:50), and [teacher forcing](/workspace/beyondwaveform/models/whisper.py:24).

The paper specifies 50 updates, \(\rho=0.1\), \(\lambda=1\), and \(f_c=6000\) Hz. Its Appendix D identifies a held-out 999-recording **test-clean** subset; this clarifies the main text's broader train-clean-100 wording. Its approximately 10 seconds/clip and four-hour total are paper claims on a 46 GB L20, not measurements from this repository ([paper Appendix D](/workspace/paper/BeyondWaveform.md:350)).

## Why it might work, and which assumptions deserve challenge

A frozen speech generator restricts the search to audio reachable from SSL features and may bias it toward structured speech. Representations can share phonetic structure across recognizers; a vocoder can turn coherent changes into waveform patterns that survive some preprocessing. Those are plausible mechanisms. Transfer and perceptual preservation still require evidence; neither follows from the construction alone.

**Synthesis changes the feasible set; it does not escape waveform space.** After consistent alignment, any generated signal can be expressed as \(x+(x_{adv}-x)\). A feature-vocoder parameterization can change the induced perturbation distribution and invalidate a particular waveform-norm threat model, but it does not logically make waveform preprocessing, purification or training defenses irrelevant. Fair evidence needs matched recordings, explicit waveform/perceptual budgets, clean and reconstruction controls, and actual held-out target measurements.

**The feature ball has no demonstrated human-content guarantee.** Locally,
\(x_{adv}-x\approx[V(q)-x]+J_V(q)\delta\). Reconstruction error, anisotropic feature geometry and the vocoder Jacobian matter; equal relative Frobenius radii need not cause equal acoustic or phonetic changes across recordings, layers, models or generators. A global budget can concentrate on a few phonemes or quiet intervals. The labels “speech-like manifold” and “content preserving” are hypotheses, not mathematical consequences of a frozen vocoder.

**The objective approximates the wrong quantities in several ways.** Teacher-forced lexical NLL is a differentiable proxy for decoded WER, not decoded WER itself; prompt grammar, special-token probability and free-running error propagation can alter the relationship. The loss contains no explicit human transcription constraint. TV penalizes the perturbed trajectory itself, not only harmful deviations from clean speech, and may suppress useful natural dynamics. A single HF ratio cannot represent all timbral, loudness, periodicity or localized artifacts. Small clean TV or HF denominators amplify penalty sensitivity; epsilon prevents division by zero without calibrating perceptual importance. WER also counts insertions and is not bounded above by 100%.

**The encoder–vocoder contract is essential in practice.** Layer dimensions alone do not make a generator compatible with a new representation. The later [layer-5/7 compatibility study](/workspace/reproduction-runs/paper-quality-feature-layers-v2/report.md) completed reconstruction checks and rejected both probes before attacks. That supports requiring reconstruction preflights; it does not establish that all other backbones or vocoders are inferior.

The essential idea is searching a bounded SSL representation through a frozen differentiable generator using only surrogate feedback, with clean-referenced perceptual regularization for the paper's full method. WavLM-Large, layer 6, regular HiFi-GAN, Adam, zero initialization, learning rate, mean/sum NLL, alpha, epsilon, STFT settings, decoder defaults and final-iterate selection are specific implementation choices. Changing encoder or generator can be a major methodological experiment; removing TV/HF is an informative ablation but no longer reproduces the full objective.

## Independent verification of saved baseline evidence

The new CPU audit recomputed **6,570 score rows across ten complete runs**, independently normalized text, calculated Levenshtein edit distances, summed corpus errors, averaged each MOS channel, checked ID/reference/variant coverage, and compared saved summaries to 10⁻¹² MOS tolerance. It rehashed **1,189 unique original recordings and 6,330 saved WAVs** and checked manifest/generated/score linkage. Every checked run contains 50 finite saved trace entries per utterance and logged projected radii within its configured tolerance. Final deltas and trajectory receipts with bound file hashes were also rehashed. The audit found no discrepancies.

This verifies saved evidence and arithmetic. It does **not** rerun WavLM, vocoder synthesis, ASR, MOS scoring, all optimizer updates, or every historical source binding. Saved GPU resynthesis/repeatability receipts remain producer evidence; they are distinguished from this audit's new CPU checks. Historical rho=.5 deltas were not saved, so their actual perturbation radii cannot now be reconstructed from tensors; their radius evidence is the saved optimization trace.

The quality scores are DNSMOS **OVRL**, NISQA **overall MOS**, and **UTMOS22 strong**, aggregated as arithmetic means across utterances. WER uses NFKC, lowercase and punctuation-to-space normalization with total edits divided by total reference words. Quality is scored on exported 16 kHz audio before defenses. See [metric implementation](/workspace/beyondwaveform/evaluation.py:16) and [saved-audio evaluation](/workspace/beyondwaveform/pipeline.py:108). These automatic estimators do not replace blinded human intelligibility or naturalness tests.

| Evidence / setting | Pool | WER % | DNSMOS | NISQA | UTMOS |
|---|---:|---:|---:|---:|---:|
| Paper anchors (claimed; protocol unmatched) | Author subset | 75.4300 | 2.9540 | 3.5260 | 3.5900 |
| historical-selected-999 | 999 | 74.1064 | 2.9638 | 3.8711 | 2.7924 |
| paper-budget-mean-999 | 999 | 64.7415 | 3.1262 | 3.9542 | 3.3582 |
| strong-sum-lr-0.1 | 24 | 74.7368 | 3.1640 | 4.0350 | 3.3186 |
| quality-sum-lr-0.02 | 24 | 45.6842 | 3.2075 | 4.1245 | 3.6761 |
| timestamp-sum-lr-0.05 | 24 | 71.7895 | 3.1273 | 4.0778 | 3.4034 |
| without-tv | 24 | 76.0000 | 3.1530 | 4.0398 | 3.3226 |
| without-hf | 24 | 74.7368 | 3.1351 | 4.0028 | 3.3058 |
| without-perceptual | 24 | 78.7368 | 3.1564 | 4.0863 | 3.3504 |
| time_energy | 24 | 71.7895 | 3.1436 | 4.0871 | 2.9045 |
| channel_std | 24 | 74.1053 | 3.1864 | 4.0641 | 3.3989 |

The two 999-clip rows use identical IDs and 20,310 reference words. Clean/reconstruction controls for the headline baseline are respectively **3.963565%/3.727228% WER** and **4.084511/4.028398 UTMOS**. Thus its high attack WER is not explained by a failed zero-delta reconstruction. The headline attack has 11,294 substitutions, 796 deletions and 2,961 insertions; the paper-budget mean-NLL control has 9,957/728/2,464. The headline's mean MOS values exceed the historical 2.5 selection gate, yet only **747/999** recordings exceed 2.5 on all three scorers; **209/999** have UTMOS ≤2.5. The paper-budget control has **963/999** above all three thresholds. A mean gate hides tails and does not establish imperceptibility.

The 24-clip rows all use the same 475 reference words and repeatedly studied development recordings. Strong and quality settings expose the WER/UTMOS tradeoff. Removing TV, HF or both has mixed effects and no ablation meets all four paper anchors. These small pilot differences cannot support significance or population-wide causal claims. The recent **channel_std** result improves UTMOS by **0.080291** with **0.631579 percentage-point** lower WER than the uniform full-objective control, passing its historical descriptive directional screen. It still misses 3.590 UTMOS and 75.43% WER; it is optimizer preconditioning, and no prospective confirmation or final test was launched. The [coordinate study](/workspace/reproduction-runs/paper-quality-coordinate-research/report.md) is newer than the consolidated thread summary.

Same-model Whisper-small WER is a surrogate result. Actual previously reported, separately evaluated transfer on the historical rho=.5 WAVs is **67.8976/68.7100/62.4274/65.5391%** for Whisper tiny/base/medium/large-v2 and **44.5446/48.2078%** for Wav2Vec2/HuBERT CTC. These transfer tables were inspected but not independently reaggregated in this ten-run audit. See [Whisper transfer](/workspace/reproduction-runs/english-transfer-999/report.md), [CTC transfer](/workspace/reproduction-runs/english-ctc-transfer-999/report.md). Available independently specified LPF/WaveGuard transform results exist, while CE-AT, DT/DE-AT, DMW, TI, sentence-level AudioPure and PVP lack recovered comparable recipes or artifacts; [defense report](/workspace/reproduction-runs/english-defenses-999/report.md) documents the exact limits.

## What is established and what remains unmatched

| Evidence class | Defensible statement |
|---|---|
| Newly checked saved measurements | Ten result sets, exact corpus WER/MOS arithmetic, linked saved WAV hashes and saved trace constraints pass the CPU audit. |
| Previously reported numerical checks | Strict explicit-reflection/math execution repeated complete trajectories bitwise on three checked development clips; general repeatability over every GPU/model/setting is unproven. [Report](/workspace/reproduction-runs/deterministic-execution-validation/report.md) |
| Paper claims | 75.43% English WER, 2.954/3.526/3.590 MOS and human/physical/defense gains belong to the supplied paper and its incompletely identified protocol. |
| Planned work | The methodological alternatives and experiment matrix in the companion planning files have no newly trained checkpoints or measured performance. |
| Unavailable evidence | Author implementation/checkpoints/subset, matched trained defenses, independently replicated human study and OTA evaluation. |

Exact parity is blocked by missing author sample IDs, WavLM layer/normalization and generator training contract, checkpoint revisions, optimizer/LR/initialization, lexical NLL reduction and orthography, alpha/epsilon/STFT pooling, decoder/timestamp behavior and scorer variants/preprocessing/evaluation pool. Mean and sum NLL are materially different: if a transcript has K scored tokens, multiplying \(-NLL_{mean}+\lambda P\) by K yields \(-NLL_{sum}+K\lambda P\). Equal numeric lambda changes the relative weighting, and finite-step Adam trajectories need not match even under constant rescaling. The historical selected configuration further uses **rho=.5/lambda=.1**, unlike the paper's **.1/1**. The paper-budget control fixes those stated numbers but does not recover the missing author choices.

The [strict parity audit](/workspace/docs/STRICT_PARITY_AUDIT.md:13) found no demonstrated quality-channel indexing error. Its general spectral configuration can accept a sample rate inconsistent with the fixed 16 kHz audio pipeline; all checked baseline configurations use 16000, and research preflights reject the mismatch. That engineering defect should be fixed before supporting arbitrary new frontends, but it does not explain the saved baseline gap. Native generation was numerically sensitive; the isolated strict differentiation path validates a limited repeatability claim rather than author parity.

## Data exposure, compute, and practical bottlenecks

The historical test manifest is independently sampled test-clean data: **999 utterances, 40 speakers, 7,402.95 seconds**. Validation contains 100 dev-clean utterances/36 speakers; the reused pilot contains 24 dev-clean utterances/18 speakers. Test999 has zero ID, source-audio-hash or speaker overlap with these dev cohorts. However, the initial 20-utterance pilot shares **two recordings** with validation100, so these legacy development pools are not independent. They must not be described as separate unbiased confirmations.

The frozen prospective cohort contains **48 unscored utterances, 28 speakers, 319.73 seconds**. Its IDs/audio hashes are disjoint from all five historical manifests audited here. All 28 speakers occur somewhere in the historical development union (27 overlap validation100; 13 overlap dev24). It can assess new utterances from previously represented speakers; it cannot establish speaker-disjoint generalization. Scanning extant per-utterance score files found **190** files intersecting audited cohorts, **15** containing test999 IDs, and **zero** containing frozen48 IDs. This is an inventory of saved exposure, not proof about absent external records. Original validation-only selection is recorded in [freeze receipt](/workspace/configs/english_selected.freeze.json); repeated historical test evaluation does not retrospectively invalidate that selection, but its observed outcomes must not tune new candidates or be presented as a fresh holdout. Freeze a genuinely new final pool prospectively for new claims.

At audit time one **TITAN RTX, 24,576 MiB** was present with 6 MiB used, 0% utilization and no GPU process. The host exposes **32 CPUs** and approximately **184 GiB RAM**; `/workspace` has roughly **17 TiB free** on a shared network filesystem. These observations describe availability at one time, not reserved capacity. Complete local assets include WavLM-Large, regular/prematched HiFi-GAN, Whisper tiny/base/small/medium/large-v2, Wav2Vec2-large/HuBERT-large and pinned quality scorers. An old `.partial` WavLM download is distinct from the complete checkpoint. LibriSpeech dev-clean/test-clean and AISHELL train/dev/test directories are present; LibriSpeech train-clean-100 is absent, so an amortized learner needs a future training-data acquisition step. Directory presence alone is not a full AISHELL completeness or licensing audit.

Measured historical attack optimization summed to **13,592.24 seconds**, averaging **13.606 seconds/clip**; the 999-job wall interval was **4.971 hours**, including loading/I/O/evaluation. Complete strict workflow intervals were **26.28 minutes** for three 24-clip settings, **63.07 minutes** for three 24-clip loss ablations and **21.96 minutes** for two 24-clip coordinate studies, with different preflight/verification overhead. Runtime forecasts for other architectures must remain ranges calibrated by future smoke measurements; these are not equivalent workloads or direct L20 benchmarks.

The main computation is 50 repeated forward/backward passes through vocoder plus Whisper per utterance; freezing model weights still requires activation memory for input gradients. Generation and scoring are separate costs, long clips dominate activation memory, repeated MOS evaluation adds CPU/I/O work, and the 24 GB GPU limits concurrent large models and new generator training. Data-reuse risk and the uncertain relation between automatic MOS and human content are at least as consequential as optimizer speed. These observations favor a staged plan that allocates budget across distinct formulations and trains only after reconstruction, objective-validity and split contracts are frozen.

Repository HEAD during audit: `d7f70e8a717db7feba35b69dacabade7de196b25`. Existing modified/untracked files were preserved. The evidence JSON records source/PDF/config hashes, manifest overlap, resource snapshots, run identifiers and exact aggregates. All new audit artifacts are under `/workspace/improvement`.
