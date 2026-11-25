# Primary literature and defense research directions

Reviewed 22 October 2026 (Asia/Taipei). This is a planning document. No implementation, training, or new robustness result is reported here.

## Evidence status and scope

**Fact** means a statement checked against the cited primary paper, proceedings page, or official repository. **Inference** means our interpretation of that evidence. **Hypothesis** means a proposed mechanism that still needs experiments. A method appearing in this document is not established as novel or effective merely because we propose testing it.

The attack of interest is the supplied local BeyondWaveform document, read through the repository transcription at `/workspace/paper/BeyondWaveform.md`. The main research audit subsequently verified its title/authors against public arXiv v1 and checked the local PDF hash; see [original-method analysis](01_ORIGINAL_METHOD.md). The local document is authoritative for this task. Its feature-vocoder threat is `V(E(x) + delta)` with a relative feature-radius constraint. Neither natural-sounding output nor membership in a vocoder's output range proves preservation of human linguistic content. This motivates human-transcription controls and clean/reconstruction/adversarial comparisons. These are inferences from the supplied threat definition, not results from the literature below.

The existing `/workspace/docs/DEFENSE_RESEARCH.md` is useful local context: it already distinguishes transformed-audio WER from WaveGuard detection, and identifies missing sentence-ASR configurations for AudioPure and the paper's unidentified training defenses. Those are repository findings, not new experiments in this task.

## 1. What would not be a defensible novelty claim

| Proposed claim | Verified prior work | Consequence for this project |
|---|---|---|
| Clean/adversarial prediction consistency is new | TRADES, VAT, adversarial logit pairing; SCADA already adapts consistency/VAT to sequential ASR | These belong in strong baselines. Changing the attack space does not by itself establish a new learning principle. |
| Joint waveform and generative-latent adversarial training is new | Dual Manifold Adversarial Training, including a TRADES extension | The contribution would need a speech-specific mechanism and evidence beyond a modality adaptation. |
| Latent invariance or suppressing base-to-finetuned ASR transfer is new | TransferBreaker, publicly submitted 6 October 2026 | Frozen-reference latent alignment and base/target gradient mixing are occupied ideas. |
| An independent articulatory representation comes free with SPARC | SPARC uses frozen WavLM Large with a learned linear articulatory projection | Sharing WavLM with the attacker can preserve exactly the vulnerable directions; independence must be measured. |
| Phoneme-space adversarial training is new | WavAugment Guided Phoneme Adversarial Training (Wapat), 2023 | A phoneme bottleneck or its AT objective requires a narrower, explicitly differentiated contribution. |
| Acoustic constraints or CTC/attention score fusion are new | Joint CTC/attention decoding (2017); LCAR acoustic reranking (August 2026) | The proposed score combination is a baseline; any contribution must go beyond generic acoustic grounding. |
| Codec quantization removes adversarial noise and is therefore robust | September 2026 revision of the neural-codec robustness study | Its stronger quantization-aware attack largely removes an apparent intermediate-depth advantage. Identity-BPDA alone is insufficient evidence. |
| Randomized chunks, enhancement, or voting is new | Temporal Dependency, Chunking Defense, Sequential Randomized Smoothing, WaveGuard | These are baselines or building blocks, not standalone novelty. |
| Generic diffusion purification for speech is new | AudioPure; WavePurifier; joint ASR-denoiser fine-tuning | Need a defensible reason it removes feature-vocoder perturbations that remain speech-like. |

### 1.1 Learning-objective precedents

**Facts.** [TRADES, ICML 2019, Eq. 5 and Algorithm 1](https://proceedings.mlr.press/v97/zhang19p/zhang19p.pdf) combines clean classification loss with a worst-case discrepancy between clean and perturbed predictions. Its classification theory does not directly establish a WER guarantee for an autoregressive recognizer. [VAT, initially submitted 13 April 2017, Eq. 17](https://arxiv.org/pdf/1704.03976) uses a local conditional-distribution divergence with the adversarial direction derived without labels. [Adversarial Logit Pairing, 16 March 2018, Section 4.2](https://arxiv.org/pdf/1803.06373) explicitly matches clean/adversarial logits. A later [primary evaluation, July 2018](https://arxiv.org/pdf/1807.10272) showed that its reported robustness did not survive a stronger evaluation. Prediction matching is a mechanism to test, not a security proof.

**Facts.** [SCADA, Interspeech 2020, Section 3](https://www.isca-archive.org/interspeech_2020/wang20aa_interspeech.pdf) applies KL/Jensen-Shannon consistency and VAT to ASR; its sequential VAT uses teacher-forced gradients. Its reported LibriSpeech generalization improvements are not evidence of resistance to the supplied feature-vocoder attack. [General-purpose Adversarial Training, Interspeech 2023](https://www.isca-archive.org/interspeech_2023/kim23l_interspeech.pdf) jointly learns a nonlinear plausible-adversarial converter and ASR with distribution-matching regularization. Learned adversarial augmentation is therefore also prior art.

**Facts.** [Wapat, arXiv:2307.12498v1, 24 July 2023, Sections 3.2–3.3](https://arxiv.org/html/2307.12498v1) attacks a frozen SpeechLM phoneme-unit tokenizer's representation using a CTC objective. WavAugment views guide perturbation generation through a KL term; its augmentations include pitch, additive noise, band rejection, time masking and reverberation. Its central evaluation concerns cross-domain recognition/generalization. **Inference:** neither phoneme-space AT nor augmentation-guided consistency is new here. A separate supervised log-Mel content recognizer with acoustic-evidence-constrained reconciliation must be distinguished from this existing tokenizer-space training method and tested under direct feature-vocoder attacks.

**Facts.** [Dual Manifold Adversarial Robustness, NeurIPS 2020, Sections 4–5.3](https://arxiv.org/html/2009.02470v1) combines generator-latent and input-space adversarial training and explicitly includes a TRADES variant. It studies images. **Inference:** a speech defense should test both feature-vocoder and waveform threats, but calling that combination alone a new defense would overstate novelty.

### 1.2 A very recent direct ASR overlap: TransferBreaker

**Facts.** [TransferBreaker, arXiv:2610.09109v1, submitted 6 October 2026, Sections 4.1–4.3](https://arxiv.org/html/2610.09109v1) trains on attacks from the public base model; aligns adversarial target latents to frozen clean base latents; and introduces HybridGrad with base/target gradient mixing. Its threat equations use additive waveform perturbations. Its Section 4.3 adaptive adversary trains a defense-aware surrogate to approximate the private target, then transfers attacks. That is a different access model from direct white-box optimization against the final defended target. [Official code](https://github.com/rohban-lab/TransferBreaker) is public but was inspected read-only, not run or revision-pinned in this task.

**Inference:** use TransferBreaker as a required comparator for any training-based proposal. A possible gap is generator-family generalization under content-preserving feature-vocoder threats, with direct adaptive access to the defended pipeline. Merely replacing waveform PGD with WavLM-feature PGD plus latent alignment is likely an incremental adaptation. This review does not establish an unoccupied theorem or algorithmic claim.

## 2. Representation and purification precedents

### 2.1 Articulatory and phonetic bottlenecks

**Facts.** [SPARC, arXiv:2406.12998v4, 14 December 2024, Section III-A](https://arxiv.org/html/2406.12998v4) maps WavLM Large layer-9 features to 12 articulatory coordinates with a linear head. The described inversion uses one speaker's 75-minute EMA dataset, 50 Hz trajectories and temporal low-pass filtering; source features and speaker identity support resynthesis. The [official repository](https://github.com/Berkeley-Speech-Group/Speech-Articulatory-Coding) is available. These are speech-coding results, not adversarial-defense results.

**Inference:** interpretable low-dimensional trajectories offer a testable bottleneck, but off-the-shelf SPARC is not an independent acoustic measurement. It shares a WavLM family and a neural-synthesis design with the threat of interest. A second head on the same frozen representation may inherit correlated failures. Supervised phones, independently trained acoustics-to-articulation, or classical source/filter features are alternatives whose clean accuracy and attack-gradient correlation need comparison. Even a different checkpoint is not proof of independence.

### 2.2 Codec bottlenecks: the version changes the conclusion

**Facts.** The current full text of [arXiv:2603.09034v2, 22 September 2026](https://arxiv.org/html/2603.09034v2) is titled *Attack-Dependent Robustness of Neural Audio Codecs for Adversarial ASR*. The March version's older title and optimistic summary still appear in search results. The revised Sections III–V compare non-adaptive PGD, identity-gradient BPDA+EOT, and a soft RVQ-assignment attack, SoftVQ-PGD. The latter optimizes through codebook-distance geometry with temperature annealing and evaluates the resulting waveform with the hard codec. It substantially changes the depth trend and largely removes the non-adaptive intermediate-depth optimum. Its defined threat is waveform `L_inf`, not feature-vocoder latent perturbation.

**Inference:** generic low-bitrate canonicalization is a valuable baseline, but quantization can create misleading gradients. A proposed discrete-phone or articulatory code defense needs quantization-aware evaluation as well as identity-BPDA. The relaxed forward objective and the true hard-forward pipeline may differ; evaluate actual hard outputs and also try hard-forward/surrogate-backward variants. Treat failure of any one surrogate as an unresolved attack limitation.

### 2.3 Diffusion and enhancement

**Facts.** [AudioPure, ICLR 2023, Section 3](https://arxiv.org/html/2303.01507) adds noise then applies reverse diffusion, and includes adaptive differentiation and EOT evaluation. Its experimental task is speech-command classification, not unrestricted sentence transcription. The [official implementation](https://github.com/cychomatica/AudioPure) is public. [WavePurifier, ACM MobiCom 2024, Section 3](https://cse.msu.edu/~qyan/paper/wavepurifier.pdf) processes chunked spectrograms using hierarchical diffusion, reconstructs with original phase, and denoises. Its Section 5.3 describes out-of-memory direct gradient tracking and appeals to attacker resource limits rather than demonstrating a complete direct adaptive attack against that pipeline.

**Facts.** [Joint Adversarial Fine-tuning with Denoiser, Interspeech 2022, Sections 2–3](https://www.isca-archive.org/interspeech_2022/joshi22_interspeech.pdf) evaluates adaptive white-box ASR-denoiser attacks and trains a tandem model, including a frozen-ASR variant. **Inference:** training an enhancer on offline examples is not enough; an adaptive attacker must attack the enhancer and recognizer together. Natural-speech priors do not automatically select the correct transcript when adversarial speech lies in the generator's range. Blind resynthesis, diffusion, and content-conditional repair require separate ablations.

**Facts.** [Zelasko et al., Adversarial Attacks and Defenses for Speech Recognition Systems, arXiv:2103.17122v1, 31 March 2021, Section IV](https://arxiv.org/pdf/2103.17122) already evaluates a ParallelWaveGAN defense reconstructing speech from log-Mel spectrograms, including an attacker that backpropagates through both vocoder and ASR. **Inference:** a different vocoder or a compressed log-Mel representation alone is established prior art, and cannot substantiate the proposed content-bottleneck contribution.

### 2.4 Temporal consistency, smoothing and voting

**Facts.** [Temporal Dependency, ICLR 2019](https://openreview.net/pdf?id=r1g4E3C9t7) studies partial/full-transcript inconsistency for detection. [Chunking Defense, Interspeech 2022, Section 3](https://www.isca-archive.org/interspeech_2022/shao22c_interspeech.pdf) independently recognizes fixed or random chunks; its main concern is targeted attacks. [Sequential Randomized Smoothing, EMNLP 2021, Sections 3–4](https://aclanthology.org/2021.emnlp-main.514.pdf) combines Gaussian-noise robustness, enhancement and ROVER word voting, with straight-through preprocessing gradients and 16-sample EOT in its adaptive attacks. [WaveGuard, USENIX Security 2021](https://www.usenix.org/conference/usenixsecurity21/presentation/hussain) compares original/transformed transcriptions for detection. Transformed-audio WER alone is not its full detector.

**Facts.** [PVP, arXiv:2603.22590v2, 20 July 2026, Sections III-C–III-F](https://arxiv.org/html/2603.22590) randomizes exposed numerical precision and uses transcription diversity for detection; its adaptive surrogate optimizes across precisions, and its hybrid includes DistriBlock uncertainty. **Inference:** multiple views are useful only if their errors are sufficiently different. If all branches confidently agree on the same wrong words, voting and disagreement detectors can both fail. A defense must report corrected WER and rejection/coverage separately.

### 2.5 Acoustic grounding and joint decoding are also prior art

**Facts.** [Joint CTC/attention decoding, ACL 2017, Sections 3.1–3.2](https://aclanthology.org/P17-1048.pdf) combines CTC and attention probabilities in training and beam search, with a shared acoustic encoder. It explicitly discusses deletion, repetition and irregular alignment failures. **Inference:** multiplying/interpolating an auxiliary acoustic score with an autoregressive score is established decoding practice. The proposed separate supervised phone path must be compared against ordinary joint/late-fusion decoding; the score equation by itself is not a novel defense.

**Facts.** [Likelihood-Constrained Acoustic Reranking (LCAR), arXiv:2608.30776v1, 31 August 2026, Section 3](https://arxiv.org/html/2608.30776v1) admits tokens near the base model's greedy likelihood and reranks them using attention-pooled audio embeddings projected through its existing LM head. It requires no independent auxiliary recognizer at inference. Its challenge suites concern acoustic-grounding hallucinations, not the supplied FV feasible set or a demonstrated joint white-box attack on a separate phone path. **Inference:** “acoustically constrained correction” is not sufficient novelty. The unresolved contribution is whether separately supervised evidence and candidate expansion retain recoverable information under jointly optimized FV attacks beyond these grounding/fusion baselines. Directly porting LCAR to Whisper may require an architecture-specific adaptation and must be labeled accordingly.

This was a bounded primary-source search, not an exhaustive novelty clearance. In particular, no absence-of-prior-work claim is established for independent phone-lattice reconciliation.

## 3. Disambiguating SpeechGuard

Three different works use the name; citing the name without a title is unsafe.

1. **Fact:** [SpeechGuard: Exploring the Adversarial Robustness of Multimodal Large Language Models, arXiv:2405.08317, May 2024](https://arxiv.org/html/2405.08317) studies speech-language-model jailbreaks. Its Sections 5.4/A.6 evaluate time-domain noise flooding and an adaptive attacker. This is relevant prior art for random-noise countermeasures, not a demonstrated sentence-ASR feature-vocoder defense.
2. **Fact:** [SpeechGuard: Online Defense against Backdoor Attacks on Speech Recognition Models, arXiv:2607.15697v1, uploaded 17 July 2026](https://arxiv.org/html/2607.15697) explicitly identifies itself as the accepted **IJCNN 2024** manuscript, DOI `10.1109/IJCNN60899.2024.10650300`. It uses S-STRIP detection and autoencoder-generated time-frequency masks for poisoned command-recognition inputs. The upload date does not make it new 2026 research; training-time poisoning differs from this task's test-time evasion threat.
3. **Fact:** [SpeechGuard: Recoverable and Customizable Speech Privacy Protection, USENIX Security 2025](https://www.usenix.org/system/files/usenixsecurity25-zhang-jingmiao.pdf) protects acoustic/content privacy using recoverable transformations and keys. It is not an ASR robustness defense.

## 4. Three substantially different research branches

These are **hypotheses and proposed experiments**. Ranking reflects current research judgment, not measured performance. All require a preregistered clean-accuracy and compute budget. No branch is implemented during this research-only phase.

### A. Train for generator-family robust transcription

**Mechanism hypothesis.** Modify the recognizer, not deployment audio. Train on the worst valid example across several attack families: the supplied WavLM/vocoder threat, additive waveform attacks, and at least one different generation/representation family. Pair the ordinary transcript loss with sequence-level invariance to clean speech. A clean branch anchors content while adversarial branches expose decision-boundary errors. Include a reconstruction-only branch for every generator so the recognizer cannot learn merely to identify vocoded audio.

For proposal purposes, let `A_j(x)` be a generator-specific feasible set and let `D_seq` compare sequence predictions. A candidate objective is

`L_clean(x,y) + beta * max_j max_{x' in A_j(x)} [L_text(x',y) + gamma * D_seq(p_theta(.|x), p_theta(.|x'))]`.

This is a **research sketch**, not a new theorem. With teacher-forced token KL it resembles existing consistency methods. A substantive contribution would require a better-defined sequence risk: e.g., aligned posterior/lattice disagreement or a differentiable expected-edit surrogate that addresses deletions/insertions and exposure to wrong prefixes. CTC and autoregressive models have different alignment and decoding issues; use the same recognizer family for baseline comparisons before claiming architecture benefits.

**Distinctive question.** Can explicitly learning invariance across generator families retain lexical evidence under an unseen encoder/vocoder, at matched clean WER and training cost? A useful claim would be about cross-family generalization with mechanism evidence, not “latent adversarial training works.”

**Key failure modes.** Overfitting the known WavLM/vocoder; teacher-forced consistency missing free-running errors; clean-teacher mistakes becoming targets; a max over incomparable unconstrained attacks selecting linguistic content changes; short inner optimization creating false robustness; forgetting rare phonemes/accented speech; large training expense; attack-family sampling hiding worst-case examples.

**Decisive experiments.** Compare clean fine-tuning, waveform AT, supplied-FV AT, naïve waveform+FV AT, TRADES/VAT-style sequence consistency, and a faithful TransferBreaker adaptation at equal update/inner-gradient budgets. Hold out an entire encoder/vocoder family and speaker split from training and selection. Compare the proposed sequence objective to teacher-forced KL and CE-only AT. Reattack the final model directly, including all defenses, using several inner losses and attack budgets. Measure paired changes on clean, reconstruction and adversarial speech. Kill the stronger claim if improvements disappear under held-out generators or direct adaptive attacks. Treat a useful known-family gain as a baseline result rather than overstating novelty.

**Research priority:** necessary strong baseline and possibly a publishable branch if the sequence mechanism and unseen-family generalization add something beyond the cited precedents.

### B. Decode through independently supervised speech content

**Mechanism hypothesis.** Change the representation and inference architecture. Train a small speech-content path using phoneme supervision or articulatory targets from an acoustic frontend that does not share the attacker's frozen SSL encoder. Combine its temporal posterior/lattice with the main recognizer's candidate hypotheses. Score candidate transcripts by independent acoustic support, phone timing and optional articulatory compatibility. A later canonicalization variant could resynthesize from these content variables in a fixed speaker/prosody template; the first scientific comparison should decode the bottleneck directly so resynthesis is not an unexplained extra cause.

The defense must not need the clean waveform or its transcript at deployment. Clean counterparts may supervise training, but inference sees only the input under test. The content path must also preserve uncertainty: a single hard predicted phone sequence can irreversibly encode an adversarial error. Constrain reconciliation acoustically rather than asking a language model to “correct” the transcript from plausibility alone. Forced alignment of an already-wrong transcript is not independent evidence.

The most concrete initial version is a separately trained **log-Mel supervised phone recognizer plus an acoustic-evidence-constrained lattice decoder**. Separate weights, data lineage or a different feature frontend do not guarantee uncorrelated errors or adversarial robustness. “Independent” here describes architectural/training provenance only; establish useful error diversity empirically and attack both models jointly.

**Distinctive question.** Does content supervision remove attack-relevant variation while keeping the linguistic distinctions that humans preserve, and does that improve corrected WER under attacks jointly targeting both paths? This is different from generic codec compression, a second recognizer vote, or blind speech enhancement. Its potential contribution is an independently trained content bottleneck plus a verifiable acoustic reconciliation rule, if ablations support that mechanism.

**Key failure modes.** Shared training/architectural biases despite different encoders; losing fricatives, stop bursts, coarticulation or Mandarin tone; phoneme inventory/lexicon mismatch; a language model overriding rare names; quantization masking gradients; attacks changing articulatory/phone predictions jointly; synthesis hallucinations; poor multilingual coverage. Off-the-shelf SPARC is an exploratory comparison, not the independent branch.

**Decisive experiments.** Compare (i) main recognizer alone, (ii) equal-compute independently trained ASR ensemble, (iii) continuous generic bottleneck, (iv) matched-size supervised phone/articulatory bottleneck, (v) codec-depth and SPARC controls, (vi) proposed acoustic reconciliation. Include shared-WavLM versus separate-encoder variants and randomized/shuffled bottleneck-target controls to test whether content supervision matters. Report phone error, phone/tone class breakdowns, main/auxiliary error correlation, clean WER and adversarial WER. If resynthesis is added, separate bottleneck decoding from canonical-audio decoding. Attack the joint scoring and all differentiable content paths, then use soft quantizer surrogates, hard-forward alternatives and query/transfer checks for discrete steps. Kill the independence claim if gains are explained by extra model capacity, encoder diversity alone or loss of difficult content.

**Research priority:** most promising conceptual branch; higher implementation/data uncertainty and no present evidence of superiority.

### C. Recover words from independently contextualized local evidence

**Mechanism hypothesis.** Keep recognizer weights fixed. Recognize overlapping speech windows with varied left/right contexts and boundary offsets. Align outputs to time/word spans, retain words supported across local contexts, and use a full-utterance branch only as another source of evidence. A deterministic fixed-window version should precede randomness. Where uncertainty is high, expand context or use a second representation; a later system could choose the number of views adaptively using thresholds frozen on validation data. This is a transcript-recovery procedure, not merely a detector.

**Distinctive question.** Are the supplied attack's errors dependent on accumulated context, or do they already exist in short acoustic units? Its total-variation regularizer encourages temporally smooth perturbations, so there is no reason to assume random cuts erase them. If local phonetic evidence is corrupted consistently, this entire branch may fail.

**Key failure modes.** Breaking words at boundaries; insufficient context for homophones, numbers and proper names; correlated wrong local transcripts; unreliable timestamps; majority voting preferring a confidently repeated mistake; randomness increasing cost without reducing risk; apparent success caused by truncated audio or fewer decoded words. Existing chunking, temporal-dependency and ROVER/smoothing papers substantially limit novelty.

**Decisive experiments.** First measure error survival under full speech, aligned short windows and varied context on the same attacked utterances. Compare the supplied FV attacks to additive attacks. Then compare fixed chunking, overlapping deterministic fusion, randomized offset fusion, ROVER smoothing and the proposed local acoustic reconciliation at equal recognizer-call budgets. Report WER including all deletions, chunk-boundary errors, latency and number of calls. Joint adaptive optimization must sample chunk offsets/context configurations, target all branches, and optimize a differentiable surrogate for the actual reconciliation. Evaluate the real alignment/voting operation on hard decoded text. Kill the mechanism if attack errors remain consistent locally or if equal-compute chunking/ROVER explains the gain.

**Research priority:** cheap diagnostic and baseline; a lower-priority novelty claim unless local-evidence recovery yields a new, well-supported mechanism.

### A held-out generative stress case for all three branches

**Facts.** [GATAS, Generative Testing of Automated Speech Recognition Systems, arXiv:2607.09833v1, 10 July 2026, Section 3 and Algorithm 1](https://arxiv.org/html/2607.09833) searches StyleTTS2 phoneme-level acoustic text embeddings using per-phoneme interpolation weights and a multi-objective optimizer that queries final ASR text. This differs from perturbing clean-waveform WavLM features and transferring from a white-box surrogate. **Inference:** it is a useful held-out generative stress case, particularly against the assumption that phonetic representations are inherently safe. The paper explicitly manipulates phonetic content; human-preserved-transcript validity must therefore be measured, rather than accepting naturalness as proof. Its headline success rate should not be compared directly to this project's WER without harmonizing data, budgets, definitions and validity controls.

## 5. Adaptive evaluation requirements derived from the review

**Facts.** [Obfuscated Gradients, ICML 2018, Sections 4.1–4.3](https://proceedings.mlr.press/v80/athalye18a/athalye18a.pdf) introduces BPDA for unhelpful derivatives and EOT for randomized transformations, and discusses reparameterizing attacks through a generative manifold. [On Adaptive Attacks, NeurIPS 2020](https://arxiv.org/pdf/2002.08347) demonstrates that an evaluation described as adaptive can still be incomplete and emphasizes defense-specific attack design. [Randomized Smoothing, ICML 2019](https://proceedings.mlr.press/v97/cohen19c/cohen19c.pdf) certifies a defined smoothed classifier in an additive `L2` neighborhood. Such a certificate does not automatically cover a feature-radius threat, arbitrary reconstructions or a heuristic word-voting system.

The following are **proposed requirements for this project**:

1. Separate non-adaptive transfer from a known surrogate, defense-aware surrogate transfer, and direct white-box access to the complete defended target. Passing the first does not establish the last.
2. Attack in the original feature-vocoder space while differentiating through the new defense. Add waveform attacks and at least one held-out generative family. For each family, freeze admissibility rules, reference transcripts and evaluation IDs before comparison.
3. Increase steps, restarts, EOT samples and loss variants until the worst observed error stabilizes or a declared resource ceiling is reached. Show curves and ceiling-limited uncertainty. An attacker out-of-memory event is not a robustness result.
4. Try true gradients, appropriate BPDA/straight-through surrogates, codebook-distance relaxations where applicable, alternative surrogate losses, and a bounded query/transfer check. For a discrete canonicalizer, evaluate the true hard pipeline after every selected attack.
5. Include recognizer loss, content-path loss, branch-consistency objectives and detector/abstention objectives where the defense uses them. Wrong but consistent predictions must count as failures.
6. Record both WER relative to the original transcript and blinded human transcription of attacked speech on a predetermined subset. Changed human linguistic content is a different validity category, not an automatic model-defense failure. Report the whole set as well as valid-content subsets; never select attractive attacks after seeing results.
7. Freeze speaker/utterance-disjoint training, tuning and final-test splits. The historical 999-clip pool can support a paired diagnostic, but prior inspection makes it unsuitable as an untouched selection-free final claim. Report corpus WER, paired speaker-aware uncertainty and clean/reconstruction/attack controls.
8. Keep recovery metrics separate from detection/abstention: corrected WER, clean WER, coverage/rejection, and failure among accepted inputs. A detector that rejects speech has not restored its transcript.
9. Match compute, data and added model capacity. Report throughput/latency, training inner-gradient counts, parameter memory and per-utterance attack cost. Keep MOS/intelligibility separate from ASR correctness.

## 6. Immediate planning conclusions

The strongest first research question is whether the attack shifts **human-preserved content evidence**, **recognizer-specific context**, or **generator-specific reconstruction cues**. The three branches target these different possibilities. Before training, design a small frozen diagnostic that can distinguish them using reconstruction controls, short-context error localization, independent phone evidence and encoder/vocoder holdouts.

A rigorous negative result is useful: if canonicalization or voting collapses under adaptive attacks, preserve it and move to the training/content branch. Do not spend a full training budget making a weaker implementation reproduce a headline. Nothing in the sources reviewed supports a present claim that any proposed branch will outperform the supplied attack or constitute a publishable contribution.

Further source verification remains necessary before selecting final baselines: pin executable repository revisions and model licenses; recover exact pretrained checkpoints and sentence-length assumptions; audit overlap with additional 2026 work; and obtain an accessible full text for the earlier DiffVC-based attack. [The primary proceedings entry, PMLR 260, 2025](https://proceedings.mlr.press/v260/wang25f.html) verifies a voice-conversion latent attack exists, but its full PDF was not successfully retrieved in this review, so detailed implementation claims are intentionally withheld.

For local data availability, use the completed [repository audit](02_REPOSITORY_AUDIT.md) and `/workspace/defense/evidence/local_audit.json`, not stale absence statements in earlier notes. They verify existing 1-hour/10-hour manifests under `/workspace/improvement/execution/data/` and a reserve200 pool associated with another active experiment. This literature review proposes no reuse of that reserve.
