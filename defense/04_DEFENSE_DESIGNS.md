# Defense hypotheses and decisions

22 October 2026 (Asia/Taipei). These are independent research proposals, not implemented methods or claimed results. The [literature review](03_LITERATURE_REVIEW.md) constrains novelty. Research priority is **independent phonetic evidence**, with **joint waveform/FV adversarial training** as the strong learned comparator and **local-context recovery** as a cheap diagnostic. Combining all three immediately would obscure why anything works.

## Three different interventions

| Branch | What changes | Proposed benefit | Main uncertainty | Current role |
|---|---|---|---|---|
| Independent phonetic evidence | Acoustic representation and decoding architecture | Recover words from supervised phone evidence when the main recognizer fails | Whether that evidence survives joint attacks and retains clean accuracy | Lead research hypothesis |
| Joint waveform/FV adversarial training | Learning problem, training data and objective | Make generated speech variation compatible with correct recognition | Family overfitting, cost, teacher-forcing mismatch; substantial prior art | Required strong baseline; possible separate contribution |
| Local-context recovery | Inference representation of time/context | Recover errors caused by long-context decoding instability | Attacked phones may remain wrong in every window | Bounded falsification experiment and baseline |

Plain filtering, same-vocoder resynthesis, generic codec compression, and generic diffusion are comparison conditions where feasible. They do not presently supply the contribution. Articulatory resynthesis is a possible later representation study; off-the-shelf SPARC shares WavLM, so it cannot establish an independent acoustic view.

## Lead hypothesis: supervised phonetic evidence constrains decoding

**Question.** Does an attack that preserves human words destroy the main recognizer's decisions more than a separately supervised phone lattice, leaving recoverable evidence? If yes, can acoustic reconciliation recover those words under a joint adaptive adversary?

### Proposed inference architecture

1. The incoming waveform `u` feeds the ordinary ASR branch and a separately parameterized log-Mel acoustic phone branch `Pψ`. The phone branch uses supervised phone-sequence CTC or a comparable alignment model. It must not reuse the attack's frozen WavLM feature extractor. A second head attached to WavLM is an explicit shared-representation control.
2. `Pψ(u)` retains soft phone/blank posteriors and timing uncertainty. A lexicon graph maps plausible phone paths to word candidates. A lexicon-free subword auxiliary decoder is a control for out-of-vocabulary names and pronunciation errors.
3. Form a candidate set from **both** the main ASR beam and the phone-derived lattice. A verifier limited to the main model's incorrect candidates cannot recover a missing correct word. Preserve an explicit fallback for missing/low-confidence phone paths.
4. Score each candidate by main-ASR sequence probability and independently computed phone-sequence probability, with weights, normalization and any pronunciation penalties frozen on development data. The phone score sums permitted CTC alignments/pronunciations; it does not force-align a wrong transcript and treat the resulting timestamps as validation.

An initial transparent scoring rule is

`score(c,u) = s_ASR(c,u) + η log Pψ(phones(c) | u) + κ s_lex(c)`.

Specify length normalization and lexical priors explicitly. `s_lex` can use a fixed lexicon/language prior; it is not an unconstrained LLM rewrite. At inference no term may depend on `x`, `y`, `E(x)` or the attack's actual `δ`. During evaluation, references may be used to measure lattice-oracle ceilings, never to choose the deployed transcript.

This is initially a **transcription defense**, with no waveform synthesis. The phonetic posterior is the constrained content representation. Keeping soft alternatives avoids treating a wrong hard phone decision as irreversible truth. Optional discretization and articulatory features should be separate later ablations, not assumed sources of robustness.

### Proposed learning setup

Phone supervision can come from training transcripts through a pinned pronunciation lexicon/G2P, accounting for alternate pronunciations. These are word-derived phone targets, not measured articulator labels. Frame alignments are latent under CTC; any explicit aligner is an extra model whose provenance must be reported. The clean waveform and transcript are available during training only.

Start from a verified separately pretrained acoustic recognizer if its weights, data lineage and license permit. Compare against an equal-resource clean auxiliary ASR ensemble. If no suitable acoustic checkpoint exists, training a small Conformer/log-Mel phone recognizer on audited train-clean-100 is a different, more expensive track. The existing 1 h subset is appropriate for adapter feasibility, **not** a credible promise to train a high-quality independent phone recognizer from scratch. This is the first unresolved implementation prerequisite.

First train/validate the acoustic branch on clean/reconstruction speech. Only after its diagnostic survives should we train it on paired content-valid FV/waveform examples. Keep “architecture alone,” “adversarially trained auxiliary,” and “jointly trained full system” as separate conditions. A gain from robust training alone must not be attributed to the bottleneck.

### Mechanism tests that can falsify the hypothesis

Measure all of the following on the same predeclared development clips:

- Phone error and lexical-decoder WER on clean, reconstruction and attack audio; analyze fricatives, stops, low-energy words, names and silence separately.
- Whether the correct word sequence remains in the phone lattice when the main branch is wrong. Oracle WER is a diagnostic upper bound, not the defense's reported accuracy.
- Per-utterance error correlation, rescued words versus newly damaged words, and calibration on wrong but confident joint predictions.
- Main-only, auxiliary-only, ordinary ensemble/ROVER, unconstrained score fusion, and the proposed phone-constrained candidate expansion at matched inference calls.
- Shared-WavLM versus separate log-Mel path, generic continuous bottleneck versus supervised phone bottleneck, and a matched phoneme-free auxiliary. Randomized-target sanity controls may expose leakage but are not a substitute for a useful matched baseline.

**Proposed early gate:** do not build a complicated decoder unless phone-lattice candidate coverage demonstrates at least a 10 WER-point oracle improvement over the main attacked decoder on a fixed development screen. This is an exploratory feasibility target, not a promised effect or a final superiority test. The eventual realized defense must still meet the clean-accuracy and adaptive-attack gates in the protocol.

**Failure means:** if phones and main ASR fail together, a content bottleneck cannot recreate missing acoustic evidence. If gains equal an ordinary extra-model ensemble, the content-specific mechanism is unsupported. If fusion loses rare words or homophones on clean speech, tune only on development or stop. Different architecture does not imply independent errors.

### Novelty boundary

Wapat already performs phoneme-space adversarial training; ordinary CTC/ASR fusion, phone lattices, bottleneck features and consistency losses also predate this task. The bounded decoder check additionally found [joint CTC/attention decoding](https://aclanthology.org/P17-1048.pdf) and [LCAR acoustic reranking](https://arxiv.org/html/2608.30776v1); the score equation above is therefore a transparent baseline, not the new contribution. SPARC supplies a WavLM-derived articulatory codec, not a new independent sensor. The possible contribution is **a particular acoustic-evidence-constrained recovery mechanism that retains content under feature-vocoder attacks and transfers to held-out generator families under joint adaptive evaluation**. Its novelty remains unestablished; the mechanism experiment and deeper comparison to grounding/fusion methods must earn the narrower claim.

No theorem follows merely from lower dimension. A conditional local statement is possible only after defining a phone-margin and bounding its change through `Pψ∘V`; estimating a Jacobian norm alone is not a certified bound. A globally normalized feature radius does not produce a phone guarantee without additional assumptions. Empirical robustness is the intended initial claim.

## Learned comparator: training over a union of speech threats

Train the recognizer itself, initially using encoder adapters/LoRA or another memory-measured fine-tuning path with an otherwise matched base model. Compare parameter-efficient and full updates only if justified by clean feasibility; no rank/optimizer recipe is selected yet.

For training attack families `A_j`, a transparent starting objective is

`E[ Ltext(Dθ(x),y) + a Ltext(Dθ(V(E(x))),y) + b max_j max_(u∈A_j(x)) Ltext(Dθ(u),y) + c Dseq(Dθ(x),Dθ(u*)) ]`.

`u*` is the selected inner adversary. First use `c=0` as a CE-only baseline. A teacher-forced lexical-token KL is a concrete comparison, not evidence of free-running sequence invariance. A stronger sequence proposal needs an explicit differentiable expected-edit or posterior/lattice loss and a check that it improves insertions/deletions under real decoding. Do not call a generic KL term a new learning paradigm.

Start with waveform and supplied-FV families, plus reconstruction-only training controls. Add a second matched generator family only after its clean reconstruction passes. Incomparable feature radii are not forced to equal numerical values. Each family has a declared feasible set and content-quality validation. Select worst training examples by normalized declared objective, and show CE-only family results separately so family sampling cannot hide failure.

Inner attacks use a frozen parameter snapshot/detached copy of the current defender; outer updates train the live model. The existing attack routine freezes modules, so direct reuse with the live trainable model is unsafe. Model parameters, optimizer state and gradient graph separation need a small preflight in the implementation phase.

Training data strategies to compare are reconstruction augmentation; offline FV pairs; online current-defender adversaries; and a bounded replay mixture. Offline attacks alone can overfit stale boundaries. Online attacks cost much more. A 1 h versus 10 h nested learning curve separates data quantity at matched architecture, while a second speaker allocation tests diversity. Additional clean-only updates and natural augmentation must receive the same training budget.

Required comparisons are clean fine-tuning, waveform-only AT, FV-only AT, naive union AT, union plus sequence consistency, and an appropriately scoped TransferBreaker adaptation. At least one generator/representation family must be absent from training and selection. Generic union AT is a strong baseline, not automatically a publishable method: TRADES/VAT, dual-manifold training and TransferBreaker are direct novelty limits.

## Inference alternative: local-context transcript recovery

Before training, test whether wrong words depend on long autoregressive context. Use deterministic overlapping windows (initial planning values: 4/8 s windows and half-window stride, with whole-utterance inference retained), then align hypotheses by time and word sequence. Do not trim unrecognized spans or lose boundary words. A fixed treatment of the last partial window, silence and short utterances is necessary.

Compare equal-call deterministic chunking, overlapping fusion and randomized offsets against ordinary whole-utterance decoding and available smoothing/ROVER baselines. Word/time support should determine fusion, not the known reference. Randomness comes after a deterministic version; its seeds are not a security secret.

If attacked phones are consistently wrong in every local context, stop this branch. If ordinary chunking explains the gain, record it as a useful baseline rather than inventing novelty. Adaptive evaluation samples the actual offsets, targets all windows/full branch and attacks the fusion surrogate; hard decoded fusion is always scored afterward.

## What would earn continuation

The lead should proceed only if independent acoustic evidence produces recoverable candidates at acceptable clean cost and realized fusion beats an ordinary equal-resource ensemble. The learned branch should proceed only if gains survive direct current-model attacks and a held-out generator. Local-context work should proceed only if a measurable context-dependent mechanism exists.

A publishable positive result needs meaningful adaptive-WER improvement, matched clean accuracy/resources, generator/speaker generalization and a mechanism ablation. A negative result exposing where bottlenecks or consistency fail is also scientifically useful if the evaluation is rigorous. No branch has passed any of these tests yet.
