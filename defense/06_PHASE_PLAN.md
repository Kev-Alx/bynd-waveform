# Gated execution plan and continuation

22 October 2026 (Asia/Taipei). The user's requested current phase is understanding, planning and research. That phase has produced this package. **Defense implementation, training and evaluation remain future work.** The gates below are scientific decision points; they are not extra permission requirements invented by this document.

## Phase 0 — completed research preparation

- Analyze the original feasible set, objective, access assumptions, architecture and limitations.
- Inspect implementation, historical and paper-budget controls, actual defense results and data exposure.
- Check primary literature through 22 October, including recent TransferBreaker and the revised codec-robustness study.
- Develop three distinct defense hypotheses and a protocol capable of falsifying them.
- Create/update handoff and source/evidence records under the approved artifact root.

No new defense has been implemented or tested. The audit script is research bookkeeping, not the defense method. Mathematical interpretation, repository observations and speculative mechanisms are distinguished in the documents.

## Phase 1 — resolve the largest uncertainties with a small diagnostic

When the implementation phase begins, first inspect the other running attack queue and avoid disrupting it. Capture baseline source/weights/config hashes and a new task-local environment manifest. Reuse compatible existing assets read-only; route all new paths and caches here.

Before GPU work, inspect candidate independent phone recognizer weights/data/license and deepen comparison to known phone-lattice/ASR acoustic reconciliation. The bounded research check already found joint CTC/attention decoding and LCAR; simple score fusion is a baseline, not a new algorithm. Exact differentiation and checkpoint viability remain gaps. Do not train a poor recognizer from 1 h and interpret its failure as proof the idea is impossible.

Freeze a 48–64 clip **development** diagnostic from exposed pools, balanced by speaker, duration and transcript length using metadata, not defense success. Register original FV paper-budget attack, reconstruction and a specified waveform control. Study (a) independent phone evidence and candidate coverage; (b) full versus short-context errors; (c) substitutions/deletions/insertions/no-speech failures; (d) generator reconstruction cues. Use existing saved audio if matched and suitable, disclosing its weaker/stale-attack scope.

Measure memory and time on a few clips before sizing the screen. Proposed diagnostic envelope: at most 2 dedicated GPU-hours after the preflight; this is a planning cap, not a measured estimate or an authorized running job. If the cap is insufficient, reduce the fixed screen prospectively and preserve the coverage limitation rather than selecting easy outcomes.

**Deliverables later:** frozen diagnostic plan, per-clip evidence, phone/local-context analysis, runtime receipt and a go/no-go decision. Continue the phone branch only if it retains a useful candidate ceiling. Stop local-context work if errors survive all contexts.

## Phase 2 — establish honest learned and inference baselines

On a dedicated GPU allocation, implement separate new modules under this directory, preserving the reproduction code. The research harness should separate `generate_inner_adversary`, `train_outer_step`, `decode_defense`, `validate_attack` and `score_saved_audio`. A named output directory and immutable config distinguish every condition.

Start with clean fine-tuning, natural/reconstruction augmentation, waveform AT, FV AT and naive union AT using the same base model/trainable parameter budget. Independently reproduce the selected TransferBreaker comparison to the degree source/checkpoints permit; otherwise mark it unavailable and narrow the comparison claim. A simple ROVER/ensemble and deterministic chunking comparison must accompany the phonetic architecture.

Use the available nested 1 h/10 h subsets only after role and overlap checks. Start adaptation with the smaller cohort; scale the same frozen design to 10 h to test data sensitivity. Architecture-from-scratch training is a separate 100 h track, not mixed into a same-data comparison. Compare offline paired training against online current-defender attacks at matched total cost.

Keep training attack iterations modest initially (e.g., 5 then 10) and evaluate with substantially stronger attacks (50/200 plus restarts), recording all inner backward calls. Avoid claims based only on the training adversary. Pilot the maximal-gradient memory path before enabling random ensembles.

Proposed baseline/first-candidate envelope: 8 additional dedicated GPU-hours, sequential conditions, with costs re-estimated from measured seconds/update. This cap may support only feasibility; it is not a promise to complete a publishable training sweep on a TITAN RTX. Stop and document an inadequate budget rather than silently reducing the adversary.

## Phase 3 — implement one justified hypothesis

Prefer the independent phone branch and acoustic candidate reconciliation only if Phase 1 passes. First implement soft posterior/lattice scoring and candidate expansion, then the matched ablations. Keep resynthesis/discrete articulation out until direct bottleneck decoding explains a benefit. Test clean accuracy, fallback/OOV behavior, duration alignment and differentiable surrogate correctness.

If the phone branch fails feasibility but union AT succeeds, narrow the research question to a specifically justified sequence objective and unseen-generator generalization. Teacher-forced KL alone is a baseline. If neither hypothesis has mechanism evidence, preserve the negative result and revise the hypothesis before another large run.

Do not combine every technique to rescue a failed screen. A later architecture-plus-AT combination requires the 2×2 control: base architecture versus phone architecture, each with clean training versus the same robust training. This separates capacity, supervision and training effects.

## Phase 4 — direct adaptive audit and conditional confirmation

Attack the complete promoted pipeline with original-transfer, defense-aware transfer and direct white-box conditions separately. Include all branches and real hard decoding; use multiple losses/restarts and EOT or quantization-aware relaxations as relevant. Keep worst valid attempts and show attack-strength curves. A larger attack that breaks the method is the result to report.

Freeze one candidate and its selection receipt before using a newly allocated confirmation split. Run at least three training seeds for a learned promoted method. Use the protocol's paired clean-regret/robust-WER gates and resource comparisons. A failure does not trigger automatic tuning on confirmation. Optional allocation for this stage should be estimated from Phase 2 timings; no fixed full-study runtime can be supported yet.

## Phase 5 — final test and contribution assessment

After confirmation and method/attack freezing, open the new reserve once. Report per-family worst-case error, clean/reconstruction controls, human-content validity, paired speaker uncertainty, training/inference cost and mechanism ablations. Preserve historical999 as a regression comparison. Add external-corpus, Mandarin or physical-playback studies only with their own frozen protocols and explicit remaining limits.

A publishable claim is conditional on results. Possible positive claim: independently supervised phonetic evidence improves recovery under generated speech attacks beyond ordinary ensembles/AT, and survives joint adaptive attacks across generator families at small clean cost. Possible negative claim: apparent canonicalization or bottleneck robustness disappears under a stronger adaptive optimizer. Neither conclusion has been established by the present work.

## Artifact layout for later phases

```text
/workspace/defense/
  HANDOFF.md
  00_RESEARCH_BRIEF.md
  01_ORIGINAL_METHOD.md
  02_REPOSITORY_AUDIT.md
  03_LITERATURE_REVIEW.md
  04_DEFENSE_DESIGNS.md
  05_EXPERIMENTAL_PROTOCOL.md
  06_PHASE_PLAN.md
  literature_sources.json
  evidence/
  tools/                         # research inventory/validation only so far
  [later] src/, configs/, data/, weights/, cache/, tmp/, runs/, reports/
```

Future scripts should set task-specific cache/temp roots, `PYTHONDONTWRITEBYTECODE=1`, task-local logging paths and `BEYONDWAVEFORM_NOTIFY_DISCORD=0` before importing model libraries. Pin and inspect dependencies instead of mutating `/opt/venv` or existing baseline caches. Never put new outputs under the historical `reproduction-runs` or `improvement` directories. Update `HANDOFF.md` after decisions, launches, failures and completions, with exact paths and current status.
