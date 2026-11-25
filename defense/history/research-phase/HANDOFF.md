# Defense research handoff

Updated 22 October 2026 (Asia/Taipei). **Status: research/planning phase complete; implementation, training and new defense evaluation have not started.**

## User objective and constraints

Develop a substantially different defense against the methodology in `BeyondWaveform.pdf`, with the ambition of a publishable research contribution. First understand the paper and reproduction, then investigate alternatives across architecture, objectives, learning paradigms, representations and data strategies. The user explicitly put implementation later.

Originally every artifact was requested in literal `/defense`. The container runs as a non-root user, cannot create that directory even after sandbox escalation, and has no sudo. The user explicitly answered **“Use /workspace/defense”**. That is the canonical root for every new task artifact, including downloads, temporary files, runtime caches, checkpoints, logs and this handoff. Read existing assets elsewhere without modifying them. No root-directory substitution remains unresolved.

## Completed artifacts

Start at [00_RESEARCH_BRIEF.md](00_RESEARCH_BRIEF.md). The six numbered supporting documents contain original-method analysis, repository audit, primary literature, distinct defense designs, experimental protocol and phased plan. [literature_sources.json](literature_sources.json) records checked primary sources and versions. [local_audit.json](evidence/local_audit.json) records 39 hashed local inputs, saved-count arithmetic, named-cohort availability/overlaps, Git state and point-in-time resources. Package checks are recorded in [package_validation.json](evidence/package_validation.json); inspect its pass/fail state rather than assuming model tests ran.

Project domain/issue-tracker instructions and the research skill were read (no glossary or ADR directory exists). Research skill path: `/home/codex/.codex/skills/research/SKILL.md`. Background agent `defense_literature` performed the skill-required primary-source research and wrote the review/initial registry; it has finished. No external messages, issues, PRs, training or inference jobs were created by this task.

## Findings that govern continuation

- Core attack: `V(E(x)+δ)` with normalized global feature-Frobenius constraint, lexical surrogate NLL and clean-normalized TV/HF terms. Frozen E/V/ASR; only per-example δ updates. Radius/MOS do not guarantee unchanged human words. Inference defense cannot require clean audio or reference text.
- Local feature/vocoder path is WavLM layer 6 / compatible kNN-VC HiFi-GAN, independently chosen; no author checkpoint identity established. Existing attack routine freezes passed modules, so a live trainable defense needs a separate inner-loop parameter snapshot/design.
- Historical999 attack at rho 0.5 / lambda 0.1: raw 74.1064% WER, LPF 73.5401%, WaveGuard 8k 71.7331%. Stronger WaveGuard 2k badly damages clean WER. These are saved results, not new inference.
- Later paper-budget rho 0.1 / lambda 1 / 50-step independent control is already complete: 64.7415% WER, UTMOS 3.3582. Do not repeat the old strict audit's obsolete claim that this test is missing. Joint paper anchors remain unmet.
- Several paper defenses have missing/mismatched definitions or weights. No exact CE-AT/DT-AT/DMW/TI/AudioPure-sentence/PVP replacement should be fabricated.
- Novelty limits: TRADES/VAT, dual-manifold training, Wapat, TransferBreaker (Oct 2026), codec robustness v2 (Sep 2026), SPARC's shared WavLM, existing vocoder defenses, chunking/smoothing/voting, joint CTC/attention decoding, and LCAR acoustic reranking. Basic phoneme AT, score fusion or quantization is not a new contribution.
- Lead hypothesis: separate log-Mel supervised phone lattice + acoustic reconciliation/candidate expansion, if phonetic evidence survives the attack and outperforms ordinary equal-resource ensembles under joint adaptive attacks. Joint waveform/FV AT is the learned comparator; local-context recovery is a bounded diagnostic. None has been tested.

## Data and concurrent-work boundaries

The historical999 and old pilot/validation pools are exposed development history for this defense. The old pilot20 shares 2 utterances with validation100. The protected confirmation48 (28 speakers) differs from the scored improvement48 (30 speakers). Existing protected48 and reserve200 belong to previous experiments; this task has not consumed them.

Current available training manifests are `/workspace/improvement/execution/data/train-1h.jsonl` (286 clips, 251 speakers) and `train-10h.jsonl` (2,807 clips, the same 251 speakers). Their audio exists beneath `/workspace/improvement/data/LibriSpeech/train-clean-100`; the 1 h pool is nested within 10 h. Reserve200 points into test-other and is not ours to allocate implicitly. No new defense data split has been frozen. A fresh dev-other speaker allocation is only a proposal, subject to complete exposure checks.

At the research snapshot, TITAN RTX 24 GB was ~92% utilized by another attack-refinement queue. `/workspace/improvement/execution_refine/progress.json` reported running/pilot-generation, supervisor PID 138912 existed. This is a historical point-in-time observation; recheck it before any later launch. Do not interrupt, adopt, restart or duplicate that job.

## Exact next steps when implementation work begins

1. Read this package and inspect current job/process state. Recheck source hashes and data reservations; preserve the dirty baseline checkout.
2. Identify a viable independently pretrained acoustic phone recognizer with auditable source/weights/license, or declare a separate larger-data from-scratch track. Deepen comparison to CTC/attention fusion and LCAR before claiming decoder novelty.
3. Materialize a small frozen development diagnostic: clean/reconstruction/FV/waveform controls, phone-lattice oracle coverage, actual acoustic recovery, local-context error survival and edit-type decomposition. Measure memory/time on a few clips first.
4. Stop the phone branch if there is no recoverable candidate ceiling. Establish clean fine-tuning, waveform/FV/union AT and equal-resource ensemble baselines before promoting a complex model.
5. If a candidate passes, implement/train solely here, attack the full defended pipeline directly, then use a separately allocated confirmation and untouched final reserve under the protocol. Selection, training and attack budgets must be recorded before final access.
6. Update this handoff after every decision, job launch, failure or completed phase. Do not claim publication-level effectiveness before evidence exists.

The user's current request is satisfied by the completed research phase. Do not infer from the future plan that long training should start automatically in this phase. This is a phase boundary from the user, not a skill-imposed approval flow.

## Execution and validation

Research audit: `PYTHONDONTWRITEBYTECODE=1 python3 /workspace/defense/tools/audit_local_evidence.py`. Package checks: `PYTHONDONTWRITEBYTECODE=1 python3 /workspace/defense/tools/validate_research_package.py`. Both write only under this artifact root. They do not run model tests.

Default shell sandbox sometimes fails with `bwrap: No permissions to create a new namespace`; reviewed escalation runs as the same non-root user and has worked for necessary commands. Bare `python` is absent; use `python3` for bookkeeping or the existing `/opt/venv/bin/python` only for later model work. System timezone data is absent, so bookkeeping uses explicit UTC+08:00 for current Taipei timestamps.

No new caches/downloads were required in this phase. Future runs must route `TMPDIR`, framework/model/data caches and all output paths beneath `/workspace/defense`; set `PYTHONDONTWRITEBYTECODE=1` and `BEYONDWAVEFORM_NOTIFY_DISCORD=0`. Do not mutate `/opt/venv`, baseline code or prior run directories. Do not send Discord or other external messages. No new defense performance has been measured.
