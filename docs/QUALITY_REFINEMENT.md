# Continuing WER and sound-quality research

**Completed batch:** all phases exited successfully; no joint WER/quality qualifier advanced to a new test. The strongest new validation candidate reached 76.0692% WER with UTMOS 3.2454. The lowercase candidate reached 66.3442% WER / 3.3306 UTMOS. See [QUALITY_CONTRACTS.md](/workspace/docs/QUALITY_CONTRACTS.md) for the next study. Historical launch details below describe this completed batch.

18 October 2026. Discord job notifications are disabled at the user's request. Current status is `/workspace/reproduction-runs/JOB_STATUS.md`; this batch lives in `/workspace/reproduction-runs/paper-quality-refinement`.

The defense baseline and first paper-budget research batch completed with exit code zero. Available defenses covered LPF 6000 Hz and WaveGuard down/up sampling at 8000 and 2000 Hz; six other paper conditions remain unidentified. See [defense report](/workspace/reproduction-runs/english-defenses-999/report.md).

The first research batch found the following tradeoff, using the same scorer checkpoints and fixed reference/scoring rules:

| Pool | Candidate | WER % | DNSMOS | NISQA | UTMOS |
|---|---|---:|---:|---:|---:|
| 100 validation | Sum NLL, Adam LR 0.02 | 51.9857 | 3.1811 | 4.1038 | 3.5874 |
| 100 validation | Sum NLL, Adam LR 0.1 | 75.5601 | 3.0978 | 4.0254 | 3.2457 |
| 999 paired test | Mean NLL, Adam LR 0.05 | 64.7415 | 3.1262 | 3.9542 | 3.3582 |
| Paper anchors, unmatched protocol | Feature attack | 75.43 | 2.954 | 3.526 | 3.590 |

These results do not simultaneously match the paper's WER and quality. The validation tradeoff motivates this batch. The completed test is retained as a paired benchmark; test utterances and defended results do not select new candidates.

First new audited pilot: lowercase mean-NLL/Adam LR=0.05 reached **109.5477% WER**, DNSMOS/NISQA/UTMOS **3.1531 / 4.3250 / 3.4548** on 12 clips (199 reference words). Its 122 insertions contribute to WER exceeding 100%. Clean/reconstruction scores match the original-case control exactly; all 36 score rows, saved WAVs, and delta bounds passed verification. This is exploratory pilot evidence, still below the UTMOS target and awaiting validation. The batch continues automatically; see its cumulative report for subsequent measurements.

## Frozen search

All candidates retain rho=0.1, lambda=1, 50 updates, saved deltas, the existing compatible WavLM layer-6/HiFi-GAN pair, fixed decoder, scoring references, spectral setup, scorer weights, and manifests. Changes are limited to assumptions identified as unspecified in the strict parity audit.

| Choice | New pilot candidates |
|---|---|
| Lexical target casing | Lowercase or sentence case, crossed with mean/sum NLL and Adam LR 0.05/0.1: eight candidates |
| Update rule | Sum NLL, SGD LR 1/5/20: three candidates |
| HF coefficient alpha | Sum NLL, Adam LR 0.1, alpha 0.1/10: two candidates |
| Initialization | Sum NLL, Adam LR 0.1, Gaussian delta initially at relative radius 0.01: one candidate |

This makes **14 new 12-clip pilots**, plus three audited cached controls: mean/LR=0.05, sum/LR=0.1, and the best previous quality candidate sum/LR=0.02. The latter two also have cached 100-clip validation measurements. Cached values are labeled explicitly and are not fresh replications. SGD rates were chosen using pilot feature/gradient norms only, rather than test feedback. Alpha and initialization remain independent research choices, not recovered author settings.

Casing changes the text passed to attack NLL teacher forcing. Original references remain in generation records and WER scoring, and recognition decoding is unchanged. The optional casing policy is recorded in model artifacts. Omitting it preserves the existing default behavior and metadata for audited controls. Sentence case lowercases the text and capitalizes its first character; it does not infer proper names or add punctuation.

## Selection and confirmation

Pilot results promote three candidates: the previous quality-first rule, the strongest remaining candidate passing all three MOS >2.5 gates, and a balanced candidate maximizing its weakest WER/quality target ratio. All decisions use pilot evidence and deterministic ties. Decision receipts include immutable evaluation hashes; rerunning an audit cannot invalidate a receipt solely because its audit timestamp changed.

A validation candidate qualifies only with WER >=73.43% and DNSMOS/NISQA/UTMOS >=2.954/3.526/3.590. The WER floor is a predeclared exploratory target two percentage points below the published WER, not a statistically established equivalence margin. Qualifiers are tried in descending validation WER on a second 100-clip run with seed 2027. Only a candidate qualifying on both runs advances to one 999-clip evaluation. If none qualifies, the batch completes with an explicit negative finding and no additional test run. Test results cannot choose a replacement.

The repeat checks observed optimizer variability; it does not establish statistical stability. Historical same-seed runs already produced different adversarial waveforms despite identical clean/reconstruction waveforms. A seed change with zero initialization does not guarantee an independent random perturbation. Close thresholds and generalization require further evidence.

The current 12/100/999 manifest IDs remain unchanged to preserve comparisons. The author subset is unknown. Repeated use of the historical test pool is a paired benchmark, not a fresh blind holdout. Scores are automatic quality estimates; there is no new human listening study.

## Verification and operation

Before changing source, all 71 previous frozen inputs and the relevant completed summary hashes were checked. Previous source/plan/results snapshots are under `history/`, with a verification receipt; completed plans and evaluations are preserved. Each new run independently verifies the saved delta against re-extracted clean features, all WAV hashes, complete score coverage, lexical edits, MOS aggregates, model/config/runtime identities, and 50 finite steps. Cached runs receive a separate `refinement-verification.json` so the original frozen audit is preserved.

`scripts/run_quality_refinement.py --prepare` freezes the new source, all serialized candidate and confirmation configs, previous evidence, manifests, and model/scorer artifacts. Preflight uses the actual generation config schemas. Each generation/evaluation/audit phase has its own log and job record, with 30-second overall heartbeats. `run_logged.py` supervises the persistent process. The status updater tracks this batch using `--research paper-quality-refinement`; it now waits for a live supervisor to finalize an exited child's record before calling a job stopped.

Launch environment:

```text
PYTHONPATH=/workspace/artifacts/defense-runtime:/workspace
OMP_NUM_THREADS=4
MKL_NUM_THREADS=4
OPENBLAS_NUM_THREADS=1
PYTHONUTF8=1
PYTHONIOENCODING=utf-8
BEYONDWAVEFORM_NOTIFY_DISCORD=0
```

Inspect `job.json` and current processes before starting another copy. Cumulative audited results are in `report.md`/`results.json`; live progress is in `research.log` and `progress.json`. Reuse on resume requires matching frozen inputs and fresh independent audits. No notification credential or access change is needed.
