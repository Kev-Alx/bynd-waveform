# BeyondWaveform research handoff
**Latest model execution:** failed_execution; read [25_learned_model_results.md](25_learned_model_results.md), `execution_model_v1/status.json` and exact exit receipts. Candidate `None`; publication planning is deferred. The original interface server remains on8888.


**Latest execution update, 27 October 2026:** publication planning is deferred. A new direct-trained learned speech-feature model comparison is registered and detached/queued behind the existing GPU job. See [current experiment and live receipts](25_learned_model_results.md) and [fixed protocol](24_learned_feature_protocol.md). No new complete performance result is available yet. The 13-hour overall ceiling includes queue wait and ends no later than **28 October, 11:59 Asia/Taipei**. The controller profiles actual throughput before long-training admission, scores eight full 48-point conditions, and opens validation/transfer only if the fixed WER/quality and declared efficiency screens pass. Existing frozen experiments and the server on 8888 are preserved. This update takes precedence over older publication-planning language.

**Latest steering, 27 October 2026:** the user says the publication plan is premature. Focus on finding a definitive measured improvement; keep the method direction open, including a trained model with fast new-audio inference. The earlier publication roadmap is background, not a commitment or a gate on exploring models. Preserve WER/quality thresholds, transfer/defense requirements, and equal priority for runtime. Any next script is queued/detached if GPU is needed, with a maximum 13-hour overall ceiling including queue wait. Current continuation artifacts are under `execution_model_v1/`; inspect its status and actual GPU processes. The audio interface remains on port8888.

**Continuation completed 26 October 2026, 17:59 Asia/Taipei.** Read [21_quality_efficiency_results.md](21_quality_efficiency_results.md) and `execution_tradeoff/results.json` first for the latest paired 48-point experiment. It completed normally under its 45-minute ceiling; The pilot completed normally, but no condition passed the unchanged joint gate. Selection is scientific stop; candidate is null. Validation and transfer/defense tests remain gated off. Matched decoder graph reuse cut loop time 14.63% with exact frozen-output parity; spectrum repair changed WER/UTMOS as recorded in the new report. Runtime and quality remain equal priorities. The user also explicitly requires publication-level contribution/novelty in the speech/NLP theme; read [22_publication_positioning.md](22_publication_positioning.md) before selecting the next method. `JOB_STATUS.json` describes the older expanded queue and remains frozen; current cycle status/exit receipts are `execution_tradeoff/status.json` and `execution_tradeoff/exit_receipt.json`. Interface regeneration now uses `execution_tradeoff/refresh_interface.py` so the new study is classified, and delivery uses `execution_tradeoff/build_interface.py` for staging immutable assets before atomic index replacement. Do not rerun its preserved staging directory; choose a new staged build for a future revision. The server remains on port 8888.



Verified **26 October 2026, 16:42 Asia/Taipei**. Workspace: `/workspace`.

**Latest user priority update, 26 October:** runtime and computational scalability are **equally important to the audio-quality problem**. Apply the priorities below when continuing; older summaries emphasizing quality alone do not capture this latest instruction.

## Start here

1. Read this file, then [the plain-language summary](17_summary_and_next_steps.md). Read [the complete results](16_all_results.md) when comparing methods. These supersede the running-state language in older reports.
2. Check `/workspace/improvement/JOB_STATUS.json` and `/workspace/improvement/audio_browser_server_status.json`, then inspect actual processes before launching anything. At handoff, **research is complete and idle; the audio web server is running**.
3. Preserve the listening interface on **`0.0.0.0:8888`**. Verify `http://127.0.0.1:8888/` returns 200. The user explicitly chose this existing Docker port.
4. Continue from the latest user request. For research continuation, inspect both the quality failures and measured compute costs, then register a hypothesis addressing the quality/runtime tradeoff before committing another long compute budget. The prior queue completed; it does not need a restart.

For the user's follow-up about runtime, future training cost and saved WAVs, read [18_runtime_and_saved_audio.md](18_runtime_and_saved_audio.md). The expanded cycle performed 144,000 per-recording optimization updates across four trajectories; it did not train a large model. Completed pilots retain their WAVs, including reused controls referenced by each `generated.jsonl`.

The original request was steps 1–4 of a research brief, but later messages authorized implementation, experiments, more data, larger models and longer runs. The latest completed deliverable is a plain-language research summary, complete metrics catalogue and working listening interface. The user requested this handoff because the thread became too long.

## User constraints and working conventions

- Put **every new deliverable, script, model/download cache, audio file, status file and log under `/workspace/improvement`**. This is the one project-root artifact directory. Preserve the original implementation and existing experiments.
- Use **48 paired recordings for performance pilots**. Small engineering/gradient checks may establish execution correctness, but are not pilot performance evidence. The user also permits advancing beyond pilots when the evidence justifies it.
- Objective: **source WER strictly above 75%, acceptable audio quality, measured transfer to unseen recognizers, persistence against the original method's defenses, and practical computational scalability**. Audio quality and runtime are equal research priorities. No joint improvement has yet been demonstrated.
- Fixed cohort-mean quality floors: **DNSMOS OVRL ≥ 3, NISQA MOS ≥ 3.75, UTMOS ≥ 3.5**. Keep the thresholds and full-cohort denominators. A high source WER alone does not qualify.
- Run long work **detached**, with persistent stage logs, ~30-second heartbeats, explicit completion/failure/exit receipts, and a declared time ceiling. The user has repeatedly been frustrated by idle or apparently stuck jobs; report meaningful progress and distinguish a scientific stop from an execution failure.
- Progress autonomously within the authorized scope. Keep the explanation/results documents current. No Discord/other external messaging is authorized; existing queues set `BEYONDWAVEFORM_NOTIFY_DISCORD=0`.
- User-facing dates/times: **Asia/Taipei (UTC+8)**.
- Follow `/workspace/AGENTS.md`. Before domain/architecture exploration, read `docs/agents/domain.md`; its glossary/ADR paths were absent. Before any GitHub issue/spec operations, read `docs/agents/issue-tracker.md`; triage has a separate label guide. No PR or commit was requested. Preserve pre-existing repository changes.

## Equal priorities: audio quality and computational scalability

The user explicitly wants the extreme runtime treated as a **method-level problem**, alongside speech quality. Earlier authorization for larger models, more data and longer runs permits investigation; it does not establish that a method taking many minutes per new recording is practical.

- **Measure the complete cost.** Separate one-time preprocessing/teacher-target generation, training, per-new-recording optimization or inference, and formal evaluation. Report generation seconds per recording and per second of audio, total GPU/wall hours, peak memory, and source/decoder call counts where available. Keep loading/calibration and activation recomputation visible. Compare equivalent hardware, clip lengths, precision, batching and conditions; distinguish measured values from forecasts.
- **Estimate scale before a long launch.** Use measured throughput to project the next 48-point study, intended training corpus and unseen-audio workload, with assumptions and a declared budget. The previous 45h 51m covered four search trajectories plus scoring; use the relevant individual method's cost when forecasting a chosen candidate.
- **Investigate efficiency explicitly.** Profile the expensive model/decoder passes and memory/recomputation tradeoff. Test justified reductions in repeated computation or optimization steps, or a reusable learned method, while retaining matched attack/quality/transfer controls. Attribute hardware upgrades separately from algorithmic gains.
- **Keep training and deployment claims separate.** A per-recording optimization has no one-time training phase that automatically makes future inputs cheap. A generator/student might reduce deployment cost, but teacher-data generation, actual training cost, and retained WER/quality/transfer must all be measured. The earlier generator failed; a future speedup is unproven.
- **Assess the joint tradeoff.** Report both quality and runtime for candidate selection and paper claims. A slow high-WER result may be a mechanism experiment, but does not resolve practical scalability. No numerical latency requirement has been supplied: state and justify proposed efficiency targets rather than inventing a user-mandated cutoff. Preserve the existing WER/quality gates and frozen evidence.

Runtime measurements, counters, conditional scaling examples and saved WAV paths: [18_runtime_and_saved_audio.md](18_runtime_and_saved_audio.md).

## Current scientific state

The expanded 48-hour queue ran **23 October 00:11 → 24 October 22:02, Taipei time**, taking **45h 51m**. Every scheduled stage exited successfully. Its frozen selection is **`stop`, candidate `null`** because no method passed WER >75% and all quality floors. The queue ended normally before its ceiling.

**No 76-point validation, held-out transfer/defense evaluation, untouched confirmation/final evaluation or human study ran.** These stages were gated off. The larger source models were generation sources, not unseen transfer targets.

The completed reporting catalogue contains **67 condition rows / 2,664 scored recordings / 48 distinct sentences / 1,602 unique audio assets**. There are 23 condition rows on the historical 24-clip cohort and 44 on the 48-clip cohort. Repeated controls and checkpoint evaluations are retained per study, not independent replications. The catalogue covers the development research and three historical original controls, not every unrelated legacy reproduction result.

Selected results, all on **48 recordings / 994 reference words**:

| Condition | WER % | DNSMOS | NISQA | UTMOS | Interpretation |
|---|---:|---:|---:|---:|---|
| Clean | 4.0241 | 3.2721 | 4.0533 | 4.0259 | Reference |
| Original quality control, FV50 / LR 0.02 | 49.7988 | 3.1739 | 4.0698 | 3.5789 | Strongest completed 48-point attack meeting all floors |
| Weighted waveform PGD50 | 43.4608 | 3.1656 | 3.7776 | 3.8873 | Quality passes; weaker attack |
| Waveform NLL-only PGD50 | 58.7525 | 3.0025 | 3.4367 | 3.5678 | Fails NISQA |
| Weighted waveform100 | 53.4205 | 3.1158 | 3.5937 | 3.8390 | Fails NISQA |
| Hard-constrained waveform100 | 43.4608 | 3.1655 | 3.7771 | 3.8871 | No WER gain; most proposed updates rejected |
| Quiet-region NLL50 | 29.1751 | 3.2752 | 4.0294 | 3.9140 | Quality passes; weak attack |
| Strong original FV1500 / LR 0.1 | 101.4085 | 3.0196 | 3.8260 | 2.9437 | Known-method control; fails UTMOS |
| Three-source raw FV ensemble500 | 90.4427 | 3.0997 | 3.9295 | 3.0557 | Fails UTMOS |
| Three-source transformed FV ensemble500 | 89.5372 | 3.1476 | 4.0211 | 3.0850 | Fails UTMOS |
| BigVGAN reconstruction | 3.9235 | 3.2981 | 4.0521 | 3.9868 | Decoder qualification passed |
| BigVGAN mel ensemble200 | 80.1811 | 2.8827 | 3.3091 | 2.2951 | Fails all quality floors |
| BigVGAN mel ensemble500 | 78.7726 | 2.6038 | 2.7068 | 1.6528 | Fails all quality floors |

WER pools substitutions + deletions + insertions over reference words. It can exceed 100% due to insertions. MOS values average recordings. Neither machine WER nor automatic MOS establishes preservation of the words for human listeners.

## What was tested; avoid repeating closed versions blindly

Original independent implementation: frozen WavLM layer 6 → per-recording feature edit → matching HiFi-GAN; Whisper-small supplies NLL gradients; feature-radius and TV/HF penalties constrain the edit. This is an independent reproduction, not verified author-checkpoint parity.

| Branch | Modification and result | Evidence |
|---|---|---|
| R1, 24 clips | Decoded candidate word-risk. Canonical candidates sometimes all word-correct, yielding no useful risk gradient. One registered deletion-candidate repair reached WER 58.5263 / UTMOS 3.3497: fails quality. | `06_initial_results.md`, `execution/r1-deletion-development24/` |
| R2, 24 clips | Local mixtures of similar speech features. Too few useful directions; WER 3.3684, near clean. Closed. | `execution/r2-canonical-zero-uniform-dev24/` |
| R3 | One seed, 1.59M-parameter waveform U-Net, 1-hour disjoint training set, 1,779 updates in about one hour. Learned almost-zero edits; near-clean WER. Direct waveform PGD subsequently showed the waveform objective can attack. | `06_initial_results.md`, `08_pilot48_results.md`, `execution/r3-waveform-autograd-repair/` |
| Waveform continuation | Weighted vs hard-constrained 100 total steps. Hard constraints rejected 1,789/2,400 attempted continuation updates; no corpus-WER gain. Weighted continuation failed NISQA. | `09_refinement_protocol.md`, `10_refinement_results.md`, `execution_refine/` |
| Quiet-region budgets | Bound edits using clean 25 ms window energy / 10 ms hops, with matched global-energy control. Quality retained, attack weakened. Closed. | `12_envelope_protocol.md`, `13_envelope_results.md`, `execution_envelope/` |
| Expanded sources/decoder | Three sources, longer optimization, sampled transformations; separate qualified BigVGAN mel-edit branch. Strong source WER, failed quality. | `14_scaled_research_protocol.md`, `15_scaled_results.md`, `execution_scaled/` |

One engineering repair in `r3_encoder_repair.py` clones a frozen upstream WavLM positional-convolution input on gradient-bearing calls, avoiding an in-place autograd failure. Forward values were verified unchanged. This is an execution repair, not a scientific contribution.

The larger source models and BigVGAN were **frozen**, not newly trained/fine-tuned. The prepared ten-hour training/distillation branch has **not** run. Merely adding models, a generator, or more steps is not itself a novelty claim.

## Research continuation: first useful investigation

The unresolved bottlenecks are **quality/content retention and computational cost**. Both must shape the next study. Proposed steps, **not queued jobs**:

1. Diagnose both problems using existing evidence: listen to clean / original quality control / stronger variants on the same sentences, and break down the saved per-method timing/call counts. Use representative successes and failures; exploratory listening is not a blinded human score.
2. Inspect saved S/D/I, perturbations and trajectories to distinguish recognition confusion from damaged speech or insertion-heavy hallucinations. In parallel with this analysis, identify expensive repeated work, the memory/speed tradeoff, and realistic scaling costs. The quality failure mechanism and achievable efficiency gains remain open questions.
3. Register **one specific change with explicit quality and efficiency outcomes**, fixed controls, a 48-point performance pilot and a measured-throughput budget. A study may isolate one bottleneck, but must track whether it worsens the other. A learned policy should follow a successful per-recording teacher and include the teacher-data/training cost. Avoid another unmotivated long-run sweep or relaxed quality thresholds.
4. Once a fixed candidate passes the source/quality screen, document its measured runtime and feasible next-stage budget, then freeze selection before 76-point development validation, unseen-model transfer and registered preprocessing evaluation. Preserve untouched confirmation/final sets and require a blinded human study before claiming intelligibility. A paper-level improvement must account for computational scalability as well as attack strength, quality, transfer and defenses.

Use the user's latest steering to choose the next concrete study. The old 48-hour ceiling belonged to the completed expanded cycle; document a new bound for any new queue. Preserve its frozen evidence rather than restarting it under a renewed timer.

## Frozen research code, models and data

Before changing an experimental path, read its plan. **Registered files are hash-bound. Version/amend changes in a new execution directory; preserve old code, plans, outputs and receipts.** `execution_scaled/plan.json` binds 72 inputs. `supervise_scaled.py` intentionally refuses to relaunch over prior registration.

Expanded implementation: `scaled_components.py`, `scaled_attack.py`, `scaled_scope.py`, `scaled_run.py`, `scaled_source_evaluate.py`, `scaled_postprocess.py`, `scaled_transfer_evaluate.py`, `supervise_scaled.py`, and `execution_scaled/audit_scaled.py`. These and referenced adapters/tests/model inputs are governed by the frozen plan. Read it for exact settings and hashes.

- Sources: Whisper-small (244M), Whisper-large-v2 (~1.55B), `facebook/data2vec-audio-base-960h`. Ensemble weights 0.5 / 0.25 / 0.25, with fixed reconstruction-gradient calibration. Large-v2 reuses `/workspace/artifacts/whisper/large-v2.pt`; inspect the plan for authoritative paths.
- New cached assets: `execution_scaled/models/data2vec/` and `execution_scaled/models/bigvgan/`. BigVGAN v2 has 112,414,512 parameters, native 24 kHz / 100 mel channels. Receipts/revisions are in `execution_scaled/assets.json` and the plan.
- FV variants: original quality control; single-source endpoints 50/150/200/500/600/1500; raw and transformed ensemble endpoints 50/200/500. Original feature geometry and penalties retained. Strong controls use LR 0.1, quality control LR 0.02.
- Transforms during optimization: identity, 129-tap Hann FIR 6 kHz low-pass, TorchAudio 16→8→16 kHz resampling. They are named optimization approximations, not proof of surviving every original-paper defense.
- Mel branch: 50/200/500 endpoints, relative ball 0.1, absolute log-mel edit bound `ln(2)`, Adam LR 0.01. Exact preprocessing, padding and clipping receipts are saved.
- Common scoring: frozen native Whisper-small, English, beam 5 / temperature 0, summed NLL for generation. Formal scoring uses the historical backend, in a separate process.
- Pilot48: `execution48/development-48.jsonl`, 30 speakers / 360.010 s, SHA-256 `9a5f56f5cca6c073dbe6e2b6d804971735045ea80c4456bea0d577b7e292748a`.
- Validation76: `execution_refine/validation-76.jsonl`, 34 speakers / 524.905 s, SHA-256 `3747ca5b726ef899c24bc8f92e83ba44f086a537bffcf98b9bcd52cba71df30a`. Utterance/hash/text disjoint from pilot48; shared speakers and historical exposure mean this is exposed development.
- Untouched utterance-level confirmation48: `/workspace/reproduction-runs/paper-quality-finalization/data/confirmation/confirmation-48.jsonl`. All its speakers overlap historical development. See `data_and_evaluation_constraints.md` before opening outcomes.
- Untouched final200: `execution/data/final-reserve-200.jsonl`, test-other / 33 speakers; reserve SHA-256 is in `execution/plan.json`. Keep it out of tuning.
- Prepared training: `execution/data/train-1h.jsonl` (286 clips) and `train-10h.jsonl` (2,807 clips), with data receipts. Training speakers were excluded from development/reserve as documented. One-hour R3 is the only learned attack trained here.
- Four planned unseen targets: `/workspace/configs/targets/{whisper_base,whisper_medium,wav2vec2_large,hubert_large}.yaml`. They remain unqueried in these improvement cycles. The planned transfer gate/defense recipes and limits are in `14_scaled_research_protocol.md` and `scaled_transfer_evaluate.py`.

The old 999-point reproduction outcomes are already exposed; do not treat them as a new final holdout. The old 20-clip pilot also overlaps validation100 by two utterances. Baseline/source/paper audit and prior work: `01_original_method_and_baseline.md`, `baseline_evidence.json`, `03_literature_review.md`, `literature_sources.json`, and `execution_scaled/{generative-prior-work,transfer-defense-prior-work}.md`.

## Execution environment

- Python: `/opt/venv/bin/python`. One TITAN RTX, 24 GB VRAM; inspect current GPU/process state before new compute.
- Shell tools in this session required `sandbox_permissions: require_escalated` because default sandbox execution failed. These were approved; use the current environment's tool policy and report any actual denial.
- Research GPU queues share `/workspace/improvement/execution/continuation.lock`.
- Long-launch pattern: `subprocess.Popen(..., stdin=DEVNULL, stdout=log, stderr=STDOUT, start_new_session=True)`, wrapping the child with `/workspace/scripts/run_logged.py --log ... --status ... -- ...`. Do not rely on the interactive terminal or chat remaining alive.
- Generation's strict environment: `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`. Formal native scoring runs fresh with `CUBLAS_WORKSPACE_CONFIG` unset. Avoid changing the evaluation backend implicitly.
- Prefer `PYTHONDONTWRITEBYTECODE=1`; CPU-only checks use `CUDA_VISIBLE_DEVICES=` and pytest can use `-p no:cacheprovider`. Use meaningful targeted checks; preserve failures and version repairs.

## Audio interface continuation

**Keep this service available.** Verified HTTP 200 at handoff. Snapshot PIDs: wrapper **174740**, server **174742**, detached with no controlling terminal. PIDs may change; check the live status before acting.

| Item | Path / purpose |
|---|---|
| Source project | `/workspace/improvement/audio_browser/` |
| Source instructions | `audio_browser/AGENTS.md`; read before editing this app |
| Authored UI | `audio_browser/src/content/dashboard/AudioLab.jsx`, `audio.css`; `DashboardContent.jsx` re-exports the entry |
| Reviewed data | `audio_browser/src/data.json`, queries `runs`, `recordings`, `clips`; stable app ID is already assigned |
| Data preparation | `build_audio_browser_data.py`: checks all source audio hashes, S/D/I/reference totals and arithmetic mean MOS against saved receipts; writes `16_all_results.md` and `all_results.csv` |
| **Complete build command** | `/opt/venv/bin/python /workspace/improvement/build_audio_browser.py` |
| Server | `serve_audio_browser.py`, binds `0.0.0.0:8888`, serves only compiled `audio_browser/dist`, supports byte ranges for seeking |
| Server logs/status | `audio_browser_server.log`, `audio_browser_server_status.json` (30 s heartbeat), `audio_browser_server_job.json` (wrapper/exit receipt) |
| Operating guide | `audio_browser/README_AUDIO_BROWSER.md` |
| Validation receipts | `audio_browser_data_receipt.json`, `audio_browser_build_receipt.json`, `audio_browser_browser_receipt.json`, `audio_browser_delivery_receipt.json` |

**Build gotcha:** the Data compiler clears `dist`. The complete build helper above restages all 1,602 WAVs and CSV afterward. Running only the Data compiler leaves the UI without audio; this was caught and fixed during browser QA. Keep `buildStatus` at `creating`/`updating` while working, then `complete` and rebuild at delivery. Data regeneration resets it to `creating`. Follow the app's protected-runtime boundaries; custom content lives under `src/content/`.

Playback preserves original WAV bytes with no normalization, resampling, trimming or lossy compression. PCM-identical recordings share an asset. The assets occupy about 762 MB; source and served copies use hard links where supported. Current capabilities: 1–3 panels, per-panel run selection/descriptions, a shared sentence list, search, explicit missing-cohort states, reference/ASR alignment and exact transcripts, clip scores, run mean/status, linked playback positions, speed/seek/stop/download, all-results filters/sorting/CSV, and explanation/future-work tab.

Actual Chromium QA passed WAV playback, position switching, seeking, speed, panel counts, sentence search, cohort/study/quality filters, reset, export reconciliation, narrow-layout containment and no uncaught runtime errors. See `audio_browser_qa/check_browser.cjs` and screenshots. QA dependencies/downloads/libraries are isolated under `audio_browser_qa/`; browsers need its `PLAYWRIGHT_BROWSERS_PATH` and `LD_LIBRARY_PATH` (recorded in prior launcher logs). Intermediate QA failures/logs remain preserved; the final receipts pass.

Delivery is the user's **local Docker port**, not a cloud deployment. A private Sites draft was registered before the user specified port 8888. Its existing opaque project ID is in `audio_browser/.openai/hosting.json`; no Site version was deployed. If cloud publication is later requested, reuse that identity and follow current Sites/Data publication instructions rather than registering a second Site. No credentials were saved to the project.

## Evidence entry points

- **Current human-facing explanation:** [17_summary_and_next_steps.md](17_summary_and_next_steps.md).
- **All development WER/MOS rows:** [16_all_results.md](16_all_results.md), [all_results.csv](all_results.csv); source receipts remain in each execution directory.
- **Expanded final result/decision:** [15_scaled_results.md](15_scaled_results.md), `execution_scaled/selection.json`, `JOB_STATUS.json`.
- **Original brief/planning:** `request.txt`, documents `01`–`04`; later user authorizations expanded that initial boundary.
- **Continuation prompt:** [NEXT_AGENT_PROMPT.md](NEXT_AGENT_PROMPT.md).

The handoff/reporting task is complete when these files are saved and the server remains available. Future research success requires new measured evidence, not merely a successfully completed job.
