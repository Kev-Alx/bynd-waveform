# BeyondWaveform handoff

## Objective and working agreement

Continue the independent reproduction of `/workspace/BeyondWaveform.pdf` using `/workspace/BeyondWaveform_Reproduction_Spec.md`: modular Python code for downloads, preparation, model loading, attacks, training, evaluation, and reporting. Most implementation already exists; continue the numerical experiments.

The user authorizes running necessary scripts. Every long task must show progress. Launch long jobs, confirm they are active, then end the chat turn while waiting and give the live log/result paths. Continue when the user returns. Do not stop before launching or repeatedly request permission. Do not claim that unavailable author artifacts have been reproduced exactly.

## Environment and launching

- Workdir: `/workspace`, inside the supplied Docker container. Python: `/opt/venv/bin/python`; plain `python` is not on the shell PATH. GPU: NVIDIA TITAN RTX (24 GB). No Git repository was present.
- The inner shell sandbox has failed with `bwrap: No permissions to create a new namespace`; shell calls have worked using `sandbox_permissions: require_escalated`. The user has authorized this local work.
- `/workspace/runs` is root-owned. Use the writable `/workspace/reproduction-runs` for new outputs.
- A previous chat-terminal process stopped after 16 clips without an error. Detached tmux then ran the entire validation grid successfully. Use detached tmux for subsequent long jobs.
- `/workspace/scripts/run_logged.py --log LOG --status JOB_JSON -- COMMAND ...` forwards existing progress indicators to the terminal and an appended log, and records `running`, `complete`, `failed`, or `interrupted`, including exit code.
- The tmux socket is `/workspace/reproduction-runs/tmux.sock`. Example: `tmux -S /workspace/reproduction-runs/tmux.sock new-session -d -s JOBNAME -c /workspace 'COMMAND'`. Confirm process, log progress, and job record. A finished tmux session disappears; its disappearance alone does not indicate success.

## Existing implementation and fidelity

Package: `/workspace/beyondwaveform`; entry points: `/workspace/scripts`; configs: `/workspace/configs`. Read `/workspace/docs/REPRODUCTION.md` for equations, assumptions, and evidence; `/workspace/README.md` has commands.

Feature attack uses frozen WavLM-Large, same-shaped feature delta, frozen feature-conditioned HiFi-GAN, and Whisper-small lexical teacher-forcing NLL. It minimizes negative NLL plus normalized temporal-variation/high-frequency penalties and projects delta into a global relative Frobenius ball after each of 50 updates. No kNN replacement or waveform residual is inserted. Encoder layer 6 and compatible regular vocoder are independently selected artifacts, not author-confirmed choices.

Pinned kNN-VC source commit: `c616845c4e309e24d5927f15adbdf277a3d65358`; regular vocoder `g_02500000.pt`. Artifacts are under `/workspace/artifacts`. DNSMOS/NISQA/UTMOS models are downloaded and real scoring works. The original 36-test suite and architecture check passed; four wrapper/resume tests subsequently passed.

Models/optimizer/STFT/scorers/decoders, author manifests, Mandarin fine-tuned weights, and defense artifacts are incompletely specified in the paper. Report independent measurements and discrepancies. Do not fabricate defense, human-study, or over-the-air results.

## Data and completed pilots

LibriSpeech dev-clean/test-clean and AISHELL-1 are downloaded and extracted. Disjoint manifests were audited:

- `/workspace/data/manifests/reconstruction-pilot.jsonl`: 12 longer pilot clips.
- `/workspace/data/manifests/english-validation.jsonl`: 100 dev-clean validation clips.
- `/workspace/data/manifests/english-test-999.jsonl`: 999 test-clean clips.

The older `/workspace/data/manifests/pilot.jsonl` overlaps validation at one item; do not use it for further tuning. The original one-clip real smoke was uninformative because clean recognition already failed.

On the 12-clip disjoint pilot (199 words), clean WER was 5.03%, reconstruction WER 6.03%. At rho=0.1 and lambda=1, 50-step Adam learning-rate results:

| LR | WER % | DNSMOS | NISQA | UTMOS |
|---:|---:|---:|---:|---:|
| 0.01 | 23.12 | 3.146 | 4.377 | 3.865 |
| 0.02 | 41.21 | 3.118 | 4.283 | 3.615 |
| 0.05 | 63.82 | 3.133 | 4.210 | 3.420 |

Scores: `/workspace/reproduction-runs/optimizer-pilot/summary.json`. LR 0.05 was frozen for the validation grid in `/workspace/configs/english_lr005.yaml`; `/workspace/configs/english.yaml` retains LR 0.01. The paper does not specify LR. All 12 final perturbations at LR 0.05 reached rho=0.1.

## Completed validation grid

`/workspace/reproduction-runs/validation-grid/job.json` reports `status: complete`, `exit_code: 0`. The log ends with `Completed 16 experiments`. All 16 per-setting `evaluation/summary.json` files exist.

The grid used LR 0.05, 50 steps, all 100 validation clips, surrogate Whisper-small evaluation and all three quality metrics. Rho values: 0.05, 0.1, 0.5, 1.0. Lambda values: 0.1, 0.5, 1.0, 2.0.

Actual output: `/workspace/reproduction-runs/validation-grid`; log: `/workspace/reproduction-runs/validation-grid.log`; plan: `/workspace/reproduction-runs/validation-grid/plan.json`. The older `/workspace/runs/validation-grid` contains configs only; it was initially run without `--execute`.

## Selected config and completed 999-clip source run

On 2026-10-10, `/workspace/scripts/verify_validation_grid.py` passed all 16 settings / 1,600 validation utterance results. It verified complete 100-clip coverage, validation-only roles, identical manifest/ordered IDs, raw surrogate and matched scoring/model/decoder provenance, JSONL hashes, waveform-hash correspondence in saved records, recomputed lexical edit counts/corpus WER/MOS means, and every finite 50-step projection trace. The disjoint reconstruction-pilot, validation and test manifests passed the split audit. No model computation was rerun. This audit does not rehash every validation WAV; final evaluation rehashes its saved WAVs.

- Verification: `/workspace/reproduction-runs/validation-grid/verification.json`.
- Existing strict quality-gated selector audit: `/workspace/reproduction-runs/validation-grid/selection.json`.
- Measured validation report from the existing report script: `/workspace/reproduction-runs/validation-grid/report.md`.
- All 16 settings passed DNSMOS/NISQA/UTMOS **each strictly >2.5**.
- Winner: **rho=0.5, lambda_perc=0.1**, validation surrogate WER **74.1344195519%**; DNSMOS **2.9317973918**, NISQA **3.9221272361**, UTMOS **2.7898755634**.
- Source: `/workspace/reproduction-runs/validation-grid/rho-0.5_lambda-0.1/run.json`; generation fingerprint `ab0ca7017eb13767d36f2b069727712831f652a008eeed265f77e87b0c8b32bd`.

The source run's complete `config` was exported to `/workspace/configs/english_selected.yaml`. YAML and `/workspace/configs/english_selected.freeze.json` are read-only (0444). All fields were retained: Adam LR 0.05, 50 steps, zero initialization, seed 2026, layer 6 / regular vocoder / `normalize: false`, native lexical-mean Whisper-small and exact decoder/STFT/perceptual settings. YAML SHA256: `7a4d61aa1d5d03c36c2d6e3fa2f61303c896b96041a0a3661df6df1e0770f59a`. Freeze receipt includes source evidence, selected artifact hashes, and the complete 999 test IDs. Test manifest SHA256: `3ad09278b9a5160add8ad37a81fafd6cfbf5510ae164b3ffefef17e6fbf32148`.

`/workspace/scripts/run_selected_test.py` performs frozen-input checks, generation, then surrogate evaluation of **clean, reconstruction, adversarial** saved WAVs with **all three quality metrics**, then the existing report script. It retains existing generation resume checks, stops on phase failure, and checks complete counts/provenance. Evaluation restarts as a complete phase if interrupted. It exposes 30-second heartbeats plus existing clip progress. `--check-only` preflight passed; 16 targeted logging, pipeline/resume and data/evaluation tests passed; both new scripts compiled.

Detached tmux session **`english-test-999`**, socket `/workspace/reproduction-runs/tmux.sock`, launched at `2026-10-10T02:44:54Z` (10:44:54 Asia/Taipei) and completed with **exit code 0** at `2026-10-10T07:43:11Z` (15:43:11 Asia/Taipei). Both overall and stage records say `complete`, covering 999 clips. The session has ended. Results were independently audited after completion; see below.

- Log: `/workspace/reproduction-runs/english-feature-test-999.log`.
- Overall completion/exit status: `/workspace/reproduction-runs/english-feature-test-999/job.json`.
- Stage/heartbeat/count status: `/workspace/reproduction-runs/english-feature-test-999/progress.json`.
- Generation metadata: `/workspace/reproduction-runs/english-feature-test-999/run.json`.
- Generation records as they finish: `/workspace/reproduction-runs/english-feature-test-999/records/`.
- Final generated index: `/workspace/reproduction-runs/english-feature-test-999/generated.jsonl`.
- Completed evaluation: `/workspace/reproduction-runs/english-feature-test-999/surrogate-eval/summary.json` and `utterances.jsonl`.
- Expanded verified report: `/workspace/reproduction-runs/english-feature-test-999/report.md`.

Launch command (do **not** start a second concurrent job; check tmux/job/process first):

```bash
tmux -S /workspace/reproduction-runs/tmux.sock new-session -d -s english-test-999 -c /workspace '/opt/venv/bin/python -u /workspace/scripts/run_logged.py --log /workspace/reproduction-runs/english-feature-test-999.log --status /workspace/reproduction-runs/english-feature-test-999/job.json -- /opt/venv/bin/python -u /workspace/scripts/run_selected_test.py --freeze /workspace/configs/english_selected.freeze.json --output /workspace/reproduction-runs/english-feature-test-999'
```

The selected rho/lambda differ from the paper's stated rho=0.1 / lambda=1. Other documented independent choices remain intact. WavLM checkpoint `cfg.normalize` is true while the explicit upstream-compatible reproduction policy is `normalize: false`; this was preserved. Missing author checkpoint/manifest/recipe IDs and different hardware prevent an exact numerical reproduction claim. Transfer, Mandarin, defenses, human study and OTA were not measured by this core run.

## Verified English core results

`/workspace/scripts/verify_test_run.py` passed the full independent audit: 999 original FLACs and 2,997 saved WAVs rehashed, float-WAV headers/lengths and generation records checked, complete `(ID, variant)` coverage, frozen config/manifest/artifacts, exact references/lexical edit counts, recomputed corpus WER and all MOS means, 49,950 finite projected updates. Evaluation IDs repeat across variants; the auditor uses pair uniqueness instead of the manifest reader. A negative check correctly rejected duplicate/missing variant pairs even with a matching modified JSONL digest. No model scoring was rerun for verification.

| Variant | Count | WER % | DNSMOS | NISQA | UTMOS |
|---|---:|---:|---:|---:|---:|
| Original clean | 999 | 3.963565 | 3.264184 | 4.029123 | 4.084511 |
| Reconstruction | 999 | 3.727228 | 3.272338 | 4.141565 | 4.028398 |
| Feature attack | 999 | 74.106352 | 2.963822 | 3.871132 | 2.792377 |

Each variant covers **20,310 reference words**. Attack WER increases 70.142787 pp from clean and 70.379124 pp from reconstruction; the latter control does not exhibit reconstruction failure. Against Table 1 Raw, clean/attack gaps are -0.786435 / -1.323648 pp. Attack Table 4 quality gaps are +0.009822 / +0.345132 / -0.797623 MOS. These are descriptive comparisons across independent artifacts/subsets/configs/decoders/scorers, not exact reproduction or statistical equivalence.

All mean attack quality scores pass >2.5. Individually 747/999 pass all three; 58 / 15 / 209 have DNSMOS/NISQA/UTMOS <=2.5. No test-based tuning occurred. Final feature radii range 0.124354–0.191633, median 0.166802, and none reach rho=0.5. Logged pre-update NLL increases for all 999 clips (means 1.067866 to 6.271019); no final post-update NLL was measured. Average recorded attack time is 13.606 s/clip / 3.776 hours total; full job wall time 4.971 hours on TITAN RTX is not an L20 GPU-hour measurement. Native vocoder WAVs are 80–321 samples shorter than clean, with no duration warping.

Evidence and figures:

- `/workspace/reproduction-runs/english-feature-test-999/verification.json`.
- `/workspace/reproduction-runs/english-feature-test-999/paper-clean-comparison.json` and `paper-feature-comparison.json` from existing comparison script.
- `/workspace/reproduction-runs/english-feature-test-999/paper-comparison.json` includes Table 1/4 gaps and source hashes.
- `/workspace/reproduction-runs/english-feature-test-999/report.md`, expanded by `/workspace/scripts/report_verified_test.py`.
- `/workspace/reproduction-runs/english-feature-test-999/figures/test-controls.png`, `attack-traces.png`, and `provenance.json`. Figures were visually inspected.
- Generation fingerprint `020f4b6423e61ff82b87ea466622ad62bdb34b9beb0f55bf5a0bdb3fcb6173da`; generated JSONL SHA256 `0ab93a117f7b5f7ec4a9dfded41794d8adc4c843b836205e85cb00290033dbf5`.

## Completed and verified raw Whisper transfer

The four-model raw transfer plan is frozen at `/workspace/reproduction-runs/english-transfer-999/plan.json` (read-only). It reuses the exact 999 source-run WAVs for **tiny, base, medium, large-v2**, with clean/reconstruction/adversarial controls. The existing independent large-v2 choice in `/workspace/configs/targets/whisper_large.yaml` is preserved; author large revision is unspecified. Tiny/base/medium target YAMLs were added. All target configs are read-only and pinned by the plan, with forced English, beam size 5 and temperature 0. Published checkpoint hashes/URLs from installed Whisper 20240930 are frozen before evaluation. Quality is audio-dependent; the verified core quality is reused, so transfer does not rerun quality scorers. No audio generation or attack tuning occurs.

`/workspace/scripts/run_whisper_transfer.py` runs existing download/evaluate scripts sequentially in separate processes, exposes download/tqdm progress and 30-second heartbeats, validates config/checkpoint/generated-index hashes, and calls the auditor on every target's 2,997 score rows. Evaluation itself rehashes every saved WAV. A completed target can be skipped only after validation; an incomplete evaluation restarts in full. Failures stop the chain. Cumulative `results.json` and `report.md` are updated after each completed target. Compilation and transfer preflight passed.

Detached tmux session **`english-transfer-999`** completed with exit code 0 at `2026-10-10T11:39:16Z` (19:39:16 Asia/Taipei), after launching at `08:46:45Z`. A fresh combined audit recomputed all **11,988 scored rows**, checked all four pinned target configs/checkpoints, and rehashed the shared 999 original FLACs / 2,997 saved WAVs once. Per-target evidence hashes match cumulative results. No model scoring was rerun during this audit.

| Target | Clean WER % | Reconstruction WER % | Attack WER % |
|---|---:|---:|---:|
| tiny | 7.651403 | 7.523387 | 67.897587 |
| base | 5.568685 | 5.657312 | 68.709995 |
| medium | 3.018218 | 3.077302 | 62.427376 |
| large-v2 | 4.140817 | 3.431807 | 65.539143 |

Every cell covers 999 clips / 20,310 reference words. Independently chosen large-v2 attack WER is **+12.579143 percentage points** above the paper's raw large anchor 52.96%; the paper's exact large revision, subset and decoder remain unspecified. Figure 3's underlying tiny/base/medium arrays are unavailable. The claimed +26.6 pp improvement cannot be assessed without comparable baselines and defenses. Do not infer exact reproduction or invent those results.

- Completed log: `/workspace/reproduction-runs/english-transfer-999.log`.
- Overall status: `/workspace/reproduction-runs/english-transfer-999/job.json`.
- Phase/target status: `/workspace/reproduction-runs/english-transfer-999/progress.json`.
- Frozen plan: `/workspace/reproduction-runs/english-transfer-999/plan.json`.
- Per-target outputs: `/workspace/reproduction-runs/english-transfer-999/{tiny,base,medium,large-v2}/summary.json`, `utterances.jsonl`, `verification.json`.
- Fresh combined audit: `/workspace/reproduction-runs/english-transfer-999/verification.json`.
- Final results: `/workspace/reproduction-runs/english-transfer-999/results.json`.
- Expanded verified report: `/workspace/reproduction-runs/english-transfer-999/report.md`, produced by `/workspace/scripts/report_whisper_transfer.py`.
- Visually inspected plot: `/workspace/reproduction-runs/english-transfer-999/figures/whisper-transfer.png`; figure source hashes: `figures/provenance.json`.

## Completed and verified raw CTC transfer

The completed frozen plan is `/workspace/reproduction-runs/english-ctc-transfer-999/plan.json` (0444; SHA256 `30784cb5d5566f34a1ce6b0182418eb85b99ecc4fb0b3db17b6a44d96e86a002`). It reused the same 999 source-run clips and all three clean/reconstruction/adversarial controls, with no generation, attack tuning, defenses, or repeated quality scoring.

The existing independent choices from `configs/targets/ctc_template.yaml` are preserved: **facebook/wav2vec2-large-960h-lv60-self** at revision `54074b1c16f4de6a5ad59affb4caa8f2ea03a119`, and **facebook/hubert-large-ls960-ft** at revision `ece5fabbf034c1073acae96d5401b25be96709d8`. Native argmax/CTC collapse uses greedy decoding without an external language model. Author CTC IDs/revisions/decoder recipes are unspecified. Target YAMLs are resolved to local snapshot paths after verified download, then frozen before evaluation.

`scripts/download_ctc.py` now verifies cached or downloaded weights against upstream LFS SHA256 and JSON/tokenizer files against upstream Git blob SHA1, plus file sizes. The frozen plan pins that upstream metadata before download. Each target gets a SHA256 receipt and read-only config/receipt/target-freeze record. Equal receipts survive resume without being overwritten; incompatible existing files fail. The existing CTC recognition backend is unchanged. `scripts/verify_test_run.py` now accepts raw CTC targets only with immutable local revisions, hashed artifacts and declared greedy decoding. Seven focused artifact, pipeline and logging tests passed, including tamper rejection and read-only receipt resume; compilation and CTC preflight passed.

`scripts/run_ctc_transfer.py` runs download and three-control evaluation for each target in separate processes, checks the frozen source and target evidence, audits each target's 2,997 score rows and writes cumulative results/report. Existing complete evaluations are audited before reuse; interrupted evaluations restart in full. It exposes download/clip progress and 30-second phase heartbeats and stops on failure.

Detached session **`english-ctc-transfer-999`** completed with exit code 0 at `2026-10-10T14:09:37Z` (22:09:37 Asia/Taipei), after launching at 14:02:25Z. A fresh post-completion audit checked all **5,994 score rows**, both target config/receipt/target-freeze hashes, upstream weight LFS SHA256 and config/tokenizer Git blob SHA1/size, frozen source evidence and corpus aggregates. It rehashed the shared **999 original FLACs / 2,997 saved WAVs** once. Existing per-target verification evidence was preserved. No model scoring was rerun.

| Target | Clean WER % | Reconstruction WER % | Attack WER % | Paper raw attack % | Gap (pp) |
|---|---:|---:|---:|---:|---:|
| Wav2Vec2 | 1.930084 | 1.994092 | 44.544559 | 42.12 | +2.424559 |
| HuBERT | 2.166420 | 2.255047 | 48.207779 | 39.80 | +8.407779 |

Each cell covers 999 clips / 20,310 reference words. These are descriptive gaps across the documented independent CTC checkpoints/greedy decoder, test IDs, selected attack config and other implementation choices. Exact author artifacts/decoders are unspecified; no defended CTC result or exact Figure 4 curve is claimed. The feature attack remains frozen.

- Completed log: `/workspace/reproduction-runs/english-ctc-transfer-999.log`.
- Overall status: `/workspace/reproduction-runs/english-ctc-transfer-999/job.json`.
- Target/phase/heartbeat status: `/workspace/reproduction-runs/english-ctc-transfer-999/progress.json`.
- Frozen plan: `/workspace/reproduction-runs/english-ctc-transfer-999/plan.json`.
- Frozen target configs: `/workspace/configs/targets/wav2vec2_large.yaml`, `/workspace/configs/targets/hubert_large.yaml`.
- Per-target evidence: `/workspace/reproduction-runs/english-ctc-transfer-999/{wav2vec2,hubert}/summary.json`, `utterances.jsonl`, `verification.json`, `target-freeze.json`.
- Fresh combined audit: `/workspace/reproduction-runs/english-ctc-transfer-999/verification.json`.
- Final results: `/workspace/reproduction-runs/english-ctc-transfer-999/results.json`.
- Expanded report: `/workspace/reproduction-runs/english-ctc-transfer-999/report.md`, generated by `scripts/report_ctc_transfer.py`.
- Visually inspected figure: `/workspace/reproduction-runs/english-ctc-transfer-999/figures/ctc-transfer.png`; figure source hashes: `figures/provenance.json`.

Historical CTC command (completed; do not relaunch):

```bash
tmux -S /workspace/reproduction-runs/tmux.sock new-session -d -s english-ctc-transfer-999 -c /workspace '/opt/venv/bin/python -u /workspace/scripts/run_logged.py --log /workspace/reproduction-runs/english-ctc-transfer-999.log --status /workspace/reproduction-runs/english-ctc-transfer-999/job.json -- /opt/venv/bin/python -u /workspace/scripts/run_ctc_transfer.py --plan /workspace/reproduction-runs/english-ctc-transfer-999/plan.json'
```

## Completed and verified loss-removal ablations

Frozen plan: `/workspace/reproduction-runs/english-ablations-999/plan.json`, 0444, SHA256 `04a294828afcbf5b4599604e3bcb5e22feab085695d141de764d8845772da0f9`. The existing `experiment_configs(..., "ablations")` defines full / without-attack / without-perceptual / without-tv / without-hf. The full condition reuses the verified English core run; the four removals generate new audio on the **same 999 test IDs**, each with clean/reconstruction/perturbed controls, followed by raw Whisper-small evaluation and all three quality metrics. No selector is applied and quality failures are reported rather than excluding a condition.

Each read-only YAML/receipt under `english-ablations-999/frozen/` derives from the untouched `/workspace/configs/english_selected.freeze.json`. Only its prescribed loss switch changes: attack=false, lambda_perc=0, temporal=false, or high_frequency=false. All other fields remain exact, including selected rho=0.5, baseline lambda=0.1, Adam LR=0.05, 50 updates, layer/vocoder/normalization, model/scorer artifacts, decoder, STFT, per-utterance seeds and original test manifest. These are ablations of the independent selected config, not a new validation selection or exact Figure 8 reproduction at paper defaults rho=0.1/lambda=1.

Without attack loss runs 50 perceptual-only updates; it is not zero-delta reconstruction. The existing implementation logs zero NLL when its computation is disabled; those zeros are placeholders, not measured NLL. Disabled TV/HF also log zero placeholders. Hold-fixed learning rate can change effective optimization balance after loss removal. Figure 8's full-loss DNSMOS conflicts with Table 4, and author exact ablation arrays/settings/scorers remain unavailable.

`scripts/ablation_protocol.py` requires an exact predefined loss removal and rejects extra changes even with updated valid hashes. `run_selected_test.py` accepts an optional ablation receipt, preserving the original selection freeze and existing generation/evaluation behavior. `verify_test_run.py` additionally checks removed-term zero placeholders and reconstructs the retained scalar objective, besides its existing 999-ID/control/trace/quality/hash audit. `scripts/run_ablation_suite.py` sequentially launches each generation/evaluation pipeline in separate processes, shows clip progress and 30-second parent/child heartbeats, checks exact decoded full-loss control sample hashes match, audits every completed removal with full WAV rehashing, and writes cumulative results/report. Completed evaluations are audited before reuse; interrupted generation uses existing checked resume, incomplete evaluation restarts in full; failures stop the chain.

Thirty-two focused protocol/attack/math/pipeline/logging tests passed, including rejection of extra LR changes and changed parent freezes, live perceptual-only optimization and loss/gradient/projection arithmetic. Compilation, suite preflight, and child-pipeline frozen-input preflight passed. Detached session **`english-ablations-999`** originally launched at `2026-10-10T14:36:20Z`. Independently exported WAV hashes differed only in libsndfile PEAK timestamps; exact samples matched. The comparison was corrected to decoded mono float32 sample SHA256 while preserving each WAV's own file-hash checks; tests reject actual sample/rate changes. The initial supervisor was deliberately interrupted after 31 clips and the same frozen plan resumed at `14:41:43Z` (22:41:43 Asia/Taipei). It completed with **exit code 0** at `2026-10-11T08:21:33Z` (16:21:33 Asia/Taipei). Original source audio/config and ablation receipts were preserved.

`scripts/verify_ablation_suite.py` passed the fresh completion audit: all **14,985 score rows** and **249,750 updates** checked, all **11,988 WAVs** from four new runs rehashed, 999 unique source FLACs checked (repeated for each new run), all five conditions' exact clean/reconstruction sample hashes matched. Full loss received a fresh metadata/control-sample audit against preserved evidence. Original per-condition verification files were not overwritten; ASR/quality inference was not rerun.

| Condition | WER % | DNSMOS | NISQA | UTMOS |
|---|---:|---:|---:|---:|
| Full loss | 74.106352 | 2.963822 | 3.871132 | 2.792377 |
| Without attack | 4.155588 | 3.065396 | 3.478466 | 3.224561 |
| Without perceptual | 75.371738 | 2.970376 | 3.940159 | 2.800914 |
| Without TV | 75.396356 | 2.959288 | 3.850383 | 2.789307 |
| Without HF | 73.382570 | 2.968427 | 3.938878 | 2.800615 |

Every condition covers 999 clips / 20,310 reference words; clean/reconstruction WER and all quality means match exactly across conditions. All mean perturbed-audio MOS gates pass. Attack loss drives the WER increase, but the paper's qualitative benefit of every penalty was **not recovered uniformly**: removing all penalties improves all three point-mean MOS and increases WER by +1.265387 pp. Do not retune or replace the selected config on this test evidence. Perceptual-only WER differs by -2.444412 pp from the paper's explicit 6.6% anchor; other removal values are plot approximations, not exact anchors.

`scripts/report_ablation_suite.py` added 5,000 paired utterance bootstrap resamples (NumPy seed 2026), corpus-WER recomputation with resampled word denominators and utterance-mean MOS. WER differences from full / 95% intervals: without attack -69.950763 [-72.187941,-68.065250] pp; without perceptual +1.265387 [-1.313864,+3.758960]; without TV +1.290005 [+0.069162,+2.729524]; without HF -0.723781 [-3.147237,+1.350249]. These intervals do not capture attack-seed or speaker-cluster variation and are independent diagnostics, not recovered paper procedures. Hold-fixed LR, different selected rho/lambda and missing author/scorer details limit attribution/equivalence.

- Completed log: `/workspace/reproduction-runs/english-ablations-999.log`.
- Overall status: `/workspace/reproduction-runs/english-ablations-999/job.json`.
- Suite experiment/phase/heartbeat status: `/workspace/reproduction-runs/english-ablations-999/progress.json`.
- Per-removal phase/count status: `/workspace/reproduction-runs/english-ablations-999/{without-attack,without-perceptual,without-tv,without-hf}/progress.json`.
- Frozen plan and exact configs/receipts: `/workspace/reproduction-runs/english-ablations-999/plan.json`, `frozen/`.
- Cumulative/final outputs: `/workspace/reproduction-runs/english-ablations-999/results.json`, `report.md`.
- Per-removal outputs: corresponding `run.json`, `generated.jsonl`, `records/`, `audio/`, `surrogate-eval/summary.json`, `surrogate-eval/utterances.jsonl`, `verification.json`.
- Full-loss reuse audit: `/workspace/reproduction-runs/english-ablations-999/full-verification.json`; original core evidence is not overwritten.
- Fresh combined audit: `/workspace/reproduction-runs/english-ablations-999/verification.json`.
- Expanded report/intervals/paper comparison: `/workspace/reproduction-runs/english-ablations-999/report.md`, `paired-bootstrap.json`, `paper-comparison.json`.
- Visually inspected plot: `/workspace/reproduction-runs/english-ablations-999/figures/loss-removals.png`; source hashes/values: `figures/provenance.json`.

Historical ablation command (completed; do not relaunch):

```bash
tmux -S /workspace/reproduction-runs/tmux.sock new-session -d -s english-ablations-999 -c /workspace '/opt/venv/bin/python -u /workspace/scripts/run_logged.py --log /workspace/reproduction-runs/english-ablations-999.log --status /workspace/reproduction-runs/english-ablations-999/job.json -- /opt/venv/bin/python -u /workspace/scripts/run_ablation_suite.py --plan /workspace/reproduction-runs/english-ablations-999/plan.json'
```

## Latest state: waveform baseline validation ACTIVE

The existing exact `waveform-grid` is frozen at `/workspace/reproduction-runs/english-waveform-validation/plan.json`, 0444, SHA256 `46a9bf2843d1c72707a6fd9dbcb82e96ae54f8cf39a27291a75a01cc1fa5f37c`. It contains **135 settings**: PGD 15, MI-FGSM 30, VMI-FGSM 90. Epsilon {0.001,0.002,0.005,0.01,0.02}; step size {epsilon/50,2*epsilon/50,epsilon/10}; MI momentum {0.5,1}; VMI momentum 1 / neighbor count {3,5} / beta {1,1.5,2}. All use 50 updates and the same previously audited **100 dev-clean validation clips**. Configs under `frozen/` are read-only and match the existing grid function exactly.

This preserves the documented independent waveform implementation: maximize native Whisper-small lexical-mean teacher-forced NLL, L-infinity projection about original audio, [-1,1] clipping, zero start, sign updates, existing momentum and variance-neighbor rules. Model/frontend/decoder/scorers and per-utterance seeds are unchanged. Baseline generation does not load encoder/vocoder models. The plan pins model/scorer/manifests and six existing computational source files by hash; no attack/model/evaluation implementation was changed. Feature-specific fields remain in copied configs as unused provenance. Paper norm/loss/start/decoder/full baseline recipes are absent; VMI adds gradient work, so equal update count does not imply equal compute. Muting Whisper/Sloth/defenses are not invented.

`scripts/run_waveform_validation.py` runs separate generation/evaluation processes sequentially. Each run saves clean and adversarial WAVs; validation evaluates adversarial only with all quality metrics, matching the existing grid protocol. `scripts/verify_waveform_run.py` checks 100 validation IDs/roles, frozen config/model/scorer provenance, full JSONL/source/WAV hashes, exact clean-source samples, WAV headers/lengths, actual saved-sample L-infinity bound/range, all 5,000 finite bounded updates, references/edit counts, corpus WER and MOS means before accepting a candidate. The overall job shows clip progress and 30-second setting/phase heartbeats and stops on error. Incomplete generation resumes through saved-hash checks; incomplete evaluation restarts in full; complete candidates are audited before reuse.

After each method finishes, the **existing strict quality-gated selector** separately maximizes validation corpus WER subject to DNSMOS/NISQA/UTMOS each **strictly >2.5**. It writes method selection audits and freezes the full winning config/receipt under `selected/`. If no candidate qualifies, it reports `no_eligible_configuration` without fallback and continues the other methods; unrelated selection errors stop the job. No test scores or target feedback are used. A 999-clip baseline test will follow selection on a later return.

Compilation, immutable-input/split/grid preflight and **29 focused waveform/attack/data/selection/logging/pipeline tests** passed; six targeted waveform/selection tests were rerun after the final audit tightening. Tests reject actual saved-sample bound/range/duration violations, enforce strict gates without fallback, and ensure a stale no-eligible audit cannot mask test-data rejection. Detached session **`english-waveform-validation`** launched at `2026-10-11T09:20:07Z` (17:20:07 Asia/Taipei), using `run_logged.py` on `/workspace/reproduction-runs/tmux.sock`. Supervisor 702633/orchestrator 702636 were confirmed running; inspect current log for actual clip progress.

- Live log: `/workspace/reproduction-runs/english-waveform-validation.log`.
- Overall status: `/workspace/reproduction-runs/english-waveform-validation/job.json`.
- Setting/method/phase/heartbeat status: `/workspace/reproduction-runs/english-waveform-validation/progress.json`.
- Frozen plan/configs: `/workspace/reproduction-runs/english-waveform-validation/plan.json`, `frozen/`.
- Cumulative/final results/report: `/workspace/reproduction-runs/english-waveform-validation/results.json`, `report.md`.
- Per-setting outputs: named run subdirectories with `run.json`, `generated.jsonl`, `records/`, `audio/`, `evaluation/summary.json`, `evaluation/utterances.jsonl`, `verification.json`.
- Eventual method selections: `/workspace/reproduction-runs/english-waveform-validation/selected/{pgd,mi-fgsm,vmi-fgsm}.selection.json`, `.yaml`, `.freeze.json` (written only after each method completes; no winner claimed at launch).

Launch/resume command (check status/process/tmux first; no concurrent duplicate):

```bash
tmux -S /workspace/reproduction-runs/tmux.sock new-session -d -s english-waveform-validation -c /workspace '/opt/venv/bin/python -u /workspace/scripts/run_logged.py --log /workspace/reproduction-runs/english-waveform-validation.log --status /workspace/reproduction-runs/english-waveform-validation/job.json -- /opt/venv/bin/python -u /workspace/scripts/run_waveform_validation.py --plan /workspace/reproduction-runs/english-waveform-validation/plan.json'
```

## Next return: inspect and continue

Latest parallel-work update (2026-10-12): the original waveform search is preserved and running; 58/135 candidates are audited, PGD and MI-FGSM have frozen validation winners, and VMI-FGSM is pending. User authorized one or two additional scripts where resources permit, with four priorities: complete/audit all 135 settings; test eligible baseline winners on the 999 held-out clips and reuse saved WAVs for transfer; investigate quality/ablation differences on validation or new disjoint data, with a new confirmatory evaluation for any new tuning; continue Mandarin training/validation/held-out CER. Baseline test generation and additional English diagnostics were NOT launched in this turn and remain pending.

GPU memory is mostly free (~2.1 GiB / 24 GiB), but a 20-second utilization sample averaged 98.05% (98–99% range). Resource evidence is `/workspace/reproduction-runs/parallel-resource-sample.json`. A five-clip PGD hardware probe on the disjoint pilot completed at `/workspace/reproduction-runs/parallel-gpu-probe`; it fit in memory but showed contention (probe ~14.3 s/clip; one overlapping VMI clip ~57.5 s versus nearby ~27 s). This is a small performance probe, not a selection experiment. No existing setting was changed. `scripts/parallel_jobs.py` provides supervision and a conservative auxiliary GPU admission helper; the GPU admission helper is not currently used by an active baseline worker.

One additional production job IS active: detached session `mandarin-preparation-training`, through `scripts/run_logged.py` and `scripts/run_mandarin_preparation_training.py`, launched at 2026-10-12 09:03:23 UTC (17:03:23 Asia/Taipei). It prepared/froze AISHELL data and runs the existing independent three-epoch Whisper-small AdamW recipe on CPU with eight threads, alongside the unchanged GPU search. Preparation/audits passed: 32 development speakers / 11,479 training clips; four separate checkpoint-validation speakers / 1,435 clips; four separate attack-reserve speakers / 1,412 clips, with a seeded 100-clip attack-validation subset; all 20 official test speakers / 7,176 test clips. IDs, audio and speakers are disjoint across the used pools. A separate attack reserve prevents checkpoint-selection and attack-selection reuse. `data-freeze.json` pins manifests, recipe, checkpoint and training/frontend source hashes. Three focused speaker-isolation/GPU-admission tests and compilation passed.

Mandarin log: `/workspace/reproduction-runs/mandarin-independent.log`; overall status: `/workspace/reproduction-runs/mandarin-independent/job.json`; phase/heartbeat status: `/workspace/reproduction-runs/mandarin-independent/progress.json`; frozen data: `/workspace/reproduction-runs/mandarin-independent/data-freeze.json`; manifests: `manifests/`; training progress/checkpoints: `training/training.json`, `training/history.json`, `training/{best,last,optimizer}.pt`. Actual CPU fine-tuning was confirmed at 228/11,479 clips in epoch one. It shows clip progress and 30-second phase heartbeats. On success the wrapper freezes the best validation-loss checkpoint in `surrogate-freeze.json`. This is checkpoint selection, not attack-config selection. Mandarin control evaluation, 16-setting attack validation, strict CER/MOS selection and held-out test CER still follow. Original trainer deliberately refuses automatic restart into incomplete outputs; investigate/preserve checkpoints before resuming an interrupted training phase. Do not claim Mandarin CER results yet.

The user asked to end the chat turn instead of waiting for GPU completion. Both detached production jobs were confirmed active and left running; continue when the user returns.

1. Read this handoff and the reproduction guide. Inspect waveform-validation `job.json`, `progress.json`, tmux/process and log. If running, report meaningful setting/clip progress and yield; do not relaunch or retune. Session disappearance is not a completion test.
2. If failed/interrupted, inspect and resume this same frozen plan with detached tmux plus `run_logged.py`. Complete candidates are audited before reuse; unfinished generation resumes through saved-hash checks and incomplete evaluation restarts in full. Preserve the 135-setting grid and computational source/model/scorer identities.
3. On completion, independently audit all 135 runs, common 100 validation IDs/raw surrogate/scorer protocol, actual saved waveform bounds, references/lexical/corpus/MOS aggregates and method-specific selector eligibility/maxima. Check all winning configs/receipts exactly match their validation source runs; report no-eligible methods honestly without fallback.
4. Freeze and launch generation/evaluation on the 999-clip English test manifest for the selected eligible baseline methods, with matched original-clean controls and saved-audio reuse for transfer. Compare only supplied exact paper anchors and label these independent NLL/L-infinity adaptations. Never choose baseline settings on test/target scores.
5. Continue wider reproduction: AISHELL speaker-disjoint preparation and Mandarin surrogate fine-tuning/evaluation. Exact defense/human/OTA recreation requires original artifacts or new data.
