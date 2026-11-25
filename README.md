# BeyondWaveform reproduction

Modular Python implementation of the [supplied paper](BeyondWaveform.pdf), with the [reproduction specification](BeyondWaveform_Reproduction_Spec.md) as the uncertainty register. The attack equations are implemented in `beyondwaveform/losses.py` and `beyondwaveform/attacks.py`. Artifacts, model variants, and optimizer details absent from the paper are explicit independent choices, recorded with every run.

The container in `Dockerfile` already includes dependencies. From `/workspace`:

```bash
/opt/venv/bin/python -m pytest -q
/opt/venv/bin/python scripts/download_models.py --components feature whisper --whisper small
/opt/venv/bin/python scripts/download_data.py --corpus librispeech --splits dev-clean test-clean
/opt/venv/bin/python scripts/prepare_data.py --corpus librispeech --split dev-clean --role validation --count 100 --output data/manifests/english-validation.jsonl
/opt/venv/bin/python scripts/prepare_data.py --corpus librispeech --split test-clean --role test --count 999 --output data/manifests/english-test-999.jsonl
/opt/venv/bin/python scripts/audit_splits.py data/manifests/english-validation.jsonl data/manifests/english-test-999.jsonl
/opt/venv/bin/python scripts/generate.py --config configs/english.yaml --manifest data/manifests/english-test-999.jsonl --output runs/english-feature
/opt/venv/bin/python scripts/evaluate.py --run runs/english-feature --target surrogate --output runs/english-feature/surrogate-eval
```

The full 999-utterance attack is compute intensive. Run the included `scripts/real_smoke.py` on a separate `role=pilot` manifest first. `--quality` on `evaluate.py` loads DNSMOS, NISQA, and UTMOS for Table 4 and validation selection; UTMOS needs `download_models.py --components utmos` once. `scripts/run_experiments.py` prepares the full radius/weight, ablation, and waveform baseline grids. Long-running scripts display file, utterance, epoch, or experiment progress. See [docs/REPRODUCTION.md](docs/REPRODUCTION.md) for exact run order, assumptions, evaluation controls, and scope limits.

For the current-implementation defense baseline, see [docs/DEFENSE_RESEARCH.md](docs/DEFENSE_RESEARCH.md). `scripts/run_defense_suite.py` freezes and audits the original 999 clips, measures the existing independent LPF and source-backed WaveGuard resampling conditions, and reports unavailable paper defenses explicitly. Install the `defenses` optional dependency for WaveGuard. This suite does not change attack settings or claim complete Table 1 parity.

The completed [first WER and quality batch](docs/QUALITY_RESEARCH.md), [refinement](docs/QUALITY_REFINEMENT.md), [feature/vocoder study](docs/QUALITY_CONTRACTS.md), and [overnight queue](docs/QUALITY_OVERNIGHT.md) document the remaining joint quality/WER gap. The current [feature-delta diagnostics](docs/QUALITY_DELTA_DIAGNOSTICS.md) compare temporal smoothing with norm-matched controls and measure objective-component gradients using saved perturbations. Live progress is in [JOB_STATUS.md](/workspace/reproduction-runs/JOB_STATUS.md). Discord job notifications are disabled.

For a long job that must outlive the chat terminal, use a detached `tmux` session and `scripts/run_logged.py`. The wrapper appends the live progress to a log and writes a JSON job record with the exit code and `running`, `complete`, `failed`, or `interrupted` status. The validation grid currently uses the learning rate chosen on the disjoint pilot:

```bash
tmux -S /workspace/reproduction-runs/tmux.sock new-session -d -s validation-grid -c /workspace '/opt/venv/bin/python -u scripts/run_logged.py --log reproduction-runs/validation-grid.log --status reproduction-runs/validation-grid/job.json -- /opt/venv/bin/python -u scripts/run_experiments.py --config configs/english_lr005.yaml --suite feature-grid --output reproduction-runs/validation-grid --execute --manifest data/manifests/english-validation.jsonl --quality'
tmux -S /workspace/reproduction-runs/tmux.sock attach -t validation-grid
```

Run the launch command once. To detach from the live session, press Ctrl-b then d. Alternatively, inspect `reproduction-runs/validation-grid.log` and `reproduction-runs/validation-grid/job.json`. The session exits when its command finishes; a disappearing session alone is not a success signal. Use the job record's exit code and the 16 per-setting evaluation summaries. Generation resumes from verified saved clips when rerun with the same config and manifest.
