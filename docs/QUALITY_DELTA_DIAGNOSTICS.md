# Feature perturbation structure diagnostics

19 October 2026, Asia/Taipei. Current study: `/workspace/reproduction-runs/feature-delta-diagnostics`. [Live status](/workspace/reproduction-runs/JOB_STATUS.md), [results](/workspace/reproduction-runs/feature-delta-diagnostics/report.md), and [overnight review](/workspace/reproduction-runs/overnight-review-20261019/report.md). Discord remains disabled.

The overnight queue completed all three studies without operational failures, covering 20 pilot and four validation score sets. All **257 frozen inputs**, 880 condition-specific clip/delta records, 2,640 score rows/WAV hashes, edit counts, MOS means, 44,000 finite updates and schedule traces passed review. Existing independently recomputed WavLM radius audits were checked by hash; the CPU review did not rerun those encoder computations.

No setting met both WER and quality targets. The strongest new validation result was timestamp sum-NLL/LR 0.05: **71.0285% WER**, DNSMOS/NISQA/UTMOS **3.1537/4.0233/3.3713**. The quality-oriented timestamp setting and matched control stayed near **51.5–52.0% WER / 3.589 UTMOS**. No confirmation or new 999-clip test was launched. The earlier regular/raw sum-NLL/LR 0.1 validation point remains **75.5601% WER / 3.2457 UTMOS**.

The apparent 91.79% cosine pilot result includes a 13-word clip producing 102 insertions and six substitutions. Those 108 edits account for 24.77% of its pilot edit total. A diagnostic leave-one-out value is 71.00%; **the official score is unchanged**. Paired bootstrap intervals for the timestamp/control comparison are broad and do not establish a clear improvement. See the review for denominators and uncertainty.

## Question and controlled comparisons

Reconstruction quality has room: validation clean/reconstruction UTMOS is **4.0534/4.0144**, while the strong attack falls to **3.2457**. The paper's temporal penalty measures TV(q+delta), which can stay near the clean value despite a rough delta. Saved-pilot descriptive correlations suggested testing temporal structure, but did not establish causation or a formula bug.

The first diagnostic reuses audited strong constant-Adam deltas on the same fresh 24-clip pilot. Source deltas came from **50 updates, rho=0.1, lambda=1, alpha=1, 6000 Hz cutoff**. It evaluates identity and three comparisons for each Gaussian width 3/5/9 feature frames:

| Case | Direction | Norm comparison |
|---|---|---|
| Identity | Original delta | Original norm |
| Smooth | Temporally smoothed delta | Its resulting norm |
| Attenuation matched | Original direction | Same norm as unscaled smoothing |
| Smooth, norm matched | Smoothed direction | Restored original norm |

Gaussian sigma is `(window-1)/4`, with replicate boundaries and independent feature channels. Only delta is smoothed; q remains unchanged. Every derived delta is independently projected/checked against the original feature ball, since replicate padding is not assumed to be a Frobenius contraction. Widths correspond approximately to 60/100/180 ms at the selected encoder frame rate. Amplitude is not normalized in waveform space.

This gives **ten pilot conditions**, including the original cached source control. Comparing smooth against matched attenuation separates directional smoothing from reduced effective budget. Comparing norm-restored smoothing against identity compares directions at the same budget. These are **post-optimization diagnostics with zero new optimization updates**, not a newly optimized 50-step attack or recovered author algorithm.

A fixed exploratory screen promotes at most two smoothing directions with at least **0.05 UTMOS gain**, no more than **five WER percentage points lost**, and all three MOS >2.5 versus their matched control. They and their matched controls then use the original 100-clip validation source deltas. Selection uses pilot results only. All conditions are reported; no 999-clip evaluation follows automatically. A successful direction would motivate a separately frozen 50-update optimizer study rather than silently being called the original algorithm.

The full diagnostic completed successfully on 19 October at **04:38:41 UTC**. Width-3 attenuation scored **67.7895% WER / 3.4762 UTMOS**, while smoothing at the matched norm scored **65.4737% / 3.4301** (identity: 72.6316% / 3.3035). At the original norm, smoothing widths 3/5/9 scored **70.9474/54.7368/30.9474% WER** and **3.2679/3.2564/3.2754 UTMOS**. Every width reduced WER and slightly worsened UTMOS relative to identity. No smoothing direction passed the fixed promotion rule, so validation and test expansion were skipped. These results do not support implementing a smoothed optimizer on the strength of this diagnostic.

Independent review checked all **397 frozen inputs**, ten score sets with **240 adversarial rows**, and **216 new deltas**. Float64 CPU reconstruction of every transform agreed with saved GPU tensors within 5.16e-7 absolute difference; maximum independently recomputed relative radius was 0.100000010. Existing GPU resynthesis receipts report exact saved-waveform reproduction. See [the review](/workspace/reproduction-runs/feature-delta-review-20261019/report.md); its CPU review checks those receipts rather than rerunning GPU synthesis.

## Mechanistic gradient probe

On the first three pilot entries in frozen manifest order, a second diagnostic evaluates the unchanged objective at zero and saved delta. It reports gradients of **-NLL**, **lambda TV**, and **lambda alpha HF**, their norms/cosines and their independently differentiated total. No model weights, q, decoder or loss coefficients are trained or tuned. Zero-vector cosines and zero-energy normalized-delta measures are reported as undefined.

The initial real probe completed successfully. At the saved deltas, ASR gradient norms were **3.14–5.72**, weighted temporal norms **0.00137–0.00341**, and weighted HF norms **0.0232–0.0347**. This is a large local gradient imbalance in the sum-NLL parent objective. It does not prove a causal account of MOS degradation, describe every optimization step, or directly predict Adam-preconditioned updates. Exact values, components and provenance are in [gradient-probe.json](/workspace/reproduction-runs/feature-delta-diagnostics/gradient-probe.json).

## Evidence and operation

Preparation preserved the completed overnight plan/results/report and inherited all frozen model/scorer/runtime sources. The new plan contains **397 input hashes**, including every source delta tensor. No existing attack/scoring source or completed result was changed; only new diagnostic utilities/drivers and tests were added. All **221 tests passed**, covering temporal filtering, norm/direction controls, zero/tiny inputs, boundary behavior, scalar/component-gradient consistency and promotion logic.

Before any transformation, the parent checker validates source audio/WAV/index/score hashes, references/edit/MOS aggregates, artifacts, identities and original 50-step traces. It re-extracts q, independently checks actual parent radii and reproduces saved parent waveforms. On all 24 pilot clips, the maximum waveform difference was **0.0**, with maximum parent radius **0.100000009**. Cached features have explicit IDs, feature metadata and tensor hashes, bound to parent-delta hashes.

Each transformation saves a new delta hash and float WAV, records zero new steps, and evaluates fixed raw Whisper and all three MOS models. Verification recomputes the transform from source tensors, checks actual radius, independently re-vocodes the saved delta, compares waveform samples and rechecks lexical/MOS aggregates. Old clean/reconstruction controls remain unchanged. The original source-control scores are labeled cached.

`scripts/run_feature_diagnostics.py` supervises parent checks, the component-gradient probe, synthesis/scoring/audits, and conditional validation. Each stage has a log/job record. `research.log`, `progress.json`, cumulative `results.json` and `report.md` show actual progress; the consolidated updater tracks `--research feature-delta-diagnostics` every 30 seconds. The environment retains UTF-8, four OMP/MKL threads, one OpenBLAS thread, `PYTHONPATH=/workspace/artifacts/defense-runtime:/workspace`, and `BEYONDWAVEFORM_NOTIFY_DISCORD=0`. Inspect current job records/PIDs before starting another copy.
