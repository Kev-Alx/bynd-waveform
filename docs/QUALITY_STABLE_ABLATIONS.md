# Complete loss ablations under repeatable paper-budget execution

20 October 2026, Asia/Taipei. The three stable-execution settings completed normally. On the fixed 24-clip development pilot, the strong sum-NLL/LR=0.1 setting reached **74.7368% WER / 3.1640 DNSMOS / 4.0350 NISQA / 3.3186 UTMOS**. The quality-focused LR=0.02 setting reached **45.6842% / 3.2075 / 4.1245 / 3.6761**, and the timestamp/LR=0.05 setting reached **71.7895% / 3.1273 / 4.0778 / 3.4034**. None matches the paper's WER and all quality targets together. Numerical repeatability was established for the fixed first three clips; that is not author-code parity or a guarantee in other environments.

The original whole-loss ablations used the historical rho=0.5/lambda=0.1 configuration. The recent same-moment Adam probes measure only an alternative current gradient, not the effects of omitting a term throughout all 50 updates. Under the new stable full-objective strong setting, mean measured HF ratio falls from about 1.004 at zero delta to 0.432 before update 50 while UTMOS remains 3.319. This supports checking what the regularizer actually contributes across complete trajectories before replacing coefficients or time pooling.

The separate [saved-audio HF pooling diagnostic](/workspace/reproduction-runs/hf-pooling-saved-audio-20261020/report.md) checked 1,245 frozen inputs and all 216 saved WAV records. Global and equal-positive-frame HF both decreased on most adversarial clips; the opposite-direction case occurs on only 1/24 strong, 1/24 quality and 3/24 timestamp clips. The existing global implementation matches bitwise on every waveform. This does not demonstrate that an alternative pooling fixes the quality gap, so no per-frame attack is promoted. The loss-ablation study has higher priority.

Use the exact existing 24 development IDs, references, cached clean q, raw-input WavLM layer-6/regular HiFi-GAN contract, original Whisper-small weights/lexical sum NLL/no-timestamps prefix, constant Adam LR=0.1, zero initialization, 50 projected updates, rho=0.1, alpha=1, lambda=1 and fc=6000. Keep original-clean denominators and the validated explicit-reflection/strict-math CUDA generation policy. Run exactly:

| Diagnostic | TV | HF | ASR |
|---|---|---|---|
| Frozen full-objective control | On | On | On |
| without-tv | Off | On | On |
| without-hf | On | Off | On |
| without-perceptual | Off | Off | On |

The lambda value remains recorded as 1; disabled terms contribute exact zero. These are deliberate loss ablations. They cannot be promoted as reproductions of the full objective even if a numerical WER/MOS target is reached.

Before the ablations, the new flag-aware collector must reproduce all 150 projected updates, loss/gradient/moment hashes and final waveforms/optimizer values of the frozen full-objective control's first three clips. Run this in a fresh strict CUDA process with cuBLAS workspace `:4096:8`. A mismatch or unexpected error stops the queue. Do not silently relax settings or alter saved results.

Save every iterate and terminal optimizer receipt for all new trajectories. Reuse the identical frozen clean/reconstruction WAV assets, verify the vocoder reconstruction still matches, and evaluate all three WAV variants with the unchanged native decoder and fixed DNSMOS OVRL/NISQA overall/UTMOS22 strong scorers. Independently resynthesize final q+delta, check all 3,600 projected iterates, disabled-term zeros, input/model/scorer hashes, all edit counts and score aggregation. The completion report compares WER and all MOS channels against the full-objective control; the paper's Figure 8 loss-guidance claim specifically discusses WER and DNSMOS.

No hyperparameter grid, candidate selection, sample filtering, fresh48 confirmation or new 999-clip test is run. All previous sources/results stay frozen. The separate saved-audio HF time-pooling diagnostic is descriptive and does not change any of these attack losses. Status/heartbeat/logs persist, and Discord remains disabled.
