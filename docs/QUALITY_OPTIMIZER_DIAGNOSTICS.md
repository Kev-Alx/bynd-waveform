# Numerical repeatability and Adam contribution diagnostics

19 October 2026. This follows the completed last-ten-iterate averaging study. Both predefined directions missed its 0.05 UTMOS gain threshold against norm-matched terminal controls: gains were **0.003445** and **0.026580**. No fresh confirmation data were scored. The averaging-window family is stopped; no additional window search is planned.

The fresh no-timestamps/LR=0.1 terminal run reached **78.5263% WER / 3.3142 UTMOS**, versus **72.6316% / 3.3035** in its earlier same-configuration pilot. The timestamp/LR=0.05 terminal run also changed. These observations do not establish the cause of variability. They motivate checking numerical repeatability before interpreting smaller differences as implementation gains. Official previous scores remain unchanged.

## Fixed-point repeatability

Use the first three IDs in the existing frozen 24-clip pilot order, not clips selected for favorable outcomes. Bind the cached clean features, original audio, saved parent deltas, feature/vocoder/Whisper artifacts and source/runtime hashes. At zero delta and the saved strong parent delta, compute the unchanged full-objective gradient three times from fresh leaves with the same per-clip seed. No optimizer update, recognition, MOS inference, parameter training or new validation/test use occurs.

Run each execution policy in a separate process:

- **Auto:** the current default attention policy, with the existing seeding, cuDNN deterministic flag and TF32 disabled.
- **Math attention:** disable the fused efficient/flash/cuDNN SDPA backends, leaving math SDPA. This is an execution-policy sensitivity diagnostic, not a claim about the authors' backend.
- **Strict math:** additionally enable strict deterministic algorithms and configure cuBLAS workspace before CUDA initialization.

Hash raw gradient values with dtype/shape, save the actual tensors and report pairwise maximum/relative differences, norms and losses. Exact equality is the predefined criterion for bitwise repeatability; a mismatch is a diagnostic result, not an infrastructure failure. Record actual attention-kernel profiler names separately from the equality repeats. Fixed-point repeatability does not establish complete 50-update trajectory reproducibility.

Strict mode may reject an operation such as CUDA reflection-padding backward, which appears in the native Whisper frontend and the centered HF STFT. Capture only known deterministic-implementation errors as **unsupported**, retaining the error and completed evidence. Do not silently disable strict mode or substitute a different frontend. Unexpected errors, including out-of-memory failures, remain failures.

## Contribution to an Adam step

A second diagnostic runs one fresh, unchanged 50-update strong-parent trajectory on each of those same three development clips. Preserve rho=0.1, lambda=1, alpha=1, cutoff=6000 Hz, constant Adam LR=0.1, zero initialization, the original no-timestamps lexical sum-NLL and the original clean denominators.

At updates **1, 10, 25 and 50**, measure the component gradients and independently differentiated total. Clone the **same current delta and Adam moment/step state** into counterfactual branches using the full gradient, the ASR-only gradient, and total gradient with the temporal or HF gradient removed. Compare the unprojected update and the displacement after the original feature-ball projection. Apply only the full-objective update to the continuing trajectory; the other branches never continue or count as additional attack updates.

Save the pre-update delta, gradients, optimizer state and branch outputs. Check that the full-gradient clone exactly matches the actual next update, that probing does not mutate the original optimizer/parameter state, and that actual iterates remain finite and within the original radius. Report gradient-sum numerical discrepancy and projected-step differences explicitly.

These are local counterfactuals with moment history held fixed. They are **not** complete no-perceptual-loss ablations, estimates of WER/MOS, or evidence that a different loss weight is correct. No replacement coefficients, optimizer settings or attack configuration are selected automatically. The earlier raw gradient-norm imbalance alone did not measure Adam's actual contribution; this diagnostic addresses that limitation.

The sample is small and includes repeated speakers. Results describe the frozen development clips and execution environment. All earlier plans, results, scorer channels, references and test IDs remain unchanged. Logs, terminal status and progress heartbeat are retained; Discord remains disabled.
