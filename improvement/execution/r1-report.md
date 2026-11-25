# R1 canonical mechanism result

**Status: negative feasibility result; the registered 24-clip condition stopped at its first clip before any optimization update.** The K=4 native beam set contained four distinct token paths, but all four normalized to the correct words. Adding the required reference path also supplied cost zero. Expected bounded word risk was therefore identically zero for this fixed candidate set. No coefficient, seed, candidate repair, replacement clip or wider beam was tried.

This is a failure of the canonical candidate construction on ordinary development data. It does not establish that every decoding-aware objective fails, or provide a transfer comparison. No confirmation/final utterance or held-out ASR target was scored.

## Implemented and verified

The isolated implementation uses one native Whisper-small beam search for four candidates, appends the correct transcript when distinct, and recomputes differentiable unfiltered lexical-plus-EOS path probabilities divided by lexical-token length plus one. Hypotheses and bounded word-edit costs are detached. The empty lexical path, when present, receives the real EOS probability with divisor one. Candidate refresh occurs every five updates. The library also provides the separately labeled EOS-free nonempty-path ablation; that ablation was not executed.

Sixteen CPU tests passed, including central finite-difference risk derivatives, edit-count expectations, EOS/empty scoring, maximization sign, calibration guards, refresh cadence and a tiny synthetic native Whisper model using the actual tokenizer, KV cache and beam implementation. The public Whisper-small, matching WavLM/HiFi-GAN artifacts and strict numerical backend were used in real execution. The parent baseline preflight matched all 150 saved control updates and three final audio tensors exactly.

## Frozen three-clip calibration and pilot

All three predeclared duration clips produced finite nonzero risk gradients. The median norm-ratio calibration fixed **c = 355.8228728875952**, unchanged thereafter.

| Duration stratum / clip | NLL gradient norm | Risk gradient norm | NLL/risk norm ratio | Scaled-risk/NLL norm | Gradient cosine |
|---|---:|---:|---:|---:|---:|
| Short / 6345-93302-0028 | 4.82047 | 0.0190471 | 253.082 | 1.406 | 0.6830 |
| Median / 8842-302196-0004 | 3.91273 | 0.0109963 | 355.823 | 1.000 | 0.9169 |
| Long / 2035-152373-0013 | 3.12151 | 0.0000987909 | 31597.176 | 0.01126 | 0.8482 |

The median calibration does **not** match all clip gradients. The long clip's scaled risk gradient is approximately **1/88.8** of its sum-NLL gradient norm. This heterogeneity was recorded without changing the registered scale.

The same three clips then completed 50 Adam updates each with rho=0.1, learning rate=0.1, zero initialization, and the original clean-referenced TV/HF terms with unit coefficients. All risk gradients remained finite/nonzero; no final source transcript was empty. The generation-time native Whisper-small source count was **25 substitutions, 0 deletions and 8 insertions / 84 reference words = 39.2857% WER**. This decoding occurred inside the strict-math generation backend and is a generation-policy observation. The historical full-objective control has **51 substitutions, 3 deletions and 14 insertions / 84 = 80.9524% WER** under its separate automatic evaluation policy. Their backend policies differ, so these numbers are not an exact paired comparison; matching-policy rescoring is required. These are development/calibration clips, and R1 quality has not yet been scored. No quality-matched or transfer conclusion follows.

## Canonical 24-clip stop

The first registered development clip, **1673-143397-0019**, failed at candidate refresh step zero. Its four source hypotheses differed only in punctuation/capitalization and all had exactly the reference's 47 normalized words. All five candidate costs, including the reference, were zero. The condition stopped with **0 of 24 completed outputs and zero updates on the failing clip**. A negative result must not be interpreted as a completed 24-clip benchmark.

The failure receipt contains every actual token/text path and the exact zero perturbation. The logging-only v2 wrapper was registered separately after the v1 pilot; it changed no model call, candidate rule, objective, optimizer, scale or cohort. The calibrated original runner and its identical v1 copy remain preserved. No canonical fallback or c=1 control was executed after this stop. The c=1 control remains unexecuted and cannot repair an all-zero-cost candidate set.

## Resource and evidence accounting

| Stage | Measured elapsed time including setup | Peak allocated / reserved CUDA memory |
|---|---:|---:|
| Three-clip calibration | 9.023 s | 11.884 / 12.522 GB |
| Three-clip, 50-update pilot | 176.683 s | 9.883 / 10.362 GB |
| Aborted canonical development attempt | 4.684 s | 5.134 / 5.744 GB |

These are process elapsed measurements, not separately instrumented GPU-active seconds. Failed work is included. Automatic quality scoring, c=1 control and held-out transfer are absent from these measurements. The GPU was released immediately after the canonical stop.

- Calibration, all candidate sets, physical-gradient probes and frozen scale: [r1-calibration/result.json](r1-calibration/result.json).
- Completed three-clip outputs, optimizer states, refresh candidates and traces: [r1-calibrated-pilot3/result.json](r1-calibrated-pilot3/result.json) and [generated.jsonl](r1-calibrated-pilot3/generated.jsonl).
- Explicit canonical stop: [r1-calibrated-development24/result.json](r1-calibrated-development24/result.json).
- All-zero-cost candidates and zero delta: [failure receipt](r1-calibrated-development24/failures/1673-143397-0019.json).
- Logging-only execution lineage: [v2 registration](r1-calibrated-development24-v2-registration.json).

No stronger content-preserving attack, transfer, quality or novelty claim is supported by this stage.
