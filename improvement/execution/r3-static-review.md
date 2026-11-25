# R3 static review before initial GPU execution

**Runtime correction:** the subsequent real GPU preflight found an upstream WavLM inplace autograd error during content-gradient calibration, before training. The initial static verdict below was not a gradient proof. The error and repair are now covered by five CPU tests using the actual pinned upstream implementation; see [repair tests](/workspace/improvement/test_r3_encoder_repair.py) and [passing receipt](/workspace/improvement/execution/r3-encoder-repair-tests.log). Original failure evidence remains at [run-failure.json](/workspace/improvement/execution/r3-waveform-reference-repair/run-failure.json).

Review scope: CPU-only source inspection of the registered seed11, one-hour, source-only path. No model inference or GPU work was launched. Registered implementation files were not edited.

Verdict: no critical execution or scientific correctness defect found in the fresh initial execution path. Runtime preflight must still establish real GPU support, finite gradients, numerical conformance, memory fit, and calibrated weights. Static inspection cannot replace those checks.

## Checked behavior

- The generator is a six-level Conv1d U-Net with 1,590,833 parameters, exact input-length output, a bounded tanh residual, conservative RMS scaling, and final audio clipping. Decoder channel dimensions and skip ordering agree.
- The source objective has the correct sign: minimize negative original-transcript lexical SUM NLL plus positive MRSTFT and relative WavLM6 penalties. Frozen source/content parameters receive no optimizer updates. AdamW uses lr1e-4, betas(.9,.999), weight decay0; four microbatches contribute gradients divided by4, followed by one global normclip1 update.
- Sequential backward passes recompute the deterministic generator three times; they avoid retaining source-ASR and WavLM activation graphs together. All three passes and the deterministic interpolation CPU derivative are included in timing.
- Against pinned upstream WavLM.extract_features(mask=False, padding_mask=None, output_layer=6), the differentiable adapter follows the same forward path, bypassing only GradMultiply/no_grad around feature extraction. The preflight separately requires bitwise native/adapter agreement and repeatability before calibration.
- Penalty calibration uses exactly the three frozen development clips and a seeded amplitude1e-4 Rademacher probe. Coefficients freeze before generator outputs; the saved calibration hash is checked before training. One global median ratio per penalty is used, with finite/nonzero guards.
- Longest-training-utterance fit performs four real microbatches and one AdamW update, allocating optimizer state before the separate training-hour clock starts. That fit model is discarded; training starts from a newly initialized seed11 generator.
- Native source/MOS evaluation runs in a fresh subprocess after CUBLAS_WORKSPACE_CONFIG is removed. The helper establishes and compares the historical backend before and after scoring, binds saved WAV hashes and unchanged development references, and loads only the source Whisper-small checkpoint. Held-out targets and confirmation/final samples are excluded.
- Checkpoint evaluation time is charged to the training wall clock, with timeout/save margin. Completed500-update checkpoints use the declared source-only automatic quality proxy and source WER/STFT/earliest tie-break. Without such a checkpoint, the terminal state is explicitly feasibility-only.
- Training progress/status files update at least every30s at microbatch boundaries. Partial accumulation is discarded and reported before a terminal checkpoint; it is never counted as a completed update.

## Training-manifest range

The frozen one-hour manifest contains286 clips. Minimum duration2.135s, maximum17.615s, maximum281840 samples; no clip exceeds the trainer's30s guard. All rows inspected declare train/train-clean-100 and16kHz. Selection/exclusion correctness belongs to the separate independent acquisition audit.

## Limitations and operational notes

- Completed preflight and completed training receipts can be reused. An interrupted incomplete training trace is deliberately not resumed in place: the runner refuses it and requires preserving the old trace and starting a named variant. This is a limitation, not a fresh-run correctness defect. A detached process should therefore be checked before launching another run.
- The native-policy evaluator rejects overwriting an existing source-summary.json. Re-scoring requires a new output directory; no blind evaluator retry into a completed folder.
- Cached-q mismatch is handled before coefficient calibration/training. A named, explicitly registered strict-reference regeneration may proceed only with the mismatch receipt preserved and no historical-cache numerical-parity claim.
- Static review cannot certify the strict CUDA backward implementations or combined parent/child GPU memory. Preflight and real subprocess receipts are required.
- A one-hour seed11 run may not reach500 updates; it does not estimate seed variance or transfer. Automatic MOS quality thresholds do not establish human intelligibility or semantic preservation.
- `make_generator` uses torch.random.fork_rng(devices=[]) around torch.manual_seed. This restores CPU RNG but does not promise restoration of CUDA RNG state. In the inspected path all trainable/frozen networks are deterministic (no generator dropout; anchors in eval), and ordering uses a separate Python RNG, so this does not change the defined initial experiment. Avoid claiming universal RNG-state preservation for that helper.

## Reviewed file bindings

- `/workspace/improvement/r3_real_runner.py`: SHA256 `40b4d9c41eab554ceb1141d3b4ffb97475ec65292e8de761d93f2059b464157e`
- `/workspace/improvement/r3_waveform.py`: SHA256 `41f710cebb79a543912f356a051ee1aacbdf8ebe00631f34364a8ba52dc87289`
- `/workspace/improvement/source_evaluate.py`: SHA256 `ec2655e5782c6023d09e16759f545231cb6db58b6e11afb6bd132aaee3fddb31`
- `/workspace/improvement/execution/data/train-1h.jsonl`: SHA256 `35a374c93963fbd2aad6efee7887e6aefcdd105b4c707d8f8fe9014f3c43b58d`

## Actual-upstream autograd repair follow-up

The pinned `TransformerEncoder.extract_features` passes `x.transpose(1,2)` to its positional convolution, then performs `x += x_conv`. That mutation invalidates the transposed input saved by Conv1d backward. An actual six-layer tiny WavLM CPU model reproduces the same saved-view version error as the real checkpoint GPU run.

The separately registered [repair adapter](/workspace/improvement/r3_encoder_repair.py) temporarily clones only the positional-convolution input in gradient-enabled calls; a `finally` block removes the hook. It leaves the pinned implementation file, model parameters, operation order, and forward values intact. The independent five-test regression confirms native bitwise forward values for both layernorm-order variants, finite nonzero waveform gradients with frozen parameters, gradient agreement with an independent actual-upstream reference changing only the offending `+=` to out-of-place addition, and hook removal after success or exception. It does not establish real full-checkpoint GPU memory or training feasibility; the new preflight must do so.

The exploratory [CPU probe](/workspace/improvement/execution/r3-autograd-probe.json) retains a failed numerical finite-difference check at epsilon1e-5. Upstream LayerNorm/GELU explicitly cast to float32, so casting the model to double does not provide a clean double-precision finite-difference oracle. That probe's repaired forward was bitwise equal and its gradient finite/nonzero. The final regression instead compares the two actual autograd graphs directly at rtol1e-6/atol1e-7 and passes; the exploratory receipt is preserved rather than silently replaced.
