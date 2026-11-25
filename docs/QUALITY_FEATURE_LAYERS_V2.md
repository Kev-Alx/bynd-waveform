# Layer probes with the original encoder execution policy

The first bounded layer study stopped at its layer-6 control on the third clip; no attack or quality score was produced. Its source, plan, failure/status and partial outputs remain intact in `paper-quality-feature-layers`. Its relative-L2/max-absolute tolerance was not relaxed.

The frozen [encoder execution diagnostic](/workspace/reproduction-runs/encoder-execution-diagnostics/report.md) then compared all 24 cached layer-6 q against fresh auto, math and strict-math forward passes, with three repeats on the fixed first three IDs. **Auto reproduces all 24 cached tensors bitwise.** Math and strict-math repeat exactly in the checked clips but differ from the cached auto values, including differences outside the original maximum-absolute threshold. This is an encoder execution-policy difference; it neither demonstrates a loss bug nor explains the WER/quality gap.

Revision two preserves the original no-gradient WavLM encoder policy (auto attention, cuBLAS workspace unset, existing seeding/cuDNN deterministic/TF32-off settings), and separately retains strict-math/CUBLAS `:4096:8` for attack derivatives and vocoder verification. The earlier successful attacks already used frozen auto-extracted q with strict derivatives. No gradient is propagated through the frozen feature extraction stage.

This revision strengthens the layer-6 control to **bitwise equality on all 24 clips**. It does not change the two predeclared layers (5/7), any reconstruction threshold, objective, reference, model/scorer weight, data selection or feature budget. See [the original bounded protocol](/workspace/docs/QUALITY_FEATURE_LAYERS.md) for those settings. The available vocoder remains trained for layer 6: these remain explicitly out-of-training-contract probes, even if an empirical reconstruction screen passes. The production loader's matching-layer guard stays unchanged.

Execution proceeds in separate fresh processes:

1. Extract layer 5/7 and synthesize reconstruction controls using the original auto encoder policy; require exact layer-6 cache equality on all clips and first-clip repeat equality for each probe.
2. Score reconstruction controls with unchanged native decoding/MOS. Independently re-extract all 24 q for each layer under the original policy and require raw-value hashes to match exactly. Bind the cache/manifest/value hashes and actual backend to an encoder verification receipt.
3. In strict-math processes, resynthesize reconstructions from cached q and apply the unchanged WER<=control+2, each MOS>=control−0.1 and >2.5 screen. A negative screen skips that layer's attacks.
4. Only passing layers use the original full-objective strict-math 50-step collector on saved q. Native evaluation is unchanged. Verification binds the prior independent encoder receipt and re-synthesizes q+delta under strict policy.

The separation preserves numerical comparability with the frozen encoder cache. It is not evidence for author checkpoint parity or a performance gain. All failures, negative gates, source revisions and conditional outputs are retained. Fresh48 and 999 remain untouched; Discord remains disabled.
