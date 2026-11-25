# Learned speech-feature model: latest measured status

Controller status: **failed_execution**. The13-hour ceiling includes GPU queue wait. Current registration and exact stage exits are in `execution_model_v1/status.json`.

The purpose is a definitive joint improvement; the method direction remains open. This is a directly source-trained shared model, not per-new-recording optimization or a claim of publication readiness.

## Complete48-point source/quality and inference results

| Condition | N | WER% | DNSMOS | NISQA | UTMOS | Mean sec/recording | p95 sec | Sec/audio sec | Peak GiB | Joint gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| initial_r050 |48|4.3260|3.3025|4.1827|4.0177|0.1424|0.2901|0.0190|2.113|fail|
| source_r025 |48|3.9235|3.2932|4.1106|3.8986|0.1478|0.3426|0.0197|2.113|fail|
| source_r050 |48|4.0241|3.2665|4.0387|3.6630|0.1413|0.2915|0.0188|2.113|fail|
| source_r100 |48|8.3501|3.2005|3.8915|2.7783|0.1406|0.2918|0.0187|2.113|fail|
| quality_r025 |48|3.9235|3.2890|4.2117|4.0072|0.1393|0.2920|0.0186|2.113|fail|
| quality_r050 |48|3.8229|3.2908|4.2198|3.9104|0.1395|0.2906|0.0186|2.113|fail|
| quality_r100 |48|4.1247|3.2308|4.2051|3.6825|0.1390|0.2896|0.0185|2.113|fail|
| quality_shuffled_r050 |48|4.0241|3.2896|4.2158|3.9658|0.1398|0.2927|0.0186|2.113|fail|

Frozen pilot selection: **unselected**, candidate `None`. Thresholds remain WER>75%, DNSMOS>=3, NISQA>=3.75, UTMOS>=3.5. All conditions retain48recordings/994words. The untrained and shuffled conditions are controls, excluded from selection.

## Costs and evidence limits

Preprocessing: 12.15minutes; common warmup and both arms: 4.077hours; selected/display arm plus warmup: 2.128hours. Queue wait: 1.947hours; active wall: 4.529hours. Exact encoding, generation, scoring and phase costs/memory are in [results.json](execution_model_v1/results.json). CUDA-event intervals are stream elapsed time, not exact kernel-active time or energy.
For `quality_r050`,1000similar recordings forecast 0.039inference hours;10000 forecast 0.387hours. Including selected-model preprocessing/training gives 2.370/2.718hours, respectively. The iterative reference itself fails quality, so this is cost arithmetic, not a successful-method deployment claim.

No per-recording ASR queries or optimization are used for candidate inference. Training still pays source/decoder gradients. Input/output IO, fresh feature encoding and policy/decoder time are included; hash checks, shared control/shuffle diagnostics and one-time loading are separately scoped. The original validation control search is not candidate inference.

Unseen recognizers/defenses were not evaluated in this cycle unless complete results above say otherwise. They remain gated by the fixed source/quality and validation requirements.

Untouched confirmation/final evaluation and blinded human word recovery remain unmeasured. Automatic MOS does not establish human intelligibility. [Protocol](24_learned_feature_protocol.md), [primary/Consensus sources](23_model_directions_and_sources.md), [per-recording latency](execution_model_v1/model_latency_per_recording.csv).
