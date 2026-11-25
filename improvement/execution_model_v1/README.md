# Learned feature policy experiment v1

Read [the protocol](../24_learned_feature_protocol.md) and [current status/result explanation](../25_learned_model_results.md). `plan.json` is the immutable registration; `status.json` and `controller.log` are live. `launch_receipt.json` declares the hard 13-hour ceiling including queue wait. Stage receipts are under `jobs/`; `gpu_exit_receipt.json` and `exit_receipt.json` are written at completion. The wrapper also writes `controller_exit.json`.

CPU preflight: four critical tests passed, all modules imported, registered inputs matched with one explicitly documented pre-existing unrelated finetune-file drift. No GPU training or inference was run before the shared queue lock. Source receipts are in `../model_direction_sources/`; new artifacts remain under improvement.
