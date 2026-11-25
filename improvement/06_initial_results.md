# Initial development results

Updated **2026-10-22T11:11:32+08:00 (Asia/Taipei)**. Status: **completed_source_only_initial_cycle**.

All numerical rows below use the same exposed 24 development clips / 475 reference words and the frozen native source-recognition policy. Higher attacked machine WER indicates more recognition errors; it does not establish human content preservation or transfer.

| Method | Source WER % | DNSMOS | NISQA | UTMOS | Automatic quality floors |
|---|---:|---:|---:|---:|---|
| Original quality control, LR 0.02 | 45.6842 | 3.2075 | 4.1245 | 3.6761 | pass |
| Clean | 3.5789 | 3.3323 | 4.0838 | 4.1886 | pass |
| Zero-edit reconstruction | 3.5789 | 3.3317 | 4.2124 | 4.1667 | pass |
| Original strong control, LR 0.1 | 74.7368 | 3.1640 | 4.0350 | 3.3186 | fail |
| R2 local geometry | 3.3684 | 3.3320 | 4.2142 | 4.1661 | pass |
| R1 with one deletion candidate | 58.5263 | 3.1778 | 3.9957 | 3.3497 | fail |
| R3 untrained seed11 control | 3.5789 | 3.3383 | 4.0808 | 4.1886 | pass |
| R3 trained seed11 | 3.5789 | 3.3318 | 4.0838 | 4.1886 | pass |

R1 canonical beam risk stopped on an all-word-correct candidate set. Its one candidate-set repair completed but failed the UTMOS floor. R2 retained quality while eliminating nearly all attack effect. Both fail the registered comparison with the quality-admissible original control.

R3 completed **1779 optimizer updates / 7116 accumulated microbatches** in **3541.89 seconds**. Stop reason: `one_hour_wall_time_cap_conservative_reserve`. Selection: automatic source-only proxy. Checkpoint source/MOS evaluations are charged to that training wall-time cap.

R3 differs from the primary original quality control by **-42.1053 source-WER points** and **+0.5125 UTMOS**. Its automatic source/quality development screen **fails**. This is a one-seed feasibility result. It is not final superiority or evidence of a publishable contribution.

No target recognizers, confirmation48, final200, human raters or defense training were used in this initial cycle. No sampling or training-seed confidence interval is claimed. A promising source-only result would still need the registered mechanism controls, genuine additional seeds, content screen and untouched evaluation before an improvement claim.

[Machine-readable comparison](execution/initial-comparison.json) · [Detailed execution report](05_execution_report.md) · [Reproduction guide](execution/reproduction-guide.md) · [Job status](JOB_STATUS.md)

[Independent R3 saved-artifact audit](execution/r3-completed-audit.md)
