# Recorded research compute ledger

Recorded scheduled time totals **2.128 hours (7660.072 seconds)**: 1.563 hours for the original development24/R3 work and 0.565 hours for the later48-utterance generation/evaluation. The original six-hour envelope minus this recorded subtotal is **3.872 hours**. This is recorded headroom, not a certified exact balance, because early timer coverage is incomplete.

The ledger sums actual run timers rather than elapsed time between chats. Whole detached process intervals take precedence over nested timers. Source recognition and all CPU MOS time count as scheduled job time, even when the GPU is idle. R3's entire detached process is counted once; its 3541.888-second training region and nested checkpoint/final evaluations are not added again. Configuration caps are not treated as actual compute consumed.

| Job | Recorded seconds | Timer coverage |
|---|---:|---|
| Baseline exact-control preflight:150 updates and final waveforms | 44.717 | Sum of3 nonoverlapping clip optimization regions; model/Python startup not recorded |
| R1 canonical calibration | 9.023 | Runner wall timer, including model load; excludes Python import/startup |
| R1 canonical initial3 mechanism check | 176.683 | Runner wall timer, including model load; excludes Python import/startup |
| R1 canonical development24 flat-risk stop | 4.684 | Runner wall timer, including model load; excludes Python import/startup |
| R1 deletion calibration | 6.439 | Runner wall timer, including model load; excludes Python import/startup |
| R1 deletion initial3 mechanism check | 130.970 | Runner wall timer, including model load; excludes Python import/startup |
| R1 deletion development24 generation | 939.639 | Runner wall timer, including model load; excludes Python import/startup |
| R2 initial3 geometry check | 35.387 | Runner wall timer including model load; excludes Python import/startup |
| R2 development24 generation | 298.048 | Runner wall timer including model load; excludes Python import/startup |
| R2 native source recognition plus all MOS scoring | 193.797 | Complete evaluator timed region including source recognition and CPU MOS work |
| r1-deletion-native-evaluation | 108.325 | Whole detached child-process wall interval including startup, failures, and nested CPU/source scoring |
| r3-native-reference-preflight | 8.317 | Whole detached child-process wall interval including startup, failures, and nested CPU/source scoring |
| r3-waveform-reference-repair-bounded-run | 8.550 | Whole detached child-process wall interval including startup, failures, and nested CPU/source scoring |
| r3-waveform-autograd-repair-bounded-run | 3661.900 | Whole detached child-process wall interval including startup, failures, and nested CPU/source scoring |
| generation48 | 1564.635 | Whole detached child-process wall interval including startup and all CPU/source scoring |
| native-evaluation48 | 468.958 | Whole detached child-process wall interval including startup and all CPU/source scoring |

Some early baseline/R1/R2 receipts omit Python startup and the baseline model load. One early R1 failure had a logging-only rerun whose earlier timer was not preserved separately. These gaps prevent a strict exact-total claim; reserve additional allowance for them. No idle gap caused by the user's limit is charged. Historical reproduction runs outside improvement/, CPU data preparation, unit tests, audits and report rendering are outside this GPU research ledger.

The [JSON ledger](/workspace/improvement/execution_refine/compute-ledger.json) binds each evidence file by SHA256, specifies the field used, and lists nested receipts excluded to avoid double counting. [Rebuild script](/workspace/improvement/execution_refine/build_compute_ledger.py).
