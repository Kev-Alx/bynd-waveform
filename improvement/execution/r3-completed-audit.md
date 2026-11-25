# Independent CPU audit of completed R3

Updated 2026-10-22T03:11:30.442035+00:00. Status: **pass**.

Run: `/workspace/improvement/execution/r3-waveform-autograd-repair`. No neural model was loaded or executed; CUDA initialized: False.

Training: 1779 AdamW updates, 7116 completed microbatches, 0 discarded partial microbatches, 3541.888 training wall seconds. Registered manifest order was replayed with Python seed 11. Accumulation, finite reported losses, fixed weights, completed checkpoint states and source-only selection were checked.

Generated 24 development examples / 72 WAVs. Clean samples equal registered source audio exactly; duration, peak, Linf and RMS bounds were independently checked for initial and trained generators.

| Variant | N | Source WER % | DNSMOS | NISQA | UTMOS |
|---|---:|---:|---:|---:|---:|
| adversarial | 24 | 3.578947 | 3.331831 | 4.083795 | 4.188555 |
| clean | 24 | 3.578947 | 3.332344 | 4.083788 | 4.188579 |
| initial-generator | 24 | 3.578947 | 3.338276 | 4.080786 | 4.188580 |

Against B2 LR=.02: delta source WER -42.105263 percentage points, delta UTMOS +0.512491. Automatic quality admissible: True; registered source/quality screen passes: False. This screen excludes efficiency, human intelligibility and held-out transfer.

## Issues

- No discrepancy found in the completed scope.

## Limits

- Saved-artifact verification only; no neural inference or optimizer replay.
- Trace finiteness verifies reported scalar losses/gradient norms, not all intermediate activation/gradient tensors.
- Checkpoint tensors are loaded weights_only on CPU; terminal hash is producer-bound, while periodic checkpoint hashes are computed at audit time.
- Generation is not reexecuted: checkpoint-to-WAV causal correspondence relies on registered producer code and receipts.
- Source transcript use is constrained by frozen IDs and implementation; full training input transcripts are not repeated in trace rows.
- MOS and source WER do not establish human intelligibility, held-out ASR transfer or improved generalization.
- No final-reserve audio or held-out-target artifact is accessed. The earlier independent cohort audit covers sealed-reserve exclusion.
