# Independent CPU audit after resumption

Updated 2026-10-22T03:11:29.380516+00:00. Status: **pass**.

R1 repaired generation: 24 development clips, 1200 updates, 240 candidate refreshes, 1440 candidate rows, 72 saved WAVs. All registered IDs/references, one middle-word deletion per refresh, lexical tokens, detached word costs, score/posterior/risk/loss arithmetic, terminal feature radii, optimizer state and saved WAV bindings were checked independently on CPU.

Fixed gradient scale recomputed from saved scalar norms: 376.686067149993; ratios [255.17518653592822, 376.6860671499931, 29688.318163906864]. CUDA initialized: False. Neural models loaded/executed: none.

| Run / variant | N | WER % | DNSMOS | NISQA | UTMOS |
|---|---:|---:|---:|---:|---:|
| r1-deletion-development24 / adversarial | 24 | 58.526316 | 3.177816 | 3.995733 | 3.349724 |
| r1-deletion-development24 / clean | 24 | 3.578947 | 3.332344 | 4.083788 | 4.188579 |
| r1-deletion-development24 / reconstruction | 24 | 3.578947 | 3.331732 | 4.212429 | 4.166712 |
| r2-canonical-zero-uniform-dev24 / adversarial | 24 | 3.368421 | 3.332028 | 4.214219 | 4.166126 |
| r2-canonical-zero-uniform-dev24 / baseline_attenuated | 24 | 4.842105 | 3.301993 | 4.192638 | 4.083689 |
| r2-canonical-zero-uniform-dev24 / baseline_frozen | 24 | 74.736842 | 3.164039 | 4.034993 | 3.318612 |
| r2-canonical-zero-uniform-dev24 / clean | 24 | 3.578947 | 3.332344 | 4.083788 | 4.188579 |
| r2-canonical-zero-uniform-dev24 / reconstruction | 24 | 3.578947 | 3.331732 | 4.212429 | 4.166712 |
| native-source / adversarial | 24 | 3.578947 | 3.332181 | 4.083824 | 4.188535 |
| native-source / adversarial | 24 | 3.578947 | 3.332581 | 4.083813 | 4.188546 |
| native-source / adversarial | 24 | 3.578947 | 3.331831 | 4.083795 | 4.188555 |
| r3-waveform-autograd-repair / adversarial | 24 | 3.578947 | 3.331831 | 4.083795 | 4.188555 |
| r3-waveform-autograd-repair / clean | 24 | 3.578947 | 3.332344 | 4.083788 | 4.188579 |
| r3-waveform-autograd-repair / initial-generator | 24 | 3.578947 | 3.338276 | 4.080786 | 4.188580 |

Primary quality-admissible comparator is B2 LR=.02, source WER 45.684211%, UTMOS 3.676064. Attenuated strong-baseline output is a norm diagnostic; it does not replace that comparator.

r1-deletion-development24: delta source WER +12.842105 percentage points; delta UTMOS -0.326340. Automatic quality admissible: False; registered source/quality screen passes: False. Efficiency, human intelligibility and transfer are not established.
r2-canonical-zero-uniform-dev24: delta source WER -42.315789 percentage points; delta UTMOS +0.490062. Automatic quality admissible: True; registered source/quality screen passes: False. Efficiency, human intelligibility and transfer are not established.
native-source: delta source WER -42.105263 percentage points; delta UTMOS +0.512471. Automatic quality admissible: True; registered source/quality screen passes: False. Efficiency, human intelligibility and transfer are not established.
native-source: delta source WER -42.105263 percentage points; delta UTMOS +0.512482. Automatic quality admissible: True; registered source/quality screen passes: False. Efficiency, human intelligibility and transfer are not established.
native-source: delta source WER -42.105263 percentage points; delta UTMOS +0.512491. Automatic quality admissible: True; registered source/quality screen passes: False. Efficiency, human intelligibility and transfer are not established.
r3-waveform-autograd-repair: delta source WER -42.105263 percentage points; delta UTMOS +0.512491. Automatic quality admissible: True; registered source/quality screen passes: False. Efficiency, human intelligibility and transfer are not established.

## Limits

- CPU saved-artifact arithmetic and bindings only; no neural inference or optimizer trajectory replay.
- Only the terminal delta and optimizer tensors are saved. Intermediate gradient/delta hashes cannot reconstruct those vectors.
- Calibration norm ratios are recomputed from producer-saved scalar norms, not independently recomputed neural gradients.
- Tokenization and decoding checks use the installed frozen public Whisper tokenizer; no ASR model is loaded.
- Independent edit dynamic programming verifies total minimum edit distance; individual S/D/I alignments can have ties.
- Feature-ball validity, automatic MOS and source recognition do not establish human intelligibility or held-out ASR transfer.
- No final-reserve audio or held-out-target artifact is accessed by this audit. This snapshot can be refreshed when pending native evaluations finish.

## Discrepancies

- No discrepancy found in this saved-artifact scope.

Refresh with `PYTHONDONTWRITEBYTECODE=1 /opt/venv/bin/python /workspace/improvement/execution/resume_audit.py`.
