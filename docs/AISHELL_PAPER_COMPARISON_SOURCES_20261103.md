# AISHELL paper comparison sources — 2026-11-03

The supplied [PDF](/workspace/BeyondWaveform.pdf) is byte-identical to [arXiv v1](https://arxiv.org/pdf/2606.05678v1), verified by SHA256: `fa1754d1f985d2791c1119fa906fe3baafa340db8a5f9a85ed97be83abf17eee` (738,098 bytes).

| Paper location | Condition | Metric | Value |
|---|---|---|---:|
| Table 1, AISHELL-1 (Chinese), Raw Model column | Clean (No Attack) | CER | 6.29% |
| Table 1, AISHELL-1 (Chinese), Raw Model column | Feature Attack (Ours) | CER | 72.25% |
| Same experiment | Zero-perturbation vocoder reconstruction | CER | Not reported |
| Same experiment | Mandarin transcription | WER | Not reported |

Values and metric assignment are explicit in [Table 1 and caption, printed p.5](https://arxiv.org/pdf/2606.05678v1#page=5).

[Section 5, Implementation Details, printed p.4](https://arxiv.org/pdf/2606.05678v1#page=4) specifies Whisper-small fine-tuned on AISHELL-1 development data, with final attack evaluation on the held-out AISHELL-1 test set; test data is excluded from training. It gives 50 updates, rho=0.1, lambda_perc=1, and 6,000-Hz frequency cutoff. [Appendix A, printed p.10](https://arxiv.org/pdf/2606.05678v1#page=10) and [Table 2, printed p.11](https://arxiv.org/pdf/2606.05678v1#page=11) confirm the selected settings.

The paper supplies no exact AISHELL evaluation count or utterance-ID manifest. Identical coverage of all 7,176 local test recordings therefore remains unverified. The reported 999-recording count concerns English LibriSpeech, not AISHELL. The 6.6% loss-ablation WER is not an AISHELL reconstruction CER.
