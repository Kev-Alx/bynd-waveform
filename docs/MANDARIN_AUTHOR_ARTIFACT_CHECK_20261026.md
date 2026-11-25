# Mandarin author-artifact check

Checked on 2026-10-26 while the independent AISHELL reproduction was running.

The primary paper is [Beyond Waveform Robustness v1](https://arxiv.org/html/2606.05678v1). Section 5 says the Mandarin surrogate starts from Whisper-small and is fine-tuned on AISHELL-1 development data. It reports 6.29% clean CER and 72.25% attacked CER. It specifies rho=0.1, lambda=1 and 50 attack updates.

The checked paper text does not identify a distributed Mandarin checkpoint, an exact development partition, fine-tuning optimizer settings, epoch count, checkpoint-selection rule or decoding configuration. Appendix C lists public research components and licenses but does not supply the paper's fine-tuned Mandarin weights. The [arXiv metadata page](https://arxiv.org/abs/2606.05678) does not link an author repository or model checkpoint.

Bounded follow-up searches used the exact paper title, arXiv identifier, author names and checkpoint terms. GitHub repository name/description searches through the configured connector for `"Beyond Waveform"`, `"2606.05678"` and `"feature-vocoder"` returned no repositories. These checks did not identify an author checkpoint; they do not prove that none exists elsewhere.

The current implementation therefore continues to use an independently fine-tuned Whisper-small. Its corrected 5,000-record clean CER is 8.576592%, with original references and normalization unchanged. A separate validation-only comparison is checking the already-trained epochs. The prepared full-development trial uses all 14,326 development recordings and independent validation speakers from the official training split. That full-split interpretation and the fixed training recipe are explicit engineering choices, not recovered author settings.

Numerical similarity to the paper must be reported as similarity of independently measured results. It does not establish checkpoint identity or exact recovery of the author's implementation.
