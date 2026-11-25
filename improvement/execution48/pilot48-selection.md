# Development pilot of 48 utterances

Frozen manifest: `/workspace/improvement/execution48/development-48.jsonl` (SHA256 `9a5f56f5cca6c073dbe6e2b6d804971735045ea80c4456bea0d577b7e292748a`).

The pilot preserves the previous 24 utterances in their original order and adds 24 from the historically exposed English validation100 pool. Additions were ordered by SHA256 of `20261022:ID` after metadata and overlap checks. No source recognition, quality, or attack outcomes were used for selection. This remains development data, with no fresh-confirmation claim.

All 48 files passed source SHA256 and native 16 kHz mono metadata checks. Whole utterances span 1.940–24.030 seconds, totaling 360.010 seconds across 30 speakers. No audio was resampled, truncated, or amplitude-normalized.

There is zero ID, exact compressed-audio hash, or normalized-transcript overlap with sealed confirmation48, sealed final200, or training manifests. There is also zero speaker overlap with final200 and training. Confirmation uses the same development split: 21 speakers overlap; this pilot does not establish speaker-independent generalization.

The known historical layer6 query-cache audit provides verified files for 24 utterances (the original 24). The 24 additions are absent from that audit and require fresh extraction before feature-space experiments. The receipt records paths and hashes; no tensors or models were loaded in this selection step.

The script, all source manifests, every selected audio file, and known cached query files are hash-bound in `development-48.receipt.json`. Older plans, results, confirmation48, and final200 were preserved.
