# Whisper loss and decoder alignment diagnostics

19 October 2026. This follows the negative [saved-delta smoothing study](/workspace/reproduction-runs/feature-delta-diagnostics/report.md). The same perturbation norm gave no quality gain from smoothing, so that result does not justify a smoothed optimizer. Joint paper WER/quality matching remains unresolved.

The next diagnostic measures whether teacher-forcing NLL rises by assigning probability to tokens that native recognition suppresses. It uses the existing 24-clip pilot and the strong sum-NLL/LR=0.1 parent at rho=0.1, lambda=1, 50 updates and a 6000 Hz cutoff. Saved clean, reconstruction and adversarial WAVs are read without alteration. Neither recognition nor MOS is rerun; no optimization, validation or test expansion occurs.

For each waveform, evaluate lexical targets under the actual no-timestamps attack prefix and the fixed initial-timestamp prefix explored previously. These are conditional teacher-forcing measurements, not free recognition. Scoring references and the full-vocabulary attack implementation remain unchanged.

At each lexical target, the diagnostic reports full-vocabulary NLL, native static-suppression probability mass and the conditional NLL after excluding statically suppressed tokens. If the reference itself is suppressed, mark the conditional loss undefined and report its coverage explicitly. On allowed targets, the exact gap is `full_NLL - conditional_NLL = -log(P(static_allowed))`. Report token-micro and utterance-macro summaries separately.

The **predeclared mechanism screen**, evaluated on the actual attack prefix and statically valid positions shared by reconstruction and adversarial audio, requires all of:

- Adversarial-minus-reconstruction full NLL is positive, with a numerical guard of 1e-8.
- The corresponding increase in static-suppression gap explains at least 10% of that NLL increase.
- Median adversarial static-suppressed mass across those tokens is at least 0.10.
- At least 95% of lexical reference tokens remain statically allowed.

This is a diagnostic threshold, not statistical evidence of improved WER or sound quality. A positive screen motivates a separately frozen, bounded experiment on the loss softmax policy. A negative screen stops that proposed intervention. No candidate is automatically launched or promoted by this script.

Native ordered blank/static/timestamp filters are inspected separately on timestamp-conditioned teacher prefixes. Record dynamically forbidden targets and when the forced reference path becomes unreachable. Later forced-prefix observations must be labeled hypothetical once an earlier reference target has been forbidden. Dynamic timestamp filtering must not be applied to the incompatible no-timestamps prefix or silently substituted for the original objective.

The new frozen plan binds prior input hashes, pilot assets, native Whisper sources, tokenizer vocabulary, effective decoding defaults and dependency versions. The pretrained checkpoint, cached feature/delta bindings and parent receipts are checked. A three-clip smoke diagnostic precedes the full 24-clip run; all 24 clips belong to the already used development pilot. Completion/failure records, stage logs and heartbeat status are retained. Discord remains disabled.

Files: [driver](/workspace/scripts/run_decoder_diagnostics.py), [measurement](/workspace/scripts/probe_decoder_alignment.py), [job status](/workspace/reproduction-runs/JOB_STATUS.md), [diagnostic report](/workspace/reproduction-runs/decoder-alignment-diagnostics/report.md).
