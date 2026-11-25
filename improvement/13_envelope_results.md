# Clean-energy waveform experiment results

Updated **2026-10-22T22:43:06+08:00 (Asia/Taipei)**.

The fixed primary candidate is envelope_nll50. All scores use the same frozen public source recognizer and native automatic quality evaluation. No human, held-out transfer or novelty claim follows.

## Pilot: 48 utterances / 30 speakers

| Condition | Source WER % | DNSMOS | NISQA | UTMOS |
|---|---:|---:|---:|---:|
| attenuated_global_nll50 | 39.9396 | 3.1112 | 3.7017 | 3.7636 |
| clean | 4.0241 | 3.2721 | 4.0533 | 4.0259 |
| envelope_full50 | 19.4165 | 3.2710 | 4.0650 | 4.0045 |
| envelope_nll50 | 29.1751 | 3.2752 | 4.0294 | 3.9140 |
| fv50 | 49.7988 | 3.1739 | 4.0698 | 3.5789 |
| global_full50 | 43.4608 | 3.1656 | 3.7776 | 3.8873 |
| global_nll50 | 58.7525 | 3.0025 | 3.4367 | 3.5678 |

Primary screen: **fails**. Exploratory shape signal: **fails**. Candidate vsFV50: -20.6237 WER points / +0.3351 UTMOS. Candidate vsRMS-matched global NLL: -10.7646 WER points.

Generation 898.36s; native source/MOS evaluation 471.96s.

## Validation

Not run: the pilot failed its frozen advancement rule.

Validation decision: **stop**. Candidate selection remains fixed.

The76-utterance validation is disjoint in utterance/hash/text from the48-point pilot, but has historical study exposure and shared speakers. Confirmation48/final200 and held-out ASRs remain unused. The clean-energy bounds protect registered-frame energy; they are not auditory masking or a human-content guarantee.

[Explanation and goals](11_changes_and_goals.md) · [Protocol](12_envelope_protocol.md) · [Live status](JOB_STATUS.md)
