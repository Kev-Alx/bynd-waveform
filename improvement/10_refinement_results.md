# Refinement and larger-validation results

Updated **2026-10-22T17:21:18+08:00 (Asia/Taipei)**.

All results use the frozen native source recognizer and automatic quality scorers. These measurements do not establish human intelligibility, black-box transfer or novelty.

## Pilot: 48 utterances / 30 speakers

| Condition | Source WER % | DNSMOS | NISQA | UTMOS |
|---|---:|---:|---:|---:|
| clean | 4.0241 | 3.2721 | 4.0533 | 4.0259 |
| fv50 | 49.7988 | 3.1739 | 4.0698 | 3.5789 |
| fv100 | 65.4930 | 3.0888 | 4.0057 | 3.3105 |
| full50 | 43.4608 | 3.1656 | 3.7776 | 3.8873 |
| weighted100 | 53.4205 | 3.1158 | 3.5937 | 3.8390 |
| constrained100 | 43.4608 | 3.1655 | 3.7771 | 3.8871 |

The primary quality-admissible control fixed on the pilot is **fv50**.

- constrained100: -6.3380 source-WER points and +0.3082 UTMOS versus that control; primary source/quality screen **fails**. Gain over the full50 initializer: +0.0000 WER points.
- weighted100: +3.6217 source-WER points and +0.2601 UTMOS versus that control; primary source/quality screen **fails**. Gain over the full50 initializer: +9.9598 WER points.

Generation 4134.29s; native source/MOS evaluation 408.11s. Constrained line-search forwards are additional cost;100 source backward calls does not mean equal total runtime.

## Validation

Not run: neither continuation earned the frozen advancement rule. The detached queue completed normally at17:21 on22 October2026 (Asia/Taipei); this was a negative scientific result, not a crashed job. The next experiment is explained in [the changes-and-goals document](11_changes_and_goals.md).

Validation decision: **stop**; fixed candidate **None**. A mechanism signal alone justifies exploratory validation; it is not a relaxed primary-improvement claim.

The validation76 has no utterance/hash/text overlap with the48-point pilot, but shares speakers and has historical study exposure. No untouched confirmation/final set or held-out recognizer was used. Caps constrain two proxy distortions, not MOS or human semantics. No final superiority claim follows.

[Protocol](09_refinement_protocol.md) · [Live status](JOB_STATUS.md) · [Prior-work check](execution_refine/prior-work.md) · [Earlier48-point results](08_pilot48_results.md)
