# Saved-artifact audit: development48

Status: **pass**.

This is an artifact-consistency audit, not a finding that the attacks succeeded. It runs on CPU and loads no models.

| Condition | n | Source WER % | UTMOS | DNSMOS | NISQA |
| --- | ---: | ---: | ---: | ---: | ---: |
| clean | 48 | 4.024145 | 4.025875 | 3.272060 | 4.053286 |
| reconstruction | 48 | 4.024145 | 4.013135 | 3.300239 | 4.171610 |
| fv_quality | 48 | 49.798793 | 3.578912 | 3.173889 | 4.069770 |
| r3_initial | 48 | 4.024145 | 4.025875 | 3.270005 | 4.055737 |
| r3_trained | 48 | 4.024145 | 4.025862 | 3.271680 | 4.053293 |
| pgd_full | 48 | 43.460765 | 3.887348 | 3.165626 | 3.777572 |
| pgd_nll | 48 | 58.752515 | 3.567782 | 3.002519 | 3.436720 |

Recorded PGD failures: 0. Failed attempts remain included among all48 and are not completed attacks.

Saved CPU geometry/waveform/hash/trace/edit-distance/MOS aggregation only; no neural gradient or model-output replay. Fresh uniform strict E6 is not compared with old cached queries. Human intelligibility/held-out transfer remain unestablished.
