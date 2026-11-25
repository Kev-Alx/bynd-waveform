# Larger-model research results

Updated **2026-10-24T22:02:54+08:00 (Asia/Taipei)**.

Goal: sourceWER>75% with unchanged automatic quality floors, followed by measured held-out transfer and defense persistence. These are separate endpoints. No human-intelligibility or novelty claim is made.

## qualification

| Condition | N | SourceWER% | DNSMOS | NISQA | UTMOS |
|---|---:|---:|---:|---:|---:|
| bigvgan_reconstruction | 48 | 3.9235 | 3.2981 | 4.0521 | 3.9868 |
| clean | 48 | 4.0241 | 3.2721 | 4.0533 | 4.0259 |
| fv_quality50 | 48 | 49.7988 | 3.1739 | 4.0698 | 3.5789 |
| fv_reconstruction | 48 | 4.0241 | 3.3002 | 4.1716 | 4.0131 |

## fvpilot

| Condition | N | SourceWER% | DNSMOS | NISQA | UTMOS |
|---|---:|---:|---:|---:|---:|
| clean | 48 | 4.0241 | 3.2721 | 4.0533 | 4.0259 |
| ensemble_eot_200 | 48 | 81.7907 | 3.1708 | 4.0672 | 3.2345 |
| ensemble_eot_50 | 48 | 64.6881 | 3.1748 | 4.0921 | 3.3771 |
| ensemble_eot_500 | 48 | 89.5372 | 3.1476 | 4.0211 | 3.0850 |
| ensemble_raw_200 | 48 | 81.9920 | 3.0532 | 3.9517 | 3.1636 |
| ensemble_raw_50 | 48 | 68.3099 | 3.1270 | 4.0665 | 3.3671 |
| ensemble_raw_500 | 48 | 90.4427 | 3.0997 | 3.9295 | 3.0557 |
| fv_quality50 | 48 | 49.7988 | 3.1739 | 4.0698 | 3.5789 |
| fv_reconstruction | 48 | 4.0241 | 3.3002 | 4.1716 | 4.0131 |
| single_150 | 48 | 97.1831 | 3.0512 | 3.9500 | 3.0866 |
| single_1500 | 48 | 101.4085 | 3.0196 | 3.8260 | 2.9437 |
| single_200 | 48 | 95.9759 | 3.0641 | 3.9187 | 3.0560 |
| single_50 | 48 | 83.9034 | 3.1056 | 3.9766 | 3.1981 |
| single_500 | 48 | 100.5030 | 3.0540 | 3.7788 | 2.9846 |
| single_600 | 48 | 93.3602 | 3.0770 | 3.8614 | 2.9859 |

## melpilot

| Condition | N | SourceWER% | DNSMOS | NISQA | UTMOS |
|---|---:|---:|---:|---:|---:|
| bigvgan_reconstruction | 48 | 3.9235 | 3.2981 | 4.0521 | 3.9868 |
| clean | 48 | 4.0241 | 3.2721 | 4.0533 | 4.0259 |
| mel_eot_200 | 48 | 80.1811 | 2.8827 | 3.3091 | 2.2951 |
| mel_eot_50 | 48 | 33.9034 | 3.1236 | 3.7549 | 3.2508 |
| mel_eot_500 | 48 | 78.7726 | 2.6038 | 2.7068 | 1.6528 |

Source-only advancement: **stop**. Candidate: **none**. Known-baseline candidate: **False**.

The76-point validation is exposed development with shared speakers. Confirmation48/final200 remain unused. LPF and WaveGuard resampling cover named preprocessing procedures; missing training-defense weights, sentence-compatible AudioPure, and full WaveGuard detection are not measured. A blinded human study remains required.

[Protocol and goals](14_scaled_research_protocol.md) · [Live status](JOB_STATUS.md) · [Changes explained](11_changes_and_goals.md)
