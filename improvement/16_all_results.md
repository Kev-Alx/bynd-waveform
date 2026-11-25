# Complete recognition and audio-quality results

Every row below comes from saved per-recording scores, reconciled against the saved summary. WER is pooled by reference words; quality scores are arithmetic recording means. Higher WER means more machine errors. Higher quality is better. WER can exceed 100% due to insertions.

Quality floors: DNSMOS ≥ 3, NISQA ≥ 3.75, UTMOS ≥ 3.5. The joint research goal additionally requires WER > 75%. 0 conditions met that joint goal. Repeated controls are retained under their actual study; they are not extra independent observations.

## 48-recording development cohort

### Waveform mechanism pilot

| Condition | N | WER % | DNSMOS | NISQA | UTMOS | Quality floors |
|---|---:|---:|---:|---:|---:|---|
| Clean speech | 48 | 4.0241 | 3.2721 | 4.0533 | 4.0259 | pass |
| Original feature method · 50 steps · LR 0.02 | 48 | 49.7988 | 3.1739 | 4.0698 | 3.5789 | pass |
| Waveform PGD · weighted loss · 50 steps | 48 | 43.4608 | 3.1656 | 3.7776 | 3.8873 | pass |
| Waveform PGD · recognition loss only | 48 | 58.7525 | 3.0025 | 3.4367 | 3.5678 | fail: NISQA |
| Generator · untrained | 48 | 4.0241 | 3.2700 | 4.0557 | 4.0259 | pass |
| Generator · trained | 48 | 4.0241 | 3.2717 | 4.0533 | 4.0259 | pass |
| HiFi-GAN reconstruction | 48 | 4.0241 | 3.3002 | 4.1716 | 4.0131 | pass |

### Quiet-region protection

| Condition | N | WER % | DNSMOS | NISQA | UTMOS | Quality floors |
|---|---:|---:|---:|---:|---:|---|
| Uniformly attenuated waveform control | 48 | 39.9396 | 3.1112 | 3.7017 | 3.7636 | fail: NISQA |
| Clean speech | 48 | 4.0241 | 3.2721 | 4.0533 | 4.0259 | pass |
| Quiet-region budget · weighted loss | 48 | 19.4165 | 3.2710 | 4.0650 | 4.0045 | pass |
| Quiet-region budget · recognition loss only | 48 | 29.1751 | 3.2752 | 4.0294 | 3.9140 | pass |
| Original feature method · 50 steps · LR 0.02 | 48 | 49.7988 | 3.1739 | 4.0698 | 3.5789 | pass |
| Waveform PGD · weighted loss · 50 steps | 48 | 43.4608 | 3.1656 | 3.7776 | 3.8873 | pass |
| Waveform PGD · recognition loss only | 48 | 58.7525 | 3.0025 | 3.4367 | 3.5678 | fail: NISQA |

### Learned speech-feature model

| Condition | N | WER % | DNSMOS | NISQA | UTMOS | Quality floors |
|---|---:|---:|---:|---:|---:|---|
| initial r050 | 48 | 4.3260 | 3.3025 | 4.1827 | 4.0177 | pass |
| source r025 | 48 | 3.9235 | 3.2932 | 4.1106 | 3.8986 | pass |
| source r050 | 48 | 4.0241 | 3.2665 | 4.0387 | 3.6630 | pass |
| source r100 | 48 | 8.3501 | 3.2005 | 3.8915 | 2.7783 | fail: UTMOS |
| quality r025 | 48 | 3.9235 | 3.2890 | 4.2117 | 4.0072 | pass |
| quality r050 | 48 | 3.8229 | 3.2908 | 4.2198 | 3.9104 | pass |
| quality r100 | 48 | 4.1247 | 3.2308 | 4.2051 | 3.6825 | pass |
| quality shuffled r050 | 48 | 4.0241 | 3.2896 | 4.2158 | 3.9658 | pass |

### Waveform refinement

| Condition | N | WER % | DNSMOS | NISQA | UTMOS | Quality floors |
|---|---:|---:|---:|---:|---:|---|
| Clean speech | 48 | 4.0241 | 3.2721 | 4.0533 | 4.0259 | pass |
| Waveform PGD · hard constraints · 100 steps | 48 | 43.4608 | 3.1655 | 3.7771 | 3.8871 | pass |
| Waveform PGD · weighted loss · 50 steps | 48 | 43.4608 | 3.1656 | 3.7776 | 3.8873 | pass |
| Original feature method · 100 steps | 48 | 65.4930 | 3.0888 | 4.0057 | 3.3105 | fail: UTMOS |
| Original feature method · 50 steps · LR 0.02 | 48 | 49.7988 | 3.1739 | 4.0698 | 3.5789 | pass |
| Waveform PGD · weighted continuation · 100 steps | 48 | 53.4205 | 3.1158 | 3.5937 | 3.8390 | fail: NISQA |

### Larger models and longer feature runs

| Condition | N | WER % | DNSMOS | NISQA | UTMOS | Quality floors |
|---|---:|---:|---:|---:|---:|---|
| Clean speech | 48 | 4.0241 | 3.2721 | 4.0533 | 4.0259 | pass |
| Three-model feature ensemble + transforms · 200 steps | 48 | 81.7907 | 3.1708 | 4.0672 | 3.2345 | fail: UTMOS |
| Three-model feature ensemble + transforms · 50 steps | 48 | 64.6881 | 3.1748 | 4.0921 | 3.3771 | fail: UTMOS |
| Three-model feature ensemble + transforms · 500 steps | 48 | 89.5372 | 3.1476 | 4.0211 | 3.0850 | fail: UTMOS |
| Three-model feature ensemble · 200 steps | 48 | 81.9920 | 3.0532 | 3.9517 | 3.1636 | fail: UTMOS |
| Three-model feature ensemble · 50 steps | 48 | 68.3099 | 3.1270 | 4.0665 | 3.3671 | fail: UTMOS |
| Three-model feature ensemble · 500 steps | 48 | 90.4427 | 3.0997 | 3.9295 | 3.0557 | fail: UTMOS |
| Original feature method · 50 steps · LR 0.02 | 48 | 49.7988 | 3.1739 | 4.0698 | 3.5789 | pass |
| HiFi-GAN reconstruction | 48 | 4.0241 | 3.3002 | 4.1716 | 4.0131 | pass |
| Original feature method · 150 steps · LR 0.1 | 48 | 97.1831 | 3.0512 | 3.9500 | 3.0866 | fail: UTMOS |
| Original feature method · 1500 steps · LR 0.1 | 48 | 101.4085 | 3.0196 | 3.8260 | 2.9437 | fail: UTMOS |
| Original feature method · 200 steps · LR 0.1 | 48 | 95.9759 | 3.0641 | 3.9187 | 3.0560 | fail: UTMOS |
| Original feature method · 50 steps · LR 0.1 | 48 | 83.9034 | 3.1056 | 3.9766 | 3.1981 | fail: UTMOS |
| Original feature method · 500 steps · LR 0.1 | 48 | 100.5030 | 3.0540 | 3.7788 | 2.9846 | fail: UTMOS |
| Original feature method · 600 steps · LR 0.1 | 48 | 93.3602 | 3.0770 | 3.8614 | 2.9859 | fail: UTMOS |

### BigVGAN mel-space pilot

| Condition | N | WER % | DNSMOS | NISQA | UTMOS | Quality floors |
|---|---:|---:|---:|---:|---:|---|
| BigVGAN reconstruction | 48 | 3.9235 | 3.2981 | 4.0521 | 3.9868 | pass |
| Clean speech | 48 | 4.0241 | 3.2721 | 4.0533 | 4.0259 | pass |
| BigVGAN mel ensemble + transforms · 200 steps | 48 | 80.1811 | 2.8827 | 3.3091 | 2.2951 | fail: DNSMOS, NISQA, UTMOS |
| BigVGAN mel ensemble + transforms · 50 steps | 48 | 33.9034 | 3.1236 | 3.7549 | 3.2508 | fail: UTMOS |
| BigVGAN mel ensemble + transforms · 500 steps | 48 | 78.7726 | 2.6038 | 2.7068 | 1.6528 | fail: DNSMOS, NISQA, UTMOS |

### Decoder qualification

| Condition | N | WER % | DNSMOS | NISQA | UTMOS | Quality floors |
|---|---:|---:|---:|---:|---:|---|
| BigVGAN reconstruction | 48 | 3.9235 | 3.2981 | 4.0521 | 3.9868 | pass |
| Clean speech | 48 | 4.0241 | 3.2721 | 4.0533 | 4.0259 | pass |
| Original feature method · 50 steps · LR 0.02 | 48 | 49.7988 | 3.1739 | 4.0698 | 3.5789 | pass |
| HiFi-GAN reconstruction | 48 | 4.0241 | 3.3002 | 4.1716 | 4.0131 | pass |

### Quality and efficiency pilot

| Condition | N | WER % | DNSMOS | NISQA | UTMOS | Quality floors |
|---|---:|---:|---:|---:|---:|---|
| FV50 · decoder graph reused | 48 | 83.9034 | 3.1056 | 3.9766 | 3.1981 | fail: UTMOS |
| FV50 · STFT round-trip control | 48 | 83.9034 | 3.1056 | 3.9766 | 3.1981 | fail: UTMOS |
| FV50 · 25% reconstruction-spectrum anchor | 48 | 72.8370 | 3.1580 | 4.0272 | 3.3604 | fail: UTMOS |

## 24-recording development cohort

### Word-risk objective

| Condition | N | WER % | DNSMOS | NISQA | UTMOS | Quality floors |
|---|---:|---:|---:|---:|---:|---|
| Word-risk with deletion candidate | 24 | 58.5263 | 3.1778 | 3.9957 | 3.3497 | fail: UTMOS |
| Clean speech | 24 | 3.5789 | 3.3323 | 4.0838 | 4.1886 | pass |
| HiFi-GAN reconstruction | 24 | 3.5789 | 3.3317 | 4.2124 | 4.1667 | pass |

### Local feature mixtures

| Condition | N | WER % | DNSMOS | NISQA | UTMOS | Quality floors |
|---|---:|---:|---:|---:|---:|---|
| Local feature mixtures | 24 | 3.3684 | 3.3320 | 4.2142 | 4.1661 | pass |
| Local-mixture amplitude control | 24 | 4.8421 | 3.3020 | 4.1926 | 4.0837 | pass |
| Original strong control · 50 steps · LR 0.1 | 24 | 74.7368 | 3.1640 | 4.0350 | 3.3186 | fail: UTMOS |
| Clean speech | 24 | 3.5789 | 3.3323 | 4.0838 | 4.1886 | pass |
| HiFi-GAN reconstruction | 24 | 3.5789 | 3.3317 | 4.2124 | 4.1667 | pass |

### Generator checkpoint · 500 updates

| Condition | N | WER % | DNSMOS | NISQA | UTMOS | Quality floors |
|---|---:|---:|---:|---:|---:|---|
| Generator · trained · 500 updates | 24 | 3.5789 | 3.3322 | 4.0838 | 4.1885 | pass |

### Generator checkpoint · 1000 updates

| Condition | N | WER % | DNSMOS | NISQA | UTMOS | Quality floors |
|---|---:|---:|---:|---:|---:|---|
| Generator · trained · 1000 updates | 24 | 3.5789 | 3.3326 | 4.0838 | 4.1885 | pass |

### Generator checkpoint · 1500 updates

| Condition | N | WER % | DNSMOS | NISQA | UTMOS | Quality floors |
|---|---:|---:|---:|---:|---:|---|
| Generator · trained · 1500 updates | 24 | 3.5789 | 3.3318 | 4.0838 | 4.1886 | pass |

### Learned waveform generator

| Condition | N | WER % | DNSMOS | NISQA | UTMOS | Quality floors |
|---|---:|---:|---:|---:|---:|---|
| Generator · trained | 24 | 3.5789 | 3.3318 | 4.0838 | 4.1886 | pass |
| Clean speech | 24 | 3.5789 | 3.3323 | 4.0838 | 4.1886 | pass |
| Generator · untrained | 24 | 3.5789 | 3.3383 | 4.0808 | 4.1886 | pass |

### Historical original control · LR 0.02

| Condition | N | WER % | DNSMOS | NISQA | UTMOS | Quality floors |
|---|---:|---:|---:|---:|---:|---|
| Clean speech | 24 | 3.5789 | 3.3323 | 4.0838 | 4.1886 | pass |
| HiFi-GAN reconstruction | 24 | 3.5789 | 3.3317 | 4.2124 | 4.1667 | pass |
| Original feature method · 50 steps · LR 0.02 | 24 | 45.6842 | 3.2075 | 4.1245 | 3.6761 | pass |

### Historical original control · LR 0.05

| Condition | N | WER % | DNSMOS | NISQA | UTMOS | Quality floors |
|---|---:|---:|---:|---:|---:|---|
| Clean speech | 24 | 3.5789 | 3.3323 | 4.0838 | 4.1886 | pass |
| HiFi-GAN reconstruction | 24 | 3.5789 | 3.3317 | 4.2124 | 4.1667 | pass |
| Original feature method · 50 steps · LR 0.05 | 24 | 71.7895 | 3.1273 | 4.0778 | 3.4034 | fail: UTMOS |

### Historical original control · LR 0.1

| Condition | N | WER % | DNSMOS | NISQA | UTMOS | Quality floors |
|---|---:|---:|---:|---:|---:|---|
| Clean speech | 24 | 3.5789 | 3.3323 | 4.0838 | 4.1886 | pass |
| HiFi-GAN reconstruction | 24 | 3.5789 | 3.3317 | 4.2124 | 4.1667 | pass |
| Original feature method · 50 steps · LR 0.1 | 24 | 74.7368 | 3.1640 | 4.0350 | 3.3186 | fail: UTMOS |

## Coverage and unavailable results

Includes every completed source-summary.json under improvement plus the three historical 24-point original controls (LR 0.02, 0.05 and 0.1), including saved generator checkpoints. This is the complete development research catalogue, not an inventory of unrelated legacy reproduction runs. Canonical word-risk failures and generation-only smoke probes have no complete WER/MOS result and are not assigned synthetic scores.

The 76-recording validation, unseen-model transfer, defense comparisons, untouched confirmation/final cohorts and a blinded human study did not run. The frozen source/quality advancement gates selected stop. Shared development speakers and repeated tuning prevent treating these tables as final generalization evidence.

Raw rows: [CSV](all_results.csv). Reproducer: [build_audio_browser_data.py](build_audio_browser_data.py). Original experiment logs and audio remain preserved.

## Runtime and computational scalability

Quality and runtime have equal priority. See [joint cost/results note](21_quality_efficiency_results.md), [per-recording measured latency](execution_tradeoff/latency_per_recording.csv), [saved prior-method latency](execution_tradeoff/saved_latency_per_recording.csv), and [registered protocol](20_quality_efficiency_protocol.md). Historical loop costs exclude loading and preprocessing; new paired measurements separate these from optimization, repair, training (zero) and scoring. Forecasts are conditional and include the teacher, not just its cheap spectral repair.
