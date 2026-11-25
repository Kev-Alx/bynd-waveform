# Evaluation provenance of the Beyond Waveform attack baselines

Researched 2026-10-23. Scope: the five baseline attacks named in §5 of [Beyond Waveform Robustness, arXiv:2606.05678v1](https://arxiv.org/html/2606.05678v1#S5). This note distinguishes the original baseline papers from the new speech experiments reported by Beyond Waveform.

## Original papers and evaluation protocols

| Baseline and original paper | Data used in the original evaluation | Original metrics | Models and comparison implications |
|---|---|---|---|
| **PGD** — Madry et al., [Towards Deep Learning Models Resistant to Adversarial Attacks](https://arxiv.org/pdf/1706.06083v3), ICLR 2018; §5 and Tables 1–2, pp. 11–13; Appendix Table 5, p. 21. | MNIST and CIFAR-10 image evaluation datasets. Appendix Table 5 explicitly reports the full **10,000-image CIFAR-10 evaluation set**. The MNIST evaluation count is not explicit in the paper passages inspected. | Classification **accuracy (%)** on natural/adversarial examples; attack loss for optimization analyses. No speech WER/CER. | MNIST convolutional networks; CIFAR-10 ResNet and a 10× wider variant. These are image robustness experiments, so their numerical results cannot be read as speech attack WER. |
| **MI-FGSM** — Dong et al., [Boosting Adversarial Attacks with Momentum](https://arxiv.org/pdf/1710.06081), CVPR 2018; §4.1–4.2, p. 5. | **1,000 images**, spanning **1,000 classes**, randomly selected from **ILSVRC 2012 validation** and correctly classified by all seven tested models. | **Attack success rate (%)**, explicitly defined as the model's misclassification rate on adversarial images; white-box and transfer results. No speech WER/CER. | Inception-v3/v4, Inception-ResNet-v2, ResNet-v2-152, and three ensemble-adversarially-trained models. Different modality, data, models, and metric from Beyond Waveform. |
| **VMI-FGSM** — Wang and He, [Enhancing the Transferability of Adversarial Attacks through Variance Tuning](https://arxiv.org/pdf/2103.15571), CVPR 2021; §4.1–4.2, p. 5. | **1,000 images**, spanning **1,000 classes**, from **ILSVRC 2012 validation**; described as almost correctly classified by all tested models. | **Attack success / misclassification rate (%)**; white-box, transfer, ensemble, and defended-image-classifier evaluations. No speech WER/CER. | Inception-v3/v4, Inception-ResNet-v2, ResNet-v2-101, three adversarially trained classifiers, plus nine defense methods. The dataset source matches MI-FGSM's, but the papers do not establish identical selected image IDs. Neither original evaluates ASR. |
| **Muting Whisper** — Raina et al., [Muting Whisper: A Universal Acoustic Adversarial Attack on Speech Foundation Models](https://aclanthology.org/2024.emnlp-main.430.pdf), EMNLP 2024; §5.1, §6.1–6.2, Appendix A.1–A.2. | Learn prefix: LibriSpeech **dev-other**, **2,864 utterances / 5.3 h**. Main evaluation: **test-other**, **2,939 / 5.1 h**. Domain transfer: TED-LIUM3 **test**, **1,155 / 2.6 h**; Artie Bias **test**, **1,712 / 2.4 h**; MGB (evaluation count unspecified). FLEURS **test**, French/German/Russian/Korean; counts unspecified. | Main attack metrics: **muted utterances (%)** and **average transcription length in words**. Appendix Table 7 separately gives unattacked-model WER. | Eight Whisper variants: tiny/base/small/medium, each multilingual and English-only. A universal **0.64-second prefix** is learned per model. Original small-model muting success is **98.7%** with average length **0.15 words** (Table 1); **98.7% is not a WER**. |
| **SlothSpeech** — Haque et al., [SlothSpeech: Denial-of-service Attack Against Speech Recognition Models](https://www.isca-archive.org/interspeech_2023/haque23_interspeech.pdf), Interspeech 2023; §4.1–4.3, pp. 1276–1277. | **LibriSpeech, OpenSLR, VCTK**. Evaluation sample counts and splits are not specified. The OpenSLR reference [10] cites Korvas et al.'s English/Czech telephone-speech corpus; the paper does not identify evaluated language/split. | **Decoder token count**, **latency (ms)**, mean/max values and percentage increases; **L2/L∞ perturbation magnitude** for quality. The paper does not report WER. | Speech2Text, Whisper, Speech2Text2, with Hugging Face pretrained weights. The objective is increased decoding cost. Original optimization uses **100 iterations, c=1**. For LibriSpeech/Whisper, mean latency rises from **376 ms** to **637 ms (L2)** or **630 ms (L∞)**; these are not error rates. |

## Beyond Waveform's common evaluation

[§5](https://arxiv.org/html/2606.05678v1#S5), [Appendix A](https://arxiv.org/html/2606.05678v1#A1), and [Appendix D](https://arxiv.org/html/2606.05678v1#A4) specify common utterances, surrogates, targets, and defenses across attacks:

- English: **999 held-out LibriSpeech test-clean utterances**; raw Whisper-small surrogate. Train-clean-100 is preparation/fine-tuning data, not the evaluation split.
- Mandarin: **AISHELL-1 test**; evaluated count unspecified. Whisper-small surrogate fine-tuned on development data; defended targets trained on training data.
- Iterative attacks: **50 updates**; validation selects strongest surrogate error satisfying DNSMOS/NISQA/UTMOS thresholds of **2.5**. This is not equal-norm matching across attack spaces.

Table 1, raw Whisper-small results:

| Method | English WER (%) | Mandarin CER (%) |
|---|---:|---:|
| Clean | 4.75 | 6.29 |
| PGD | 61.06 | 69.01 |
| MI-FGSM | 77.65 | 79.87 |
| VMI-FGSM | 78.84 | 80.36 |
| Muting Whisper | 99.52 | 94.47 |
| SlothSpeech | 39.50 | 53.11 |
| Feature Attack | 75.43 | 72.25 |

These are **new measurements**, not copied original results. Exact utterance IDs, validation sizes, decoding/scoring details remain unspecified.

## Interpretation

Common evaluation supports comparing attacks within Beyond Waveform. Across original papers, modalities, subsets, goals, and metrics differ. Muting Whisper evaluates **test-other**, whereas Beyond Waveform evaluates a **test-clean subset**. SlothSpeech's LibriSpeech split is unspecified. The image papers supply no original ASR WER to match.

Do not mistake Muting Whisper's MGB description of **1,600 hours of challenge training data** for attack-evaluation size. Do not substitute standard corpus sizes or local reproduction sizes for unspecified paper sample counts.
