# Independent acoustic checkpoint research

22 October 2026 (Asia/Taipei). Research subtask for the first defense implementation. No model weights, packages, datasets or archives were downloaded by this subtask, and no GPU computation was run. Only this note was created. Network checks read primary documentation, official model metadata and HTTP headers.

## Recommendation

**Use the official SpeechBrain TIMIT CRDNN phone-CTC checkpoint first. Its download is reachable from this container.** The official README's model link resolves to a 496,781,950-byte ZIP named `CTC.zip`; an HTTP HEAD request returned status 200 on 22 October 2026 at 17:35 Asia/Taipei. The browser tool could not access Dropbox, but `/opt/venv/bin/python` with `urllib.request` could. This removes the presumed need to fit a new phone head before the first feasibility screen. Archive contents, checkpoint compatibility and the label map still need inspection by the implementation task.

Use `speechbrain/asr-crdnn-rnnlm-librispeech` as the auditable log-Mel backbone fallback, with a newly trained phone head. It supplies pretrained subword acoustic evidence rather than pretrained phones. Keep its ordinary subword decoder as the matched extra-recognizer control. A Wav2Vec2 phone checkpoint is useful only as a disclosed SSL-family comparison.

## TIMIT: genuine supervised phone recognizer

The [official TIMIT CTC recipe README](https://github.com/speechbrain/speechbrain/tree/develop/recipes/TIMIT/ASR/CTC) identifies a 20 May 2022 release, validation PER 12.80%, test PER 14.78%, and its [official checkpoint archive](https://www.dropbox.com/sh/xjh9qrat7v8ssuq/AAA4inQHBzRcbAeB_I8GSXi_a?dl=1). These are the authors' TIMIT results, not reproduced results and not evidence of robustness to BeyondWaveform. TIMIT training data are obtained from LDC according to that README; inference with existing weights need not fetch the dataset.

The [current configuration](https://raw.githubusercontent.com/speechbrain/speechbrain/develop/recipes/TIMIT/ASR/CTC/hparams/train.yaml) specifies 16 kHz audio, FFT size 400, 40 Mel features, global normalization, CNN channels 128/256, four bidirectional recurrent layers with 512 units, two 512-unit DNN blocks, and a 40-class linear CTC head with blank index zero. It uses time pooling and default LiGRU recurrent layers; the [CRDNN implementation](https://raw.githubusercontent.com/speechbrain/speechbrain/develop/speechbrain/lobes/models/CRDNN.py) establishes those defaults, including time-pooling size two. **Use the archive's saved configuration rather than assuming current develop exactly matches the 2022 weights.**

The [training source](https://raw.githubusercontent.com/speechbrain/speechbrain/develop/recipes/TIMIT/ASR/CTC/train.py) computes `Fbank → normalize → model → output → log_softmax`. It persists phone IDs in `label_encoder.txt`, built from the training dataset. The necessary inference state is the model, output head, saved normalizer and original label encoder. A fabricated alphabetical phone-ID ordering would invalidate decoding and candidate scoring. The saved configuration/checkpoint key names must determine actual file locations.

The [TIMIT preparation source](https://raw.githubusercontent.com/speechbrain/speechbrain/develop/recipes/TIMIT/timit_prepare.py) defaults to the 39-phone reduction, including silence. It maps, for example, `ao→aa`, `zh→sh`, and closures to silence. Therefore CMUdict's ARPAbet inventory needs an explicit, documented conversion, and silence handling must be consistent. This conversion is lossy; lexical distinctions removed by the phone inventory cannot be recovered by acoustic confidence alone.

The [Fbank source](https://raw.githubusercontent.com/speechbrain/speechbrain/develop/speechbrain/lobes/features.py) uses a 25 ms window, 10 ms hop, STFT magnitude and Mel filterbank, without a pretrained SSL encoder. This satisfies the proposed independent feature route, subject to confirming the archived configuration. Architectural difference does not establish statistical or adversarial independence.

Archive accessibility is verified by headers only; its members, hashes and model-specific license were not inspected. [SpeechBrain's source license](https://raw.githubusercontent.com/speechbrain/speechbrain/develop/LICENSE) is Apache 2.0, but this alone does not prove the terms attached to every Dropbox checkpoint. Record any archive license/attribution material during extraction. The official Hugging Face API query `list_models(search='timit', author='speechbrain')` returned no models during this audit; this is a bounded search result, not a proof that no mirror exists.

## LibriSpeech CRDNN fallback and ordinary ensemble control

The [official model card](https://huggingface.co/speechbrain/asr-crdnn-rnnlm-librispeech) describes a supervised CRDNN with CTC/attention, a unigram subword tokenizer and an optional RNN language model. It declares Apache 2.0. The card reports WER 3.09% for its release but does not make that a validated clean-performance guarantee for this defense's clips. It cites SpeechBrain training commit `2abd9f01`.

The [hyperparameters](https://huggingface.co/speechbrain/asr-crdnn-rnnlm-librispeech/raw/main/hyperparams.yaml) specify 40-bin Fbank, global normalization, two CNN blocks 128/256, four bidirectional LSTM layers of 1,024 units, two DNN blocks of 512 units and 1,000 subword output classes. Time pooling is four. A frozen phone adaptation should take the encoder's 512-dimensional output and fit a new CTC head using training-only phone targets; keep the feature extractor/normalizer consistent and report this adaptation separately from a pretrained phone model. CTC input-length feasibility must be checked after downsampling.

Official [Hugging Face file metadata](https://huggingface.co/speechbrain/asr-crdnn-rnnlm-librispeech/tree/main) was additionally read with `HfApi.model_info(..., files_metadata=True)`; pin revision **`979a53a7a3f6c9291c02c040fd8ebfb2471cf8a3`**. Exact byte counts and LFS SHA-256 values:

| File | Bytes | SHA-256 |
|---|---:|---|
| `asr.ckpt` | 479,555,971 | `1e795c7e18f3bab6bd5f47060ab852233deb33d7d550e989994c8683901e18d5` |
| `normalizer.ckpt` | 1,409 | `e11bfd7dbe13a266d13c00f6ff042a00fdbd40f3f5973928f9b49c33da32b512` |
| `tokenizer.ckpt` | 253,217 | `37a6cba34cd520b33fd83612d5efc8ba7e351166541eb2726642bb3032234d31` |
| `lm.ckpt` (optional control) | 212,420,087 | `3f73e243f5f0eb070a05a2069ba5b9014232e926384cc7d5ba24cde060c84997` |

`asr.ckpt` packages the encoder, embedding, attention decoder, CTC head and sequence head together; it is not a phone checkpoint. Loading the full default inference YAML also constructs an RNNLM and large beam decoder. For encoder-only adaptation, load only the necessary saved modules through a local, audited configuration so the default language-model download and runtime do not become hidden costs. All caches and checkpoint paths must be explicitly rooted under `/workspace/defense`.

## Phone-free acoustic control: torchaudio Emformer RNN-T

The [official RNNT bundle documentation](https://docs.pytorch.org/audio/stable/generated/torchaudio.pipelines.RNNTBundle.html) exposes a LibriSpeech recognizer with feature extraction, RNN-T beam decoding and SentencePiece output. It documents a roughly 293 MB model download. This is an ordinary supervised recognizer suitable for measuring whether gains come from adding another acoustic model rather than from a phone bottleneck.

The [versioned bundle source](https://docs.pytorch.org/audio/2.7.0/_modules/torchaudio/pipelines/rnnt_pipeline.html) and the installed torchaudio 2.4.1 source agree on the stable assets: `models/emformer_rnnt_base_librispeech.pt`, `pipeline-assets/global_stats_rnnt_librispeech.json`, and `pipeline-assets/spm_bpe_4096_librispeech.model` under `https://download.pytorch.org/torchaudio/`. Settings are 16 kHz, FFT400, 80 Mel bins, hop160, 16-frame segments, four-frame right context, 4,096 BPE symbols plus blank4096. Its feature transformation includes gain scaling, a piecewise log function and fixed global normalization. Replacing these with a generic log-Mel transform changes the model input distribution.

The [model factory source](https://docs.pytorch.org/audio/2.7.0/_modules/torchaudio/models/rnnt.html) constructs an Emformer transcriber, recurrent predictor and joiner. RNN-T scores are not CTC posteriors: do not apply phone-CTC candidate scoring to this model. Use its decoded hypotheses or a proper transducer sequence score. The [torchaudio source license](https://raw.githubusercontent.com/pytorch/audio/v2.7.0/LICENSE) is BSD 2-Clause; checkpoint-specific attribution should be preserved when fetched.

## SSL phone fallback: clearly separate from the main hypothesis

[Meta's `facebook/wav2vec2-xlsr-53-espeak-cv-ft`](https://huggingface.co/facebook/wav2vec2-xlsr-53-espeak-cv-ft) is a real 16 kHz Wav2Vec2-Large-XLSR-53 model fine-tuned on multilingual Common Voice phonetic labels, declared Apache 2.0. It returns phonetic strings and requires a lexicon to map phones to words. Its [file listing](https://huggingface.co/facebook/wav2vec2-xlsr-53-espeak-cv-ft/tree/main) shows a 1.26 GB `pytorch_model.bin`, with vocabulary and processor configuration. eSpeak/IPA symbols must not be assumed equivalent to TIMIT/CMUdict phones.

This checkpoint has independent parameters from WavLM but belongs to the related waveform-SSL family. It can test whether phonetic supervision helps within that family. It cannot establish the planned independent log-Mel acoustic-evidence mechanism. Do not silently promote it as a substitute for the TIMIT route.

## Runtime and immediate implementation checks

Read-only inspection found the task runtime at `/opt/venv/bin/python`, with torch `2.4.1+cu121`, torchaudio `2.4.1+cu121`, transformers `4.46.3` and huggingface-hub `0.26.5`. The system `/usr` Python had none of those ML packages; the `python` command itself was absent in the shell. SpeechBrain, HyperPyYAML and SentencePiece were absent before the parent began provisioning. Treat later package state as mutable and record the actual environment used.

Recommended next steps, in order:

1. Fetch `CTC.zip` only into `/workspace/defense`; record the exact archive hash, enumerate paths safely and inspect the saved YAML, normalization state, head and label encoder. Do not fetch TIMIT data merely to run inference.
2. Build a minimal phone-posterior inference adapter; test clean/reconstruction/attack audio on development clips, then assess correct-transcript phone-CTC score and phone-path coverage. References remain evaluation-only.
3. Preserve differentiability through the acoustic feature extractor and model for later joint adaptive attacks; avoid a `no_grad` wrapper inside the scoring API. Inference-only convenience wrappers may freeze parameters while still allowing input gradients in a separate call.
4. Use clean data to validate symbol mapping and sequence-length accounting before interpreting attacked scores. Distinguish TIMIT's domain mismatch and phone-inventory loss from adversarial failure.
5. If archived weights are incompatible or incomplete, fit a new head on the pinned LibriSpeech CRDNN encoder using training-only supervision; report training hours, split and source. A short adapter run is a feasibility experiment, not evidence of a high-quality independent recognizer.

No defense has passed a clean-accuracy or robustness gate on the basis of this research note.
