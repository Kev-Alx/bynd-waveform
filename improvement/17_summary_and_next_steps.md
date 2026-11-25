# What changed, what the results mean, and what comes next
## Latest model experiment

Direct-trained model status: **failed_execution**. Pilot selection: **unselected**, candidate `None`. Full measured quality/WER, inference and training costs: [25_learned_model_results.md](25_learned_model_results.md). Publication planning is deferred; the next direction follows the evidence.



**Latest execution update, 27 October 2026:** publication planning is deferred. A new direct-trained learned speech-feature model comparison is registered and detached/queued behind the existing GPU job. See [current experiment and live receipts](25_learned_model_results.md) and [fixed protocol](24_learned_feature_protocol.md). No new complete performance result is available yet. The 13-hour overall ceiling includes queue wait and ends no later than **28 October, 11:59 Asia/Taipei**. The controller profiles actual throughput before long-training admission, scores eight full 48-point conditions, and opens validation/transfer only if the fixed WER/quality and declared efficiency screens pass. Existing frozen experiments and the server on 8888 are preserved. This update takes precedence over older publication-planning language.


Updated 26 October 2026, Asia/Taipei. [Complete tables](16_all_results.md) · [All conditions as CSV](all_results.csv) · [Audio interface](http://localhost:8888/).

The expanded experiment finished at **22:02 on 24 October**, after **45 hours 51 minutes**, within its 48-hour ceiling. The code completed normally. The scientific result is negative: **no tested condition achieved WER above 75% while meeting all three fixed audio-quality thresholds**. Transferability and resistance to the named defenses have not been established for our candidates.

## Latest continuation: quality and runtime together

Updated 26 October 2026, 17:59 Asia/Taipei. **Audio quality and computational scalability are equal research priorities.** The new 48-point paired pilot completed in 42.4 minutes within its 45-minute ceiling. The pilot completed normally, but no condition passed the unchanged joint gate. Selection is scientific stop; candidate is null. Validation and transfer/defense tests remain gated off.

Reusing the decoder graph preserved every frozen raw WAV exactly and reduced matched generation-loop wall time by **14.63%**, with peak allocated memory 6.88 → 6.93 GiB. The fixed spectrum repair changed WER 83.90% → 72.84% and UTMOS 3.198 → 3.360. A speed gain is not a quality solution, and a quality gain must retain attack strength.

| Condition | N | WER % | DNSMOS | NISQA | UTMOS | Joint gate |
|---|---:|---:|---:|---:|---:|---|
| FV50, reused decoder graph | 48 | 83.9034 | 3.1056 | 3.9766 | 3.1981 | fail |
| STFT round-trip control | 48 | 83.9034 | 3.1056 | 3.9766 | 3.1981 | fail |
| 25% reconstruction-spectrum anchor | 48 | 72.8370 | 3.1580 | 4.0272 | 3.3604 | fail |

The generation search is still required for each new recording. Including measured feature extraction and repair, 1,000 similar new clips are forecast at 4.71 hours and ten audio hours of teacher data at 22.62 allocated GPU-hours, before additional corpus scoring or any student training. No reusable student was trained. See [complete joint results and cost breakdown](21_quality_efficiency_results.md); the interface now includes a **Runtime & scale** view and the new recordings.

Publication is also an explicit research objective: these engineering/repair changes alone do not establish novelty. See [publication positioning](22_publication_positioning.md) for the next-method contribution and evidence plan.

The historical expanded-cycle account follows.

## The original implementation

Think of the original method as editing an internal representation of a recording, then turning that representation back into speech. WavLM extracts the speech features; HiFi-GAN turns the edited features into audio. Whisper-small supplies the gradient telling the optimizer how to make recognition harder. This search happens separately for each recording. The models remain frozen.

The original code balances recognition disruption against feature/spectral penalties and limits how far the features can move. Its stronger learning-rate setting already creates many recognition errors, but loses audio quality. Our main reference therefore uses the quality-admissible original setting, not simply whichever setting has the highest WER.

On the 48-recording development cohort, that original 50-step quality control has **49.7988% WER**, DNSMOS **3.1739**, NISQA **4.0698**, and UTMOS **3.5789**.

## Changes we actually tested

| Change | What we changed in simple terms | Why | What happened |
|---|---|---|---|
| Word-risk objective (R1) | Asked the optimizer to favor word mistakes among candidate transcripts instead of relying only on token loss. | Make the objective closer to the WER we ultimately measure. | Native candidate sets sometimes had no useful word errors. A registered deletion-candidate repair reached 58.53% WER on 24 clips but failed UTMOS. |
| Local feature mixtures (R2) | Allowed only mixtures of nearby, similar speech features. | Keep edits closer to natural speech. | Quality was high, but WER stayed near clean speech: 3.37% on 24 clips. |
| Shared waveform generator (R3) | Trained a 1.59M-parameter U-Net to predict small audio edits using one hour of disjoint speech. | Learn reusable edits and avoid a fresh long optimization for every recording. | One seed and about one hour of training produced an almost-zero perturbation. WER stayed near clean speech. |
| Direct waveform optimization | Edited audio samples directly, with and without spectral/content penalties. | Test whether waveform attacks work even when the learned generator fails. | They work against the source recognizer. Full penalties gave 43.46% WER; removing them gave 58.75% but failed NISQA. |
| Longer weighted and hard-constrained waveform runs | Added 50 steps, either with weighted penalties or with hard limits on two distortion proxies. | Recover attack strength without increasing speech distortion. | Weighted100 reached 53.42% but failed NISQA. Hard constraints rejected most proposed updates and left WER at 43.46%. |
| Quiet-region protection | Allowed smaller waveform edits in quieter speech windows. | Avoid concentrating audible noise where the original speech is weak. | The primary version retained better quality but reached only 29.18% WER. It did not beat the original quality-admissible tradeoff. |
| Larger source-model ensemble | Combined Whisper-small, Whisper-large-v2 and Data2Vec CTC gradients. | Seek edits that affect more than one model and potentially transfer better. | The raw 500-step ensemble reached 90.44% source WER, but UTMOS was only 3.0557. No held-out transfer test was triggered. |
| Transformations during optimization | Included unchanged audio, filtering and resampling when computing attack gradients. | Encourage edits to survive those transformations. | The feature ensemble with transformations reached 89.54% WER at 500 steps, but UTMOS was 3.0850. This does not prove defense resistance. |
| Longer original-method controls | Ran the stronger original feature method for up to 1,500 steps. | Check whether apparent gains merely come from extra optimization or stronger settings. | WER reached 101.41%, with UTMOS 2.9437. These are known-method controls, not a new scientific contribution. |
| BigVGAN decoder and mel edits | Replaced the feature decoder with a 112.4M-parameter BigVGAN and optimized bounded mel-spectrogram changes. | Test whether a larger speech generator offers a better attack/quality tradeoff. | Reconstruction itself was good. The 200-step attack reached 80.18% WER but failed all three quality floors; 500 steps damaged quality further. |

The larger source models and BigVGAN were **frozen**, not newly trained or fine-tuned. A larger training corpus was prepared, but the ten-hour teacher/distillation branch has **not** been run. Increasing pilot data from 24 to 48 recordings improved the evidence coverage; it was not an independent final confirmation.

## How to read the results

WER measures mistakes made by the machine: substitutions, deletions and inserted words divided by the number of reference words. It is pooled over the cohort, so long sentences contribute more reference words. **WER can exceed 100%** when the recognizer inserts enough extra words; that is not a bug and is not a human-intelligibility score.

DNSMOS, NISQA and UTMOS are automatic estimates of audio quality; higher is better. The fixed thresholds apply to **the mean across the complete condition**: DNSMOS ≥ 3, NISQA ≥ 3.75 and UTMOS ≥ 3.5. Passing these proxies would still not prove that people can understand every original word.

The [complete tables](16_all_results.md) include **70 saved conditions and 2,808 scored recordings**, including generator checkpoints, reconstruction controls and the three historical original controls. They separate the 24-clip/475-word and 48-clip/994-word cohorts. Repeated controls are retained per study for completeness and must not be counted as independent replications. Generation-only engineering probes and failed objectives without complete scores are explicitly unavailable, not filled with invented values.

## What this tells us

We can force the source recognizer above 75% WER. The unresolved problems are **keeping the audio good enough and making the computation scale in practice**. More steps and bigger models alone did not solve it. BigVGAN's good reconstruction score also shows that a good decoder by itself does not guarantee that optimized attacks will preserve speech quality.

The expanded queue selected **stop** because no candidate passed the registered joint gate. Consequently, the 76-recording validation, unseen recognizers and preprocessing-defense comparisons were skipped. The untouched confirmation48/final200 cohorts and blinded human study remain unused. We cannot claim that the original method's transfer or defense behavior has been retained.

## Next steps, in order

1. **Use the audio browser to inspect the tradeoff.** Compare clean speech, the original quality control and stronger variants on the same sentences. Include both good and bad examples, rather than choosing only impressive attack transcripts. Check whether words remain understandable and whether the timbre, rhythm or speaker changes. Listening here is exploratory, not a blinded human score.
2. **Inspect why quality drops.** Use the saved trajectories and S/D/I counts to distinguish recognition confusion from erased/corrupted speech or long hallucinations. Check whether UTMOS failures correspond to audible damage or disagreement among quality proxies. The current evidence does not establish which explanation dominates.
3. **Register one quality/efficiency change before another long run.** The next study should test a specific content/quality and efficiency hypothesis, with measured throughput forecasts, a fixed baseline, compute budget and 48-point pilot. Repeating longer runs or lowering the quality floors would not resolve the present result. A larger learned policy should follow a successful per-recording teacher, rather than distilling a failed tradeoff.
4. **Advance only after a qualifying source result.** Freeze one candidate that passes WER > 75% and all quality floors, then run the 76-recording exposed-development validation. Use unseen recognizers and the registered preprocessing tests to measure transfer and persistence explicitly. Finally evaluate untouched data and conduct a blinded human intelligibility/quality study before making a paper-level improvement claim.

The earlier reporting/interface request launched only the audio server. The new continuation ran the separately registered paired pilot above; its explicit status, logs and exit receipts are under `execution_tradeoff/`.

## Listening interface and reproducibility

Open **http://localhost:8888/** through the existing Docker mapping, or use the Docker host's address with port 8888. The server binds `0.0.0.0:8888`. It is detached from the chat terminal.

- Choose one, two or three run panels. Each panel has a run name, description, full-run WER, quality status, the selected recording's audio, reference/recognizer transcript alignment, exact saved transcript, and per-recording metrics.
- Select the same sentence across methods; search by words, recording ID or speaker. A 24-recording method explicitly reports when a selected sentence is absent.
- Switch audio while retaining playback position, change playback speed, seek, and download the original WAV. Only one recording plays at a time so an A/B/C comparison does not mix signals.
- The **All results** view has cohort/study/quality filters, sorting and CSV export. **Changes & next steps** contains the plain-language interpretation.
- Playback uses original WAV bytes with no normalization, trimming, resampling or lossy conversion. PCM-identical recordings share an asset. Waveforms use a common amplitude scale. Reported scores remain the original saved scores.

Data preparation: `build_audio_browser_data.py`; verification receipt: `audio_browser_data_receipt.json`. Server: `serve_audio_browser.py`; status: `audio_browser_server_status.json`; log: `audio_browser_server.log`; process/exit receipt: `audio_browser_server_job.json`. All new files remain under `/workspace/improvement`.
