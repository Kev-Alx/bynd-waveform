# Study and configuration explanations

This companion to the weekly update gives two paragraphs for each of the 15 studies, eight pilot configurations and two full-scale runs. The first paragraph explains the purpose and method; the second explains the findings. Each row title links to its source report or result record.

WER measures transcription errors: higher attacked WER means a stronger attack. WER can exceed 100% when the recognizer inserts extra words. DNSMOS, NISQA and UTMOS estimate sound quality, with higher scores meaning better estimated quality. The paper reference values used in our reports are 75.43% WER, 2.954 DNSMOS, 3.526 NISQA and 3.590 UTMOS. These automatic scores are not human listening results, and the authors’ exact model, sample and scoring choices remain partly unresolved.

## The 15 studies

### 1. [Loss calculation + step size — paper-quality-research](/workspace/reproduction-runs/paper-quality-research/report.md)

This study tested whether the attack worked better when transcript losses were added together or averaged, and when Adam used different update sizes. Adding versus averaging also changes the balance between the attack and quality penalties. We used 12 clips per pilot, 100 per validation run and one predefined 999-clip test control.

Larger updates generally produced stronger attacks but lower UTMOS. One 100-clip setting reached 75.56% WER with only 3.246 UTMOS. The separate 999-clip control reached 64.74% WER and 3.358 UTMOS. Neither matched attack strength and all quality targets together, so changing these settings alone did not close the gap.

### 2. [Text case + update choices — paper-quality-refinement](/workspace/reproduction-runs/paper-quality-refinement/report.md)

This study explored choices the paper did not fully specify: capitalization of the transcript used during the attack, Adam versus another optimizer, high-frequency penalty strength and random starting changes. It used 12-clip pilots and 100-clip validation. Recognition settings and scoring references stayed fixed, so changing text case did not change how WER was counted.

Some settings looked strong on the small pilot but failed to deliver both strength and quality on validation. For example, a lowercase-transcript setting fell from 109.55% pilot WER to 66.34% validation WER. No candidate met the combined validation requirements, so this study did not advance to another 999-clip test.

### 3. [Vocoder + input normalization — paper-quality-contracts](/workspace/reproduction-runs/paper-quality-contracts/report.md)

This study checked whether the audio generator, called a vocoder, or scaling its inputs explained the quality gap. We compared the original generator with an alternative checkpoint and tried normalized inputs. Clean speech was first reconstructed without an attack to check compatibility, using 12-clip pilots and 100-clip validation.

Both normalized paths failed the reconstruction checks, so their remaining attack settings were skipped. The alternative raw-input generator also failed to establish a better attack/quality balance. We retained the original compatible encoder and generator pair because these changes did not produce a qualifying improvement.

### 4. [Three overnight experiments — paper-quality-overnight](/workspace/reproduction-runs/paper-quality-overnight/report.md)

The overnight queue tested three ideas: adding a starting timestamp to the attack prompt, reducing update sizes during optimization and using updates with a controlled gradient length. The experiments shared a fresh 24-clip pilot, and selected candidates were checked on 100 validation clips. The permitted overall feature change stayed fixed.

The small pilot sometimes gave encouraging results that did not hold on validation. For example, a decreasing-step-size setting reached 91.79% pilot WER but 68.58% validation WER. None met the combined validation targets, so the queue finished without a confirmation run or another 999-clip test.

### 5. [Smooth the saved perturbation — feature-delta-diagnostics](/workspace/reproduction-runs/feature-delta-diagnostics/report.md)

This study tested whether smoothing an already computed feature change could improve the resulting audio. It used saved changes from 24 clips, applied several smoothing widths and regenerated the audio without further attack optimization. Controls matched the size of the changes so we could distinguish smoothing from simply making the attack weaker.

Smoothing often improved some quality scores while sharply reducing attack strength. Once change sizes were matched, no smoothing option met the predefined improvement requirement. We stopped this direction because it did not show a useful quality gain while preserving enough attack strength.

### 6. [Check decoder token filters — decoder-alignment-diagnostics](/workspace/reproduction-runs/decoder-alignment-diagnostics/report.md)

This diagnostic checked whether the attack loss and the recognizer treated possible output tokens differently. The concern was that the attack might spend effort on tokens the recognizer later blocks. We examined saved clean, reconstructed and attacked audio from 24 clips under two prompt formats, without generating new attacks or measuring new WER/MOS scores.

Static token filtering explained only about 0.24% of the attack-related increase in transcript loss. That was far below the predefined threshold for pursuing this explanation. The result did not support replacing the loss calculation; it also did not demonstrate a sound-quality improvement.

### 7. [Average the final ten updates — paper-quality-finalization](/workspace/reproduction-runs/paper-quality-finalization/report.md)

This study tested whether averaging feature changes from optimization steps 41–50 produced a better result than using step 50 alone. Two fixed settings each ran on 24 clips. The averaged change was compared with a final-step control of the same size, so a smaller change could not be mistaken for a benefit from averaging.

The UTMOS improvements over the matched controls were only about 0.003 and 0.027. Both missed the required improvement of 0.05, so neither advanced to fresh confirmation. The six reported output sets include different outputs from shared attack trajectories; they are not six independent batches of optimization.

### 8. [GPU / Adam diagnostics — optimizer-mechanism-diagnostics-v2](/workspace/reproduction-runs/optimizer-mechanism-diagnostics-v2/report.md)

This diagnostic investigated why repeated runs with the same settings could produce different results. On three fixed clips, we repeated gradient calculations at identical inputs and examined how each loss component affected an Adam update. These local comparisons held the optimizer’s accumulated history fixed so the current gradient’s contribution could be measured.

Repeated GPU gradients differed under the older execution paths, motivating the repeatability work. Removing the quality-loss contribution changed the first projected Adam update by roughly 21–42%, showing that it affected optimization. These local probes did not measure final WER or quality after removing a loss throughout an entire attack.

### 9. [Repeatable frontend — deterministic-execution-validation](/workspace/reproduction-runs/deterministic-execution-validation/report.md)

This study built an equivalent way to pad the waveform before calculating the attack loss, allowing strict repeatable GPU execution. It checked that the new calculation matched the original forward values, then ran the same 50-update attack twice in separate processes on three fixed development clips. The objective and feature-change limit stayed fixed.

All 150 paired updates and the final audio matched exactly across the two runs. This established repeatability on those checked clips and in that environment. No new WER or quality scores were measured here; its value was providing a consistent basis for later experiments.

### 10. [Repeatable attack controls — paper-quality-stable-execution](/workspace/reproduction-runs/paper-quality-stable-execution/report.md)

After checking numerical repeatability, we reran three existing settings on the same 24-clip pilot: a strong attack, a smaller-step quality-focused attack and a timestamp-prompt attack. Each used the full objective and 50 updates. This established reference results for subsequent loss-removal and coordinate-weighting comparisons.

The strong setting reached 74.74% WER and 3.319 UTMOS. Smaller steps raised UTMOS to 3.676 but reduced WER to 45.68%; the timestamp setting reached 71.79% WER and 3.403 UTMOS. Consistent execution therefore preserved the central tradeoff rather than resolving it.

### 11. [HF pooling diagnostic — hf-pooling-saved-audio-20261020](/workspace/reproduction-runs/hf-pooling-saved-audio-20261020/report.md)

This diagnostic checked two ways to average high-frequency sound energy over time: a whole-clip measure and an average that gives each non-silent time window equal weight. We measured 216 saved WAV records covering 24 source clips, three settings and clean/reconstructed/attacked audio. No new attacks or neural quality scoring were performed.

Both measures generally showed less high-frequency energy in attacked audio than in the original clean audio. Their differences did not establish that the averaging choice explained the quality gap or would fix it. The study therefore provided no basis for promoting an alternative high-frequency loss.

### 12. [Remove loss components — paper-quality-stable-ablations](/workspace/reproduction-runs/paper-quality-stable-ablations/report.md)

This study tested what happened when quality penalties were removed throughout all 50 attack updates. On the same 24 clips, we removed the penalty for abrupt changes between feature frames, the high-frequency penalty or both. Other settings matched the strong full-objective control, allowing complete attack trajectories to be compared.

Removing both penalties raised WER from 74.74% to 78.74%, while UTMOS rose only from 3.319 to 3.350. Removing individual penalties also failed to close the quality gap. These results describe how the components behave on this pilot; the modified objectives omit required parts of the paper’s method.

### 13. [Check encoder execution — encoder-execution-diagnostics](/workspace/reproduction-runs/encoder-execution-diagnostics/report.md)

This diagnostic checked whether changing the encoder’s GPU execution mode changed the speech features used as attack inputs. We re-extracted features for all 24 clips under three modes and compared them with the frozen feature cache. Repeats on the first three clips also checked consistency within each mode.

Only the original mode reproduced the cached features exactly. The other modes produced repeatable features but differed from the original beyond the existing tolerance. We retained the original encoder mode so later comparisons would start from the same speech representation; no attack or quality improvement was measured here.

### 14. [Try adjacent feature layers — paper-quality-feature-layers-v2](/workspace/reproduction-runs/paper-quality-feature-layers-v2/report.md)

This study tested features from encoder layers 5 and 7 instead of layer 6, using the same 24 clips. The existing generator was trained for layer 6, so these were compatibility probes. Each alternative first had to reconstruct speech with acceptable quality before any attack could run.

Both alternatives failed the reconstruction-quality checks. Reconstruction UTMOS fell from 4.167 for layer 6 to 4.027 for layer 5 and 3.784 for layer 7. Their reconstruction WER remained low, but that did not compensate for the quality loss. Neither layer proceeded to attack generation.

### 15. [Weight Adam coordinates — paper-quality-coordinate-research](/workspace/reproduction-runs/paper-quality-coordinate-research/report.md)

This study changed how Adam distributed updates across the speech representation while retaining the full loss and the same overall feature-change limit. One version used the original waveform’s energy over time; the other used how much each feature dimension varied over time. Both were predefined and tested on the existing 24-clip pilot.

Time-based weighting reduced UTMOS by 0.414 and weakened the attack. Channel-based weighting improved UTMOS by 0.080 while losing only 0.63 WER percentage points. The channel result passed the study’s descriptive improvement screen, but still missed the paper’s quality target and requires fresh confirmation because the pilot has been reused.

## The eight pilot configurations

All eight configurations below use the same 24 source clips. Their results are development measurements. Each uses 50 updates and the same feature-change limit; the loss-removal rows deliberately change the objective. Results are listed as WER, DNSMOS, NISQA and UTMOS, matching the supplied table.

### 16. [strong-sum-lr-0.1](/workspace/reproduction-runs/paper-quality-stable-execution/report.md)

This is the main strong-attack reference. “Sum” means adding the transcript-token losses, and “lr-0.1” means an Adam learning rate of 0.1. It uses the full attack objective, including both quality penalties, with the validated execution path. The later loss-removal and coordinate experiments are compared against this control.

It reached 74.74% WER, 3.164 DNSMOS, 4.035 NISQA and 3.319 UTMOS. WER was close to the paper’s 75.43%, and DNSMOS/NISQA exceeded their reference values, but UTMOS remained below 3.590. This control therefore shows strong attack performance with an unresolved quality gap.

### 17. [quality-sum-lr-0.02](/workspace/reproduction-runs/paper-quality-stable-execution/report.md)

This uses the same full objective as the strong reference but reduces Adam’s learning rate from 0.1 to 0.02. The question was whether gentler updates would preserve sound quality while keeping the attack effective. The models, clip set, number of updates and permitted overall feature change stayed fixed.

It reached 45.68% WER, 3.207 DNSMOS, 4.125 NISQA and 3.676 UTMOS. All three quality scores exceeded the paper references, but WER was about 29.05 percentage points below the strong control. Smaller updates improved the measured quality at a substantial cost to attack strength.

### 18. [timestamp-sum-lr-0.05](/workspace/reproduction-runs/paper-quality-stable-execution/report.md)

This adds a fixed 0.00-second timestamp before the transcript used to calculate the attack loss and uses Adam learning rate 0.05. It tests whether this attack prompt and update size produce a better balance. Recognition decoding and scoring references remain unchanged; the timestamp is part of attack optimization.

It reached 71.79% WER, 3.127 DNSMOS, 4.078 NISQA and 3.403 UTMOS. UTMOS improved over the strong control, while WER fell, and both remained below their paper targets. Because this row changes the prompt and learning rate together, its comparison with the strong control does not isolate the timestamp’s effect.

### 19. [without-tv](/workspace/reproduction-runs/paper-quality-stable-ablations/report.md)

This starts from the strong reference and switches off the temporal-variation, or TV, penalty throughout optimization. TV penalizes abrupt changes between neighbouring frames of the speech features. The high-frequency penalty remains active. The experiment checks whether TV helps or interferes with the final attack/quality balance.

It reached 76.00% WER, 3.153 DNSMOS, 4.040 NISQA and 3.323 UTMOS. WER rose by 1.26 percentage points, but UTMOS increased by only 0.004 and still missed the target. This pilot result provides little quality benefit and cannot represent the full method because TV was removed.

### 20. [without-hf](/workspace/reproduction-runs/paper-quality-stable-ablations/report.md)

This switches off the high-frequency, or HF, penalty while keeping TV and the attack loss. The HF term penalizes the proportion of generated sound energy above 6 kHz relative to the original clean audio’s proportion. Removing it tests whether that constraint helps or harms the final result.

It reached 74.74% WER, 3.135 DNSMOS, 4.003 NISQA and 3.306 UTMOS. Aggregate WER matched the strong control, while all three quality means were slightly lower. On this pilot, removing HF offered no benefit; it also omitted a required component of the paper’s objective.

### 21. [without-perceptual](/workspace/reproduction-runs/paper-quality-stable-ablations/report.md)

This switches off both TV and HF, leaving the transcript attack loss to guide optimization within the same permitted feature-change limit. It tests what both quality penalties contribute across a complete attack. The optimizer, learning rate and number of updates match the strong reference.

It reached 78.74% WER, 3.156 DNSMOS, 4.086 NISQA and 3.350 UTMOS. WER rose by 4.00 percentage points and UTMOS by about 0.032, but UTMOS still missed 3.590 and DNSMOS fell slightly. Its stronger attack does not establish reproduction because both required quality penalties were removed.

### 22. [time_energy](/workspace/reproduction-runs/paper-quality-coordinate-research/report.md)

This uses the original audio’s energy in short time windows to scale the optimizer’s coordinates. Louder windows receive larger weights than quieter windows, with the aim of steering changes away from quiet portions. The full objective and overall feature-change limit remain the same; the experiment changes the path Adam takes through optimization.

It reached 71.79% WER, 3.144 DNSMOS, 4.087 NISQA and 2.904 UTMOS. Compared with the strong control, WER fell by 2.95 percentage points and UTMOS by 0.414. Although NISQA rose slightly, the weaker attack and lower UTMOS made this direction unsuccessful under the predefined screen.

### 23. [channel_std](/workspace/reproduction-runs/paper-quality-coordinate-research/report.md)

This scales the optimizer’s coordinates using how much each original speech-feature dimension varies over time. More variable dimensions receive larger weights. “Channel” means a dimension of the model’s speech representation, not a left or right audio channel. The full objective and overall feature-change limit remain unchanged.

It reached 74.11% WER, 3.186 DNSMOS, 4.064 NISQA and 3.399 UTMOS. All three quality means improved over the strong control; UTMOS gained 0.080 with only a 0.63-point WER loss. It passed the descriptive improvement screen but missed the joint paper targets and still needs confirmation on fresh clips.

## The two full-scale 999-audio runs

Both runs use exactly the same 999 utterances and 20,310 reference words, the same recorded encoder/generator choices and Whisper-small recognizer. Both use mean transcript loss, Adam learning rate 0.05, zero initialization and 50 updates. Their clean and reconstructed controls match at the precision shown in the weekly table, so the different attacked results do not come from changing the test clips.

### 24. [english-feature-test-999](/workspace/reproduction-runs/english-feature-test-999/report.md)

This is the older full-scale baseline, also used as the source audio for the defense tests. It allowed a relative feature-change radius of rho=0.5 and used quality-loss weight lambda=0.1. Its permitted radius was five times the paper-stated 0.1, while its quality penalties had one tenth of the paper-stated weight. The radius describes feature-space changes, not waveform volume.

The attacked audio reached 74.11% WER, 2.964 DNSMOS, 3.871 NISQA and 2.792 UTMOS. WER approached the paper reference, but UTMOS was substantially below 3.590. Because this run allowed larger changes and used weaker quality weighting than the paper states, it is a historical comparison rather than evidence of performance under the paper’s stated budget.

### 25. [paper-setting-test-999](/workspace/reproduction-runs/paper-quality-research/paper-setting-test-999/evaluation/summary.json)

This is the newer fixed control inside paper-quality-research. It reduces rho from 0.5 to 0.1 and raises lambda from 0.1 to 1, matching the paper-stated radius and quality weight. Mean transcript loss, Adam learning rate 0.05 and the 999 test IDs stay fixed. It also saves the final feature changes for verification; unspecified author choices remain unresolved.

The attacked audio reached 64.74% WER, 3.126 DNSMOS, 3.954 NISQA and 3.358 UTMOS. Compared with the older run, WER fell by 9.36 percentage points while UTMOS rose by 0.566; DNSMOS and NISQA also improved. It delivered better estimated quality with a weaker attack. Because rho and lambda changed together, this comparison cannot separate their individual effects; it still misses the joint paper targets.
