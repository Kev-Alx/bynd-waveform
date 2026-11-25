# Bounded feature-trajectory finalization study

19 October 2026. Study root: `/workspace/reproduction-runs/paper-quality-finalization`. Discord remains disabled. Joint WER and sound-quality matching is unresolved.

## Evidence and question

The saved-delta smoothing study found no directional improvement at matched norms. The subsequent [decoder-alignment diagnostic](/workspace/reproduction-runs/decoder-alignment-diagnostics/report.md) completed on all 24 pilot clips, three waveform variants and two prefixes. All 942 lexical targets survived native static suppression. Static suppression explained only **0.241647%** of the adversarial-minus-reconstruction lexical NLL increase; median adversarial suppressed mass was **0.0004703**. The predeclared screen failed. This does not support a filtered-softmax intervention; the original loss and recognition remain fixed.

This study tests a different author-unspecified choice: returning the terminal optimizer iterate versus an average of projected late iterates. It is a bounded experiment, not a demonstrated explanation of the quality gap or recovery of an author recipe. No optimizer schedule, loss term, feature/vocoder path, decoder, scorer, reference or stated attack budget changes.

## Two trajectories and matched outputs

The fixed settings are original-case, sum-NLL, constant Adam with zero initialization:

| Family | Attack conditioning | Learning rate |
|---|---|---:|
| Strong control | No-timestamps prefix | 0.10 |
| Timestamp lead | Fixed unscored initial 0.00 timestamp | 0.05 |

Both use **50 updates, rho=0.1, lambda=1, alpha=1 and 6000 Hz cutoff**. Each setting runs one fresh trajectory per clip and retains the projected perturbations from updates 41–50. These ten snapshots produce exactly three outputs:

1. **Terminal:** delta50 from the fresh trajectory.
2. **Average:** `torch.stack([delta41, ..., delta50]).mean(0)` in the saved float32 dtype.
3. **Terminal, norm matched:** delta50's direction, scaled to the average's Frobenius norm.

There are no extra optimization updates for derived outputs and no averaging-window search. Compare average against norm-matched terminal to isolate direction from reduced radius. Compare the matched terminal against the unscaled terminal to quantify the radius effect. Fresh terminal controls share the exact trajectory with their paired outputs; cached historical attacks are not substituted, since numerical variability between repeated optimization runs has been observed.

The shared trace always describes the actual 50 optimizer updates and ends at delta50. Its final radius must not be replaced with the average's radius. Each output records its own saved perturbation, norm, finalization rule, source snapshots and waveform identity. Arithmetic means of feasible perturbations remain in the convex global feature ball, subject to independent float32 checks. No waveform amplitude adjustment or q smoothing is applied.

## Pools, screening and stopping

Run both families on the existing 24-clip development pilot, yielding six paired output conditions from 48 actual trajectories. Before any new pilot output, sample and freeze **48 fresh dev-clean clips**, seed **20261019**, from the sorted eligible source index. Exclude all previous reconstruction-pilot, pilot, validation, 999-test and fresh-24 IDs **or** audio hashes. Record the union of exclusions, sampling recipe, source hashes, durations and speakers. This is a fresh development confirmation pool, not a final test set.

For each family, promote the average to fresh confirmation only if it gains **at least 0.05 UTMOS** over its norm-matched terminal control, loses **at most five WER percentage points**, and all three averaged MOS values exceed 2.5. The terminal reconstruction must also pass the existing reconstruction screen: WER at most clean+5 percentage points and all three quality anchors met. Both families finish the pilot before any promotion decision. The decision and score evidence are frozen; at most two families advance, always with both terminal controls and the average output.

All six pilot results are reported, including negative results and their controls. Passing this directional screen does not mean the paper's WER/quality targets were met. Confirmation uses the identical settings and one fresh 50-update trajectory per clip, without additional tuning. **No automatic historical-100 validation or 999-clip test follows.** If neither family passes, stop this finalization family rather than expanding the window or optimizer grid based on these outcomes.

## Verification and operation

The standalone [collector](/workspace/beyondwaveform/trajectory_diagnostics.py) leaves the existing attack source intact. Deterministic CPU tests require exact agreement with its terminal waveform, delta and every trace value, exactly 50 NLL calls, projected snapshots, frozen model parameters and detached clean features. This checks implementation equivalence; it does not establish bitwise CUDA reproducibility.

The [runner](/workspace/scripts/run_finalization_research.py) freezes the plan, source/runtime/model/scorer hashes, configurations and both pools. Real pretrained one-clip smoke checks cover both families and all three output types before launch. Verification independently re-extracts clean q, checks all ten snapshot radii, reconstructs the specified mean and norm-matched control, checks actual returned radii, re-vocodes each saved output and compares waveform samples. It also checks complete ID coverage, references, edit counts, corpus WER, utterance-MOS means, scorer channels/runtime, generation identity and all 50 finite trace entries. Terminal trace radius and each output's returned radius are reported separately.

The main job, phase records, research log and progress heartbeat are retained under the study root. The consolidated [status file](/workspace/reproduction-runs/JOB_STATUS.md) must track `--research paper-quality-finalization`. Launch with UTF-8 and `BEYONDWAVEFORM_NOTIFY_DISCORD=0`. Inspect actual job records and process state before restarting a completed or interrupted run.
