"""Detached epoch 3 continuation; never launches CER selection or attack evaluation."""
import argparse
from datetime import datetime, timedelta, timezone
import fcntl
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from beyondwaveform.io import atomic_json
from beyondwaveform.training_resume import resume_finetune

TAIPEI = timezone(timedelta(hours=8))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--deadline-unix", required=True, type=float)
    parser.add_argument("--max-runtime-hours", type=float, default=5)
    parser.add_argument("--save-reserve-seconds", type=float, default=600)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()

    def progress(state):
        updated = datetime.now(TAIPEI)
        terminal = state["status"] in {"complete", "stopped", "failed"}
        position, total = state["train_position"], state["train_count"]
        validation = state["validation_position"]
        elapsed = state["elapsed_seconds"]
        remaining = None
        if position >= 48 and elapsed > 0:
            rate = position / elapsed
            remaining = (total - position) / rate + (state["validation_count"] - validation) / (2 * rate) + 150
        if terminal:
            remaining = 0
        state.update(updated_at=updated.isoformat(timespec="seconds"), scope="Resume saved epoch 2 model and AdamW optimizer; finish epoch 3 only; five-hour wall-clock limit including saving",
                     clip_progress={"completed": position, "total": total, "percent": round(100 * position / total, 1)},
                     deadline_taipei=datetime.fromtimestamp(args.deadline_unix, TAIPEI).isoformat(timespec="seconds"),
                     automatic_followup_allowed=False, eta_basis="Observed continuation throughput; final loss check and checkpoint writes included approximately")
        if remaining is not None:
            state.update(estimated_remaining_seconds=round(remaining), estimated_remaining_range_seconds=[round(remaining * .7), round(remaining * 1.8)],
                         estimated_completion_taipei=(updated + timedelta(seconds=remaining)).isoformat(timespec="seconds"))
        atomic_json(output / "progress.json", state)
        eta = f"Estimated completion: **{state['estimated_completion_taipei']} (Asia/Taipei)**." if remaining is not None and not terminal else "Calibrating ETA from the first training recordings."
        if terminal:
            eta = "This GPU job has ended; no additional GPU work is queued."
        text = (f"# AISHELL epoch 3 continuation\n\nUpdated: {state['updated_at']} (Asia/Taipei).\n\n"
                f"Status: **{state['status']}**; phase: **{state['phase']}**.\n\n"
                f"Epoch 3 training: **{position}/{total}**; loss validation: **{validation}/{state['validation_count']}**.\n\n"
                f"{eta}\n\nMaximum runtime deadline: **{state['deadline_taipei']} (Asia/Taipei)**. "
                f"Training stops at an optimizer boundary with **{args.save_reserve_seconds/60:.0f} minutes reserved for saving** if the time budget runs short.\n\n"
                "The epoch 2 weights and AdamW optimizer are restored together. Original epoch 2 artifacts are preserved. "
                "The original trainer did not save RNG state, so bitwise identity to uninterrupted training is not promised; "
                "new continuation checkpoints include RNG state and the exact data position.\n\n"
                "Decoded-CER selection, attack pilots, and the 2,000-record evaluation will not start automatically.\n\n"
                f"- [Runtime policy]({output}/runtime-policy.json)\n- [Job receipt]({output}/job.json)\n"
                f"- [Training log]({output}/run.log)\n- [Training metadata]({output}/training/training.json)\n")
        if terminal and state["status"] != "failed":
            text += f"- [Saved continuation state]({output}/training/resume-state.json)\n"
        for path in [output / "JOB_STATUS.md", ROOT / "reproduction-runs/mandarin-cer-full-dev/JOB_STATUS.md",
                     ROOT / "reproduction-runs/mandarin-cer-corrected/WORKFLOW_STATUS.md"]:
            path.write_text(text)
        from update_job_status import update
        update(ROOT / "reproduction-runs", "paper-quality-coordinate-research")

    with (ROOT / "reproduction-runs/auxiliary-gpu.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        max_seconds = min(args.max_runtime_hours * 3600, args.deadline_unix - time.time())
        if max_seconds <= args.save_reserve_seconds:
            raise RuntimeError("Insufficient time remains to restore and save a continuation")
        result = resume_finetune(args.resume, output / "training", max_seconds=max_seconds,
                                 save_reserve_seconds=args.save_reserve_seconds, progress=progress)
        print(json.dumps({"status": result["status"], "completed_epochs": result["completed_epochs"],
                          "elapsed_seconds": time.time() - started, "checkpoint": result["checkpoint"]}), flush=True)


if __name__ == "__main__":
    main()
