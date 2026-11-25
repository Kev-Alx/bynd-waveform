"""Stop this AISHELL trial at a saved epoch boundary, without starting follow-up jobs."""
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reproduction-runs/mandarin-cer-full-dev"
TRAINING = OUT / "training"
POLICY = OUT / "stop-after-epoch-2.json"
TAIPEI = timezone(timedelta(hours=8))


def now():
    return datetime.now(TAIPEI).isoformat(timespec="seconds")


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".stop.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(2**20):
            result.update(chunk)
    return result.hexdigest()


def process_state(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except FileNotFoundError:
        return None


def verify_process(pid, expected):
    command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
    assert expected in command, f"Refusing to signal a different process: {pid}"
    return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]


def terminate(pid, expected, start_ticks):
    if process_state(pid) in {None, "Z"}:
        return
    assert verify_process(pid, expected) == start_ticks, "Process identity changed"
    os.kill(pid, signal.SIGTERM)
    # A stopped process must run briefly to receive pending SIGTERM.
    os.kill(pid, signal.SIGCONT)
    deadline = time.monotonic() + 20
    while process_state(pid) not in {None, "Z"}:
        assert time.monotonic() < deadline, f"Process {pid} did not exit"
        time.sleep(.1)


def publish(policy, finished=False):
    log = OUT / "phases/training/stage.log"
    with log.open("rb") as stream:
        stream.seek(max(0, log.stat().st_size - 32000))
        tail = stream.read().decode(errors="replace")
    matches = list(re.finditer(r"(Fine-tune|Validating) epoch (\d+):[^\r\n]*?(\d+)/(\d+)\s*\[([^\]]*)\]", tail))
    observation = matches[-1].groups() if matches else None
    eta = None
    detail = "Waiting for epoch 2 training, its loss validation, and checkpoint writes."
    if observation:
        kind, epoch, position, total, timing = observation
        detail = f"{kind} epoch {epoch}: {position}/{total}; {timing}."
        rate_match = re.search(r"([\d.]+)(it|utterances)/s", timing)
        if rate_match and int(epoch) == 2:
            rate = float(rate_match.group(1))
            # Validation has no backward pass. Estimate it conservatively at the training rate.
            pending = int(total) - int(position) + (1435 if kind == "Fine-tune" else 0)
            eta = pending / rate + 60
    state = {"status": "stopped" if finished else "running", "phase": "stopped-after-epoch-2" if finished else "finishing-epoch-2-only",
             "updated_at": now(), "stop_after_epoch": 2, "automatic_followup_allowed": False,
             "latest_training_progress": detail, "estimated_remaining_seconds": 0 if finished else (round(eta) if eta else None),
             "eta_basis": "Current training/validation throughput plus checkpoint allowance; epoch 3 and decoded-CER evaluation are excluded."}
    if eta and not finished:
        state["estimated_completion_taipei"] = (datetime.now(TAIPEI) + timedelta(seconds=eta)).isoformat(timespec="seconds")
    if finished:
        state.update(completed_epochs=2, gpu_released=True, resume_state=str(TRAINING / "resume-state.json"))
    save(OUT / "progress.json", state)
    eta_text = "GPU released; no further GPU work is queued." if finished else (
        f"Estimated GPU release: **{state['estimated_completion_taipei']} (Asia/Taipei)**, about **{eta/60:.0f} minutes**."
        if eta else "GPU release ETA is being estimated from current progress.")
    text = (f"# AISHELL training stopped at epoch 2\n\nUpdated: {state['updated_at']} (Asia/Taipei).\n\n"
            f"Status: **{state['status']}**; phase **{state['phase']}**.\n\n{detail}\n\n{eta_text}\n\n"
            "User requested stopping after epoch 2 to use the GPU elsewhere. Epoch 3, decoded-CER checkpoint selection, "
            "the attack pilot, and the requested 2,000-record evaluation will not start automatically.\n\n"
            "Epoch 2 model weights, optimizer state, original configuration, data manifests, and history are retained "
            "for a later explicit continuation to epoch 3. The original three-epoch plan remains as provenance.\n\n"
            f"- [Stop policy]({POLICY})\n- [Training history]({TRAINING}/history.json)\n")
    if finished:
        text += f"- [Resume state]({TRAINING}/resume-state.json)\n- [Epoch 3 continuation notes]({OUT}/RESUME_EPOCH_3.md)\n"
    (OUT / "JOB_STATUS.md").write_text(text)
    (ROOT / "reproduction-runs/mandarin-cer-corrected/WORKFLOW_STATUS.md").write_text(text)


def main():
    policy = read(POLICY)
    worker, controller = policy["worker_pid"], policy["controller_pid"]
    worker_start = verify_process(worker, "scripts/finetune.py")
    controller_start = verify_process(controller, "scripts/run_mandarin_full_dev_trial.py")
    assert process_state(controller) in {"T", "t"}, "Controller must be suspended before arming"
    policy.update(status="waiting-for-epoch-2-checkpoint", watcher_pid=os.getpid(), watcher_started_at=now())
    save(POLICY, policy)
    last_publish = 0
    while True:
        history = read(TRAINING / "history.json")
        assert max(row["epoch"] for row in history) <= 2, "Epoch 3 already saved; review before stopping"
        if any(row["epoch"] == 2 for row in history):
            assert verify_process(worker, "scripts/finetune.py") == worker_start
            os.kill(worker, signal.SIGSTOP)
            break
        assert process_state(worker) not in {None, "Z"}, "Training ended before the requested checkpoint"
        if time.monotonic() - last_publish >= 20:
            publish(policy)
            last_publish = time.monotonic()
        time.sleep(.05)

    # history.json is written last at the boundary: weights, optimizer, and epoch index are durable.
    snapshots = read(TRAINING / "epoch-checkpoints.json")
    checkpoint = next(row for row in snapshots if row["epoch"] == 2)
    assert digest(Path(checkpoint["path"])) == checkpoint["sha256"]
    optimizer = TRAINING / "optimizer.pt"
    assert optimizer.exists() and optimizer.stat().st_size > 0
    preserved_optimizer = TRAINING / "optimizer-epoch-002.pt"
    if not preserved_optimizer.exists():
        os.link(optimizer, preserved_optimizer)
    # Terminate only this trial's worker and controller; their log supervisors reap them normally.
    terminate(worker, "scripts/finetune.py", worker_start)
    terminate(controller, "scripts/run_mandarin_full_dev_trial.py", controller_start)

    import torch
    saved = torch.load(preserved_optimizer, map_location="cpu", weights_only=False)
    assert saved["epoch"] == 2 and saved["optimizer"]["state"], "Optimizer is not an epoch 2 continuation state"
    del saved
    resume = {"status": "saved; not scheduled", "saved_at": now(), "completed_epochs": 2, "next_epoch": 3,
              "checkpoint": checkpoint, "optimizer": str(preserved_optimizer), "optimizer_sha256": digest(preserved_optimizer),
              "training_config": str(OUT / "training-config.json"), "training_config_sha256": digest(OUT / "training-config.json"),
              "train_manifest": str(OUT / "manifests/finetune.jsonl"), "train_manifest_sha256": digest(OUT / "manifests/finetune.jsonl"),
              "validation_manifest": str(OUT / "manifests/checkpoint-validation.jsonl"),
              "validation_manifest_sha256": digest(OUT / "manifests/checkpoint-validation.jsonl"),
              "history": history, "next_epoch_shuffle_seed": 2028, "automatic_followup_allowed": False,
              "rng_state_saved": False, "resume_note": "Continue model and AdamW state for one more epoch; no bitwise RNG-continuation guarantee. Existing finetune CLI does not restore optimizer state automatically."}
    save(TRAINING / "resume-state.json", resume)
    metadata = read(TRAINING / "training.json")
    save(TRAINING / "training-before-user-stop.json", metadata)
    metadata.update(status="stopped", stop_reason="User requested finish epoch 2 and release GPU", completed_epochs=2,
                    history=history, epoch_checkpoints=snapshots, resumable=True, resume_state=str(TRAINING / "resume-state.json"),
                    stopped_at=now(), best_sha256=digest(TRAINING / "best.pt"))
    save(TRAINING / "training.json", metadata)
    (OUT / "RESUME_EPOCH_3.md").write_text(
        "# Continue epoch 3 later\n\nNo continuation or evaluation is queued.\n\n"
        "The saved `training/resume-state.json` records the SHA-256 hashes and paths of epoch 2 model weights, "
        "the epoch 2 AdamW optimizer state, original configuration and unchanged data manifests. "
        "The original three-epoch configuration is retained; the effective completed epoch count is two.\n\n"
        "For a later explicit continuation, load `epoch-002.pt` and restore the optimizer from "
        "`optimizer-epoch-002.pt`, keep the original optimizer settings and data, and train only epoch 3. "
        "The zero-based epoch index is 2 and deterministic data-order seed is 2028. Save the continuation "
        "to a fresh output directory and keep these epoch 2 artifacts intact.\n\n"
        "The current finetune CLI does not restore optimizer state automatically. Setting its epochs to 1 "
        "would start a new optimization run; a continuation must explicitly restore the optimizer. "
        "Model and optimizer state are preserved, but RNG state was not captured by the original trainer, "
        "so bitwise identity to an uninterrupted third epoch is not promised.\n")
    # Retain real signal exit codes in the wrapper receipts and explain the intentional stop.
    for path in [OUT / "job.json", OUT / "phases/training/job.json"]:
        deadline = time.monotonic() + 15
        while read(path).get("status") == "running" and time.monotonic() < deadline:
            time.sleep(.1)
        receipt = read(path)
        assert receipt["status"] != "running", f"Supervisor has not finalized {path}"
        receipt.update(status="stopped", stop_reason="User-requested epoch 2 boundary stop", completed_epochs=2,
                       stop_policy=str(POLICY), resumable=True)
        save(path, receipt)
    policy.update(status="stopped-after-epoch-2", stopped_at=now(), controller_suspended=False,
                  gpu_released=True, resume_state=str(TRAINING / "resume-state.json"))
    save(POLICY, policy)
    publish(policy, finished=True)
    print(json.dumps(policy, indent=2), flush=True)


if __name__ == "__main__":
    main()
