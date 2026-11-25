"""Keep one human-readable and JSON status file for the active research jobs.

Read-only with respect to the experiments; it does not change frozen inputs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import time

TAIPEI = timezone(timedelta(hours=8))
TERMINAL = {"complete", "failed", "interrupted", "stopped"}


def read(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def alive(pid):
    if not isinstance(pid, int):
        return None
    try:
        os.kill(pid, 0)
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
        return state != "Z"
    except ProcessLookupError:
        return False
    except PermissionError:
        return None
    except (OSError, IndexError):
        return False


def clip_progress(path):
    try:
        with path.open("rb") as stream:
            stream.seek(max(0, path.stat().st_size - 16384))
            tail = stream.read().decode("utf-8", errors="replace")
        matches = re.findall(r"(\d+)/(\d+)\s*\[", tail)
        if matches:
            done, total = map(int, matches[-1])
            return {"completed": done, "total": total, "percent": round(100*done/total, 1) if total else None}
    except OSError:
        pass
    return None


def snapshot(root, name, label):
    folder = root / name
    job, progress, results = [read(folder / n) for n in ("job.json", "progress.json", "results.json")]
    status = job.get("status", "pending")
    phase = progress.get("phase", "pending")
    process_alive = alive(job.get("child_pid")) if status == "running" else None
    if status == "running" and process_alive is False:
        # A child may finish between the first read and the liveness check.
        latest = read(folder / "job.json")
        if latest.get("status") in TERMINAL:
            job, status = latest, latest["status"]
        elif alive(job.get("supervisor_pid")) is True:
            status = "finishing"
        else:
            status = "stopped"
    if status == "complete" and progress.get("status") in {"failed", "interrupted"}:
        status = progress["status"]
    if status == "running" and phase.startswith("waiting-"):
        status = "waiting"
    phase_log = folder / "phases" / phase / "stage.log"
    if phase == "waiting-for-existing-lpf":
        phase_log = folder / "lpf-6000.log"
    if progress.get("clip_progress"):
        counts = progress["clip_progress"]
    elif phase == "waiting-for-defense-baseline":
        counts = None
    else:
        counts = clip_progress(phase_log)
    completed = results.get("rows", []) if name == "english-defenses-999" else results.get("results", [])
    if name.startswith("mandarin-cer-"):
        completed = [{"name": variant, **values} for variant, values in results.get("summary", {}).items()]
    return {"name": name, "label": label, "status": status, "reported_job_status": job.get("status"),
            "phase": phase, "clip_progress": counts, "process_alive": process_alive,
            "started_at": job.get("started_at"), "last_heartbeat": progress.get("updated_at"),
            "finished_at": job.get("finished_at"), "exit_code": job.get("exit_code"),
            "error": progress.get("error") or job.get("error"),
            "scope": progress.get("scope"),
            "generated_utterances": progress.get("generated_utterances"),
            "estimated_remaining_seconds": progress.get("estimated_remaining_seconds"),
            "estimated_remaining_range_seconds": progress.get("estimated_remaining_range_seconds"),
            "estimated_completion_taipei": progress.get("estimated_completion_taipei"),
            "eta_basis": progress.get("eta_basis"),
            "completed_results": [{k: row[k] for k in ("name", "role", "summary", "error_percent", "quality") if k in row}
                                  for row in completed],
            "job_file": str(folder / "job.json"), "progress_file": str(folder / "progress.json"),
            "log": job.get("log"), "phase_log": str(phase_log) if counts else None,
            "report": str(folder / "report.md")}


def update(root, research="paper-quality-research"):
    jobs = [snapshot(root, "english-defenses-999", "Defense tests"),
            snapshot(root, research, "WER and sound-quality research")]
    if (root / "mandarin-cer-evaluation" / "plan.json").exists():
        jobs.append(snapshot(root, "mandarin-cer-evaluation", "AISHELL-1 CER evaluation"))
    if (root / "mandarin-cer-next-3000" / "plan.json").exists():
        jobs.append(snapshot(root, "mandarin-cer-next-3000", "AISHELL-1 CER continuation (+3,000; 5,000 total)"))
    if (root / "mandarin-cer-corrected" / "plan.json").exists():
        jobs.append(snapshot(root, "mandarin-cer-corrected", "AISHELL-1 corrected CER (48-audio pilot first)"))
    if (root / "mandarin-cer-checkpoint-selection" / "plan.json").exists():
        jobs.append(snapshot(root, "mandarin-cer-checkpoint-selection", "Mandarin checkpoint CER selection (48 clips × 2 checkpoints first)"))
    if (root / "mandarin-cer-full-dev" / "plan.json").exists():
        jobs.append(snapshot(root, "mandarin-cer-full-dev", "Full-development Mandarin surrogate trial"))
        for continuation in sorted((root / "mandarin-cer-full-dev").glob("epoch-3-continuation-*")):
            if (continuation / "job.json").exists():
                jobs.append(snapshot(root, str(continuation.relative_to(root)), "Mandarin epoch 3 continuation (maximum 5 hours)"))
    sequence = read(root / "mandarin-cer-corrected" / "sequence.json")
    if sequence.get("status") in {"failed", "needs_review"} or any(j["status"] in {"failed", "interrupted", "stopped"} for j in jobs):
        overall = "needs_attention"
    elif sequence.get("status") == "running":
        overall = "running"
    elif all(j["status"] == "complete" for j in jobs):
        overall = "complete"
    else:
        overall = "running"
    receipts = list(root.glob("english-defenses-999/**/job.discord.json")) + list(root.glob(research + "/**/job.discord.json"))
    failed_notifications = sum(read(p).get("status") == "failed" for p in receipts)
    state = {"updated_at": datetime.now(TAIPEI).isoformat(timespec="seconds"), "timezone": "Asia/Taipei",
             "status": overall, "refresh_seconds": 30, "jobs": jobs,
             "failed_notification_receipts": failed_notifications,
             "corrected_sequence": {k: sequence[k] for k in ("status", "phase", "error", "tracked_supervisor_pid") if k in sequence},
             "note": "Available defense conditions only; full paper defense parity remains incomplete."}
    lines = ["# Job status", "", f"Updated: **{state['updated_at']} (Asia/Taipei)**. Overall: **{overall}**.", "",
             "Refreshes every 30 seconds while the updater is running.", "",
             "| Job | Status | Current phase | Clip progress | Completed results | Estimated remaining |",
             "|---|---|---|---|---:|---|"]
    for job in jobs:
        count = job["clip_progress"]
        counter = f"{count['completed']}/{count['total']} ({count['percent']}%)" if count else "—"
        remaining = job.get("estimated_remaining_seconds")
        eta = f"{remaining/3600:.1f} hours" if remaining is not None else "—"
        lines.append(f"| {job['label']} | {job['status']} | {job['phase']} | {counter} | {len(job['completed_results'])} | {eta} |")
    lines += ["", "Completed defense results include the historical raw control. Clip progress counts utterances, each with the configured clean/reconstruction/adversarial controls.", ""]
    if sequence:
        workflow = root / "mandarin-cer-corrected" / "WORKFLOW_STATUS.md"
        lines += [f"Corrected AISHELL workflow: **{sequence.get('status', 'unknown')}**; {sequence.get('phase', 'unknown')}. "
                  f"[Whole-workflow status and conditional ETA]({workflow}). Per-job ETAs below describe the current stage.", ""]
    for job in jobs:
        lines += [f"## {job['label']}", "", f"- Status record: [{job['job_file']}]({job['job_file']})",
                  f"- Phase/heartbeat: [{job['progress_file']}]({job['progress_file']})",
                  f"- Results: [report]({job['report']})"]
        if job["log"]:
            lines.append(f"- Live log: [{job['log']}]({job['log']})")
        if job.get("scope"):
            lines.append(f"- Scope: {job['scope']}")
        if job.get("generated_utterances") is not None:
            lines.append(f"- Test utterances generated: {job['generated_utterances']}")
        if job.get("estimated_completion_taipei") and job["status"] not in {"failed", "interrupted", "stopped"}:
            lines.append(f"- Estimated completion: **{job['estimated_completion_taipei']} (Asia/Taipei)**")
            lower, upper = job["estimated_remaining_range_seconds"]
            lines.append(f"- Approximate remaining range: {lower/3600:.1f}–{upper/3600:.1f} hours. Basis: {job['eta_basis']}")
        elif job.get("eta_basis") and job["status"] not in TERMINAL:
            lines.append(f"- Estimated completion: calculating. {job['eta_basis']}")
        if job["error"]:
            lines.append(f"- Last error: {job['error']}")
        if job["exit_code"] is not None:
            lines.append(f"- Exit code: {job['exit_code']}")
        lines.append("")
    lines += [f"Failed Discord delivery receipts: **{failed_notifications}**. Compute status is tracked separately.", "",
              state["note"], ""]
    root.mkdir(parents=True, exist_ok=True)
    for name, content in [("JOB_STATUS.json", json.dumps(state, indent=2, ensure_ascii=False) + "\n"),
                          ("JOB_STATUS.md", "\n".join(lines))]:
        path = root / name
        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_text(content, encoding="utf-8")
        temp.replace(path)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/workspace/reproduction-runs"))
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--research", default="paper-quality-research", help="Research directory name inside --root")
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / "job-status-updater.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Status updater is already running; no duplicate started.", flush=True)
            return
        while True:
            state = update(root, args.research)
            print(f"{state['updated_at']} {state['status']}: " + "; ".join(
                f"{j['name']}={j['status']}/{j['phase']}" for j in state["jobs"]), flush=True)
            if args.once or (all(j["status"] in TERMINAL for j in state["jobs"])
                             and state.get("corrected_sequence", {}).get("status") != "running"):
                return
            time.sleep(30)


if __name__ == "__main__":
    main()
