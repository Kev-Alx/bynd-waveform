"""Evaluate epoch 2 and epoch 3 sequentially on one frozen 100-audio manifest."""
import argparse
from datetime import datetime, timedelta, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from beyondwaveform.io import atomic_json, digest, read_jsonl

TAIPEI = timezone(timedelta(hours=8))
BASE = ROOT / "reproduction-runs/mandarin-cer-full-dev"


def run(output):
    selection = json.loads((output / "selection-plan.json").read_text())
    manifest = output / "validation-100.jsonl"
    assert digest(manifest) == selection["manifest_sha256"]
    assert len(read_jsonl(manifest)) == 100
    targets = {
        2: {"metadata": BASE / "training/training.json", "checkpoint": BASE / "training/epoch-002.pt"},
        3: {"metadata": BASE / "epoch-3-continuation-20261027/training/training.json",
            "checkpoint": BASE / "epoch-3-continuation-20261027/training/model.pt"},
    }
    plan = {"status": "frozen", "selection_plan_sha256": digest(output / "selection-plan.json"),
            "manifest_sha256": digest(manifest), "epochs": [2, 3], "audio_count": 100,
            "source_sha256": digest(__file__), "evaluator_sha256": digest(ROOT / "scripts/run_mandarin_clean_pilot.py"),
            "targets": {str(epoch): {"checkpoint": str(target["checkpoint"]), "checkpoint_sha256": digest(target["checkpoint"]),
                                    "metadata_sha256": digest(target["metadata"])} for epoch, target in targets.items()}}
    if (output / "plan.json").exists():
        assert json.loads((output / "plan.json").read_text()) == plan
    else:
        atomic_json(output / "plan.json", plan)
    started = time.monotonic()
    summaries = {}
    completed = 0

    def publish(epoch, within=0, status="running"):
        done = completed + within
        now = datetime.now(TAIPEI)
        remaining = (200 - done) * ((time.monotonic() - started) / done if done else .5)
        if status == "complete":
            remaining = 0
        progress = {"status": status, "phase": f"epoch-{epoch}-clean-validation" if status == "running" else status,
                    "updated_at": now.isoformat(timespec="seconds"), "scope": "Clean recognition: epoch 2 and epoch 3, the same 100 fresh validation audios",
                    "clip_progress": {"completed": done, "total": 200, "percent": round(done / 2, 1)},
                    "unique_audio_count": 100, "epoch": epoch, "estimated_remaining_seconds": round(remaining),
                    "estimated_remaining_range_seconds": [round(.7 * remaining), round(1.8 * remaining)],
                    "estimated_completion_taipei": (now + timedelta(seconds=remaining)).isoformat(timespec="seconds"),
                    "eta_basis": "Observed end-to-end scoring throughput; initially approximately 0.5 seconds per audio from the 24-audio pilots",
                    "automatic_followup_allowed": False}
        atomic_json(output / "progress.json", progress)
        state = (f"# Epoch 2 and 3 — 100-audio clean validation\n\nUpdated: {progress['updated_at']} (Asia/Taipei).\n\n"
                 f"Status: **{status}**; current checkpoint **epoch {epoch}**. Progress: **{done}/200** checkpoint/audio observations "
                 "(**100 unique audios**, each decoded once per epoch).\n\n")
        state += (f"Estimated completion: **{progress['estimated_completion_taipei']} (Asia/Taipei)**; approximately **{remaining/60:.1f} minutes** remaining.\n\n"
                  if status == "running" else "The suite has ended; no more GPU work is queued.\n\n")
        state += ("This run uses 100 fresh held-out validation recordings, excluding all 48 recordings in the earlier pilot pool. "
                  "Both checkpoints use the same manifest, reference transcripts, Mandarin decoder and scoring settings.\n\n"
                  f"- [Results and per-epoch reports]({output}/report.md)\n- [Frozen data selection]({output}/selection-plan.json)\n"
                  f"- [Job receipt]({output}/job.json)\n- [Live log]({output}/run.log)\n")
        for path in [output / "JOB_STATUS.md", BASE / "JOB_STATUS.md", ROOT / "reproduction-runs/mandarin-cer-corrected/WORKFLOW_STATUS.md"]:
            path.write_text(state)
        text = (f"# Epoch 2 and 3: 100-audio clean validation\n\nStatus: **{status}**. Updated: {progress['updated_at']} (Asia/Taipei).\n\n"
                "| Epoch | Audios | Clean CER | Clean WER (jieba) |\n|---|---:|---:|---:|\n")
        for candidate in [2, 3]:
            if candidate in summaries:
                s = summaries[candidate]
                text += f"| {candidate} | 100 | {s['clean_cer']['error_percent']:.4f}% | {s['clean_wer_jieba']['error_percent']:.4f}% |\n"
            else:
                text += f"| {candidate} | — | pending | pending |\n"
        text += ("\nCER is the Mandarin comparison metric. WER uses jieba 0.42.1 with HMM disabled on identically normalized references and hypotheses. "
                 "Each corpus rate sums errors and reference units before dividing.\n\n"
                 f"Selection: {selection['selection']}. The same fixed 100 fresh records are used for both epochs; all 48 previous pilot records are excluded. "
                 "These validation speakers, IDs and audio hashes are disjoint from the surrogate's fine-tuning data. "
                 "References retain the official AISHELL transcript strings.\n\n"
                 "Decoding: forced Mandarin, beam 5, temperature 0, timestamps disabled, fp32, previous-text conditioning disabled.\n\n"
                 f"- [Data selection and manifest hash]({output}/selection-plan.json)\n- [Validation manifest]({manifest})\n")
        for candidate in [2, 3]:
            text += f"- [Epoch {candidate} report and all transcriptions]({output}/epoch-{candidate}/report.md)\n"
        if status == "complete":
            text += f"- [Cross-checkpoint comparison audit]({output}/comparison-audit.json)\n"
        (output / "report.md").write_text(text)

    for epoch, target in targets.items():
        folder = output / f"epoch-{epoch}"
        folder.mkdir(exist_ok=True)
        command = [sys.executable, "-u", str(ROOT / "scripts/run_logged.py"), "--log", str(folder / "run.log"),
                   "--status", str(folder / "job.json"), "--", sys.executable, "-u", str(ROOT / "scripts/run_mandarin_clean_pilot.py"),
                   "--checkpoint", str(target["checkpoint"]), "--training-metadata", str(target["metadata"]),
                   "--epoch", str(epoch), "--manifest", str(manifest), "--selection-note", selection["selection"],
                   "--output", str(folder), "--tokenizer-path", str(ROOT / "artifacts/python/jieba-0.42.1"), "--local-status-only"]
        publish(epoch)
        child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                 env={**os.environ, "BEYONDWAVEFORM_NOTIFY_DISCORD": "0"})
        try:
            while child.poll() is None:
                path = folder / "progress.json"
                within = json.loads(path.read_text())["clip_progress"]["completed"] if path.exists() else 0
                publish(epoch, within)
                time.sleep(2)
            assert child.returncode == 0, f"Epoch {epoch} failed; inspect {folder / 'run.log'}"
        finally:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
        result = json.loads((folder / "results.json").read_text())
        audit = json.loads((folder / "independent-audit.json").read_text())
        assert result["status"] == "complete" and result["epoch"] == epoch and result["count"] == 100
        assert result["target"]["sha256"] == plan["targets"][str(epoch)]["checkpoint_sha256"]
        assert audit["status"] == "passed" and audit["count"] == 100
        assert digest(folder / "validation-100.jsonl") == plan["manifest_sha256"]
        summaries[epoch] = result
        completed += 100
        atomic_json(output / "results.json", {"status": "complete" if completed == 200 else "partial", "audio_count": 100,
                    "epochs": {str(key): value for key, value in summaries.items()}, "plan_sha256": digest(output / "plan.json")})
    records = {epoch: read_jsonl(output / f"epoch-{epoch}/utterances.jsonl") for epoch in [2, 3]}
    assert [(r["id"], r["reference"], r["audio_sha256"]) for r in records[2]] == [(r["id"], r["reference"], r["audio_sha256"]) for r in records[3]]
    for key in ["language", "dimensions", "decoding", "nll_reduction"]:
        assert summaries[2]["target"][key] == summaries[3]["target"][key]
    atomic_json(output / "comparison-audit.json", {"status": "passed", "unique_audio_count": 100,
                "checkpoint_audio_observations": 200, "same_audio_references_and_decoder": True,
                "independent_character_distance_checks": 200, "independent_word_distance_checks": 200,
                "manifest_sha256": plan["manifest_sha256"], "results_sha256": digest(output / "results.json")})
    publish(3, status="complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    def interrupt(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    with (args.output / "suite.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args.output.resolve())
