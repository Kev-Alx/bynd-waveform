"""Prepare, detach, stop and resume the epoch 3 AISHELL evaluation."""
import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT / "reproduction-runs/aishell-epoch3-2000"
FIVE_THOUSAND = ROOT / "reproduction-runs/aishell-epoch3-5000"
DEFAULT = ROOT / "reproduction-runs/aishell-epoch3-7176"
PYTHON = "/opt/venv/bin/python"
WORKER = ROOT / "scripts/run_aishell_epoch3.py"
EXTENDED_WORKER = ROOT / "scripts/run_aishell_epoch3_extended.py"
RUNTIME_WORKER = ROOT / "scripts/run_aishell_epoch3_full_run.py"
COMPLETE_TEST_WORKER = ROOT / "scripts/run_aishell_epoch3_complete_test.py"
TAIPEI = timezone(timedelta(hours=8))


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".control.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(2**20):
            result.update(chunk)
    return result.hexdigest()


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def active_pid(output):
    launch = read(output / "launch.json")
    pid = launch.get("pid")
    if not isinstance(pid, int):
        return None
    try:
        stat = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
        worker = Path(launch.get("worker_script", read(output / "plan.json").get("worker_script", str(WORKER))))
        if stat[0] != "Z" and stat[19] == launch["start_ticks"] and str(worker) in command and str(output) in command:
            return pid
    except FileNotFoundError:
        pass
    return None


def original_five_thousand(first, following, full):
    assert len(first) == 2000 and len(following) == 3000 and len(full) == 5000
    selected = first + following
    assert len({row["id"] for row in selected}) == 5000, "Original subsets overlap"
    assert len({row["audio_sha256"] for row in selected}) == 5000, "Duplicate source audio"
    official = {row["id"]: row for row in full}
    assert all(official.get(row["id"]) == row for row in selected), "Original 5,000 reference rows differ"
    return selected


def remaining_official_test(completed, official):
    assert len(completed) == 5000 and len(official) == 7176
    assert len({row["id"] for row in completed}) == 5000
    assert len({row["id"] for row in official}) == 7176, "Duplicate official test IDs"
    assert len({row["audio_sha256"] for row in official}) == 7176, "Duplicate official test audio"
    by_id = {row["id"]: row for row in official}
    assert all(by_id.get(row["id"]) == row for row in completed), "Completed references differ from official test"
    used = {row["id"] for row in completed}
    remaining = [row for row in official if row["id"] not in used]
    assert len(remaining) == 2176
    return remaining


def prepare_remaining(output, parent=None):
    parent = FIVE_THOUSAND if parent is None else parent
    assert not active_pid(parent), "Parent evaluation is still running"
    with (parent / "control.lock").open("a+") as control, (parent / "worker.lock").open("a+") as worker_lock:
        fcntl.flock(control, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(worker_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = read(parent / "plan.json")
        job = read(parent / "job.json")
        previous = read(parent / "results.json")
        audit = read(parent / "independent-audit.json")
        assert plan["count"] == 5000 and plan["epoch"] == 3
        assert job["status"] == previous["status"] == "complete" and audit["status"] == "passed"
        assert job["generated"] == audit["generation_records"] == 5000
        assert job["scored_variants"] == audit["character_distance_checks"] == audit["word_distance_checks"] == 15000
        for name, expected in {**plan["input_hashes"], **plan["source_hashes"]}.items():
            assert sha(name) == expected, f"Frozen parent input changed: {name}"
        old_evaluation = ROOT / "reproduction-runs/mandarin-cer-evaluation/plan.json"
        full_info = read(old_evaluation)
        official_path = Path(full_info["full_test_manifest"])
        assert sha(official_path) == full_info["full_test_manifest_sha256"] == plan["input_hashes"][str(official_path)]
        completed = rows(parent / "manifest.jsonl")
        official = rows(official_path)
        remaining = remaining_official_test(completed, official)
        expected_ids = {row["id"] for row in official}
        wav_root = ROOT / "data/data_aishell/wav/test"
        assert {path.stem for path in wav_root.rglob("*.wav")} == expected_ids, "Full manifest differs from official test directory"
        assert all(row["split"] == row["role"] == "test" and row["language"] == "zh" for row in official)
        assert all(Path(row["audio"]).is_relative_to(wav_root) for row in official)
        truth_path = ROOT / "data/data_aishell/transcript/aishell_transcript_v0.8.txt"
        truth = dict(line.split(maxsplit=1) for line in truth_path.read_text().splitlines() if line.strip())
        assert all(row["text"] == truth[row["id"]].strip() for row in official)
        for filename in ["finetune.jsonl", "checkpoint-validation.jsonl", "attack-pilot-48.jsonl"]:
            pool = rows(ROOT / "reproduction-runs/mandarin-cer-full-dev/manifests" / filename)
            for field in ["id", "speaker", "audio_sha256"]:
                assert not {row[field] for row in official} & {row[field] for row in pool}, f"Test overlap in {field}"
        utterances_path = parent / "utterances.jsonl"
        assert sha(utterances_path) == audit["utterances_sha256"]
        utterances = rows(utterances_path)
        expected_scores = {(row["id"], variant) for row in completed for variant in ["clean", "reconstruction", "adversarial"]}
        assert len(utterances) == 15000 and {(item["id"], item["variant"]) for item in utterances} == expected_scores
        sources = {row["id"]: row for row in completed}
        for variant in ["clean", "reconstruction", "adversarial"]:
            items = [item for item in utterances if item["variant"] == variant]
            assert previous["summary"][variant]["count"] == len(items) == 5000
            assert all(item["target_sha256"] == plan["checkpoint_sha256"] and item["reference"] == sources[item["id"]]["text"] for item in items)
            for stored, metric in [("cer", "cer"), ("wer_jieba", "wer")]:
                for key in ["substitutions", "deletions", "insertions", "reference_units"]:
                    assert sum(item[metric][key] for item in items) == previous["summary"][variant][stored][key]
        (output / "config.json").write_bytes((parent / "config.json").read_bytes())
        (output / "manifest.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in remaining))
        inputs = [parent / name for name in ["plan.json", "manifest.jsonl", "config.json", "results.json", "independent-audit.json", "utterances.jsonl"]]
        source_paths = [RUNTIME_WORKER, COMPLETE_TEST_WORKER]
        frozen = {**plan, "count": 2176, "official_test_count": 7176, "session_max_hours": None,
                  "runtime_worker": str(COMPLETE_TEST_WORKER), "worker_script": str(COMPLETE_TEST_WORKER),
                  "full_test_manifest": str(official_path), "manifest_policy": "All 2,176 official test recordings outside the completed 5,000; preserve official order and references",
                  "combined_parent": {"output": str(parent), "count": 5000, "plan_path": str(parent / "plan.json"),
                                      "plan_sha256": sha(parent / "plan.json"), "results_sha256": sha(parent / "results.json")},
                  "data_audit": {"official_references_exact": 7176, "training_validation_overlap": 0,
                                 "remaining_count": 2176, "completed_count": 5000, "official_test_directory_exact": True},
                  "source_hashes": {**plan["source_hashes"], **{str(path): sha(path) for path in source_paths}},
                  "input_hashes": {**plan["input_hashes"], **{str(path): sha(path) for path in inputs},
                                   str(output / "config.json"): sha(output / "config.json"), str(output / "manifest.jsonl"): sha(output / "manifest.jsonl")}}
        frozen.pop("resume_parent", None)
        save(output / "plan.json", frozen)
        save(output / "extension-audit.json", {"status": "passed", "official_test_count": 7176, "remaining_count": 2176,
             "completed_count": 5000, "official_directory_ids_exact": True, "official_references_exact": 7176,
             "training_validation_overlap": 0, "previous_score_counts_recomputed": 15000,
             "parent_plan_sha256": sha(parent / "plan.json"), "parent_results_sha256": sha(parent / "results.json"),
             "previous_audio_and_score_files_untouched": True})
        (output / "worker-source.py").write_bytes(COMPLETE_TEST_WORKER.read_bytes())
        for path in [output / "config.json", output / "manifest.jsonl", output / "plan.json", output / "worker-source.py"]:
            path.chmod(0o444)
        save(parent / "successor.json", {"output": str(output), "parent_plan_sha256": sha(parent / "plan.json")})


def prepare_extension(output, parent=LEGACY):
    assert not active_pid(parent), "Stop the parent worker before extending its scope"
    with (parent / "control.lock").open("a+") as control, (parent / "worker.lock").open("a+") as worker_lock:
        fcntl.flock(control, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(worker_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        parent_plan = read(parent / "plan.json")
        parent_sha = sha(parent / "plan.json")
        assert parent_plan["count"] == 2000 and parent_plan["epoch"] == 3
        for path, expected in {**parent_plan["input_hashes"], **parent_plan["source_hashes"]}.items():
            assert sha(path) == expected, f"Frozen parent input changed: {path}"
        following_path = ROOT / "reproduction-runs/mandarin-cer-next-3000/test-subset.jsonl"
        following_plan_path = following_path.parent / "plan.json"
        following_plan = read(following_plan_path)
        assert sha(following_path) == following_plan["test_manifest_sha256"]
        full_path = ROOT / "reproduction-runs/mandarin-cer-full-dev/manifests/test-5000.jsonl"
        data_plan_path = full_path.parent.parent / "data-plan.json"
        assert sha(full_path) == read(data_plan_path)["manifest_sha256"][str(full_path)]
        first = rows(parent / "manifest.jsonl")
        selected = original_five_thousand(first, rows(following_path), rows(full_path))
        assert set(Counter(row["speaker"] for row in selected).values()) == {250}
        assert all(row["role"] == "test" and row["split"] == "test" and row["language"] == "zh" for row in selected)
        first_ids = {row["id"] for row in first}
        truth_path = ROOT / "data/data_aishell/transcript/aishell_transcript_v0.8.txt"
        truth = dict(line.split(maxsplit=1) for line in truth_path.read_text().splitlines() if line.strip())
        assert all(row["text"] == truth[row["id"]].strip() for row in selected)
        for name in ["finetune.jsonl", "checkpoint-validation.jsonl", "attack-pilot-48.jsonl"]:
            pool = rows(ROOT / "reproduction-runs/mandarin-cer-full-dev/manifests" / name)
            for field in ["id", "speaker", "audio_sha256"]:
                assert not {row[field] for row in selected} & {row[field] for row in pool}
        (output / "config.json").write_bytes((parent / "config.json").read_bytes())
        (output / "manifest.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected))
        reused = {"generated": 0, "transcriptions": 0}
        copied_hashes = {}
        for name, key in [("generation/records", "generated"), ("scores", "transcriptions")]:
            target = output / name
            target.mkdir(parents=True, exist_ok=True)
            for source in sorted((parent / name).glob("*.json")):
                item = read(source)
                assert item["id"] in first_ids and item["plan_sha256"] == parent_sha
                if key == "transcriptions":
                    assert item["target_sha256"] == parent_plan["checkpoint_sha256"]
                destination = target / source.name
                expected = sha(source)
                if not destination.exists():
                    try:
                        os.link(source, destination)
                    except OSError:
                        shutil.copy2(source, destination)
                assert sha(destination) == expected
                copied_hashes[str(source)] = expected
                reused[key] += 1
        plan = {**parent_plan, "count": 5000, "worker_script": str(EXTENDED_WORKER),
                "manifest_policy": "Original first 2,000 test recordings followed by the original next 3,000; 250 recordings per each of 20 speakers; unchanged official references",
                "resume_parent": {"plan_path": str(parent / "plan.json"), "plan_sha256": parent_sha,
                                  "count": 2000, "reused_generated": reused["generated"], "reused_transcriptions": reused["transcriptions"]},
                "data_audit": {"official_references_exact": 5000, "training_validation_overlap": 0},
                "source_hashes": {**parent_plan["source_hashes"], str(EXTENDED_WORKER): sha(EXTENDED_WORKER)},
                "input_hashes": {**parent_plan["input_hashes"], str(output / "config.json"): sha(output / "config.json"),
                                 str(output / "manifest.jsonl"): sha(output / "manifest.jsonl"), str(parent / "plan.json"): parent_sha,
                                 str(following_path): sha(following_path), str(following_plan_path): sha(following_plan_path),
                                 str(full_path): sha(full_path), str(data_plan_path): sha(data_plan_path)}}
        save(output / "plan.json", plan)
        save(output / "extension-audit.json", {"status": "prepared; worker will verify audio and all score distances before reuse",
             "parent_plan_sha256": parent_sha, "count": 5000, "original_2000_first": True, "reused": reused,
             "original_cache_hashes": copied_hashes, "all_cache_json_bytes_preserved": True})
        (output / "worker-source.py").write_bytes(EXTENDED_WORKER.read_bytes())
        for path in [output / "config.json", output / "manifest.jsonl", output / "plan.json", output / "worker-source.py"]:
            path.chmod(0o444)
        save(parent / "successor.json", {"output": str(output), "parent_plan_sha256": parent_sha})
        (parent / "JOB_STATUS.md").write_text(
            f"# AISHELL scope extended to the original 5,000 recordings\n\nThe original {reused['generated']} generated audios and "
            f"{reused['transcriptions']} transcripts are retained unchanged. The unfinished original 2,000 will finish first, then the original next 3,000.\n\n"
            f"[Current status, PID, stop/resume commands and ETA]({output}/JOB_STATUS.md)\n")


def prepare(output):
    if (output / "plan.json").exists():
        return
    if output == DEFAULT:
        prepare_remaining(output)
        return
    if output == FIVE_THOUSAND:
        prepare_extension(output)
        return
    base = ROOT / "reproduction-runs/mandarin-cer-full-dev"
    metadata_path = base / "epoch-3-continuation-20261027/training/training.json"
    metadata = read(metadata_path)
    assert metadata["status"] == "complete" and metadata["completed_epochs"] == 3
    checkpoint = metadata["checkpoint"]
    assert checkpoint["epoch"] == 3 and sha(checkpoint["path"]) == checkpoint["sha256"]
    english_freeze_path = ROOT / "configs/english_selected.freeze.json"
    english = read(english_freeze_path)
    assert sha(english["config_path"]) == english["config_sha256"]
    assert sha(english["selection_path"]) == english["selection_sha256"]
    selection = read(Path(english["selection_path"]))
    assert max((r for r in selection["candidates"] if r["eligible"]), key=lambda r: r["error_percent"]) == english["selected"]
    config = json.loads(json.dumps(english["config"]))
    config["features"]["artifact_root"] = str(ROOT / "artifacts")
    config["surrogate"] = {"checkpoint": checkpoint["path"], "language": "zh", "nll_reduction": "mean",
                           "decoding": {"beam_size": 5, "temperature": 0., "without_timestamps": True}}
    old_plan_path = ROOT / "reproduction-runs/mandarin-cer-evaluation/plan.json"
    old = read(old_plan_path)
    source = Path(old["test_manifest"])
    assert sha(source) == old["test_manifest_sha256"]
    official_path = Path(old["full_test_manifest"])
    assert sha(official_path) == old["full_test_manifest_sha256"]
    selected, official = rows(source), {row["id"]: row for row in rows(official_path)}
    assert len(selected) == len({row["id"] for row in selected}) == 2000
    assert len(official) == 7176 and all(official.get(row["id"]) == row for row in selected)
    assert set(Counter(row["speaker"] for row in selected).values()) == {100}
    assert all(row["corpus"] == "aishell" and row["split"] == "test" and row["role"] == "test" for row in selected)
    truth_path = ROOT / "data/data_aishell/transcript/aishell_transcript_v0.8.txt"
    truth = dict(line.split(maxsplit=1) for line in truth_path.read_text().splitlines() if line.strip())
    assert all(row["text"] == truth[row["id"]].strip() for row in selected)
    role_paths = [base / "manifests" / filename for filename in ["finetune.jsonl", "checkpoint-validation.jsonl", "attack-pilot-48.jsonl"]]
    for path in role_paths:
        pool = rows(path)
        for field in ["id", "speaker", "audio_sha256"]:
            assert not {row[field] for row in selected} & {row[field] for row in pool}, f"Overlap in {field}"
    (output / "manifest.jsonl").write_bytes(source.read_bytes())
    save(output / "config.json", config)
    tokenizer = ROOT / "artifacts/python/jieba-0.42.1"
    assert (tokenizer / "jieba/dict.txt").exists()
    input_paths = [output / "config.json", output / "manifest.jsonl", metadata_path, Path(checkpoint["path"]),
                   ROOT / "artifacts/checkpoints/WavLM-Large.pt", ROOT / "artifacts/checkpoints/g_02500000.pt",
                   english_freeze_path, Path(english["config_path"]), Path(english["selection_path"]),
                   source, official_path, truth_path, *role_paths, tokenizer / "jieba/dict.txt"]
    source_paths = [WORKER, *[ROOT / "beyondwaveform" / name for name in
                    ["attacks.py", "pipeline.py", "config.py", "evaluation.py", "io.py", "losses.py", "models/whisper.py", "models/feature.py"]]]
    package_code = "import importlib.metadata,json; print(json.dumps({n:importlib.metadata.version(n) for n in ['torch','torchaudio','numpy','openai-whisper','jiwer']}))"
    packages = json.loads(subprocess.check_output([PYTHON, "-c", package_code], text=True))
    plan = {"status": "frozen", "epoch": 3, "count": 2000, "official_test_count": 7176,
            "checkpoint_sha256": checkpoint["sha256"], "checkpoint": checkpoint["path"],
            "input_hashes": {str(p): sha(p) for p in input_paths}, "source_hashes": {str(p): sha(p) for p in source_paths},
            "packages": packages, "tokenizer_path": str(tokenizer), "tokenizer_dictionary_sha256": sha(tokenizer / "jieba/dict.txt"),
            "initial_generation_seconds": 11.887, "initial_score_seconds": .762, "session_max_hours": 5,
            "manifest_policy": old["subset_policy"], "data_audit": {"official_references_exact": 2000, "training_validation_overlap": 0},
            "resume_policy": "One process; atomic per-audio generation and per-variant scores. Interrupted items repeat; completed records are hash-verified and reused.",
            "english_selection_sha256": sha(english_freeze_path), "automatic_followup_allowed": False}
    save(output / "plan.json", plan)
    (output / "worker-source.py").write_bytes(WORKER.read_bytes())
    for path in [output / "manifest.jsonl", output / "config.json", output / "plan.json", output / "worker-source.py"]:
        path.chmod(0o444)


def start(output, hours):
    pid = active_pid(output)
    if pid:
        print(json.dumps({"status": "already running", "pid": pid, "stop": f"kill {pid}"}));return
    old_job = read(output / "job.json")
    if old_job.get("status") == "complete":
        print(json.dumps({"status": "already complete", "report": str(output / "report.md")}));return
    with (output / "worker.lock").open("a+") as worker_lock:
        fcntl.flock(worker_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    with (ROOT / "reproduction-runs/auxiliary-gpu.lock").open("a+") as gpu:
        fcntl.flock(gpu, fcntl.LOCK_EX | fcntl.LOCK_NB)
    available = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True)
    assert int(available.splitlines()[0]) >= 8 * 1024, "Need at least 8 GiB free to start this GPU worker"
    if old_job.get("status") == "running":
        (output / "sessions").mkdir(exist_ok=True)
        save(output / "sessions" / f"recovered-{old_job['pid']}.json", {**old_job, "status": "interrupted", "recovered_after_worker_exit": True})
    started = time.time()
    deadline = started + hours * 3600 if hours is not None else None
    worker = Path(read(output / "plan.json").get("runtime_worker", str(RUNTIME_WORKER)))
    command = [PYTHON, "-u", str(worker), "--output", str(output)]
    if deadline is not None:
        command.extend(["--deadline-unix", str(deadline)])
    env = {**os.environ, "BEYONDWAVEFORM_NOTIFY_DISCORD": "0", "OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8", "OPENBLAS_NUM_THREADS": "1"}
    with (output / "run.log").open("ab") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True, close_fds=True, env=env)
    stat = Path(f"/proc/{process.pid}/stat").read_text().rsplit(")", 1)[1].split()
    launch = {"pid": process.pid, "start_ticks": stat[19], "session_id": os.getsid(process.pid),
              "detached": os.getsid(process.pid) == process.pid, "started_at": datetime.fromtimestamp(started, TAIPEI).isoformat(timespec="seconds"),
              "deadline_taipei": datetime.fromtimestamp(deadline, TAIPEI).isoformat(timespec="seconds") if deadline is not None else None,
              "max_hours": hours, "worker_script": str(worker), "worker_sha256": sha(worker), "command": command}
    save(output / "launch.json", launch)
    (output / "worker.pid").write_text(str(process.pid) + "\n")
    print(json.dumps({**launch, "stop": f"kill {process.pid}",
          "resume": f"python3 {Path(__file__).resolve()} resume --output {output}", "status_file": str(output / "JOB_STATUS.md")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "start", "resume", "stop", "status"])
    parser.add_argument("--output", type=Path, default=DEFAULT)
    parser.add_argument("--max-hours", type=float, default=None,
                        help="Optional session limit; by default run until all recordings finish")
    args = parser.parse_args()
    if args.max_hours is not None and (not math.isfinite(args.max_hours) or args.max_hours <= 0):
        parser.error("Session duration must be a finite positive number")
    output = args.output.resolve()
    visited = set()
    while successor := read(output / "successor.json"):
        assert output not in visited, "Cyclic run successor"
        visited.add(output)
        target = Path(successor["output"]).resolve()
        target_plan = read(target / "plan.json")
        lineage = target_plan.get("combined_parent", target_plan.get("resume_parent"))
        assert lineage, "Successor has no verified parent"
        assert Path(lineage["plan_path"]).resolve() == output / "plan.json"
        assert lineage["plan_sha256"] == successor["parent_plan_sha256"] == sha(output / "plan.json")
        output = target
    output.mkdir(parents=True, exist_ok=True)
    with (output / "control.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.action == "prepare":
            prepare(output)
            print(json.dumps({"status": "prepared", "output": str(output)}))
        elif args.action in {"start", "resume"}:
            if args.action == "start":
                prepare(output)
            assert (output / "plan.json").exists(), "No frozen run exists at this output directory"
            start(output, args.max_hours)
        elif args.action == "stop":
            pid = active_pid(output)
            if pid:
                os.kill(pid, signal.SIGTERM)
            print(json.dumps({"stop_requested": pid, "already_stopped": pid is None}))
        else:
            job = read(output / "job.json")
            pid = active_pid(output)
            print(json.dumps({"active_pid": pid, "effective_status": "running" if pid else
                              ("interrupted" if job.get("status") == "running" else job.get("status", "not started")),
                              "job": job, "progress": read(output / "progress.json")}, indent=2))
