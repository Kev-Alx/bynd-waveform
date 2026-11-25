"""Run a fixed AISHELL test protocol through the existing generation/scoring code.

An eight-clip development benchmark estimates runtime, without selecting settings.
Test chunks retain the normal resumable generation records and CER transcripts.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from beyondwaveform.config import load_config
from beyondwaveform.data import audit_disjoint
from beyondwaveform.evaluation import aggregate_errors, error_counts
from beyondwaveform.io import atomic_json, digest, read_jsonl, write_jsonl

TAIPEI = timezone(timedelta(hours=8))
VARIANTS = ["clean", "reconstruction", "adversarial"]


def now():
    return datetime.now(timezone.utc).isoformat()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def freeze_json(path, value):
    if path.exists():
        require(json.loads(path.read_text(encoding="utf-8")) == value, f"Frozen record differs: {path}")
    else:
        atomic_json(path, value)
        path.chmod(0o444)


def freeze_manifest(path, rows):
    if path.exists():
        require(read_jsonl(path) == rows, f"Frozen manifest differs: {path}")
    else:
        write_jsonl(path, rows)
        path.chmod(0o444)


def prepare(output, chunk_size, test_count=7176):
    original = ROOT / "reproduction-runs/mandarin-independent"
    data_path, surrogate_path = original / "data-freeze.json", original / "surrogate-freeze.json"
    data = json.loads(data_path.read_text(encoding="utf-8"))
    surrogate = json.loads(surrogate_path.read_text(encoding="utf-8"))
    require(surrogate["status"] == "frozen", "Mandarin surrogate is not frozen")
    require(digest(data_path) == surrogate["data_freeze_sha256"], "Mandarin data receipt changed")
    require(digest(surrogate["checkpoint"]) == surrogate["checkpoint_sha256"], "Mandarin checkpoint changed")
    require(digest(original / "training/training.json") == surrogate["training_metadata_sha256"], "Training receipt changed")
    manifests = [data["paths"][name] for name in ["finetune", "checkpoint-validation", "attack-validation"]]
    manifests.append(data["test_manifest"])
    for path in manifests:
        require(digest(path) == data["proofs"][path], f"Frozen manifest changed: {path}")
    audit = audit_disjoint(manifests)
    pools = [read_jsonl(path) for path in manifests]
    speakers = [{row["speaker"] for row in rows} for rows in pools]
    require(not any(left & right for i, left in enumerate(speakers) for right in speakers[i+1:]), "Speaker leakage")
    test, validation = pools[-1], pools[-2]
    require(len(test) == data["test_count"] == 7176, "Expected all 7,176 official AISHELL test clips")
    require(all(row["corpus"] == "aishell" and row["split"] == "test" and row["role"] == "test"
                and row["language"] == "zh" for row in test), "Wrong test corpus/split/role/language")
    full_test_manifest = data["test_manifest"]
    test_manifest = full_test_manifest
    subset_policy = "Complete official test split"
    if test_count != len(test):
        subset_path = output / "test-subset.jsonl"
        selected = read_jsonl(subset_path)
        official = {row["id"]: row for row in test}
        require(len(selected) == test_count and all(official.get(row["id"]) == row for row in selected),
                "Subset must contain the requested count of unchanged official test rows")
        test = selected
        test_manifest = str(subset_path)
        subset = json.loads(subset_path.with_suffix(".jsonl.metadata.json").read_text(encoding="utf-8"))
        require(subset["count"] == test_count and subset["manifest_sha256"] == digest(subset_path), "Subset receipt changed")
        subset_policy = subset["selection"]
    english_freeze_path = ROOT / "configs/english_selected.freeze.json"
    english_freeze = json.loads(english_freeze_path.read_text(encoding="utf-8"))
    base_path = Path(english_freeze["config_path"])
    require(digest(base_path) == english_freeze["config_sha256"], "Selected English config changed")
    require(digest(english_freeze["selection_path"]) == english_freeze["selection_sha256"], "English selection changed")
    selection = json.loads(Path(english_freeze["selection_path"]).read_text(encoding="utf-8"))
    winner = max((row for row in selection["candidates"] if row["eligible"]), key=lambda row: row["error_percent"])
    require(winner == selection["selected"] == english_freeze["selected"], "English selection is not its eligible WER winner")
    require(digest(winner["summary"]) == english_freeze["selected_summary_sha256"], "Selected English summary changed")
    config = load_config(base_path)
    config["features"]["artifact_root"] = str(ROOT / "artifacts")
    config["surrogate"] = load_config(ROOT / "configs/mandarin_finetune.yaml")["model"]
    config["surrogate"]["checkpoint"] = surrogate["checkpoint"]
    config_path = output / "config.json"
    freeze_json(config_path, config)
    # Duration quantiles from the separate attack-validation pool, chosen without scores.
    ordered = sorted(validation, key=lambda row: (row["duration"], row["id"]))
    benchmark = [ordered[round((i + 0.5) * len(ordered) / 8 - 0.5)] for i in range(8)]
    benchmark_path = output / "benchmark.jsonl"
    freeze_manifest(benchmark_path, benchmark)
    chunks = []
    for start in range(0, len(test), chunk_size):
        rows = test[start:start+chunk_size]
        folder = output / "test" / f"chunk-{start//chunk_size:03d}"
        path = folder / "manifest.jsonl"
        freeze_manifest(path, rows)
        chunks.append({"folder": str(folder), "manifest": str(path), "count": len(rows), "sha256": digest(path)})
    source_paths = [*sorted((ROOT / "beyondwaveform").glob("*.py")),
                    *sorted((ROOT / "beyondwaveform/models").glob("*.py")), Path(__file__),
                    ROOT / "scripts/generate.py", ROOT / "scripts/evaluate.py"]
    artifact_paths = [Path(surrogate["checkpoint"]), ROOT / "artifacts/checkpoints/WavLM-Large.pt",
                      ROOT / "artifacts/checkpoints/g_02500000.pt"]
    plan = {"status": "frozen", "scope": f"Fixed-setting AISHELL-1 CER; {len(test)} official test recordings; raw Mandarin Whisper-small; no defenses",
            "test_count": len(test), "test_manifest": test_manifest,
            "test_manifest_sha256": digest(test_manifest), "full_test_count": data["test_count"],
            "full_test_manifest": full_test_manifest, "full_test_manifest_sha256": digest(full_test_manifest),
            "subset_policy": subset_policy, "chunk_size": chunk_size, "chunks": chunks,
            "benchmark_manifest": str(benchmark_path), "benchmark_sha256": digest(benchmark_path),
            "config_path": str(config_path), "config_sha256": digest(config_path),
            "checkpoint_sha256": surrogate["checkpoint_sha256"], "split_audit": audit,
            "source_hashes": {str(path): digest(path) for path in source_paths},
            "artifact_hashes": {str(path): digest(path) for path in artifact_paths},
            "variants": VARIANTS, "metric": "CER", "base_config": str(base_path),
            "english_selection": {"freeze": str(english_freeze_path), "freeze_sha256": digest(english_freeze_path),
                                  "selection": english_freeze["selection_path"], "selection_sha256": english_freeze["selection_sha256"],
                                  "validation_wer": winner["error_percent"], "summary": winner["summary"]},
            "protocol": f"User-requested English WER winner: rho={config['attack']['rho']}, "
                        f"lambda_perc={config['attack']['lambda_perc']}, {config['attack']['steps']} steps, "
                        f"Adam LR={config['attack']['learning_rate']} from the frozen English selection. "
                        "No Mandarin hyperparameter search or test-based tuning. Validation benchmark is execution/timing only. "
                        "Corpus character edit counts; existing NFKC/lowercase/punctuation/whitespace normalization. "
                        "Forced zh; beam size 5; temperature 0; fp32. Speech quality, transfer and defenses are not measured."}
    freeze_json(output / "plan.json", plan)
    return plan


class Runner:
    def __init__(self, output, plan, device, threads):
        self.output, self.plan, self.device, self.threads = output, plan, device, threads
        self.started = time.monotonic()
        self.state = {"status": "running", "phase": "preflight", "started_at": now(),
                      "scope": plan["scope"], "total_utterances": plan["test_count"],
                      "generated_utterances": 0, "evaluated_utterances": 0,
                      "eta_basis": "Calibrating on eight development clips; no test-based tuning"}
        self.gen_done = self.eval_done = 0
        self.gen_seconds = self.eval_seconds = self.gen_overhead = 0.0
        self.current_folder = None
        self.all_rows = []

    def progress(self, **values):
        self.state.update(values, updated_at=now())
        generated, current_seconds = self.gen_done, 0.0
        if self.current_folder is not None:
            records = [json.loads(p.read_text(encoding="utf-8")) for p in (self.current_folder / "run/records").glob("*.json")]
            generated += len(records)
            current_seconds = sum(r["generation_seconds"] for r in records)
        self.state.update(generated_utterances=generated, evaluated_utterances=self.eval_done,
                          clip_progress={"completed": self.eval_done, "total": self.plan["test_count"],
                                         "percent": round(100 * self.eval_done / self.plan["test_count"], 1)})
        timing_path = self.output / "timing.json"
        if timing_path.exists():
            timing = json.loads(timing_path.read_text(encoding="utf-8"))
            gen_rate = (self.gen_seconds + current_seconds) / generated if generated else timing["generation_seconds_per_clip"]
            eval_rate = self.eval_seconds / self.eval_done if self.eval_done else timing["evaluation_seconds_per_clip"]
            chunks_left = math.ceil(max(0, self.plan["test_count"]-generated) / self.plan["chunk_size"])
            overhead = self.gen_overhead / max(1, math.ceil(self.gen_done/self.plan["chunk_size"])) if self.gen_done else timing["generation_setup_seconds"]
            remaining = max(0, (self.plan["test_count"]-generated)*gen_rate +
                            (self.plan["test_count"]-self.eval_done)*eval_rate + chunks_left*overhead)
            if self.state["status"] == "complete":
                remaining = 0
            completion = datetime.now(TAIPEI) + timedelta(seconds=remaining)
            self.state.update(estimated_remaining_seconds=round(remaining),
                              estimated_remaining_range_seconds=[round(remaining*0.8), round(remaining*1.5)],
                              estimated_completion_taipei=completion.isoformat(timespec="seconds"),
                              eta_basis="Eight development clips, updated with observed test execution times; approximate 0.8–1.5× range",
                              generation_seconds_per_clip=round(gen_rate, 3), evaluation_seconds_per_clip=round(eval_rate, 3))
        atomic_json(self.output / "progress.json", self.state)
        self.write_status()

    def write_status(self):
        state = self.state
        lines = ["# AISHELL-1 CER evaluation", "", f"Updated: {datetime.now(TAIPEI).isoformat(timespec='seconds')} (Asia/Taipei)",
                 f"Status: **{state['status']}**. Phase: **{state['phase']}**.", "",
                 f"Evaluation recordings from the official test split: **{self.plan['test_count']} / {self.plan['full_test_count']}**.",
                 f"Generated: **{state['generated_utterances']}**; evaluated with all three variants: **{self.eval_done}**."]
        if "estimated_remaining_seconds" in state:
            low, high = state["estimated_remaining_range_seconds"]
            lines += ["", f"Estimated remaining: **{state['estimated_remaining_seconds']/3600:.1f} hours** "
                      f"(rough range **{low/3600:.1f}–{high/3600:.1f} hours**).",
                      f"Estimated completion: **{state['estimated_completion_taipei']} (Asia/Taipei)**."]
        else:
            lines += ["", "Estimated completion: calculating from the eight-clip development timing run."]
        lines += [f"ETA basis: {state['eta_basis']}.", "", f"Subset: {self.plan['subset_policy']}.", "", self.plan["protocol"], "",
                  f"[Live log]({self.output}/suite.log) · [Progress]({self.output}/progress.json) · "
                  f"[Results]({self.output}/results.json) · [Report]({self.output}/report.md)", ""]
        if state.get("error"):
            lines += [f"Error: {state['error']}", ""]
        path = self.output / "JOB_STATUS.md"
        temporary = path.with_suffix(".md.tmp")
        temporary.write_text("\n".join(lines), encoding="utf-8")
        temporary.replace(path)

    def phase(self, name, command):
        folder = self.output / "phases" / name
        folder.mkdir(parents=True, exist_ok=True)
        self.progress(phase=name, phase_started_at=now(), phase_elapsed_seconds=0, command=command)
        env = {**os.environ, "OMP_NUM_THREADS": str(self.threads), "MKL_NUM_THREADS": str(self.threads),
               "OPENBLAS_NUM_THREADS": "1", "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8",
               "BEYONDWAVEFORM_NOTIFY_DISCORD": "0"}
        wrapped = [sys.executable, "-u", str(ROOT / "scripts/run_logged.py"), "--log", str(folder / "stage.log"),
                   "--status", str(folder / "job.json"), "--", *command]
        print(f"Starting {name}", flush=True)
        start = time.monotonic()
        child = subprocess.Popen(wrapped, cwd=ROOT, env=env)
        self.progress(phase_pid=child.pid)
        try:
            while True:
                try:
                    code = child.wait(timeout=30)
                    break
                except subprocess.TimeoutExpired:
                    self.progress(phase_elapsed_seconds=round(time.monotonic()-start, 1))
                    print(f"Heartbeat: {name}; generated {self.state['generated_utterances']}; scored {self.eval_done}", flush=True)
            require(code == 0, f"{name} exited {code}; see {folder}/stage.log")
        finally:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
        return time.monotonic()-start

    def audit(self, folder, manifest, expected_role):
        rows = read_jsonl(manifest)
        by_id = {row["id"]: row for row in rows}
        run = json.loads((folder / "run/run.json").read_text(encoding="utf-8"))
        result = json.loads((folder / "evaluation/summary.json").read_text(encoding="utf-8"))
        require(run["status"] == "complete" and run["count"] == len(rows) and run["ids"] == list(by_id), "Incomplete generation")
        require(run["manifest_sha256"] == digest(manifest) == result["manifest_sha256"], "Manifest mismatch")
        require(run["config"] == load_config(self.plan["config_path"]), "Config changed")
        require(run["surrogate"]["sha256"] == self.plan["checkpoint_sha256"], "Wrong Mandarin model")
        artifacts = self.plan["artifact_hashes"]
        require(run["features"]["encoder_sha256"] == artifacts[str(ROOT / "artifacts/checkpoints/WavLM-Large.pt")]
                and run["features"]["vocoder_sha256"] == artifacts[str(ROOT / "artifacts/checkpoints/g_02500000.pt")], "Feature weights changed")
        require(result["metric"] == "CER" and result["evaluation_scope"] == "surrogate"
                and result["target_artifacts"] == run["surrogate"] and result["roles"] == [expected_role]
                and result["evaluation_ids"] == list(by_id), "Wrong evaluation protocol")
        records = read_jsonl(folder / "run/generated.jsonl")
        require(digest(folder / "run/generated.jsonl") == run["generated_sha256"], "Generated index changed")
        radius = run["config"]["attack"]["rho"]
        for record in records:
            require(record["text"] == by_id[record["id"]]["text"] and record["language"] == "zh", "Wrong reference")
            require(record["input_audio_sha256"] == by_id[record["id"]]["audio_sha256"], "Wrong source audio")
            require(len(record["trace"]) == 50 and all(math.isfinite(step["loss_before_update"])
                    and step["radius_after_update"] <= radius+1e-6 for step in record["trace"]), "Invalid attack trace")
            for asset in record["audio"].values():
                require(digest(asset["path"]) == asset["sha256"], "Saved audio changed")
        scored = [json.loads(line) for line in (folder / "evaluation/utterances.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        require(digest(folder / "evaluation/utterances.jsonl") == result["utterances_sha256"], "Transcripts changed")
        require({(row["id"], row["variant"]) for row in scored} == {(uid, v) for uid in by_id for v in VARIANTS}
                and len(scored) == len(rows)*3, "Missing or duplicate scored variants")
        for row in scored:
            require(row["reference"] == by_id[row["id"]]["text"], "Scored reference changed")
            counts = error_counts(row["reference"], row["hypothesis"], "zh")
            require(all(row[key] == value for key, value in counts.items()), "Character counts do not recompute")
        for variant in VARIANTS:
            counts = aggregate_errors([row for row in scored if row["variant"] == variant])
            require(result["summary"][variant] == {"count": len(rows), **counts}, "CER summary does not recompute")
        atomic_json(folder / "verification.json", {"status": "passed", "count": len(rows), "metric": "CER",
                    "summary_sha256": digest(folder / "evaluation/summary.json"), "checked_at": now()})
        return scored, run

    def run_chunk(self, folder, manifest, name, role):
        result_path = folder / "evaluation/summary.json"
        gen_wall = eval_wall = 0.0
        if not result_path.exists():
            generation = [sys.executable, "-u", str(ROOT / "scripts/generate.py"), "--config", self.plan["config_path"],
                          "--manifest", str(manifest), "--output", str(folder / "run"), "--device", self.device]
            gen_wall = self.phase(f"{name}-generation", generation)
            evaluation = [sys.executable, "-u", str(ROOT / "scripts/evaluate.py"), "--run", str(folder / "run"),
                          "--target", "surrogate", "--output", str(folder / "evaluation"), "--device", self.device,
                          "--variants", *VARIANTS]
            eval_wall = self.phase(f"{name}-evaluation", evaluation)
            atomic_json(folder / "timing.json", {"generation_wall_seconds": gen_wall, "evaluation_wall_seconds": eval_wall})
        timing = json.loads((folder / "timing.json").read_text(encoding="utf-8"))
        scored, run = self.audit(folder, manifest, role)
        return scored, run, timing

    def results(self, complete=False):
        summary = {v: {"count": sum(row["variant"] == v for row in self.all_rows),
                       **aggregate_errors([row for row in self.all_rows if row["variant"] == v])} for v in VARIANTS}
        result = {"status": "complete" if complete else "partial", "metric": "CER", "scope": self.plan["scope"],
                  "expected_count": self.plan["test_count"], "completed_count": self.eval_done,
                  "plan_sha256": digest(self.output / "plan.json"), "summary": summary,
                  "protocol": self.plan["protocol"], "updated_at": now()}
        atomic_json(self.output / "results.json", result)
        lines = ["# AISHELL-1 CER results", "", f"Status: **{result['status']}**; {self.eval_done}/{self.plan['test_count']} official test recordings.",
                 "", "| Condition | Utterances | CER (%) | Substitutions | Deletions | Insertions | Reference characters |",
                 "|---|---:|---:|---:|---:|---:|---:|"]
        for variant, counts in summary.items():
            lines.append(f"| {variant} | {counts['count']} | {counts['error_percent']:.6f} | {counts['substitutions']} | "
                         f"{counts['deletions']} | {counts['insertions']} | {counts['reference_units']} |")
        lines += ["", self.plan["protocol"], "", "These are independent implementation measurements, not exact paper reproduction.", ""]
        (self.output / "report.md").write_text("\n".join(lines), encoding="utf-8")
        if complete:
            write_jsonl(self.output / "utterances.jsonl", self.all_rows)
            atomic_json(self.output / "verification.json", {"status": "passed", "count": self.eval_done,
                        "pairs": len(self.all_rows), "results_sha256": digest(self.output / "results.json"),
                        "utterances_sha256": digest(self.output / "utterances.jsonl"), "checked_at": now()})

    def run(self, benchmark_only=False):
        self.progress()
        benchmark = self.output / "benchmark"
        _, run, timing = self.run_chunk(benchmark, self.plan["benchmark_manifest"], "development-benchmark", "validation")
        atomic_json(self.output / "timing.json", {
            "benchmark_count": run["count"], "generation_seconds_per_clip": run["generation_seconds"]/run["count"],
            "evaluation_seconds_per_clip": timing["evaluation_wall_seconds"]/run["count"],
            "generation_setup_seconds": max(0, timing["generation_wall_seconds"]-run["generation_seconds"]),
            "basis": "Development timing only; configuration was frozen before execution"})
        self.progress(phase="benchmark-complete")
        if benchmark_only:
            self.progress(status="benchmark_complete", phase="benchmark-complete")
            return
        for number, chunk in enumerate(self.plan["chunks"]):
            for path, expected in self.plan["source_hashes"].items():
                require(digest(path) == expected, f"Frozen evaluation source changed: {path}")
            require(digest(chunk["manifest"]) == chunk["sha256"], "Test chunk changed")
            self.current_folder = Path(chunk["folder"])
            rows, run, timing = self.run_chunk(self.current_folder, chunk["manifest"], f"test-{number:03d}", "test")
            self.all_rows.extend(rows)
            self.gen_done += run["count"]
            self.eval_done += run["count"]
            self.gen_seconds += run["generation_seconds"]
            self.eval_seconds += timing["evaluation_wall_seconds"]
            self.gen_overhead += max(0, timing["generation_wall_seconds"]-run["generation_seconds"])
            self.current_folder = None
            self.results()
            self.progress(phase=f"test-{number:03d}-verified")
        require(self.eval_done == self.plan["test_count"] and
                {row["id"] for row in self.all_rows} == {row["id"] for row in read_jsonl(self.plan["test_manifest"])}, "Final coverage differs")
        self.results(complete=True)
        self.progress(status="complete", phase="complete", finished_at=now())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reproduction-runs/mandarin-cer-evaluation")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--test-count", type=int, default=7176)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--benchmark-only", action="store_true")
    args = parser.parse_args()
    require(args.chunk_size > 0 and args.threads > 0, "Positive chunk size/threads required")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"Interrupted by signal {signum}")
    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, interrupt)
    with (output / "evaluation.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = prepare(output, args.chunk_size, args.test_count)
        if args.check_only:
            print(f"Preflight passed: {plan['test_count']} official test clips; frozen checkpoint/config; disjoint speaker pools.", flush=True)
            return
        with (output.parent / "auxiliary-gpu.lock").open("a+") as gpu_lock:
            fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            runner = Runner(output, plan, args.device, args.threads)
            try:
                runner.run(args.benchmark_only)
            except BaseException as error:
                runner.current_folder = None
                runner.progress(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                                error=str(error), finished_at=now())
                raise


if __name__ == "__main__":
    main()
