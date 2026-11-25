"""Single-process, per-audio resumable AISHELL generation and recognition."""
import argparse
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import jiwer
from beyondwaveform.attacks import feature_attack
from beyondwaveform.config import AttackConfig, SpectralConfig
from beyondwaveform.evaluation import aggregate_errors, error_counts
from beyondwaveform.io import atomic_json, digest, load_audio, read_jsonl, save_audio, seed_everything, write_jsonl
from beyondwaveform.models.feature import load_feature_models
from beyondwaveform.models.whisper import load_whisper

TAIPEI = timezone(timedelta(hours=8))
VARIANTS = ("clean", "reconstruction", "adversarial")


class RequestedStop(BaseException):
    pass


def distance(a, b):
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


class Worker:
    def __init__(self, output, deadline, device="cuda"):
        self.output = Path(output).resolve()
        self.deadline, self.device = deadline, device
        self.plan = json.loads((self.output / "plan.json").read_text())
        self.plan_sha = digest(self.output / "plan.json")
        self.config = json.loads((self.output / "config.json").read_text())
        self.rows = read_jsonl(self.output / "manifest.jsonl")
        self.originals = {row["id"]: row for row in self.rows}
        self.generated, self.scores = {}, {}
        self.status, self.phase = "running", "preflight"
        self.started = time.monotonic()
        self.job = {"status": "running", "pid": os.getpid(), "session_id": os.getsid(0),
                    "started_at": datetime.now(TAIPEI).isoformat(timespec="seconds"),
                    "deadline_taipei": datetime.fromtimestamp(deadline, TAIPEI).isoformat(timespec="seconds"),
                    "plan_sha256": self.plan_sha, "command": sys.argv}
        self.session_file = self.output / "sessions" / f"{time.time_ns()}-{os.getpid()}.json"

    def validate_inputs(self):
        for path, expected in {**self.plan["input_hashes"], **self.plan["source_hashes"]}.items():
            if digest(path) != expected:
                raise ValueError(f"Frozen input changed; refusing to mix results: {path}")
        assert len(self.rows) == self.plan["count"] == len(self.originals)
        for package, expected in self.plan.get("packages", {}).items():
            assert importlib.metadata.version(package) == expected, f"Package version changed: {package}"

    def validate_generation(self, record):
        source = self.originals[record["id"]]
        assert record["plan_sha256"] == self.plan_sha
        assert record["text"] == source["text"] and record["input_audio_sha256"] == source["audio_sha256"]
        assert digest(source["audio"]) == source["audio_sha256"]
        assert set(record["audio"]) == set(VARIANTS)
        for asset in record["audio"].values():
            assert digest(asset["path"]) == asset["sha256"], "Saved waveform changed"
        trace = record["trace"]
        assert len(trace) == self.config["attack"]["steps"]
        for index, step in enumerate(trace, 1):
            assert step["step"] == index and all(math.isfinite(v) for v in step.values())
            assert step["radius_after_update"] <= self.config["attack"]["rho"] + 1e-6

    def validate_score(self, item):
        source = self.originals[item["id"]]
        asset = self.generated[item["id"]]["audio"][item["variant"]]
        assert item["plan_sha256"] == self.plan_sha and item["target_sha256"] == self.plan["checkpoint_sha256"]
        assert item["reference"] == source["text"] and item["audio_sha256"] == asset["sha256"]
        normalize = lambda text: "".join(c for c in unicodedata.normalize("NFKC", text).lower()
                                        if not c.isspace() and not unicodedata.category(c).startswith("P"))
        ref, hyp = normalize(item["reference"]), normalize(item["hypothesis"])
        assert ref == item["cer"]["reference_normalized"] and hyp == item["cer"]["hypothesis_normalized"]
        for metric, a, b in [("cer", ref, hyp), ("wer", item["reference_word_tokens"], item["hypothesis_word_tokens"])]:
            assert len(a) == item[metric]["reference_units"]
            assert distance(a, b) == sum(item[metric][key] for key in ("substitutions", "deletions", "insertions"))

    def restore(self):
        for path in sorted((self.output / "generation/records").glob("*.json")):
            record = json.loads(path.read_text())
            self.validate_generation(record)
            self.generated[record["id"]] = record
        for path in sorted((self.output / "scores").glob("*.json")):
            item = json.loads(path.read_text())
            self.validate_score(item)
            self.scores[(item["id"], item["variant"])] = item
        self.job.update(reused_generated=len(self.generated), reused_transcriptions=len(self.scores))

    def publish(self):
        now = datetime.now(TAIPEI)
        count = len(self.rows)
        generation_rate = (sum(r["generation_seconds"] for r in self.generated.values()) / len(self.generated)
                           if self.generated else self.plan["initial_generation_seconds"])
        score_rate = sum(r["seconds"] for r in self.scores.values()) / len(self.scores) if self.scores else self.plan["initial_score_seconds"]
        remaining = (count - len(self.generated)) * generation_rate + (3 * count - len(self.scores)) * score_rate
        if self.status == "complete":
            remaining = 0
        progress = {"status": self.status, "phase": self.phase, "updated_at": now.isoformat(timespec="seconds"),
                    "pid": os.getpid(), "session_id": os.getsid(0), "epoch": 3, "generated": len(self.generated),
                    "scored_variants": len(self.scores), "count": count, "clip_progress": {"completed": len(self.scores), "total": 3 * count,
                    "percent": round(100 * len(self.scores) / (3 * count), 1)}, "estimated_remaining_seconds": round(remaining),
                    "estimated_remaining_range_seconds": [round(remaining * .7), round(remaining * 1.7)],
                    "estimated_completion_taipei": (now + timedelta(seconds=remaining)).isoformat(timespec="seconds"),
                    "session_deadline_taipei": self.job["deadline_taipei"],
                    "eta_basis": "Completed per-audio generation and per-variant decoding times; initial estimate uses the previous AISHELL timing run",
                    "automatic_followup_allowed": False}
        atomic_json(self.output / "progress.json", progress)
        summary = {}
        for variant in VARIANTS:
            items = [row for (uid, name), row in self.scores.items() if name == variant]
            if items:
                summary[variant] = {"count": len(items), "cer": aggregate_errors([row["cer"] for row in items]),
                                    "wer_jieba": aggregate_errors([row["wer"] for row in items])}
        atomic_json(self.output / "results.json", {"status": self.status, "epoch": 3, "count": count,
                    "plan_sha256": self.plan_sha, "summary": summary})
        resume = f"python3 {ROOT}/scripts/manage_aishell_epoch3.py resume --output {self.output}"
        controls = (f"Stop this single GPU worker: `kill {os.getpid()}`. After it exits, resume detached with:\n\n"
                    f"```bash\n{resume}\n```\n\n"
                    "Completed audio and transcription records are reused. An interrupted audio/variant is repeated. "
                    "Even `kill -9` retains committed records; the resume command detects a dead worker. The PID changes after each resume.\n\n")
        status = (f"# AISHELL epoch 3 — {count} test recordings\n\nUpdated: {progress['updated_at']} (Asia/Taipei).\n\n"
                  f"Status: **{self.status}**; phase: **{self.phase}**. Worker PID: **{os.getpid()}**.\n\n"
                  f"Generated: **{len(self.generated)}/{count}**; transcribed: **{len(self.scores)}/{3 * count}** clean/reconstruction/adversarial variants.\n\n"
                  f"Estimated remaining compute: **{remaining/3600:.2f} hours**; projected completion **{progress['estimated_completion_taipei']} (Asia/Taipei)**. "
                  "This estimate excludes time spent stopped.\n\n"
                  f"This session stops automatically by **{self.job['deadline_taipei']} (Asia/Taipei)** after at most five hours. Resume starts another bounded session.\n\n"
                  f"Attack: rho **{self.config['attack']['rho']}**, lambda **{self.config['attack']['lambda_perc']}**, "
                  f"**{self.config['attack']['steps']} steps**, Adam LR **{self.config['attack']['learning_rate']}**.\n\n" + controls +
                  f"- [Results]({self.output}/report.md)\n- [Live log]({self.output}/run.log)\n- [Job receipt]({self.output}/job.json)\n"
                  f"- [Frozen model, data and settings]({self.output}/plan.json)\n")
        for path in [self.output / "JOB_STATUS.md", ROOT / "reproduction-runs/mandarin-cer-full-dev/JOB_STATUS.md",
                     ROOT / "reproduction-runs/mandarin-cer-corrected/WORKFLOW_STATUS.md"]:
            path.write_text(status)
        report = (f"# AISHELL epoch 3 evaluation\n\nStatus: **{self.status}**. {count} official test recordings; "
                  f"{len(self.generated)} generated, {len(self.scores)} variant transcriptions complete.\n\n"
                  "| Audio | Count scored | CER | WER (jieba) |\n|---|---:|---:|---:|\n")
        for variant in VARIANTS:
            values = summary.get(variant)
            report += (f"| {variant} | {values['count']} | {values['cer']['error_percent']:.4f}% | {values['wer_jieba']['error_percent']:.4f}% |\n"
                       if values else f"| {variant} | 0 | pending | pending |\n")
        report += ("\nCorpus rates sum substitutions, deletions and insertions across recordings before dividing by reference characters or word tokens. "
                   "Mandarin CER is the comparison metric; WER uses jieba 0.42.1, HMM disabled, on identically normalized references and hypotheses. "
                   "Decoder: forced zh, beam 5, temperature 0, fp32, timestamps disabled, no previous-text conditioning.\n\n"
                   "The same frozen 2,000 official test IDs are retained from the prior run, with unchanged official references. "
                   "All new audio is generated with epoch 3; existing audio from another checkpoint is not reused.\n\n" + controls +
                   f"- [Machine-readable scores]({self.output}/results.json)\n- [Manifest]({self.output}/manifest.jsonl)\n")
        (self.output / "report.md").write_text(report)

    def generate_one(self, row, asr, encoder, vocoder):
        uid = row["id"]
        assert digest(row["audio"]) == row["audio_sha256"]
        seed = (int(self.config.get("seed", 2026)) + int(hashlib.sha256(uid.encode()).hexdigest()[:8], 16)) % 2**32
        seed_everything(seed)
        audio = load_audio(row["audio"]).to(self.device)
        started = time.perf_counter()
        result = feature_attack(audio, row["text"], encoder, vocoder, asr,
                                AttackConfig(**self.config["attack"]), SpectralConfig(**self.config["spectral"]))
        assets = {variant: save_audio(self.output / "generation/audio" / variant / f"{uid}.wav", value)
                  for variant, value in [("clean", audio), ("reconstruction", result.reconstruction), ("adversarial", result.audio)]}
        record = {"id": uid, "text": row["text"], "input_audio_sha256": row["audio_sha256"],
                  "plan_sha256": self.plan_sha, "seed": seed, "audio": assets, "trace": result.trace,
                  "generation_seconds": time.perf_counter() - started}
        self.validate_generation(record)
        atomic_json(self.output / "generation/records" / f"{uid}.json", record)
        self.generated[uid] = record

    def score_one(self, row, variant, asr, tokenizer):
        uid = row["id"]
        asset = self.generated[uid]["audio"][variant]
        started = time.perf_counter()
        assert digest(asset["path"]) == asset["sha256"]
        hypothesis = asr.transcribe(load_audio(asset["path"]).to(self.device))
        cer = error_counts(row["text"], hypothesis, "zh")
        ref_words = list(tokenizer.cut(cer["reference_normalized"], HMM=False))
        hyp_words = list(tokenizer.cut(cer["hypothesis_normalized"], HMM=False))
        words = jiwer.process_words(" ".join(ref_words), " ".join(hyp_words))
        item = {"id": uid, "variant": variant, "reference": row["text"], "hypothesis": hypothesis,
                "audio_sha256": asset["sha256"], "target_sha256": self.plan["checkpoint_sha256"], "plan_sha256": self.plan_sha,
                "cer": cer, "wer": {"substitutions": words.substitutions, "deletions": words.deletions,
                                     "insertions": words.insertions, "reference_units": len(ref_words)},
                "reference_word_tokens": ref_words, "hypothesis_word_tokens": hyp_words, "seconds": time.perf_counter() - started}
        self.validate_score(item)
        atomic_json(self.output / "scores" / f"{uid}.{variant}.json", item)
        self.scores[(uid, variant)] = item

    def run(self):
        previous = {}

        def stop(signum, frame):
            raise RequestedStop("five-hour session limit" if signum == signal.SIGALRM else f"signal {signum}")

        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGALRM):
            previous[sig] = signal.signal(sig, stop)
        signal.setitimer(signal.ITIMER_REAL, max(.01, self.deadline - time.time() - 3))
        try:
            atomic_json(self.output / "job.json", self.job)
            self.validate_inputs()
            self.phase = "verifying-committed-checkpoints"
            self.restore()
            self.publish()
            torch.set_num_threads(8)
            sys.path.insert(0, self.plan["tokenizer_path"])
            import jieba
            assert jieba.__version__ == "0.42.1" and digest(Path(jieba.__file__).with_name("dict.txt")) == self.plan["tokenizer_dictionary_sha256"]
            tokenizer = jieba.Tokenizer()
            tokenizer.tmp_dir = str(self.output)
            tokenizer.initialize()
            print(f"Resuming with {len(self.generated)} generated audios and {len(self.scores)} transcriptions already committed.", flush=True)
            asr, meta = load_whisper(self.config["surrogate"], self.device)
            assert meta["sha256"] == self.plan["checkpoint_sha256"]
            encoder, vocoder, feature_meta = load_feature_models(self.config["features"], self.device)
            for row in self.rows:
                uid = row["id"]
                if uid not in self.generated:
                    self.phase = f"generating:{uid}"
                    self.publish()
                    self.generate_one(row, asr, encoder, vocoder)
                    print(f"Generated {len(self.generated)}/{len(self.rows)}: {uid}", flush=True)
                    self.publish()
                for variant in VARIANTS:
                    if (uid, variant) not in self.scores:
                        self.phase = f"scoring:{uid}:{variant}"
                        self.score_one(row, variant, asr, tokenizer)
                        self.publish()
            write_jsonl(self.output / "utterances.jsonl", list(self.scores.values()))
            self.validate_inputs()
            atomic_json(self.output / "independent-audit.json", {"status": "passed", "count": len(self.rows),
                        "generation_records": len(self.generated), "character_distance_checks": len(self.scores),
                        "word_distance_checks": len(self.scores), "plan_sha256": self.plan_sha,
                        "utterances_sha256": digest(self.output / "utterances.jsonl")})
            self.status, self.phase = "complete", "complete"
        except RequestedStop as reason:
            self.status, self.phase = "stopped", "stopped-with-committed-checkpoints"
            self.job["stop_reason"] = str(reason)
        except BaseException as error:
            self.status, self.phase = "failed", "failed"
            self.job["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            for sig in previous:
                signal.signal(sig, signal.SIG_IGN)
            self.job.update(status=self.status, finished_at=datetime.now(TAIPEI).isoformat(timespec="seconds"),
                            generated=len(self.generated), scored_variants=len(self.scores))
            atomic_json(self.output / "job.json", self.job)
            atomic_json(self.session_file, self.job)
            self.publish()
            for sig, handler in previous.items():
                signal.signal(sig, handler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deadline-unix", type=float, required=True)
    args = parser.parse_args()
    with (args.output / "worker.lock").open("a+") as lock, (ROOT / "reproduction-runs/auxiliary-gpu.lock").open("a+") as gpu:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(gpu, fcntl.LOCK_EX | fcntl.LOCK_NB)
        Worker(args.output, args.deadline_unix).run()
