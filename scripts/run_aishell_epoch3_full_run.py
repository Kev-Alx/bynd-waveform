"""Run the frozen AISHELL protocol with an optional operational time limit.

Numerical methods and cache validation come from the unchanged, hash-pinned
workers. This entry point changes only lifecycle controls and their status text.
"""
import argparse
from datetime import datetime
import fcntl
import json
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_aishell_epoch3 as frozen
from run_aishell_epoch3_extended import ExtendedWorker


class RuntimeControls:
    def __init__(self, output, deadline=None, device="cuda"):
        # The frozen constructor requires a timestamp. Replace its operational
        # metadata immediately; the timestamp never controls an unlimited run.
        super().__init__(output, time.time() if deadline is None else deadline, device)
        self.deadline = deadline
        self.runtime_sha = frozen.digest(__file__)
        self.job.update(deadline_taipei=(datetime.fromtimestamp(deadline, frozen.TAIPEI).isoformat(timespec="seconds")
                                        if deadline is not None else None),
                        runtime_worker=str(Path(__file__).resolve()), runtime_worker_sha256=self.runtime_sha,
                        runtime_limit="none" if deadline is None else "explicit deadline",
                        overrides_frozen_session_limit=True)

    def validate_inputs(self):
        super().validate_inputs()
        assert frozen.digest(__file__) == self.runtime_sha, "Runtime controls changed while running"

    def publish(self):
        super().publish()
        old = (f"This session stops automatically by **{self.job['deadline_taipei']} (Asia/Taipei)** "
               "after at most five hours. Resume starts another bounded session.")
        replacement = ("**No runtime limit.** This detached worker continues until all recordings finish or you stop it. "
                       "The resume command also runs without a time limit by default. "
                       "This runtime policy supersedes the historical five-hour session limit in the frozen plan."
                       if self.deadline is None else
                       f"This session stops automatically by **{self.job['deadline_taipei']} (Asia/Taipei)**. "
                       "Resume runs without a time limit unless --max-hours is supplied.")
        for path in [self.output / "JOB_STATUS.md", frozen.ROOT / "reproduction-runs/mandarin-cer-full-dev/JOB_STATUS.md",
                     frozen.ROOT / "reproduction-runs/mandarin-cer-corrected/WORKFLOW_STATUS.md"]:
            path.write_text(path.read_text().replace(old, replacement))
        path = self.output / "progress.json"
        progress = json.loads(path.read_text())
        progress.update(runtime_limit=self.job["runtime_limit"], runtime_worker_sha256=self.runtime_sha)
        frozen.atomic_json(path, progress)

    def run(self):
        previous = {}

        def stop(signum, frame):
            raise frozen.RequestedStop("session time limit" if signum == signal.SIGALRM else f"signal {signum}")

        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGALRM):
            previous[sig] = signal.signal(sig, stop)
        # Zero explicitly disarms the timer for unlimited execution.
        signal.setitimer(signal.ITIMER_REAL, 0 if self.deadline is None else max(.01, self.deadline - time.time() - 3))
        try:
            frozen.atomic_json(self.output / "job.json", self.job)
            self.validate_inputs()
            self.phase = "verifying-committed-checkpoints"
            self.restore()
            self.publish()
            frozen.torch.set_num_threads(8)
            sys.path.insert(0, self.plan["tokenizer_path"])
            import jieba
            assert jieba.__version__ == "0.42.1" and frozen.digest(Path(jieba.__file__).with_name("dict.txt")) == self.plan["tokenizer_dictionary_sha256"]
            tokenizer = jieba.Tokenizer()
            tokenizer.tmp_dir = str(self.output)
            tokenizer.initialize()
            print(f"Resuming with {len(self.generated)} generated audios and {len(self.scores)} transcriptions already committed.", flush=True)
            asr, meta = frozen.load_whisper(self.config["surrogate"], self.device)
            assert meta["sha256"] == self.plan["checkpoint_sha256"]
            encoder, vocoder, feature_meta = frozen.load_feature_models(self.config["features"], self.device)
            for row in self.rows:
                uid = row["id"]
                if uid not in self.generated:
                    self.phase = f"generating:{uid}"
                    self.publish()
                    self.generate_one(row, asr, encoder, vocoder)
                    print(f"Generated {len(self.generated)}/{len(self.rows)}: {uid}", flush=True)
                    self.publish()
                for variant in frozen.VARIANTS:
                    if (uid, variant) not in self.scores:
                        self.phase = f"scoring:{uid}:{variant}"
                        self.score_one(row, variant, asr, tokenizer)
                        self.publish()
            frozen.write_jsonl(self.output / "utterances.jsonl", list(self.scores.values()))
            self.validate_inputs()
            frozen.atomic_json(self.output / "independent-audit.json", {"status": "passed", "count": len(self.rows),
                               "generation_records": len(self.generated), "character_distance_checks": len(self.scores),
                               "word_distance_checks": len(self.scores), "plan_sha256": self.plan_sha,
                               "utterances_sha256": frozen.digest(self.output / "utterances.jsonl")})
            self.status, self.phase = "complete", "complete"
        except frozen.RequestedStop as reason:
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
            self.job.update(status=self.status, finished_at=datetime.now(frozen.TAIPEI).isoformat(timespec="seconds"),
                            generated=len(self.generated), scored_variants=len(self.scores))
            frozen.atomic_json(self.output / "job.json", self.job)
            frozen.atomic_json(self.session_file, self.job)
            self.publish()
            for sig, handler in previous.items():
                signal.signal(sig, handler)


class FullRunWorker(RuntimeControls, frozen.Worker):
    pass


class FullRunExtendedWorker(RuntimeControls, ExtendedWorker):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deadline-unix", type=float, default=None)
    args = parser.parse_args()
    plan = json.loads((args.output / "plan.json").read_text())
    worker_type = FullRunExtendedWorker if "resume_parent" in plan else FullRunWorker
    with (args.output / "worker.lock").open("a+") as lock, (ROOT / "reproduction-runs/auxiliary-gpu.lock").open("a+") as gpu:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(gpu, fcntl.LOCK_EX | fcntl.LOCK_NB)
        worker_type(args.output, args.deadline_unix).run()
