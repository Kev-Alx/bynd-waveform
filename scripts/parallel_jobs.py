"""Progress supervision and conservative GPU admission for auxiliary jobs."""
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from beyondwaveform.io import atomic_json


def now():
    return datetime.now(timezone.utc).isoformat()


class Supervisor:
    def __init__(self, output):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.state = {"status": "running", "started_at": now()}
        for sig in [signal.SIGTERM, signal.SIGHUP]:
            signal.signal(sig, self.interrupt)
        self.progress(phase="preflight")

    def interrupt(self, signum, frame):
        raise KeyboardInterrupt(f"Interrupted by signal {signum}")

    def progress(self, **values):
        self.state.update(values, updated_at=now())
        atomic_json(self.output / "progress.json", self.state)

    def phase(self, name, command, threads=8):
        self.progress(phase=name, command=command, phase_started_at=now())
        print(f"Starting {name}: {' '.join(command)}", flush=True)
        env = {**os.environ, "OMP_NUM_THREADS": str(threads), "MKL_NUM_THREADS": str(threads), "OPENBLAS_NUM_THREADS": "1"}
        child = subprocess.Popen(command, cwd="/workspace", env=env)
        self.progress(phase_pid=child.pid, cpu_threads=threads)
        started = time.monotonic()
        try:
            while True:
                try:
                    code = child.wait(timeout=30)
                    break
                except subprocess.TimeoutExpired:
                    self.progress(phase_elapsed_seconds=round(time.monotonic()-started, 1))
                    print(f"Heartbeat: {name}, elapsed {self.state['phase_elapsed_seconds']}s", flush=True)
            if code:
                raise RuntimeError(f"{name} failed with exit code {code}")
        finally:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()

    def finish(self, **values):
        self.progress(status="complete", phase="complete", finished_at=now(), **values)

    def fail(self, error):
        self.progress(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=str(error), finished_at=now())


class GpuAdmission:
    """Leave the original job untouched; serialize auxiliary GPU operations."""
    def __init__(self, primary, output):
        self.primary, self.output = Path(primary), Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.lock = Path('/workspace/reproduction-runs/auxiliary-gpu.lock')

    @staticmethod
    def allowed(primary_status, primary_phase, utilization):
        return primary_status == "complete" or (primary_status == "running" and
            primary_phase != "generation" and utilization <= 65)

    @contextmanager
    def slot(self, operation):
        start, last_log = time.monotonic(), 0
        with self.lock.open("a+") as lock:
            while True:
                job = json.loads((self.primary / "job.json").read_text())
                stage = json.loads((self.primary / "progress.json").read_text())
                if job["status"] in {"failed", "interrupted"}:
                    raise RuntimeError("Primary waveform search stopped; auxiliary GPU work waits for its investigation")
                query = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.free", "--format=csv,noheader,nounits"],
                                       capture_output=True, text=True, check=True)
                utilization, free = [float(s.strip()) for s in query.stdout.strip().split(',')]
                admitted = False
                if self.allowed(job["status"], stage.get("phase"), utilization) and free >= 4096:
                    try:
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        admitted = True
                    except BlockingIOError:
                        pass
                if admitted:
                    break
                elapsed = time.monotonic()-start
                atomic_json(self.output / "gpu-admission.json", {"status": "waiting", "operation": operation,
                    "elapsed_seconds": round(elapsed, 1), "primary_phase": stage.get("phase"),
                    "gpu_utilization": utilization, "free_memory_MiB": free, "updated_at": now()})
                if elapsed-last_log >= 30:
                    print(f"GPU queue: {operation}, primary {stage.get('phase')}, utilization {utilization}%, waited {elapsed:.0f}s", flush=True)
                    last_log = elapsed
                time.sleep(5)
            atomic_json(self.output / "gpu-admission.json", {"status": "active", "operation": operation,
                "wait_seconds": round(time.monotonic()-start, 1), "updated_at": now()})
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

