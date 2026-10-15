"""Run a command with a live log and an explicit completion/failure record.

Use inside a detached tmux session for jobs that must outlive the chat terminal.
The wrapped command keeps its existing progress indicators.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys


def now():
    return datetime.now(timezone.utc).isoformat()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True)
    parser.add_argument("--status", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("provide a command after --")
    log_path, status_path = Path(args.log).resolve(), Path(args.status).resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.parent.mkdir(parents=True, exist_ok=True)
    state = {"status": "running", "started_at": now(), "command": command,
             "cwd": str(Path.cwd()), "supervisor_pid": os.getpid(), "log": str(log_path)}

    def save_status():
        temporary = status_path.with_suffix(status_path.suffix + ".tmp")
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(status_path)

    def interrupt(signum, frame):
        state["signal"] = signum
        raise KeyboardInterrupt

    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, interrupt)
    process = None
    save_status()
    with log_path.open("ab") as log:
        def emit(data):
            log.write(data)
            log.flush()
            try:
                sys.stdout.buffer.write(data)
                sys.stdout.buffer.flush()
            except BrokenPipeError:
                pass

        emit(f"\nJob started at {state['started_at']} (supervisor {os.getpid()})\n".encode())
        try:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       env={**os.environ, "PYTHONUNBUFFERED": "1"})
            state["child_pid"] = process.pid
            save_status()
            while chunk := process.stdout.read1(4096):
                emit(chunk)
            code = process.wait()
            state.update(status="complete" if code == 0 else "failed", exit_code=code)
        except KeyboardInterrupt:
            state.update(status="interrupted", exit_code=128 + state.get("signal", signal.SIGINT))
        except OSError as error:
            state.update(status="failed", exit_code=127, error=str(error))
            emit(f"Job error: {error}\n".encode())
        finally:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            state["finished_at"] = now()
            save_status()
        emit(f"\nJob {state['status']} at {state['finished_at']}; exit code {state['exit_code']}\n".encode())
    return state["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
