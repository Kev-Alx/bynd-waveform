"""The job record must distinguish completed work from a stopped command."""
import json
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("code, expected", [(0, "complete"), (7, "failed")])
def test_job_records_exit_and_captures_output(tmp_path, code, expected):
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_logged.py"
    log, status = tmp_path / "job.log", tmp_path / "job.json"
    result = subprocess.run([sys.executable, str(script), "--log", str(log), "--status", str(status),
                             "--", sys.executable, "-c", f"print('progress 1/1'); raise SystemExit({code})"],
                            capture_output=True, timeout=10)
    record = json.loads(status.read_text())
    assert result.returncode == code
    assert record["status"] == expected
    assert record["exit_code"] == code
    assert record["started_at"] <= record["finished_at"]
    assert "progress 1/1" in log.read_text()
    assert "progress 1/1" in result.stdout.decode()


def test_missing_executable_records_failure(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_logged.py"
    status = tmp_path / "job.json"
    result = subprocess.run([sys.executable, str(script), "--log", str(tmp_path / "job.log"),
                             "--status", str(status), "--", str(tmp_path / "missing")],
                            capture_output=True, timeout=10)
    record = json.loads(status.read_text())
    assert result.returncode == 127
    assert record["status"] == "failed"
    assert "error" in record
