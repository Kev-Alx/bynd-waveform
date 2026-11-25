import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import update_job_status as status


@pytest.mark.parametrize("supervisor_alive,expected", [(True, "finishing"), (False, "stopped")])
def test_exited_child_waits_for_live_supervisor_final_record(tmp_path, monkeypatch, supervisor_alive, expected):
    folder = tmp_path / "research"
    folder.mkdir()
    (folder / "job.json").write_text(json.dumps({"status": "running", "child_pid": 12, "supervisor_pid": 11}))
    (folder / "progress.json").write_text(json.dumps({"status": "complete", "phase": "complete"}))
    monkeypatch.setattr(status, "alive", lambda pid: supervisor_alive if pid == 11 else False)
    assert status.snapshot(tmp_path, "research", "Research")["status"] == expected


def test_final_exit_zero_has_priority_over_process_liveness(tmp_path, monkeypatch):
    folder = tmp_path / "research"
    folder.mkdir()
    (folder / "job.json").write_text(json.dumps({"status": "complete", "exit_code": 0, "child_pid": 12}))
    (folder / "progress.json").write_text(json.dumps({"status": "complete", "phase": "complete"}))
    monkeypatch.setattr(status, "alive", lambda pid: False)
    assert status.snapshot(tmp_path, "research", "Research")["status"] == "complete"


def test_mandarin_eta_preserves_completed_jobs_in_shared_status(tmp_path, monkeypatch):
    for name in ["english-defenses-999", "research"]:
        folder = tmp_path / name
        folder.mkdir()
        (folder / "job.json").write_text(json.dumps({"status": "complete", "exit_code": 0}))
        (folder / "progress.json").write_text(json.dumps({"phase": "complete"}))
    folder = tmp_path / "mandarin-cer-evaluation"
    folder.mkdir()
    (folder / "plan.json").write_text("{}")
    (folder / "job.json").write_text(json.dumps({"status": "running", "child_pid": 12}))
    (folder / "progress.json").write_text(json.dumps({
        "phase": "test-000-generation", "clip_progress": {"completed": 64, "total": 7176, "percent": 0.9},
        "estimated_remaining_seconds": 36000, "estimated_remaining_range_seconds": [28800, 54000],
        "estimated_completion_taipei": "2026-10-25T16:00:00+08:00", "eta_basis": "Observed validation timing"}))
    monkeypatch.setattr(status, "alive", lambda pid: True)
    result = status.update(tmp_path, "research")
    rendered = (tmp_path / "JOB_STATUS.md").read_text()
    assert result["status"] == "running"
    assert [job["status"] for job in result["jobs"]] == ["complete", "complete", "running"]
    assert "64/7176" in rendered and "10.0 hours" in rendered
    assert "2026-10-25T16:00:00+08:00" in rendered and "Asia/Taipei" in rendered


def test_continuation_keeps_original_batch_complete(tmp_path, monkeypatch):
    for name in ["english-defenses-999", "research", "mandarin-cer-evaluation"]:
        folder = tmp_path / name
        folder.mkdir()
        (folder / "job.json").write_text(json.dumps({"status": "complete", "exit_code": 0}))
        (folder / "progress.json").write_text(json.dumps({"phase": "complete"}))
    (tmp_path / "mandarin-cer-evaluation/plan.json").write_text("{}")
    folder = tmp_path / "mandarin-cer-next-3000"
    folder.mkdir()
    (folder / "plan.json").write_text("{}")
    (folder / "job.json").write_text(json.dumps({"status": "running", "child_pid": 15}))
    (folder / "progress.json").write_text(json.dumps({
        "phase": "test-000-generation", "clip_progress": {"completed": 0, "total": 3000, "percent": 0.0},
        "estimated_remaining_seconds": 43200, "estimated_remaining_range_seconds": [34560, 64800],
        "estimated_completion_taipei": "2026-10-26T07:00:00+08:00", "eta_basis": "Completed batch timing"}))
    monkeypatch.setattr(status, "alive", lambda pid: True)
    result = status.update(tmp_path, "research")
    rendered = (tmp_path / "JOB_STATUS.md").read_text()
    assert [job["status"] for job in result["jobs"]] == ["complete", "complete", "complete", "running"]
    assert "0/3000" in rendered and "12.0 hours" in rendered and "5,000 total" in rendered


def test_corrected_cer_pilot_is_visible_alongside_prior_runs(tmp_path, monkeypatch):
    for name in ['english-defenses-999', 'research']:
        folder = tmp_path / name
        folder.mkdir()
        (folder / 'job.json').write_text(json.dumps({'status': 'complete', 'exit_code': 0}))
    folder = tmp_path / 'mandarin-cer-corrected'
    folder.mkdir()
    (folder / 'plan.json').write_text('{}')
    (folder / 'job.json').write_text(json.dumps({'status': 'running', 'child_pid': 15}))
    (folder / 'progress.json').write_text(json.dumps({
        'phase': 'pilot-clean-selected-scoring', 'clip_progress': {'completed': 12, 'total': 48, 'percent': 25.0},
        'estimated_remaining_seconds': 180, 'estimated_remaining_range_seconds': [126, 324],
        'estimated_completion_taipei': '2026-10-26T20:00:00+08:00', 'eta_basis': 'Observed pilot timing'}))
    monkeypatch.setattr(status, 'alive', lambda pid: True)
    result = status.update(tmp_path, 'research')
    assert result['jobs'][-1]['name'] == 'mandarin-cer-corrected'
    assert result['status'] == 'running'
    assert '12/48' in (tmp_path / 'JOB_STATUS.md').read_text()


@pytest.mark.parametrize('sequence_status,expected', [('running','running'),('needs_review','needs_attention'),('criteria_met','complete')])
def test_corrected_sequence_controls_overall_completion(tmp_path,sequence_status,expected):
    for name in ['english-defenses-999','research','mandarin-cer-corrected']:
        folder=tmp_path/name
        folder.mkdir()
        (folder/'job.json').write_text(json.dumps({'status':'complete','exit_code':0}))
    folder=tmp_path/'mandarin-cer-corrected'
    (folder/'plan.json').write_text('{}')
    (folder/'sequence.json').write_text(json.dumps({'status':sequence_status,'phase':'comparison-gate'}))
    assert status.update(tmp_path,'research')['status']==expected
