from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_overnight_research as overnight
from run_quality_research import candidates, validate_candidate_files


def test_fresh_pilot_is_deterministic_and_excludes_ids_and_audio():
    index = [{"id": str(i), "audio_sha256": str(i), "role": "validation"} for i in range(60)]
    excluded = [{"id": "0", "audio_sha256": "1"}]
    original = deepcopy(index)
    result = overnight.fresh_pilot(index, excluded)
    assert result == overnight.fresh_pilot(list(reversed(index)), excluded)
    assert index == original
    assert len({r["id"] for r in result}) == 24
    assert all(r["id"] not in {"0", "1"} and r["role"] == "pilot" for r in result)


def test_three_batches_keep_budget_scorers_and_decoder(tmp_path):
    from beyondwaveform.config import load_config
    from beyondwaveform.io import atomic_json
    base = load_config(Path(__file__).resolve().parents[1] / "configs/english_lr005.yaml")
    original = {"candidates": candidates(base)}
    before = deepcopy(original)
    entries = overnight.candidate_grid(original)
    assert original == before
    assert len(entries) == len({e["name"] for e in entries}) == 20
    assert {g: sum(e["group"] == g for e in entries) for g in overnight.GROUPS} == {
        "conditioning": 8, "schedules": 4, "l2-updates": 6}
    for entry in entries:
        config = entry["config"]
        assert config["attack"]["rho"] == 0.1 and config["attack"]["lambda_perc"] == 1 and config["attack"]["steps"] == 50
        assert config["spectral"]["cutoff_hz"] == 6000
        assert config["features"] == base["features"]
        assert config["surrogate"]["decoding"] == base["surrogate"]["decoding"]
        path = tmp_path / (entry["name"] + ".json")
        atomic_json(path, config)
        entry["config_path"] = str(path)
    validate_candidate_files(entries)


def test_optional_timestamp_artifact_identity_keeps_controls_unchanged():
    base = {"base_surrogate_artifacts": {"checkpoint": "small", "nll_reduction": "mean"}}
    assert overnight.effective_plan(base, {"surrogate": {}}) == base
    changed = overnight.effective_plan(base, {"surrogate": {"nll_timestamp_policy": "initial_timestamp"}})
    assert changed["base_surrogate_artifacts"]["nll_timestamp_policy"] == "initial_timestamp"
    assert "nll_timestamp_policy" not in base["base_surrogate_artifacts"]


def test_archived_sources_cannot_be_silently_replaced(tmp_path):
    from beyondwaveform.io import digest
    snapshot = tmp_path / "historical.py"
    snapshot.write_bytes(b"original")
    previous = {"frozen_files": {"/original.py": digest(snapshot)}}
    receipt = {"historical_sources": {"/original.py": {"snapshot": str(snapshot), "sha256": digest(snapshot)}}}
    assert overnight.historical_hashes(previous, receipt) == {str(snapshot): digest(snapshot)}
    snapshot.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Changed historical evidence"):
        overnight.historical_hashes(previous, receipt)


def test_independent_failure_does_not_skip_other_experiments(tmp_path, monkeypatch):
    entries = [{"name": group, "group": group} for group in overnight.GROUPS]
    plan = {"candidates": entries}
    monkeypatch.setattr(sys, "argv", ["overnight", "--output", str(tmp_path)])
    monkeypatch.setattr(overnight.protocol, "read", lambda path: plan)
    monkeypatch.setattr(overnight, "check", lambda plan: None)
    monkeypatch.setattr(overnight, "promotions", lambda rows: [])
    monkeypatch.setattr(overnight, "frozen_decision", lambda *args: None)

    class Supervisor:
        def __init__(self, *args): pass
        def progress(self, **kwargs): pass
        def fail(self, *args): pass
    monkeypatch.setattr(overnight, "Supervisor", Supervisor)
    called, reports = [], []
    def run(output, plan, entry, role, rows, failures, supervisor):
        called.append(entry["group"])
        if entry["group"] == "conditioning":
            raise RuntimeError("synthetic independent failure")
        rows.append({"name": entry["name"], "pool": "pilot", "group": entry["group"], "run": str(tmp_path),
                     "summary_sha256": "fixture", "error_percent": 50.0,
                     "quality": {"dnsmos": 3.5, "nisqa": 3.5, "utmos": 3.5}})
    monkeypatch.setattr(overnight, "run_case", run)
    monkeypatch.setattr(overnight, "report", lambda output, plan, rows, failures, status, outcome=None: reports.append(status))
    with pytest.raises(RuntimeError, match="1 case.*failed"):
        overnight.main()
    assert called == list(overnight.GROUPS)
    assert reports[-1] == "failed"
