"""CPU registration and execution-gate tests; no model loading/inference."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from improvement import r2_real_runner as runner


@pytest.fixture
def registered_inputs(monkeypatch):
    with TemporaryDirectory(prefix=".r2-runner-tests-", dir=Path(__file__).parent) as temporary:
        folder = Path(temporary)
        run = folder / "execution"
        run.mkdir()
        (run / "plan.json").write_text("{}")
        monkeypatch.setattr(runner.rt, "RUN", run)
        monkeypatch.setattr(runner.rt, "check", lambda plan: True)
        rows, cache, baseline = [], [], {}
        for index in range(3):
            uid = f"test-{index}"
            audio, features, delta, attacked = [folder / f"{uid}-{kind}.bin" for kind in ("audio", "q", "delta", "attacked")]
            for path in (audio, features, delta, attacked):
                path.write_bytes(path.name.encode())
            rows.append({"id": uid, "audio": str(audio), "audio_sha256": runner.rt.digest(audio), "text": "A TEST"})
            cache.append({"id": uid, "q_path": str(features), "q_sha256": runner.rt.digest(features)})
            baseline[uid] = {"id": uid, "input_audio_sha256": rows[-1]["audio_sha256"],
                             "delta": {"path": str(delta), "file_sha256": runner.rt.digest(delta), "value_sha256": "bound-values"},
                             "audio": {"adversarial": {"path": str(attacked), "sha256": runner.rt.digest(attacked)}}}
        cache_path = folder / "cache.json"
        cache_path.write_text(json.dumps({"items": cache}))
        plan = {"cache_audit": str(cache_path)}
        monkeypatch.setattr(runner, "_rows", lambda plan, smoke3=False: rows)
        monkeypatch.setattr(runner, "_baseline_records", lambda: baseline)
        yield plan, run, rows, baseline


def test_registration_freezes_complete_calibration_cohort_without_models(registered_inputs, monkeypatch):
    plan, run, rows, baseline = registered_inputs
    monkeypatch.setattr(runner.rt, "models", lambda *a, **kw: pytest.fail("No models during registration"))
    value = runner.register(plan, "r2-test", smoke3=True)
    assert value["cohort_ids"] == [row["id"] for row in rows]
    assert value["registered_count"] == 3
    assert value["method"]["gradient_bootstrap"] is False
    assert value["condition"] == runner.CANONICAL
    assert len(value["source_files"]) == 3
    runner.check_registration(value)
    assert runner.register(plan, "r2-test", smoke3=True) == value


def test_changed_physical_baseline_input_is_rejected(registered_inputs):
    plan, run, rows, baseline = registered_inputs
    value = runner.register(plan, "r2-test", smoke3=True)
    Path(baseline[rows[0]["id"]]["delta"]["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="Changed registered input"):
        runner.check_registration(value)


def test_calibration_cannot_silently_replace_full_development_cohort(registered_inputs):
    plan, run, rows, baseline = registered_inputs
    with pytest.raises(ValueError, match="complete registered"):
        runner.register(plan, "r2-test", smoke3=False)


def test_failed_exact_baseline_gate_prevents_model_loading(registered_inputs, monkeypatch):
    plan, run, rows, baseline = registered_inputs
    value = runner.register(plan, "r2-test", smoke3=True)
    runner.rt.atomic_json(run / "baseline-preflight/result.json", {"status": "failed", "checked_updates": 0})
    monkeypatch.setattr(runner.rt, "models", lambda *a, **kw: pytest.fail("No models after failed gate"))
    with pytest.raises(ValueError, match="preflight must pass"):
        runner.generate(plan, value)


def test_repair_needs_explicit_prior_actual_cancellation_and_cannot_replace_canonical(registered_inputs):
    plan, run, rows, baseline = registered_inputs
    canonical = runner.register(plan, "canonical", smoke3=True)
    runner.rt.atomic_json(run / "canonical/generation-summary.json", {"status": "completed_all_registered", "condition": runner.CANONICAL})
    outputs = [{"id": row["id"], "r2_status": "completed"} for row in rows]
    runner.rt.write_jsonl(run / "canonical/generated.jsonl", outputs)
    with pytest.raises(ValueError, match="no recorded actual"):
        runner.register(plan, "repair", smoke3=True, repair_ids=[rows[0]["id"]], canonical_name="canonical")
    outputs[0]["r2_status"] = "uniform_start_stalled_with_feasible_descent"
    runner.rt.write_jsonl(run / "canonical/generated.jsonl", outputs)
    repair = runner.register(plan, "repair", smoke3=True, repair_ids=[rows[0]["id"]], canonical_name="canonical")
    assert repair["condition"] == runner.REPAIR
    assert repair["cohort_ids"] == [rows[0]["id"]]
    assert repair["method"]["gradient_bootstrap"] is True
    assert runner.rt.read(run / "canonical/registration.json") == canonical
    assert "Exploratory" in repair["cohort_interpretation"]


def test_incomplete_per_clip_record_is_not_accepted_for_resume():
    with pytest.raises(ValueError, match="Incomplete per-clip"):
        runner._verify_completed_record({"record_complete": False})


@pytest.mark.parametrize("name", ["../outside", "/workspace/elsewhere", "a/b", ".", ""])
def test_run_names_cannot_escape_artifact_directory(name):
    with pytest.raises(ValueError):
        runner.safe_name(name)
