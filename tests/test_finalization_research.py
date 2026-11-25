from pathlib import Path
import copy
import sys

import pytest
import torch

from beyondwaveform.config import AttackConfig, SpectralConfig
from beyondwaveform.io import atomic_json, digest, read_jsonl, save_audio, write_jsonl
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from scripts import run_finalization_research as study


@pytest.mark.parametrize("field,value", [
    ("rate", .02), ("prefix", "initial_timestamp"), ("reduction", "mean"), ("case", "lower")])
def test_fixed_family_cannot_silently_change_the_scientific_question(field, value):
    config = {"attack": vars(AttackConfig(learning_rate=.1)), "spectral": vars(SpectralConfig()),
              "surrogate": {"nll_reduction": "sum"}}
    family = {"name": study.FAMILIES[0], "config": copy.deepcopy(config)}
    study.validate_family(family)
    if field == "rate":
        family["config"]["attack"]["learning_rate"] = value
    else:
        key = {"prefix": "nll_timestamp_policy", "reduction": "nll_reduction", "case": "nll_transcript_policy"}[field]
        family["config"]["surrogate"][key] = value
    with pytest.raises(ValueError, match="Changed fixed family"):
        study.validate_family(family)


def score(wer=70, utmos=3.5):
    return {"error_percent": wer, "quality": {"dnsmos": 3.1, "nisqa": 4.0, "utmos": utmos}}


def terminal_summary(reconstruction_utmos=4.0):
    return {"clean": score(4, 4), "reconstruction": score(5, reconstruction_utmos)}


def test_directional_gate_compares_average_to_its_matched_fresh_control():
    gate = study.directional_gate(terminal_summary(), score(72, 3.6), score(75, 3.5))
    assert gate["eligible"]
    assert gate["utmos_gain"] == pytest.approx(.1)
    assert gate["wer_point_loss"] == 3


@pytest.mark.parametrize("average,matched,rec,failed", [
    (score(72, 3.54), score(75, 3.5), 4, "utmos_gain"),
    (score(69, 3.6), score(75, 3.5), 4, "wer_loss"),
    (score(72, 3.6), score(75, 3.5), 3.4, "terminal_reconstruction"),
    (score(72, 2.5), score(75, 2.4), 4, "finite_quality"),
])
def test_predeclared_promotion_requires_every_directional_and_reconstruction_check(average, matched, rec, failed):
    gate = study.directional_gate(terminal_summary(rec), average, matched)
    assert not gate["eligible"] and not gate["checks"][failed]


def fixture(tmp_path, monkeypatch):
    import beyondwaveform.pipeline as pipeline
    monkeypatch.setattr(study.protocol, "runtime", lambda: {"test": "cpu"})
    monkeypatch.setattr(study.protocol, "configure_scorers", lambda manifest: None)
    audio = save_audio(tmp_path / "input.wav", torch.linspace(-.3, .3, 1024))
    manifest = tmp_path / "pilot.jsonl"
    source = [{"id": f"clip{i:02}", "text": "FIXED REFERENCE", "language": "en", "role": "pilot",
               "audio": audio["path"], "audio_sha256": audio["sha256"]} for i in range(24)]
    write_jsonl(manifest, source)
    config = {"seed": 2026, "features": {}, "surrogate": {"checkpoint": "toy", "language": "en"},
        "attack": vars(AttackConfig(learning_rate=.05)),
        "spectral": vars(SpectralConfig(n_fft=8, hop_length=2, win_length=8))}
    family = {"name": study.FAMILIES[0], "config": config, "surrogate_artifacts": {"toy": "surrogate"}}
    plan = {"families": [family], "pools": {"pilot": {"manifest": str(manifest), "sha256": digest(manifest),
        "ids": [row["id"] for row in source]}}, "feature_artifacts": {"toy": "features"},
        "quality": {"toy": "quality"}, "runtime": {"test": "cpu"}, "scope": "CPU integrity fixture", "waveform_tolerance": 2e-5}

    class Encoder(torch.nn.Module):
        def forward(self, x):
            return x[:256].reshape(32, 8)

    class Vocoder(torch.nn.Module):
        def forward(self, q):
            return torch.tanh(q.flatten().repeat_interleave(4))

    class Surrogate(torch.nn.Module):
        def nll(self, audio, text):
            return (audio+.3).square().mean()

        def transcribe(self, audio):
            return "FIXED REFERENCE"

    monkeypatch.setattr(study, "load_models", lambda plan, family, device="cuda", surrogate=True:
        (Encoder(), Vocoder(), Surrogate() if surrogate else None))
    monkeypatch.setattr(pipeline, "load_target", lambda config, device: (Surrogate(), family["surrogate_artifacts"]))
    monkeypatch.setattr(pipeline, "quality_provenance", lambda root: {"toy": "fixed"})

    class Scorer:
        metrics = ("dnsmos", "nisqa", "utmos")

        def __init__(self, root):
            pass

        def __call__(self, audio):
            return {"dnsmos": 3.1, "nisqa": 4.0, "utmos": 3.8}

    monkeypatch.setattr(pipeline, "QualityScorer", Scorer)
    monkeypatch.setattr(study, "evaluate", lambda *args, **kwargs: pipeline.evaluate(*args, device="cpu", **kwargs))
    return plan, family, source


def test_smoke_integrity_runs_one_shared_trajectory_three_exports_and_reaudits(tmp_path, monkeypatch):
    plan, family, source = fixture(tmp_path, monkeypatch)
    output = tmp_path / "research"
    calls = []
    original = study.collect_trajectory

    def collect(*args):
        calls.append(True)
        return original(*args)

    monkeypatch.setattr(study, "collect_trajectory", collect)
    study.generate_family(output, plan, family, "pilot", smoke=True, device="cpu")
    assert len(calls) == 1
    root = study.family_root(output, "pilot", family, True)
    shared = study.protocol.read(root / "shared/records/clip00.json")
    assert len(shared["snapshots"]) == 10
    assert set(shared["deltas"]) == set(study.MODES)
    assert len(shared["trace"]) == 50
    assert not (root / "shared/clip00.incomplete").exists()
    for mode in study.MODES:
        data = read_jsonl(root / mode / "generated.jsonl")
        assert [r["id"] for r in data] == [source[0]["id"]]
        assert data[0]["trace_scope"] == "terminal optimizer trajectory"
        assert data[0]["terminal_trace_radius"] == shared["trace"][-1]["radius_after_update"]
        study.evaluate_mode(output, plan, family, "pilot", mode, smoke=True)
    study.verify_family(output, plan, family, "pilot", smoke=True, device="cpu")
    rows = study.rows_for_family(output, plan, family, "pilot", True)
    assert len(rows) == 3 and all(row["smoke"] for row in rows)
    assert all(row["count"] == 1 and row["reference_units"] == 2 for row in rows)
    for mode in study.MODES:
        audit = study.protocol.read(root / mode / "verification.json")
        assert audit["count"] == 1 and audit["additional_optimization_updates"] == 0
        assert audit["items"][0]["max_resynthesis_abs_error"] == 0
    # A complete shared receipt resumes without another optimization run.
    study.generate_family(output, plan, family, "pilot", smoke=True, device="cpu")
    assert len(calls) == 1
    # Corruption is rejected even when metadata and completed outputs exist.
    saved = Path(shared["snapshots"]["41"]["path"])
    saved.write_bytes(b"changed snapshot")
    with pytest.raises(ValueError, match="Changed shared trajectory artifact"):
        study.generate_family(output, plan, family, "pilot", smoke=True, device="cpu")


def test_partial_per_clip_trajectory_is_refused_before_another_attack(tmp_path, monkeypatch):
    plan, family, _ = fixture(tmp_path, monkeypatch)
    output = tmp_path / "research"
    root = study.family_root(output, "pilot", family, True)
    marker = root / "shared/clip00.incomplete"
    marker.parent.mkdir(parents=True)
    marker.write_text("interrupted before commit")
    monkeypatch.setattr(study, "collect_trajectory", lambda *args: pytest.fail("Partial trajectory must not be rerun"))
    with pytest.raises(ValueError, match="Partial trajectory"):
        study.generate_family(output, plan, family, "pilot", smoke=True, device="cpu")


def test_mode_identity_explicitly_distinguishes_returned_delta_and_terminal_trace(tmp_path, monkeypatch):
    plan, family, _ = fixture(tmp_path, monkeypatch)
    one = study.identity(plan, family, "pilot", "terminal", smoke=True)
    averaged = study.identity(plan, family, "pilot", "average_41_50", smoke=True)
    assert one["ids"] == ["clip00"]
    assert len(study.identity(plan, family, "pilot", "terminal")["ids"]) == 24
    assert study.config_fingerprint(one) != study.config_fingerprint(averaged)
    assert averaged["config"]["shared_optimization_updates"] == 50
    assert averaged["config"]["additional_optimization_updates"] == 0
    assert averaged["config"]["snapshot_steps"] == list(range(41, 51))
    assert averaged["config"]["trace_scope"].startswith("terminal")


def test_verification_rejects_saved_returned_delta_even_if_receipt_is_updated(tmp_path, monkeypatch):
    plan, family, _ = fixture(tmp_path, monkeypatch)
    output = tmp_path / "research"
    study.generate_family(output, plan, family, "pilot", smoke=True, device="cpu")
    for mode in study.MODES:
        study.evaluate_mode(output, plan, family, "pilot", mode, smoke=True)
    root = study.family_root(output, "pilot", family, True)
    shared_path = root / "shared/records/clip00.json"
    shared = study.protocol.read(shared_path)
    path = Path(shared["deltas"]["average_41_50"]["path"])
    delta = torch.load(path, weights_only=True)
    torch.save(delta+.001, path)
    # Rewriting the local shared receipt cannot replace the frozen arithmetic.
    shared["deltas"]["average_41_50"]["sha256"] = digest(path)
    atomic_json(shared_path, shared)
    for mode in study.MODES:
        folder = root / mode
        rows = read_jsonl(folder / "generated.jsonl")
        rows[0]["shared_trajectory"]["sha256"] = digest(shared_path)
        if mode == "average_41_50":
            rows[0]["delta"]["sha256"] = digest(path)
        write_jsonl(folder / "generated.jsonl", rows)
        metadata = study.protocol.read(folder / "run.json")
        metadata["generated_sha256"] = digest(folder / "generated.jsonl")
        atomic_json(folder / "run.json", metadata)
    with pytest.raises(ValueError, match="fixed CPU finalization arithmetic"):
        study.verify_family(output, plan, family, "pilot", smoke=True, device="cpu")
