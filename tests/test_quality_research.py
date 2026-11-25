from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_quality_research import (candidates, promotions, configure_scorers, verify_stage,
                                  validate_candidate_files, defense_check_plan)


def test_all_serialized_research_configs_pass_the_actual_generation_schema(tmp_path):
    from beyondwaveform.config import load_config
    from beyondwaveform.io import atomic_json
    base = load_config(Path(__file__).resolve().parents[1] / "configs/english_lr005.yaml")
    entries = candidates(base)
    for entry in entries:
        path = tmp_path / (entry["name"] + ".json")
        atomic_json(path, entry["config"])
        entry["config_path"] = str(path)
    validate_candidate_files(entries)


def test_completed_defense_plan_uses_verified_historical_snapshot(tmp_path):
    from beyondwaveform.io import atomic_json, digest
    snapshot = tmp_path / "historical-config.py"
    snapshot.write_bytes(b"original loader")
    config = tmp_path / "defense.json"
    atomic_json(config, {"kind": "whisper", "decoding": {"temperature": 0.0}})
    original = {"frozen_files": {"/original/config.py": digest(snapshot)}, "settings": [{"target": str(config)}]}
    unchanged = deepcopy(original)
    plan = {"historical_defense_sources": {"/original/config.py": {"snapshot": str(snapshot), "sha256": digest(snapshot)}}}
    checked = defense_check_plan(plan, original)
    assert original == unchanged
    assert checked["frozen_files"] == {str(snapshot): digest(snapshot)}
    snapshot.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="Historical source snapshot changed"):
        defense_check_plan(plan, original)


def row(name, wer, dnsmos=3.1, nisqa=3.8, utmos=3.7):
    return {"name": name, "error_percent": wer,
            "quality": {"dnsmos": dnsmos, "nisqa": nisqa, "utmos": utmos}}


def test_research_grid_preserves_paper_budget_and_input_config():
    base = {"features": {"normalize": False}, "save_delta": False,
            "surrogate": {"nll_reduction": "mean"}, "attack": {"alpha_hf": 1.0}}
    original = deepcopy(base)
    grid = candidates(base)
    assert base == original
    assert len(grid) == 6
    assert {(r["config"]["surrogate"]["nll_reduction"], r["config"]["attack"]["learning_rate"])
            for r in grid} == {(reduction, lr) for reduction in ("mean", "sum") for lr in (0.02, 0.05, 0.1)}
    assert all(r["config"]["attack"]["rho"] == 0.1 and r["config"]["attack"]["lambda_perc"] == 1 and
               r["config"]["attack"]["steps"] == 50 and r["config"]["save_delta"] for r in grid)


def test_promote_quality_match_then_attack_strength_comparator():
    rows = [row("good-quality", 69), row("strong-low-quality", 78, utmos=3.2),
            row("weak-good-quality", 50), row("fails-gate", 99, utmos=2.5)]
    outcome = promotions(rows)
    assert outcome["names"] == ["good-quality", "strong-low-quality"]
    assert outcome["quality_target_available"]
    assert "fails-gate" not in outcome["eligible"]


def test_promotion_fallback_preserves_quality_and_reports_target_miss():
    rows = [row("closest-quality", 65, utmos=3.5), row("stronger", 78, utmos=3.1)]
    outcome = promotions(rows)
    assert outcome["names"] == ["closest-quality", "stronger"]
    assert not outcome["quality_target_available"]


def test_nan_quality_and_nonfinite_wer_cannot_win_selection():
    rows = [row("valid", 65), row("nan-mos", 99, utmos=float("nan")), row("nan-wer", float("nan"))]
    assert promotions(rows)["names"] == ["valid"]
    with pytest.raises(ValueError, match="No pilot candidate"):
        promotions([row("bad", 99, utmos=2.49)])


def test_quality_scorer_rejects_changed_weights_before_inference(tmp_path):
    checkpoint = tmp_path / "changed.onnx"
    checkpoint.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Changed quality weights"):
        configure_scorers({"hashes": {str(checkpoint): "incorrect"}})


@pytest.mark.parametrize("delta_scale,passes", [(0.1, True), (0.2, False)])
def test_saved_delta_bound_is_independently_checked(tmp_path, monkeypatch, delta_scale, passes):
    import torch
    from beyondwaveform.evaluation import aggregate_errors, error_counts
    from beyondwaveform.io import atomic_json, digest, save_audio, write_jsonl
    from beyondwaveform.pipeline import config_fingerprint
    import beyondwaveform.models.feature as features
    run = tmp_path / "run"
    run.mkdir()
    asset = save_audio(tmp_path / "source.wav", torch.linspace(-0.1, 0.1, 1600))
    source = {"id": "one", "text": "hello world", "audio": asset["path"], "audio_sha256": asset["sha256"]}
    manifest = tmp_path / "pilot.jsonl"
    write_jsonl(manifest, [source])
    config = {"features": {}, "surrogate": {"nll_reduction": "mean"}, "attack": {"eps": 1e-8}}
    identity = {"config": config, "manifest_sha256": digest(manifest), "ids": ["one"]}
    fingerprint = config_fingerprint(identity)
    variants = ("clean", "reconstruction", "adversarial")
    record = {"id": "one", "text": "hello world", "input_audio_sha256": asset["sha256"],
              "audio": {v: asset for v in variants},
              "trace": [{"step": i+1, "radius_after_update": 0.1} for i in range(50)]}
    write_jsonl(run / "generated.jsonl", [record])
    atomic_json(run / "run.json", {**identity, "fingerprint": fingerprint, "status": "complete", "count": 1,
                "features": {}, "surrogate": {"nll_reduction": "mean"}, "generated_sha256": digest(run / "generated.jsonl")})
    rows = [{"id": "one", "variant": v, "reference": "hello world", "hypothesis": "hello world",
             "audio_sha256": asset["sha256"], "quality": {m: 3.5 for m in ("dnsmos", "nisqa", "utmos")},
             **error_counts("hello world", "hello world")} for v in variants]
    write_jsonl(run / "evaluation/utterances.jsonl", rows)
    atomic_json(run / "evaluation/summary.json", {"target_artifacts": {"nll_reduction": "mean"},
        "target": {"kind": "whisper", "nll_reduction": "mean"}, "evaluation_scope": "surrogate", "roles": ["pilot"],
        "metric": "WER", "generation_fingerprint": fingerprint, "manifest_sha256": digest(manifest),
        "generation_run": str(run.resolve()), "research_scorer_manifest": {}, "evaluation_environment": {},
        "utterances_sha256": digest(run / "evaluation/utterances.jsonl"), "evaluation_ids": ["one"],
        "summary": {v: {"count": 1, **aggregate_errors([rows[i]]), "quality": rows[i]["quality"]} for i, v in enumerate(variants)}})
    torch.save(torch.ones(4, 8) * delta_scale, run / "one.delta.pt")
    monkeypatch.setattr(features, "load_feature_models", lambda *args: (lambda audio: torch.ones(4, 8), None, {}))
    monkeypatch.setattr(torch.Tensor, "cuda", lambda tensor, *args, **kwargs: tensor)
    original_load = torch.load
    monkeypatch.setattr(torch, "load", lambda path, **kwargs: original_load(path, map_location="cpu", weights_only=True))
    plan = {"pools": {"pilot": {"manifest": str(manifest), "sha256": digest(manifest), "ids": ["one"]}},
            "base_surrogate_artifacts": {}, "feature_artifacts": {}, "quality": {}, "runtime": {}}
    if passes:
        verify_stage(plan, run, config, "pilot")
        assert (run / "verification.json").exists()
    else:
        with pytest.raises(ValueError, match="Saved delta does not satisfy"):
            verify_stage(plan, run, config, "pilot")
