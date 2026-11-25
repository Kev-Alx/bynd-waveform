from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_quality_research import candidates, validate_candidate_files
from run_quality_contracts import candidate_grid, contract_key, reconstruction_gate, effective_plan, frozen_decision, read_scores


def test_contract_grid_changes_only_preprocessing_and_generator(tmp_path):
    from beyondwaveform.config import load_config
    from beyondwaveform.io import atomic_json
    base = load_config(Path(__file__).resolve().parents[1] / "configs/english_lr005.yaml")
    previous = {"candidates": candidates(base)}
    original = deepcopy(previous)
    entries = candidate_grid(previous, tmp_path / "old")
    assert previous == original
    assert len(entries) == len({e["name"] for e in entries}) == 24
    assert sum(not e.get("cached") for e in entries) == 18
    assert sum(e.get("contract_probe", False) for e in entries) == 3
    for entry in entries:
        config = deepcopy(entry["config"])
        assert contract_key(config["features"]) == entry["contract"]
        config["features"]["normalize"] = False
        config["features"]["vocoder_variant"] = "regular"
        assert config in [c["config"] for c in previous["candidates"]]
        assert config["spectral"]["cutoff_hz"] == 6000
        path = tmp_path / (entry["name"] + ".json")
        atomic_json(path, entry["config"])
        entry["config_path"] = str(path)
    validate_candidate_files(entries)


@pytest.mark.parametrize("wer,utmos,passes", [(8.0, 3.6, True), (8.01, 3.6, False),
    (5.0, 3.589, False), (float("nan"), 3.8, False), (5.0, float("inf"), False)])
def test_reconstruction_screen_checks_both_intelligibility_and_quality(wer, utmos, passes):
    result = {"clean": {"error_percent": 3.0}, "reconstruction": {"error_percent": wer,
              "quality": {"dnsmos": 3.1, "nisqa": 3.8, "utmos": utmos}}}
    assert reconstruction_gate(result) is passes


def test_per_candidate_artifact_contract_preserves_base_plan():
    metadata = {"vocoder_variant": "prematched", "normalize": True, "layer": 6, "vocoder_sha256": "different"}
    plan = {"feature_artifacts": {"old": True}, "feature_contracts": {"prematched-normalized": metadata}}
    original = deepcopy(plan)
    config = {"features": {"vocoder_variant": "prematched", "normalize": True, "layer": 6}}
    assert effective_plan(plan, config)["feature_artifacts"] == metadata
    assert plan == original
    config["features"]["layer"] = 7
    with pytest.raises(ValueError, match="Wrong feature contract"):
        effective_plan(plan, config)


def test_reconstruction_or_selection_decision_cannot_change_on_resume(tmp_path):
    path = tmp_path / "gates/a.json"
    frozen_decision(path, {"passed": False, "summary_sha256": "fixed"})
    frozen_decision(path, {"passed": False, "summary_sha256": "fixed"})
    with pytest.raises(ValueError, match="Decision changed on resume"):
        frozen_decision(path, {"passed": True, "summary_sha256": "changed"})


def test_score_review_accepts_variants_and_rejects_duplicate_pairs(tmp_path):
    import json
    path = tmp_path / "scores.jsonl"
    rows = [{"id": "one", "variant": v} for v in ("clean", "reconstruction", "adversarial")]
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    assert read_scores(path) == rows
    path.write_text("\n".join(json.dumps(r) for r in [*rows, rows[0]]), encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate evaluation ID/variant pairs"):
        read_scores(path)
