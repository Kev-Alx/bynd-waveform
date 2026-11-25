from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_quality_research import candidates, validate_candidate_files
from run_quality_refinement import candidate_grid, select_validation, qualifies, decision_evidence


def row(name, wer, utmos=3.7, **extras):
    return {"name": name, "error_percent": wer,
            "quality": {"dnsmos": 3.1, "nisqa": 3.8, "utmos": utmos}, **extras}


def test_new_grid_serializes_and_preserves_stated_budget(tmp_path):
    from beyondwaveform.config import load_config
    from beyondwaveform.io import atomic_json
    base = load_config(Path(__file__).resolve().parents[1] / "configs/english_lr005.yaml")
    previous = {"candidates": candidates(base)}
    original = deepcopy(previous)
    grid = candidate_grid(previous, tmp_path / "previous")
    assert previous == original
    assert len(grid) == len({e["name"] for e in grid}) == 17
    assert sum(not e.get("cached") for e in grid) == 14
    assert {e["name"] for e in grid if "validation" in e.get("cached", {})} == {"sum-lr-0.02", "sum-lr-0.1"}
    for entry in grid:
        config = entry["config"]
        assert config["attack"]["rho"] == 0.1
        assert config["attack"]["lambda_perc"] == 1
        assert config["attack"]["steps"] == 50
        assert config["features"] == base["features"]
        assert config["save_delta"]
        assert config["surrogate"]["decoding"] == base["surrogate"]["decoding"]
        path = tmp_path / (entry["name"] + ".json")
        atomic_json(path, config)
        entry["config_path"] = str(path)
    validate_candidate_files(grid)


def test_validation_promotes_quality_strength_and_balanced_only():
    rows = [row("quality", 65), row("strength", 85, utmos=3.0),
            row("balanced", 74, utmos=3.55), row("weak", 45), row("invalid", 100, utmos=float("nan"))]
    assert select_validation(rows) == ["quality", "strength", "balanced"]


@pytest.mark.parametrize("wer,utmos,expected", [(73.43, 3.590, True), (73.42, 3.8, False),
    (80, 3.589, False), (float("nan"), 3.8, False), (80, float("inf"), False)])
def test_confirmation_requires_joint_target(wer, utmos, expected):
    assert qualifies(row("candidate", wer, utmos)) is expected


def test_reauditing_cannot_change_frozen_promotion_evidence():
    before = row("candidate", 76, pool="validation", run="run", summary_sha256="fixed", audit_sha256="audit-1")
    after = {**before, "audit_sha256": "audit-2"}
    assert decision_evidence([before]) == decision_evidence([after])
    after["summary_sha256"] = "changed"
    assert decision_evidence([before]) != decision_evidence([after])
