from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_feature_diagnostics import cases, promotion, diagnostic_config


def rows():
    return [{**case, "error_percent": 75.0, "quality": {"dnsmos": 3.1, "nisqa": 3.8, "utmos": 3.3}}
            for case in cases()]


def test_all_smoothing_widths_have_both_norm_controls():
    grid = cases()
    assert len(grid) == len({r["name"] for r in grid}) == 10
    for window in (3, 5, 9):
        assert {r["mode"] for r in grid if r["window"] == window} == {"smooth", "smooth_norm_matched", "attenuation_matched"}


def test_promotion_does_not_confuse_weaker_attack_with_directional_gain():
    measurements = rows()
    for row in measurements:
        if row["name"] in {"smooth-3", "attenuation_matched-3"}:
            row["quality"]["utmos"] = 3.7
            row["error_percent"] = 60.0
    assert promotion(measurements)["names"] == []


def test_promising_direction_promotes_its_matched_control():
    measurements = rows()
    for row in measurements:
        if row["name"] == "smooth-5":
            row["quality"]["utmos"] = 3.6
    decision = promotion(measurements)
    assert decision["promising_directions"] == ["smooth-5"]
    assert decision["names"] == ["identity", "attenuation_matched-5", "smooth-5"]


def test_norm_restored_direction_compares_against_identity_and_rejects_wer_collapse():
    measurements = rows()
    for row in measurements:
        if row["name"] == "smooth_norm_matched-3":
            row["quality"]["utmos"] = 3.6
            row["error_percent"] = 69.0
    assert promotion(measurements)["names"] == []
    next(r for r in measurements if r["name"] == "smooth_norm_matched-3")["error_percent"] = 70.0
    assert promotion(measurements)["names"] == ["identity", "smooth_norm_matched-3"]


def test_derivation_records_zero_new_steps_and_original_model_contract():
    parent = {"run": "parent", "config": {"features": {"layer": 6}, "surrogate": {"checkpoint": "small"}}}
    config = diagnostic_config({"parents": {"pilot": parent}}, cases()[1], "pilot")
    assert config["new_optimization_steps"] == 0 and config["parent_optimization_steps"] == 50
    assert config["features"] == parent["config"]["features"]
    assert config["surrogate"] == parent["config"]["surrogate"]
