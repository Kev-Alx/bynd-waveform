from pathlib import Path
import sys

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from verify_waveform_run import waveform_bound
from run_waveform_validation import select_method


def pair(tmp_path, changed):
    clean = np.linspace(-0.2, 0.2, 1000, dtype=np.float32)
    a, b = tmp_path / "clean.wav", tmp_path / "perturbed.wav"
    sf.write(a, clean, 16000, subtype="FLOAT")
    sf.write(b, changed(clean), 16000, subtype="FLOAT")
    return a, b


def test_bound_checks_exported_samples(tmp_path):
    a, b = pair(tmp_path, lambda x: x + 0.005)
    assert waveform_bound(a, b, 0.005) == pytest.approx(0.005, abs=1e-7)


def test_bound_rejects_out_of_ball_audio_even_if_trace_could_be_valid(tmp_path):
    a, b = pair(tmp_path, lambda x: x + 0.015)
    with pytest.raises(ValueError, match="L-infinity"):
        waveform_bound(a, b, 0.005)


def test_rejects_duration_change(tmp_path):
    a, b = pair(tmp_path, lambda x: x[:-1])
    with pytest.raises(ValueError, match="length"):
        waveform_bound(a, b, 0.02)


def test_rejects_out_of_range_even_with_large_ball(tmp_path):
    a, b = pair(tmp_path, lambda x: x + 1.1)
    with pytest.raises(ValueError, match="range"):
        waveform_bound(a, b, 2)


def selection_candidate(tmp_path, role="validation"):
    import json
    path = tmp_path / "summary.json"
    path.write_text(json.dumps({"evaluation_scope": "surrogate", "roles": [role], "manifest_sha256": "fixture",
        "evaluation_ids": ["fixture"], "target_artifacts": {}, "quality_provenance": {}, "metric": "WER",
        "normalization": "fixture", "generation_run": "fixture", "generation_fingerprint": "fixture",
        "summary": {"adversarial": {"error_percent": 80, "quality": {"dnsmos": 2.5, "nisqa": 3, "utmos": 3}}}}))
    return [{"summary_path": str(path)}]


def test_no_eligible_setting_is_reported_without_fallback(tmp_path):
    rows = selection_candidate(tmp_path)
    result = select_method(tmp_path, {}, "pgd", rows)
    assert result["status"] == "no_eligible_configuration" and result["selected"] is None
    assert not (tmp_path / "selected/pgd.yaml").exists()


def test_old_no_eligible_audit_cannot_mask_test_data_rejection(tmp_path):
    import json
    folder = tmp_path / "selected"
    folder.mkdir()
    (folder / "pgd.selection.json").write_text(json.dumps({"selected": None}))
    with pytest.raises(ValueError, match="validation"):
        select_method(tmp_path, {}, "pgd", selection_candidate(tmp_path, role="test"))
