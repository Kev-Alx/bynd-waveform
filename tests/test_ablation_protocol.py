import json
from pathlib import Path
import sys

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from ablation_protocol import control_sample_digest, load_ablation_freeze
from beyondwaveform.experiments import experiment_configs
from beyondwaveform.io import atomic_json, digest


def frozen_fixture(tmp_path, name="without-attack"):
    config = {"seed": 2026, "method": "feature", "attack": {"rho": 0.5, "lambda_perc": 0.1,
        "learning_rate": 0.05, "steps": 50, "attack": True, "temporal": True, "high_frequency": True}}
    base_config = tmp_path / "selected.yaml"
    base_config.write_text(yaml.safe_dump(config))
    base = tmp_path / "selected.freeze.json"
    atomic_json(base, {"config": config, "config_path": str(base_config), "config_sha256": digest(base_config)})
    changed = dict(experiment_configs(config, "ablations"))[name]
    changed_path = tmp_path / "removal.yaml"
    changed_path.write_text(yaml.safe_dump(changed))
    receipt_path = tmp_path / "removal.freeze.json"
    receipt = {"status": "frozen", "scope": "predefined_loss_removal", "name": name,
        "base_freeze_sha256": digest(base), "config": changed, "config_path": str(changed_path),
        "config_sha256": digest(changed_path)}
    atomic_json(receipt_path, receipt)
    return base, receipt_path, receipt


@pytest.mark.parametrize("name", ["without-attack", "without-perceptual", "without-tv", "without-hf"])
def test_exact_removal_keeps_original_freeze(tmp_path, name):
    base, receipt, _ = frozen_fixture(tmp_path, name)
    before = digest(base)
    protocol, removal = load_ablation_freeze(base, receipt)
    assert protocol["config"] == removal["config"]
    assert digest(base) == before


def test_rejects_extra_change_even_with_new_valid_file_hash(tmp_path):
    base, path, receipt = frozen_fixture(tmp_path)
    receipt["config"]["attack"]["learning_rate"] = 0.1
    config_path = Path(receipt["config_path"])
    config_path.write_text(yaml.safe_dump(receipt["config"]))
    receipt["config_sha256"] = digest(config_path)
    atomic_json(path, receipt)
    with pytest.raises(ValueError, match="only its predefined"):
        load_ablation_freeze(base, path)


def test_rejects_changed_base_freeze(tmp_path):
    base, receipt, _ = frozen_fixture(tmp_path)
    with base.open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="base freeze"):
        load_ablation_freeze(base, receipt)


def test_control_samples_ignore_peak_timestamp_but_detect_audio_changes(tmp_path):
    import numpy as np
    import soundfile as sf
    import struct
    original, changed_metadata, changed_audio = [tmp_path / n for n in ["a.wav", "b.wav", "c.wav"]]
    samples = np.linspace(-0.2, 0.2, 1000, dtype=np.float32)
    sf.write(original, samples, 16000, subtype="FLOAT")
    contents = bytearray(original.read_bytes())
    pos = 12
    while pos + 8 <= len(contents):
        tag, size = struct.unpack_from("<4sI", contents, pos)
        if tag == b"PEAK":
            timestamp = struct.unpack_from("<I", contents, pos + 12)[0]
            struct.pack_into("<I", contents, pos + 12, timestamp + 1)
            break
        pos += 8 + size + size % 2
    else:
        pytest.fail("Expected libsndfile PEAK chunk")
    changed_metadata.write_bytes(contents)
    assert digest(original) != digest(changed_metadata)
    assert control_sample_digest(original) == control_sample_digest(changed_metadata)
    samples[10] += 0.0001
    sf.write(changed_audio, samples, 16000, subtype="FLOAT")
    assert control_sample_digest(original) != control_sample_digest(changed_audio)


def test_control_rejects_different_sample_rate(tmp_path):
    import numpy as np
    import soundfile as sf
    path = tmp_path / "wrong-rate.wav"
    sf.write(path, np.zeros(1000, dtype=np.float32), 8000, subtype="FLOAT")
    with pytest.raises(ValueError, match="16 kHz"):
        control_sample_digest(path)
