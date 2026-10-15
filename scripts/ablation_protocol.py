"""Accept only predefined loss removals of the immutable validation-selected config."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from beyondwaveform.config import load_config
from beyondwaveform.experiments import experiment_configs
from beyondwaveform.io import digest


def control_sample_digest(path):
    """Hash exact decoded float32 samples; WAV PEAK timestamps vary on export."""
    import soundfile as sf
    info = sf.info(path)
    if info.subtype != "FLOAT" or info.samplerate != 16000 or info.channels != 1:
        raise ValueError("Control requires mono float32 WAV at 16 kHz")
    samples, _ = sf.read(path, dtype="float32", always_2d=True)
    return hashlib.sha256(samples.astype("<f4", copy=False).tobytes()).hexdigest()


def load_ablation_freeze(base_path, receipt_path):
    base = json.loads(Path(base_path).read_text())
    receipt = json.loads(Path(receipt_path).read_text())
    presets = dict(experiment_configs(base["config"], "ablations"))
    if (receipt["status"] != "frozen" or receipt["scope"] != "predefined_loss_removal" or
            receipt["base_freeze_sha256"] != digest(base_path) or
            receipt["name"] not in set(presets) - {"full"}):
        raise ValueError("Invalid ablation name, scope or base freeze")
    if (receipt["config"] != presets[receipt["name"]] or
            digest(receipt["config_path"]) != receipt["config_sha256"] or
            load_config(receipt["config_path"]) != receipt["config"]):
        raise ValueError("Ablation must change only its predefined loss component")
    if digest(base["config_path"]) != base["config_sha256"] or load_config(base["config_path"]) != base["config"]:
        raise ValueError("Original selected config changed")
    protocol = deepcopy(base)
    protocol.update({k: receipt[k] for k in ["config_path", "config_sha256", "config"]})
    return protocol, receipt
