"""Check the pinned real HiFi-GAN architecture using random weights (not MOS).

This runs without large pretrained checkpoints and does not claim speech quality.
"""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from beyondwaveform.assets import KNN_REVISION
from beyondwaveform.io import atomic_json
from beyondwaveform.models.feature import FeatureVocoder

torch.set_num_threads(2)
root = Path("artifacts/sources") / f"knn-vc-{KNN_REVISION}"
sys.path.insert(0, str(root.resolve()))
from hifigan.models import Generator
from hifigan.utils import AttrDict

config = AttrDict(json.loads((root / "hifigan/config_v1_wavlm.json").read_text()))
generator = Generator(config).eval()
state_keys = sorted(generator.state_dict())
generator.remove_weight_norm()
vocoder = FeatureVocoder(generator)
measurements = []
for frames in [3, 7, 11]:
    features = torch.randn(frames, 1024, requires_grad=True)
    waveform = vocoder(features)
    assert waveform.shape == (frames * 320,)
    grad, = torch.autograd.grad(waveform.square().mean(), features)
    assert torch.isfinite(grad).all() and grad.norm() > 0
    assert all(parameter.grad is None for parameter in generator.parameters())
    measurements.append({"frames": frames, "samples": waveform.numel(), "gradient_norm": grad.norm().item()})
result = {"status": "passed", "checkpoint": "random initialization, architecture test only",
          "source_revision": KNN_REVISION, "parameters": sum(p.numel() for p in generator.parameters()),
          "config": dict(config), "state_dict_keys_before_weightnorm_removal": state_keys,
          "measurements": measurements}
atomic_json("verification/architecture.json", result)
print(json.dumps({key: result[key] for key in ["status", "parameters", "measurements"]}, indent=2))

