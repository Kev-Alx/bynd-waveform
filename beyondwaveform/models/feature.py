"""Verified upstream WavLM/HiFi-GAN interface; no kNN voice conversion."""
from __future__ import annotations
import importlib
import json
import sys
from pathlib import Path
import torch
from torch import nn
from ..assets import KNN_REVISION
from ..attacks import freeze
from ..io import digest


class WavLMEncoder(nn.Module):
    def __init__(self, model, layer=6, normalize=False):
        super().__init__()
        self.model = freeze(model)
        self.layer = layer
        self.normalize = normalize

    def forward(self, x):
        if x.ndim != 1:
            raise ValueError("One unpadded utterance required")
        if self.normalize:
            x = torch.nn.functional.layer_norm(x, x.shape)
        return self.model.extract_features(x[None], output_layer=self.layer, ret_layer_results=False)[0][0]


class FeatureVocoder(nn.Module):
    def __init__(self, generator, input_dim=1024):
        super().__init__()
        self.generator = freeze(generator)
        self.input_dim = input_dim

    def forward(self, z):
        if z.ndim != 2 or z.shape[1] != self.input_dim:
            raise ValueError(f"Vocoder expects (T,{self.input_dim}), got {tuple(z.shape)}")
        # Do NOT call upstream matcher.vocode(): it uses inference_mode.
        return self.generator(z[None])[0, 0]


def load_feature_models(config, device):
    root = Path(config.get("artifact_root", "artifacts"))
    source = root / "sources" / f"knn-vc-{KNN_REVISION}"
    if not source.exists():
        raise FileNotFoundError("Run scripts/download_models.py --components feature first")
    if config.get("layer", 6) != 6:
        raise ValueError("The selected upstream vocoder is trained on WavLM layer 6; other layers require matching weights")
    sys.path.insert(0, str(source.resolve()))
    try:
        from wavlm.WavLM import WavLM, WavLMConfig
        from hifigan.models import Generator
        from hifigan.utils import AttrDict
    finally:
        sys.path.pop(0)
    encoder_path = root / "checkpoints" / "WavLM-Large.pt"
    variant = config.get("vocoder_variant", "regular")
    if variant not in {"regular", "prematched"}:
        raise ValueError("Unknown vocoder variant")
    vocoder_path = root / "checkpoints" / ("g_02500000.pt" if variant == "regular" else "prematch_g_02500000.pt")
    checkpoint = torch.load(encoder_path, map_location="cpu", weights_only=True)
    wavlm = WavLM(WavLMConfig(checkpoint["cfg"]))
    wavlm.load_state_dict(checkpoint["model"], strict=True)
    h = AttrDict(json.loads((source / "hifigan" / "config_v1_wavlm.json").read_text()))
    generator = Generator(h)
    generator.load_state_dict(torch.load(vocoder_path, map_location="cpu", weights_only=True)["generator"], strict=True)
    generator.remove_weight_norm()
    encoder = WavLMEncoder(wavlm, layer=6, normalize=config.get("normalize", False)).to(device)
    vocoder = FeatureVocoder(generator, h.hubert_dim).to(device)
    provenance = {"source_revision": KNN_REVISION, "encoder_sha256": digest(encoder_path),
                  "vocoder_sha256": digest(vocoder_path), "vocoder_variant": variant,
                  "layer": 6, "normalize": encoder.normalize, "encoder_config": checkpoint["cfg"],
                  "vocoder_config": dict(h), "output_samples": "320 * feature_frames (native, no duration warping)",
                  "status": "compatible upstream independent choice; author checkpoint unspecified"}
    return encoder, vocoder, provenance

