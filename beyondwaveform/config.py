from dataclasses import dataclass
from pathlib import Path
import yaml


@dataclass(frozen=True)
class SpectralConfig:
    sample_rate: int = 16000
    cutoff_hz: float = 6000.0
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    center: bool = True

    def __post_init__(self):
        if not 0 <= self.cutoff_hz < self.sample_rate / 2:
            raise ValueError("HF cutoff must be below Nyquist")
        if not 0 < self.win_length <= self.n_fft or self.hop_length <= 0:
            raise ValueError("Invalid STFT configuration")


@dataclass(frozen=True)
class AttackConfig:
    steps: int = 50
    rho: float = 0.1
    lambda_perc: float = 1.0
    alpha_hf: float = 1.0
    eps: float = 1e-8
    optimizer: str = "adam"
    learning_rate: float = 0.01
    initialization: str = "zero"
    init_fraction: float = 0.01
    attack: bool = True
    temporal: bool = True
    high_frequency: bool = True

    def __post_init__(self):
        if self.steps < 0 or self.rho < 0 or self.eps <= 0 or self.learning_rate <= 0:
            raise ValueError("Invalid attack steps, radius, epsilon, or learning rate")
        if self.lambda_perc < 0 or self.alpha_hf < 0 or not 0 <= self.init_fraction <= 1:
            raise ValueError("Invalid loss weight or initialization fraction")
        if self.optimizer not in {"adam", "sgd"} or self.initialization not in {"zero", "normal"}:
            raise ValueError("Unsupported optimizer/initialization")


def load_config(path):
    config = yaml.safe_load(Path(path).read_text())
    if not isinstance(config, dict):
        raise ValueError("Expected a YAML mapping")
    return config

