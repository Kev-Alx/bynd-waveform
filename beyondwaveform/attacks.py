"""Per-example attacks. Generation has no target/defense model dependency."""
from __future__ import annotations
from dataclasses import dataclass
import time
import torch
from .config import AttackConfig, SpectralConfig
from .losses import CleanReference


def freeze(module):
    module.eval()
    module.requires_grad_(False)
    return module


def relative_radius(delta, q, eps=1e-8):
    return torch.linalg.vector_norm(delta) / (torch.linalg.vector_norm(q.detach()) + eps)


@torch.no_grad()
def project_relative_ball_(delta, q, rho, eps=1e-8):
    """Euclidean projection onto Eq.6, not elementwise clipping."""
    radius = rho * (torch.linalg.vector_norm(q.detach()) + eps)
    norm = torch.linalg.vector_norm(delta)
    if norm > radius:
        delta.mul_(radius / norm)
    return delta


@dataclass
class AttackResult:
    audio: torch.Tensor
    reconstruction: torch.Tensor
    delta: torch.Tensor
    trace: list[dict]
    seconds: float


def feature_attack(x, transcript, encoder, vocoder, surrogate,
                   config=AttackConfig(), spectral=SpectralConfig()):
    for module in (encoder, vocoder, surrogate):
        freeze(module)
    start = time.perf_counter()
    with torch.no_grad():
        q = encoder(x).detach()
        reconstruction = vocoder(q).detach()
    if q.ndim != 2 or q.shape[0] < 2 or not torch.isfinite(q).all():
        raise ValueError("Encoder must produce finite (T,D), T>=2 features")
    reference = CleanReference(q, x, spectral, config.eps)
    delta = torch.zeros_like(q)
    if config.initialization == "normal" and config.rho > 0:
        delta.normal_()
        delta *= config.init_fraction * config.rho * (q.norm() + config.eps) / delta.norm()
    delta.requires_grad_()
    optimizer_class = torch.optim.Adam if config.optimizer == "adam" else torch.optim.SGD
    optimizer = optimizer_class([delta], lr=config.learning_rate)
    trace = []
    for step in range(config.steps):
        optimizer.zero_grad(set_to_none=True)
        z = q + delta
        audio = vocoder(z)  # frozen weights, live input gradients
        nll = surrogate.nll(audio, transcript) if config.attack else z.new_zeros(())
        tv, hf = reference.terms(z, audio,
                                 temporal=config.temporal and config.lambda_perc > 0,
                                 high_frequency=config.high_frequency and config.lambda_perc > 0 and config.alpha_hf > 0)
        loss = -nll + config.lambda_perc * (tv + config.alpha_hf * hf)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite objective at step {step}")
        if loss.requires_grad:
            gradient, = torch.autograd.grad(loss, delta)
        else:
            gradient = torch.zeros_like(delta)
        if not torch.isfinite(gradient).all():
            raise FloatingPointError(f"Nonfinite gradient at step {step}")
        delta.grad = gradient
        optimizer.step()
        project_relative_ball_(delta, q, config.rho, config.eps)
        radius = relative_radius(delta, q, config.eps).item()
        if radius > config.rho + 1e-6:
            raise RuntimeError("Feature radius violated")
        trace.append({"step": step + 1, "loss_before_update": loss.item(), "nll_before_update": nll.item(),
                      "tv_ratio": tv.item(), "hf_ratio": hf.item(), "gradient_norm": gradient.norm().item(),
                      "radius_after_update": radius})
    with torch.no_grad():
        final = vocoder(q + delta).detach()
    if x.is_cuda:
        torch.cuda.synchronize(x.device)
    return AttackResult(final, reconstruction, delta.detach(), trace, time.perf_counter() - start)


def waveform_attack(x, transcript, surrogate, method="pgd", steps=50, epsilon=0.02,
                    step_size=0.0008, momentum=1.0, variance_samples=5, beta=1.5,
                    random_start=False):
    """Independent L-infinity PGD/MI-FGSM/VMI-FGSM adaptations.

    Selected Table 2 defaults; paper does not specify the waveform norm or
    full baseline objectives. All maximize the same lexical NLL here.
    VMI uses the previous iterate's variance correction (Wang & He, Alg.1).
    """
    if method not in {"pgd", "mi-fgsm", "vmi-fgsm"}:
        raise ValueError(f"Unknown waveform attack: {method}")
    if steps < 0 or epsilon < 0 or step_size <= 0 or variance_samples < 1:
        raise ValueError("Invalid waveform attack parameters")
    freeze(surrogate)
    current = x.detach().clone()
    if random_start:
        current = (current + torch.empty_like(x).uniform_(-epsilon, epsilon)).clamp(-1, 1)
    velocity, variance = torch.zeros_like(x), torch.zeros_like(x)
    trace = []

    def gradient_at(wave):
        wave = wave.detach().requires_grad_()
        nll = surrogate.nll(wave, transcript)
        grad, = torch.autograd.grad(nll, wave)
        if not torch.isfinite(grad).all():
            raise FloatingPointError("Nonfinite baseline gradient")
        return grad, nll.item()

    for i in range(steps):
        gradient, nll = gradient_at(current)
        direction = gradient
        if method != "pgd":
            corrected = gradient + variance if method == "vmi-fgsm" else gradient
            velocity = momentum * velocity + corrected / corrected.abs().mean().clamp_min(1e-12)
            direction = velocity
        if method == "vmi-fgsm":
            neighbors = [gradient_at(current + torch.empty_like(x).uniform_(-beta * epsilon, beta * epsilon))[0]
                         for _ in range(variance_samples)]
            variance = torch.stack(neighbors).mean(0) - gradient
        current = current.detach() + step_size * direction.sign()
        current = (x + (current - x).clamp(-epsilon, epsilon)).clamp(-1, 1).detach()
        trace.append({"step": i + 1, "nll_before_update": nll, "linf": (current - x).abs().max().item()})
    return current, trace
