"""Fixed clean-energy waveform constraints; no model loading or score selection."""
from dataclasses import dataclass
import numpy as np
import torch
from r3_waveform import multiresolution_stft_loss, relative_feature_loss
from waveform_pgd import (PGDConfig, PGDResult, sequential_objective_gradients,
                          waveform_bounds, _checked_scalar)


@dataclass(frozen=True)
class EnvelopeConfig:
    frame: int = 400
    hop: int = 160
    local_snr_db: float = 30.
    epsilon: float = .003
    updates: int = 50

    def __post_init__(self):
        if (self.frame, self.hop, self.local_snr_db, self.epsilon, self.updates) != (400, 160, 30., .003, 50):
            raise ValueError('Use the single frozen envelope configuration')


def energy_envelope(original, config=None):
    """CPU float64 construction; every overlapping frame supplies a box cap.

    For each 25ms frame (10ms hop, plus an end-aligned frame), cap each
    covered sample at min(.003, frame_RMS * 10**(-30/20)). Taking the minimum
    across covering frames guarantees frame perturbation RMS <= clean RMS /
    sqrt(1000), before waveform-storage roundoff. This is not auditory masking.
    """
    cfg = config or EnvelopeConfig()
    values = original.detach().cpu().numpy().astype(np.float64)
    if values.ndim != 1 or len(values) <= 512 or not np.isfinite(values).all() or np.abs(values).max() > 1:
        raise ValueError('Finite native mono waveform longer than512 samples required')
    starts = list(range(0, len(values) - cfg.frame + 1, cfg.hop))
    if starts[-1] != len(values) - cfg.frame:
        starts.append(len(values) - cfg.frame)
    bounds = np.full(len(values), cfg.epsilon, dtype=np.float64)
    frame_caps = []
    for start in starts:
        rms = float(np.sqrt(np.mean(values[start:start + cfg.frame] ** 2)))
        cap = min(cfg.epsilon, rms * 10 ** (-cfg.local_snr_db / 20))
        bounds[start:start + cfg.frame] = np.minimum(bounds[start:start + cfg.frame], cap)
        frame_caps.append(cap)
    return torch.as_tensor(bounds, device=original.device, dtype=original.dtype), {
        'frame_samples': cfg.frame, 'hop_samples': cfg.hop, 'local_snr_db': cfg.local_snr_db,
        'starts': starts, 'frame_caps': frame_caps, 'minimum_bound': float(bounds.min()),
        'median_bound': float(np.median(bounds)), 'maximum_bound': float(bounds.max()),
        'zero_bound_samples': int((bounds == 0).sum()), 'rule': 'minimum of all overlapping clean-RMS frame caps'}


def project_envelope(original, residual, bounds):
    if residual.shape != original.shape or bounds.shape != original.shape or not torch.isfinite(residual).all():
        raise ValueError('Invalid envelope projection input')
    residual = torch.maximum(torch.minimum(residual, bounds), -bounds)
    return (original + residual).clamp(-1., 1.) - original


class EnvelopePGD:
    def __init__(self, source_nll, encoder, *, condition='nll_only'):
        if condition not in ('full_r3', 'nll_only'):
            raise ValueError('Unknown fixed objective')
        self.source_nll, self.encoder, self.condition = source_nll, encoder, condition
        self.config = PGDConfig()

    def run(self, original, text, *, clean_features, on_step=None):
        if not isinstance(text, str) or not text.strip():
            raise ValueError('Unchanged nonempty reference required')
        original = original.detach()
        q = clean_features.detach()
        if not torch.isfinite(q).all():
            raise ValueError('Nonfinite fixed content reference')
        bounds, envelope = energy_envelope(original)
        residual = torch.zeros_like(original)
        counts = {f'{term}_{op}_calls': 0 for term in ('source', 'stft', 'content') for op in ('forward', 'backward')}
        counts.update(clean_reference_forward_calls=0, completed_updates=0,
            final_source_forward_calls=0, final_stft_forward_calls=0, final_content_forward_calls=0)
        trace = []
        for step in range(50):
            gradient, row = sequential_objective_gradients(original + residual,
                lambda x: self.source_nll(x, text), lambda x: multiresolution_stft_loss(x, original),
                lambda x: relative_feature_loss(self.encoder(x), q), condition=self.condition,
                config=self.config, counts=counts)
            with torch.no_grad():
                residual = project_envelope(original, residual - (2 * bounds / 50) * gradient.sign(), bounds).detach()
            measured = waveform_bounds(original, residual)
            excess = float((residual.abs() - bounds).max())
            if excess > 4 * torch.finfo(original.dtype).eps:
                raise ValueError('Measured envelope projection violation')
            row.update(step=step + 1, loss_evaluation_iterate=step, objective_condition=self.condition,
                envelope_maximum_excess=excess, update_direction='subtract (2*clean_envelope/50)*sign(objective gradient)', **measured)
            counts['completed_updates'] += 1
            trace.append(row)
            if on_step: on_step(dict(row))
        adversarial = (original + residual).detach()
        with torch.no_grad():
            counts['final_source_forward_calls'] += 1
            nll = _checked_scalar(self.source_nll(adversarial, text), 'final source')
            counts['final_stft_forward_calls'] += 1
            stft = _checked_scalar(multiresolution_stft_loss(adversarial, original), 'final stft')
            counts['final_content_forward_calls'] += 1
            content = _checked_scalar(relative_feature_loss(self.encoder(adversarial), q), 'final content')
        objective = -nll + self.config.stft_weight * stft + self.config.content_weight * content
        final = {'source_sum_nll': nll, 'stft_loss': stft, 'content_loss': content,
            'full_r3_objective': objective, 'optimized_objective': objective if self.condition == 'full_r3' else -nll,
            'objective_condition': self.condition, 'selected_iterate': 50,
            'selection': 'Final50 iterate; no decoding or MOS selection', 'energy_envelope': envelope,
            'envelope_maximum_excess': float((residual.abs() - bounds).max()),
            **waveform_bounds(original, residual)}
        return PGDResult(adversarial, residual, trace, final, counts)


def attenuate_to_rms(original, global_adversarial, candidate):
    """Same final residual RMS, uniform attenuation only; no decoded feedback."""
    residual = global_adversarial.double() - original.double()
    target = (candidate.double() - original.double()).square().mean().sqrt()
    available = residual.square().mean().sqrt()
    if not torch.isfinite(target) or not torch.isfinite(available) or target > available + 1e-12:
        raise ValueError('RMS attenuation control cannot silently amplify its reference')
    scale = float(target / available) if available > 0 else 0.
    output = (original.double() + scale * residual).to(original.dtype)
    return output, {'uniform_attenuation': scale, 'target_rms': float(target),
        'reference_rms': float(available), 'actual_rms': float((output.double() - original.double()).square().mean().sqrt()),
        'source': 'frozen global NLL50 output', 'selection_uses_decode_or_quality': False}
