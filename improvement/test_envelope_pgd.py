"""Analytic bounds, real loss gradients, tail coverage and control integrity."""
import numpy as np
import pytest
import torch
from envelope_pgd import EnvelopeConfig, EnvelopePGD, energy_envelope, project_envelope, attenuate_to_rms


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_all_registered_frames_obey_energy_budget_including_tail():
    rng = torch.Generator().manual_seed(71)
    x = torch.randn(2273, dtype=torch.float64, generator=rng) * torch.linspace(.001, .2, 2273)
    bounds, meta = energy_envelope(x)
    assert meta['starts'][-1] == len(x) - 400
    assert len(meta['starts']) == len(set(meta['starts']))
    delta = project_envelope(x, torch.randn(2273, dtype=x.dtype, generator=rng), bounds)
    assert torch.all(delta.abs() <= bounds + 1e-16)
    for start in meta['starts']:
        assert delta[start:start + 400].square().sum() <= x[start:start + 400].square().sum() / 1000 + 1e-16


def test_zero_energy_frames_cannot_be_perturbed():
    x = torch.zeros(960, dtype=torch.float64); x[600:] = .1
    bounds, meta = energy_envelope(x)
    assert torch.equal(bounds[:560], torch.zeros_like(bounds[:560]))
    output = x + project_envelope(x, torch.ones_like(x), bounds)
    assert torch.equal(output[:560], x[:560])
    assert meta['zero_bound_samples'] >= 560


def test_bound_is_minimum_of_all_covering_frames():
    x = torch.ones(800, dtype=torch.float64) * .1; x[:400] = .001
    bounds, meta = energy_envelope(x)
    for index in range(len(x)):
        expected = min(cap for start, cap in zip(meta['starts'], meta['frame_caps']) if start <= index < start + 400)
        assert bounds[index].item() == expected


@pytest.mark.parametrize('condition', ['nll_only', 'full_r3'])
def test_real_fifty_step_gradients_keep_duration_caps_and_actual_counters(condition):
    x = torch.sin(torch.arange(640, dtype=torch.float64) * .09) * .03
    encoder = lambda a: a[::16][None]
    source = lambda a, text: (a * torch.cos(torch.arange(len(a), dtype=a.dtype) * .07)).sum()
    run = EnvelopePGD(source, encoder, condition=condition).run(x, 'ORIGINAL WORDS', clean_features=encoder(x))
    bounds, _ = energy_envelope(x)
    assert run.adversarial.shape == x.shape
    assert torch.all(run.residual.abs() <= bounds + 1e-16)
    assert len(run.trace) == run.counts['completed_updates'] == 50
    assert run.final_metrics['selected_iterate'] == 50
    for term in ('source', 'stft', 'content'):
        assert run.counts[term + '_forward_calls'] == 50
        assert run.counts['final_' + term + '_forward_calls'] == 1
        assert run.counts[term + '_backward_calls'] == (50 if term == 'source' or condition == 'full_r3' else 0)


def test_rms_control_matches_energy_with_one_uniform_attenuation():
    x = torch.sin(torch.arange(900, dtype=torch.float64) * .1) * .2
    global_audio = x + torch.cos(torch.arange(900, dtype=torch.float64)) * .002
    candidate = x + torch.sin(torch.arange(900, dtype=torch.float64)) * .0003
    output, receipt = attenuate_to_rms(x, global_audio, candidate)
    assert torch.allclose(output - x, receipt['uniform_attenuation'] * (global_audio - x), atol=1e-16)
    assert (output - x).square().mean() == pytest.approx((candidate - x).square().mean(), abs=1e-18)
    assert not receipt['selection_uses_decode_or_quality']


def test_rms_control_refuses_amplification():
    x = torch.ones(640) * .1
    with pytest.raises(ValueError, match='amplify'):
        attenuate_to_rms(x, x + .0001, x + .001)


@pytest.mark.parametrize('field,value', [('frame', 401), ('local_snr_db', 25.), ('updates', 100)])
def test_unregistered_envelope_configuration_rejected(field, value):
    with pytest.raises(ValueError): EnvelopeConfig(**{field: value})


def test_selection_never_promotes_high_wer_with_failed_quality_floor():
    from postprocess_envelope import decision
    def item(wer, nisqa, utmos):
        return {'error_percent': wer, 'quality': {'dnsmos': 3.2, 'nisqa': nisqa, 'utmos': utmos}}
    s = {'fv50': item(49.8, 4., 3.58), 'global_full50': item(43.46, 3.78, 3.89),
         'attenuated_global_nll50': item(39., 4., 3.9), 'envelope_nll50': item(65., 3.6, 3.9)}
    assert not decision(s, [])['earns_larger_validation']
    s['envelope_nll50'] = item(43.46, 3.90, 3.90)
    result = decision(s, [])
    assert not result['primary_source_quality_screen']
    assert result['exploratory_shape_signal'] and result['earns_larger_validation']
    s['attenuated_global_nll50']['error_percent'] = 42.
    assert not decision(s, [])['earns_larger_validation']
    s['envelope_full50'] = item(100., 4., 4.)
    assert not decision(s, [])['earns_larger_validation']


def test_recorded_execution_failure_prevents_promotion():
    from postprocess_envelope import decision
    item = lambda wer, ut: {'error_percent': wer, 'quality': {'dnsmos': 3.2, 'nisqa': 4., 'utmos': ut}}
    s = {'fv50': item(49., 3.6), 'global_full50': item(43., 3.9),
         'attenuated_global_nll50': item(40., 3.9), 'envelope_nll50': item(55., 3.9)}
    assert decision(s, [])['primary_source_quality_screen']
    assert not decision(s, [{'id': 'failed'}])['earns_larger_validation']
