"""Trajectory identity and independent weighted Adam reference checks."""
import copy

import pytest
import torch

from beyondwaveform.config import AttackConfig
from beyondwaveform.coordinate_diagnostics import SPEC, coordinate_weights
from scripts.probe_deterministic_execution import ExplicitReflectionReference, trajectory
from scripts.run_coordinate_research import cases, collect, direction_signal, verify_coordinate_receipt
from scripts.run_deterministic_execution import compare_trajectories


def fixture():
    class Vocoder(torch.nn.Module):
        def forward(self, q):
            return (q.flatten().repeat_interleave(4) * .7).tanh()

    class Surrogate(torch.nn.Module):
        def nll(self, audio, text):
            assert text == 'FIXED REFERENCE'
            return (audio + .2).square().mean() * 1.3

    from beyondwaveform.config import SpectralConfig
    generator = torch.Generator().manual_seed(27)
    q = torch.randn(4, 8, dtype=torch.float64, generator=generator)
    x = torch.randn(1360, dtype=torch.float64, generator=generator)
    x[:640] *= .01
    return q, x, Vocoder(), Surrogate(), SpectralConfig(n_fft=8, hop_length=2, win_length=8)


def test_cases_only_add_coordinate_choice_without_changing_paper_config():
    base = {'attack': vars(AttackConfig(learning_rate=.1)), 'surrogate': {'nll_reduction': 'sum'}}
    before = copy.deepcopy(base)
    result = cases(base)
    assert [r['name'] for r in result] == ['time_energy', 'channel_std']
    for row in result:
        config = copy.deepcopy(row['config'])
        spec = config.pop('optimization_coordinates')
        assert spec == {'kind': row['name'], **SPEC} and config == base
    assert base == before


def test_uniform_collector_matches_all_fifty_original_updates_and_final_states(tmp_path):
    q, x, vocoder, surrogate, spectral = fixture()
    attack = AttackConfig(learning_rate=.1)
    old = trajectory(q, x, 'FIXED REFERENCE', vocoder, surrogate, attack, spectral, tmp_path / 'old')
    new = collect(q, x, 'FIXED REFERENCE', vocoder, surrogate, attack, spectral, tmp_path / 'new')
    result = compare_trajectories({'items': [{'id': 'fixed', 'seed': 27, **old}]},
        {'items': [{'id': 'fixed', 'seed': 27, **new}]})
    assert result['passed'] and len(result['items'][0]['updates']) == 50
    verify_coordinate_receipt(new, q, x, 'uniform')


@pytest.mark.parametrize('kind', ['time_energy', 'channel_std'])
def test_weighted_collector_matches_independent_full_objective_adam_in_theta(tmp_path, kind):
    q, x, vocoder, surrogate, spectral = fixture()
    attack = AttackConfig(learning_rate=.1)
    result = collect(q, x, 'FIXED REFERENCE', vocoder, surrogate, attack, spectral, tmp_path / kind, kind)
    weights = coordinate_weights(q, x, kind)
    theta = torch.zeros_like(q, requires_grad=True)
    optimizer = torch.optim.Adam([theta], lr=.1)
    reference = ExplicitReflectionReference(q, x, spectral, 1e-8)
    for row in result['trace']:
        optimizer.zero_grad(set_to_none=True)
        z = q + theta * weights
        wave = vocoder(z)
        nll = surrogate.nll(wave, 'FIXED REFERENCE')
        tv, hf = reference.terms(z, wave)
        loss = -nll + (tv + hf)
        gradient, = torch.autograd.grad(loss, theta)
        theta.grad = gradient
        optimizer.step()
        # Independent physical-space projection, mapped back by the same scalar.
        with torch.no_grad():
            delta = theta * weights
            limit = .1 * (q.norm() + 1e-8)
            if delta.norm() > limit:
                theta.mul_(limit / delta.norm())
        assert torch.equal(theta, torch.load(row['theta']['path'], weights_only=True))
        assert torch.equal(theta * weights, torch.load(row['delta']['path'], weights_only=True))
    verify_coordinate_receipt(result, q, x, kind)
    assert not q.requires_grad and q.grad is None and x.grad is None


@pytest.mark.parametrize('kind', ['time_energy', 'channel_std'])
def test_coordinate_gradient_obeys_chain_rule_and_changes_physical_adam_step(kind):
    q, x, *_ = fixture()
    weights = coordinate_weights(q, x, kind)
    theta = torch.zeros_like(q, requires_grad=True)
    physical = torch.zeros_like(q, requires_grad=True)
    gradient, = torch.autograd.grad((q + theta * weights).square().sum(), theta)
    direct, = torch.autograd.grad((q + physical).square().sum(), physical)
    assert torch.equal(gradient, direct * weights)
    # Adam cancels constant gradient scaling; scaling the physical parameterization is necessary.
    opt = torch.optim.Adam([theta], lr=.1)
    uniform = torch.optim.Adam([physical], lr=.1)
    theta.grad = gradient; physical.grad = direct
    opt.step(); uniform.step()
    assert not torch.allclose(theta * weights, physical, atol=1e-6, rtol=1e-6)


def test_coordinate_audit_rejects_theta_weight_and_loss_tampering(tmp_path):
    from scripts.probe_deterministic_execution import save_tensor
    q, x, vocoder, surrogate, spectral = fixture()
    result = collect(q, x, 'FIXED REFERENCE', vocoder, surrogate, AttackConfig(learning_rate=.1),
        spectral, tmp_path / 'result', 'channel_std')
    altered = copy.deepcopy(result)
    theta = torch.load(altered['trace'][7]['theta']['path'], weights_only=True)
    altered['trace'][7]['theta'] = save_tensor(theta + .001, tmp_path / 'wrong-theta.pt')
    with pytest.raises(ValueError, match='physical delta does not equal'):
        verify_coordinate_receipt(altered, q, x, 'channel_std')
    altered = copy.deepcopy(result)
    altered['weights'] = save_tensor(torch.ones_like(coordinate_weights(q, x, 'channel_std')), tmp_path / 'wrong-weights.pt')
    with pytest.raises(ValueError, match='Changed clean coordinate weights'):
        verify_coordinate_receipt(altered, q, x, 'channel_std')
    altered = copy.deepcopy(result); altered['trace'][0]['loss_values'][0] += 1
    with pytest.raises(ValueError, match='Changed loss/gradient-coordinate'):
        verify_coordinate_receipt(altered, q, x, 'channel_std')


def test_only_full_loss_fifty_update_protocol_is_allowed(tmp_path):
    q, x, vocoder, surrogate, spectral = fixture()
    for attack in (AttackConfig(steps=49), AttackConfig(optimizer='sgd'), AttackConfig(temporal=False)):
        with pytest.raises(ValueError, match='fixed fifty-update full-objective'):
            collect(q, x, 'FIXED REFERENCE', vocoder, surrogate, attack, spectral, tmp_path / 'invalid')
    assert not (tmp_path / 'invalid').exists()


def test_direction_signal_is_not_a_relaxation_of_published_joint_targets():
    from scripts.run_stable_quality_research import joint_target
    baseline = {'error_percent': 74.7368, 'quality': {'dnsmos': 3.164, 'nisqa': 4.035, 'utmos': 3.3186}}
    row = {'error_percent': 72., 'quality': {'dnsmos': 3.1, 'nisqa': 4., 'utmos': 3.4}}
    assert direction_signal(row, baseline) and not joint_target(row)
    row['error_percent'] = 65.
    assert not direction_signal(row, baseline)


def test_coordinate_choice_is_bound_into_generation_fingerprint():
    from beyondwaveform.pipeline import config_fingerprint
    from scripts.run_stable_quality_research import identity
    plan = {'pilot': {'sha256': 'fixed', 'ids': ['clip']}, 'execution': {'mode': 'strict_math'}}
    rows = cases({'attack': vars(AttackConfig(learning_rate=.1))})
    assert config_fingerprint(identity(plan, rows[0])) != config_fingerprint(identity(plan, rows[1]))


def test_check_rejects_frozen_source_or_experiment_definition_changes(tmp_path, monkeypatch):
    from beyondwaveform.config import SpectralConfig
    from beyondwaveform.io import digest, write_jsonl
    from scripts import run_coordinate_research as study
    manifest = tmp_path / 'manifest.jsonl'
    ids = [str(i) for i in range(24)]; write_jsonl(manifest, [{'id': uid} for uid in ids])
    frozen = tmp_path / 'source.py'; frozen.write_text('original')
    base = {'attack': vars(AttackConfig(learning_rate=.1)), 'spectral': vars(SpectralConfig()),
        'surrogate': {'nll_reduction': 'sum'}}
    plan = {'runtime': {}, 'additional_dependencies': {}, 'base_config': base, 'cases': cases(base),
        'coordinate_spec': SPEC, 'descriptive_direction_signal': study.SIGNAL, 'targets': study.stable.TARGETS,
        'scope': study.SCOPE, 'discord_enabled': False, 'control_preflight_ids': ids[:3],
        'execution': {'mode': 'strict_math', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8', 'explicit_reflection': True},
        'pilot': {'ids': ids, 'manifest': str(manifest), 'sha256': digest(manifest)},
        'frozen_files': {str(frozen): digest(frozen)}}
    monkeypatch.setattr(study.protocol, 'runtime', lambda: {})
    study.check(plan)
    changed = copy.deepcopy(plan)
    changed['cases'][0]['config']['optimization_coordinates']['floor_relative_to_max'] = .2
    with pytest.raises(ValueError, match='Changed coordinate experiment definitions'):
        study.check(changed)
    frozen.write_text('changed')
    with pytest.raises(ValueError, match='Changed frozen coordinate input'):
        study.check(plan)
