"""Two clean-statistic Adam coordinate experiments, with the full paper loss.

The fixed positive map delta=W*theta changes finite-step optimization only.
Keep the matched encoder/vocoder, original physical ball and native scoring.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from beyondwaveform.attacks import relative_radius
from beyondwaveform.config import AttackConfig, SpectralConfig
from beyondwaveform.coordinate_diagnostics import KINDS, SPEC, coordinate_weights, physical_delta, project_coordinates_
from beyondwaveform.io import atomic_json, digest, load_audio, read_jsonl, save_audio, seed_everything, write_jsonl
from beyondwaveform.pipeline import config_fingerprint
from parallel_jobs import Supervisor, now
from probe_deterministic_execution import ExplicitReflectionReference, save_tensor, tensor_tree
from probe_numerical_repeatability import backend_manifest, backend_mode, tensor_value_sha256
import run_deterministic_execution as numerical
import run_feature_layer_research_v2 as layers
import run_quality_research as protocol
import run_quality_refinement as stages
import run_stable_ablation_research as prior
import run_stable_quality_research as stable

DEFAULT = ROOT / 'reproduction-runs/paper-quality-coordinate-research'
REVIEW = ROOT / 'reproduction-runs/feature-layer-completion-review-20261020'
NAMES = KINDS
ENABLED = {'asr': True, 'tv': True, 'hf': True}
SIGNAL = {'min_UTMOS_gain': .05, 'max_WER_loss_pp': 5., 'min_each_MOS_exclusive': 2.5}
SCOPE = ('Two predefined clean-statistic optimizer coordinates on the fixed 24 development clips. '
    'Matched raw WavLM layer-6/regular HiFi-GAN, original sum NLL/full perceptual loss, '
    'rho=.1, lambda=1, alpha=1, fc=6000, zero initialization, Adam LR=.1, fifty updates. '
    'Positive weights change Adam coordinates, not the physical feasible set or loss. '
    'Native references/decoding/scorers and strict explicit-reflection differentiation retained. '
    'The pilot is repeatedly used development data. No sample selection, weight grid, automatic '
    'fresh48/999 evaluation or claim of author-recipe recovery or human sound quality.')


def cases(base):
    result = []
    for kind in KINDS:
        config = copy.deepcopy(base)
        config['optimization_coordinates'] = {'kind': kind, **SPEC}
        result.append({'name': kind, 'config': config})
    return result


def collect(q, x, text, vocoder, surrogate, attack, spectral, directory, kind='uniform'):
    protocol.require(attack.steps == 50 and attack.optimizer == 'adam' and attack.lr_schedule == 'constant'
        and attack.initialization == 'zero' and prior.active_terms(attack) == ENABLED,
        'Only fixed fifty-update full-objective Adam is supported')
    q, x = q.detach(), x.detach()
    weights = coordinate_weights(q, x, kind).to(q.device)
    identity = kind == 'uniform'
    theta = torch.zeros_like(q, requires_grad=True)
    reference = ExplicitReflectionReference(q, x, spectral, attack.eps)
    optimizer = torch.optim.Adam([theta], lr=attack.learning_rate)
    weight_asset = save_tensor(weights, directory / 'weights.pt')
    trace = []
    for step in range(1, 51):
        optimizer.zero_grad(set_to_none=True)
        delta = physical_delta(theta, weights, identity)
        loss, values = prior.ablation_objective(q, delta, text, vocoder, surrogate, attack, reference)
        gradient, = torch.autograd.grad(loss, theta)
        protocol.require(torch.isfinite(gradient).all().item(), 'Nonfinite coordinate gradient')
        theta.grad = gradient
        optimizer.step()
        delta = project_coordinates_(theta, weights, q, attack.rho, attack.eps, identity)
        radius = relative_radius(delta, q, attack.eps).item()
        protocol.require(torch.isfinite(delta).all().item() and radius <= attack.rho + 1e-6,
            'Invalid projected physical iterate')
        trace.append({'step': step, 'loss_values': values.tolist(),
            'loss_value_sha256': tensor_value_sha256(values),
            'gradient_value_sha256': tensor_value_sha256(gradient), 'gradient_coordinate': 'theta',
            'radius': radius, 'optimizer_value_receipt': tensor_tree(optimizer.state_dict()),
            'theta': save_tensor(theta, directory / f'theta-{step:02d}.pt'),
            'delta': save_tensor(delta, directory / f'delta-{step:02d}.pt')})
        if step in (1, 10, 25, 50):
            print(f'Coordinate trajectory {directory.name}: {step}/50, physical radius={radius:.8f}', flush=True)
    with torch.no_grad():
        audio = vocoder(q + physical_delta(theta, weights, identity))
    state_path = directory / 'final-optimizer.pt'
    torch.save(optimizer.state_dict(), state_path)
    protocol.require(not any(p.grad is not None for m in (vocoder, surrogate) for p in m.parameters()),
        'Accumulated model gradients')
    return {'trace': trace, 'actual_optimization_updates': 50, 'active_terms': ENABLED,
        'coordinate_kind': kind, 'coordinate_spec': SPEC, 'weights': weight_asset,
        'final_audio': save_tensor(audio, directory / 'final-audio.pt'),
        'final_optimizer': {'path': str(state_path.resolve()), 'file_sha256': digest(state_path),
            'value_receipt': tensor_tree(optimizer.state_dict())}}


def prepare(output):
    protocol.require(not (output / 'plan.json').exists(), 'Plan already exists')
    parent = protocol.read(layers.DEFAULT / 'plan.json')
    layers.check(parent)
    audit = protocol.read(REVIEW / 'review.json')
    protocol.require(protocol.read(layers.DEFAULT / 'job.json')['status'] == 'complete'
        and audit['all_checks_passed'] and audit['source_plan_sha256'] == digest(layers.DEFAULT / 'plan.json')
        and audit['frozen_inputs_verified'] == len(parent['frozen_files'])
        and audit['feature_tensors_verified'] == 48 and audit['score_WAV_records_verified'] == 96
        and audit['layer6_exact_controls'] == 24 and not any(r['gate'] for r in audit['results'])
        and audit['audit_source_sha256'] == digest(REVIEW / 'audit.py'), 'Review completed layer experiments first')
    entries = cases(parent['base_config'])
    tested = protocol.read(output / 'preflight/final-cpu-tests-job.json')
    protocol.require(tested['status'] == 'complete' and tested['exit_code'] == 0, 'Complete targeted CPU tests first')
    files = [Path(__file__).resolve(), ROOT / 'beyondwaveform/coordinate_diagnostics.py',
        ROOT / 'tests/test_coordinate_diagnostics.py', ROOT / 'tests/test_coordinate_research.py',
        ROOT / 'docs/QUALITY_COORDINATE_RESEARCH.md', output / 'preflight/final-cpu-tests.log',
        output / 'preflight/final-cpu-tests-job.json']
    files.extend(REVIEW / n for n in ('audit.py', 'review.json', 'report.md'))
    files.extend(layers.DEFAULT / n for n in ('plan.json', 'job.json', 'results.json', 'report.md'))
    # Bind the independently reviewed outputs, not just their high-level summary.
    frozen = dict(parent['frozen_files'])
    frozen.update(audit['checked_asset_hashes'])
    for case in entries:
        path = output / 'configs' / (case['name'] + '.json')
        atomic_json(path, case['config']); path.chmod(0o444)
        case['config_path'] = str(path); files.append(path)
    frozen.update({str(p.resolve()): digest(p) for p in files})
    plan = {'created_at': now(), 'frozen_files': frozen, 'runtime': parent['runtime'],
        'additional_dependencies': parent['additional_dependencies'], 'pilot': parent['pilot'],
        'cache_audit': parent['baseline_cache_audit'], 'feature_artifacts': parent['feature_artifacts'],
        'quality': parent['quality'], 'execution': parent['execution'], 'base_config': parent['base_config'],
        'cases': entries, 'baseline': parent['baseline'], 'targets': stable.TARGETS,
        'control_preflight_ids': parent['pilot']['ids'][:3], 'coordinate_spec': SPEC,
        'descriptive_direction_signal': SIGNAL, 'discord_enabled': False, 'scope': SCOPE}
    atomic_json(output / 'plan.json', plan); (output / 'plan.json').chmod(0o444)
    return plan


def check(plan):
    import importlib.metadata
    protocol.require(protocol.runtime() == plan['runtime'] and all(importlib.metadata.version(k) == v
        for k, v in plan['additional_dependencies'].items()), 'Changed runtime')
    stable.validate_case({'name': stable.NAMES[0], 'config': plan['base_config']})
    protocol.require([{k: r[k] for k in ('name', 'config')} for r in plan['cases']] == cases(plan['base_config'])
        and plan['coordinate_spec'] == SPEC and plan['descriptive_direction_signal'] == SIGNAL
        and plan['targets'] == stable.TARGETS and plan['scope'] == SCOPE and not plan['discord_enabled'],
        'Changed coordinate experiment definitions')
    protocol.require(len(plan['pilot']['ids']) == 24 and plan['control_preflight_ids'] == plan['pilot']['ids'][:3]
        and plan['execution'] == {'mode': 'strict_math', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
        'explicit_reflection': True}, 'Changed cohort/execution')
    for path, expected in plan['frozen_files'].items():
        protocol.require(digest(path) == expected, f'Changed frozen coordinate input: {path}')
    protocol.require(digest(plan['pilot']['manifest']) == plan['pilot']['sha256']
        and [r['id'] for r in read_jsonl(plan['pilot']['manifest'])] == plan['pilot']['ids'], 'Changed cohort')


def control_preflight(output, plan):
    protocol.require(not (output / 'control-preflight.json').exists()
        and not (output / 'control-trajectories').exists(), 'Preserve existing preflight evidence')
    protocol.require(os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8', 'Fresh CUDA process needs workspace config')
    base = plan['base_config']
    cache = {r['id']: r for r in protocol.read(plan['cache_audit'])['items']}
    original = {r['id']: r for r in read_jsonl(prior.BASELINE / 'generated.jsonl')}
    new, old = [], []
    with backend_mode('strict_math'):
        seed_everything(base['seed']); vocoder, surrogate, _ = prior.models(plan, base)
        flags = backend_manifest()
        for row in read_jsonl(plan['pilot']['manifest'])[:3]:
            uid = row['id']; record = original[uid]; seed_everything(record['seed'])
            q = torch.load(cache[uid]['q_path'], map_location='cuda', weights_only=True)
            x = load_audio(row['audio']).cuda()
            result = collect(q, x, row['text'], vocoder, surrogate, AttackConfig(**base['attack']),
                SpectralConfig(**base['spectral']), output / 'control-trajectories' / uid)
            verify_coordinate_receipt(result, q.cpu(), x.cpu(), 'uniform')
            new.append({'id': uid, 'seed': record['seed'], **result})
            old.append(protocol.read(record['trajectory_receipt']['path']))
    comparison = numerical.compare_trajectories({'items': old}, {'items': new})
    atomic_json(output / 'control-preflight.json', {'status': 'complete', 'backend': flags,
        'comparison': comparison, 'passed': comparison['passed'], 'items': new,
        'scope': 'Uniform coordinates must match all 150 control updates, gradients, losses, moments, final audio and Adam state.'})
    protocol.require(comparison['passed'], 'Identity coordinate collector differs from frozen control')


def generate(output, plan, case):
    folder = output / 'pilot' / case['name']; config = case['config']
    protocol.require(not (folder / 'run.json').exists(), 'Preserve partial/completed generation before retry')
    protocol.require(os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8', 'Fresh CUDA process needs workspace config')
    protocol.require(protocol.read(output / 'control-preflight.json')['passed'], 'Control preflight did not pass')
    original = {r['id']: r for r in read_jsonl(prior.BASELINE / 'generated.jsonl')}
    cache = {r['id']: r for r in protocol.read(plan['cache_audit'])['items']}
    attack, spectral = AttackConfig(**config['attack']), SpectralConfig(**config['spectral'])
    records = []
    with backend_mode('strict_math'):
        seed_everything(config['seed']); vocoder, surrogate, surrogate_meta = prior.models(plan, config)
        expected = stable.identity(plan, case)
        metadata = {**expected, 'fingerprint': config_fingerprint(expected), 'manifest': plan['pilot']['manifest'],
            'roles': ['pilot'], 'status': 'running', 'features': plan['feature_artifacts'], 'surrogate': surrogate_meta,
            'generation_runtime': protocol.runtime(), 'generation_backend': backend_manifest(),
            'active_terms': ENABLED, 'optimization_coordinates': config['optimization_coordinates'], 'scope': plan['scope']}
        atomic_json(folder / 'run.json', metadata)
        for index, row in enumerate(read_jsonl(plan['pilot']['manifest']), 1):
            uid = row['id']; parent = original[uid]; seed_everything(parent['seed'])
            protocol.require(digest(row['audio']) == row['audio_sha256'] and digest(cache[uid]['q_path']) == cache[uid]['q_sha256'],
                'Changed q/audio')
            q = torch.load(cache[uid]['q_path'], map_location='cuda', weights_only=True).detach()
            x = load_audio(row['audio']).cuda()
            value = collect(q, x, row['text'], vocoder, surrogate, attack, spectral,
                folder / 'trajectories' / uid, kind=case['name'])
            verify_coordinate_receipt(value, q.cpu(), x.cpu(), case['name'])
            delta = numerical.verify_tensor(value['trace'][-1]['delta']).cuda()
            with torch.no_grad():
                audio = vocoder(q + delta); reconstruction = vocoder(q)
            protocol.require(tensor_value_sha256(audio) == value['final_audio']['value_sha256']
                and torch.equal(reconstruction.cpu(), load_audio(parent['audio']['reconstruction']['path'], max_seconds=None)),
                'Resynthesis/clean reconstruction changed')
            assets = {k: parent['audio'][k] for k in ('clean', 'reconstruction')}
            assets['adversarial'] = save_audio(folder / 'audio/adversarial' / (uid + '.wav'), audio)
            receipt = folder / 'trajectories' / uid / 'receipt.json'
            atomic_json(receipt, {'id': uid, 'seed': parent['seed'], 'q_path': cache[uid]['q_path'],
                'q_sha256': cache[uid]['q_sha256'], 'q_value_sha256': tensor_value_sha256(q), **value})
            trace = [{'step': r['step'], 'loss_total': r['loss_values'][0], 'nll': r['loss_values'][1],
                'tv_ratio': r['loss_values'][2], 'hf_ratio': r['loss_values'][3], 'radius_after_update': r['radius']} for r in value['trace']]
            record = {'id': uid, 'text': row['text'], 'language': row['language'], 'role': 'pilot',
                'input_audio_sha256': row['audio_sha256'], 'seed': parent['seed'], 'audio': assets, 'trace': trace,
                'delta': value['trace'][-1]['delta'], 'trajectory_receipt': {'path': str(receipt), 'sha256': digest(receipt)},
                'actual_optimization_updates': 50, 'returned_delta': 'projected update 50', 'active_terms': ENABLED,
                'optimization_coordinates': config['optimization_coordinates']}
            atomic_json(folder / 'records' / (uid + '.json'), record); records.append(record)
            print(f'Generated {case["name"]}: {index}/24 [{uid}]', flush=True)
    write_jsonl(folder / 'generated.jsonl', records)
    metadata.update(status='complete', count=24, generated_sha256=digest(folder / 'generated.jsonl'))
    atomic_json(folder / 'run.json', metadata)


def verify_coordinate_receipt(receipt, q, x, kind):
    protocol.require(receipt['coordinate_kind'] == kind and receipt['coordinate_spec'] == SPEC
        and receipt['active_terms'] == ENABLED and receipt['actual_optimization_updates'] == 50
        and len(receipt['trace']) == 50, 'Changed coordinate receipt')
    weights = numerical.verify_tensor(receipt['weights'])
    expected = coordinate_weights(q, x, kind)
    protocol.require(torch.equal(weights, expected) and (weights > 0).all().item()
        and weights.max().item() / weights.min().item() <= 10.00001, 'Changed clean coordinate weights')
    for number, row in enumerate(receipt['trace'], 1):
        theta = numerical.verify_tensor(row['theta']); delta = numerical.verify_tensor(row['delta'])
        protocol.require(row['step'] == number and theta.shape == delta.shape == q.shape
            and theta.dtype == delta.dtype == weights.dtype == q.dtype
            and torch.equal(delta, physical_delta(theta, weights, kind == 'uniform')),
            'Saved physical delta does not equal weighted theta')
        radius = (delta.double().norm() / (q.double().norm() + 1e-8)).item()
        protocol.require(radius <= .100001 and abs(radius - row['radius']) <= 1e-6, 'Physical radius receipt mismatch')
        protocol.require(row['gradient_coordinate'] == 'theta'
            and tensor_value_sha256(torch.tensor(row['loss_values'], dtype=q.dtype)) == row['loss_value_sha256'],
            'Changed loss/gradient-coordinate receipt')
    state = receipt['final_optimizer']
    protocol.require(digest(state['path']) == state['file_sha256']
        and tensor_tree(torch.load(state['path'], map_location='cpu', weights_only=True))
        == state['value_receipt'] == receipt['trace'][-1]['optimizer_value_receipt'], 'Changed coordinate Adam state')


def verify(output, plan, case):
    # Full saved-output/score audit plus scalar, active-term, terminal-audio and optimizer checks.
    prior.verify(output, plan, case)
    folder = output / 'pilot' / case['name']
    protocol.require(protocol.read(folder / 'run.json')['optimization_coordinates'] == case['config']['optimization_coordinates'],
        'Changed coordinate metadata')
    source = {r['id']: r for r in read_jsonl(plan['pilot']['manifest'])}
    def controls(path):
        return {(r['id'], r['variant']): r for r in
            (json.loads(line) for line in path.read_text().splitlines() if line.strip())
            if r['variant'] in ('clean', 'reconstruction')}
    protocol.require(controls(folder / 'evaluation/utterances.jsonl')
        == controls(prior.BASELINE / 'evaluation/utterances.jsonl'), 'Clean/reconstruction scores differ from full control')
    for record in read_jsonl(folder / 'generated.jsonl'):
        receipt = protocol.read(record['trajectory_receipt']['path'])
        protocol.require(record['optimization_coordinates'] == case['config']['optimization_coordinates'], 'Changed record coordinates')
        q = torch.load(receipt['q_path'], map_location='cpu', weights_only=True)
        protocol.require(digest(receipt['q_path']) == receipt['q_sha256']
            and tensor_value_sha256(q) == receipt['q_value_sha256'], 'Changed clean q binding')
        verify_coordinate_receipt(receipt, q, load_audio(source[record['id']]['audio']), case['name'])
    path = folder / 'verification.json'; result = protocol.read(path)
    result.update(coordinate_weights_rebuilt=24, theta_delta_pairs_verified=1200,
        coordinate_kind=case['name'], original_physical_ball_verified=True, exact_control_score_rows=48)
    atomic_json(path, result)


def direction_signal(row, baseline):
    return (row['quality']['utmos'] >= baseline['quality']['utmos'] + SIGNAL['min_UTMOS_gain']
        and row['error_percent'] >= baseline['error_percent'] - SIGNAL['max_WER_loss_pp']
        and all(row['quality'][m] > SIGNAL['min_each_MOS_exclusive'] for m in protocol.METRICS))


def report(output, rows, status, plan):
    baseline = plan['baseline']
    atomic_json(output / 'results.json', {'status': status, 'updated_at': now(), 'baseline': baseline,
        'results': rows, 'scope': plan['scope']})
    lines = ['# Clean-statistic Adam coordinates', '', f'Status: **{status}**.', '',
        '| Coordinates | WER (%) | DNSMOS | NISQA | UTMOS | Delta WER (pp) | Delta UTMOS | Direction signal | Joint targets |',
        '|---|---:|---:|---:|---:|---:|---:|---|---|', '| Paper | 75.4300 | 2.9540 | 3.5260 | 3.5900 | — | — | — | — |']
    for row in [dict(baseline, name='Uniform (frozen control)'), *rows]:
        q = row['quality']
        lines.append(f"| {row['name']} | {row['error_percent']:.4f} | {q['dnsmos']:.4f} | {q['nisqa']:.4f} | {q['utmos']:.4f} | "
            f"{row['error_percent']-baseline['error_percent']:+.4f} | {q['utmos']-baseline['quality']['utmos']:+.4f} | "
            f"{direction_signal(row, baseline)} | {stable.joint_target(row)} |")
    lines += ['', plan['scope'], '', 'The frozen direction signal (UTMOS gain >=0.05, WER loss <=5 percentage points, '
        'each MOS >2.5) is descriptive and does not trigger confirmation, selection or test expansion. '
        'Published joint targets are reported separately without relaxing them.', '']
    (output / 'report.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT)
    parser.add_argument('--prepare', action='store_true'); parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--stage', choices=('control-preflight', 'generate', 'evaluate', 'verify'))
    parser.add_argument('--case', choices=NAMES); args = parser.parse_args()
    os.environ['BEYONDWAVEFORM_NOTIFY_DISCORD'] = '0'; output = args.output.resolve()
    plan = prepare(output) if args.prepare else protocol.read(output / 'plan.json'); check(plan)
    if args.prepare or args.check_only:
        print(f'Coordinate preflight passed: {len(plan["frozen_files"])} frozen inputs.', flush=True); return
    if args.stage:
        if args.stage == 'control-preflight':
            control_preflight(output, plan)
        else:
            protocol.require(args.case is not None, 'Stage needs fixed coordinates')
            case = next(r for r in plan['cases'] if r['name'] == args.case)
            {'generate': generate, 'evaluate': stable.evaluate_case, 'verify': verify}[args.stage](output, plan, case)
        check(plan); return
    supervisor = Supervisor(output); rows = []
    try:
        report(output, rows, 'running', plan)
        base = [sys.executable, '-u', str(Path(__file__).resolve()), '--output', str(output)]
        stages.phase(supervisor, 'uniform-control-preflight', ['/usr/bin/env', 'CUBLAS_WORKSPACE_CONFIG=:4096:8',
            *base, '--stage', 'control-preflight'])
        for case in plan['cases']:
            for stage in ('generate', 'evaluate', 'verify'):
                command = [*base, '--case', case['name'], '--stage', stage]
                if stage in ('generate', 'verify'):
                    command = ['/usr/bin/env', 'CUBLAS_WORKSPACE_CONFIG=:4096:8', *command]
                stages.phase(supervisor, case['name'] + '-' + stage, command)
            folder = output / 'pilot' / case['name']; summary = protocol.read(folder / 'evaluation/summary.json')
            rows.append({'name': case['name'], 'pool': 'pilot', 'run': str(folder), **summary['summary']['adversarial'],
                'summary_sha256': digest(folder / 'evaluation/summary.json'), 'audit_sha256': digest(folder / 'verification.json')})
            report(output, rows, 'running', plan)
        check(plan); report(output, rows, 'complete', plan); supervisor.finish(scope=plan['scope'])
    except BaseException as error:
        report(output, rows, 'failed', plan); supervisor.fail(error); raise


if __name__ == '__main__':
    main()
