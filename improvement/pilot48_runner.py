"""Registered48-utterance mechanism test; the detached supervisor owns the GPU."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import gc
from pathlib import Path
import time
import traceback

import torch
import execution_runtime as rt
import runtime48 as cycle
from r3_real_runner_v2 import load_encoder_only
from r3_waveform import make_generator, multiresolution_stft_loss, relative_feature_loss
from waveform_pgd import PGDConfig, PGDFailure, WaveformPGD

FOLDER = cycle.RUN / 'mechanism-pilot48'


def progress(**values):
    item = {'updated_at_utc': rt.stamp(), 'pilot_count': 48, **values}
    rt.atomic_json(FOLDER / 'generation-status.json', item)
    print(item, flush=True)


def fv_control(q, audio, text, vocoder, source, on_step):
    objective = rt.objective(q, audio, text, vocoder, source)
    delta = torch.zeros_like(q, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=.02)
    trace = []
    for update in range(50):
        optimizer.zero_grad(set_to_none=True)
        terms = objective(q + delta)
        gradient, = torch.autograd.grad(terms['loss'], delta)
        if not torch.isfinite(gradient).all() or not torch.isfinite(terms['loss']):
            raise RuntimeError('Nonfinite original feature-control objective/gradient')
        delta.grad = gradient
        optimizer.step()
        rt.project_relative_ball_(delta, q, .1)
        row = {key: float(value.detach()) for key, value in terms.items()}
        row.update(update=update + 1, gradient_norm=float(gradient.double().norm()),
                   relative_radius=float(rt.relative_radius(delta, q)))
        trace.append(row)
        on_step(update + 1)
    with torch.no_grad():
        adversarial = vocoder(q + delta)
    return adversarial, delta.detach(), trace


def generate():
    plan = cycle.check()
    old = rt.read(plan['prior_plan'])
    rt.check(old)
    if (FOLDER / 'generated.jsonl').exists() or (FOLDER / 'records').exists():
        raise FileExistsError('Preserve prior/partial48-point generation; use an explicitly registered continuation')
    FOLDER.mkdir(parents=True, exist_ok=True)
    rows = rt.read_jsonl(plan['development_manifest'])
    if len(rows) != 48:
        raise ValueError('Exactly48 utterances required')
    result = {'status': 'running', 'source_only': True, 'target_models_evaluated': [],
              'pilot_count': 48, 'actual_count': 0, 'pgd_mathematical_failures': [],
              'plan_sha256': rt.digest(cycle.RUN / 'plan.json'), 'started_at_utc': rt.stamp()}
    rt.atomic_json(FOLDER / 'generation-result.json', result)
    started = time.perf_counter()
    try:
        with rt.backend_mode('strict_math'):
            vocoder, source, native_source, source_metadata = rt.models(old)
            native_encoder, encoder, encoder_metadata = load_encoder_only(old)
            trained = make_generator(11).cuda().eval()
            untrained = make_generator(11).cuda().eval()
            state = torch.load(plan['generator_checkpoint'], map_location='cpu', weights_only=True)
            trained.load_state_dict(state['generator'], strict=True)
            del state
            config = PGDConfig()
            if asdict(config)['stft_weight'] != plan['pgd']['weights']['stft'] or config.content_weight != plan['pgd']['weights']['content']:
                raise ValueError('PGD loss must match frozen R3 coefficients exactly')
            torch.cuda.reset_peak_memory_stats()
            records = []
            for index, row in enumerate(rows, 1):
                uid = row['id']
                clip_started = time.perf_counter()
                if rt.digest(row['audio']) != row['audio_sha256']:
                    raise ValueError('Changed registered source audio')
                audio = rt.load_audio(row['audio']).cuda()
                with torch.no_grad():
                    q = encoder(audio).detach()
                    q_repeat = native_encoder(audio).detach()
                    if not torch.equal(q, q_repeat):
                        raise ValueError('Gradient adapter changes strict native encoder output')
                    clean_nll = float(source.nll(audio, row['text']))
                    recon = vocoder(q)
                    initial, learned = untrained(audio), trained(audio)
                q_path = FOLDER / 'clean-features' / f'{uid}.pt'
                q_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(q.cpu(), q_path)
                assets = {variant: rt.save_audio(FOLDER / 'audio' / variant / f'{uid}.wav', waveform)
                          for variant, waveform in [('clean', audio), ('reconstruction', recon),
                                                    ('r3_initial', initial), ('r3_trained', learned)]}
                diagnostics = {'clean_sum_nll': clean_nll, 'pgd': {}, 'generator': {}}
                with torch.no_grad():
                    for label, waveform in [('r3_initial', initial), ('r3_trained', learned)]:
                        residual = waveform - audio
                        diagnostics['generator'][label] = {
                            'sum_nll': float(source.nll(waveform, row['text'])),
                            'stft_loss': float(multiresolution_stft_loss(waveform, audio)),
                            'content_loss': float(relative_feature_loss(encoder(waveform), q)),
                            'linf': float(residual.abs().max()),
                            'rms': float(residual.square().mean().sqrt()),
                            'samples': waveform.numel()}
                del initial, learned, recon, q_repeat

                def status_step(variant, update):
                    if update in (1, 10, 25, 50):
                        progress(phase='generation', status='running', clip=index,
                                 total=48, id=uid, variant=variant, update=update, updates=50)

                fv_started = time.perf_counter()
                baseline, delta, trace = fv_control(q, audio, row['text'], vocoder, source,
                                                    lambda step: status_step('fv_quality', step))
                assets['fv_quality'] = rt.save_audio(FOLDER / 'audio/fv_quality' / f'{uid}.wav', baseline)
                delta_path = FOLDER / 'fv-deltas' / f'{uid}.pt'
                delta_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(delta.cpu(), delta_path)
                rt.atomic_json(FOLDER / 'fv-traces' / f'{uid}.json', trace)
                diagnostics['fv'] = {'delta': {'path': str(delta_path), 'sha256': rt.digest(delta_path)},
                    'trace': str(FOLDER / 'fv-traces' / f'{uid}.json'),
                    'trace_sha256': rt.digest(FOLDER / 'fv-traces' / f'{uid}.json'),
                    'relative_radius': float(rt.relative_radius(delta, q)),
                    'source_backward_calls': 50, 'wall_seconds': time.perf_counter() - fv_started}
                del baseline, delta, trace

                for condition, variant in [('full_r3', 'pgd_full'), ('nll_only', 'pgd_nll')]:
                    before = time.perf_counter()
                    optimizer = WaveformPGD(source.nll, encoder, condition=condition, config=config)
                    try:
                        attack = optimizer.run(audio, row['text'], clean_features=q,
                            on_step=lambda step, v=variant: status_step(v, step['step']))
                        output = attack.adversarial
                        receipt = {'status': 'completed', 'condition': condition,
                            'trace': attack.trace, 'final_metrics': attack.final_metrics, 'counts': attack.counts}
                    except PGDFailure as error:
                        if error.receipt.get('failure_type') != 'mathematical':
                            raise RuntimeError('PGD contract failure must stop the job') from error
                        if error.adversarial is None:
                            raise RuntimeError('PGD failure has no valid attempted waveform') from error
                        output = error.adversarial
                        receipt = error.receipt
                        result['pgd_mathematical_failures'].append({'id': uid, 'variant': variant,
                            'reason': str(error), 'completed_updates': receipt['counts']['completed_updates']})
                    torch.cuda.synchronize()
                    receipt['wall_seconds'] = time.perf_counter() - before
                    receipt_path = FOLDER / 'pgd-traces' / variant / f'{uid}.json'
                    rt.atomic_json(receipt_path, receipt)
                    assets[variant] = rt.save_audio(FOLDER / 'audio' / variant / f'{uid}.wav', output)
                    diagnostics['pgd'][variant] = {'receipt': str(receipt_path),
                        'sha256': rt.digest(receipt_path), 'status': receipt['status']}
                    del optimizer, output, receipt
                    if 'attack' in locals():
                        del attack
                torch.cuda.synchronize()
                record = {'id': uid, 'text': row['text'], 'speaker': row['speaker'],
                    'language': 'en', 'role': 'pilot', 'pilot_count': 48,
                    'source_only': True, 'input_audio_sha256': row['audio_sha256'],
                    'audio': assets, 'diagnostics': diagnostics,
                    'clean_features': {'path': str(q_path), 'sha256': rt.digest(q_path)},
                    'wall_seconds': time.perf_counter() - clip_started,
                    'plan_sha256': result['plan_sha256']}
                rt.atomic_json(FOLDER / 'records' / f'{uid}.json', record)
                records.append(record)
                result['actual_count'] = index
                rt.atomic_json(FOLDER / 'generation-result.json', result)
                progress(phase='generation', status='running', completed=index, total=48, id=uid)
                del q, audio, assets, diagnostics
                gc.collect()
            rt.write_jsonl(FOLDER / 'generated.jsonl', records)
            result.update(status='completed', completed_at_utc=rt.stamp(),
                wall_seconds=time.perf_counter() - started,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                source_metadata=source_metadata, encoder_metadata=encoder_metadata,
                backend=rt.backend_manifest(),
                generated_sha256=rt.digest(FOLDER / 'generated.jsonl'))
            rt.atomic_json(FOLDER / 'generation-result.json', result)
            progress(phase='generation', status='completed', completed=48, total=48)
    except Exception as error:
        result.update(status='failed_infrastructure', error=str(error), error_type=type(error).__name__,
            wall_seconds=time.perf_counter() - started, traceback=traceback.format_exc())
        rt.atomic_json(FOLDER / 'generation-result.json', result)
        progress(phase='generation', status='failed_infrastructure', error=str(error))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['register', 'generate'])
    args = parser.parse_args()
    if args.stage == 'register':
        print(cycle.register())
    else:
        generate()
