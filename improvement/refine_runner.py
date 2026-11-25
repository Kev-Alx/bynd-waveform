"""Real source-only refinement; pilot48, then a frozen conditional validation76."""
from __future__ import annotations
import argparse
import gc
from pathlib import Path
import time
import traceback
import torch
import execution_runtime as rt
import runtime_refine as cycle
from r3_real_runner_v2 import load_encoder_only
from waveform_pgd import WaveformPGD, PGDFailure
from constrained_refine import WaveformRefiner, RefineFailure


def generate(cohort):
    plan = cycle.check()
    if cohort == 'validation':
        cycle.validation_freeze()
    old = rt.read(rt.RUN / 'plan.json'); rt.check(old)
    info = plan['cohorts'][cohort]
    folder = cycle.RUN / cohort
    if (folder / 'records').exists():
        raise FileExistsError('Preserve previous or partial refinement outputs')
    folder.mkdir(parents=True, exist_ok=True)
    result = {'status': 'running', 'cohort': cohort, 'count': info['count'], 'actual_count': 0,
        'source_only': True, 'target_models_evaluated': [], 'failures': [],
        'plan_sha256': rt.digest(cycle.RUN / 'plan.json'), 'started_at_utc': rt.stamp()}
    rt.atomic_json(folder / 'generation-result.json', result)
    started = time.perf_counter()

    def progress(**kwargs):
        value = {'updated_at_utc': rt.stamp(), 'cohort': cohort, 'total': info['count'], **kwargs}
        rt.atomic_json(folder / 'generation-status.json', value); print(value, flush=True)

    try:
        with rt.backend_mode('strict_math'):
            vocoder, source, native_source, source_metadata = rt.models(old)
            native_encoder, encoder, encoder_metadata = load_encoder_only(old)
            torch.cuda.reset_peak_memory_stats()
            historical = {r['id']: r for r in rt.read_jsonl(cycle.OUT / 'execution48/mechanism-pilot48/generated.jsonl')}
            records = []
            for index, row in enumerate(rt.read_jsonl(info['manifest']), 1):
                uid = row['id']; begin = time.perf_counter()
                if rt.digest(row['audio']) != row['audio_sha256']:
                    raise ValueError('Changed source audio')
                audio = rt.load_audio(row['audio']).cuda()
                assets = {'clean': rt.save_audio(folder / 'audio/clean' / f'{uid}.wav', audio)}
                diagnostics = {}
                if cohort == 'pilot':
                    prior = historical[uid]
                    q_asset = prior['clean_features']; rt_digest = rt.digest(q_asset['path'])
                    if rt_digest != q_asset['sha256']:
                        raise ValueError('Changed fixed strict feature reference')
                    q = torch.load(q_asset['path'], map_location='cuda', weights_only=True).detach()
                    initial_asset = prior['audio']['pgd_full']
                    if rt.digest(initial_asset['path']) != initial_asset['sha256']:
                        raise ValueError('Changed full50 initialization')
                    initial = rt.load_audio(initial_asset['path']).cuda()
                    assets['full50'] = initial_asset
                    diagnostics['full50'] = prior['diagnostics']['pgd']['pgd_full']
                else:
                    with torch.no_grad():
                        q = encoder(audio).detach()
                        if not torch.equal(q, native_encoder(audio)):
                            raise ValueError('Strict adapter changes encoder values')
                    initial_run = WaveformPGD(source.nll, encoder, condition='full_r3').run(audio, row['text'], clean_features=q)
                    initial = initial_run.adversarial
                    assets['full50'] = rt.save_audio(folder / 'audio/full50' / f'{uid}.wav', initial)
                    initial_path = folder / 'traces/full50' / f'{uid}.json'
                    rt.atomic_json(initial_path, {'status': 'completed', 'condition': 'full_r3',
                        'trace': initial_run.trace, 'final_metrics': initial_run.final_metrics, 'counts': initial_run.counts})
                    diagnostics['full50'] = {'receipt': str(initial_path), 'sha256': rt.digest(initial_path), 'status': 'completed'}
                    del initial_run
                q_path = folder / 'clean-features' / f'{uid}.pt'; q_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(q.cpu(), q_path)
                diagnostics['q'] = {'path': str(q_path), 'sha256': rt.digest(q_path)}

                # Save both50 and100 from the same Adam trajectory; the pilot
                # step50 delta must reproduce its previously frozen control.
                fv_begin = time.perf_counter()
                objective = rt.objective(q, audio, row['text'], vocoder, source)
                delta = torch.zeros_like(q, requires_grad=True)
                optimizer = torch.optim.Adam([delta], lr=.02); fv_trace = []
                for step in range(100):
                    optimizer.zero_grad(set_to_none=True)
                    terms = objective(q + delta)
                    gradient, = torch.autograd.grad(terms['loss'], delta)
                    if not torch.isfinite(gradient).all() or not torch.isfinite(terms['loss']):
                        raise RuntimeError('Nonfinite original FV control')
                    delta.grad = gradient; optimizer.step(); rt.project_relative_ball_(delta, q, .1)
                    fv_trace.append({'step': step + 1, **{k: float(v.detach()) for k, v in terms.items()},
                        'relative_radius': float(rt.relative_radius(delta, q))})
                    if step + 1 in (50, 100):
                        variant = f'fv{step + 1}'
                        if cohort == 'pilot' and step == 49:
                            prior_delta_asset = historical[uid]['diagnostics']['fv']['delta']
                            if rt.digest(prior_delta_asset['path']) != prior_delta_asset['sha256']:
                                raise ValueError('Changed pilot FV50 replay reference')
                            prior_delta = torch.load(prior_delta_asset['path'], map_location='cuda', weights_only=True)
                            if not torch.equal(delta, prior_delta):
                                rt.atomic_json(folder / 'fv-replay-mismatch.json', {'id': uid,
                                    'maximum_absolute_error': float((delta - prior_delta).abs().max())})
                                raise RuntimeError('FV50 replay mismatch; preserve before further optimization')
                            del prior_delta
                        with torch.no_grad():
                            assets[variant] = rt.save_audio(folder / 'audio' / variant / f'{uid}.wav', vocoder(q + delta))
                        delta_path = folder / 'fv-deltas' / f'{uid}-{step + 1}.pt'
                        delta_path.parent.mkdir(parents=True, exist_ok=True); torch.save(delta.detach().cpu(), delta_path)
                        trace_path = folder / 'traces' / variant / f'{uid}.json'
                        rt.atomic_json(trace_path, list(fv_trace))
                        diagnostics[variant] = {'delta': {'path': str(delta_path), 'sha256': rt.digest(delta_path)},
                            'trace': str(trace_path), 'trace_sha256': rt.digest(trace_path),
                            'relative_radius': float(rt.relative_radius(delta, q)), 'source_backward_calls': step + 1,
                            'wall_seconds_cumulative': time.perf_counter() - fv_begin,
                            'pilot50_exact_replay': cohort == 'pilot'}
                    if step + 1 in (1, 50, 100):
                        progress(status='running', clip=index, id=uid, variant='fv100', update=step + 1)
                del delta, optimizer, fv_trace, objective

                for condition, variant in [('weighted_control', 'weighted100'), ('constrained', 'constrained100')]:
                    before = time.perf_counter()
                    def on_step(item, label=variant):
                        if item['step'] in (1, 10, 25, 50):
                            progress(status='running', clip=index, id=uid, variant=label, update=item['step'])
                    try:
                        refined = WaveformRefiner(source.nll, encoder, condition=condition).run(
                            audio, row['text'], initial_waveform=initial, clean_features=q, on_step=on_step)
                        output = refined.adversarial
                        receipt = {'status': 'completed', 'condition': condition, 'trace': refined.trace,
                            'final_metrics': refined.final_metrics, 'counts': refined.counts}
                    except RefineFailure as error:
                        # Fixed caps/direction failures are recorded; unexpected
                        # external exceptions propagate and stop infrastructure.
                        if error.adversarial is None:
                            raise
                        output = error.adversarial; receipt = error.receipt
                        result['failures'].append({'id': uid, 'variant': variant, 'reason': str(error)})
                    torch.cuda.synchronize(); receipt['wall_seconds'] = time.perf_counter() - before
                    receipt_path = folder / 'traces' / variant / f'{uid}.json'; rt.atomic_json(receipt_path, receipt)
                    assets[variant] = rt.save_audio(folder / 'audio' / variant / f'{uid}.wav', output)
                    diagnostics[variant] = {'receipt': str(receipt_path), 'sha256': rt.digest(receipt_path), 'status': receipt['status']}
                    del output, receipt
                    if 'refined' in locals(): del refined
                record = {'id': uid, 'text': row['text'], 'speaker': row['speaker'], 'language': 'en',
                    'role': cohort, 'source_only': True, 'count': info['count'], 'input_audio_sha256': row['audio_sha256'],
                    'audio': assets, 'diagnostics': diagnostics, 'plan_sha256': result['plan_sha256'],
                    'wall_seconds': time.perf_counter() - begin}
                rt.atomic_json(folder / 'records' / f'{uid}.json', record); records.append(record)
                result['actual_count'] = index; rt.atomic_json(folder / 'generation-result.json', result)
                progress(status='running', completed=index, total=info['count'], id=uid)
                del q, audio, initial, assets, diagnostics
                gc.collect()
            rt.write_jsonl(folder / 'generated.jsonl', records)
            result.update(status='completed', wall_seconds=time.perf_counter() - started,
                generated_sha256=rt.digest(folder / 'generated.jsonl'), source_metadata=source_metadata,
                encoder_metadata=encoder_metadata, backend=rt.backend_manifest(),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                completed_at_utc=rt.stamp())
            rt.atomic_json(folder / 'generation-result.json', result)
            progress(status='completed', completed=info['count'], total=info['count'])
    except Exception as error:
        result.update(status='negative_initialization' if isinstance(error, PGDFailure) else 'failed_infrastructure', error=str(error), error_type=type(error).__name__,
            wall_seconds=time.perf_counter() - started, traceback=traceback.format_exc())
        rt.atomic_json(folder / 'generation-result.json', result); progress(status='failed_infrastructure', error=str(error))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['register', 'generate'])
    parser.add_argument('--cohort', choices=['pilot', 'validation'], default='pilot')
    args = parser.parse_args()
    print(cycle.register()) if args.stage == 'register' else generate(args.cohort)
