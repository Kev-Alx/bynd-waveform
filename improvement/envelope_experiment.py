"""Frozen clean-energy experiment:48-point comparison, conditional validation76."""
import argparse
import gc
from pathlib import Path
import time
import traceback
import torch
import execution_runtime as rt
from envelope_pgd import EnvelopePGD, attenuate_to_rms
from waveform_pgd import WaveformPGD
from r3_real_runner_v2 import load_encoder_only

OUT = Path(__file__).resolve().parent
RUN = OUT / 'execution_envelope'
VARIANTS = ('clean', 'fv50', 'global_full50', 'global_nll50', 'envelope_nll50',
            'envelope_full50', 'attenuated_global_nll50')


def register():
    old = rt.read(rt.RUN / 'plan.json'); rt.check(old)
    cohorts = {'pilot': {'count': 48, 'manifest': str(OUT / 'execution48/development-48.jsonl')},
        'validation': {'count': 76, 'manifest': str(OUT / 'execution_refine/validation-76.jsonl')}}
    for info in cohorts.values():
        rows = rt.read_jsonl(info['manifest'])
        if len(rows) != info['count'] or len({r['id'] for r in rows}) != info['count']:
            raise ValueError('Incomplete cohort registration')
        info['sha256'] = rt.digest(info['manifest'])
    names = ('envelope_pgd.py', 'envelope_experiment.py', 'source_evaluate_envelope.py',
        'source_evaluate_refine.py', 'supervise_envelope.py', 'postprocess_envelope.py',
        'test_envelope_pgd.py', '12_envelope_protocol.md', 'waveform_pgd.py', 'r3_waveform.py',
        'execution_runtime.py', 'r3_real_runner_v2.py', 'r3_encoder_repair.py', 'supervise_pilot48.py',
        'execution48/audit_pilot48.py', 'execution_refine/audit_refinement.py',
        'execution48/mechanism-pilot48/generated.jsonl', 'execution48/mechanism-pilot48/source-summary.json',
        'execution_refine/pilot/generated.jsonl', 'execution_refine/pilot/source-summary.json',
        'execution_refine/selection.json', 'execution_refine/validation-76.receipt.json',
        'execution_refine/create_validation76.py')
    files = [OUT / name for name in names] + [RUN / name for name in
        ('prior-work.md', 'audit_envelope.py', 'tests.log', 'test_audit_envelope.py', 'audit-tests.log', 'compute-ledger.json')]
    files += [Path(info['manifest']) for info in cohorts.values()] + [OUT.parent / 'scripts/run_logged.py']
    value = {'created_at_utc': rt.stamp(), 'cohorts': cohorts, 'variants': list(VARIANTS),
        'source_only': True, 'target_models_evaluated': [], 'confirmation_or_final_inference': False,
        'maximum_cycle_wall_seconds': 7200, 'candidate': 'envelope_nll50', 'control': 'fv50',
        'algorithm': {'epsilon': .003, 'frame_samples': 400, 'hop_samples': 160, 'local_snr_db': 30,
            'tail': 'Additional end-aligned frame unless already present', 'bound': 'minimum cap of every covering frame',
            'updates': 50, 'initialization': 'zero', 'step': '2*pointwise_bound/50', 'selection': 'final iterate',
            'full_objective_ablation_weights': {'stft': 65.08367870148994, 'content': 186.6123439723859}},
        'validation_gate': 'Failure-free primary candidate passes all quality floors and primary G1 vsFV50 OR exploratory quality/frontier rule: WER>=global_full50-2pp, NISQA>=global_full50+.10, UTMOS>=global_full50-.05, WER>=attenuated_global_nll50+3pp. Fixed single candidate; no validation reselection.',
        'scientific_limits': 'Engineering feasible-set/coordinate-step intervention, not auditory masking or novelty. Historical-exposure source validation; no human or held-out inference.',
        'bindings': {str(path): rt.digest(path) for path in files}}
    path = RUN / 'plan.json'
    if path.exists():
        raise FileExistsError('Preserve earlier envelope registration')
    rt.atomic_json(path, value); path.chmod(0o444)
    return value


def check():
    plan = rt.read(RUN / 'plan.json')
    for path, expected in plan['bindings'].items():
        if rt.digest(path) != expected: raise ValueError(f'Changed frozen input: {path}')
    if not plan['source_only'] or plan['target_models_evaluated'] or plan['confirmation_or_final_inference']:
        raise ValueError('Changed source-only scope')
    return plan


def validation_freeze():
    freeze = rt.read(RUN / 'validation-freeze.json')
    if freeze['status'] != 'validate' or freeze['candidate'] != 'envelope_nll50' or freeze['control'] != 'fv50':
        raise ValueError('No qualifying frozen envelope candidate')
    for path, expected in freeze['bindings'].items():
        if rt.digest(path) != expected: raise ValueError('Changed pre-validation freeze')
    return freeze


def generate(cohort):
    plan = check()
    if cohort == 'validation': validation_freeze()
    previous = rt.read(rt.RUN / 'plan.json'); rt.check(previous)
    info = plan['cohorts'][cohort]; folder = RUN / cohort
    if (folder / 'records').exists(): raise FileExistsError('Preserve existing/partial cohort')
    folder.mkdir(parents=True, exist_ok=True)
    begin = time.perf_counter()
    result = {'status': 'running', 'cohort': cohort, 'count': info['count'], 'actual_count': 0,
        'source_only': True, 'target_models_evaluated': [], 'failures': [],
        'started_at_utc': rt.stamp(), 'plan_sha256': rt.digest(RUN / 'plan.json')}
    rt.atomic_json(folder / 'generation-result.json', result)
    def progress(**value):
        state = {'updated_at_utc': rt.stamp(), 'cohort': cohort, 'total': info['count'], **value}
        rt.atomic_json(folder / 'generation-status.json', state); print(state, flush=True)
    def save_trace(variant, uid, run, condition):
        path = folder / 'traces' / variant / f'{uid}.json'
        rt.atomic_json(path, {'status': 'completed', 'condition': condition, 'trace': run.trace,
            'final_metrics': run.final_metrics, 'counts': run.counts})
        return {'receipt': str(path), 'sha256': rt.digest(path), 'status': 'completed'}
    try:
        with rt.backend_mode('strict_math'):
            vocoder, source, native, source_metadata = rt.models(previous)
            native_encoder, encoder, encoder_metadata = load_encoder_only(previous)
            torch.cuda.reset_peak_memory_stats()
            history = {r['id']: r for r in rt.read_jsonl(OUT / 'execution48/mechanism-pilot48/generated.jsonl')}
            fv_history = {r['id']: r for r in rt.read_jsonl(OUT / 'execution_refine/pilot/generated.jsonl')}
            records = []
            for index, row in enumerate(rt.read_jsonl(info['manifest']), 1):
                uid = row['id']; clip_begin = time.perf_counter()
                if rt.digest(row['audio']) != row['audio_sha256']: raise ValueError('Changed original audio')
                audio = rt.load_audio(row['audio']).cuda()
                assets = {'clean': rt.save_audio(folder / 'audio/clean' / f'{uid}.wav', audio)}
                diagnostics = {}
                if cohort == 'pilot':
                    prior = history[uid]
                    q_asset = prior['clean_features']
                    if rt.digest(q_asset['path']) != q_asset['sha256']: raise ValueError('Changed clean reference')
                    q = torch.load(q_asset['path'], weights_only=True, map_location='cuda').detach()
                    assets['fv50'] = fv_history[uid]['audio']['fv50']
                    diagnostics['fv50'] = fv_history[uid]['diagnostics']['fv50']
                    for variant, old_variant in [('global_full50', 'pgd_full'), ('global_nll50', 'pgd_nll')]:
                        assets[variant] = prior['audio'][old_variant]
                        diagnostics[variant] = prior['diagnostics']['pgd'][old_variant]
                else:
                    with torch.no_grad():
                        q = encoder(audio).detach()
                        if not torch.equal(q, native_encoder(audio)): raise ValueError('Content adapter changed values')
                    delta = torch.zeros_like(q, requires_grad=True)
                    optimizer = torch.optim.Adam([delta], lr=.02)
                    objective = rt.objective(q, audio, row['text'], vocoder, source); trace = []
                    for step in range(50):
                        optimizer.zero_grad(set_to_none=True); terms = objective(q + delta)
                        gradient, = torch.autograd.grad(terms['loss'], delta)
                        if not torch.isfinite(gradient).all() or not torch.isfinite(terms['loss']):
                            raise ValueError('Invalid FV control')
                        delta.grad = gradient; optimizer.step(); rt.project_relative_ball_(delta, q, .1)
                        trace.append({'step': step + 1, **{k: float(v.detach()) for k, v in terms.items()},
                            'relative_radius': float(rt.relative_radius(delta, q))})
                    with torch.no_grad(): assets['fv50'] = rt.save_audio(folder / 'audio/fv50' / f'{uid}.wav', vocoder(q + delta))
                    dpath = folder / 'fv-deltas' / f'{uid}.pt'; dpath.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(delta.detach().cpu(), dpath)
                    tpath = folder / 'traces/fv50' / f'{uid}.json'; rt.atomic_json(tpath, trace)
                    diagnostics['fv50'] = {'delta': {'path': str(dpath), 'sha256': rt.digest(dpath)},
                        'trace': str(tpath), 'trace_sha256': rt.digest(tpath),
                        'relative_radius': float(rt.relative_radius(delta, q)), 'source_backward_calls': 50,
                        'pilot50_exact_replay': False}
                    del delta, optimizer, objective, trace
                    for variant, condition in [('global_full50', 'full_r3'), ('global_nll50', 'nll_only')]:
                        run = WaveformPGD(source.nll, encoder, condition=condition).run(audio, row['text'], clean_features=q)
                        assets[variant] = rt.save_audio(folder / 'audio' / variant / f'{uid}.wav', run.adversarial)
                        diagnostics[variant] = save_trace(variant, uid, run, condition); del run
                qpath = folder / 'clean-features' / f'{uid}.pt'; qpath.parent.mkdir(parents=True, exist_ok=True)
                torch.save(q.cpu(), qpath); diagnostics['q'] = {'path': str(qpath), 'sha256': rt.digest(qpath)}
                for variant, condition in [('envelope_nll50', 'nll_only'), ('envelope_full50', 'full_r3')]:
                    def on_step(item, label=variant):
                        if item['step'] in (1, 10, 25, 50):
                            progress(status='running', clip=index, id=uid, variant=label, update=item['step'])
                    run = EnvelopePGD(source.nll, encoder, condition=condition).run(audio, row['text'], clean_features=q, on_step=on_step)
                    assets[variant] = rt.save_audio(folder / 'audio' / variant / f'{uid}.wav', run.adversarial)
                    diagnostics[variant] = save_trace(variant, uid, run, condition)
                    if variant == 'envelope_nll50':
                        global_asset = assets['global_nll50']
                        if rt.digest(global_asset['path']) != global_asset['sha256']: raise ValueError('Changed global control')
                        global_audio = rt.load_audio(global_asset['path']).cuda()
                        matched, receipt = attenuate_to_rms(audio, global_audio, run.adversarial)
                        assets['attenuated_global_nll50'] = rt.save_audio(folder / 'audio/attenuated_global_nll50' / f'{uid}.wav', matched)
                        diagnostics['attenuation'] = receipt
                        del global_audio, matched
                    del run
                record = {'id': uid, 'text': row['text'], 'speaker': row['speaker'], 'language': 'en',
                    'role': cohort, 'source_only': True, 'count': info['count'], 'input_audio_sha256': row['audio_sha256'],
                    'audio': assets, 'diagnostics': diagnostics, 'plan_sha256': result['plan_sha256'],
                    'wall_seconds': time.perf_counter() - clip_begin}
                rt.atomic_json(folder / 'records' / f'{uid}.json', record); records.append(record)
                result['actual_count'] = index; rt.atomic_json(folder / 'generation-result.json', result)
                progress(status='running', completed=index, id=uid)
                del q, audio, assets, diagnostics
                gc.collect()
            rt.write_jsonl(folder / 'generated.jsonl', records)
            result.update(status='completed', wall_seconds=time.perf_counter() - begin,
                generated_sha256=rt.digest(folder / 'generated.jsonl'), source_metadata=source_metadata,
                encoder_metadata=encoder_metadata, backend=rt.backend_manifest(),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(), completed_at_utc=rt.stamp())
            rt.atomic_json(folder / 'generation-result.json', result)
            progress(status='completed', completed=info['count'])
    except Exception as error:
        result.update(status='failed_execution', error=str(error), error_type=type(error).__name__,
            wall_seconds=time.perf_counter() - begin, traceback=traceback.format_exc())
        rt.atomic_json(folder / 'generation-result.json', result); progress(status='failed', error=str(error)); raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['register', 'generate'])
    parser.add_argument('--cohort', choices=['pilot', 'validation'], default='pilot')
    args = parser.parse_args()
    print(register()) if args.stage == 'register' else generate(args.cohort)
