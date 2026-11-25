"""CPU saved-artifact audit; independently reconstructs all clean-energy caps."""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import numpy as np
import soundfile as sf
import torch

ROOT = Path('/workspace/improvement')
CYCLE = ROOT / 'execution_envelope'
VARIANTS = ('clean', 'fv50', 'global_full50', 'global_nll50', 'envelope_nll50',
            'envelope_full50', 'attenuated_global_nll50')


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


base = load('independent_refinement_audit', ROOT / 'execution_refine/audit_refinement.py')
pgd = load('independent_pgd_audit', ROOT / 'execution48/audit_pilot48.py')
base.CYCLE, base.VARIANTS = CYCLE, VARIANTS
pgd.ISSUES, pgd.CHECKED_FILES = base.ISSUES, base.HASHES


def check_envelope(clean, adversarial, receipt, label):
    clean, adversarial = clean.astype(np.float64), adversarial.astype(np.float64)
    starts = list(range(0, len(clean) - 399, 160))
    if starts[-1] != len(clean) - 400: starts.append(len(clean) - 400)
    bounds = np.full(len(clean), .003)
    caps = []
    for start in starts:
        cap = min(.003, np.sqrt(np.sum(clean[start:start + 400] ** 2) / 400) / np.sqrt(1000))
        caps.append(float(cap)); bounds[start:start + 400] = np.minimum(bounds[start:start + 400], cap)
    saved = receipt['final_metrics']['energy_envelope']
    base.verify(saved['frame_samples'] == 400 and saved['hop_samples'] == 160 and saved['local_snr_db'] == 30,
        label + ': different registered frame rule')
    base.verify(saved['starts'] == starts and len(saved['frame_caps']) == len(caps), label + ': frame coverage differs')
    base.verify(np.allclose(saved['frame_caps'], caps, atol=1e-15, rtol=1e-12), label + ': clean-derived caps differ')
    base.close(saved['minimum_bound'], float(bounds.min()), label + ': minimum cap differs', atol=1e-15)
    base.close(saved['median_bound'], float(np.median(bounds)), label + ': median cap differs', atol=1e-15)
    base.close(saved['maximum_bound'], float(bounds.max()), label + ': maximum cap differs', atol=1e-15)
    base.verify(saved['zero_bound_samples'] == int((bounds == 0).sum()), label + ': zero-cap inventory differs')
    residual = adversarial - clean
    # Account for float32 bound rounding and waveform addition at the sample's
    # actual magnitude; this allowance stays exactly zero at silent zero caps.
    tolerance = 4 * np.finfo(np.float32).eps * (np.abs(clean) + bounds)
    base.verify(np.all(np.abs(residual) <= bounds + tolerance), label + ': pointwise clean-energy cap violation')
    ratios = []
    for start in starts:
        energy = float(np.sum(clean[start:start + 400] ** 2))
        actual = float(np.sum(residual[start:start + 400] ** 2))
        allowed = float(np.sum((bounds[start:start + 400] + tolerance[start:start + 400]) ** 2))
        base.verify(actual <= allowed + 1e-30, label + ': frame energy cap violation')
        if energy > 0: ratios.append(actual / energy)
        else: base.verify(actual == 0, label + ': perturbed an exactly silent frame')
    for row in receipt['trace']:
        base.verify(row['envelope_maximum_excess'] <= 4 * np.finfo(np.float32).eps,
            label + ': recorded envelope update violated bounds')
    return {'frames': len(starts), 'largest_frame_residual_to_clean_energy': max(ratios, default=0),
        'maximum_pointwise_excess': float(np.max(np.abs(residual) - bounds)),
        'roundoff': '4*float32eps*(abs(clean_sample)+bound_sample), zero at true silent caps'}


def audit(cohort_name):
    base.ISSUES.clear(); base.HASHES.clear()
    plan = base.read(CYCLE / 'plan.json'); previous = base.read(ROOT / 'execution/plan.json')
    base.verify(tuple(plan['variants']) == VARIANTS and plan['source_only'] and not plan['target_models_evaluated']
        and not plan['confirmation_or_final_inference'], 'Changed source-only registration')
    for path, expected in plan['bindings'].items(): base.bind(path, expected)
    for path, expected in previous['frozen_files'].items(): base.bind(path, expected)
    info = plan['cohorts'][cohort_name]; folder = CYCLE / cohort_name
    cohort = base.rows(info['manifest']); count = 48 if cohort_name == 'pilot' else 76
    base.bind(info['manifest'], info['sha256'])
    base.verify(len(cohort) == len({r['id'] for r in cohort}) == info['count'] == count, 'Changed cohort')
    generation = base.read(folder / 'generation-result.json')
    base.verify(generation['status'] == 'completed' and generation['count'] == generation['actual_count'] == count,
        'Generation incomplete')
    base.verify(generation['cohort'] == cohort_name and generation['source_only'] and not generation['target_models_evaluated']
        and not generation['failures'], 'Changed generation scope/failures')
    base.verify(generation['plan_sha256'] == base.sha(CYCLE / 'plan.json'), 'Generation plan changed')
    historical_generation = base.read(ROOT / 'execution48/mechanism-pilot48/generation-result.json')
    base.verify(generation['backend'] == historical_generation['backend'], 'Changed strict generation backend')
    base.verify(generation['encoder_metadata'] == historical_generation['encoder_metadata'], 'Changed content model')
    if cohort_name == 'validation':
        freeze = base.read(CYCLE / 'validation-freeze.json')
        base.verify(freeze['status'] == 'validate' and freeze['candidate'] == 'envelope_nll50' and freeze['control'] == 'fv50',
            'Missing qualifying candidate freeze')
        for path, expected in freeze['bindings'].items(): base.bind(path, expected)
        base.verify(datetime.fromisoformat(freeze['created_at_utc']) < datetime.fromisoformat(generation['started_at_utc']),
            'Validation began before candidate freeze')
    base.bind(folder / 'generated.jsonl', generation['generated_sha256'])
    records = base.rows(folder / 'generated.jsonl')
    base.verify([r['id'] for r in records] == [r['id'] for r in cohort], 'Incomplete/reordered generated cohort')
    old = {r['id']: r for r in base.rows(ROOT / 'execution48/mechanism-pilot48/generated.jsonl')}
    old_fv = {r['id']: r for r in base.rows(ROOT / 'execution_refine/pilot/generated.jsonl')}
    checks = []
    for record, row in zip(records, cohort):
        uid = row['id']; tag = f'{cohort_name}/{uid}'
        base.verify(base.read(folder / 'records' / f'{uid}.json') == record, tag + ': individual record differs')
        base.verify(record['text'] == row['text'] and record['speaker'] == row['speaker']
            and record['input_audio_sha256'] == row['audio_sha256'], tag + ': reference/input changed')
        base.verify(record['role'] == cohort_name and record['count'] == count and record['source_only']
            and record['plan_sha256'] == base.sha(CYCLE / 'plan.json'), tag + ': record scope differs')
        base.verify(set(record['audio']) == set(VARIANTS), tag + ': missing variants')
        base.bind(row['audio'], row['audio_sha256'])
        source, rate = sf.read(row['audio'], dtype='float32')
        base.verify(rate == 16000 and source.shape == (row['samples'],), tag + ': changed source format')
        audio = {v: base.waveform(record['audio'][v], tag + '/' + v) for v in VARIANTS}
        base.verify(np.array_equal(source, audio['clean']), tag + ': transformed clean audio')
        q = base.tensor(record['diagnostics']['q'], tag + '/q')
        prior = old[uid] if cohort_name == 'pilot' else None
        if prior:
            base.verify(torch.equal(q, base.tensor(prior['clean_features'], tag + '/prior-q')), tag + ': changed fixed clean features')
            base.verify(record['audio']['fv50'] == old_fv[uid]['audio']['fv50']
                and record['diagnostics']['fv50'] == old_fv[uid]['diagnostics']['fv50'], tag + ': changed frozen FV control')
            for variant, old_variant in [('global_full50', 'pgd_full'), ('global_nll50', 'pgd_nll')]:
                base.verify(record['audio'][variant] == prior['audio'][old_variant]
                    and record['diagnostics'][variant] == prior['diagnostics']['pgd'][old_variant], tag + ': changed global control')
        base.audit_fv(record['diagnostics']['fv50'], q, uid, 'fv50', old=prior)
        for variant in ('global_full50', 'global_nll50', 'envelope_nll50', 'envelope_full50'):
            actual = base.physical(audio['clean'], audio[variant], tag + '/' + variant)
            diag = record['diagnostics'][variant]; base.bind(diag['receipt'], diag['sha256'])
            receipt = base.read(diag['receipt'])
            base.verify(diag['status'] == receipt['status'] == 'completed', tag + ': incomplete attack')
            pgd.audit_pgd(diag['receipt'], uid, 'pgd_nll' if 'nll' in variant else 'pgd_full', actual,
                {'stft': 65.08367870148994, 'content': 186.6123439723859})
            if variant.startswith('envelope'):
                checks.append({'id': uid, 'variant': variant, **check_envelope(audio['clean'], audio[variant], receipt, tag + '/' + variant)})
        diag = record['diagnostics']['attenuation']; clean = audio['clean'].astype(np.float64)
        global_residual = audio['global_nll50'].astype(np.float64) - clean
        target = float(np.sqrt(np.mean((audio['envelope_nll50'].astype(np.float64) - clean) ** 2)))
        reference = float(np.sqrt(np.mean(global_residual ** 2)))
        base.verify(0 <= diag['uniform_attenuation'] <= 1 and not diag['selection_uses_decode_or_quality'], tag + ': attenuation control changed')
        base.close(diag['target_rms'], target, tag + ': wrong target RMS', atol=1e-12)
        base.close(diag['reference_rms'], reference, tag + ': wrong reference RMS', atol=1e-12)
        base.close(diag['uniform_attenuation'], target / reference if reference else 0., tag + ': wrong uniform scaling', atol=1e-12)
        expected = (clean + diag['uniform_attenuation'] * global_residual).astype(np.float32)
        base.verify(np.allclose(audio['attenuated_global_nll50'], expected, atol=2e-8, rtol=0), tag + ': uniform attenuation waveform differs')
        actual = base.physical(audio['clean'], audio['attenuated_global_nll50'], tag + '/attenuation')
        base.close(actual['rms'], target, tag + ': unmatched residual RMS', atol=2e-9)
        base.close(diag['actual_rms'], actual['rms'], tag + ': wrong actual attenuation RMS', atol=1e-12)
    scores = base.audit_scores(folder, cohort, records, generation, plan, previous)
    value = {'status': 'fail' if base.ISSUES else 'pass', 'checked_at_utc': datetime.now(timezone.utc).isoformat(),
        'cohort': cohort_name, 'count': count, 'source_only': True, 'target_models_evaluated': [],
        'issues': base.ISSUES, 'hashes_checked': len(base.HASHES), 'envelope_checks': checks,
        'independent_score_recount': scores, 'scope': 'Saved audio/caps/50-step receipts/actual counters/source scores; no neural gradients or human/transfer claims'}
    (folder / 'audit.json').write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: value[k] for k in ('status', 'cohort', 'count', 'issues')}, indent=2))
    return bool(base.ISSUES)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--cohort', choices=['pilot', 'validation'], required=True)
    args = parser.parse_args()
    raise SystemExit(audit(args.cohort))
