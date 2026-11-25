"""Frozen source-only refinement experiment and conditional larger validation."""
from pathlib import Path
import execution_runtime as rt

OUT = Path(__file__).resolve().parent
RUN = OUT / 'execution_refine'
VARIANTS = ('clean', 'fv50', 'full50', 'weighted100', 'constrained100', 'fv100')


def register():
    previous = rt.read(rt.RUN / 'plan.json')
    rt.check(previous)
    cohorts = {'pilot': {'count': 48, 'manifest': str(OUT / 'execution48/development-48.jsonl')},
               'validation': {'count': 76, 'manifest': str(RUN / 'validation-76.jsonl')}}
    for value in cohorts.values():
        rows = rt.read_jsonl(value['manifest'])
        if len(rows) != value['count'] or len({r['id'] for r in rows}) != value['count']:
            raise ValueError('Changed registered cohort size/identity')
        value['sha256'] = rt.digest(value['manifest'])
    files = [OUT / name for name in ('runtime_refine.py', 'refine_runner.py', 'source_evaluate_refine.py',
        'constrained_refine.py', 'test_constrained_refine.py', 'supervise_refinement.py',
        'postprocess_refinement.py', 'r3_waveform.py', 'r3_encoder_repair.py', 'r3_real_runner_v2.py',
        'waveform_pgd.py', 'execution_runtime.py', 'supervise_pilot48.py', 'test_refinement_scope.py', '09_refinement_protocol.md')]
    files += [RUN / name for name in ('validation-76.jsonl', 'validation-76.receipt.json',
        'create_validation76.py', 'prior-work.md', 'core-tests.log', 'audit_refinement.py',
        'test_audit_refinement.py', 'audit-tests.log', 'integration-tests.log',
        'compute-ledger.json', 'compute-ledger.md', 'build_compute_ledger.py')]
    files += [Path(value['manifest']) for value in cohorts.values()]
    files += [OUT / 'execution48/audit_pilot48.py', OUT / 'execution48/plan.json', OUT / 'execution48/mechanism-pilot48/generated.jsonl',
        OUT / 'execution48/mechanism-pilot48/source-summary.json', rt.RUN / 'plan.json',
        OUT.parent / 'scripts/run_logged.py']
    value = {'created_at_utc': rt.stamp(), 'cohorts': cohorts, 'variants': list(VARIANTS),
        'source_only': True, 'target_models_evaluated': [], 'confirmation_or_final_inference': False,
        'maximum_cycle_wall_seconds': 12600, 'original_initial_gpu_envelope_hours': 6,
        'authorization': 'User requested continued paper improvement and larger validation/full runs when justified; pilots48',
        'initialization': 'Full-R3 waveform PGD50; pilot reuses frozen prior output, validation generates the identical algorithm',
        'method': {'additional_updates': 50, 'step_size': .00012, 'epsilon': .003, 'rms_limit': .003,
            'inward_fraction': .01, 'backtracking_trials': 10,
            'quality_caps': 'Per-example fixed initial full50 STFT and WavLM6 distortion values',
            'tolerances': 'Caps max(1e-12,32*finfo.eps*abs(cap)); source max(1e-7,8*finfo.eps*abs(NLL))',
            'direction': 'Simultaneous2-plane Euclidean projection of box-free source-gradient sign; fixed inward margin, zero-margin fallback; positive Linf normalization; actual nonlinear checks',
            'selected_iterate': 'last feasible after50 attempted additional steps; no decoding/MOS iterate selection'},
        'fv': {'learning_rate': .02, 'steps': 100, 'rho': .1, 'full_original_tv_hf': True,
            'save_steps': [50, 100], 'pilot_step50': 'Require exact tensor equality with frozen FV50 delta'},
        'validation_rule': 'Freeze one quality-admissible candidate with either primary G1 signal or a>=3pp source gain vs full50 without>0.05UTMOS loss (or>=0.05UTMOS gain with<=2pp source loss); primary superiority is separately labelled. If none, stop rather than repeat pilots.',
        'selection_rule': 'Choose highest source WER among qualifying constrained100/weighted100, then higher UTMOS, then constrained100; select admissible FV50/FV100 control by highest source WER, then UTMOS, then FV50. Freeze before validation.',
        'scientific_limits': 'Source-only exposed-development validation; no human or transfer claim; waveform and feature budgets differ; match source-backward counts but disclose extra line-search forwards.',
        'bindings': {str(path): rt.digest(path) for path in files}}
    path = RUN / 'plan.json'
    if path.exists():
        saved = rt.read(path); value['created_at_utc'] = saved['created_at_utc']
        if saved != value:
            raise ValueError('Preserve registered refinement source and configuration')
        return saved
    rt.atomic_json(path, value); path.chmod(0o444)
    return value


def check():
    plan = rt.read(RUN / 'plan.json')
    for path, expected in plan['bindings'].items():
        if rt.digest(path) != expected:
            raise ValueError(f'Changed registered refinement input: {path}')
    if not plan['source_only'] or plan['target_models_evaluated'] or plan['confirmation_or_final_inference']:
        raise ValueError('Changed source-only scope')
    return plan


def validation_freeze():
    freeze = rt.read(RUN / 'validation-freeze.json')
    if freeze['status'] != 'validate':
        raise ValueError('Larger validation did not earn its registered continuation')
    for path, expected in freeze['bindings'].items():
        if rt.digest(path) != expected:
            raise ValueError('Changed selection after candidate freeze')
    return freeze
