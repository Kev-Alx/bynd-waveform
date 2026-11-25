"""New 48-point development contract; all historical registrations are intact."""
from pathlib import Path
import execution_runtime as rt

OUT = Path(__file__).resolve().parent
RUN = OUT / 'execution48'
VARIANTS = ('clean', 'reconstruction', 'fv_quality', 'r3_initial', 'r3_trained', 'pgd_full', 'pgd_nll')


def register():
    old = rt.read(rt.RUN / 'plan.json')
    rt.check(old)
    cohort = RUN / 'development-48.jsonl'
    receipt = RUN / 'development-48.receipt.json'
    selection = rt.read(receipt)
    rows = rt.read_jsonl(cohort)
    if len(rows) != 48 or len({r['id'] for r in rows}) != 48:
        raise ValueError('Every new pilot must contain exactly48 unique development utterances')
    if selection['manifest_sha256'] != rt.digest(cohort):
        raise ValueError('Changed frozen pilot48')
    training = rt.read(rt.RUN / 'r3-waveform-autograd-repair/training.json')
    calibration = rt.RUN / 'r3-waveform-autograd-repair/penalty-calibration.json'
    checkpoint = Path(training['selected_checkpoint'])
    files = [OUT / n for n in ('runtime48.py', 'pilot48_runner.py', 'source_evaluate48.py', 'test_pilot48_scope.py',
        'waveform_pgd.py', 'test_waveform_pgd.py', 'r3_waveform.py', 'r3_encoder_repair.py',
        'r3_real_runner_v2.py', 'execution_runtime.py', 'supervise_pilot48.py', 'postprocess_pilot48.py')]
    files += [cohort, receipt, calibration, checkpoint, rt.RUN / 'plan.json',
              RUN / 'diagnostics_saved_r3.py', RUN / 'waveform-pgd-tests.log', RUN / 'pilot48-tests.log',
              OUT / '07_pilot48_protocol.md', RUN / 'audit_pilot48.py',
              RUN / 'test_audit_pilot48.py', RUN / 'audit-tests.log',
              OUT.parent / 'scripts/run_logged.py']
    value = {'created_at_utc': rt.stamp(), 'phase': 'source_only_development48',
        'authorization': 'User requested continue, and48 data points for pilot tests',
        'pilot_count': 48, 'development_manifest': str(cohort),
        'manifest_sha256': rt.digest(cohort), 'prior_plan': str(rt.RUN / 'plan.json'),
        'target_models_evaluated': [], 'confirmation_or_final_inference': False,
        'maximum_cycle_wall_seconds': 7200, 'original_initial_gpu_envelope_hours': 6,
        'variants': list(VARIANTS), 'seed': 11,
        'pgd': {'conditions': ['full_r3', 'nll_only'], 'steps': 50, 'initialization': 'zero',
            'step_size': 2 * .003 / 50, 'step_rule': 'projected descent of full loss using sign of waveform gradient',
            'linf': .003, 'rms_limit': .003, 'peak_limit': 1, 'selected_iterate': 'final50',
            'weights': rt.read(calibration)['weights'], 'new_coefficient_search': False},
        'fv_quality': {'steps': 50, 'optimizer': 'Adam', 'learning_rate': .02,
            'rho': .1, 'lambda_perc': 1, 'alpha_hf': 1, 'initialization': 'zero',
            'objective': 'original uppercase lexicalSUM NLL with full TV/HF'},
        'q_policy': 'Regenerate all48 clean WavLM6 features using the same strict transform; no parity with historical numerical feature cache claimed',
        'generator_checkpoint': str(checkpoint), 'generator_sha256': rt.digest(checkpoint),
        'generator_selection': 'Already frozen from prior24; no new checkpoint selection on48',
        'failure_policy': 'Retain all48; mathematical PGD failure saves and scores its last feasible attempted waveform with explicit failure flag; infrastructure errors stop the job',
        'scientific_limits': 'Exposed development only; no human/transfer claim. FV radius and waveform amplitude are different constraints, compared on measured quality, not presumed equal.',
        'bindings': {str(p): rt.digest(p) for p in files}}
    path = RUN / 'plan.json'
    if path.exists():
        saved = rt.read(path)
        value['created_at_utc'] = saved['created_at_utc']
        if saved != value:
            raise ValueError('Preserve registered48-point cycle; changed source requires a named new variant')
        return saved
    rt.atomic_json(path, value)
    path.chmod(0o444)
    return value


def check():
    plan = rt.read(RUN / 'plan.json')
    for path, expected in plan['bindings'].items():
        if rt.digest(path) != expected:
            raise ValueError(f'Changed registered48-point input: {path}')
    if plan['pilot_count'] != 48 or plan['target_models_evaluated'] or plan['confirmation_or_final_inference']:
        raise ValueError('Changed pilot count or source-only scope')
    return plan
