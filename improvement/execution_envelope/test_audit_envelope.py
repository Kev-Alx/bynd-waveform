import copy
import importlib.util
from pathlib import Path
import numpy as np
import torch
from envelope_pgd import energy_envelope

spec = importlib.util.spec_from_file_location('saved_envelope_audit', Path(__file__).with_name('audit_envelope.py'))
audit = importlib.util.module_from_spec(spec); spec.loader.exec_module(audit)


def fixture():
    x = np.zeros(960, dtype=np.float32); x[600:] = .1
    _, meta = energy_envelope(torch.from_numpy(x))
    return x, {'final_metrics': {'energy_envelope': meta}, 'trace': [{'envelope_maximum_excess': 0.}]}


def test_independent_frame_audit_accepts_zero_delta():
    x, receipt = fixture(); audit.base.ISSUES.clear()
    audit.check_envelope(x, x.copy(), receipt, 'test')
    assert audit.base.ISSUES == []


def test_independent_audit_detects_even_tiny_noise_in_exact_silence():
    x, receipt = fixture(); y = x.copy(); y[100] = 1e-9
    audit.base.ISSUES.clear(); audit.check_envelope(x, y, receipt, 'test')
    assert any('pointwise' in issue for issue in audit.base.ISSUES)
    assert any('exactly silent' in issue for issue in audit.base.ISSUES)


def test_independent_audit_detects_changed_frame_recipe():
    x, receipt = fixture(); receipt['final_metrics']['energy_envelope']['frame_caps'][-1] *= 2
    audit.base.ISSUES.clear(); audit.check_envelope(x, x.copy(), receipt, 'test')
    assert any('caps differ' in issue for issue in audit.base.ISSUES)
