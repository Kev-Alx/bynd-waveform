from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_optimizer_diagnostics as driver


def test_numerical_study_rejects_new_scope_before_model_work(monkeypatch):
    plan = {"count": 3, "repeats": 3, "modes": list(driver.MODES), "adam_capture_steps": [1, 10, 25, 50]}
    for field, value in (("count", 24), ("repeats", 1), ("modes", ["auto"]), ("adam_capture_steps", [50])):
        changed = {**plan, field: value}
        with pytest.raises(ValueError, match="Changed diagnostic scope"):
            driver.check(changed)
