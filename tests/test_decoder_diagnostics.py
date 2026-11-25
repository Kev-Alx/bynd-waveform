import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_decoder_diagnostics as driver


def test_frozen_decoder_diagnostic_rejects_changed_input(tmp_path, monkeypatch):
    item = tmp_path / "native-decoding.py"
    item.write_text("frozen primary source")
    plan = {"runtime": {}, "count": 24, "variants": ["clean", "reconstruction", "adversarial"],
        "prefixes": ["no_timestamps", "initial_timestamp"],
        "frozen_files": {str(item): driver.digest(item)}, "parent_plan": "parent.json"}
    monkeypatch.setattr(driver.protocol, "runtime", lambda: {})
    monkeypatch.setattr(driver.protocol, "read", lambda path: {})
    monkeypatch.setattr(driver.parent_protocol, "check", lambda parent: None)
    driver.check(plan)
    item.write_text("changed primary source")
    with pytest.raises(ValueError, match="Changed frozen diagnostic input"):
        driver.check(plan)


def test_diagnostic_scope_cannot_expand_silently(monkeypatch):
    plan = {"runtime": {}, "count": 24, "variants": ["clean", "reconstruction", "adversarial"],
        "prefixes": ["no_timestamps", "initial_timestamp"], "frozen_files": {}, "parent_plan": "parent.json"}
    monkeypatch.setattr(driver.protocol, "runtime", lambda: {})
    monkeypatch.setattr(driver.protocol, "read", lambda path: {})
    monkeypatch.setattr(driver.parent_protocol, "check", lambda parent: None)
    for key, value in (("count", 100), ("variants", ["adversarial"]), ("prefixes", ["no_timestamps"])):
        changed = copy.deepcopy(plan)
        changed[key] = value
        with pytest.raises(ValueError, match="Changed diagnostic scope"):
            driver.check(changed)
