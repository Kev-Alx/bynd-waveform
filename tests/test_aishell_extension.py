import copy
import hashlib
import json
from pathlib import Path
import sys
import time

import pytest
import torch

from beyondwaveform.io import atomic_json, digest, save_audio, write_jsonl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from manage_aishell_epoch3 import original_five_thousand
from run_aishell_epoch3_extended import ExtendedWorker


def test_original_union_keeps_the_unfinished_2000_before_the_original_3000():
    full = [{"id": f"audio-{i}", "audio_sha256": hashlib.sha256(str(i).encode()).hexdigest(), "text": "你好"} for i in range(5000)]
    first, following = full[:2000], full[2000:]
    selected = original_five_thousand(first, following, list(reversed(full)))
    assert selected[:2000] == first and selected[2000:] == following
    overlap = [first[0], *following[1:]]
    with pytest.raises(AssertionError, match="overlap"):
        original_five_thousand(first, overlap, full)
    changed = copy.deepcopy(first)
    changed[0]["text"] = "changed reference"
    with pytest.raises(AssertionError, match="reference rows differ"):
        original_five_thousand(changed, following, full)


@pytest.fixture
def extension(tmp_path):
    parent, output = tmp_path / "parent", tmp_path / "expanded"
    parent.mkdir();output.mkdir()
    audio = tmp_path / "source.wav"
    asset = save_audio(audio, torch.full((1024,), .1))
    records = [{"id": uid, "audio": str(audio), "audio_sha256": asset["sha256"], "text": "你好"} for uid in ["old-1", "old-2", "new-1"]]
    config = {"attack": {"steps": 1, "rho": .5, "lambda_perc": .1}}
    atomic_json(parent / "config.json", config)
    (output / "config.json").write_bytes((parent / "config.json").read_bytes())
    write_jsonl(parent / "manifest.jsonl", records[:2])
    write_jsonl(output / "manifest.jsonl", records)
    parent_plan = {"checkpoint_sha256": "same-checkpoint", "input_hashes": {str(parent / "config.json"): digest(parent / "config.json")}}
    atomic_json(parent / "plan.json", parent_plan)
    plan = {"checkpoint_sha256": "same-checkpoint", "resume_parent": {"plan_path": str(parent / "plan.json"), "plan_sha256": digest(parent / "plan.json")}}
    atomic_json(output / "plan.json", plan)
    return parent, output, records, asset


def test_verified_parent_records_are_reused_without_rewriting_their_fingerprint(extension):
    parent, output, rows, asset = extension
    worker = ExtendedWorker(output, time.time() + 60, device="cpu")
    record = {"id": "old-1", "text": "你好", "input_audio_sha256": asset["sha256"],
              "plan_sha256": digest(parent / "plan.json"), "audio": {name: asset for name in ["clean", "reconstruction", "adversarial"]},
              "trace": [{"step": 1, "radius_after_update": .1}]}
    before = json.dumps(record, sort_keys=True)
    worker.validate_generation(record)
    worker.generated[record["id"]] = record
    item = {"id": "old-1", "variant": "clean", "reference": "你好", "hypothesis": "你好",
            "plan_sha256": record["plan_sha256"], "target_sha256": "same-checkpoint", "audio_sha256": asset["sha256"],
            "cer": {"substitutions": 0, "deletions": 0, "insertions": 0, "reference_units": 2,
                    "reference_normalized": "你好", "hypothesis_normalized": "你好"},
            "wer": {"substitutions": 0, "deletions": 0, "insertions": 0, "reference_units": 1},
            "reference_word_tokens": ["你好"], "hypothesis_word_tokens": ["你好"]}
    worker.validate_score(item)
    assert json.dumps(record, sort_keys=True) == before
    assert item["plan_sha256"] == digest(parent / "plan.json")
    for invalid in [{**record, "plan_sha256": "unknown"}, {**record, "id": "new-1"}]:
        with pytest.raises(AssertionError, match="Untrusted cached protocol"):
            worker.validate_generation(invalid)


@pytest.mark.parametrize("change", ["checkpoint", "settings", "order"])
def test_extension_rejects_changed_models_settings_and_original_processing_order(extension, change):
    parent, output, rows, asset = extension
    if change == "checkpoint":
        plan = json.loads((output / "plan.json").read_text());plan["checkpoint_sha256"] = "different"
        atomic_json(output / "plan.json", plan)
    elif change == "settings":
        atomic_json(output / "config.json", {"attack": {"steps": 1, "rho": .5, "lambda_perc": .2}})
    else:
        write_jsonl(output / "manifest.jsonl", [rows[1], rows[0], rows[2]])
    with pytest.raises(AssertionError):
        ExtendedWorker(output, time.time() + 60, device="cpu")
