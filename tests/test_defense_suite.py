from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_defense_suite import verify
from beyondwaveform.evaluation import aggregate_errors, error_counts
from beyondwaveform.io import atomic_json, digest, write_jsonl


@pytest.fixture
def saved_evaluation(tmp_path):
    run, evaluation = tmp_path / "run", tmp_path / "evaluation"
    target = {"kind": "whisper", "checkpoint": "frozen-small.pt"}
    ids = [f"clip-{i}" for i in range(999)]
    plan = {"generation_run": str(run), "ids": ids, "generation_fingerprint": "fixed"}
    records, rows = [], []
    for i, uid in enumerate(ids):
        # 330*21 + 669*20 = 20,310 words, matching the historical denominator.
        text = " ".join(["word"] * (21 if i < 330 else 20))
        audio = {v: {"path": str(tmp_path / "unused.wav"), "sha256": f"hash-{uid}-{v}"}
                 for v in ("clean", "reconstruction", "adversarial")}
        records.append({"id": uid, "text": text, "audio": audio})
        for variant, asset in audio.items():
            rows.append({"id": uid, "variant": variant, "reference": text, "hypothesis": text,
                         "audio_sha256": asset["sha256"], **error_counts(text, text)})
    write_jsonl(run / "generated.jsonl", records)
    atomic_json(run / "run.json", {"manifest_sha256": "manifest", "surrogate": {"sha256": "checkpoint"}})
    write_jsonl(evaluation / "utterances.jsonl", rows)
    summary = {v: {"count": 999, **aggregate_errors([r for r in rows if r["variant"] == v])}
               for v in ("clean", "reconstruction", "adversarial")}
    atomic_json(evaluation / "summary.json", {
        "target": target, "target_artifacts": {"sha256": "checkpoint"}, "evaluation_ids": ids,
        "roles": ["test"], "manifest_sha256": "manifest", "metric": "WER",
        "generation_fingerprint": "fixed", "generation_run": str(run),
        "normalization": "NFKC/lowercase/punctuation-to-space; Chinese whitespace removed",
        "utterances_sha256": digest(evaluation / "utterances.jsonl"), "summary": summary})
    return plan, evaluation, target, rows


def test_defense_audit_accepts_complete_controls(saved_evaluation):
    plan, evaluation, target, _ = saved_evaluation
    result = verify(plan, evaluation, target)
    assert result["rows"] == 2997
    assert result["summary"]["adversarial"]["reference_units"] == 20310


def test_defense_audit_rejects_duplicate_even_with_updated_digest(saved_evaluation):
    plan, evaluation, target, rows = saved_evaluation
    rows[-1] = deepcopy(rows[0])
    write_jsonl(evaluation / "utterances.jsonl", rows)
    summary = json.loads((evaluation / "summary.json").read_text())
    summary["utterances_sha256"] = digest(evaluation / "utterances.jsonl")
    atomic_json(evaluation / "summary.json", summary)
    with pytest.raises(ValueError, match="Duplicate/missing"):
        verify(plan, evaluation, target)


def test_defense_audit_rejects_invented_aggregate(saved_evaluation):
    plan, evaluation, target, _ = saved_evaluation
    path = evaluation / "summary.json"
    summary = json.loads(path.read_text())
    summary["summary"]["adversarial"]["error_percent"] = 75.43
    atomic_json(path, summary)
    with pytest.raises(ValueError, match="Incorrect corpus error_percent"):
        verify(plan, evaluation, target)
