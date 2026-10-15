import io
import json
import tarfile
import pytest
import torch
from beyondwaveform.assets import safe_extract
from beyondwaveform.data import audit_disjoint, prepare_manifest
from beyondwaveform.evaluation import aggregate_errors, error_counts
from beyondwaveform.experiments import experiment_configs
from beyondwaveform.io import load_audio, read_jsonl, save_audio, write_jsonl
from beyondwaveform.selection import select_configuration


def test_corpus_errors_and_insertions_above_100():
    one = error_counts("one two", "one three")
    assert aggregate_errors([one])["error_percent"] == 50
    chinese = error_counts("你好", "你", "zh")
    assert aggregate_errors([chinese])["error_percent"] == 50
    insertions = error_counts("one", "one two three four")
    assert aggregate_errors([insertions])["error_percent"] == 300
    assert aggregate_errors([one, error_counts("a b c d", "a b c d")])["error_percent"] == pytest.approx(100/6)
    with pytest.raises(ValueError):
        error_counts("", "one")


def test_export_preserves_float_values(tmp_path):
    x = torch.linspace(-0.9, 0.9, 2000)
    path = tmp_path / "roundtrip.wav"
    result = save_audio(path, x)
    torch.testing.assert_close(load_audio(path), x, atol=0, rtol=0)
    assert result["format"] == "WAV/FLOAT"


def test_no_silent_truncation(tmp_path):
    path = tmp_path / "long.wav"
    save_audio(path, torch.ones(16001)*0.1)
    with pytest.raises(ValueError, match="truncation"):
        load_audio(path, max_seconds=1)


def test_split_overlap_by_audio_content(tmp_path):
    manifests = []
    for i in range(2):
        p = tmp_path / f"{i}.jsonl"
        write_jsonl(p, [{"id": str(i), "audio_sha256": "same", "role": "validation"}])
        manifests.append(p)
    with pytest.raises(ValueError, match="leakage"):
        audit_disjoint(manifests)


def test_unsafe_archive_rejected(tmp_path):
    path = tmp_path / "unsafe.tar"
    with tarfile.open(path, "w") as archive:
        info = tarfile.TarInfo("../escape")
        info.size = 1
        archive.addfile(info, io.BytesIO(b"x"))
    with pytest.raises(ValueError):
        safe_extract(path, tmp_path / "out")
    assert not (tmp_path / "escape").exists()


def candidate(tmp_path, name, error=60, score=3, scope="surrogate", role="validation"):
    p = tmp_path / f"{name}.json"
    p.write_text(json.dumps({"evaluation_scope": scope, "roles": [role], "manifest_sha256": "abc",
                             "evaluation_ids": ["one"], "target_artifacts": {}, "quality_provenance": {},
                             "metric": "WER", "normalization": "test",
                             "generation_run": name, "generation_fingerprint": name,
                             "summary": {"adversarial": {"error_percent": error,
                                         "quality": {"dnsmos": score, "nisqa": score, "utmos": score}}}}))
    return p


def test_selection_gate_and_maximum(tmp_path):
    good = candidate(tmp_path, "good", error=60)
    best = candidate(tmp_path, "best", error=70)
    bad = candidate(tmp_path, "bad", error=95, score=2.5)
    result = select_configuration([good, best, bad], tmp_path / "selection.json")
    assert result["selected"]["generation_run"] == "best"
    assert not result["candidates"][-1]["eligible"]


@pytest.mark.parametrize("scope,role", [("target", "validation"), ("surrogate", "test"), ("surrogate", "pilot")])
def test_selection_rejects_target_feedback_and_test_data(tmp_path, scope, role):
    source = candidate(tmp_path, "invalid", scope=scope, role=role)
    with pytest.raises(ValueError, match="validation"):
        select_configuration([source], tmp_path / "selection.json")


def test_missing_quality_metric_is_not_ignored(tmp_path):
    source = candidate(tmp_path, "missing")
    value = json.loads(source.read_text())
    del value["summary"]["adversarial"]["quality"]["utmos"]
    source.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="three"):
        select_configuration([source], tmp_path / "selection.json")


def test_paper_grids_and_ablation_switches():
    base = {"attack": {"rho": 0.1, "lambda_perc": 1.0}}
    grid = experiment_configs(base, "feature-grid")
    assert len(grid) == 16
    assert {c["attack"]["rho"] for _, c in grid} == {0.05, 0.1, 0.5, 1.0}
    ablations = dict(experiment_configs(base, "ablations"))
    assert ablations["without-attack"]["attack"]["attack"] is False
    assert ablations["without-perceptual"]["attack"]["lambda_perc"] == 0
    assert base["attack"] == {"rho": 0.1, "lambda_perc": 1.0}
    assert len(experiment_configs(base, "waveform-grid")) == 135


def test_selection_rejects_different_limits_of_same_manifest(tmp_path):
    a, b = candidate(tmp_path, "a"), candidate(tmp_path, "b")
    contents = json.loads(b.read_text())
    contents["evaluation_ids"] = ["two"]
    b.write_text(json.dumps(contents))
    with pytest.raises(ValueError, match="subsets"):
        select_configuration([a, b], tmp_path / "selection.json")
