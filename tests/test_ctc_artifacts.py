"""Cached CTC weights must still match their pinned source and local receipts."""
import hashlib
import json
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

import pytest
import yaml
from beyondwaveform.io import digest
from scripts.verify_test_run import verify_ctc_target


@pytest.fixture
def snapshot(tmp_path, monkeypatch):
    import huggingface_hub
    revision = "a" * 40
    folder = tmp_path / "snapshots" / revision
    folder.mkdir(parents=True)
    contents = {"config.json": b'{"model_type":"wav2vec2"}', "pytorch_model.bin": b"fixture weights"}
    siblings = []
    for name, value in contents.items():
        (folder / name).write_bytes(value)
        blob = hashlib.sha1(f"blob {len(value)}\0".encode() + value).hexdigest()
        lfs = SimpleNamespace(sha256=hashlib.sha256(value).hexdigest()) if name.endswith(".bin") else None
        siblings.append(SimpleNamespace(rfilename=name, size=len(value), blob_id=blob, lfs=lfs))
    class FakeAPI:
        def model_info(self, repo, revision=None, files_metadata=False):
            assert revision == "a" * 40 and files_metadata
            return SimpleNamespace(sha=revision, siblings=siblings)
    monkeypatch.setattr(huggingface_hub, "HfApi", FakeAPI)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *a, **k: str(folder))
    output = tmp_path / "target.yaml"
    monkeypatch.setattr(sys, "argv", ["download_ctc.py", "--model-id", "fixture/ctc", "--revision", revision,
                                      "--output", str(output), "--cache-dir", str(tmp_path / "cache")])
    return folder, output


def download():
    runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/download_ctc.py"), run_name="__main__")


def test_verified_download_preserves_readonly_receipts_on_resume(snapshot):
    folder, output = snapshot
    download()
    receipt = Path(str(output) + ".receipt.json")
    config = yaml.safe_load(output.read_text())
    verify_ctc_target(config)
    assert config["artifact_hashes"]["pytorch_model.bin"] == digest(folder / "pytorch_model.bin")
    assert json.loads(receipt.read_text())["upstream_files"]["pytorch_model.bin"]["lfs_sha256"] == digest(folder / "pytorch_model.bin")
    before = digest(output), digest(receipt)
    output.chmod(0o444)
    receipt.chmod(0o444)
    download()
    assert (digest(output), digest(receipt)) == before


def test_cached_weight_tampering_is_rejected_against_upstream_digest(snapshot):
    folder, output = snapshot
    file = folder / "pytorch_model.bin"
    file.write_bytes(b"corrupt weights")  # same length; size alone is insufficient
    with pytest.raises(ValueError, match="LFS SHA256 mismatch"):
        download()
    assert not output.exists()


def test_auditor_rejects_local_tampering_and_mutable_revision(snapshot):
    folder, output = snapshot
    download()
    config = yaml.safe_load(output.read_text())
    (folder / "config.json").write_text('{"model_type":"changed"}')
    with pytest.raises(ValueError, match="CTC artifact changed"):
        verify_ctc_target(config)
    config["revision"] = "main"
    with pytest.raises(ValueError, match="Unpinned CTC"):
        verify_ctc_target(config)
