from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from parallel_jobs import GpuAdmission
from prepare_mandarin_independent import split_speakers


def test_three_speaker_pools_are_deterministic_and_preserve_official_dev():
    rows = [{"speaker": f"S{i:02}", "id": f"u{i}-{j}", "split": "dev", "corpus": "aishell", "role": "finetune"}
            for i in range(20) for j in range(3)]
    pools = split_speakers(rows)
    assert pools == split_speakers(rows)
    speakers = [{r["speaker"] for r in p} for p in pools.values()]
    assert all(not a & b for i, a in enumerate(speakers) for b in speakers[i+1:])
    assert sum(len(p) for p in pools.values()) == len(rows)
    assert all(r["role"] == "finetune" for r in pools["finetune"])
    assert all(r["role"] == "validation" for r in pools["checkpoint-validation"] + pools["attack-reserve"])
    assert all(r["role"] == "finetune" for r in rows)


def test_test_audio_cannot_enter_development_pools():
    rows = [{"speaker": str(i), "split": "test", "corpus": "aishell"} for i in range(5)]
    with pytest.raises(ValueError, match="development"):
        split_speakers(rows)


def test_auxiliary_gpu_admission_preserves_generation_priority():
    assert not GpuAdmission.allowed("running", "generation", 5)
    assert not GpuAdmission.allowed("running", "evaluation", 98)
    assert GpuAdmission.allowed("running", "evaluation", 40)
    assert GpuAdmission.allowed("complete", "complete", 98)
    assert not GpuAdmission.allowed("failed", "generation", 5)
