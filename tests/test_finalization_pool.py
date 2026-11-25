import json
from pathlib import Path
import random

import pytest

from beyondwaveform.io import digest, read_jsonl, write_jsonl
from scripts.finalization_pool import prepare_pool


def index_fixture(tmp_path, count=12):
    rows = []
    for i in range(count):
        audio = tmp_path / f"audio-{i}.flac"
        audio.write_bytes(f"fixed source audio {i}".encode())
        rows.append({"id": f"u{i:03d}", "text": "FIXED REFERENCE", "audio": str(audio),
            "audio_sha256": digest(audio), "corpus": "librispeech", "split": "dev-clean",
            "role": "pilot", "language": "en", "speaker": str(i % 3), "duration": 1.0 + i / 10})
    path = tmp_path / "index.jsonl"
    write_jsonl(path, rows)
    return path, rows


def test_pool_excludes_union_of_ids_and_audio_with_overlapping_manifests(tmp_path):
    index, rows = index_fixture(tmp_path, 10)
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    write_jsonl(a, [{"id": rows[0]["id"], "audio_sha256": rows[1]["audio_sha256"]}])
    write_jsonl(b, [{"id": rows[0]["id"], "audio_sha256": rows[2]["audio_sha256"]},
                   {"id": "outside-index", "audio_sha256": rows[1]["audio_sha256"]}])
    pool = prepare_pool(tmp_path / "pool", index, [a, b], count=7)
    assert pool["ids"] == [row["id"] for row in rows[3:]]
    receipt = json.loads(Path(pool["receipt"]).read_text())
    assert receipt["exclusion_counts"] == {"rows_across_manifests": 3, "unique_ids": 2,
        "unique_audio_sha256": 2, "index_rows_blocked_by_id": 1,
        "index_rows_blocked_by_audio_sha256": 2, "index_rows_blocked_by_both": 0,
        "index_rows_blocked_union": 3, "eligible_index_rows": 7}
    assert receipt["index"]["sha256"] == digest(index)
    assert receipt["manifest_sha256"] == digest(pool["manifest"]) == pool["sha256"]
    assert receipt["speakers"]["count"] == 3
    assert receipt["duration_seconds"]["total"] == pytest.approx(sum(row["duration"] for row in rows[3:]))
    assert [row["audio_sha256"] for row in receipt["selected"]] == [row["audio_sha256"] for row in rows[3:]]
    assert all(row["role"] == "validation" for row in read_jsonl(pool["manifest"]))
    assert prepare_pool(tmp_path / "pool", index, [b, a], count=7) == pool


def test_sampling_is_exact_sorted_id_algorithm_and_input_order_independent(tmp_path):
    index, rows = index_fixture(tmp_path)
    excluded = tmp_path / "excluded.jsonl"
    write_jsonl(excluded, [rows[0]])
    pool = prepare_pool(tmp_path / "one", index, [excluded], seed=20261019, count=5)
    expected = sorted(row["id"] for row in random.Random(20261019).sample(rows[1:], 5))
    assert pool["ids"] == expected
    reversed_index = tmp_path / "reversed.jsonl"
    write_jsonl(reversed_index, list(reversed(rows)))
    other = prepare_pool(tmp_path / "two", reversed_index, [excluded], seed=20261019, count=5)
    assert other["ids"] == pool["ids"]
    assert other["sha256"] == pool["sha256"]
    assert read_jsonl(index) == rows


def test_insufficient_pool_creates_no_partial_evidence(tmp_path):
    index, rows = index_fixture(tmp_path, 3)
    excluded = tmp_path / "excluded.jsonl"
    write_jsonl(excluded, [rows[0]])
    with pytest.raises(ValueError, match="only 2 eligible"):
        prepare_pool(tmp_path / "pool", index, [excluded], count=3)
    assert not (tmp_path / "pool").exists()


def test_changed_input_evidence_is_not_silently_overwritten(tmp_path):
    index, rows = index_fixture(tmp_path)
    excluded = tmp_path / "excluded.jsonl"
    write_jsonl(excluded, [rows[0]])
    pool = prepare_pool(tmp_path / "pool", index, [excluded], count=5)
    original_manifest = Path(pool["manifest"]).read_bytes()
    original_receipt = Path(pool["receipt"]).read_bytes()
    write_jsonl(index, list(reversed(rows)))
    with pytest.raises(ValueError, match="Refusing to overwrite different confirmation evidence"):
        prepare_pool(tmp_path / "pool", index, [excluded], count=5)
    assert Path(pool["manifest"]).read_bytes() == original_manifest
    assert Path(pool["receipt"]).read_bytes() == original_receipt


def test_selected_source_hashes_are_verified_without_model_inference(tmp_path):
    index, rows = index_fixture(tmp_path, 4)
    excluded = tmp_path / "excluded.jsonl"
    write_jsonl(excluded, [rows[0]])
    Path(rows[2]["audio"]).write_bytes(b"changed source")
    with pytest.raises(ValueError, match="Selected source audio changed"):
        prepare_pool(tmp_path / "pool", index, [excluded], count=3)
