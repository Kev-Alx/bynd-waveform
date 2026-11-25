"""Freeze a fresh dev-clean confirmation pool without consulting outcomes."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import random
from collections import Counter

from beyondwaveform.io import atomic_json, digest, read_jsonl, write_jsonl


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _stable_manifest(path):
    path = Path(path).resolve()
    expected = digest(path)
    rows = read_jsonl(path)
    _require(digest(path) == expected, f"Manifest changed while reading: {path}")
    _require(all(isinstance(row.get("audio_sha256"), str) and row["audio_sha256"] for row in rows),
             f"Manifest is missing audio hashes: {path}")
    return path, expected, rows


def prepare_pool(output_dir, index_path, exclusion_paths, seed=20261019, count=48):
    """Return a protocol pool and write exact, reusable manifest/receipt files.

    Sampling is ``random.Random(seed).sample(sorted(eligible, key=id), count)``;
    selected rows are sorted again and retain the standard validation role.
    Exclusions form a union across every provided manifest, using ID OR audio
    SHA256. No WER, MOS, model output, or prior rank is accessed.
    """
    _require(isinstance(count, int) and not isinstance(count, bool) and count > 0,
             "count must be a positive integer")
    _require(isinstance(seed, int) and not isinstance(seed, bool), "seed must be an integer")
    output_dir = Path(output_dir).resolve()
    index_path, index_sha, index = _stable_manifest(index_path)
    _require(all(row.get("corpus") == "librispeech" and row.get("split") == "dev-clean" and
                 row.get("language") == "en" for row in index), "Confirmation source must be English LibriSpeech dev-clean")
    _require(all(isinstance(row.get("duration"), (int, float)) and math.isfinite(row["duration"]) and
                 .045 <= row["duration"] <= 30 and row.get("speaker") is not None and row.get("audio")
                 for row in index), "Invalid dev-clean duration, speaker or audio metadata")
    _require(len({row["audio_sha256"] for row in index}) == len(index),
             "Source index contains duplicate audio hashes; review before sampling")
    paths = sorted({Path(path).resolve() for path in exclusion_paths}, key=str)
    _require(paths, "At least one exclusion manifest is required")
    exclusions, excluded_rows = [], []
    for path in paths:
        path, sha, rows = _stable_manifest(path)
        exclusions.append({"manifest": str(path), "sha256": sha, "count": len(rows),
                           "unique_ids": len({r["id"] for r in rows}),
                           "unique_audio_sha256": len({r["audio_sha256"] for r in rows})})
        excluded_rows.extend(rows)
    blocked_ids = {row["id"] for row in excluded_rows}
    blocked_hashes = {row["audio_sha256"] for row in excluded_rows}
    by_id = {row["id"] for row in index if row["id"] in blocked_ids}
    by_audio = {row["id"] for row in index if row["audio_sha256"] in blocked_hashes}
    eligible = sorted((row for row in index if row["id"] not in blocked_ids and
                       row["audio_sha256"] not in blocked_hashes), key=lambda row: row["id"])
    _require(count <= len(eligible), f"Requested {count} confirmation clips; only {len(eligible)} eligible")
    selected = sorted(({**row, "role": "validation"} for row in random.Random(seed).sample(eligible, count)),
                      key=lambda row: row["id"])
    for row in selected:
        _require(digest(row["audio"]) == row["audio_sha256"], f"Selected source audio changed: {row['id']}")
    manifest = output_dir / f"confirmation-{count}.jsonl"
    receipt = output_dir / f"confirmation-{count}.receipt.json"
    manifest_bytes = "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in selected).encode("utf-8")
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    durations = [row["duration"] for row in selected]
    speakers = Counter(str(row["speaker"]) for row in selected)
    evidence = {"purpose": "fresh_development_confirmation", "corpus": "librispeech", "split": "dev-clean",
        "role": "validation", "seed": seed, "count": count,
        "algorithm": "random.Random(seed).sample(eligible sorted by ID, count); sort selected IDs; exclusions use ID OR audio SHA256 union",
        "index": {"manifest": str(index_path), "sha256": index_sha, "count": len(index)},
        "exclusions": exclusions,
        "exclusion_counts": {"rows_across_manifests": len(excluded_rows),
            "unique_ids": len(blocked_ids), "unique_audio_sha256": len(blocked_hashes),
            "index_rows_blocked_by_id": len(by_id), "index_rows_blocked_by_audio_sha256": len(by_audio),
            "index_rows_blocked_by_both": len(by_id & by_audio),
            "index_rows_blocked_union": len(by_id | by_audio), "eligible_index_rows": len(eligible)},
        "manifest": str(manifest), "manifest_sha256": manifest_sha,
        "selected_ids": [row["id"] for row in selected],
        "selected": [{key: row[key] for key in ("id", "audio_sha256", "duration", "speaker")} for row in selected],
        "duration_seconds": {"total": sum(durations), "mean": sum(durations) / count,
                             "minimum": min(durations), "maximum": max(durations)},
        "speakers": {"count": len(speakers), "ids": sorted(speakers), "clip_counts": dict(sorted(speakers.items()))},
        "selection_inputs": "Manifest metadata and source audio hashes only; no model outputs or score-based filtering",
        "limitation": "Fresh utterances from dev-clean; speakers may overlap prior development pools. This is not the author's subset or a final test pool."}
    receipt_bytes = (json.dumps(evidence, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    # Check both files before writing either, including a partially completed
    # matching construction. Changed evidence is never silently regenerated.
    for path, expected in ((manifest, manifest_bytes), (receipt, receipt_bytes)):
        _require(not path.exists() or path.read_bytes() == expected,
                 f"Refusing to overwrite different confirmation evidence: {path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    if not manifest.exists():
        write_jsonl(manifest, selected)
    if not receipt.exists():
        atomic_json(receipt, evidence)
    _require(digest(manifest) == manifest_sha and receipt.read_bytes() == receipt_bytes,
             "Confirmation evidence differs from the deterministic construction")
    manifest.chmod(0o444)
    receipt.chmod(0o444)
    return {"manifest": str(manifest), "sha256": manifest_sha, "ids": evidence["selected_ids"],
            "count": count, "seed": seed, "role": "validation", "purpose": evidence["purpose"],
            "receipt": str(receipt), "receipt_sha256": digest(receipt)}
