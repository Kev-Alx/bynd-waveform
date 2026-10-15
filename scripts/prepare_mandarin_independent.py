"""Freeze speaker-disjoint AISHELL development pools and the official held-out test."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from beyondwaveform.data import audit_disjoint, prepare_manifest
from beyondwaveform.io import atomic_json, digest, read_jsonl, write_jsonl


def split_speakers(rows, seed=2026):
    speakers = sorted({r["speaker"] for r in rows})
    if len(speakers) < 3:
        raise ValueError("Three independent development speaker groups are required")
    random.Random(seed).shuffle(speakers)
    reserve = max(1, min((len(speakers)-1)//2, round(len(speakers)*0.1)))
    cp, attack = set(speakers[:reserve]), set(speakers[reserve:2*reserve])
    pools = {"finetune": [], "checkpoint-validation": [], "attack-reserve": []}
    for original in rows:
        if original["split"] != "dev" or original["corpus"] != "aishell":
            raise ValueError("Only official AISHELL development audio can enter these pools")
        row = dict(original)
        key = "checkpoint-validation" if row["speaker"] in cp else "attack-reserve" if row["speaker"] in attack else "finetune"
        row["role"] = "finetune" if key == "finetune" else "validation"
        pools[key].append(row)
    return pools


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    manifest_dir = output / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    frozen = output / "data-freeze.json"
    if frozen.exists():
        receipt = json.loads(frozen.read_text())
        for path, sha in receipt["proofs"].items():
            if digest(path) != sha:
                raise ValueError(f"Frozen Mandarin input changed: {path}")
        print("Prepared Mandarin data already frozen; hashes passed.", flush=True)
        return
    dev_path, test_path = manifest_dir / "dev-all.jsonl", manifest_dir / "test.jsonl"
    dev = prepare_manifest("aishell", "/workspace/data", "dev", "finetune", dev_path)
    test = prepare_manifest("aishell", "/workspace/data", "test", "test", test_path)
    pools = split_speakers(dev)
    paths = {}
    for name, rows in pools.items():
        path = manifest_dir / f"{name}.jsonl"
        write_jsonl(path, rows)
        paths[name] = str(path)
    attack = random.Random(2026).sample(pools["attack-reserve"], 100)
    attack.sort(key=lambda r: r["id"])
    attack_path = manifest_dir / "attack-validation-100.jsonl"
    write_jsonl(attack_path, attack)
    paths["attack-validation"] = str(attack_path)
    used = [paths["finetune"], paths["checkpoint-validation"], paths["attack-validation"], str(test_path)]
    audit = audit_disjoint(used)
    speaker_sets = [{r["speaker"] for r in read_jsonl(p)} for p in used]
    if any(a & b for i, a in enumerate(speaker_sets) for b in speaker_sets[i+1:]):
        raise ValueError("Mandarin speaker leakage between training, validation or official test")
    cfg = ROOT / "configs/mandarin_finetune.yaml"
    checkpoint = ROOT / "artifacts/whisper/small.pt"
    proofs = {str(p): digest(p) for p in [dev_path, test_path, *map(Path, paths.values()), cfg, checkpoint,
        ROOT / "beyondwaveform/training.py", ROOT / "beyondwaveform/models/whisper.py"]}
    receipt = {"status": "frozen", "frozen_at": datetime.now(timezone.utc).isoformat(), "seed": 2026,
        "paths": paths, "test_manifest": str(test_path), "counts": {name: len(rows) for name, rows in pools.items()},
        "attack_validation_count": 100, "test_count": len(test), "speaker_counts": [len(s) for s in speaker_sets],
        "split_audit": audit, "proofs": proofs, "training_config": str(cfg),
        "fidelity": "Independent 80/10/10 development-speaker partition: train/checkpoint selection/attack selection. "
                    "Attack grid uses a seeded 100-clip subset of its separate reserve. All duration-eligible official test clips retained. "
                    "Author split/fine-tuned weights/training recipe and Mandarin test IDs/count unspecified."}
    atomic_json(frozen, receipt)
    for path in [dev_path, test_path, *map(Path, paths.values()), frozen]:
        path.chmod(0o444)
    print(json.dumps({k: receipt[k] for k in ["counts", "attack_validation_count", "test_count", "speaker_counts"]}), flush=True)


if __name__ == "__main__":
    main()
