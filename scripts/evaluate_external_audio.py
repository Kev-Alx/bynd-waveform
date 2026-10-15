"""Score user-supplied re-recordings (e.g. OTA) on a pinned target.

JSONL rows: id, condition, audio, text, language. Audio must already be
recorded; this script does not simulate a physical channel.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tqdm import tqdm
from beyondwaveform.config import load_config
from beyondwaveform.evaluation import aggregate_errors, error_counts
from beyondwaveform.io import atomic_json, digest, load_audio, write_jsonl
from beyondwaveform.models.targets import load_target

parser = argparse.ArgumentParser()
parser.add_argument("--manifest", required=True)
parser.add_argument("--target", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--device", default="cuda")
args = parser.parse_args()
rows = [json.loads(line) for line in Path(args.manifest).read_text().splitlines() if line.strip()]
if not rows or len({(r["condition"], r["id"]) for r in rows}) != len(rows):
    raise ValueError("Manifest must contain unique (condition, id) recordings")
if len({r["language"] for r in rows}) != 1:
    raise ValueError("Score one language at a time")
print("Loading target model...", flush=True)
target, target_meta = load_target(load_config(args.target), args.device)
results = []
for row in tqdm(rows, desc="Scoring external audio", unit="recordings"):
    audio = load_audio(row["audio"])
    hypothesis = target.transcribe(audio.to(args.device))
    unique_id = hashlib.sha256(f"{row['condition']}\0{row['id']}".encode()).hexdigest()
    results.append({"id": unique_id, "source_id": row["id"], "condition": row["condition"], "audio_sha256": digest(row["audio"]),
                    "reference": row["text"], "hypothesis": hypothesis,
                    **error_counts(row["text"], hypothesis, row["language"])})
summary = {condition: {"count": sum(r["condition"] == condition for r in results),
                       **aggregate_errors([r for r in results if r["condition"] == condition])}
           for condition in sorted({r["condition"] for r in results})}
output = Path(args.output)
write_jsonl(output / "utterances.jsonl", results)
atomic_json(output / "summary.json", {"manifest_sha256": digest(args.manifest), "target": target_meta,
                                             "metric": "CER" if rows[0]["language"] == "zh" else "WER",
                                             "summary": summary, "status": "external recordings; provenance and setup supplied by operator"})
print(json.dumps(summary, indent=2))
