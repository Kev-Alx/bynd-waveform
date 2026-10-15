"""Run the paper's default 50-step feature attack on disjoint pilot audio."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from beyondwaveform.config import load_config
from beyondwaveform.io import read_jsonl
from beyondwaveform.pipeline import generate, evaluate

parser = argparse.ArgumentParser()
parser.add_argument("--manifest", required=True)
parser.add_argument("--config", default="configs/english.yaml")
parser.add_argument("--output", default="reproduction-runs/attack-pilot")
parser.add_argument("--device", default="cuda")
args = parser.parse_args()
if any(r["role"] != "pilot" for r in read_jsonl(args.manifest)):
    raise ValueError("Pilot must use a separate role=pilot manifest")
config = load_config(args.config)
if config.get("method") != "feature":
    raise ValueError("Expected the feature attack")
print("Generating 50-step feature attacks on pilot audio...", flush=True)
generate(config, args.manifest, args.output, args.device)
print("Scoring saved clean, reconstruction, and adversarial audio...", flush=True)
summary = evaluate(args.output, {"kind": "whisper", **config["surrogate"]},
                   Path(args.output) / "evaluation", args.device)
print(json.dumps(summary["summary"], indent=2), flush=True)
print(f"Results: {Path(args.output, 'evaluation', 'summary.json').resolve()}", flush=True)
