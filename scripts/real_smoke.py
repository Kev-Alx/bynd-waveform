"""Real pretrained x / V(q) / two-step attack check on a supplied pilot manifest."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from beyondwaveform.config import load_config
from beyondwaveform.io import read_jsonl, atomic_json
from beyondwaveform.pipeline import generate, evaluate

parser = argparse.ArgumentParser()
parser.add_argument("--manifest", required=True)
parser.add_argument("--config", default="configs/english.yaml")
parser.add_argument("--output", default="runs/real-smoke")
parser.add_argument("--device", default="cuda")
args = parser.parse_args()
if any(r["role"] != "pilot" for r in read_jsonl(args.manifest)):
    raise ValueError("Use a pilot manifest, separate from final evaluation")
torch.set_num_threads(4)
config = load_config(args.config)
config["attack"]["steps"] = 2
config["save_delta"] = True
generate(config, args.manifest, args.output, args.device, limit=1)
summary = evaluate(args.output, {"kind": "whisper", **config["surrogate"]},
                   Path(args.output) / "evaluation", args.device)
print(json.dumps(summary["summary"], indent=2))

