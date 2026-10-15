"""Engineering sensitivity study for the paper's unspecified attack LR.

Uses only the disjoint pilot pool, never final test or target feedback.
"""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tqdm import tqdm
from beyondwaveform.config import load_config
from beyondwaveform.io import read_jsonl, atomic_json
from beyondwaveform.pipeline import generate, evaluate

parser = argparse.ArgumentParser()
parser.add_argument("--manifest", required=True)
parser.add_argument("--config", default="configs/english.yaml")
parser.add_argument("--base-run", default="reproduction-runs/attack-pilot")
parser.add_argument("--output", default="reproduction-runs/optimizer-pilot")
parser.add_argument("--learning-rates", type=float, nargs="+", default=[0.02, 0.05])
parser.add_argument("--device", default="cuda")
args = parser.parse_args()
if any(r["role"] != "pilot" for r in read_jsonl(args.manifest)):
    raise ValueError("Optimizer study must use a disjoint pilot pool")
if any(rate <= 0 for rate in args.learning_rates):
    raise ValueError("Learning rates must be positive")
config = load_config(args.config)
results = {}
base = Path(args.base_run)
print("Scoring baseline learning rate on saved audio...", flush=True)
base_result = evaluate(base, {"kind": "whisper", **config["surrogate"]},
                       Path(args.output) / "lr-0.01" / "evaluation", args.device,
                       variants=["adversarial"], quality=True, artifact_root=config["features"]["artifact_root"])
results["0.01"] = base_result["summary"]["adversarial"]
for rate in tqdm(args.learning_rates, desc="Optimizer sensitivity", unit="runs"):
    config["attack"]["learning_rate"] = rate
    run = Path(args.output) / f"lr-{rate:g}"
    print(f"Running lr={rate:g} on {args.manifest}", flush=True)
    generate(config, args.manifest, run, args.device)
    outcome = evaluate(run, {"kind": "whisper", **config["surrogate"]}, run / "evaluation",
                       args.device, variants=["adversarial"], quality=True,
                       artifact_root=config["features"]["artifact_root"])
    results[str(rate)] = outcome["summary"]["adversarial"]
    atomic_json(Path(args.output) / "summary.json", results)
print(json.dumps(results, indent=2), flush=True)
