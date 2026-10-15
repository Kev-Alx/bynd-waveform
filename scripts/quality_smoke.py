"""Check each real quality scorer on one exported speech clip."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tqdm import tqdm
from beyondwaveform.evaluation import QualityScorer
from beyondwaveform.io import atomic_json, load_audio

parser = argparse.ArgumentParser()
parser.add_argument("--audio", required=True)
parser.add_argument("--artifact-root", default="artifacts")
parser.add_argument("--output", default="reproduction-runs/quality-smoke.json")
args = parser.parse_args()
x = load_audio(args.audio)
scores = {}
for name in tqdm(["dnsmos", "nisqa", "utmos"], desc="Checking MOS scorers", unit="models"):
    scores.update(QualityScorer(args.artifact_root, metrics=(name,))(x))
    print(f"{name}: {scores[name]:.3f}", flush=True)
atomic_json(args.output, {"audio": str(Path(args.audio).resolve()), "scores": scores})
print(json.dumps(scores, indent=2), flush=True)
