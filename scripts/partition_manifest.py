"""Reserve disjoint validation speakers from a non-test manifest."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from beyondwaveform.data import partition_manifest

parser = argparse.ArgumentParser()
parser.add_argument("--manifest", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--validation-fraction", type=float, default=0.1)
parser.add_argument("--seed", type=int, default=2026)
parser.add_argument("--remainder-role", choices=["finetune", "train", "pilot"], default="finetune")
args = parser.parse_args()
for path in partition_manifest(args.manifest, args.output, args.validation_fraction, args.seed, args.remainder_role):
    print(path)

