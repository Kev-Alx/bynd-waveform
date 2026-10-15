"""Pick readable pilot utterances disjoint from the validation set."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from beyondwaveform.data import audit_disjoint
from beyondwaveform.io import read_jsonl, write_jsonl

parser = argparse.ArgumentParser()
parser.add_argument("--pilot", required=True)
parser.add_argument("--validation", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--min-words", type=int, default=4)
parser.add_argument("--min-seconds", type=float, default=3)
args = parser.parse_args()
validation = {r["id"] for r in read_jsonl(args.validation)}
eligible = [r for r in read_jsonl(args.pilot)
            if r["id"] not in validation and len(r["text"].split()) >= args.min_words
            and r["duration"] >= args.min_seconds]
if not eligible:
    raise ValueError("No eligible pilot utterances")
write_jsonl(args.output, eligible)
audit_disjoint([args.output, args.validation])
print(f"Selected {len(eligible)} readable pilot utterances at {args.output}", flush=True)

