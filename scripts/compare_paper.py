"""Compare a measured summary with a named exact Table 1 anchor."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from beyondwaveform.io import atomic_json
from beyondwaveform.references import DEFENSES, TABLE_1, compare_table_1

parser = argparse.ArgumentParser()
parser.add_argument("--summary", required=True)
parser.add_argument("--language", choices=["en", "zh"], required=True)
parser.add_argument("--condition", choices=list(TABLE_1["en"]), required=True)
parser.add_argument("--defense", choices=DEFENSES, default="Raw")
parser.add_argument("--variant", choices=["clean", "reconstruction", "adversarial"], default="adversarial")
parser.add_argument("--output", required=True)
args = parser.parse_args()
summary = json.loads(Path(args.summary).read_text())
if (summary["metric"] == "CER") != (args.language == "zh"):
    raise ValueError("Reference language does not match measured metric")
result = compare_table_1(summary, args.language, args.condition, args.defense, args.variant)
atomic_json(args.output, result)
print(json.dumps(result, indent=2))

