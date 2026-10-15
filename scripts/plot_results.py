"""Plot measured rho/lambda sweeps or ablations; never digitize invented values."""
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("summaries", nargs="+")
parser.add_argument("--axis", choices=["rho", "lambda_perc", "ablation"], required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()
rows = []
for path in args.summaries:
    result = json.loads(Path(path).read_text())
    run = json.loads((Path(result["generation_run"]) / "run.json").read_text())
    x = Path(result["generation_run"]).name if args.axis == "ablation" else run["config"]["attack"][args.axis]
    rows.append((x, result["summary"]["adversarial"], result["metric"]))
rows.sort(key=lambda r: r[0])
if len({row[0] for row in rows}) != len(rows):
    raise ValueError("Select one fixed-setting sweep; duplicate x values indicate mixed experiments")
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot([r[0] for r in rows], [r[1]["error_percent"] for r in rows], "o-")
axes[0].set_ylabel(f"{rows[0][2]} (%)")
for metric in ["dnsmos", "nisqa", "utmos"]:
    if all(metric in r[1].get("quality", {}) for r in rows):
        axes[1].plot([r[0] for r in rows], [r[1]["quality"][metric] for r in rows], "o-", label=metric.upper())
axes[1].set_ylabel("Predicted MOS")
if axes[1].lines:
    axes[1].legend()
for axis in axes:
    axis.set_xlabel(args.axis)
    axis.grid(alpha=0.25)
    if args.axis == "ablation":
        axis.tick_params(axis="x", rotation=35)
fig.suptitle("Independent reproduction — measured results")
fig.tight_layout()
Path(args.output).parent.mkdir(parents=True, exist_ok=True)
fig.savefig(args.output, dpi=180)

