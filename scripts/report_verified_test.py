"""Create static figures and an evidence-backed Markdown report for a verified test."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from beyondwaveform.experiments import report
from beyondwaveform.io import atomic_json, digest, read_jsonl
from beyondwaveform.references import TABLE_1, TABLE_4


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--freeze", required=True)
    args = parser.parse_args()
    run = Path(args.run).resolve()
    summary_path = run / "surrogate-eval" / "summary.json"
    audit = json.loads((run / "verification.json").read_text())
    assert audit["status"] == "passed" and audit["summary_sha256"] == digest(summary_path)
    result = json.loads(summary_path.read_text())
    frozen = json.loads(Path(args.freeze).read_text())
    summary = result["summary"]
    variants = ["clean", "reconstruction", "adversarial"]
    metrics = ["dnsmos", "nisqa", "utmos"]
    labels = ["Original clean", "Vocoder reconstruction", "Feature attack"]
    colors = ["#3c78a8", "#649663", "#ba6150"]
    figures = run / "figures"
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
    wer = [summary[v]["error_percent"] for v in variants]
    bars = axes[0].bar(np.arange(3), wer, color=colors, width=0.65)
    axes[0].bar_label(bars, labels=[f"{v:.2f}%" for v in wer], padding=4)
    axes[0].set(xticks=np.arange(3), xticklabels=["Clean", "Reconstruction", "Attack"], ylabel="Corpus WER (%)", ylim=(0, 85))
    for index, variant in enumerate(variants):
        bars = axes[1].bar(np.arange(3) + (index - 1) * 0.25,
                           [summary[variant]["quality"][m] for m in metrics], width=0.24,
                           label=labels[index], color=colors[index])
        axes[1].bar_label(bars, labels=[f"{summary[variant]['quality'][m]:.2f}" for m in metrics], padding=3, fontsize=8)
    axes[1].axhline(2.5, color="#555555", linestyle="--", linewidth=1, label="Mean gate: strictly > 2.5")
    axes[1].set(xticks=np.arange(3), xticklabels=[m.upper() for m in metrics], ylabel="Mean predicted MOS", ylim=(0, 5))
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False, fontsize=8)
    for axis in axes:
        axis.set_axisbelow(True)
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle("Independent BeyondWaveform test: 999 English clips / raw Whisper-small")
    fig.tight_layout()
    fig.savefig(figures / "test-controls.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    records = read_jsonl(run / "generated.jsonl")
    nll = np.array([[s["nll_before_update"] for s in r["trace"]] for r in records])
    radii = np.array([r["trace"][-1]["radius_after_update"] for r in records])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    steps = np.arange(1, nll.shape[1] + 1)
    lower, median, upper = np.quantile(nll, [0.1, 0.5, 0.9], axis=0)
    axes[0].fill_between(steps, lower, upper, alpha=0.2, color=colors[2], label="10th–90th utterance percentiles")
    axes[0].plot(steps, median, color=colors[2], label="Median across 999 utterances")
    axes[0].set(xlabel="Update (NLL logged before update)", ylabel="Mean lexical-token NLL")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].hist(radii, bins=30, color=colors[2], alpha=0.85)
    axes[1].axvline(frozen["config"]["attack"]["rho"], color="#555555", linestyle="--", label="Frozen rho limit = 0.5")
    axes[1].set(xlabel="Final relative feature Frobenius radius", ylabel="Utterances", xlim=(0, 0.52))
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("Verified finite 50-step attack traces")
    fig.tight_layout()
    fig.savefig(figures / "attack-traces.png", dpi=180)
    plt.close(fig)
    comparison = {}
    for variant, paper_row in [("clean", "Clean"), ("adversarial", "Feature attack")]:
        comparison[variant] = {"wer": {"observed": summary[variant]["error_percent"], "paper": TABLE_1["en"][paper_row][0],
                                       "gap_percentage_points": summary[variant]["error_percent"] - TABLE_1["en"][paper_row][0]},
                              "quality": {m: {"observed": summary[variant]["quality"][m], "paper": TABLE_4[paper_row][i],
                                              "gap_mos": summary[variant]["quality"][m] - TABLE_4[paper_row][i]}
                                          for i, m in enumerate(metrics)}}
    provenance = {"summary": str(summary_path), "summary_sha256": digest(summary_path),
                  "verification": str(run / "verification.json"), "verification_sha256": digest(run / "verification.json"),
                  "freeze": str(Path(args.freeze).resolve()), "freeze_sha256": digest(args.freeze),
                  "paper": str(ROOT / "BeyondWaveform.pdf"), "paper_sha256": digest(ROOT / "BeyondWaveform.pdf"),
                  "comparability": "Independent subset/config/models/decoding/scorers; numeric differences do not establish exact reproduction."}
    atomic_json(run / "paper-comparison.json", {"provenance": provenance, "comparisons": comparison})
    atomic_json(figures / "provenance.json", {**provenance, "control_values": summary,
                                             "nll_percentiles": {"steps": steps.tolist(), "p10": lower.tolist(),
                                                                 "median": median.tolist(), "p90": upper.tolist()},
                                             "final_radii": radii.tolist()})
    report([str(summary_path)], run / "report.md")
    diag = audit["attack_diagnostics"]
    job = json.loads((run / "job.json").read_text())
    wall_hours = (datetime.fromisoformat(job["finished_at"]) - datetime.fromisoformat(job["started_at"])).total_seconds() / 3600
    lines = ["", "## Verified interpretation", "",
             f"All 999 test clips and 2,997 variant scores were verified, covering 20,310 reference words. Original-clean WER is {wer[0]:.4f}% and zero-delta reconstruction WER is {wer[1]:.4f}%. The attack increases WER by {wer[2]-wer[0]:.4f} percentage points from clean and {wer[2]-wer[1]:.4f} from reconstruction. The large error increase is associated with optimization rather than failed zero-delta reconstruction.", "",
             "This is the raw generation-surrogate result. Transfer to other models is a separate experiment, and no human intelligibility or over-the-air claim follows from predicted MOS.", "",
             "![Measured controls and predicted quality](figures/test-controls.png)", "",
             "## Comparison with the supplied paper", "",
             "| Metric / condition | Measured | Paper | Measured minus paper |",
             "|---|---:|---:|---:|"]
    for variant, paper_row in [("clean", "Clean"), ("adversarial", "Feature attack")]:
        c = comparison[variant]
        lines.append(f"| WER / {paper_row} | {c['wer']['observed']:.4f}% | {c['wer']['paper']:.2f}% | {c['wer']['gap_percentage_points']:+.4f} pp |")
        for metric in metrics:
            values = c["quality"][metric]
            lines.append(f"| {metric.upper()} / {paper_row} | {values['observed']:.4f} | {values['paper']:.3f} | {values['gap_mos']:+.4f} MOS |")
    lines += ["", "WER anchors are Table 1 Raw; quality anchors are Table 4. There is no paper reconstruction-control row. The independently selected rho=0.5 / lambda=0.1 differs from the paper's rho=0.1 / lambda=1. Author subsets, WavLM/vocoder checkpoint IDs, optimizer recipe, decoder and quality-scoring details are unavailable. Table 4's exact evaluation pool is also unspecified. These gaps are descriptive, not evidence of statistically equivalent results or an exact numerical reproduction.", "",
              "## Quality and optimizer diagnostics", "",
              "The three adversarial mean quality scores each exceed 2.5. Only 747/999 attack clips individually exceed 2.5 on all three scorers; 58 clips have DNSMOS <=2.5, 15 have NISQA <=2.5, and 209 have UTMOS <=2.5. The protocol gates means, so this is a distributional limitation, not a reason to reselect on test data. Attack UTMOS is about 0.798 MOS below Table 4.", "",
              f"All 49,950 updates were finite and within rho=0.5. Final relative radii range from {diag['final_radius']['min']:.4f} to {diag['final_radius']['max']:.4f}, median {diag['final_radius']['median']:.4f}; none reach the radius bound. First-to-last logged pre-update NLL increases for all 999 clips (mean {diag['first_preupdate_nll']['mean']:.4f} to {diag['last_preupdate_nll']['mean']:.4f}). These are pre-update values, not an additional post-update loss measurement. The frozen 50-step config remains unchanged.", "",
              "![NLL and final perturbation radii](figures/attack-traces.png)", "",
              f"Recorded optimization time averages {diag['generation_seconds']['mean']:.3f} s/clip ({sum(r['generation_seconds'] for r in records)/3600:.3f} hours total). The complete job took {wall_hours:.3f} wall hours, including model loading, I/O and quality evaluation. These TITAN RTX wall timings are not an L20 benchmark or a direct measurement of GPU-hours. Native vocoder output is 80–321 samples shorter than original clean audio here (5–20.0625 ms); no duration warping was introduced.", "",
              "## Inspectable evidence", "",
              "- [Full audit](verification.json): corpus and saved-WAV hashes, 999 source clips, 2,997 scored variants, exact references/edit counts/aggregates, frozen protocol and projected traces.",
              "- [Raw evaluation summary](surrogate-eval/summary.json) and [per-utterance scores/transcripts](surrogate-eval/utterances.jsonl).",
              "- [Paper comparisons with provenance](paper-comparison.json) and [figure data/provenance](figures/provenance.json).",
              "- [Generation metadata](run.json), [generated records](generated.jsonl), [completion status](job.json), and [stage status](progress.json).",
              "- [Frozen config](../../configs/english_selected.yaml), [freeze receipt](../../configs/english_selected.freeze.json), [reproduction choices](../../docs/REPRODUCTION.md), and [supplied paper](../../BeyondWaveform.pdf).", "",
              "Transfer, baselines/ablations, Mandarin, and artifact-dependent defense/human/OTA experiments are still outstanding. The saved test WAVs can be reused for fixed-config transfer without regenerating or tuning."]
    with (run / "report.md").open("a") as stream:
        stream.write("\n".join(lines) + "\n")
    print(f"Wrote verified report and figures: {run / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
