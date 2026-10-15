"""Plot the audited raw Whisper transfer controls and expand their measured report."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from beyondwaveform.io import atomic_json, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    run = Path(args.run).resolve()
    results_path = run / "results.json"
    results = json.loads(results_path.read_text())
    audit = json.loads((run / "verification.json").read_text())
    assert audit["status"] == "passed" and audit["results_sha256"] == digest(results_path)
    models = results["targets"]
    assert [r["model"] for r in models] == ["tiny", "base", "medium", "large-v2"]
    figures = run / "figures"
    figures.mkdir(exist_ok=True)
    colors = ["#3c78a8", "#649663", "#ba6150"]
    fig, axis = plt.subplots(figsize=(11, 5))
    x = np.arange(len(models))
    for index, (variant, label) in enumerate(zip(["clean", "reconstruction", "adversarial"], ["Clean", "Reconstruction", "Attack"])):
        values = [r["summary"][variant]["error_percent"] for r in models]
        bars = axis.bar(x + (index - 1) * 0.25, values, width=0.24, color=colors[index], label=label)
        axis.bar_label(bars, labels=[f"{v:.2f}" for v in values], padding=3, fontsize=9)
    paper = results["paper_large_raw_comparison"]
    axis.scatter([x[-1] + 0.25], [paper["paper_large_wer"]], marker="D", s=45, color="#303030", zorder=5,
                 label="Paper large raw attack: 52.96% (revision unspecified)")
    axis.set(xticks=x, xticklabels=[r["model"] for r in models], ylim=(0, 80), ylabel="Corpus WER (%)",
             title="Raw Whisper transfer: same 999 frozen Whisper-small-generated clips")
    axis.spines[["top", "right"]].set_visible(False)
    axis.set_axisbelow(True)
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.10), fontsize=9)
    fig.tight_layout()
    fig.savefig(figures / "whisper-transfer.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    atomic_json(figures / "provenance.json", {"results": str(results_path), "results_sha256": digest(results_path),
                "verification": str(run / "verification.json"), "verification_sha256": digest(run / "verification.json"),
                "values": models, "paper_anchor": paper,
                "caveat": "Independent target checkpoints, corpus subset, attack config and decoder; the paper's large revision is unidentified."})
    lines = ["# Verified raw Whisper transfer", "", "The attack remains frozen at rho=0.5, lambda=0.1, Adam LR=0.05 and 50 steps. All four targets used the identical 999 saved English test clips and three audio controls. Target models were loaded only for evaluation; no generation, target-dependent tuning or quality rescoring occurred.", "",
             "| Raw target | Clean WER % | Reconstruction WER % | Attack WER % | Attack minus clean (pp) |",
             "|---|---:|---:|---:|---:|"]
    for row in models:
        s = row["summary"]
        values = [s[v]["error_percent"] for v in ["clean", "reconstruction", "adversarial"]]
        lines.append(f"| {row['model']} | {values[0]:.4f} | {values[1]:.4f} | {values[2]:.4f} | {values[2]-values[0]:.4f} |")
    lines += ["", "Every cell covers 999 utterances / 20,310 reference words. Raw multilingual targets use forced English, beam size five, temperature zero, and the pinned official checkpoint SHA256 receipts. Native frontends and the original saved audio policy are unchanged.", "",
              "![Measured Whisper transfer controls](figures/whisper-transfer.png)", "",
              f"Large-v2 attack WER is {paper['observed_large_v2_wer']:.4f}%, versus the paper's large raw anchor of {paper['paper_large_wer']:.2f}%: **{paper['gap_percentage_points']:+.4f} percentage points**. The existing independent large-v2 checkpoint choice is preserved. Paper target revisions, decoder recipe, subset IDs, and original selected config differ or are unavailable, so this gap does not establish an implementation error or exact numerical reproduction. Exact Figure 3 tiny/base/medium curve arrays are unavailable; no values were invented or digitized. Baseline/defense comparisons are still needed to assess the paper's aggregate transfer-gain claim.", "",
              f"The post-completion audit recomputed all **{audit['evaluation_rows']:,}** `(utterance, variant, target)` score rows and confirmed identical generation fingerprints, audio hashes, references, roles, counts, target configurations, checkpoint hashes, lexical edit counts and corpus aggregation. Source FLACs and all **{audit['saved_wavs_rehashed_once']:,}** saved WAVs were rehashed once; per-target evaluation had also rehashed each WAV. All four per-target audits passed. Models were not rerun for this audit.", "",
              "The [verified source quality report](../english-feature-test-999/report.md) remains applicable to these same raw WAVs: attack DNSMOS/NISQA/UTMOS 2.963822 / 3.871132 / 2.792377. Transfer WER does not establish human intelligibility, defended-model robustness, or over-the-air performance.", "",
              "Evidence: [combined post-completion audit](verification.json), [measured results](results.json), [frozen plan](plan.json), [figure data/provenance](figures/provenance.json), and each target's `summary.json`, `utterances.jsonl`, and `verification.json`. The next experiment is raw independently pinned Wav2Vec2/HuBERT CTC transfer using these same WAVs and the documented greedy decoder."]
    (run / "report.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote verified Whisper report/figure: {run / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
