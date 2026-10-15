"""Report audited raw CTC transfer measurements and exact available paper anchors."""
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
    results = json.loads((run / "results.json").read_text())
    audit = json.loads((run / "verification.json").read_text())
    assert results["status"] == "complete" and audit["status"] == "passed"
    assert audit["results_sha256"] == digest(run / "results.json")
    models = results["targets"]
    assert [m["model"] for m in models] == ["wav2vec2", "hubert"]
    figures = run / "figures"
    figures.mkdir(exist_ok=True)
    fig, axis = plt.subplots(figsize=(9, 5))
    x = np.arange(2)
    for i, (variant, label, color) in enumerate(zip(
            ["clean", "reconstruction", "adversarial"], ["Clean", "Reconstruction", "Attack"],
            ["#3c78a8", "#649663", "#ba6150"])):
        values = [m["summary"][variant]["error_percent"] for m in models]
        bars = axis.bar(x + (i - 1) * 0.25, values, width=0.24, color=color, label=label)
        axis.bar_label(bars, labels=[f"{v:.2f}" for v in values], padding=3)
    anchors = [m["paper_raw_comparison"]["paper_wer"] for m in models]
    axis.scatter(x + 0.25, anchors, marker="D", s=45, color="#303030", zorder=5,
                 label="Paper raw attack (checkpoint/decoder unspecified)")
    axis.set(xticks=x, xticklabels=["Wav2Vec2 large", "HuBERT large"], ylim=(0, 60),
             ylabel="Corpus WER (%)", title="Raw CTC transfer: same 999 frozen Whisper-small-generated clips")
    axis.spines[["top", "right"]].set_visible(False)
    axis.set_axisbelow(True)
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.10), fontsize=9)
    fig.tight_layout()
    fig.savefig(figures / "ctc-transfer.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    atomic_json(figures / "provenance.json", {
        "results_sha256": digest(run / "results.json"), "verification_sha256": digest(run / "verification.json"),
        "plan_sha256": digest(run / "plan.json"), "values": models,
        "caveat": "Independent checkpoint revisions, greedy decoder, corpus IDs and selected attack settings."})
    lines = ["# Verified raw CTC transfer", "",
        "Two independently pinned CTC targets evaluated the same 999 saved English test clips generated with frozen Whisper-small, rho=0.5, lambda=0.1, Adam LR=0.05 and 50 updates. All three audio controls were retained. No audio regeneration, attack tuning or quality rescoring occurred.", "",
        "| Raw target | Clean WER % | Reconstruction WER % | Attack WER % | Paper attack WER % | Gap (pp) |",
        "|---|---:|---:|---:|---:|---:|"]
    for m in models:
        s, p = m["summary"], m["paper_raw_comparison"]
        lines.append(f"| {m['model']} | {s['clean']['error_percent']:.4f} | {s['reconstruction']['error_percent']:.4f} | "
                     f"{s['adversarial']['error_percent']:.4f} | {p['paper_wer']:.2f} | {p['gap_percentage_points']:+.4f} |")
    lines += ["", "Each cell covers 999 utterances and 20,310 reference words. Recognition uses each checkpoint's native processor at 16 kHz, framewise argmax and native CTC collapse, with no external language model. Reconstruction WER stays close to clean; optimized feature perturbations account for the substantial additional errors.", "",
              "![Measured CTC transfer controls](figures/ctc-transfer.png)", "",
              "The paper does not identify the target checkpoint revisions, CTC decoder recipe or utterance IDs. These gaps therefore describe independent measurements; they do not establish exact numerical reproduction. The already selected attack remains unchanged. Figure 4's complete curve arrays and defended/baseline measurements are unavailable.", "",
              f"The fresh post-completion audit recomputed all **{audit['evaluation_rows']:,}** target/control score rows, rehashed **{audit['source_flacs_rehashed']:,} original FLACs and {audit['saved_wavs_rehashed']:,} saved WAVs**, and verified immutable target configs/receipts, published weight LFS SHA256, config/tokenizer Git blob hashes, generation fingerprints, references and corpus edit-count aggregation. No recognition was rerun during auditing.", "",
              "The [verified source quality report](../english-feature-test-999/report.md) applies to these unchanged WAVs: attack DNSMOS/NISQA/UTMOS 2.963822 / 3.871132 / 2.792377. Automatic quality does not establish human intelligibility or over-the-air performance.", "",
              "Frozen target artifacts:"]
    for m in models:
        lines.append(f"- `{m['repo']}` at `{m['revision']}`; evidence under `{m['model']}/`.")
    lines += ["", "Evidence: [combined verification](verification.json), [measured results](results.json), [frozen plan](plan.json), [figure source hashes](figures/provenance.json), and each target's summary, scored rows, verification and immutable target-freeze record. The next experiment removes the four documented loss components with all remaining settings held fixed."]
    (run / "report.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote verified CTC report and figure: {run / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
