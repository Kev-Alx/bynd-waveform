"""Evaluate the remaining AISHELL test recordings and write the full-test report."""
import argparse
import fcntl
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_aishell_epoch3_full_run import FullRunWorker
from run_aishell_epoch3 import VARIANTS
from beyondwaveform.io import atomic_json, digest, read_jsonl


def combine_summaries(previous, current):
    """Pool edit counts and reference lengths, rather than averaging percentages."""
    combined = {}
    for variant in VARIANTS:
        old, new = previous[variant], current.get(variant, {})
        values = {"count": old["count"] + new.get("count", 0)}
        for metric in ("cer", "wer_jieba"):
            counts = {key: old[metric][key] + new.get(metric, {}).get(key, 0)
                      for key in ("substitutions", "deletions", "insertions", "reference_units")}
            counts["error_percent"] = 100 * sum(counts[key] for key in ("substitutions", "deletions", "insertions")) / counts["reference_units"]
            values[metric] = counts
        combined[variant] = values
    return combined


class CompleteTestWorker(FullRunWorker):
    def __init__(self, output, deadline=None, device="cuda"):
        super().__init__(output, deadline, device)
        lineage = self.plan["combined_parent"]
        self.parent = Path(lineage["output"])
        assert digest(self.parent / "plan.json") == lineage["plan_sha256"]
        parent_plan = json.loads((self.parent / "plan.json").read_text())
        assert parent_plan["checkpoint_sha256"] == self.plan["checkpoint_sha256"]
        assert digest(self.parent / "config.json") == digest(self.output / "config.json"), "Numerical settings changed"
        self.previous = json.loads((self.parent / "results.json").read_text())
        assert digest(self.parent / "results.json") == lineage["results_sha256"]
        assert self.previous["status"] == "complete"
        assert all(value["count"] == lineage["count"] for value in self.previous["summary"].values())
        self.previous_count = lineage["count"]
        old_rows = read_jsonl(self.parent / "manifest.jsonl")
        official = read_jsonl(Path(self.plan["full_test_manifest"]))
        selected = old_rows + self.rows
        assert len(selected) == len({row["id"] for row in selected}) == len(official) == self.plan["official_test_count"]
        assert {row["id"]: row for row in selected} == {row["id"]: row for row in official}

    def publish(self):
        super().publish()
        current = json.loads((self.output / "results.json").read_text())
        combined = combine_summaries(self.previous["summary"], current["summary"])
        target = self.plan["official_test_count"]
        if self.status == "complete":
            assert all(value["count"] == target for value in combined.values())
        atomic_json(self.output / "combined-results.json", {
            "status": self.status, "epoch": 3, "count": target,
            "previous_count": self.previous_count, "remaining_count": len(self.rows),
            "plan_sha256": self.plan_sha, "parent_plan_sha256": self.plan["combined_parent"]["plan_sha256"],
            "summary": combined})
        report = (f"# AISHELL-1 complete test split — epoch 3\n\nStatus: **{self.status}**. "
                  f"Target: **{target:,} official test recordings**. The completed {self.previous_count:,} recordings are retained; "
                  f"this detached job evaluates the remaining {len(self.rows):,}.\n\n"
                  "| Audio | Recordings scored | CER | WER (jieba) |\n|---|---:|---:|---:|\n")
        for variant, value in combined.items():
            report += f"| {variant} | {value['count']:,} | {value['cer']['error_percent']:.4f}% | {value['wer_jieba']['error_percent']:.4f}% |\n"
        report += ("\nRates pool substitutions, deletions, insertions and reference lengths over all recordings; subset percentages are never averaged. "
                   "Until status is complete, the table contains the completed 5,000 plus any newly scored recordings.\n\n"
                   "CER is the primary Mandarin metric. WER uses jieba 0.42.1 with HMM disabled on identically normalized references and hypotheses. "
                   "WER may exceed 100% when insertions are numerous. Official transcript strings are retained; normalization uses NFKC, lowercase, "
                   "and removal of punctuation and whitespace.\n\n"
                   "ASR: epoch 3 fine-tuned Whisper small, forced zh, beam 5, temperature 0, fp32, timestamps disabled, no previous-text conditioning. "
                   f"Attack: rho {self.config['attack']['rho']}, lambda {self.config['attack']['lambda_perc']}, "
                   f"{self.config['attack']['steps']} steps, Adam LR {self.config['attack']['learning_rate']}. "
                   "The checkpoint, attack and decoding settings match the completed 5,000. No test data enters fine-tuning or checkpoint selection.\n\n"
                   f"- [Full-test scores and edit counts]({self.output}/combined-results.json)\n"
                   f"- [Status, ETA and stop/resume commands]({self.output}/JOB_STATUS.md)\n"
                   f"- [Completed 5,000-recording results]({self.parent}/results.json)\n"
                   f"- [Remaining-recording scores]({self.output}/results.json)\n"
                   f"- [Frozen protocol]({self.output}/plan.json)\n")
        (self.output / "report.md").write_text(report)
        progress_path = self.output / "progress.json"
        progress = json.loads(progress_path.read_text())
        progress.update(full_test_count=target, previous_completed=self.previous_count,
                        full_test_generated=self.previous_count + len(self.generated),
                        full_test_scored_variants=3 * self.previous_count + len(self.scores))
        atomic_json(progress_path, progress)
        prefix = (f"Full AISHELL test evaluation: **{target:,} recordings**. Previously completed: **{self.previous_count:,}**. "
                  f"New work: **{len(self.generated)}/{len(self.rows)} generated**, **{len(self.scores)}/{3 * len(self.rows)} variant scores**. "
                  f"[Combined CER and WER report]({self.output}/report.md).\n\n")
        for path in [self.output / "JOB_STATUS.md", ROOT / "reproduction-runs/mandarin-cer-full-dev/JOB_STATUS.md",
                     ROOT / "reproduction-runs/mandarin-cer-corrected/WORKFLOW_STATUS.md"]:
            path.write_text(prefix + path.read_text())
        if self.status == "complete":
            audit = json.loads((self.output / "independent-audit.json").read_text())
            parent_audit = json.loads((self.parent / "independent-audit.json").read_text())
            assert audit["status"] == parent_audit["status"] == "passed"
            atomic_json(self.output / "combined-audit.json", {
                "status": "passed", "count": target, "generation_records": self.previous_count + audit["generation_records"],
                "character_distance_checks": parent_audit["character_distance_checks"] + audit["character_distance_checks"],
                "word_distance_checks": parent_audit["word_distance_checks"] + audit["word_distance_checks"],
                "combined_results_sha256": digest(self.output / "combined-results.json"),
                "report_sha256": digest(self.output / "report.md"), "plan_sha256": self.plan_sha})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deadline-unix", type=float, default=None)
    args = parser.parse_args()
    with (args.output / "worker.lock").open("a+") as lock, (ROOT / "reproduction-runs/auxiliary-gpu.lock").open("a+") as gpu:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(gpu, fcntl.LOCK_EX | fcntl.LOCK_NB)
        CompleteTestWorker(args.output, args.deadline_unix).run()
