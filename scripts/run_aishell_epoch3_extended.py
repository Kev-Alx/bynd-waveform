"""Extend the original epoch 3 run while preserving its immutable record provenance."""
import argparse
import fcntl
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_aishell_epoch3 import Worker, VARIANTS
from beyondwaveform.io import atomic_json, digest, read_jsonl


class ExtendedWorker(Worker):
    def __init__(self, output, deadline, device="cuda"):
        super().__init__(output, deadline, device)
        lineage = self.plan["resume_parent"]
        parent_path = Path(lineage["plan_path"])
        assert digest(parent_path) == lineage["plan_sha256"], "Parent protocol changed"
        parent = json.loads(parent_path.read_text())
        assert parent["checkpoint_sha256"] == self.plan["checkpoint_sha256"], "Cannot reuse another checkpoint"
        parent_config = parent_path.parent / "config.json"
        assert digest(parent_config) == parent["input_hashes"][str(parent_config)]
        assert digest(self.output / "config.json") == digest(parent_config), "Attack or decoding settings differ"
        parent_rows = read_jsonl(parent_path.parent / "manifest.jsonl")
        assert self.rows[:len(parent_rows)] == parent_rows, "Finish the unchanged original subset first"
        self.parent_sha = lineage["plan_sha256"]
        self.parent_ids = {row["id"] for row in parent_rows}

    def authorized(self, item):
        recorded = item["plan_sha256"]
        assert recorded == self.plan_sha or (recorded == self.parent_sha and item["id"] in self.parent_ids), "Untrusted cached protocol"
        # Validate the old record against this expanded, identical numerical protocol.
        # Its published JSON and original fingerprint remain unchanged on disk.
        return {**item, "plan_sha256": self.plan_sha}

    def validate_generation(self, record):
        super().validate_generation(self.authorized(record))

    def validate_score(self, item):
        super().validate_score(self.authorized(item))

    def publish(self):
        super().publish()
        progress_path = self.output / "progress.json"
        progress = json.loads(progress_path.read_text())
        first_generated = sum(uid in self.parent_ids for uid in self.generated)
        first_scored = sum(uid in self.parent_ids for uid, variant in self.scores)
        progress.update(original_2000_generated=first_generated, original_2000_scored_variants=first_scored,
                        next_3000_generated=len(self.generated) - first_generated,
                        next_3000_scored_variants=len(self.scores) - first_scored)
        atomic_json(progress_path, progress)
        report_path = self.output / "report.md"
        report = report_path.read_text().replace(
            "The same frozen 2,000 official test IDs are retained from the prior run, with unchanged official references.",
            "The original 5,000 official test IDs are retained: the unchanged first 2,000, followed by the original next 3,000. "
            "Committed epoch 3 audio and transcripts from the parent run are hash-verified and reused with their original protocol fingerprints. "
            "The frozen parent protocol has identical model, attack and decoding settings. References are unchanged.")
        report_path.write_text(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deadline-unix", type=float, required=True)
    args = parser.parse_args()
    with (args.output / "worker.lock").open("a+") as lock, (ROOT / "reproduction-runs/auxiliary-gpu.lock").open("a+") as gpu:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(gpu, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ExtendedWorker(args.output, args.deadline_unix).run()
