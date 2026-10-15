"""Parallel CPU data preparation and the documented independent Mandarin recipe."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from beyondwaveform.io import atomic_json, digest
from parallel_jobs import Supervisor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    supervisor = Supervisor(output)
    try:
        supervisor.phase("speaker_disjoint_preparation", [sys.executable, "-u", str(ROOT / "scripts/prepare_mandarin_independent.py"),
            "--output", str(output)], threads=args.threads)
        data = json.loads((output / "data-freeze.json").read_text())
        training = output / "training"
        record = training / "training.json"
        if not record.exists():
            supervisor.phase("cpu_fine_tuning", [sys.executable, "-u", str(ROOT / "scripts/finetune.py"),
                "--config", data["training_config"], "--train-manifest", data["paths"]["finetune"],
                "--validation-manifest", data["paths"]["checkpoint-validation"], "--output", str(training),
                "--device", "cpu"], threads=args.threads)
        elif json.loads(record.read_text())["status"] != "complete":
            raise RuntimeError("Training is incomplete; preserve its checkpoints and inspect before resuming")
        state = json.loads(record.read_text())
        for path, sha in data["proofs"].items():
            if digest(path) != sha:
                raise ValueError(f"Frozen Mandarin training input changed: {path}")
        if state["status"] != "complete" or digest(training / "best.pt") != state["best_sha256"]:
            raise ValueError("Incomplete or changed selected Mandarin checkpoint")
        receipt = {"status": "frozen", "scope": "checkpoint_validation_loss_only", "checkpoint": str(training / "best.pt"),
            "checkpoint_sha256": state["best_sha256"], "data_freeze_sha256": digest(output / "data-freeze.json"),
            "training_metadata_sha256": digest(record), "history": state["history"], "cpu_threads": args.threads,
            "fidelity": "Existing three-epoch AdamW recipe unchanged; CPU execution uses spare compute. Separate speakers for "
                        "checkpoint selection and attack validation; official test excluded from training/selection. "
                        "Next: Mandarin reconstruction control, 16-setting validation grid, strict CER/MOS selection, full held-out test."}
        path = output / "surrogate-freeze.json"
        if path.exists() and json.loads(path.read_text()) != receipt:
            raise ValueError("Existing frozen Mandarin surrogate differs")
        if not path.exists():
            atomic_json(path, receipt)
            path.chmod(0o444)
        (training / "best.pt").chmod(0o444)
        supervisor.finish(surrogate_freeze=str(path), next_stage="Mandarin controls and attack validation; held-out test CER remains pending")
    except BaseException as error:
        supervisor.fail(error)
        raise


if __name__ == "__main__":
    main()
