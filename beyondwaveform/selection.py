"""Eq.16: select ONLY using validation surrogate error and all three MOS gates."""
import json
import math
from pathlib import Path
from .io import atomic_json


def select_configuration(summaries, output, threshold=2.5, strict=True):
    candidates = []
    manifest = None
    comparison_protocol = None
    for path in summaries:
        result = json.loads(Path(path).read_text())
        if result["evaluation_scope"] != "surrogate" or result["roles"] != ["validation"]:
            raise ValueError("Selection requires exclusively validation data and the raw generation surrogate")
        if manifest is None:
            manifest = result["manifest_sha256"]
        elif manifest != result["manifest_sha256"]:
            raise ValueError("Candidates must use the identical validation manifest")
        protocol = {"ids": result["evaluation_ids"], "target": result["target_artifacts"],
                    "quality": result["quality_provenance"], "metric": result["metric"],
                    "normalization": result["normalization"]}
        if comparison_protocol is None:
            comparison_protocol = protocol
        elif protocol != comparison_protocol:
            raise ValueError("Candidate subsets, surrogate, decoding, and scoring protocols must match")
        attack = result["summary"]["adversarial"]
        metrics = attack.get("quality", {})
        if set(metrics) != {"dnsmos", "nisqa", "utmos"} or not all(math.isfinite(v) for v in metrics.values()):
            raise ValueError("Selection needs all three finite quality metrics; no silent gate bypass")
        error = attack["error_percent"]
        if not math.isfinite(error):
            raise ValueError("Nonfinite error rate")
        eligible = all(value > threshold if strict else value >= threshold for value in metrics.values())
        candidates.append({"summary": str(Path(path).resolve()), "generation_run": result["generation_run"],
                           "generation_fingerprint": result["generation_fingerprint"], "error_percent": error,
                           "quality": metrics, "eligible": eligible})
    eligible = [r for r in candidates if r["eligible"]]
    winner = max(eligible, key=lambda r: r["error_percent"]) if eligible else None
    selection = {"threshold": threshold, "strict": strict, "aggregation": "corpus error; mean utterance MOS",
                 "manifest_sha256": manifest, "candidates": candidates, "selected": winner}
    atomic_json(output, selection)
    if winner is None:
        raise ValueError("No configuration passed all three quality gates; audit saved")
    return selection
