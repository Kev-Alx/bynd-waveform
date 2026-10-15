"""Audit validation-only waveform baselines, including saved-sample L-infinity bounds."""
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import soundfile as sf
from beyondwaveform.config import load_config
from beyondwaveform.evaluation import aggregate_errors, error_counts
from beyondwaveform.io import digest, load_audio, read_jsonl
from beyondwaveform.pipeline import config_fingerprint


def require(condition, message):
    if not condition:
        raise ValueError(message)


def waveform_bound(clean_path, adversarial_path, epsilon):
    for path in [clean_path, adversarial_path]:
        info = sf.info(path)
        require(info.subtype == "FLOAT" and info.channels == 1 and info.samplerate == 16000, "Wrong waveform export policy")
    clean, _ = sf.read(clean_path, dtype="float32")
    adversarial, _ = sf.read(adversarial_path, dtype="float32")
    require(clean.shape == adversarial.shape and len(clean) > 0 and np.isfinite(clean).all() and
            np.isfinite(adversarial).all(), "Changed length or nonfinite waveform")
    radius = float(np.max(np.abs(adversarial-clean)))
    require(radius <= epsilon + 1e-6, "Saved waveform violates L-infinity bound")
    require(float(np.max(np.abs(adversarial))) <= 1 + 1e-6, "Saved waveform violates [-1,1] range")
    return radius


def verify_waveform_run(run, entry, plan):
    run = Path(run).resolve()
    require(digest(entry["config"]) == entry["config_sha256"], "Frozen baseline config changed")
    config = load_config(entry["config"])
    metadata = json.loads((run / "run.json").read_text())
    evaluation = run / "evaluation"
    result = json.loads((evaluation / "summary.json").read_text())
    rows = read_jsonl(plan["validation_manifest"])
    ids = [r["id"] for r in rows]
    require(len(ids) == 100 and all(r["role"] == "validation" and r["split"] == "dev-clean" and
            r["language"] == "en" for r in rows), "Wrong validation pool")
    require(metadata["status"] == "complete" and metadata["count"] == 100 and metadata["config"] == config and
            metadata["features"] is None, "Incomplete/wrong waveform generation")
    require(metadata["ids"] == result["evaluation_ids"] == ids and metadata["roles"] == result["roles"] == ["validation"],
            "Wrong IDs or mixed roles")
    require(digest(plan["validation_manifest"]) == metadata["manifest_sha256"] == result["manifest_sha256"] ==
            plan["validation_manifest_sha256"], "Validation manifest changed")
    require(config["method"] == entry["method"] and config["waveform"]["steps"] == 50 and
            config_fingerprint({k: metadata[k] for k in ["config", "manifest_sha256", "ids"]}) ==
            metadata["fingerprint"] == result["generation_fingerprint"], "Wrong method/steps/fingerprint")
    require(metadata["surrogate"] == result["target_artifacts"] == plan["surrogate_artifacts"] and
            result["target"] == {"kind": "whisper", **config["surrogate"]} and result["evaluation_scope"] == "surrogate",
            "Wrong/defended target or surrogate")
    require(result["metric"] == "WER" and result["quality_provenance"] == plan["quality_provenance"] and
            result["normalization"] == "NFKC/lowercase/punctuation-to-space; Chinese whitespace removed",
            "Changed scoring protocol")
    require(digest(run / "generated.jsonl") == metadata["generated_sha256"] and
            digest(evaluation / "utterances.jsonl") == result["utterances_sha256"], "Changed JSONL evidence")
    generated = read_jsonl(run / "generated.jsonl")
    scored = read_jsonl(evaluation / "utterances.jsonl")
    require([r["id"] for r in generated] == [r["id"] for r in scored] == ids and
            set(result["summary"]) == {"adversarial"}, "Incomplete/extra generation/evaluation rows")
    require({p.stem for p in (run / "records").glob("*.json")} == set(ids), "Missing/extra records")
    for variant in ["clean", "adversarial"]:
        require({p.stem for p in (run / "audio" / variant).glob("*.wav")} == set(ids), "Missing/extra baseline WAVs")
    radii, seconds = [], []
    epsilon = config["waveform"]["epsilon"]
    for source, record, score in zip(rows, generated, scored):
        uid = source["id"]
        require(record == json.loads((run / "records" / f"{uid}.json").read_text()), "Record/index mismatch")
        require(record["text"] == source["text"] == score["reference"] and record["role"] == "validation" and
                record["language"] == "en" and record["input_audio_sha256"] == source["audio_sha256"] == digest(source["audio"]),
                "Changed reference, source or role")
        seed = (int(config.get("seed", 2026)) + int(hashlib.sha256(uid.encode()).hexdigest()[:8], 16)) % 2**32
        require(record["seed"] == seed and set(record["audio"]) == {"clean", "adversarial"}, "Changed seed/controls")
        for variant, asset in record["audio"].items():
            require(Path(asset["path"]).resolve() == run / "audio" / variant / f"{uid}.wav" and
                    digest(asset["path"]) == asset["sha256"] and asset["format"] == "WAV/FLOAT" and
                    asset["sample_rate"] == 16000 and math.isfinite(asset["peak"]), "Changed/invalid saved WAV")
            info = sf.info(asset["path"])
            require(info.frames == asset["samples"], "Saved WAV length mismatch")
        require(score["variant"] == "adversarial" and score["audio_sha256"] == record["audio"]["adversarial"]["sha256"],
                "Wrong evaluated WAV")
        for key, value in error_counts(source["text"], score["hypothesis"], "en").items():
            require(score[key] == value, "Incorrect lexical edit counts")
        require(set(score["quality"]) == {"dnsmos", "nisqa", "utmos"} and
                all(math.isfinite(v) for v in score["quality"].values()), "Missing/nonfinite quality")
        require([t["step"] for t in record["trace"]] == list(range(1, 51)), "Incomplete update trace")
        require(all(all(math.isfinite(v) for v in t.values()) and 0 <= t["linf"] <= epsilon + 1e-6
                    for t in record["trace"]), "Nonfinite/out-of-bound update trace")
        saved_clean, _ = sf.read(record["audio"]["clean"]["path"], dtype="float32")
        require(np.array_equal(saved_clean, load_audio(source["audio"]).numpy()), "Saved clean samples differ from source")
        radii.append(waveform_bound(record["audio"]["clean"]["path"], record["audio"]["adversarial"]["path"], epsilon))
        require(math.isfinite(record["generation_seconds"]) and record["generation_seconds"] > 0, "Invalid generation timing")
        seconds.append(record["generation_seconds"])
    summary = result["summary"]["adversarial"]
    require(summary["count"] == 100, "Wrong summary count")
    for key, value in aggregate_errors(scored).items():
        require(math.isclose(summary[key], value, rel_tol=1e-12), "Incorrect corpus aggregate")
    for metric in ["dnsmos", "nisqa", "utmos"]:
        require(math.isclose(summary["quality"][metric], sum(r["quality"][metric] for r in scored)/100, rel_tol=1e-12),
                "Incorrect quality aggregate")
    require(math.isclose(sum(seconds), metadata["generation_seconds"], rel_tol=1e-12), "Incorrect total generation timing")
    return {"status": "passed", "name": entry["name"], "method": entry["method"], "count": 100,
        "summary_sha256": digest(evaluation / "summary.json"), "generated_sha256": metadata["generated_sha256"],
        "utterances_sha256": result["utterances_sha256"], "generation_fingerprint": metadata["fingerprint"],
        "config_sha256": entry["config_sha256"], "validation_manifest_sha256": plan["validation_manifest_sha256"],
        "source_flacs_rehashed": 100, "saved_wavs_rehashed": 200, "finite_bounded_updates": 5000,
        "max_saved_linf": max(radii), "summary": summary,
        "scope": "Frozen validation IDs/config/raw surrogate/scorers, full source/WAV hashes, samples/range/L-infinity bounds, "
                 "50 finite bounded updates, references/edit counts/corpus WER/MOS. Models not rerun."}
