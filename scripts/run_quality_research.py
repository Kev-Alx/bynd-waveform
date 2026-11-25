"""Paper-budget WER/quality research, queued after the current defense suite.

Six disjoint-pilot candidates vary only unspecified NLL reduction and Adam LR.
Two candidates advance to validation by predeclared quality/attack criteria.
A separate fixed paper-budget control then evaluates the historical 999 IDs.
"""
from __future__ import annotations

import argparse
import copy
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from beyondwaveform.config import AttackConfig, SpectralConfig, load_config
from beyondwaveform.data import audit_disjoint
from beyondwaveform.evaluation import QualityScorer, aggregate_errors, error_counts
from beyondwaveform.io import atomic_json, digest, environment, load_audio, read_jsonl
from beyondwaveform.pipeline import config_fingerprint
from parallel_jobs import Supervisor, now

METRICS = ("dnsmos", "nisqa", "utmos")
QUALITY_TARGET = dict(zip(METRICS, (2.954, 3.526, 3.590)))
PAPER_WER = 75.43
DEFAULT_OUTPUT = ROOT / "reproduction-runs/paper-quality-research"


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def runtime():
    result = environment()
    for name in ("librosa", "onnxruntime", "resampy", "scipy", "soundfile"):
        result["packages"][name] = importlib.metadata.version(name)
    return result


def candidates(base):
    result = []
    for reduction in ("mean", "sum"):
        for rate in (0.02, 0.05, 0.1):
            config = copy.deepcopy(base)
            config["save_delta"] = True
            config["surrogate"]["nll_reduction"] = reduction
            config["attack"].update(rho=0.1, lambda_perc=1.0, steps=50, learning_rate=rate)
            result.append({"name": f"{reduction}-lr-{rate:g}", "config": config})
    return result


def promotions(rows):
    """No test/defense feedback: quality-first candidate plus attack comparator."""
    eligible = [r for r in rows if all(math.isfinite(r["quality"][m]) and r["quality"][m] > 2.5
                                     for m in METRICS) and math.isfinite(r["error_percent"])]
    require(bool(eligible), "No pilot candidate passes all three >2.5 quality gates")
    matched = [r for r in eligible if all(r["quality"][m] >= QUALITY_TARGET[m] for m in METRICS)]
    attack_key = lambda r: (r["error_percent"], r["quality"]["utmos"], r["name"])
    first = max(matched, key=attack_key) if matched else max(
        eligible, key=lambda r: (min(r["quality"][m] / QUALITY_TARGET[m] for m in METRICS), *attack_key(r)))
    remaining = [r for r in eligible if r["name"] != first["name"]]
    selected = [first["name"]]
    if remaining:
        selected.append(max(remaining, key=attack_key)["name"])
    return {"names": selected, "quality_target_available": bool(matched),
            "eligible": [r["name"] for r in eligible],
            "rule": "Highest WER meeting all Table 4 feature-quality targets; otherwise maximize weakest quality/target ratio. Also promote highest-WER remaining >2.5-gated comparator."}


def scorer_setup(cache, historical):
    """Relocate historical, hash-verified weights without changing HOME."""
    import requests
    expected = historical["quality_provenance"]["artifact_hashes"]
    sources = {
        "DNSMOS/DNSMOS/sig_bak_ovr.onnx": "https://raw.githubusercontent.com/microsoft/DNS-Challenge/master/DNSMOS/DNSMOS/sig_bak_ovr.onnx",
        "DNSMOS/DNSMOS/model_v8.onnx": "https://raw.githubusercontent.com/microsoft/DNS-Challenge/master/DNSMOS/DNSMOS/model_v8.onnx",
        "NISQA/nisqa.tar": "https://raw.githubusercontent.com/gabrielmittag/NISQA/master/weights/nisqa.tar",
    }
    hashes, provenance = {}, {}
    for relative, url in sources.items():
        wanted = next(v for k, v in expected.items() if k.endswith("/" + relative))
        path = cache / relative
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            old = Path("/tmp/beyondwaveform-quality-audit") / relative
            if old.is_file() and digest(old) == wanted:
                shutil.copy2(old, path)
            else:
                temporary = path.with_suffix(path.suffix + ".partial")
                print(f"Downloading scorer: {relative}", flush=True)
                with requests.get(url, stream=True, timeout=(15, 60)) as response:
                    response.raise_for_status()
                    with temporary.open("wb") as stream:
                        for chunk in response.iter_content(1024 * 1024):
                            stream.write(chunk)
                require(digest(temporary) == wanted, f"Downloaded scorer differs from historical hash: {relative}")
                temporary.replace(path)
        require(digest(path) == wanted, f"Scorer hash mismatch: {path}")
        hashes[str(path)] = wanted
        provenance[relative] = {"url": url, "sha256": wanted}
    utmos = ROOT / "artifacts/torchhub/checkpoints/utmos22_strong_step7459_v1.pt"
    wanted = next(v for k, v in expected.items() if k.endswith("/" + utmos.name))
    require(digest(utmos) == wanted, "Historical UTMOS checkpoint differs")
    hashes[str(utmos)] = wanted
    return {"cache": str(cache), "hashes": hashes, "download_sources": provenance,
            "runtime": runtime(), "dnsmos": {"channel": "OVRL", "personalized": False,
                "short_audio": "native repeat to 9.01 seconds", "window_hop_seconds": 1},
            "nisqa": {"channel": "MOS", "segmentation": "native checkpoint parameters"},
            "utmos": {"variant": "utmos22_strong", "channel": "native calibrated mean"},
            "input": "saved mono float32 16 kHz WAV; no amplitude normalization/clipping or defense",
            "aggregation": "arithmetic utterance mean"}


def configure_scorers(manifest):
    for path, expected in manifest["hashes"].items():
        require(digest(path) == expected, f"Changed quality weights: {path}")
    cache = Path(manifest["cache"])
    importlib.import_module("torchmetrics.functional.audio.dnsmos").DNSMOS_DIR = str(cache / "DNSMOS")
    importlib.import_module("torchmetrics.functional.audio.nisqa").NISQA_DIR = str(cache / "NISQA")


def prepare(output, defense):
    require(not (output / "plan.json").exists(), "Research plan already exists")
    output.mkdir(parents=True, exist_ok=True)
    old_run = ROOT / "reproduction-runs/english-feature-test-999"
    old = read(old_run / "run.json")
    historical = read(old_run / "surrogate-eval/summary.json")
    freeze = read(ROOT / "configs/english_selected.freeze.json")
    manifests = {"pilot": ROOT / "data/manifests/reconstruction-pilot.jsonl",
                 "validation": Path(freeze["validation_manifest"]), "test": Path(freeze["test_manifest"])}
    audit_disjoint(list(manifests.values()))
    counts = {"pilot": 12, "validation": 100, "test": 999}
    pools = {}
    for role, path in manifests.items():
        rows = read_jsonl(path)
        require(len(rows) == counts[role] and all(r["role"] == role and r["language"] == "en" for r in rows),
                f"Unexpected {role} pool")
        for row in rows:
            require(digest(row["audio"]) == row["audio_sha256"], f"Changed source clip: {row['id']}")
        pools[role] = {"manifest": str(path), "sha256": digest(path), "ids": [r["id"] for r in rows]}
    require(pools["test"]["ids"] == freeze["test_ids"] == old["ids"], "Changed historical test subset")
    base = load_config(ROOT / "configs/english_lr005.yaml")
    require(base["attack"]["rho"] == 0.1 and base["attack"]["lambda_perc"] == 1 and
            base["attack"]["steps"] == 50 and base["spectral"]["sample_rate"] == 16000,
            "Expected stated paper settings and actual audio sample rate")
    entries = candidates(base)
    files = [Path(__file__).resolve(), ROOT / "configs/english_lr005.yaml",
             ROOT / "configs/english_selected.freeze.json", old_run / "run.json",
             old_run / "surrogate-eval/summary.json", old_run / "surrogate-eval/utterances.jsonl",
             defense / "plan.json", *manifests.values()]
    for row in entries:
        path = output / "configs" / (row["name"] + ".json")
        atomic_json(path, row["config"])
        path.chmod(0o444)
        row["config_path"] = str(path)
        files.append(path)
    quality = scorer_setup(ROOT / "artifacts/quality-research-scorers", historical)
    configure_scorers(quality)
    # Confirm internal scoring reproducibility before comparing candidate MOS.
    scorer = QualityScorer(str(ROOT / "artifacts"))
    record = read_jsonl(old_run / "generated.jsonl")[0]
    stored = [json.loads(line) for line in (old_run / "surrogate-eval/utterances.jsonl").read_text(encoding="utf-8").splitlines()]
    smoke = []
    for variant in ("clean", "adversarial"):
        print(f"Verifying historical {variant} quality fixture", flush=True)
        asset = record["audio"][variant]
        require(digest(asset["path"]) == asset["sha256"], "Changed quality fixture WAV")
        measured = scorer(load_audio(asset["path"]))
        prior = next(r["quality"] for r in stored if r["id"] == record["id"] and r["variant"] == variant)
        require(all(abs(measured[m] - prior[m]) < 1e-5 for m in METRICS), "Quality fixture did not reproduce historical scores")
        smoke.append({"variant": variant, "id": record["id"], "measured": measured, "historical": prior})
    atomic_json(output / "quality-smoke.json", smoke)
    files.append(output / "quality-smoke.json")
    files.extend(p for p in (ROOT / "beyondwaveform").rglob("*.py"))
    files.extend(ROOT / "scripts" / name for name in
                 ("generate.py", "parallel_jobs.py", "run_logged.py", "discord_gpu_report.py"))
    for folder in (ROOT / "artifacts/sources").iterdir():
        if folder.is_dir() and (folder.name.startswith("knn-vc-") or folder.name.startswith("SpeechMOS-")):
            files.extend(p for p in folder.rglob("*") if p.suffix in {".py", ".json", ".yaml", ".yml"})
    hashes = {str(p.resolve()): digest(p) for p in files}
    for path, expected in freeze["artifact_hashes"].items():
        if path.startswith(str(ROOT / "artifacts")):
            require(digest(path) == expected, f"Historical model changed: {path}")
            hashes[path] = expected
    hashes.update(quality["hashes"])
    plan = {"created_at": now(), "pools": pools, "candidates": entries, "quality": quality,
            "runtime": runtime(), "frozen_files": hashes, "defense_dependency": str(defense),
            "defense_plan_sha256": digest(defense / "plan.json"), "feature_artifacts": old["features"],
            "base_surrogate_artifacts": old["surrogate"], "paper_quality_targets": QUALITY_TARGET,
            "promotion_rule": promotions([{ "name": "rule-example", "error_percent": 0,
                                         "quality": {m: 4.0 for m in METRICS}}])["rule"],
            "fixed_test_control": "mean-lr-0.05", "test_control_selection": "Predeclared before new pilot/validation outcomes; no test-based selection.",
            "scope": "Independent paper-budget comparison, not recovered author optimizer/model/scoring parity."}
    atomic_json(output / "plan.json", plan)
    (output / "plan.json").chmod(0o444)
    print(f"Frozen six candidates, three disjoint pools, scorer fixtures, and {len(hashes)} input hashes.", flush=True)
    return plan


def check(plan):
    require(runtime() == plan["runtime"], "Research runtime changed")
    for path, expected in plan["frozen_files"].items():
        require(digest(path) == expected, f"Frozen research input changed: {path}")
    validate_candidate_files(plan["candidates"])


def validate_candidate_files(entries):
    """Validate the serialized configs consumed by the actual generation CLI."""
    for entry in entries:
        config = load_config(entry["config_path"])
        require(config == entry["config"], f"Saved config changed values/types: {entry['name']}")
        AttackConfig(**config["attack"])
        spectral = SpectralConfig(**config["spectral"])
        require(spectral.sample_rate == 16000, "Spectral/audio sample rates disagree")


def defense_check_plan(plan, defense_plan):
    """Check completed defenses against preserved historical source snapshots.

    An explicitly archived config loader can differ from current research code;
    the completed defense plan and all its data/config/weights stay immutable.
    """
    import yaml
    effective = copy.deepcopy(defense_plan)
    for original, archived in plan.get("historical_defense_sources", {}).items():
        require(effective["frozen_files"].get(original) == archived["sha256"], "Historical source hash differs")
        require(digest(archived["snapshot"]) == archived["sha256"], "Historical source snapshot changed")
        del effective["frozen_files"][original]
        effective["frozen_files"][archived["snapshot"]] = archived["sha256"]
    for setting in defense_plan["settings"]:
        path = Path(setting["target"])
        require(yaml.safe_load(path.read_text(encoding="utf-8")) == load_config(path),
                "Config-loader repair changes a completed defense configuration")
    return effective


def evaluate_stage(plan, run):
    from beyondwaveform.pipeline import evaluate
    metadata = read(run / "run.json")
    configure_scorers(plan["quality"])
    result = evaluate(run, {"kind": "whisper", **metadata["config"]["surrogate"]}, run / "evaluation",
                      variants=["clean", "reconstruction", "adversarial"], quality=True,
                      artifact_root=str(ROOT / "artifacts"))
    result["research_scorer_manifest"] = plan["quality"]
    result["evaluation_environment"] = runtime()
    atomic_json(run / "evaluation/summary.json", result)


def verify_stage(plan, run, config, role, audit_name="verification.json"):
    """Audit scores and independently recompute final delta radii with WavLM."""
    import torch
    from tqdm import tqdm
    from beyondwaveform.models.feature import load_feature_models
    from beyondwaveform.attacks import relative_radius
    pool = plan["pools"][role]
    source = read_jsonl(pool["manifest"])
    metadata = read(run / "run.json")
    result = read(run / "evaluation/summary.json")
    records = read_jsonl(run / "generated.jsonl")
    identity = {"config": config, "manifest_sha256": pool["sha256"], "ids": pool["ids"]}
    require(metadata["status"] == "complete" and metadata["count"] == len(source) and
            metadata["config"] == config and metadata["fingerprint"] == config_fingerprint(identity), "Changed generation identity")
    expected_surrogate = {**plan["base_surrogate_artifacts"], "nll_reduction": config["surrogate"]["nll_reduction"]}
    if "nll_transcript_policy" in config["surrogate"]:
        expected_surrogate["nll_transcript_policy"] = config["surrogate"]["nll_transcript_policy"]
    require(metadata["features"] == plan["feature_artifacts"] and metadata["surrogate"] == expected_surrogate,
            "Changed generation artifacts")
    require(result["target_artifacts"] == expected_surrogate and result["target"] == {"kind": "whisper", **config["surrogate"]} and
            result["evaluation_scope"] == "surrogate" and result["roles"] == [role] and result["metric"] == "WER" and
            result["generation_fingerprint"] == metadata["fingerprint"] and result["manifest_sha256"] == pool["sha256"] and
            Path(result["generation_run"]).resolve() == run.resolve(), "Changed evaluation protocol")
    require(result["research_scorer_manifest"] == plan["quality"] and result["evaluation_environment"] == plan["runtime"],
            "Changed scorer/runtime contract")
    require(digest(run / "generated.jsonl") == metadata["generated_sha256"] and
            digest(run / "evaluation/utterances.jsonl") == result["utterances_sha256"], "Altered output index")
    require([r["id"] for r in records] == result["evaluation_ids"] == pool["ids"], "Changed evaluated IDs")
    rows = [json.loads(line) for line in (run / "evaluation/utterances.jsonl").read_text(encoding="utf-8").splitlines()]
    variants = ("clean", "reconstruction", "adversarial")
    pairs = {(r["id"], r["variant"]): r for r in rows}
    require(len(pairs) == len(rows) == len(source)*3 and set(pairs) == {(uid, v) for uid in pool["ids"] for v in variants},
            "Duplicate/missing score rows")
    encoder, _, features = load_feature_models(config["features"], "cuda")
    require(features == plan["feature_artifacts"], "Changed radius-verification encoder")
    radii = []
    for original, record in tqdm(list(zip(source, records)), desc="Auditing saved deltas/WAVs", unit="clips"):
        require(record["text"] == original["text"] and record["input_audio_sha256"] == original["audio_sha256"], "Changed source reference")
        for variant in variants:
            asset, row = record["audio"][variant], pairs[(record["id"], variant)]
            require(digest(asset["path"]) == asset["sha256"] == row["audio_sha256"], "Changed saved waveform")
            require(row["reference"] == original["text"], "Changed score reference")
            for key, value in error_counts(original["text"], row["hypothesis"], "en").items():
                require(row[key] == value, f"Incorrect lexical {key}")
        delta = torch.load(run / f"{record['id']}.delta.pt", map_location="cuda", weights_only=True)
        with torch.no_grad():
            q = encoder(load_audio(original["audio"]).cuda()).detach()
        require(delta.shape == q.shape and torch.isfinite(delta).all(), "Invalid saved feature delta")
        radius = relative_radius(delta, q, config["attack"]["eps"]).item()
        require(radius <= 0.100001 and abs(radius-record["trace"][-1]["radius_after_update"]) < 1e-6,
                "Saved delta does not satisfy/match the stated rho=0.1 constraint")
        require(len(record["trace"]) == 50 and all(math.isfinite(value) for step in record["trace"] for value in step.values()),
                "Incomplete/nonfinite optimization trace")
        radii.append(radius)
    for variant in variants:
        subset = [r for r in rows if r["variant"] == variant]
        summary = result["summary"][variant]
        require(summary["count"] == len(source), "Wrong aggregate count")
        for key, value in aggregate_errors(subset).items():
            require(summary[key] == value, f"Wrong aggregate {key}")
        for metric in METRICS:
            values = [r["quality"][metric] for r in subset]
            require(all(math.isfinite(v) for v in values) and abs(sum(values)/len(values)-summary["quality"][metric]) < 1e-10,
                    f"Wrong aggregate {metric}")
    atomic_json(run / audit_name, {"verified_at": now(), "summary_sha256": digest(run / "evaluation/summary.json"),
                "rows": len(rows), "count": len(source), "max_independent_delta_radius": max(radii),
                "summary": result["summary"], "role": role, "config": config})


def write_report(output, results, status):
    atomic_json(output / "results.json", {"status": status, "updated_at": now(), "results": results,
                "paper_wer": PAPER_WER, "paper_quality_targets": QUALITY_TARGET})
    lines = ["# Paper-budget WER and sound-quality research", "", f"Status: **{status}**.", "",
             "All new candidates use rho=0.1, lambda=1, 50 steps, and the existing compatible encoder/vocoder. Saved deltas are independently checked against re-extracted clean features.", "",
             "| Pool | Candidate | WER % | DNSMOS | NISQA | UTMOS | All paper quality targets met |",
             "|---|---|---:|---:|---:|---:|---|"]
    for row in results:
        q = row["quality"]
        lines.append(f"| {row['role']} | {row['name']} | {row['error_percent']:.4f} | {q['dnsmos']:.4f} | {q['nisqa']:.4f} | {q['utmos']:.4f} | {all(q[m] >= QUALITY_TARGET[m] for m in METRICS)} |")
    lines += ["", "Paper anchors: WER 75.43%; DNSMOS/NISQA/UTMOS 2.954 / 3.526 / 3.590. Absolute quality targets are an explicit research preference; scorer/model/pool parity with the authors remains unresolved.", "",
              "Pilot results guide two validation promotions. The fixed 999-clip mean-NLL/LR=0.05 control was predeclared before these results; no test or defended-target feedback selects it. Historical test IDs remain unchanged. No new selected-config test is automatically launched.", "",
              "Compared with sum NLL, mean NLL weights the perceptual term by transcript token count. Equal numeric lambda explores that unresolved author choice; it does not imply equal objectives or identical optimizer trajectories.", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def wait_defenses(plan, supervisor):
    dependency = Path(plan["defense_dependency"])
    while True:
        job, progress = read(dependency / "job.json"), read(dependency / "progress.json")
        require(job["status"] not in {"failed", "interrupted"}, "Defense suite failed; research must wait for its repair")
        if job["status"] == progress["status"] == "complete":
            from run_defense_suite import check as check_defenses, verify as verify_defense
            defense_plan = read(dependency / "plan.json")
            require(digest(dependency / "plan.json") == plan["defense_plan_sha256"], "Defense plan changed")
            check_defenses(defense_check_plan(plan, defense_plan))
            for setting in defense_plan["settings"]:
                verify_defense(defense_plan, dependency / setting["name"], load_config(setting["target"]))
            supervisor.progress(defense_baseline_verified=True)
            return
        supervisor.progress(phase="waiting-for-defense-baseline", defense_phase=progress.get("phase"))
        print(f"Heartbeat: waiting for defenses; active phase {progress.get('phase')}", flush=True)
        time.sleep(30)


def run_case(output, plan, candidate, role, results, supervisor):
    run = output / ("paper-setting-test-999" if role == "test" else role + "/" + candidate["name"])
    config_path = candidate["config_path"]
    check(plan)
    if not (run / "evaluation/summary.json").exists():
        supervisor.phase(f"{role}-{candidate['name']}-generate", [sys.executable, "-u", "scripts/generate.py",
            "--config", config_path, "--manifest", plan["pools"][role]["manifest"], "--output", str(run)], threads=4)
        supervisor.phase(f"{role}-{candidate['name']}-evaluate", [sys.executable, "-u", str(Path(__file__).resolve()),
            "--output", str(output), "--stage", "evaluate", "--run", str(run)], threads=4)
    supervisor.phase(f"{role}-{candidate['name']}-verify", [sys.executable, "-u", str(Path(__file__).resolve()),
        "--output", str(output), "--stage", "verify", "--run", str(run), "--config", config_path, "--role", role], threads=4)
    audit = read(run / "verification.json")
    summary = audit["summary"]["adversarial"]
    row = {"name": candidate["name"], "role": role, "run": str(run), **summary,
           "verification_sha256": digest(run / "verification.json"), "summary_sha256": audit["summary_sha256"]}
    results.append(row)
    write_report(output, results, "running")
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--defenses", type=Path, default=ROOT / "reproduction-runs/english-defenses-999")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--stage", choices=["evaluate", "verify"])
    parser.add_argument("--run", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--role", choices=["pilot", "validation", "test"])
    args = parser.parse_args()
    output = args.output.resolve()
    plan = prepare(output, args.defenses.resolve()) if args.prepare else read(output / "plan.json")
    check(plan)
    if args.prepare or args.check_only:
        print("Research preflight passed.", flush=True)
        return
    if args.stage:
        require(args.run is not None, "Stage requires --run")
        if args.stage == "evaluate":
            evaluate_stage(plan, args.run)
        else:
            require(args.config is not None and args.role is not None, "Verification requires config and role")
            verify_stage(plan, args.run, load_config(args.config), args.role)
        return
    supervisor, results = Supervisor(output), []
    try:
        write_report(output, results, "waiting-for-defense-baseline")
        wait_defenses(plan, supervisor)
        pilot = [run_case(output, plan, candidate, "pilot", results, supervisor) for candidate in plan["candidates"]]
        promotion = promotions(pilot)
        promotion_path = output / "pilot-promotion.json"
        if promotion_path.exists():
            require(read(promotion_path) == promotion, "Pilot promotion differs on resume")
        else:
            atomic_json(promotion_path, promotion)
            promotion_path.chmod(0o444)
        for name in promotion["names"]:
            run_case(output, plan, next(c for c in plan["candidates"] if c["name"] == name), "validation", results, supervisor)
        control = next(c for c in plan["candidates"] if c["name"] == plan["fixed_test_control"])
        run_case(output, plan, control, "test", results, supervisor)
        check(plan)
        write_report(output, results, "complete")
        supervisor.finish(results=str(output / "results.json"))
    except BaseException as error:
        write_report(output, results, "failed")
        supervisor.fail(error)
        raise


if __name__ == "__main__":
    main()
