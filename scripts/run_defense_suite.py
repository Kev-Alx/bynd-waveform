"""Freeze, run, and audit available defenses on the unchanged 999-clip attack.

An existing LPF job may be adopted after checking its command and completion.
WaveGuard settings are predeclared sensitivity tests, not a recovered Table 1
recipe. Unavailable defenses stay explicit in the plan and report.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from beyondwaveform.config import load_config
from beyondwaveform.evaluation import aggregate_errors, error_counts
from beyondwaveform.io import atomic_json, digest, environment, read_jsonl
from beyondwaveform.references import DEFENSES, TABLE_1
from parallel_jobs import Supervisor, now

VARIANTS = ("clean", "reconstruction", "adversarial")
UNAVAILABLE = {
    "CE-AT": "No defended Whisper-small weights or ASR training recipe supplied.",
    "DT-AT / DE-AT": "Inconsistent paper label; no trajectory objective, recipe, or weights supplied.",
    "DMW": "Cited instance-reweighting method does not identify trained ASR weights or its adaptation.",
    "TI": "Cited translation-invariant work describes an attack; the defended ASR recipe and weights are unidentified.",
    "AudioPure": "Official released configuration/model targets one-second SC09 keyword audio. Sentence purification, checkpoint, time level, segmentation, and aggregation used in Table 1 are unidentified.",
    "PVP Vote": "Paper cites Acoustic-decoy, not the recovered Precision-Varying Prediction paper. Vote definition/precisions/ties are absent; Titan RTX lacks native BF16 support.",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def runtime():
    value = environment()
    for name in ("librosa", "resampy", "scipy", "soundfile"):
        value["packages"][name] = importlib.metadata.version(name)
    return value


def prepare(output, run):
    output.mkdir(parents=True, exist_ok=True)
    require(not (output / "plan.json").exists(), "Plan already exists; use --check-only or run it")
    meta = read_json(run / "run.json")
    raw = read_json(run / "surrogate-eval/summary.json")
    records = read_jsonl(run / "generated.jsonl")
    freeze = read_json(ROOT / "configs/english_selected.freeze.json")
    require(meta["status"] == "complete" and meta["count"] == len(records) == 999,
            "Expected the complete historical 999-clip run")
    require(meta["config"] == freeze["config"] and meta["ids"] == freeze["test_ids"] and
            meta["manifest_sha256"] == freeze["test_manifest_sha256"], "Source differs from historical freeze")
    require(meta["generated_sha256"] == digest(run / "generated.jsonl"), "Altered generated index")
    require(meta["surrogate"]["sha256"] == digest(meta["surrogate"]["checkpoint"]), "Altered surrogate")
    base = {"kind": "whisper", **meta["config"]["surrogate"]}
    settings = [{"name": "lpf-6000", "paper_label": "LPF",
                 "target": str(ROOT / "configs/targets/lpf_example.yaml"),
                 "note": "Existing independent two-pole biquad, 6000 Hz, Q=0.70710678; author filter unidentified."}]
    for hz in (8000, 2000):
        name = f"waveguard-downup-{hz}"
        config = {"name": name, "kind": "extension", "factory": "beyondwaveform.defenses:WaveGuardDownUpASR",
                  "kwargs": {"target_config": base, "checkpoint_sha256": meta["surrogate"]["sha256"],
                             "downsample_hz": hz, "device": "cuda"},
                  "protocol": {"upstream_revision": "4320cfe12d0452e7f34f2903a4e302780662f5e1",
                               "resampler": "librosa/kaiser_best/fix=True/scale=False",
                               "io": "clip/truncate PCM16 then divide by 32768; native resampled length",
                               "scope": "WaveGuard resampling preprocessing only; no detection threshold",
                               "paper_parity": "independent settings; Table 1 transformation unspecified"}}
        path = output / "configs" / f"{name}.json"
        atomic_json(path, config)  # JSON is a YAML subset accepted by load_config.
        path.chmod(0o444)
        settings.append({"name": name, "paper_label": "WaveGuard", "target": str(path),
                         "note": "Upstream default 8 kHz and predeclared 2 kHz sensitivity; neither selected from test WER."})
    files = [run / "run.json", run / "generated.jsonl", run / "surrogate-eval/summary.json",
             run / "surrogate-eval/utterances.jsonl", Path(freeze["test_manifest"]),
             ROOT / "configs/english_selected.freeze.json", Path(freeze["config_path"]),
             Path(meta["surrogate"]["checkpoint"]), Path(__file__).resolve(),
             *[ROOT / f"beyondwaveform/{p}.py" for p in
               ("defenses", "pipeline", "evaluation", "io", "config", "losses", "references", "cli",
                "models/targets", "models/whisper")], ROOT / "scripts/parallel_jobs.py",
             ROOT / "scripts/run_logged.py", ROOT / "scripts/evaluate.py",
             *[Path(s["target"]) for s in settings],
             *[p for p in (ROOT / "verification/defense-research").rglob("*") if p.is_file()]]
    plan = {"created_at": now(), "generation_run": str(run), "count": 999,
            "generation_fingerprint": meta["fingerprint"], "ids": [r["id"] for r in records],
            "variants": list(VARIANTS), "runtime": runtime(), "settings": settings,
            "unavailable": UNAVAILABLE, "frozen_files": {str(p.resolve()): digest(p) for p in files},
            "selection": "No target/defense/test feedback used to select attacks, subsets, or defense settings.",
            "quality": "Reuse historical pre-defense quality of identical saved WAVs; no post-defense MOS is claimed.",
            "scope": "English Whisper-small current feature attack; partial independent defense coverage, not full Table 1."}
    require(raw["evaluation_ids"] == plan["ids"], "Raw evaluation IDs differ")
    atomic_json(output / "plan.json", plan)
    (output / "plan.json").chmod(0o444)
    return plan


def check(plan):
    for path, expected in plan["frozen_files"].items():
        require(digest(path) == expected, f"Frozen input changed: {path}")
    require(runtime() == plan["runtime"], "Evaluation runtime differs from frozen plan")


def verify(plan, evaluation, target, hash_audio=False):
    """Recompute every lexical edit count and corpus aggregate, not just totals."""
    run = Path(plan["generation_run"])
    records = read_jsonl(run / "generated.jsonl")
    meta = read_json(run / "run.json")
    result = read_json(evaluation / "summary.json")
    require(result["target"] == target, "Different target/defense config")
    require(result["evaluation_ids"] == plan["ids"] and result["roles"] == ["test"] and
            result["manifest_sha256"] == meta["manifest_sha256"] and result["metric"] == "WER" and
            result["generation_fingerprint"] == plan["generation_fingerprint"] and
            Path(result["generation_run"]).resolve() == run, "Evaluation source/protocol differs")
    require(result["normalization"] == "NFKC/lowercase/punctuation-to-space; Chinese whitespace removed",
            "Changed text scoring policy")
    if target["kind"] == "whisper":
        require(result["target_artifacts"] == meta["surrogate"], "Different Whisper artifacts/decoder")
    else:
        require(result["target_artifacts"] == target, "Changed extension provenance")
    path = evaluation / "utterances.jsonl"
    require(digest(path) == result["utterances_sha256"], "Altered evaluation rows")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    pairs = {(r["id"], r["variant"]): r for r in rows}
    require(len(pairs) == len(rows) == 2997 and set(pairs) ==
            {(uid, v) for uid in plan["ids"] for v in VARIANTS}, "Duplicate/missing control rows")
    require(set(result["summary"]) == set(VARIANTS), "Missing summary controls")
    for record in records:
        for variant in VARIANTS:
            row, asset = pairs[(record["id"], variant)], record["audio"][variant]
            require(row["reference"] == record["text"] and row["audio_sha256"] == asset["sha256"],
                    "Changed reference or saved audio")
            if hash_audio:
                require(digest(asset["path"]) == asset["sha256"], "Saved waveform changed")
            for key, expected in error_counts(record["text"], row["hypothesis"], "en").items():
                require(row[key] == expected, f"Incorrect {key} in {row['id']}/{variant}")
    for variant in VARIANTS:
        summary = result["summary"][variant]
        require(summary["count"] == 999, "Wrong variant count")
        for key, expected in aggregate_errors([r for r in rows if r["variant"] == variant]).items():
            require(summary[key] == expected, f"Incorrect corpus {key}")
        require(summary["reference_units"] == 20310, "Changed reference denominator")
    audit = {"verified_at": now(), "summary_sha256": digest(evaluation / "summary.json"),
             "utterances_sha256": digest(path), "rows": len(rows), "count_per_variant": 999,
             "rehash_saved_audio": hash_audio, "target": target, "summary": result["summary"]}
    if evaluation != run / "surrogate-eval":
        atomic_json(evaluation / "verification.json", audit)
    return audit


def report(output, plan, audits, status):
    rows = []
    for name, audit in audits.items():
        label = "Raw" if name == "raw" else next(s["paper_label"] for s in plan["settings"] if s["name"] == name)
        summary = audit["summary"]
        clean, attack = [summary[v]["error_percent"] for v in ("clean", "adversarial")]
        rows.append({"name": name, "paper_label": label, "summary": summary,
                     "attack_minus_clean_pp": attack-clean,
                     "paper_clean_wer": TABLE_1["en"]["Clean"][DEFENSES.index(label)],
                     "paper_attack_wer": TABLE_1["en"]["Feature attack"][DEFENSES.index(label)],
                     "verification": audit})
    atomic_json(output / "results.json", {"status": status, "updated_at": now(), "rows": rows,
                "unavailable": plan["unavailable"], "plan_sha256": digest(output / "plan.json"),
                "complete_paper_defense_suite": False})
    lines = ["# Current-implementation defense tests", "", f"Status: **{status}**. English Whisper-small; frozen current attack and original 999 IDs.", "",
             "Every completed condition includes 999 clean, reconstruction, and attack clips / 20,310 reference words. WER is corpus edit totals divided by reference words.", "",
             "| Condition | Clean WER % | Reconstruction WER % | Attack WER % | Attack − clean pp | Paper clean / attack % |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        s = row["summary"]
        lines.append(f"| {row['name']} | {s['clean']['error_percent']:.6f} | {s['reconstruction']['error_percent']:.6f} | {s['adversarial']['error_percent']:.6f} | {row['attack_minus_clean_pp']:.6f} | {row['paper_clean_wer']:.2f} / {row['paper_attack_wer']:.2f} |")
    lines += ["", "Paper columns are descriptive anchors across unmatched protocols. LPF and WaveGuard settings are independent choices; no exact reproduction claim.", "",
              "WaveGuard here uses its official down/up resampling transformation and PCM16 behavior, with modern librosa explicitly set to the historical kaiser_best filter. 8 kHz is the upstream default; 2 kHz is a predeclared stronger sensitivity condition. Neither setting was selected from test results. This is preprocessing WER, not the complete WaveGuard consistency detector/AUC.", "",
              "Quality of the unchanged exported attack WAVs remains DNSMOS/NISQA/UTMOS 2.963822 / 3.871132 / 2.792377. The paper reports 2.954 / 3.526 / 3.590. Post-defense quality is not measured.", "", "## Unavailable Table 1 conditions", ""]
    lines += [f"- **{name}**: {reason}" for name, reason in plan["unavailable"].items()]
    lines += ["", "Research and primary-source provenance: [defense review](../../docs/DEFENSE_RESEARCH.md).", "",
              "No attack tuning or subset changes occur in this job. Next research stage follows defense evaluation: paper-budget rho=0.1/lambda=1 control, validation-only optimizer/NLL/transcript sensitivity, and quality-aware selection. Any new subset must be seeded and frozen before measuring attacks; retain this historical pool for paired comparisons.", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def wait_for_adopted_lpf(output, setting, supervisor):
    folder = output / setting["name"]
    job_path = folder / "job.json"
    if not job_path.exists():
        return
    expected = ["/opt/venv/bin/python", "-u", "scripts/evaluate.py", "--run",
                "reproduction-runs/english-feature-test-999", "--target", "configs/targets/lpf_example.yaml",
                "--output", str(folder)]
    while True:
        job = read_json(job_path)
        require(job["command"] == expected and job["cwd"] == str(ROOT), "Unrecognized adopted LPF job")
        if job["status"] != "running":
            require(job["status"] == "complete" and job["exit_code"] == 0, "Adopted LPF job failed; investigate before resuming")
            require((folder / "summary.json").is_file() and (folder / "utterances.jsonl").is_file(),
                    "Completed LPF job has missing results")
            return
        for key in ("supervisor_pid", "child_pid"):
            os.kill(job[key], 0)
        supervisor.progress(phase="waiting-for-existing-lpf", lpf_job=job)
        print("Heartbeat: existing 999-clip LPF evaluation is running; see lpf-6000.log", flush=True)
        time.sleep(30)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reproduction-runs/english-defenses-999")
    parser.add_argument("--run", type=Path, default=ROOT / "reproduction-runs/english-feature-test-999")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    output, run = args.output.resolve(), args.run.resolve()
    plan = prepare(output, run) if args.prepare else read_json(output / "plan.json")
    check(plan)
    audits = {"raw": verify(plan, run / "surrogate-eval", {"kind": "whisper", **read_json(run / "run.json")["config"]["surrogate"]}, hash_audio=True)}
    if args.prepare or args.check_only:
        report(output, plan, audits, "prepared" if args.prepare else "preflight-passed")
        print("Frozen source, all 2,997 WAV hashes, raw scores, targets and runtime verified.", flush=True)
        return
    supervisor = Supervisor(output)
    try:
        report(output, plan, audits, "running")
        for setting in plan["settings"]:
            check(plan)
            folder = output / setting["name"]
            if setting["name"] == "lpf-6000":
                wait_for_adopted_lpf(output, setting, supervisor)
            if not (folder / "summary.json").exists():
                supervisor.phase(setting["name"], ["/opt/venv/bin/python", "-u", "scripts/evaluate.py",
                    "--run", str(run), "--target", setting["target"], "--output", str(folder)], threads=4)
            audits[setting["name"]] = verify(plan, folder, load_config(setting["target"]))
            report(output, plan, audits, "running")
            print(f"Verified {setting['name']}: {audits[setting['name']]['summary']}", flush=True)
        check(plan)
        report(output, plan, audits, "available-conditions-complete")
        supervisor.finish(completed=list(audits), complete_paper_defense_suite=False)
    except BaseException as error:
        report(output, plan, audits, "failed")
        supervisor.fail(error)
        raise


if __name__ == "__main__":
    main()
