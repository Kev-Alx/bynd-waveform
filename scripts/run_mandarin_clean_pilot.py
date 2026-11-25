"""Score a fixed, speaker-balanced clean pilot and produce a Markdown report."""
import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import fcntl
import json
from pathlib import Path
import random
import sys
import time
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from beyondwaveform.evaluation import aggregate_errors, error_counts, normalize_text
from beyondwaveform.io import atomic_json, digest, load_audio, read_jsonl, seed_everything, write_jsonl
from beyondwaveform.models.whisper import load_whisper

TAIPEI = timezone(timedelta(hours=8))


def distance(a, b):
    """Independent Levenshtein distance, for auditing the scorer's counts."""
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


def independent_normalize(text):
    text = unicodedata.normalize("NFKC", text).lower()
    return "".join(c for c in text if not c.isspace() and not unicodedata.category(c).startswith("P"))


def freeze(path, value):
    if path.exists():
        assert json.loads(path.read_text()) == value, f"Frozen inputs changed: {path}"
    else:
        atomic_json(path, value)


def run(args):
    import torch
    import jiwer
    sys.path.insert(0, str(args.tokenizer_path.resolve()))
    import jieba
    assert jieba.__version__ == "0.42.1"
    torch.set_num_threads(8)
    seed_everything(2026)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    base = ROOT / "reproduction-runs/mandarin-cer-full-dev"
    source = args.manifest or base / "manifests/checkpoint-pilot-48.jsonl"
    original = read_jsonl(source)
    if args.manifest:
        selected = original
        pool = {row["id"]: row for row in read_jsonl(base / "manifests/checkpoint-validation.jsonl")}
        assert all(pool.get(row["id"]) == row for row in selected), "Use unchanged held-out validation rows"
        selection = args.selection_note or "Provided frozen validation manifest, prepared before decoding"
    else:
        by_speaker = {speaker: sorted((r for r in original if r["speaker"] == speaker), key=lambda r: (r["duration"], r["id"]))
                      for speaker in sorted({r["speaker"] for r in original})}
        assert len(original) == 48 and len(by_speaker) == 8 and all(len(rows) == 6 for rows in by_speaker.values())
        rng = random.Random(2026)
        selected = [rng.choice(rows[start:start + 2]) for rows in by_speaker.values() for start in [0, 2, 4]]
        selected.sort(key=lambda r: r["id"])
        assert len(selected) == 24
        selection = "Seed 2026; one recording from each short/medium/long pair per each of eight existing held-out pilot speakers; selected before decoding"
    count = len(selected)
    assert count == len({r["id"] for r in selected}) and count > 0
    train = read_jsonl(base / "manifests/finetune.jsonl")
    for field in ["id", "speaker", "audio_sha256"]:
        assert not {r[field] for r in selected} & {r[field] for r in train}, f"Training overlap: {field}"
    assert all(r["role"] == "validation" and r["split"] == "train" for r in selected)
    truth_path = ROOT / "data/data_aishell/transcript/aishell_transcript_v0.8.txt"
    truth = dict(line.split(maxsplit=1) for line in truth_path.read_text().splitlines() if line.strip())
    assert all(r["text"] == truth[r["id"]].strip() for r in selected)
    metadata_path = args.training_metadata or base / "training/training.json"
    training = json.loads(metadata_path.read_text())
    snapshots = list(training.get("epoch_checkpoints", []))
    if training.get("checkpoint"):
        snapshots.append(training["checkpoint"])
    checkpoint = next(row for row in snapshots if row["epoch"] == args.epoch
                      and Path(row["path"]).resolve() == args.checkpoint.resolve())
    assert Path(checkpoint["path"]).resolve() == args.checkpoint.resolve()
    assert digest(args.checkpoint) == checkpoint["sha256"]
    assert any(row["epoch"] == args.epoch for row in training["history"])
    target_config = {"checkpoint": str(args.checkpoint.resolve()), "language": "zh", "nll_reduction": "mean",
                     "decoding": {"beam_size": 5, "temperature": 0., "without_timestamps": True}}
    plan = {"status": "frozen", "epoch": args.epoch, "checkpoint": checkpoint, "target": target_config,
            "training_metadata": str(metadata_path.resolve()), "training_metadata_sha256": digest(metadata_path),
            "count": count, "speaker_counts": dict(Counter(r["speaker"] for r in selected)),
            "selection": selection,
            "source_pilot": str(source), "source_pilot_sha256": digest(source),
            "training_manifest_sha256": digest(base / "manifests/finetune.jsonl"), "official_transcripts_sha256": digest(truth_path),
            "pilot_ids": [r["id"] for r in selected], "source_script_sha256": digest(__file__),
            "wer_tokenizer": {"name": "jieba", "version": jieba.__version__, "HMM": False,
                              "dictionary_sha256": digest(Path(jieba.__file__).with_name("dict.txt")),
                              "input_policy": "Apply identical Mandarin normalization to reference and hypothesis, then tokenize both with the same dictionary"},
            "normalization": "NFKC, lowercase, remove Unicode punctuation and whitespace; no number rewriting",
            "data_audit": {"training_id_overlap": 0, "training_speaker_overlap": 0, "training_audio_hash_overlap": 0,
                           "unchanged_official_references": count}, "automatic_followup_allowed": False}
    freeze(output / "plan.json", plan)
    manifest = output / (f"validation-{count}.jsonl" if args.manifest else "pilot-24.jsonl")
    if manifest.exists():
        assert read_jsonl(manifest) == selected
    else:
        write_jsonl(manifest, selected)
    tokenizer = jieba.Tokenizer()
    tokenizer.tmp_dir = str(output)
    tokenizer.cache_file = "jieba-0.42.1.cache"
    tokenizer.initialize()
    scored = []
    timings = []
    began = time.monotonic()

    def publish(complete=False):
        now = datetime.now(TAIPEI)
        remaining = (count - len(scored)) * sum(timings) / len(timings) if timings else None
        progress = {"status": "complete" if complete else "running", "phase": "clean-pilot-complete" if complete else "clean-pilot-decoding",
                    "updated_at": now.isoformat(timespec="seconds"), "epoch": args.epoch,
                    "clip_progress": {"completed": len(scored), "total": count, "percent": round(100 * len(scored) / count, 1)},
                    "scope": f"Epoch {args.epoch} clean recognition on {count} held-out AISHELL validation recordings", "automatic_followup_allowed": False}
        if remaining is not None:
            progress.update(estimated_remaining_seconds=round(remaining),
                            estimated_remaining_range_seconds=[round(remaining * .7), round(remaining * 1.8)],
                            estimated_completion_taipei=(now + timedelta(seconds=remaining)).isoformat(timespec="seconds"),
                            eta_basis=f"Observed decoding time on this fixed {count}-audio validation manifest")
        atomic_json(output / "progress.json", progress)
        eta = f"Estimated completion: **{progress['estimated_completion_taipei']} (Asia/Taipei)**." if remaining is not None and not complete else "Loading the model; ETA will follow the first audio."
        if complete:
            eta = "Pilot complete; this GPU job has ended."
        text = (f"# Epoch {args.epoch} clean AISHELL pilot\n\nUpdated: {progress['updated_at']} (Asia/Taipei).\n\n"
                f"Status: **{progress['status']}**; progress **{len(scored)}/{count}**.\n\n{eta}\n\n"
                f"Epoch {args.epoch} was already fully trained and saved. The later epoch 3 continuation also completed; "
                f"this pilot uses the explicitly requested epoch {args.epoch} weights.\n\n"
                f"- [Clean CER and WER report]({output}/report.md)\n- [Job receipt]({output}/job.json)\n"
                f"- [Frozen selection and decoder]({output}/plan.json)\n")
        status_paths = [output / "JOB_STATUS.md"]
        if not args.local_status_only:
            status_paths += [base / "JOB_STATUS.md", ROOT / "reproduction-runs/mandarin-cer-corrected/WORKFLOW_STATUS.md"]
        for path in status_paths:
            path.write_text(text)
        lines = [f"# Clean AISHELL validation — epoch {args.epoch}", "", f"Status: **{'complete' if complete else 'running'}**, **{len(scored)}/{count}** audios. Updated: {progress['updated_at']} (Asia/Taipei).", ""]
        if scored:
            cer = aggregate_errors([r["cer"] for r in scored])
            wer = aggregate_errors([r["wer"] for r in scored])
            lines += ["| Metric | Error rate | Substitutions | Deletions | Insertions | Reference units |",
                      "|---|---:|---:|---:|---:|---:|",
                      *[f"| {name} | {values['error_percent']:.4f}% | {values['substitutions']} | {values['deletions']} | {values['insertions']} | {values['reference_units']} |"
                        for name, values in [("Clean CER", cer), ("Clean WER (jieba)", wer)]], ""]
        lines += ["CER is the primary Mandarin metric used for comparison in the existing AISHELL workflow. WER segments both references and hypotheses with jieba 0.42.1, HMM disabled; word segmentation affects this number.", "",
                  f"The corpus rate is 100 × sum(substitutions + deletions + insertions) / sum(reference units), with each audio decoded once. This validation run uses {count} recordings from {len(set(r['speaker'] for r in selected))} speakers held out from model fine-tuning. Selection: {selection}.", "",
                  f"Checkpoint: `{checkpoint['path']}` (epoch {args.epoch}). SHA-256: `{checkpoint['sha256']}`.", "",
                  "Decoding: forced Mandarin, beam size 5, temperature 0, timestamps disabled, fp32, previous-text conditioning disabled. Ground truth is the unchanged official AISHELL transcript. Normalization: NFKC, lowercase, remove punctuation and whitespace; no number rewriting.", "",
                  f"- [Frozen plan]({output}/plan.json)", f"- [Pilot manifest]({manifest})", f"- [Per-audio machine-readable results]({output}/utterances.jsonl)", ""]
        if complete:
            lines += [f"- [Independent distance and reference audit]({output}/independent-audit.json)", "",
                      f"Elapsed evaluation time: {time.monotonic() - began:.1f} seconds.", "",
                      "| Audio | CER | WER (jieba) | Official reference | Hypothesis |", "|---|---:|---:|---|---|"]
            for row in scored:
                safe = lambda s: s.replace("|", "\\|").replace("\n", " ")
                percentage = lambda metric: 100 * sum(row[metric][k] for k in ["substitutions", "deletions", "insertions"]) / row[metric]["reference_units"]
                lines.append(f"| {row['id']} | {percentage('cer'):.2f}% | {percentage('wer'):.2f}% | {safe(row['reference'])} | {safe(row['hypothesis'])} |")
        (output / "report.md").write_text("\n".join(lines) + "\n")

    publish()
    target, target_meta = load_whisper(target_config, "cuda")
    assert target_meta["sha256"] == checkpoint["sha256"]
    for row in selected:
        assert digest(row["audio"]) == row["audio_sha256"]
        cache = output / "records" / f"{row['id']}.json"
        started = time.monotonic()
        if cache.exists():
            item = json.loads(cache.read_text())
            assert item["target"] == target_meta and item["reference"] == row["text"] and item["audio_sha256"] == row["audio_sha256"]
        else:
            hypothesis = target.transcribe(load_audio(row["audio"]).to("cuda"))
            cer = error_counts(row["text"], hypothesis, "zh")
            reference_tokens = list(tokenizer.cut(cer["reference_normalized"], HMM=False))
            hypothesis_tokens = list(tokenizer.cut(cer["hypothesis_normalized"], HMM=False))
            words = jiwer.process_words(" ".join(reference_tokens), " ".join(hypothesis_tokens))
            wer = {"substitutions": words.substitutions, "deletions": words.deletions, "insertions": words.insertions,
                   "reference_units": len(reference_tokens)}
            item = {"id": row["id"], "speaker": row["speaker"], "reference": row["text"], "hypothesis": hypothesis,
                    "audio_sha256": row["audio_sha256"], "target": target_meta, "cer": cer, "wer": wer,
                    "reference_word_tokens": reference_tokens, "hypothesis_word_tokens": hypothesis_tokens,
                    "seconds": time.monotonic() - started}
            atomic_json(cache, item)
        ref, hyp = independent_normalize(item["reference"]), independent_normalize(item["hypothesis"])
        assert ref == independent_normalize(truth[row["id"]]) == item["cer"]["reference_normalized"]
        assert hyp == item["cer"]["hypothesis_normalized"]
        for metric, reference, hypothesis in [("cer", ref, hyp), ("wer", item["reference_word_tokens"], item["hypothesis_word_tokens"])]:
            assert len(reference) == item[metric]["reference_units"]
            assert distance(reference, hypothesis) == sum(item[metric][k] for k in ["substitutions", "deletions", "insertions"])
        scored.append(item)
        timings.append(item["seconds"])
        write_jsonl(output / "utterances.jsonl", scored)
        publish()
        print(f"Clean validation {len(scored)}/{count}: {row['id']}", flush=True)
    assert len(scored) == len({r["id"] for r in scored}) == count
    result = {"status": "complete", "epoch": args.epoch, "count": count, "target": target_meta,
              "clean_cer": aggregate_errors([r["cer"] for r in scored]), "clean_wer_jieba": aggregate_errors([r["wer"] for r in scored]),
              "plan_sha256": digest(output / "plan.json"), "utterances_sha256": digest(output / "utterances.jsonl")}
    atomic_json(output / "results.json", result)
    atomic_json(output / "independent-audit.json", {"status": "passed", "count": count, "unchanged_official_references": count,
                "independent_character_distance_checks": count, "independent_word_distance_checks": count,
                "training_id_speaker_audio_overlap": 0, "results_sha256": digest(output / "results.json")})
    publish(complete=True)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-metadata", type=Path, help="Completed training metadata for separately resumed epochs")
    parser.add_argument("--manifest", type=Path, help="Frozen subset of the held-out checkpoint-validation manifest")
    parser.add_argument("--selection-note", help="Description of the prespecified manifest selection")
    parser.add_argument("--local-status-only", action="store_true", help="Let a suite controller own the shared status files")
    parser.add_argument("--epoch", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    args = parser.parse_args()
    with (ROOT / "reproduction-runs/auxiliary-gpu.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args)
