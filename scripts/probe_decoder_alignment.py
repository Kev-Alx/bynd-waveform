"""Read-only teacher-forcing/native-decoder probability alignment diagnostic.

No decoding, optimization, vocoding or MOS inference occurs. All probability
measurements use saved parent WAVs and native Whisper logit filters. A static
conditional softmax is a diagnostic distribution, not a modified attack loss.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import importlib
import math
from pathlib import Path
import statistics
import sys

import torch
from whisper.decoding import DecodingOptions, DecodingTask, SuppressTokens

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from beyondwaveform.attacks import freeze
from beyondwaveform.io import atomic_json, digest, load_audio, read_jsonl, seed_everything
from beyondwaveform.losses import lexical_nll
from beyondwaveform.models.whisper import load_whisper, nll_transcript, teacher_forcing_prefix
from beyondwaveform.pipeline import config_fingerprint

VARIANTS = ("clean", "reconstruction", "adversarial")
VIEWS = ("attack", "native_timestamp0")
GATE_THRESHOLDS = {"minimum_static_gap_share": 0.10,
                   "minimum_adversarial_token_median_suppressed_mass": 0.10,
                   "minimum_static_valid_coverage": 0.95,
                   "minimum_positive_nll_increase": 1e-8}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def source_paths():
    paths = [Path(__file__).resolve(), ROOT / "beyondwaveform/models/whisper.py",
             ROOT / "beyondwaveform/losses.py", ROOT / "beyondwaveform/io.py"]
    paths += [Path(importlib.import_module("whisper." + name).__file__).resolve()
              for name in ("decoding", "tokenizer", "model", "audio", "transcribe")]
    assets = Path(importlib.import_module("whisper").__file__).resolve().parent / "assets"
    paths += sorted(assets.glob("*.tiktoken")) + [assets / "mel_filters.npz"]
    return paths


def make_tasks(adapter):
    """Instantiate native filters, without running their model inference path."""
    options = DecodingOptions(language=adapter.language, task="transcribe", fp16=False, **adapter.decoding)
    require(options.prompt is None and options.prefix is None, "Probe requires the unprompted parent decoder")
    native = DecodingTask(adapter.model, options)
    attack_without = adapter.nll_timestamp_policy == "no_timestamps"
    attack = DecodingTask(adapter.model, replace(options, without_timestamps=attack_without))
    timestamped = DecodingTask(adapter.model, replace(options, without_timestamps=False))
    return native, {"attack": attack, "native_timestamp0": timestamped}


def static_token_ids(task):
    """The exact enabled native SuppressTokens list; no invented exclusions."""
    return sorted({token for filt in task.logit_filters if isinstance(filt, SuppressTokens)
                   for token in filt.suppress_tokens})


def token_diagnostic(logits, context, task, target_id=None, prefix_reachable=True):
    """Attribute removed original-softmax mass to ordered native filters.

    Static gap = raw logsumexp - static-allowed logsumexp = -log(P_allowed).
    If a target is allowed, raw NLL = conditional NLL + gap. A forbidden
    target has null conditional NLL. Native-filter removal is sequential and
    disjoint, unlike the explicitly overlapping token-category summaries.
    """
    require(logits.ndim == 1 and torch.isfinite(logits).all().item(), "Expected finite vocabulary logits")
    require(context.ndim == 2 and context.shape[0] == 1, "Expected one native prefix")
    raw = logits.detach().double()
    raw_lse = raw.logsumexp(0)
    probabilities = (raw - raw_lse).exp()
    static_mask = torch.zeros_like(raw, dtype=torch.bool)
    ids = static_token_ids(task)
    if ids:
        static_mask[ids] = True
    require((~static_mask).any().item(), "Static suppression removes the entire vocabulary")
    static_lse = raw.masked_fill(static_mask, -torch.inf).logsumexp(0)
    static_allowed = target_id is None or not static_mask[target_id].item()
    filtered = logits.detach().float().clone()[None]
    removed = torch.zeros_like(static_mask)
    attribution = []
    for filt in task.logit_filters:
        filt.apply(filtered, context)
        require(not torch.isnan(filtered).any().item() and torch.isfinite(filtered).any().item(),
                "Native filters left no finite candidate")
        now_removed = torch.isneginf(filtered[0])
        newly_removed = now_removed & ~removed
        attribution.append({"filter": type(filt).__name__, "newly_masked_count": newly_removed.sum().item(),
                            "removed_original_probability_mass": probabilities[newly_removed].sum().item()})
        removed = now_removed
    native_allowed = target_id is None or not removed[target_id].item()
    native_lse = raw.masked_fill(removed, -torch.inf).logsumexp(0)
    tokenizer = task.tokenizer
    blank_ids = tokenizer.encode(" ")
    raw_argmax = raw.argmax().item()
    row = {"target_id": target_id,
        "raw_nll": (raw_lse - raw[target_id]).item() if target_id is not None else None,
        "raw_target_probability": probabilities[target_id].item() if target_id is not None else None,
        "raw_argmax_id": raw_argmax,
        "raw_argmax_statically_suppressed": static_mask[raw_argmax].item(),
        "raw_categories": {"eot": probabilities[tokenizer.eot].item(),
            "blank": probabilities[blank_ids].sum().item(),
            "timestamps": probabilities[tokenizer.timestamp_begin:].sum().item(),
            "nonlexical_specials_below_timestamps": probabilities[tokenizer.eot:tokenizer.timestamp_begin].sum().item()},
        "static_target_allowed": static_allowed,
        "static_suppressed_mass": probabilities[static_mask].sum().item(),
        "static_gap": (raw_lse - static_lse).item(),
        "static_conditional_nll": (static_lse - raw[target_id]).item() if target_id is not None and static_allowed else None,
        "static_argmax_id": raw.masked_fill(static_mask, -torch.inf).argmax().item(),
        "native": {"target_allowed": native_allowed,
            "prefix_reachable": bool(prefix_reachable),
            "target_reachable": bool(prefix_reachable and native_allowed),
            "masked_original_probability_mass": probabilities[removed].sum().item(),
            "gap": (raw_lse - native_lse).item(),
            "conditional_nll": (native_lse - raw[target_id]).item() if target_id is not None and native_allowed else None,
            "argmax_id": filtered[0].argmax().item(), "ordered_filter_attribution": attribution}}
    require(abs(sum(r["removed_original_probability_mass"] for r in attribution) -
                row["native"]["masked_original_probability_mass"]) < 1e-10,
            "Native probability attribution is not disjoint")
    return row


@torch.no_grad()
def probe_audio(adapter, audio, text, tasks):
    target = nll_transcript(text, adapter.nll_transcript_policy)
    lexical = adapter.tokenizer.encode(target)
    require(lexical and all(token < adapter.tokenizer.eot for token in lexical), "Invalid lexical transcript")
    audio_features = adapter.model.embed_audio(adapter.mel(audio))
    views = {}
    for name in VIEWS:
        task = tasks[name]
        if name == "attack":
            prefix = teacher_forcing_prefix(adapter.tokenizer, adapter.nll_timestamp_policy)
        else:
            prefix = [*task.initial_tokens, adapter.tokenizer.timestamp_begin]
        expected_prefix = list(task.initial_tokens)
        if not task.options.without_timestamps:
            expected_prefix.append(adapter.tokenizer.timestamp_begin)
        require(prefix == expected_prefix, "Teacher prefix does not match its declared native task")
        inputs = torch.tensor([prefix + lexical], device=audio.device)
        require(inputs.shape[1] <= adapter.model.dims.n_text_ctx, "Transcript exceeds context")
        logits = adapter.model.logits(inputs, audio_features)[0]
        start = len(prefix) - 1
        initial = None
        reachable = True
        if not task.options.without_timestamps:
            index = len(task.initial_tokens) - 1
            initial = token_diagnostic(logits[index], inputs[:, :index+1], task,
                                       adapter.tokenizer.timestamp_begin, True)
            reachable = initial["native"]["target_reachable"]
        rows = []
        for index, token in enumerate(lexical):
            position = start + index
            row = token_diagnostic(logits[position], inputs[:, :position+1], task, token, reachable)
            row.update(index=index, target_text=adapter.tokenizer.decode([token]))
            rows.append(row)
            reachable = row["native"]["target_reachable"]
        selected = logits[start:start+len(lexical)][None]
        labels = torch.tensor([lexical], device=audio.device)
        parent_nll = lexical_nll(selected, labels, torch.ones_like(labels, dtype=torch.bool), adapter.reduction).item()
        reported_nll = sum(r["raw_nll"] for r in rows)
        if adapter.reduction == "mean":
            reported_nll /= len(rows)
        require(abs(parent_nll-reported_nll) <= 1e-5 + 1e-5*abs(parent_nll), "Raw NLL does not match parent reduction")
        views[name] = {"prefix": prefix, "lexical_ids": lexical, "tokens": rows,
            "parent_reduction": adapter.reduction, "parent_reduced_raw_nll_float32": parent_nll,
            "initial_timestamp_step": initial, "all_reference_targets_native_reachable": reachable,
            "effective_options": asdict(task.options), "sample_begin": task.sample_begin,
            "static_suppressed_token_ids": static_token_ids(task),
            "native_filter_order": [type(f).__name__ for f in task.logit_filters]}
    require(not any(p.grad is not None for p in adapter.parameters()), "Probe accumulated model gradients")
    return {"views": views, "model_parameters_frozen": not any(p.requires_grad for p in adapter.parameters()),
            "parameter_gradients_present": False}


def mean(values):
    return statistics.fmean(values) if values else None


def summarize_tokens(tokens):
    valid = [r for r in tokens if r["static_target_allowed"]]
    return {"token_count": len(tokens), "static_valid_count": len(valid),
        "static_valid_coverage": len(valid)/len(tokens) if tokens else None,
        "raw_nll_mean": mean([r["raw_nll"] for r in tokens]),
        "raw_nll_sum": sum(r["raw_nll"] for r in tokens),
        "raw_nll_mean_static_valid": mean([r["raw_nll"] for r in valid]),
        "static_conditional_nll_mean_valid": mean([r["static_conditional_nll"] for r in valid]),
        "static_gap_mean_valid": mean([r["static_gap"] for r in valid]),
        "static_suppressed_mass_mean_valid": mean([r["static_suppressed_mass"] for r in valid]),
        "static_suppressed_mass_median_valid": statistics.median([r["static_suppressed_mass"] for r in valid]) if valid else None,
        "native_locally_allowed_count": sum(r["native"]["target_allowed"] for r in tokens),
        "native_reachable_target_count": sum(r["native"]["target_reachable"] for r in tokens),
        "native_masked_mass_mean": mean([r["native"]["masked_original_probability_mass"] for r in tokens]),
        "raw_category_mass_means": {category: mean([r["raw_categories"][category] for r in tokens])
            for category in ("eot", "blank", "timestamps", "nonlexical_specials_below_timestamps")}}


def aggregate(items):
    """Token-micro and utterance-macro aggregates; no word-weighted scores."""
    require(items, "Cannot aggregate an empty diagnostic")
    result = {"utterance_count": len(items), "views": {},
        "aggregation": "Token-micro concatenates lexical positions; utterance-macro equally weights each clip's token mean. No word weighting, new WER or MOS."}
    for view in VIEWS:
        states = {}
        for variant in VARIANTS:
            per_clip = [summarize_tokens(item["states"][variant]["views"][view]["tokens"]) for item in items]
            flat = [token for item in items for token in item["states"][variant]["views"][view]["tokens"]]
            states[variant] = {"token_micro": summarize_tokens(flat),
                "utterance_macro": {key: mean([row[key] for row in per_clip if row[key] is not None])
                    for key in ("raw_nll_mean", "raw_nll_mean_static_valid", "static_conditional_nll_mean_valid",
                                "static_gap_mean_valid", "static_suppressed_mass_mean_valid", "native_masked_mass_mean")}}
        matched = []
        total = 0
        for item in items:
            rec = item["states"]["reconstruction"]["views"][view]["tokens"]
            adv = item["states"]["adversarial"]["views"][view]["tokens"]
            require(len(rec) == len(adv), "Unpaired lexical coverage")
            total += len(rec)
            for left, right in zip(rec, adv):
                require(left["index"] == right["index"] and left["target_id"] == right["target_id"], "Unpaired lexical targets")
                if left["static_target_allowed"] and right["static_target_allowed"]:
                    matched.append((left, right))
        raw_increase = mean([right["raw_nll"]-left["raw_nll"] for left, right in matched])
        gap_increase = mean([right["static_gap"]-left["static_gap"] for left, right in matched])
        conditional_increase = mean([right["static_conditional_nll"]-left["static_conditional_nll"] for left, right in matched])
        if matched:
            require(abs(raw_increase-gap_increase-conditional_increase) < 1e-9,
                    "Paired NLL decomposition does not recover the raw increase")
        paired = {"matched_static_valid_count": len(matched), "reference_token_count": total,
            "static_valid_coverage": len(matched)/total if total else None,
            "mean_raw_nll_increase": raw_increase, "mean_static_gap_increase": gap_increase,
            "mean_static_conditional_nll_increase": conditional_increase,
            "static_gap_share_of_raw_nll_increase": gap_increase/raw_increase
                if raw_increase is not None and raw_increase > GATE_THRESHOLDS["minimum_positive_nll_increase"] else None,
            "adversarial_token_median_static_suppressed_mass": statistics.median([right["static_suppressed_mass"] for _, right in matched]) if matched else None}
        result["views"][view] = {"states": states, "paired_adversarial_vs_reconstruction": paired}
    return result


def mechanism_gate(summary):
    evidence = summary["views"]["attack"]["paired_adversarial_vs_reconstruction"]
    checks = {"positive_raw_nll_increase": evidence["mean_raw_nll_increase"] is not None and
            evidence["mean_raw_nll_increase"] > GATE_THRESHOLDS["minimum_positive_nll_increase"],
        "static_gap_share": evidence["static_gap_share_of_raw_nll_increase"] is not None and
            evidence["static_gap_share_of_raw_nll_increase"] >= GATE_THRESHOLDS["minimum_static_gap_share"],
        "adversarial_median_mass": evidence["adversarial_token_median_static_suppressed_mass"] is not None and
            evidence["adversarial_token_median_static_suppressed_mass"] >= GATE_THRESHOLDS["minimum_adversarial_token_median_suppressed_mass"],
        "static_valid_coverage": evidence["static_valid_coverage"] is not None and
            evidence["static_valid_coverage"] >= GATE_THRESHOLDS["minimum_static_valid_coverage"]}
    return {"eligible": all(checks.values()), "checks": checks, "thresholds": GATE_THRESHOLDS,
            "evidence": evidence, "interpretation": "A descriptive screen for a separately labeled filtered-softmax experiment, not causal proof or author-code identification. No experiment is launched by this script."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True, help="Frozen prior feature-delta diagnostic plan")
    parser.add_argument("--output", type=Path, required=True, help="New JSON report, never overwritten")
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--progress", type=Path, help="Optional separate progress JSON; do not reuse supervisor progress")
    args = parser.parse_args(argv)
    from run_feature_diagnostics import check
    import run_quality_research as protocol
    plan_path, output = args.plan.resolve(), args.output.resolve()
    require(not output.exists(), "Refusing to overwrite existing decoder diagnostic")
    require(args.count > 0, "count must be positive")
    plan = protocol.read(plan_path)
    check(plan)
    plan_sha = digest(plan_path)
    source_hashes = {str(path): digest(path) for path in source_paths()}
    pool, parent = plan["pools"]["pilot"], plan["parents"]["pilot"]
    source = read_jsonl(pool["manifest"])
    require(args.count <= len(source), "count exceeds pilot pool")
    parent_run = Path(parent["run"])
    metadata = protocol.read(parent_run / "run.json")
    evaluation = protocol.read(parent_run / "evaluation/summary.json")
    records = read_jsonl(parent_run / "generated.jsonl")
    identity = {"config": parent["config"], "manifest_sha256": pool["sha256"], "ids": pool["ids"]}
    require(metadata["status"] == "complete" and metadata["config"] == parent["config"] and
            metadata["features"] == plan["feature_artifacts"] and
            metadata["fingerprint"] == config_fingerprint(identity) == evaluation["generation_fingerprint"] and
            digest(parent_run / "generated.jsonl") == metadata["generated_sha256"] and
            digest(parent_run / "evaluation/summary.json") == parent["summary_sha256"] and
            digest(parent_run / "evaluation/utterances.jsonl") == evaluation["utterances_sha256"] and
            [row["id"] for row in records] == pool["ids"], "Changed parent generation/evaluation identity")
    cache_path = plan_path.parent / "parent-audits/pilot.json"
    cache = protocol.read(cache_path)
    cache_sha = digest(cache_path)
    require(cache["role"] == "pilot" and cache["features"] == plan["feature_artifacts"] and
            cache["parent_summary_sha256"] == parent["summary_sha256"] and
            [row["id"] for row in cache["items"]] == pool["ids"], "Changed clean-feature cache identity")
    cache_items = {row["id"]: row for row in cache["items"]}
    seed_everything(parent["config"]["seed"])
    adapter, surrogate = load_whisper(parent["config"]["surrogate"], args.device)
    freeze(adapter)
    require(surrogate == metadata["surrogate"], "Changed parent Whisper contract")
    native, tasks = make_tasks(adapter)
    checked_assets = {}
    items = []
    for original, record in zip(source[:args.count], records[:args.count]):
        uid, cached = original["id"], cache_items[original["id"]]
        require(record["text"] == original["text"] and record["input_audio_sha256"] == original["audio_sha256"], "Changed parent reference/source")
        checked_assets[original["audio"]] = original["audio_sha256"]
        checked_assets[cached["q_path"]] = cached["q_sha256"]
        delta_path = str(parent_run / (uid + ".delta.pt"))
        checked_assets[delta_path] = parent["delta_hashes"][uid]
        require(cached["parent_delta_sha256"] == parent["delta_hashes"][uid] and
                cached["parent_radius"] <= 0.100001 and
                cached["resynthesis_max_abs_error"] <= plan["waveform_reproduction_tolerance"], "Changed cached delta/resynthesis evidence")
        for path in (original["audio"], cached["q_path"], delta_path):
            require(digest(path) == checked_assets[path], "Changed source/cache/delta artifact")
        states = {}
        for variant in VARIANTS:
            asset = record["audio"][variant]
            require(digest(asset["path"]) == asset["sha256"], "Changed saved parent WAV")
            checked_assets[asset["path"]] = asset["sha256"]
            seed_everything(record["seed"])
            states[variant] = probe_audio(adapter, load_audio(asset["path"]).to(args.device), original["text"], tasks)
            print(f"Decoder alignment: {uid} {variant}; both teacher prefixes measured.", flush=True)
        items.append({"id": uid, "selection": "first entries in frozen pilot order; no result-based selection",
            "reference": original["text"], "source_audio_sha256": original["audio_sha256"],
            "q_sha256": cached["q_sha256"], "parent_delta_sha256": parent["delta_hashes"][uid],
            "audio": record["audio"], "states": states})
        if args.progress:
            atomic_json(args.progress, {"status": "running", "completed_clips": len(items), "total_clips": args.count})
    check(plan)
    require(digest(plan_path) == plan_sha and digest(cache_path) == cache_sha, "Prior plan/cache receipt changed during probe")
    require(all(digest(path) == expected for path, expected in {**source_hashes, **checked_assets}.items()), "Source/asset changed during probe")
    summary = aggregate(items)
    report = {"status": "complete", "created_at": datetime.now(timezone.utc).isoformat(),
        "plan": str(plan_path), "plan_sha256": plan_sha, "parent_run": str(parent_run),
        "parent_cache_audit_sha256": cache_sha, "source_hashes": source_hashes,
        "checked_asset_hashes": checked_assets, "runtime": protocol.runtime(),
        "surrogate": surrogate, "effective_native_recognition_options": asdict(native.options),
        "transcribe_controls": {"condition_on_previous_text": False, "fp16": False,
            "no_speech_threshold": 0.6, "logprob_threshold": -1.0, "compression_ratio_threshold": 2.4,
            "note": "These are the parent wrapper's effective native transcribe defaults. This probe does not run transcribe or its segment-skip/retry logic."},
        "native_recognition_static_suppressed_token_ids": static_token_ids(native),
        "variant_order": VARIANTS, "teacher_views": VIEWS, "count": len(items), "items": items,
        "summary": summary, "mechanism_gate": mechanism_gate(summary),
        "scope": "Read-only saved-WAV teacher-forced probability diagnostic; zero new attacks, updates, decoding or MOS inference.",
        "semantics": {"static_gap": "raw_logsumexp - static_allowed_logsumexp = -log(original probability of static-allowed vocabulary)",
            "conditional_nll": "Defined only when the reference target survives the applicable mask; otherwise null and explicitly counted as forbidden.",
            "native_path": "Native filters are applied in native order. Once a forced reference target is forbidden, subsequent reference prefixes are hypothetical and marked unreachable.",
            "category_mass": "Raw probability categories may overlap; sequential native-filter attribution is disjoint.",
            "native_timestamp0": "A fixed unscored 0.00 timestamp is supplied after native initial tokens. It is not oracle alignment or a replay of autoregressive decoding.",
            "precision": "Native filters operate in float32; probability/NLL summaries use float64 logsumexp over those logits, checked against parent float32 lexical NLL.",
            "aggregation": summary["aggregation"]},
        "limitations": ["Ground-truth teacher forcing differs from decoded histories and cannot establish attack effectiveness.",
            "The attack-prefix view uses matching no-timestamps grammar when its conditioning contains no_timestamps; timestamp rules are not applied to that incompatible prefix.",
            "Native initial-timestamp prediction is reported separately and excluded from all lexical NLL aggregates.",
            "Cached features/deltas and prior resynthesis evidence are hash-verified; no re-extraction or resynthesis is performed.",
            "The mechanism gate motivates a separate experiment only; suppression masking is not claimed as a paper correction."]}
    atomic_json(output, report)
    if args.progress:
        atomic_json(args.progress, {"status": "complete", "completed_clips": len(items), "total_clips": args.count,
                                   "report_sha256": digest(output)})
    print(f"Decoder diagnostic complete: {len(items)} clips; mechanism gate={report['mechanism_gate']['eligible']}.", flush=True)
    return report


if __name__ == "__main__":
    main()
