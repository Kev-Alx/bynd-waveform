import math
from types import SimpleNamespace

import pytest
import torch
from whisper.decoding import DecodingOptions, DecodingTask

from beyondwaveform.attacks import freeze
from beyondwaveform.models.whisper import WhisperASR
from scripts.probe_decoder_alignment import (
    aggregate, make_tasks, mechanism_gate, probe_audio, static_token_ids, token_diagnostic,
)


def native_task(without_timestamps=True):
    # Native filter construction requires dimensions and decoder block names;
    # no fake inference is used in the analytical filter tests.
    model = SimpleNamespace(is_multilingual=True, num_languages=99,
        dims=SimpleNamespace(n_text_ctx=448, n_audio_ctx=1500),
        decoder=SimpleNamespace(blocks=[]))
    return DecodingTask(model, DecodingOptions(language="en", task="transcribe", fp16=False,
                                              without_timestamps=without_timestamps))


def sparse_logits(task, weights):
    logits = torch.full((task.tokenizer.encoding.n_vocab,), -100.0, dtype=torch.float64)
    for token, weight in weights.items():
        logits[token] = math.log(weight)
    return logits


def lexical_token(task, text="hello"):
    return task.tokenizer.encode(text)[0]


def test_static_suppression_has_exact_nll_decomposition_and_mass():
    task = native_task()
    target, alternative = lexical_token(task), lexical_token(task, "world")
    banned = next(token for token in static_token_ids(task) if token not in {target, alternative})
    logits = sparse_logits(task, {target: .25, alternative: .25, banned: .5})
    context = torch.tensor([[*task.initial_tokens, target]])
    row = token_diagnostic(logits, context, task, target)
    assert row["static_target_allowed"]
    assert row["raw_nll"] == pytest.approx(math.log(4))
    assert row["static_conditional_nll"] == pytest.approx(math.log(2))
    assert row["static_gap"] == pytest.approx(math.log(2))
    assert row["static_suppressed_mass"] == pytest.approx(.5)
    assert row["raw_nll"] == pytest.approx(row["static_conditional_nll"]+row["static_gap"])
    attribution = row["native"]["ordered_filter_attribution"]
    assert sum(r["removed_original_probability_mass"] for r in attribution) == pytest.approx(.5)
    assert row["raw_argmax_statically_suppressed"]
    assert torch.equal(logits, sparse_logits(task, {target: .25, alternative: .25, banned: .5}))


def test_statically_forbidden_reference_has_null_conditional_nll():
    task = native_task()
    target = static_token_ids(task)[0]
    allowed = lexical_token(task)
    row = token_diagnostic(sparse_logits(task, {target: .5, allowed: .5}),
                           torch.tensor([[*task.initial_tokens, allowed]]), task, target)
    assert not row["static_target_allowed"]
    assert row["static_conditional_nll"] is None
    assert not row["native"]["target_allowed"]
    assert row["native"]["conditional_nll"] is None
    assert math.isfinite(row["raw_nll"])


def test_blank_filter_suppresses_space_and_eot_only_at_first_decode_step():
    task = native_task()
    word = lexical_token(task)
    blank = task.tokenizer.encode(" ")[0]
    eot = task.tokenizer.eot
    logits = sparse_logits(task, {eot: .4, blank: .3, word: .3})
    first = token_diagnostic(logits, torch.tensor([task.initial_tokens]), task, eot)
    later = token_diagnostic(logits, torch.tensor([[*task.initial_tokens, word]]), task, eot)
    assert first["static_target_allowed"] and later["static_target_allowed"]
    assert not first["native"]["target_allowed"]
    assert later["native"]["target_allowed"]
    first_blank = first["native"]["ordered_filter_attribution"][0]
    assert first_blank["filter"] == "SuppressBlank"
    assert first_blank["removed_original_probability_mass"] == pytest.approx(.7)
    assert later["native"]["ordered_filter_attribution"][0]["removed_original_probability_mass"] == 0
    assert first["raw_categories"]["eot"] == pytest.approx(.4)
    assert first["raw_categories"]["blank"] == pytest.approx(.3)


def test_timestamp_initial_step_forces_time_and_limits_initial_time_range():
    task = native_task(False)
    timestamp = task.tokenizer.timestamp_begin
    word = lexical_token(task)
    logits = sparse_logits(task, {word: 5, timestamp: 1, timestamp+60: 3})
    context = torch.tensor([task.initial_tokens])
    first = token_diagnostic(logits, context, task, timestamp)
    forbidden_text = token_diagnostic(logits, context, task, word)
    forbidden_late_time = token_diagnostic(logits, context, task, timestamp+60)
    assert first["native"]["target_allowed"]
    assert not forbidden_text["native"]["target_allowed"]
    assert not forbidden_late_time["native"]["target_allowed"]
    assert first["native"]["argmax_id"] == timestamp
    assert first["native"]["ordered_filter_attribution"][-1]["filter"] == "ApplyTimestampRules"


def test_timestamp_rules_use_combined_time_mass_and_actual_prefix_state():
    task = native_task(False)
    timestamp, word = task.tokenizer.timestamp_begin, lexical_token(task)
    # Each time token is weaker than the text token; their combined mass is
    # stronger. The native rule therefore forces a timestamp after text.
    logits = sparse_logits(task, {word: 1.5, timestamp+1: 1, timestamp+2: 1})
    after_text = token_diagnostic(logits, torch.tensor([[*task.initial_tokens, timestamp, word]]), task, word)
    after_initial_time = token_diagnostic(logits, torch.tensor([[*task.initial_tokens, timestamp]]), task, word)
    assert after_text["raw_argmax_id"] == word
    assert not after_text["native"]["target_allowed"]
    assert after_text["native"]["conditional_nll"] is None
    assert after_initial_time["native"]["target_allowed"]
    assert after_initial_time["native"]["argmax_id"] == word


def test_unreachable_native_prefix_is_not_reclassified_when_local_label_survives():
    task = native_task()
    word = lexical_token(task)
    row = token_diagnostic(sparse_logits(task, {word: 1}),
        torch.tensor([[*task.initial_tokens, word]]), task, word, prefix_reachable=False)
    assert row["native"]["target_allowed"]
    assert not row["native"]["prefix_reachable"]
    assert not row["native"]["target_reachable"]


def test_probe_reuses_audio_encoding_masks_only_lexical_positions_and_records_paths():
    class TinyNative(torch.nn.Module):
        is_multilingual = True
        num_languages = 99
        dims = SimpleNamespace(n_text_ctx=448, n_audio_ctx=1500)
        decoder = SimpleNamespace(blocks=[])

        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(1.0))
            self.encodes = 0

        def embed_audio(self, mel):
            self.encodes += 1
            return mel

        def logits(self, tokens, features):
            return torch.zeros(1, tokens.shape[1], 51865) + self.scale

    adapter = freeze(WhisperASR(TinyNative(), reduction="sum"))
    adapter.mel = lambda audio: torch.zeros(1)
    native, tasks = make_tasks(adapter)
    result = probe_audio(adapter, torch.zeros(1000), "HELLO WORLD FRIEND", tasks)
    assert adapter.model.encodes == 1
    assert not native.options.without_timestamps
    assert tasks["attack"].options.without_timestamps
    assert not tasks["native_timestamp0"].options.without_timestamps
    assert result["model_parameters_frozen"] and not result["parameter_gradients_present"]
    attack, timestamped = (result["views"][view] for view in ("attack", "native_timestamp0"))
    assert attack["lexical_ids"] == timestamped["lexical_ids"]
    assert len(attack["tokens"]) == len(adapter.tokenizer.encode("HELLO WORLD FRIEND"))
    assert attack["initial_timestamp_step"] is None
    assert timestamped["initial_timestamp_step"]["target_id"] == adapter.tokenizer.timestamp_begin
    assert attack["parent_reduced_raw_nll_float32"] == pytest.approx(len(attack["tokens"])*math.log(51865))
    # Uniform logits make combined timestamp mass force a timestamp after the
    # first lexical token. Later teacher-forced histories are hypothetical.
    assert timestamped["tokens"][0]["native"]["target_reachable"]
    assert not timestamped["tokens"][1]["native"]["target_allowed"]
    assert not timestamped["tokens"][2]["native"]["prefix_reachable"]


def synthetic_items(*, valid=True, adversarial_mass=.2, raw_increase=2, gap_increase=.4):
    def token(index, raw, gap, allowed=True, mass=0):
        return {"index": index, "target_id": index+10, "raw_nll": raw,
            "static_target_allowed": allowed, "static_gap": gap,
            "static_conditional_nll": raw-gap if allowed else None,
            "static_suppressed_mass": mass, "raw_categories": {key: 0 for key in
                ("eot", "blank", "timestamps", "nonlexical_specials_below_timestamps")},
            "native": {"target_allowed": allowed, "target_reachable": allowed,
                       "masked_original_probability_mass": mass}}
    items = []
    for uid, baseline, length in (("long", 2, 2), ("short", 4, 1)):
        states = {}
        for variant in ("clean", "reconstruction", "adversarial"):
            raw = baseline + (raw_increase if variant == "adversarial" else 0)
            gap = .1 + (gap_increase if variant == "adversarial" else 0)
            rows = [token(i, raw, gap, mass=adversarial_mass if variant == "adversarial" else .01)
                    for i in range(length)]
            if not valid and uid == "long":
                rows.append(token(length, raw, gap, False))
            states[variant] = {"views": {view: {"tokens": rows} for view in ("attack", "native_timestamp0")}}
        items.append({"id": uid, "states": states})
    return items


def test_aggregate_pairs_reference_positions_and_distinguishes_token_and_clip_weighting():
    result = aggregate(synthetic_items())
    states = result["views"]["attack"]["states"]
    assert states["adversarial"]["token_micro"]["raw_nll_mean"] == pytest.approx(14/3)
    assert states["adversarial"]["utterance_macro"]["raw_nll_mean"] == 5
    paired = result["views"]["attack"]["paired_adversarial_vs_reconstruction"]
    assert paired["matched_static_valid_count"] == 3
    assert paired["mean_raw_nll_increase"] == 2
    assert paired["mean_static_gap_increase"] == pytest.approx(.4)
    assert paired["mean_static_conditional_nll_increase"] == pytest.approx(1.6)
    assert paired["static_gap_share_of_raw_nll_increase"] == pytest.approx(.2)
    assert mechanism_gate(result)["eligible"]


@pytest.mark.parametrize("changes,failed", [
    ({"valid": False}, "static_valid_coverage"),
    ({"adversarial_mass": .09}, "adversarial_median_mass"),
    ({"gap_increase": .1}, "static_gap_share"),
    ({"raw_increase": 0}, "positive_raw_nll_increase"),
])
def test_mechanism_gate_requires_every_predeclared_condition(changes, failed):
    result = aggregate(synthetic_items(**changes))
    gate = mechanism_gate(result)
    assert not gate["eligible"]
    assert not gate["checks"][failed]
    if not changes.get("valid", True):
        evidence = gate["evidence"]
        assert evidence["matched_static_valid_count"] == 3
        assert evidence["reference_token_count"] == 4
        assert evidence["static_valid_coverage"] == .75


def test_aggregate_rejects_misaligned_reference_positions():
    items = synthetic_items()
    items[0]["states"]["adversarial"]["views"]["attack"]["tokens"][0]["target_id"] = 99
    with pytest.raises(ValueError, match="Unpaired lexical targets"):
        aggregate(items)
