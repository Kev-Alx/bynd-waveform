"""Exact candidate-construction contracts for the sole allowed R1 repair."""
from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]

import r1_sequence
from r1_deletion_repair import DeletionRepairObjective, ORIGIN
from r1_sequence import DegenerateRiskError


class WordTokenizer:
    eot = 100
    def __init__(self, texts):
        self.texts = dict(texts)
    def decode(self, tokens):
        return self.texts[tuple(tokens)]
    def encode(self, text):
        existing = next((tokens for tokens, value in self.texts.items() if value == text), None)
        if existing is not None:
            return list(existing)
        tokens = (len(self.texts) + 20,)
        self.texts[tokens] = text
        return list(tokens)


def build(monkeypatch, texts):
    tokenizer = WordTokenizer(texts)
    paths = [(tokens, -index) for index, tokens in enumerate(texts)]
    monkeypatch.setattr(r1_sequence, "native_nbest", lambda *_: (paths, {}))
    surrogate = SimpleNamespace(tokenizer=tokenizer, language="en", nll_transcript_policy="original")
    return DeletionRepairObjective(surrogate)


def test_all_correct_native_paths_receive_exactly_one_middle_word_deletion(monkeypatch):
    objective = build(monkeypatch, {(0,): "Alpha beta gamma delta.", (1,): "Alpha, beta gamma delta.",
                                    (2,): "alpha beta gamma delta.", (3,): "Alpha beta gamma delta"})
    candidates = objective.refresh(torch.zeros(1), "Alpha beta gamma delta.")
    synthetic = [candidate for candidate in candidates.candidates if candidate.origin == ORIGIN]
    assert len(synthetic) == 1
    assert synthetic[0].text == "Alpha beta delta."
    assert synthetic[0].cost == .25
    assert synthetic[0].deletions == 1
    assert len([candidate for candidate in candidates.candidates if candidate.origin == "native_beam"]) == 4
    rule = candidates.search_options["deletion_repair"]
    assert rule["deleted_word_index"] == 2
    assert rule["deleted_word"] == "gamma"
    assert rule["posterior_beam_hypothesis"] is False


def test_single_word_source_can_produce_empty_path(monkeypatch):
    objective = build(monkeypatch, {(0,): "Hello!", (1,): "hello!", (2,): "Hello", (): ""})
    candidates = objective.refresh(torch.zeros(1), "Hello!")
    synthetic = [candidate for candidate in candidates.candidates if candidate.origin == ORIGIN]
    # The empty path was already a native candidate, so no duplicate is added.
    assert synthetic == []
    assert candidates.search_options["deletion_repair"]["synthetic_tokens"] == ()
    assert candidates.search_options["deletion_repair"]["synthetic_text"] == ""
    assert any(not candidate.tokens and candidate.cost == 1 for candidate in candidates.candidates)


def test_source_choice_skips_top_empty_text_and_preserves_remaining_punctuation(monkeypatch):
    objective = build(monkeypatch, {(): "", (0,): "One, TWO three!", (1,): "One two three.", (2,): "One two, three!"})
    candidates = objective.refresh(torch.zeros(1), "One two three")
    synthetic = next(candidate for candidate in candidates.candidates if candidate.origin == ORIGIN)
    assert synthetic.text == "One, three!"
    assert candidates.search_options["deletion_repair"]["source_tokens"] == (0,)


def test_deduplication_does_not_try_a_second_deletion(monkeypatch):
    objective = build(monkeypatch, {(0,): "a b c", (1,): "a c", (2,): "a B c", (3,): "a b c."})
    candidates = objective.refresh(torch.zeros(1), "a b c")
    assert len(candidates.candidates) == 4
    assert not any(candidate.origin == ORIGIN for candidate in candidates.candidates)
    assert candidates.search_options["deletion_repair"]["added_after_deduplication"] is False


def test_real_whisper_tokenizer_empty_deletion_path_is_retained(monkeypatch):
    from whisper.tokenizer import get_tokenizer
    tokenizer = get_tokenizer(True, num_languages=99, language="en", task="transcribe")
    texts = ["Hello!", "hello!", "Hello", "hello"]
    paths = [(tuple(tokenizer.encode(text)), -index) for index, text in enumerate(texts)]
    monkeypatch.setattr(r1_sequence, "native_nbest", lambda *_: (paths, {}))
    surrogate = SimpleNamespace(tokenizer=tokenizer, language="en", nll_transcript_policy="original")
    candidates = DeletionRepairObjective(surrogate).refresh(torch.zeros(1), "Hello!")
    synthetic = next(candidate for candidate in candidates.candidates if candidate.origin == ORIGIN)
    assert synthetic.tokens == ()
    assert synthetic.text == ""
    assert synthetic.cost == 1


def test_native_beam_failure_is_not_repaired_using_stale_paths(monkeypatch):
    objective = build(monkeypatch, {(0,): "a b c", (1,): "a b c.", (2,): "A b c", (3,): "a B c"})
    objective.refresh(torch.zeros(1), "a b c")
    def failure(*_):
        raise DegenerateRiskError("Native beam search returned no hypotheses")
    monkeypatch.setattr(r1_sequence, "native_nbest", failure)
    with pytest.raises(DegenerateRiskError, match="Native beam"):
        objective.refresh(torch.zeros(1), "a b c", step=5)


def test_single_registered_deletion_can_still_fail_and_does_not_expand(monkeypatch):
    objective = build(monkeypatch, {(0,): "a ... b", (1,): "a b", (2,): "a, b", (3,): "A b"})
    with pytest.raises(DegenerateRiskError, match="still has no incorrect"):
        objective.refresh(torch.zeros(1), "a b")
    assert len(objective.candidate_set.candidates) == 4
    assert objective.candidate_set.search_options["deletion_repair"]["added_after_deduplication"] is False
