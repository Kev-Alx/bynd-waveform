"""Independent numerical contracts for R1, including native Whisper beams."""
from __future__ import annotations

import math
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "improvement")]

from r1_sequence import (CandidateSet, DegenerateRiskError, SequenceRiskObjective,
                         calibrate_scale, expected_risk, gradient_probe,
                         make_candidate, native_nbest, normalized_path_score, word_cost)


def test_expected_edit_risk_matches_manual_value_and_detaches_costs():
    # Equal path probabilities make the risk an independently calculable mean.
    scores = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    costs = torch.tensor([0., .5, 1.], dtype=torch.float64, requires_grad=True)
    risk, posterior = expected_risk(scores, costs)
    assert risk.item() == pytest.approx(.5)
    assert posterior.tolist() == pytest.approx([1 / 3] * 3)
    score_gradient, cost_gradient = torch.autograd.grad(risk, (scores, costs), allow_unused=True)
    assert score_gradient.tolist() == pytest.approx([-1 / 6, 0., 1 / 6])
    assert cost_gradient is None


def test_risk_audio_gradient_agrees_with_central_finite_difference():
    # Each path has a different nonlinear response to an underlying physical
    # coordinate; costs and candidates stay fixed during this derivative.
    coordinates = torch.tensor([.2, -.3], dtype=torch.float64, requires_grad=True)
    costs = torch.tensor([0., .5, 1.], dtype=torch.float64)

    def objective(value):
        scores = torch.stack([value[0].sin(), value[1].square(), value[0] * value[1]])
        return expected_risk(scores, costs, tau=.7)[0]

    gradient, = torch.autograd.grad(objective(coordinates), coordinates)
    finite_difference = []
    for index in range(coordinates.numel()):
        perturbation = torch.zeros_like(coordinates)
        perturbation[index] = 1e-5
        finite_difference.append(((objective(coordinates + perturbation) - objective(coordinates - perturbation)) / 2e-5).item())
    assert gradient.tolist() == pytest.approx(finite_difference, rel=1e-8, abs=1e-10)


def test_lexical_plus_eos_scoring_excludes_all_prefix_positions():
    probabilities = torch.tensor([[.2, .3, .1, .4], [.1, .6, .2, .1],
                                  [.1, .1, .2, .6]], dtype=torch.float64)
    logits = probabilities.log().requires_grad_()
    # Prefix length two -> its last position predicts lexical token one;
    # the lexical-token position predicts EOS three. Row zero is unscored.
    score = normalized_path_score(logits, 2, [1], 3)
    assert score.item() == pytest.approx(math.log(.6))
    gradient, = torch.autograd.grad(score, logits)
    assert torch.equal(gradient[0], torch.zeros_like(gradient[0]))
    lexical_only = normalized_path_score(logits, 2, [1], 3, include_eos=False)
    assert lexical_only.item() == pytest.approx(math.log(.6))


def test_empty_path_is_actual_eos_probability_and_has_gradient():
    logits = torch.tensor([[100., -100., 0.], [0., 0., math.log(2.)]],
                          dtype=torch.float64, requires_grad=True)
    score = normalized_path_score(logits, 2, [], eot=2)
    assert score.item() == pytest.approx(math.log(.5))
    gradient, = torch.autograd.grad(score, logits)
    assert gradient[-1, 2].item() == pytest.approx(.5)
    assert gradient[0].abs().sum().item() == 0
    with pytest.raises(ValueError, match="cannot score empty"):
        normalized_path_score(logits, 2, [], 2, include_eos=False)


@pytest.mark.parametrize("hypothesis,cost,expected", [
    ("a b c d", 0., (0, 0, 0)),
    ("a z c d", .25, (1, 0, 0)),
    ("a b", .5, (0, 2, 0)),
    ("a b c d e f", .5, (0, 0, 2)),
    ("", 1., (0, 4, 0)),
    ("a b c d e f g h i", 1., (0, 0, 5)),
])
def test_word_risk_costs_and_uncapped_edit_accounting(hypothesis, cost, expected):
    measured, counts = word_cost("A, b c d!", hypothesis)
    assert measured == cost
    assert tuple(counts[k] for k in ("substitutions", "deletions", "insertions")) == expected
    assert counts["reference_units"] == 4


def test_minimizing_negative_risk_increases_error_mass():
    scores = torch.tensor([.2, -.1], dtype=torch.float64, requires_grad=True)
    costs = torch.tensor([0., 1.], dtype=torch.float64)
    old_risk, _ = expected_risk(scores, costs)
    minimized_loss = -old_risk * 3
    gradient, = torch.autograd.grad(minimized_loss, scores)
    new_risk, _ = expected_risk(scores.detach() - .1 * gradient, costs)
    assert new_risk > old_risk


def test_calibration_uses_gradient_norms_in_same_physical_coordinates():
    probes = []
    for index, (stratum, factor) in enumerate(zip(("short", "median", "long"), (2., 5., 3.))):
        delta = torch.tensor([.2, -.1], dtype=torch.float64, requires_grad=True)
        risk = delta[0] + 2 * delta[1]
        nll = factor * risk
        probe = gradient_probe(risk, nll, delta, str(index), stratum)
        assert probe["ratio"] == pytest.approx(factor)
        assert probe["gradient_cosine"] == pytest.approx(1.)
        probes.append(probe)
    result = calibrate_scale(probes)
    assert result["scale"] == pytest.approx(3.)
    assert result["uncalibrated_control_scale"] == 1.


def test_calibration_refuses_zero_gradient_and_does_not_inflate_scale():
    delta = torch.tensor([1., 2.], requires_grad=True)
    flat_risk = (delta * 0).sum()
    probe = gradient_probe(flat_risk, delta.sum(), delta, "a", "short")
    assert probe["risk_gradient_norm"] == 0
    assert probe["ratio"] is None
    rows = [dict(probe, clip_id=str(index), duration_stratum=stratum)
            for index, stratum in enumerate(("short", "median", "long"))]
    with pytest.raises(DegenerateRiskError, match="zero/nonfinite"):
        calibrate_scale(rows)
    for row in rows:
        row.update(nll_gradient_norm=1., risk_gradient_norm=1e-7)
    with pytest.raises(DegenerateRiskError, match="outside"):
        calibrate_scale(rows)


def test_candidate_scores_handle_prefix_lexical_and_empty_paths_together():
    # A simple token-conditioned table permits a complete independent expected
    # path calculation and catches accidentally scoring task/prompt tokens.
    class Tokenizer:
        eot = 3
        sot_sequence_including_notimestamps = (4, 5)
        def decode(self, tokens):
            return " ".join("abc"[t] for t in tokens)

    class Decoder:
        def __call__(self, inputs, features):
            rows = []
            for token in inputs[0]:
                table = torch.tensor([.1, .2, .3, .4, .0, .0], dtype=torch.float64)
                if token == 5:
                    table = torch.tensor([.4, .2, .1, .3, .0, .0], dtype=torch.float64)
                # The additive feature distinguishes gradients without changing
                # the token-independent expected values at feature zero.
                rows.append(table.clamp_min(1e-30).log() + features * torch.arange(6, dtype=torch.float64))
            return torch.stack(rows)[None]

    surrogate = SimpleNamespace(language="en", nll_timestamp_policy="no_timestamps", tokenizer=Tokenizer(),
                                model=SimpleNamespace(encoder=lambda mel: mel,
                                                      decoder=Decoder(), dims=SimpleNamespace(n_text_ctx=20)),
                                mel=lambda audio: audio.sum())
    rows = tuple(make_candidate(surrogate, tokens, "a", "test") for tokens in ((0,), (), (1,)))
    candidates = CandidateSet(rows, "a", 0, 3, False, {})
    audio = torch.zeros(2, dtype=torch.float64, requires_grad=True)
    scores = SequenceRiskObjective(surrogate).score_candidates(audio, candidates)
    assert scores.tolist() == pytest.approx([math.log(.4), math.log(.3), (math.log(.2) + math.log(.4)) / 2])
    gradient, = torch.autograd.grad(scores.sum(), audio)
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0


def test_refresh_happens_once_per_five_updates_and_retains_degenerate_evidence(monkeypatch):
    import r1_sequence

    class Tokenizer:
        eot = 3
        def encode(self, text):
            return [0]
        def decode(self, tokens):
            return " ".join("abc"[token] for token in tokens)

    surrogate = SimpleNamespace(language="en", tokenizer=Tokenizer(), nll_transcript_policy="original")
    calls = []
    def decode(surrogate, audio, k):
        calls.append(audio.detach().clone())
        return [((0,), -.2), ((1,), -.3), ((2,), -.4), ((), -.5)], {}

    monkeypatch.setattr(r1_sequence, "native_nbest", decode)
    objective = SequenceRiskObjective(surrogate)
    objective.score_candidates = lambda audio, candidates: audio.sum() * torch.arange(len(candidates.candidates))
    for step in range(11):
        audio = torch.tensor([float(step)], requires_grad=True)
        result = objective.evaluate(audio, "a", step)
        assert result.candidate_set.refreshed_at == (step // 5) * 5
    assert len(calls) == 3

    # Distinct token sequences can all normalize to the correct words. Their
    # diversity does not justify pretending a nonzero risk gradient exists.
    surrogate.tokenizer.decode = lambda tokens: "a"
    monkeypatch.setattr(r1_sequence, "native_nbest", lambda *_: ([((0,) * length, -.1) for length in range(1, 5)], {}))
    objective = SequenceRiskObjective(surrogate)
    with pytest.raises(DegenerateRiskError, match="No incorrect"):
        objective.evaluate(torch.ones(1, requires_grad=True), "a", 0)
    assert len(objective.candidate_set.receipt()["candidates"]) == 4
    with pytest.raises(DegenerateRiskError, match="No incorrect"):
        objective.evaluate(torch.ones(1, requires_grad=True), "a", 1)


def test_real_native_whisper_architecture_produces_distinct_beam_candidates_and_audio_gradient():
    # A random, deliberately tiny native Whisper uses real tokenization, KV
    # caching, beam search, decoder EOS finalization and full differentiable
    # scoring. No checkpoint, network, GPU or held-out target is used.
    from whisper.model import ModelDimensions, Whisper
    from whisper.tokenizer import get_tokenizer
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(11)
        dims = ModelDimensions(n_mels=80, n_audio_ctx=4, n_audio_state=8, n_audio_head=2,
                               n_audio_layer=1, n_vocab=51865, n_text_ctx=24,
                               n_text_state=8, n_text_head=2, n_text_layer=1)
        model = Whisper(dims).eval().requires_grad_(False)
        # Native construction expects a checkpoint to populate torch.empty
        # decoder positions. This synthetic model has no checkpoint.
        torch.nn.init.normal_(model.decoder.positional_embedding, std=.02)
        tokenizer = get_tokenizer(model.is_multilingual, num_languages=model.num_languages,
                                  language="en", task="transcribe")
        # Real encoder context, with all input coordinates participating.
        ramp = torch.linspace(-1., 1., 640).reshape(1, 80, 8)
        surrogate = SimpleNamespace(language="en", model=model, tokenizer=tokenizer,
                                    nll_timestamp_policy="no_timestamps", nll_transcript_policy="original",
                                    mel=lambda audio: ramp + audio.mean())
        audio = torch.tensor([.1, -.1], requires_grad=True)
        paths, policy = native_nbest(surrogate, audio)
        assert len(paths) == len({path for path, _ in paths}) == 4
        assert policy["beam_size"] == 4
        objective = SequenceRiskObjective(surrogate)
        result = objective.evaluate(audio, "a")
        assert result.candidate_set.decoded_count == 4
        assert 4 <= len(result.candidate_set.candidates) <= 5
        assert any(c.cost == 0 for c in result.candidate_set.candidates)
        assert torch.isfinite(result.scores).all()
        gradient, = torch.autograd.grad(result.risk, audio)
        assert torch.isfinite(gradient).all()
        assert gradient.abs().sum() > 0
        assert all(parameter.grad is None for parameter in model.parameters())
        assert not any(module._forward_hooks for module in model.modules())
    finally:
        torch.set_num_threads(old_threads)
