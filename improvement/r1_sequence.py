"""R1: detached native Whisper beam candidates and differentiable word risk.

This implementation stays outside the historical reproduction sources. Candidate
search is one native beam search, not repeated greedy decoding. Beam log scores
are used only to order candidates; the objective always recomputes *unfiltered*
lexical-path probabilities, including EOS, with gradients through the audio.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from statistics import median
from typing import Mapping, Sequence

import torch
from torch import Tensor
from torch.nn import functional as F
from whisper.decoding import DecodingOptions, DecodingTask, LogitFilter, SequenceRanker

from beyondwaveform.evaluation import error_counts, normalize_text
from beyondwaveform.models.whisper import nll_transcript, teacher_forcing_prefix


class DegenerateRiskError(RuntimeError):
    """A declared candidate/calibration gate failed; do not silently repair it."""


@dataclass(frozen=True)
class Candidate:
    tokens: tuple[int, ...]
    text: str
    cost: float
    substitutions: int
    deletions: int
    insertions: int
    reference_words: int
    origin: str
    search_score: float | None = None


@dataclass(frozen=True)
class CandidateSet:
    candidates: tuple[Candidate, ...]
    reference: str
    refreshed_at: int
    decoded_count: int
    decoded_top_empty: bool
    search_options: dict

    def receipt(self) -> dict:
        return asdict(self)


@dataclass
class RiskResult:
    risk: Tensor
    scores: Tensor
    posterior: Tensor
    costs: Tensor
    candidate_set: CandidateSet

    def diagnostics(self) -> dict:
        p = self.posterior.detach()
        candidates = self.candidate_set.candidates
        return {
            "risk": self.risk.detach().item(),
            "scores": self.scores.detach().tolist(),
            "posterior": p.tolist(),
            "costs": self.costs.detach().tolist(),
            "posterior_entropy": -(p * p.clamp_min(torch.finfo(p.dtype).tiny).log()).sum().item(),
            "candidate_count": len(candidates),
            "unique_token_paths": len({c.tokens for c in candidates}),
            "unique_normalized_texts": len({normalize_text(c.text, "en") for c in candidates}),
            "decoded_top_empty": self.candidate_set.decoded_top_empty,
            "empty_candidate_count": sum(not c.tokens for c in candidates),
            "refreshed_at": self.candidate_set.refreshed_at,
        }


class _LexicalOrEOS(LogitFilter):
    """No timestamp, task, language or other special tokens inside a path."""
    def __init__(self, eot: int):
        self.eot = eot

    def apply(self, logits: Tensor, tokens: Tensor) -> None:
        logits[:, self.eot + 1:] = -torch.inf


class _CaptureRanker(SequenceRanker):
    """Capture native BeamSearchDecoder.finalize output before top-one collapse."""
    def __init__(self):
        self.paths: list[tuple[tuple[int, ...], float]] = []

    def rank(self, tokens, sum_logprobs):
        if len(tokens) != 1:
            raise ValueError("R1 decoding accepts exactly one utterance")
        paths = [(tuple(t.tolist()), float(lp) / (len(t) + 1))
                 for t, lp in zip(tokens[0], sum_logprobs[0])]
        if not paths:
            raise DegenerateRiskError("Native beam search returned no hypotheses")
        self.paths = sorted(paths, key=lambda item: (-item[1], item[0]))
        # Native's default ranker divides by zero on an empty lexical path.
        # L+1 is both empty-safe and the registered scoring normalization.
        return [max(range(len(paths)), key=lambda i: paths[i][1])]


@torch.no_grad()
def native_nbest(surrogate, audio: Tensor, k: int = 4) -> tuple[list, dict]:
    """Return K distinct paths from one native no-timestamp beam search.

    Native finalization appends EOS to unfinished paths at its context/sample
    limit. Those paths are subsequently scored with the true EOS probability;
    their filtered search scores never enter the differentiable objective.
    Initial EOS is allowed so an empty candidate is retained and diagnosed.
    """
    if isinstance(k, bool) or not isinstance(k, int) or k < 2:
        raise ValueError("N-best beam width must be an integer >= 2")
    if surrogate.nll_timestamp_policy != "no_timestamps":
        raise ValueError("R1 initial condition requires no-timestamps conditioning")
    options = DecodingOptions(language=surrogate.language, task="transcribe",
                              beam_size=k, temperature=0, patience=1,
                              without_timestamps=True, suppress_blank=False,
                              fp16=False)
    task = DecodingTask(surrogate.model, options)
    task.logit_filters.append(_LexicalOrEOS(surrogate.tokenizer.eot))
    capture = _CaptureRanker()
    task.sequence_ranker = capture
    task.run(surrogate.mel(audio.detach()))
    paths = []
    seen = set()
    for tokens, score in capture.paths:
        if any(token < 0 or token >= surrogate.tokenizer.eot for token in tokens):
            raise DegenerateRiskError("Native hypothesis contains a nonlexical token")
        if tokens not in seen:
            seen.add(tokens)
            paths.append((tokens, score))
    if len(paths) != k:
        raise DegenerateRiskError(f"Expected {k} distinct native beam paths; got {len(paths)}")
    policy = asdict(options)
    policy.update(lexical_or_eos_only=True, rank_normalization="lexical_length+1",
                  search="one native BeamSearchDecoder.finalize call",
                  context_limit_eos="native finalize appends; rescored with actual EOS probability")
    return paths, policy


def word_cost(reference: str, hypothesis: str) -> tuple[float, dict]:
    """Detached conventional word edits, capped at one for the risk only."""
    counts = error_counts(reference, hypothesis, "en")
    edits = sum(counts[key] for key in ("substitutions", "deletions", "insertions"))
    return min(edits / max(1, counts["reference_units"]), 1.0), counts


def make_candidate(surrogate, tokens: Sequence[int], reference: str, origin: str,
                   search_score: float | None = None) -> Candidate:
    path = tuple(int(t) for t in tokens)
    if any(t < 0 or t >= surrogate.tokenizer.eot for t in path):
        raise ValueError("Candidate paths must contain lexical tokens only (EOS is implicit)")
    text = surrogate.tokenizer.decode(list(path)).strip()
    cost, counts = word_cost(reference, text)
    return Candidate(path, text, cost, counts["substitutions"], counts["deletions"],
                     counts["insertions"], counts["reference_units"], origin, search_score)


def normalized_path_score(logits: Tensor, prefix_length: int, lexical: Sequence[int],
                          eot: int, include_eos: bool = True) -> Tensor:
    """Score lexical next tokens and terminal EOS; never score prefix tokens.

    ``logits`` covers decoder input ``prefix + lexical``. The empty path is
    exactly log p(EOS | prefix,audio), with divisor one. The optional EOS-free
    ablation rejects empty paths and normalizes by lexical length.
    """
    if logits.ndim == 3:
        if logits.shape[0] != 1:
            raise ValueError("Expected single-path logits")
        logits = logits[0]
    if logits.ndim != 2 or prefix_length < 1:
        raise ValueError("Invalid path logits or prefix")
    lexical = tuple(lexical)
    if any(t < 0 or t >= eot for t in lexical):
        raise ValueError("Path includes a nonlexical token")
    if not include_eos and not lexical:
        raise ValueError("EOS-free nonempty-candidate ablation cannot score empty text")
    labels = (*lexical, eot) if include_eos else lexical
    start = prefix_length - 1
    stop = start + len(labels)
    if stop > logits.shape[0]:
        raise ValueError("Decoder outputs do not cover all lexical/EOS labels")
    ids = torch.tensor(labels, dtype=torch.long, device=logits.device)
    # Preserve float64 in finite-difference tests; production Whisper is fp32.
    values = logits[start:stop]
    if values.dtype in (torch.float16, torch.bfloat16):
        values = values.float()
    return F.log_softmax(values, dim=-1).gather(1, ids[:, None]).sum() / len(labels)


def expected_risk(scores: Tensor, costs: Tensor, tau: float = 1.0) -> tuple[Tensor, Tensor]:
    if scores.ndim != 1 or costs.shape != scores.shape or scores.numel() < 2:
        raise ValueError("Risk requires >=2 aligned scalar candidate scores/costs")
    if not math.isfinite(tau) or tau <= 0:
        raise ValueError("Risk temperature must be finite and positive")
    if not torch.isfinite(scores).all() or not torch.isfinite(costs).all():
        raise DegenerateRiskError("Nonfinite candidate score/cost")
    detached_costs = costs.detach().to(device=scores.device, dtype=scores.dtype)
    if (detached_costs < 0).any() or (detached_costs > 1).any():
        raise ValueError("Risk costs must be bounded in [0,1]")
    posterior = torch.softmax(scores / tau, dim=0)
    return (posterior * detached_costs).sum(), posterior


class SequenceRiskObjective:
    """Refresh hypotheses every five zero-based optimizer updates, score each step."""
    def __init__(self, surrogate, k: int = 4, refresh_every: int = 5,
                 tau: float = 1.0, include_eos: bool = True):
        if surrogate.language != "en":
            raise ValueError("Registered R1 word-risk condition currently supports English only")
        if refresh_every < 1 or not isinstance(refresh_every, int):
            raise ValueError("refresh_every must be a positive integer")
        if not math.isfinite(tau) or tau <= 0:
            raise ValueError("Risk temperature must be finite and positive")
        self.surrogate, self.k, self.refresh_every = surrogate, k, refresh_every
        self.tau, self.include_eos = tau, include_eos
        self.candidate_set: CandidateSet | None = None

    @torch.no_grad()
    def refresh(self, audio: Tensor, text: str, step: int = 0) -> CandidateSet:
        if step < 0 or not isinstance(step, int):
            raise ValueError("Refresh step must be a zero-based nonnegative integer")
        if not normalize_text(text, "en"):
            raise ValueError("Empty normalized reference")
        paths, policy = native_nbest(self.surrogate, audio, self.k)
        decoded_top_empty = not paths[0][0]
        rows = [make_candidate(self.surrogate, tokens, text, "native_beam", score)
                for tokens, score in paths if self.include_eos or tokens]
        target = nll_transcript(text, self.surrogate.nll_transcript_policy)
        reference_tokens = tuple(self.surrogate.tokenizer.encode(target))
        if reference_tokens not in {c.tokens for c in rows}:
            rows.append(make_candidate(self.surrogate, reference_tokens, text, "reference"))
        self.candidate_set = CandidateSet(tuple(rows), text, step, len(paths),
                                          decoded_top_empty, policy)
        if len(rows) < 2 or not any(c.cost > 0 for c in rows):
            # Retain the actual candidates so a stopped canonical condition has
            # inspectable evidence rather than only an exception message.
            raise DegenerateRiskError("No incorrect native candidate; fallback is a separate condition")
        return self.candidate_set

    def score_candidates(self, audio: Tensor, candidate_set: CandidateSet) -> Tensor:
        prefix = teacher_forcing_prefix(self.surrogate.tokenizer,
                                       self.surrogate.nll_timestamp_policy)
        mel = self.surrogate.mel(audio)
        features = self.surrogate.model.encoder(mel)
        scores = []
        for candidate in candidate_set.candidates:
            context = (*prefix, *candidate.tokens)
            if len(context) > self.surrogate.model.dims.n_text_ctx:
                raise ValueError("Candidate exceeds Whisper text context")
            inputs = torch.tensor([context], dtype=torch.long, device=audio.device)
            logits = self.surrogate.model.decoder(inputs, features)
            scores.append(normalized_path_score(logits, len(prefix), candidate.tokens,
                                                self.surrogate.tokenizer.eot, self.include_eos))
        return torch.stack(scores)

    def evaluate(self, audio: Tensor, text: str, step: int = 0) -> RiskResult:
        if step < 0 or not isinstance(step, int):
            raise ValueError("Optimizer step must be a zero-based nonnegative integer")
        current = self.candidate_set
        # A new reference always starts a new candidate cache. Callers reusing an
        # instance across utterances with identical text must explicitly refresh.
        if current is None or current.reference != text or step >= current.refreshed_at + self.refresh_every:
            current = self.refresh(audio, text, step)
        elif step < current.refreshed_at:
            raise ValueError("Candidate cache is from a later optimizer step; refresh for a new utterance")
        if len(current.candidates) < 2 or not any(c.cost > 0 for c in current.candidates):
            raise DegenerateRiskError("No incorrect native candidate; fallback is a separate condition")
        scores = self.score_candidates(audio, current)
        costs = scores.new_tensor([c.cost for c in current.candidates]).detach()
        risk, posterior = expected_risk(scores, costs, self.tau)
        return RiskResult(risk, scores, posterior, costs, current)

    def loss(self, audio: Tensor, text: str, step: int, scale: float) -> tuple[Tensor, RiskResult]:
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("R1 maximization scale must be finite and positive")
        result = self.evaluate(audio, text, step)
        return -scale * result.risk, result


def gradient_probe(risk: Tensor, nll: Tensor, delta: Tensor, clip_id: str,
                   duration_stratum: str) -> dict:
    """Measure source losses against the exact same physical-feature delta."""
    if risk.ndim or nll.ndim:
        raise ValueError("Calibration losses must be scalars")
    risk_gradient, = torch.autograd.grad(risk, delta, retain_graph=True)
    nll_gradient, = torch.autograd.grad(nll, delta, retain_graph=True)
    if not torch.isfinite(risk_gradient).all() or not torch.isfinite(nll_gradient).all():
        raise DegenerateRiskError("Nonfinite physical-feature gradient")
    gn = torch.linalg.vector_norm(nll_gradient.double()).item()
    gr = torch.linalg.vector_norm(risk_gradient.double()).item()
    valid = math.isfinite(gn) and math.isfinite(gr) and gr > 1e-8 and gn > 0
    ratio = gn / gr if valid else None
    cosine = (nll_gradient.double().flatten() @ risk_gradient.double().flatten()).item() / (gn * gr) if valid else None
    return {"clip_id": clip_id, "duration_stratum": duration_stratum,
            "nll_gradient_norm": gn, "risk_gradient_norm": gr,
            "ratio": ratio, "gradient_cosine": cosine, "valid": valid,
            "nll": nll.detach().item(), "risk": risk.detach().item()}


def calibrate_scale(probes: Sequence[Mapping]) -> dict:
    """Freeze exactly three source-only probes; no zero-norm coefficient inflation."""
    if len(probes) != 3 or len({p["clip_id"] for p in probes}) != 3:
        raise ValueError("Calibration requires exactly three distinct declared development clips")
    if {p["duration_stratum"] for p in probes} != {"short", "median", "long"}:
        raise ValueError("Calibration requires the declared short/median/long duration strata")
    ratios = []
    for probe in probes:
        gn, gr = probe["nll_gradient_norm"], probe["risk_gradient_norm"]
        if not math.isfinite(gn) or not math.isfinite(gr) or gn <= 0 or gr <= 1e-8:
            raise DegenerateRiskError("R1 calibration has a zero/nonfinite physical-feature gradient")
        ratio = gn / gr
        if not math.isfinite(ratio):
            raise DegenerateRiskError("R1 calibration ratio is nonfinite")
        ratios.append(ratio)
    scale = median(ratios)
    if not 1e-4 <= scale <= 1e4:
        raise DegenerateRiskError(f"R1 calibrated scale {scale} is outside [1e-4,1e4]")
    return {"scale": scale, "method": "median lexical-sum-NLL / bounded-risk physical-feature gradient norm",
            "probes": [dict(p) for p in probes], "ratios": ratios,
            "uncalibrated_control_scale": 1.0, "frozen": True}
