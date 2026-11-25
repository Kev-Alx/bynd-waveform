"""The single preregistered R1 repair: one source-derived deletion candidate.

The added text is a synthetic counterfactual, NOT a posterior beam hypothesis.
It is scored with the same differentiable lexical+EOS path probability as all
other candidates. Only candidate construction changes; transcript/audio content
requirements and the frozen source-only calibration protocol remain in force.
"""
from __future__ import annotations

from dataclasses import replace

import torch

from r1_sequence import DegenerateRiskError, SequenceRiskObjective, make_candidate


ORIGIN = "synthetic_source_best_middle_word_deletion"
RULE = {
    "name": "R1-native-plus-deletion",
    "maximum_synthetic_candidates": 1,
    "source": "highest-ranked nonempty native beam hypothesis text",
    "operation": "split on whitespace; delete words[len(words)//2]; join remaining words with one space",
    "preserve": "all remaining words, case and punctuation",
    "tokenization": "same frozen public Whisper-small tokenizer encode",
    "empty_deletion_result": "retain and score actual EOS probability",
    "deduplication": "if token path already exists, add nothing and try no other deletion",
    "origin": ORIGIN,
    "posterior_beam_hypothesis": False,
    "additional_fallbacks": False,
}


class DeletionRepairObjective(SequenceRiskObjective):
    """Native K=4 plus reference plus at most one fixed deletion path."""

    @torch.no_grad()
    def refresh(self, audio, text, step=0):
        previous = self.candidate_set
        try:
            native = super().refresh(audio, text, step)
        except DegenerateRiskError as error:
            # The canonical all-correct set is intentionally retained by the
            # frozen base class. Other beam failures must never reuse old paths.
            if (not str(error).startswith("No incorrect native candidate;")
                    or self.candidate_set is previous or self.candidate_set is None):
                raise
            native = self.candidate_set
        source = next((candidate for candidate in native.candidates
                       if candidate.origin == "native_beam" and candidate.text.split()), None)
        if source is None:
            raise DegenerateRiskError("Deletion repair has no nonempty native source text; no other fallback")
        words = source.text.split()
        index = len(words) // 2
        deletion_text = " ".join(words[:index] + words[index + 1:])
        tokens = tuple(self.surrogate.tokenizer.encode(deletion_text))
        rows = list(native.candidates)
        added = tokens not in {candidate.tokens for candidate in rows}
        if added:
            rows.append(make_candidate(self.surrogate, tokens, text, ORIGIN))
        policy = dict(native.search_options)
        policy["deletion_repair"] = {**RULE, "source_tokens": source.tokens,
                                      "source_text": source.text, "deleted_word_index": index,
                                      "deleted_word": words[index], "synthetic_text": deletion_text,
                                      "synthetic_tokens": tokens, "added_after_deduplication": added}
        self.candidate_set = replace(native, candidates=tuple(rows), search_options=policy)
        if len(rows) < 2 or not any(candidate.cost > 0 for candidate in rows):
            raise DegenerateRiskError("Single deletion repair still has no incorrect candidate; no other fallback")
        return self.candidate_set
