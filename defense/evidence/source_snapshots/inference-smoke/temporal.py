"""Overlapping local decoding with an explicit partition of word ownership."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Window:
    start: float
    end: float
    own_start: float
    own_end: float


def windows(duration, width=8., stride=4.):
    if duration <= 0 or width <= 0 or not 0 < stride <= width:
        raise ValueError('Require positive duration and 0 < stride <= width')
    if duration <= width:
        return [Window(0., duration, 0., duration)]
    starts = [0.]
    while starts[-1] + width < duration:
        nxt = min(starts[-1] + stride, duration - width)
        if nxt <= starts[-1]:
            break
        starts.append(nxt)
    centers = [(s + min(s + width, duration)) / 2 for s in starts]
    boundaries = [0.] + [(a+b)/2 for a,b in zip(centers,centers[1:])] + [duration]
    return [Window(s, min(s+width,duration), boundaries[i], boundaries[i+1])
            for i,s in enumerate(starts)]


def stitch_words(parts, spans):
    """parts contain local model timestamps, without any clean/reference input."""
    if len(parts) != len(spans):
        raise ValueError('One hypothesis list is required per window')
    selected = []
    for index, (words, span) in enumerate(zip(parts,spans)):
        for word in words:
            start, end = span.start + float(word['start']), span.start + float(word['end'])
            if end < start or start < span.start - .01 or end > span.end + .1:
                raise ValueError('Invalid local word timestamps')
            midpoint = (start+end)/2
            if span.own_start <= midpoint < span.own_end or (
                    index == len(spans)-1 and midpoint == span.own_end):
                selected.append({'word':word['word'].strip(), 'start':start, 'end':end,
                                 'window_index':index, 'probability':word.get('probability')})
    selected.sort(key=lambda w: ((w['start']+w['end'])/2, w['window_index']))
    return ' '.join(w['word'] for w in selected if w['word']), selected
