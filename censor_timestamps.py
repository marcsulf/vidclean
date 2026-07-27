"""Interval logic for the video censor tool.

Given transcribed word spans and a :class:`WordMatcher`, this module builds
the list of ``(start, end)`` audio time intervals that should be muted or
beeped in the final video output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Protocol, Sequence, Tuple


class _MatcherProtocol(Protocol):
    def matches(self, token: str) -> bool: ...

    def find_matches(self, tokens: Sequence[str]) -> List[Tuple[int, int]]: ...


@dataclass(frozen=True)
class WordSpan:
    """A single transcribed word with start/end time in seconds."""

    text: str
    start: float
    end: float


def _clamp(value: float, low: float, high: float | None) -> float:
    if value < low:
        return low
    if high is not None and value > high:
        return high
    return value


def build_mute_intervals(
    words: Sequence[WordSpan],
    matcher: _MatcherProtocol,
    pad_s: float | None = None,
    total_duration_s: float | None = None,
    pre_pad_s: float | None = None,
    post_pad_s: float | None = None,
) -> List[Tuple[float, float]]:
    """Return sorted, non-overlapping ``(start, end)`` mute intervals.

    Parameters
    ----------
    words:
        Transcribed word spans in ascending time order.
    matcher:
        A :class:`WordMatcher` used to decide which words should be muted.
    pad_s:
        Legacy symmetric padding. If ``pre_pad_s`` / ``post_pad_s`` are not
        provided, this value is used for both sides. Defaults to 0.05 s when
        no explicit padding is given.
    pre_pad_s:
        Seconds to expand each matching word interval on the leading edge.
        Overrides ``pad_s`` on that side. Must be >= 0.
    post_pad_s:
        Seconds to expand each matching word interval on the trailing edge.
        Overrides ``pad_s`` on that side. Must be >= 0.
    total_duration_s:
        Optional media duration used to clamp the upper bound of intervals.
    """
    default_sym = 0.05 if pad_s is None else pad_s
    pre = default_sym if pre_pad_s is None else pre_pad_s
    post = default_sym if post_pad_s is None else post_pad_s
    if pre < 0:
        pre = 0.0
    if post < 0:
        post = 0.0

    raw: List[Tuple[float, float]] = []
    # Filter to indexable, non-empty spans up front so index math lines up.
    valid_words: List[WordSpan] = [
        w for w in words if w.text and w.end > w.start
    ]
    if valid_words:
        # Prefer the phrase-aware find_matches API when available; fall back
        # to per-token matches() for older matcher shapes.
        matches: List[Tuple[int, int]] = []
        if hasattr(matcher, "find_matches"):
            try:
                matches = list(matcher.find_matches([w.text for w in valid_words]))
            except Exception:  # noqa: BLE001 - never let matcher errors crash pipeline
                matches = []
        if not matches:
            matches = [
                (i, i + 1) for i, w in enumerate(valid_words) if matcher.matches(w.text)
            ]

        for start_idx, end_idx in matches:
            if start_idx < 0 or end_idx > len(valid_words) or end_idx <= start_idx:
                continue
            span_start = valid_words[start_idx].start
            span_end = valid_words[end_idx - 1].end
            s = _clamp(span_start - pre, 0.0, total_duration_s)
            e = _clamp(span_end + post, 0.0, total_duration_s)
            if e > s:
                raw.append((s, e))

    if not raw:
        return []

    raw.sort(key=lambda x: x[0])
    merged: List[Tuple[float, float]] = [raw[0]]
    for s, e in raw[1:]:
        prev_s, prev_e = merged[-1]
        if s <= prev_e:
            if e > prev_e:
                merged[-1] = (prev_s, e)
        else:
            merged.append((s, e))
    return merged


def flatten_transcript_words(segments: Iterable[dict]) -> List[WordSpan]:
    """Flatten Whisper's ``segments`` structure into :class:`WordSpan` items.

    Whisper's transcribe() result has ``segments[i]["words"] = [{word, start, end}, ...]``
    when ``word_timestamps=True``.
    """
    out: List[WordSpan] = []
    for seg in segments:
        words = seg.get("words") or []
        for w in words:
            text = str(w.get("word", "")).strip()
            start = float(w.get("start", 0.0))
            end = float(w.get("end", start))
            if not text or end <= start:
                continue
            out.append(WordSpan(text=text, start=start, end=end))
    return out
