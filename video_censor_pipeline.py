"""Orchestrates the full video-censor pipeline.

Sequence:
    1. probe input duration
    2. extract audio to a temp WAV
    3. transcribe with Whisper (word-level timestamps)
    4. match transcribed words against censor list, build padded intervals
    5. run ffmpeg to mute or beep the intervals in the output video
    6. clean up temp files

Progress is normalized to 0.0-1.0 across the whole pipeline using fixed
phase weights, and forwarded to ``progress_cb`` together with a phase label.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event
from typing import Callable, List, Optional, Sequence, Tuple

from censor_audio_io import (
    MuteJobConfig,
    OUTPUT_MODE_ADD_TRACK,
    OUTPUT_MODE_REPLACE,
    VALID_OUTPUT_MODES,
    adjust_output_path_for_mode,
    cleanup_temp_file,
    extract_audio_wav,
    make_temp_wav_path,
    probe_duration_s,
    run_censor_encode,
    safe_default_output_path,
)
from censor_cache import (
    CachedTranscript,
    cache_path_for,
    load_cache,
    save_cache,
)
from censor_timestamps import WordSpan, build_mute_intervals
from censor_transcribe import transcribe
from censor_wordlist import build_matcher, load_wordlist


LogCB = Callable[[str], None]
# progress_cb signature: (fraction_0_1, phase_label, eta_seconds_or_None)
ProgressCB = Callable[[float, str, Optional[float]], None]


@dataclass
class PipelineConfig:
    input_video: str
    output_video: str = ""
    wordlist_path: str = "censor_words.txt"
    model_size: str = "medium"
    device: str = "auto"  # "auto" | "cuda" | "cpu"
    engine: str = "openai-whisper"  # "openai-whisper" | "faster-whisper"
    model_path: str = ""  # optional local dir or HF repo id; overrides model_size when set
    mode: str = "mute"  # "mute" | "beep"
    pre_pad_ms: int = 150
    post_pad_ms: int = 50
    beep_freq_hz: float = 1000.0
    audio_delay_ms: int = 0  # positive = delay audio; negative = advance audio
    output_mode: str = OUTPUT_MODE_REPLACE  # "replace" | "add_track"
    ffmpeg_dir: str = ""  # optional folder containing ffmpeg/ffprobe binaries
    use_transcript_cache: bool = True
    force_retranscribe: bool = False

    def resolve(self) -> "PipelineConfig":
        out = self.output_video.strip() or safe_default_output_path(self.input_video)
        mode = (self.output_mode or OUTPUT_MODE_REPLACE).lower()
        if mode not in VALID_OUTPUT_MODES:
            mode = OUTPUT_MODE_REPLACE
        # For add_track, containers that don't reliably support multiple audio
        # streams (e.g. .webm, .ogv, .flv) get switched to .mp4 automatically.
        out = adjust_output_path_for_mode(out, mode)
        return PipelineConfig(
            input_video=str(Path(self.input_video)),
            output_video=str(Path(out)),
            wordlist_path=str(Path(self.wordlist_path)),
            model_size=self.model_size,
            device=self.device,
            engine=self.engine,
            model_path=self.model_path.strip() if self.model_path else "",
            mode=self.mode,
            pre_pad_ms=int(self.pre_pad_ms),
            post_pad_ms=int(self.post_pad_ms),
            beep_freq_hz=float(self.beep_freq_hz),
            audio_delay_ms=int(self.audio_delay_ms),
            output_mode=mode,
            ffmpeg_dir=self.ffmpeg_dir.strip(),
            use_transcript_cache=bool(self.use_transcript_cache),
            force_retranscribe=bool(self.force_retranscribe),
        )


@dataclass
class PipelineResult:
    output_video: str
    intervals: List[Tuple[float, float]] = field(default_factory=list)
    words_transcribed: int = 0
    words_matched: int = 0
    duration_s: float = 0.0
    censor_words: List[str] = field(default_factory=list)
    used_cached_transcript: bool = False


# Phase weights for a full run (must sum to 1.0). Transcription dominates.
_PHASE_WEIGHTS_FULL = {
    "probe": 0.01,
    "extract": 0.04,
    "transcribe": 0.85,
    "match": 0.02,
    "encode": 0.08,
}

# Phase weights when the transcript cache is hit and extract+transcribe are skipped.
_PHASE_WEIGHTS_CACHED = {
    "probe": 0.02,
    "match": 0.08,
    "encode": 0.90,
}


class _CancelledError(RuntimeError):
    pass


def run_pipeline(
    config: PipelineConfig,
    log_cb: Optional[LogCB] = None,
    progress_cb: Optional[ProgressCB] = None,
    cancel_flag: Optional[Event] = None,
) -> PipelineResult:
    """Run the full pipeline. Raises on error. Raises ``RuntimeError('Cancelled')``
    if ``cancel_flag`` is set at a check point."""
    cfg = config.resolve()
    log = log_cb or (lambda _msg: None)
    start_time = time.time()

    # Decide up front whether we will use the cache, so progress weights
    # match reality from the very first tick.
    cached: Optional[CachedTranscript] = None
    if cfg.use_transcript_cache and not cfg.force_retranscribe:
        try:
            cache_identifier = cfg.model_path.strip() or cfg.model_size
            cached = load_cache(cfg.input_video, cache_identifier, engine=cfg.engine)
        except Exception:  # noqa: BLE001 - never let cache errors block a run
            cached = None

    phase_weights = _PHASE_WEIGHTS_CACHED if cached else _PHASE_WEIGHTS_FULL

    def cancel_check() -> bool:
        return bool(cancel_flag and cancel_flag.is_set())

    def _raise_if_cancelled() -> None:
        if cancel_check():
            raise _CancelledError("Cancelled")

    completed_fraction = 0.0

    # Rolling-window ETA state. Whisper often emits several fast ticks up
    # front (early chunks / warm-up) that make ``elapsed / overall`` a bad
    # estimator of the true rate. Instead we keep a short history of
    # (wall_time, overall_fraction) samples and compute the rate over the
    # last ~60 seconds, giving a much more realistic ETA once real chunks
    # start flowing at their steady cadence.
    _ETA_WINDOW_S = 60.0
    _ETA_MIN_SAMPLES = 3
    _eta_history: list[Tuple[float, float]] = []

    def _emit(fraction_in_phase: float, phase: str) -> None:
        if progress_cb is None:
            return
        weight = phase_weights.get(phase, 0.0)
        overall = completed_fraction + max(0.0, min(1.0, fraction_in_phase)) * weight
        overall = max(0.0, min(0.999, overall))
        now = time.time()
        elapsed = now - start_time

        # Update rolling history and drop samples older than the window
        # (but always keep at least the last _ETA_MIN_SAMPLES samples so we
        # can compute a rate even during long lulls between real ticks).
        _eta_history.append((now, overall))
        cutoff = now - _ETA_WINDOW_S
        while (
            len(_eta_history) > _ETA_MIN_SAMPLES
            and _eta_history[0][0] < cutoff
        ):
            _eta_history.pop(0)

        eta: Optional[float]
        if len(_eta_history) >= 2 and overall > 0.01:
            oldest_ts, oldest_frac = _eta_history[0]
            delta_time = now - oldest_ts
            delta_frac = overall - oldest_frac
            if delta_time > 0.5 and delta_frac > 1e-4:
                rate = delta_frac / delta_time  # fraction per second
                eta = (1.0 - overall) / rate
            elif overall > 0.05:
                # Fall back to the classic estimator only once we've made
                # enough overall progress that it's likely stable.
                eta = elapsed * (1.0 - overall) / overall
            else:
                eta = None
        elif overall > 0.05:
            eta = elapsed * (1.0 - overall) / overall
        else:
            eta = None
        progress_cb(overall, phase, eta)

    try:
        # -- Phase: probe -----------------------------------------------------
        log(f"Input video: {cfg.input_video}")
        log(f"Output video: {cfg.output_video}")
        log(f"Output mode: {cfg.output_mode}")
        if cfg.ffmpeg_dir:
            log(f"FFmpeg folder override: {cfg.ffmpeg_dir}")
        _emit(0.0, "probe")
        duration_s = probe_duration_s(cfg.input_video, ffmpeg_dir=cfg.ffmpeg_dir or None)
        log(f"Media duration: {duration_s:.2f} s")
        _emit(1.0, "probe")
        completed_fraction += phase_weights["probe"]
        _raise_if_cancelled()

        # -- Phase: word list -------------------------------------------------
        log(f"Loading censor word list from {cfg.wordlist_path}")
        words = load_wordlist(cfg.wordlist_path)
        if not words:
            raise RuntimeError(
                f"Censor word list at '{cfg.wordlist_path}' is empty. "
                f"Add at least one word."
            )
        matcher = build_matcher(words)
        log(f"Loaded {len(words)} censor word(s): {', '.join(words)}")
        _raise_if_cancelled()

        # -- Phase: extract audio + transcribe (or cache) --------------------
        used_cached_transcript = False
        if cached is not None:
            log(
                f"Using cached transcript: {cache_path_for(cfg.input_video)} "
                f"({len(cached.words)} words, model={cfg.model_size})."
            )
            transcription_words = cached.words
            transcription_language = cached.detected_language
            used_cached_transcript = True
        else:
            if cfg.force_retranscribe:
                log("Force re-transcribe: ignoring any existing transcript cache.")
            elif cfg.use_transcript_cache:
                log("No valid transcript cache found; running full transcription.")
            else:
                log("Transcript cache disabled; running full transcription.")

            wav_path = make_temp_wav_path(cfg.input_video)
            try:
                extract_audio_wav(
                    cfg.input_video,
                    wav_path,
                    total_duration_s=duration_s,
                    log=log,
                    progress=lambda elapsed, total: _emit(elapsed / total if total else 0.0, "extract"),
                    cancel_check=cancel_check,
                    ffmpeg_dir=cfg.ffmpeg_dir or None,
                )
                _emit(1.0, "extract")
                completed_fraction += phase_weights["extract"]
                _raise_if_cancelled()

                fresh = transcribe(
                    wav_path=wav_path,
                    model_size=cfg.model_size,
                    device=cfg.device,
                    engine=cfg.engine,
                    model_path=cfg.model_path,
                    total_duration_s=duration_s,
                    log=log,
                    progress=lambda elapsed, total: _emit(elapsed / total if total else 0.0, "transcribe"),
                    cancel_check=cancel_check,
                    ffmpeg_dir=cfg.ffmpeg_dir or None,
                )
                _emit(1.0, "transcribe")
                completed_fraction += phase_weights["transcribe"]
                _raise_if_cancelled()
            finally:
                cleanup_temp_file(wav_path)

            transcription_words = fresh.words
            transcription_language = fresh.detected_language

            # Persist the sidecar for future runs.
            if cfg.use_transcript_cache:
                try:
                    cache_identifier = cfg.model_path.strip() or cfg.model_size
                    written = save_cache(
                        cfg.input_video,
                        cache_identifier,
                        words=fresh.words,
                        detected_language=fresh.detected_language,
                        duration_s=duration_s,
                        engine=cfg.engine,
                    )
                    log(f"Saved transcript cache: {written}")
                except OSError as exc:
                    log(f"Warning: could not save transcript cache: {exc}")

        # -- Phase: match / build intervals ----------------------------------
        log("Matching transcribed words against censor list...")
        intervals = build_mute_intervals(
            transcription_words,
            matcher,
            pre_pad_s=max(0, cfg.pre_pad_ms) / 1000.0,
            post_pad_s=max(0, cfg.post_pad_ms) / 1000.0,
            total_duration_s=duration_s,
        )
        matched_count, per_entry = _count_matches_detailed(transcription_words, matcher)
        log(
            f"Matched {matched_count} word occurrence(s), producing "
            f"{len(intervals)} mute interval(s) after padding/merging."
        )
        if per_entry:
            log("Censored entries (occurrences):")
            for entry_text, count in per_entry:
                log(f"  {count:>4} x  {entry_text}")
        _emit(1.0, "match")
        completed_fraction += phase_weights["match"]
        _raise_if_cancelled()

        # -- Phase: encode ---------------------------------------------------
        if not intervals:
            log("No censored intervals produced; copying audio through unchanged.")
        job = MuteJobConfig(
            input_video=cfg.input_video,
            output_video=cfg.output_video,
            intervals=intervals,
            mode=cfg.mode,
            beep_freq_hz=cfg.beep_freq_hz,
            audio_delay_ms=cfg.audio_delay_ms,
            output_mode=cfg.output_mode,
        )
        run_censor_encode(
            job,
            total_duration_s=duration_s,
            log=log,
            progress=lambda elapsed, total: _emit(elapsed / total if total else 0.0, "encode"),
            cancel_check=cancel_check,
            ffmpeg_dir=cfg.ffmpeg_dir or None,
        )
        _emit(1.0, "encode")
        completed_fraction = 1.0

        log(f"Done. Wrote censored video: {cfg.output_video}")
        if progress_cb:
            progress_cb(1.0, "done", 0.0)

        return PipelineResult(
            output_video=cfg.output_video,
            intervals=intervals,
            words_transcribed=len(transcription_words),
            words_matched=matched_count,
            duration_s=duration_s,
            censor_words=words,
            used_cached_transcript=used_cached_transcript,
        )
    except _CancelledError:
        log("Pipeline cancelled by user.")
        raise RuntimeError("Cancelled")
    except RuntimeError as exc:
        # censor_transcribe raises RuntimeError("Cancelled during transcription.")
        # when the tqdm-hook cancellation path fires. Normalize the log message
        # but preserve the wording so the GUI's "Cancelled..." detection works.
        if str(exc).lower().startswith("cancelled"):
            log("Pipeline cancelled by user (during transcription).")
        raise


def _count_matches(words: Sequence[WordSpan], matcher) -> int:
    """Count how many transcribed words end up muted, including tokens that
    only participate in a multi-word phrase match."""
    if hasattr(matcher, "find_matches"):
        try:
            spans = matcher.find_matches([w.text for w in words])
            return sum(e - s for s, e in spans)
        except Exception:  # noqa: BLE001
            pass
    return sum(1 for w in words if matcher.matches(w.text))


def _count_matches_detailed(
    words: Sequence[WordSpan], matcher
) -> tuple[int, List[Tuple[str, int]]]:
    """Return ``(total_words_muted, [(entry_text, occurrences), ...])``.

    The per-entry list is sorted by descending occurrence count, then
    alphabetically. Falls back to the coarse counter for legacy matchers
    that don't implement ``find_matches_detailed``.
    """
    if not hasattr(matcher, "find_matches_detailed"):
        return _count_matches(words, matcher), []
    try:
        spans = matcher.find_matches_detailed([w.text for w in words])
    except Exception:  # noqa: BLE001
        return _count_matches(words, matcher), []
    total = sum(e - s for s, e, _ in spans)
    counts: dict[str, int] = {}
    for _s, _e, entry_text in spans:
        counts[entry_text] = counts.get(entry_text, 0) + 1
    per_entry = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0].lower()))
    return total, per_entry
