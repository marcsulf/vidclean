"""FFmpeg / FFprobe helpers for the video censor tool.

Responsibilities:

* Probe the input video for total duration (used for ETA and clamping).
* Extract a mono 16 kHz WAV file from the input for Whisper.
* Build and run the final ffmpeg command that mutes or beeps out a list of
  time intervals in the output video (video stream copied, audio re-encoded).

Progress from ffmpeg is parsed from its ``-progress pipe:2`` output so the
GUI can display a meaningful percentage and ETA.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple


LogCB = Callable[[str], None]
ProgressCB = Callable[[float, float], None]  # (elapsed_s, total_s)


class FFmpegError(RuntimeError):
    pass


def _resolve_binary(name: str, ffmpeg_dir: str | Path | None = None) -> str:
    """Resolve an ffmpeg-suite binary.

    If ``ffmpeg_dir`` is provided, look for the binary there first (both as
    a directory containing the exe, and as a direct path to the exe itself).
    Otherwise fall back to :func:`shutil.which` against ``PATH``.
    """
    exe_ext = ".exe" if sys.platform.startswith("win") else ""
    if ffmpeg_dir:
        p = Path(str(ffmpeg_dir).strip().strip('"'))
        # Directory case: expect "<dir>/<name>[.exe]" or a nested "bin/" subdir.
        if p.is_dir():
            for candidate in (p / f"{name}{exe_ext}", p / "bin" / f"{name}{exe_ext}"):
                if candidate.exists():
                    return str(candidate)
        # File case: user pointed directly at an executable that happens to
        # be the requested binary (only useful when name matches its stem).
        elif p.is_file():
            if p.stem.lower() == name.lower():
                return str(p)
            # If they gave a path to e.g. ffmpeg.exe, still try to find the
            # sibling ffprobe.exe.
            sibling = p.with_name(f"{name}{exe_ext}")
            if sibling.exists():
                return str(sibling)

    exe = shutil.which(name)
    if exe:
        return exe

    hint = f" or in '{ffmpeg_dir}'" if ffmpeg_dir else ""
    raise FFmpegError(
        f"'{name}' was not found on PATH{hint}. Install ffmpeg and either "
        f"add its folder to PATH, or set the FFmpeg folder in the GUI."
    )


def probe_duration_s(video_path: str | Path, ffmpeg_dir: str | Path | None = None) -> float:
    """Return the media duration of ``video_path`` in seconds."""
    ffprobe = _resolve_binary("ffprobe", ffmpeg_dir)
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(video_path),
    ]
    try:
        raw = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as exc:
        raise FFmpegError(
            f"ffprobe failed for '{video_path}': {exc.output.decode(errors='replace')}"
        ) from exc
    data = json.loads(raw.decode("utf-8", errors="replace"))
    dur = float(data.get("format", {}).get("duration", 0.0))
    if dur <= 0:
        raise FFmpegError(f"ffprobe reported non-positive duration for '{video_path}'.")
    return dur


def extract_audio_wav(
    video_path: str | Path,
    wav_path: str | Path,
    total_duration_s: float | None = None,
    log: Optional[LogCB] = None,
    progress: Optional[ProgressCB] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
    ffmpeg_dir: str | Path | None = None,
) -> None:
    """Extract a mono 16 kHz 16-bit WAV suitable for Whisper."""
    ffmpeg = _resolve_binary("ffmpeg", ffmpeg_dir)
    wav_path = str(wav_path)
    Path(wav_path).parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        "-progress",
        "pipe:1",
        wav_path,
    ]
    if log:
        log("Extracting audio (mono 16 kHz WAV) with ffmpeg...")
    _run_ffmpeg_with_progress(cmd, total_duration_s, log, progress, cancel_check)


def _run_ffmpeg_with_progress(
    cmd: Sequence[str],
    total_duration_s: float | None,
    log: Optional[LogCB],
    progress: Optional[ProgressCB],
    cancel_check: Optional[Callable[[], bool]],
) -> None:
    """Run an ffmpeg command, parsing ``key=value`` progress lines from stdout."""
    # Windows: hide console window when launched from GUI.
    creationflags = 0
    if sys.platform.startswith("win"):
        creationflags = 0x08000000  # CREATE_NO_WINDOW

    proc = subprocess.Popen(
        list(cmd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        creationflags=creationflags,
    )
    stderr_lines: List[str] = []

    def _drain_stderr() -> None:
        assert proc.stderr is not None
        for line in proc.stderr:
            stderr_lines.append(line.rstrip())
            if log:
                stripped = line.strip()
                if stripped:
                    log(f"[ffmpeg] {stripped}")

    stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
    stderr_thread.start()

    try:
        assert proc.stdout is not None
        for raw in proc.stdout:
            if cancel_check and cancel_check():
                proc.terminate()
                raise FFmpegError("Cancelled during ffmpeg processing.")
            line = raw.strip()
            if not line or "=" not in line:
                continue
            key, _, value = line.partition("=")
            if key == "out_time_ms" and progress and total_duration_s:
                try:
                    elapsed = float(value) / 1_000_000.0
                except ValueError:
                    continue
                if elapsed < 0:
                    elapsed = 0.0
                if elapsed > total_duration_s:
                    elapsed = total_duration_s
                progress(elapsed, total_duration_s)
            elif key == "progress" and value == "end" and progress and total_duration_s:
                progress(total_duration_s, total_duration_s)
    finally:
        proc.wait()
        stderr_thread.join(timeout=1.0)

    if proc.returncode != 0:
        tail = "\n".join(stderr_lines[-20:])
        raise FFmpegError(
            f"ffmpeg exited with code {proc.returncode}. Last output:\n{tail}"
        )


def _audio_delay_filter(audio_delay_ms: int) -> str:
    """Return an ffmpeg audio filter that shifts audio in time.

    * Positive ``audio_delay_ms``: delay audio (pad silence at the start) via
      ``adelay=N:all=1``. Use this when the audio track plays *ahead* of the
      video and needs to be pushed later.
    * Negative ``audio_delay_ms``: advance audio (trim the first |N| ms) via
      ``atrim=start=X,asetpts=PTS-STARTPTS``. Use when the audio plays behind
      the video.
    * Zero: returns an empty string (caller should omit the step).
    """
    n = int(audio_delay_ms)
    if n == 0:
        return ""
    if n > 0:
        return f"adelay={n}:all=1"
    start_s = abs(n) / 1000.0
    return f"atrim=start={start_s:.3f},asetpts=PTS-STARTPTS"


def build_mute_filter(
    intervals: Iterable[Tuple[float, float]],
    audio_delay_ms: int = 0,
) -> str:
    """Build an ffmpeg ``-af`` filter string that silences the given intervals.

    Each interval becomes one ``volume=enable='between(t,s,e)':volume=0`` clause,
    chained with commas. When ``audio_delay_ms`` is non-zero, the shift is
    applied *after* the mute clauses so the interval timings still reference
    the original audio timeline.
    """
    parts: List[str] = []
    for s, e in intervals:
        if e <= s:
            continue
        parts.append(f"volume=enable='between(t,{s:.3f},{e:.3f})':volume=0")
    delay_step = _audio_delay_filter(audio_delay_ms)
    if not parts:
        # No mute clauses. Still force re-encoding; include the delay if any.
        return delay_step or "anull"
    chain = ",".join(parts)
    if delay_step:
        chain = f"{chain},{delay_step}"
    return chain


def build_beep_filter_complex(
    intervals: Iterable[Tuple[float, float]],
    beep_freq_hz: float = 1000.0,
    audio_delay_ms: int = 0,
) -> Tuple[str, List[str]]:
    """Build ``-filter_complex`` arguments that beep-out the given intervals.

    Strategy:
      1. Mute the original audio at each interval (volume=0 between t=s,e).
      2. Generate a single sine tone stream and gate it to only the intervals.
      3. Mix the muted original and the gated tone.

    Returns a tuple ``(filter_complex_string, extra_input_args)``. ``extra_input_args``
    is currently empty (we synthesize the tone in-filter via ``sine`` source).
    """
    intervals = [(s, e) for s, e in intervals if e > s]
    delay_step = _audio_delay_filter(audio_delay_ms)
    if not intervals:
        base = "[0:a]anull"
        if delay_step:
            return f"{base},{delay_step}[aout]", []
        return f"{base}[aout]", []

    mute_clauses = ",".join(
        f"volume=enable='between(t,{s:.3f},{e:.3f})':volume=0" for s, e in intervals
    )
    beep_gate_clauses = ",".join(
        # Invert the gate: keep tone only inside intervals, mute elsewhere.
        f"volume=enable='between(t,{s:.3f},{e:.3f})':volume=1"
        for s, e in intervals
    )
    # We need the tone to be silent outside intervals. Start silent and gate
    # up inside intervals by multiplying with a switch built from between().
    # Simpler: use aevalsrc with a between() expression as amplitude.
    conds = "+".join(f"between(t,{s:.3f},{e:.3f})" for s, e in intervals)
    tone_expr = f"aevalsrc=exprs='0.4*sin(2*PI*{beep_freq_hz:.1f}*t)*({conds})':s=16000:c=mono"

    if delay_step:
        filter_complex = (
            f"[0:a]{mute_clauses}[muted];"
            f"{tone_expr}[tone];"
            f"[muted][tone]amix=inputs=2:duration=first:normalize=0[mixed];"
            f"[mixed]{delay_step}[aout]"
        )
    else:
        filter_complex = (
            f"[0:a]{mute_clauses}[muted];"
            f"{tone_expr}[tone];"
            f"[muted][tone]amix=inputs=2:duration=first:normalize=0[aout]"
        )
    # ``beep_gate_clauses`` is retained above as documentation of the intent
    # even though we ultimately use aevalsrc; silence the linter:
    _ = beep_gate_clauses
    return filter_complex, []


# Output modes for the censor encode step.
OUTPUT_MODE_REPLACE = "replace"       # replace the original audio (single audio track)
OUTPUT_MODE_ADD_TRACK = "add_track"   # keep original audio + append censored track

VALID_OUTPUT_MODES = frozenset({OUTPUT_MODE_REPLACE, OUTPUT_MODE_ADD_TRACK})

# Container extensions that do not reliably support multiple audio streams
# (or the AAC codec we produce). When the user requests ``add_track`` output
# and the input has one of these extensions, we switch the output container
# to ``.mp4``. AVI is included because its multi-audio support is limited and
# player compatibility is inconsistent.
_SINGLE_AUDIO_CONTAINERS: frozenset = frozenset({
    ".webm", ".ogv", ".ogg", ".oga", ".flv", ".3gp", ".gif", ".avi",
})


def container_supports_multiple_audio(path: str | Path) -> bool:
    """Return True when the file extension of ``path`` is known to support
    multiple audio streams with AAC. Extensions not in the known-bad set are
    assumed to support it."""
    ext = Path(str(path)).suffix.lower()
    return ext not in _SINGLE_AUDIO_CONTAINERS


def adjust_output_path_for_mode(
    output_video: str | Path,
    output_mode: str,
    fallback_ext: str = ".mp4",
) -> str:
    """If ``output_mode == 'add_track'`` and the requested output container
    can't hold multiple audio streams, swap its extension for ``fallback_ext``.
    Otherwise return ``output_video`` unchanged."""
    mode = (output_mode or OUTPUT_MODE_REPLACE).lower()
    if mode != OUTPUT_MODE_ADD_TRACK:
        return str(output_video)
    p = Path(str(output_video))
    if container_supports_multiple_audio(p):
        return str(p)
    return str(p.with_suffix(fallback_ext))


def probe_audio_stream_count(
    video_path: str | Path,
    ffmpeg_dir: str | Path | None = None,
) -> int:
    """Return the number of audio streams in ``video_path``."""
    ffprobe = _resolve_binary("ffprobe", ffmpeg_dir)
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "a",
        "-show_entries",
        "stream=index",
        "-of",
        "json",
        str(video_path),
    ]
    try:
        raw = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError:
        return 0
    try:
        data = json.loads(raw.decode("utf-8", errors="replace"))
    except json.JSONDecodeError:
        return 0
    streams = data.get("streams") or []
    return len(streams)


@dataclass
class MuteJobConfig:
    input_video: str
    output_video: str
    intervals: List[Tuple[float, float]]
    mode: str = "mute"  # "mute" or "beep"
    beep_freq_hz: float = 1000.0
    audio_delay_ms: int = 0  # positive = delay audio; negative = advance audio
    output_mode: str = OUTPUT_MODE_REPLACE  # "replace" | "add_track"


def build_censor_ffmpeg_args(
    config: MuteJobConfig,
    ffmpeg_exe: str,
    source_audio_count: int,
    out_path: str,
) -> List[str]:
    """Return the full argv for the censor-encode ffmpeg invocation.

    Split out from :func:`run_censor_encode` so it can be unit-tested without
    launching ffmpeg. ``source_audio_count`` is the number of audio streams in
    the input video (used only for ``add_track`` mode to place per-stream
    codec/metadata options on the newly appended track).
    """
    mode = (config.mode or "mute").lower()
    if mode not in {"mute", "beep"}:
        raise ValueError(f"Unknown censor mode: {config.mode!r}")

    output_mode = (config.output_mode or OUTPUT_MODE_REPLACE).lower()
    if output_mode not in VALID_OUTPUT_MODES:
        raise ValueError(f"Unknown output_mode: {config.output_mode!r}")

    cmd: List[str] = [
        ffmpeg_exe,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-i",
        str(config.input_video),
    ]

    if output_mode == OUTPUT_MODE_REPLACE:
        if mode == "mute":
            af = build_mute_filter(config.intervals, audio_delay_ms=config.audio_delay_ms)
            cmd += [
                "-map", "0:v:0",
                "-map", "0:a:0?",
                "-c:v", "copy",
                "-af", af,
                "-c:a", "aac",
                "-b:a", "192k",
            ]
        else:  # beep
            filter_complex, extra_inputs = build_beep_filter_complex(
                config.intervals,
                beep_freq_hz=config.beep_freq_hz,
                audio_delay_ms=config.audio_delay_ms,
            )
            cmd += extra_inputs
            cmd += [
                "-filter_complex", filter_complex,
                "-map", "0:v:0",
                "-map", "[aout]",
                "-c:v", "copy",
                "-c:a", "aac",
                "-b:a", "192k",
            ]
    else:  # add_track: keep original audio + append censored track
        # Index of the new censored track within the output's audio streams.
        new_idx = max(0, int(source_audio_count))

        if mode == "mute":
            af = build_mute_filter(config.intervals, audio_delay_ms=config.audio_delay_ms)
            filter_complex = f"[0:a:0]{af}[cens]"
            extra_inputs: List[str] = []
        else:  # beep
            beep_expr, extra_inputs = build_beep_filter_complex(
                config.intervals,
                beep_freq_hz=config.beep_freq_hz,
                audio_delay_ms=config.audio_delay_ms,
            )
            # The beep builder already labels its output as [aout]; rename to [cens].
            filter_complex = beep_expr.replace("[aout]", "[cens]")

        cmd += extra_inputs
        cmd += [
            "-filter_complex", filter_complex,
            "-map", "0:v",
            "-map", "0:a?",
            "-map", "[cens]",
            "-c:v", "copy",
            "-c:a", "copy",
            f"-c:a:{new_idx}", "aac",
            f"-b:a:{new_idx}", "192k",
            f"-metadata:s:a:{new_idx}", "title=Censored",
            f"-metadata:s:a:{new_idx}", "language=eng",
            f"-disposition:a:{new_idx}", "0",
        ]

    cmd += ["-progress", "pipe:1", out_path]
    return cmd


def run_censor_encode(
    config: MuteJobConfig,
    total_duration_s: float,
    log: Optional[LogCB] = None,
    progress: Optional[ProgressCB] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
    ffmpeg_dir: str | Path | None = None,
) -> None:
    """Run the final ffmpeg command that produces the censored output video."""
    ffmpeg = _resolve_binary("ffmpeg", ffmpeg_dir)
    in_path = str(config.input_video)
    out_path = str(config.output_video)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    output_mode = (config.output_mode or OUTPUT_MODE_REPLACE).lower()
    if output_mode == OUTPUT_MODE_ADD_TRACK:
        source_audio_count = probe_audio_stream_count(in_path, ffmpeg_dir=ffmpeg_dir)
    else:
        source_audio_count = 0

    cmd = build_censor_ffmpeg_args(
        config,
        ffmpeg_exe=ffmpeg,
        source_audio_count=source_audio_count,
        out_path=out_path,
    )

    if log:
        n = len(config.intervals)
        delay_note = (
            f", audio_delay_ms={config.audio_delay_ms}"
            if config.audio_delay_ms
            else ""
        )
        mode = (config.mode or "mute").lower()
        if output_mode == OUTPUT_MODE_ADD_TRACK:
            log(
                f"Encoding output with {n} censored interval(s), mode={mode}, "
                f"output_mode=add_track (source audio streams: {source_audio_count}, "
                f"new track index: {source_audio_count}){delay_note}..."
            )
        else:
            log(
                f"Encoding output with {n} censored interval(s), "
                f"mode={mode}, output_mode=replace{delay_note}..."
            )
    _run_ffmpeg_with_progress(cmd, total_duration_s, log, progress, cancel_check)


def safe_default_output_path(input_video: str | Path) -> str:
    """Return ``<name>_censored.<ext>`` alongside the input file."""
    p = Path(input_video)
    return str(p.with_name(f"{p.stem}_censored{p.suffix}"))


def make_temp_wav_path(input_video: str | Path) -> str:
    """Return a scratch WAV path alongside the input video."""
    p = Path(input_video)
    return str(p.with_name(f".{p.stem}_censor_tmp.wav"))


def cleanup_temp_file(path: str | Path) -> None:
    try:
        os.remove(path)
    except OSError:
        pass
