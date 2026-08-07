"""Transcript caching for the video censor tool.

Whisper transcription is the slowest step in the pipeline. When the user
re-runs against the same input video (e.g. after editing the censor word
list), we can skip transcription entirely by reading a JSON sidecar
written the first time around.

Cache layout
------------
A JSON file next to the input video, named ``<input>.transcript.json``, with:

    {
        "cache_version": 1,
        "engine": "openai-whisper",
        "model_size": "medium",
        "language": "en",
        "duration_s": 917.88,
        "input_path": "/path/to/video/test.mp4",
        "input_size_bytes": 123456789,
        "input_mtime": 1720368000.123,
        "words": [
            {"text": "hello", "start": 0.12, "end": 0.44},
            ...
        ]
    }

Cache validity
--------------
The cache is used only when *all* of the following match the current input:

* file size in bytes
* file mtime (nanosecond precision on modern filesystems)
* model_size
* engine (missing engine field is treated as ``"openai-whisper"`` for
  backward compatibility with caches written before engine selection was
  added)
* **server_url** and **remote_model_version** (only when engine is
  ``"remote"`` — prevents mixing results from different servers or model
  versions)

Any mismatch causes a fresh transcription.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from censor_timestamps import WordSpan


CACHE_VERSION = 1


def cache_path_for(input_video: str | Path) -> Path:
    """Return the sidecar transcript path for ``input_video``."""
    p = Path(input_video)
    return p.with_suffix(p.suffix + ".transcript.json")


def _stat_tuple(input_video: str | Path) -> tuple[int, float]:
    st = os.stat(str(input_video))
    return int(st.st_size), float(st.st_mtime)


@dataclass
class CachedTranscript:
    words: list
    detected_language: str
    duration_s: float


def load_cache(
    input_video: str | Path,
    model_size: str,
    engine: str = "openai-whisper",
    *,
    server_url: str = "",
    remote_model_version: str = "",
) -> Optional[CachedTranscript]:
    """Return a cached transcript if one exists and is still valid.

    Returns ``None`` if the cache file is missing, malformed, or stale.
    A cache is considered stale if the recorded ``engine`` differs from
    the requested one; caches predating engine selection are treated as
    ``"openai-whisper"``.

    When ``engine == "remote"``, the cached ``server_url`` and
    ``remote_model_version`` must also match (passed as keyword-only args).
    """
    path = cache_path_for(input_video)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if int(data.get("cache_version", 0)) != CACHE_VERSION:
        return None
    if str(data.get("model_size", "")) != str(model_size):
        return None
    cached_engine = str(data.get("engine", "openai-whisper"))
    if cached_engine != str(engine):
        return None
    # Remote-engine cache must also match server URL and model version.
    if engine == "remote":
        if str(data.get("server_url", "")) != str(server_url):
            return None
        if str(data.get("remote_model_version", "")) != str(remote_model_version):
            return None
    try:
        size, mtime = _stat_tuple(input_video)
    except OSError:
        return None
    if int(data.get("input_size_bytes", -1)) != size:
        return None
    # Allow ~1 ms mtime slop for filesystems with coarser resolution.
    if abs(float(data.get("input_mtime", -1.0)) - mtime) > 0.001:
        return None
    try:
        words = [
            WordSpan(
                text=str(w["text"]),
                start=float(w["start"]),
                end=float(w["end"]),
            )
            for w in data.get("words", [])
        ]
    except (KeyError, TypeError, ValueError):
        return None
    return CachedTranscript(
        words=words,
        detected_language=str(data.get("language", "en")),
        duration_s=float(data.get("duration_s", 0.0)),
    )


def save_cache(
    input_video: str | Path,
    model_size: str,
    words: list,
    detected_language: str,
    duration_s: float,
    engine: str = "openai-whisper",
    *,
    server_url: str = "",
    remote_model_version: str = "",
) -> Path:
    """Write a transcript sidecar for ``input_video``. Returns the path.

    When ``engine == "remote"``, pass ``server_url`` and
    ``remote_model_version`` so future cache hits can verify they match
    the same server and model version.
    """
    path = cache_path_for(input_video)
    try:
        size, mtime = _stat_tuple(input_video)
    except OSError:
        # If we cannot stat the source, we cannot make a valid cache.
        raise
    payload: dict = {
        "cache_version": CACHE_VERSION,
        "engine": str(engine),
        "model_size": str(model_size),
        "language": str(detected_language),
        "duration_s": float(duration_s),
        "input_path": str(Path(input_video)),
        "input_size_bytes": int(size),
        "input_mtime": float(mtime),
        "words": [
            {"text": w.text, "start": float(w.start), "end": float(w.end)}
            for w in words
        ],
    }
    # Store remote-identity fields when using the remote engine so that
    # caches from different servers or model versions never collide.
    if engine == "remote":
        payload["server_url"] = str(server_url)
        payload["remote_model_version"] = str(remote_model_version)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1, ensure_ascii=False)
    return path


def clear_cache(input_video: str | Path) -> bool:
    """Delete the sidecar transcript for ``input_video`` if it exists.

    Returns True if a file was deleted, False otherwise.
    """
    path = cache_path_for(input_video)
    try:
        path.unlink()
        return True
    except FileNotFoundError:
        return False
    except OSError:
        return False
