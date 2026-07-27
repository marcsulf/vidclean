"""Whisper transcription wrapper for the video censor tool.

Handles device auto-detection (CUDA vs CPU), model loading, transcription
with word-level timestamps, and per-segment progress reporting.

Two backend engines are supported and are selected via the ``engine``
parameter:

* ``"openai-whisper"`` — the reference implementation. Slower but well
  established. Progress is captured by monkey-patching its internal
  ``tqdm`` bar.
* ``"faster-whisper"`` — a CTranslate2 reimplementation, roughly 4-8x
  faster on the same hardware. It exposes a streaming segment generator
  which we iterate ourselves; progress comes from the segment ``end``
  time versus the total audio duration, so it updates smoothly.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

from censor_timestamps import WordSpan, flatten_transcript_words


ProgressCB = Callable[[float, float], None]
"""Signature: (elapsed_audio_seconds, total_audio_seconds) -> None."""

LogCB = Callable[[str], None]


AVAILABLE_MODELS = (
    "tiny",
    "tiny.en",
    "base",
    "base.en",
    "small",
    "small.en",
    "medium",
    "medium.en",
    "large-v3",
)

# Canonical engine identifiers, stored in settings and the transcript cache.
AVAILABLE_ENGINES = ("openai-whisper", "faster-whisper")


def _normalize_engine(engine: str | None) -> str:
    """Return a canonical engine name, defaulting to ``openai-whisper``."""
    if not engine:
        return "openai-whisper"
    e = str(engine).strip().lower().replace("_", "-").replace(" ", "-")
    if e in {"openai-whisper", "openai", "whisper", "openaiwhisper"}:
        return "openai-whisper"
    if e in {"faster-whisper", "faster", "fasterwhisper"}:
        return "faster-whisper"
    return e  # unknown; will error later


# Simple single-slot model cache. Whisper models are large (medium is ~1.4 GB
# on CPU, ~3 GB on GPU), so we keep at most one loaded at a time to avoid
# leaking VRAM/RAM when the user changes engine or model size between
# batches. Keyed by (engine, model_size, resolved_device). Access is not
# protected by a lock because the pipeline is single-threaded from the
# caller's perspective — one batch worker at a time drives all transcribe()
# calls.
_MODEL_CACHE: dict[tuple[str, str, str], object] = {}


def clear_model_cache() -> None:
    """Drop any cached Whisper model. Frees VRAM/RAM."""
    _MODEL_CACHE.clear()
    try:
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # pragma: no cover
        pass


def _looks_like_ssl_error(exc: BaseException) -> bool:
    """Return True if ``exc`` (or any cause in its chain) looks like an SSL
    / TLS certificate verification failure. Corporate MITM proxies commonly
    inject self-signed certificates that break `huggingface_hub`'s HTTPS
    downloads.
    """
    seen: set[int] = set()
    e: BaseException | None = exc
    while e is not None and id(e) not in seen:
        seen.add(id(e))
        msg = str(e).lower()
        if (
            "certificate verify failed" in msg
            or "self-signed certificate" in msg
            or "self signed certificate" in msg
            or "ssl:" in msg
            or "sslerror" in msg
            or "sslcertverificationerror" in msg
        ):
            return True
        e = e.__cause__ or e.__context__
    return False


# Guards against installing the unverified-TLS override more than once
# per process.
_HF_SSL_BACKEND_DISABLED = False


def _disable_hf_ssl_verification(log: Optional[LogCB] = None) -> bool:
    """Disable TLS certificate verification for HTTPS calls made by
    ``huggingface_hub`` (and best-effort for other clients) in the current
    process. Returns True if at least one override took effect.

    ``huggingface_hub``'s API surface has moved around across versions and
    the version installed here (1.x) uses ``httpx`` \u2014 not ``requests``.
    We therefore install every override we know about:

    1. **`huggingface_hub.set_client_factory` / `set_async_client_factory`**
       (huggingface_hub 1.x, httpx-based) \u2014 factories that return
       ``httpx.Client(verify=False, ...)`` matching the library's own
       defaults (event hooks, ``follow_redirects=True``, ``timeout=None``).
    2. **`huggingface_hub.configure_http_backend`** (0.x, requests-based)
       \u2014 unverified ``requests.Session`` factory. Harmless when the
       function is absent.\n    3. Patch ``requests.Session.__init__`` so every ``requests`` session\n       is created with ``verify=False``. Cheap belt-and-suspenders for\n       anything else in the process that talks HTTPS via ``requests``.\n    4. Set ``os.environ['CURL_CA_BUNDLE'] = ''`` and\n       ``HF_HUB_DISABLE_SSL_VERIFY`` (some libraries honor these hints).\n    5. Silence ``urllib3``'s ``InsecureRequestWarning`` and ``httpx``'s\n       equivalent so the retry log stays readable.\n    """
    global _HF_SSL_BACKEND_DISABLED
    if _HF_SSL_BACKEND_DISABLED:
        return True

    patched_any = False

    # ---- 1 & 2: huggingface_hub factories --------------------------------
    try:
        import huggingface_hub  # type: ignore[import-not-found]

        # Newer hf_hub (1.x, httpx-based).
        set_factory = getattr(huggingface_hub, "set_client_factory", None)
        set_async_factory = getattr(huggingface_hub, "set_async_client_factory", None)
        if callable(set_factory):
            try:
                import httpx  # type: ignore[import-not-found]

                # Import the library's default event hooks so we behave like
                # ``default_client_factory`` \u2014 minus the TLS check.
                try:
                    from huggingface_hub.utils._http import (  # type: ignore[import-not-found]
                        hf_request_event_hook,
                    )

                    request_hooks = [hf_request_event_hook]
                except Exception:  # pragma: no cover
                    request_hooks = []

                def _client_factory():
                    return httpx.Client(
                        event_hooks={"request": request_hooks},
                        follow_redirects=True,
                        timeout=None,
                        verify=False,
                    )

                set_factory(_client_factory)
                patched_any = True

                if callable(set_async_factory):
                    try:
                        from huggingface_hub.utils._http import (  # type: ignore[import-not-found]
                            async_hf_request_event_hook,
                            async_hf_response_event_hook,
                        )

                        async_req_hooks = [async_hf_request_event_hook]
                        async_res_hooks = [async_hf_response_event_hook]
                    except Exception:  # pragma: no cover
                        async_req_hooks = []
                        async_res_hooks = []

                    def _async_client_factory():
                        return httpx.AsyncClient(
                            event_hooks={
                                "request": async_req_hooks,
                                "response": async_res_hooks,
                            },
                            follow_redirects=True,
                            timeout=None,
                            verify=False,
                        )

                    try:
                        set_async_factory(_async_client_factory)
                    except Exception:  # pragma: no cover
                        pass
            except Exception:  # pragma: no cover
                pass

        # Older hf_hub (0.x, requests-based).
        configure = getattr(huggingface_hub, "configure_http_backend", None)
        if callable(configure):
            def _requests_factory():
                import requests as _rq  # picks up the patched Session below

                return _rq.Session()

            try:
                configure(backend_factory=_requests_factory)
                patched_any = True
            except Exception:  # pragma: no cover
                pass
    except Exception:  # pragma: no cover
        pass

    # ---- 3: requests.Session ---------------------------------------------
    try:
        import requests.sessions as _rq_sessions  # type: ignore[import-not-found]

        _orig_session_init = _rq_sessions.Session.__init__

        def _patched_session_init(self, *args, **kwargs):
            _orig_session_init(self, *args, **kwargs)
            self.verify = False

        _rq_sessions.Session.__init__ = _patched_session_init  # type: ignore[assignment]
        patched_any = True
    except Exception:  # pragma: no cover
        pass

    # ---- 4: environment hints --------------------------------------------
    try:
        os.environ.setdefault("HF_HUB_DISABLE_SSL_VERIFY", "1")
        os.environ.setdefault("CURL_CA_BUNDLE", "")
    except Exception:  # pragma: no cover
        pass

    # ---- 5: warning silencing --------------------------------------------
    try:
        import urllib3  # type: ignore[import-not-found]

        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover
        pass
    warnings.filterwarnings(
        "ignore",
        message=r".*Unverified HTTPS request.*",
    )

    if patched_any:
        _HF_SSL_BACKEND_DISABLED = True
        if log:
            log(
                "Disabled TLS verification for huggingface_hub for this session "
                "(corporate SSL workaround)."
            )
    elif log:  # pragma: no cover
        log("Could not disable TLS verification: huggingface_hub not reachable.")
    return patched_any


def _looks_like_connection_reset(exc: BaseException) -> bool:
    """Return True if ``exc`` looks like a corporate proxy actively
    refusing / closing the connection (rather than a TLS failure).

    Examples: ``WinError 10054`` ("An existing connection was forcibly
    closed by the remote host"), ``ConnectionResetError``,
    ``httpx.ConnectError`` with 'Connection refused', 'Failed to establish
    a new connection', etc.
    """
    seen: set[int] = set()
    e: BaseException | None = exc
    while e is not None and id(e) not in seen:
        seen.add(id(e))
        msg = str(e).lower()
        if (
            "10054" in msg
            or "connection reset" in msg
            or "connection was forcibly closed" in msg
            or "connection refused" in msg
            or "connection aborted" in msg
            or "failed to establish a new connection" in msg
            or "remote end closed connection" in msg
            or "connectionreseterror" in msg
        ):
            return True
        e = e.__cause__ or e.__context__
    return False


def _load_faster_whisper_with_ssl_fallback(
    WhisperModel,
    model_id: str,
    resolved_device: str,
    compute_type: str,
    log: Optional[LogCB] = None,
):
    """Load a faster-whisper ``WhisperModel``, retrying on SSL cert errors.

    ``model_id`` is passed straight through as the first positional arg to
    ``WhisperModel(...)``. It may be a well-known size (``"medium"``), an
    HF repo id (``"Systran/faster-whisper-medium"``), or a local
    directory containing the CT2 model files.

    huggingface_hub raises ``ConnectError`` / ``LocalEntryNotFoundError`` on
    the first attempt when a corporate proxy's certificate isn't trusted.
    We install a session factory that disables verification and try once
    more; if the second attempt still fails, the exception propagates.
    """
    try:
        return WhisperModel(model_id, device=resolved_device, compute_type=compute_type)
    except BaseException as exc:  # noqa: BLE001
        if not _looks_like_ssl_error(exc):
            raise
        if log:
            log(
                "faster-whisper model download failed with an SSL certificate "
                "verification error. Retrying with certificate verification "
                "DISABLED (corporate SSL proxy workaround). This is safe if "
                "you trust the network you are on."
            )
        if not _disable_hf_ssl_verification(log=log):
            raise
        # Retry once with the unverified session factory now in place.
        return WhisperModel(model_id, device=resolved_device, compute_type=compute_type)


def _get_or_load_model(
    engine: str,
    model_size: str,
    resolved_device: str,
    log: Optional[LogCB] = None,
    model_path: str = "",
):
    """Return a cached transcription model, loading it (once) if necessary.

    Only one model is kept cached at a time. If the requested
    ``(engine, identifier, device)`` differs from the cached entry, the
    old one is dropped first so we don't hold two large models in memory
    simultaneously.

    ``model_path`` (optional) overrides ``model_size`` as the identifier
    passed to the backend, and is used verbatim. For faster-whisper it may
    be a local directory (containing ``model.bin``, ``config.json``, etc.)
    or an HF repo id (e.g. ``Systran/faster-whisper-medium``). For
    openai-whisper it may be a path to a downloaded ``.pt`` checkpoint.
    """
    engine = _normalize_engine(engine)
    identifier = (model_path or "").strip() or model_size
    key = (engine, identifier, resolved_device)
    cached = _MODEL_CACHE.get(key)
    if cached is not None:
        if log:
            log(
                f"Reusing already-loaded {engine} model '{identifier}' on {resolved_device}."
            )
        return cached

    # Different key requested; free any previously-cached model first.
    if _MODEL_CACHE:
        old_key = next(iter(_MODEL_CACHE))
        if log:
            log(
                f"Transcription model changed from {old_key} to {key}; unloading old model."
            )
        clear_model_cache()

    if engine == "openai-whisper":
        try:
            import whisper
        except ImportError as exc:
            raise RuntimeError(
                "openai-whisper is required for this engine. Install with:  pip install openai-whisper"
            ) from exc
        if log:
            log(f"Loading openai-whisper model '{identifier}' on {resolved_device}...")
        model = whisper.load_model(identifier, device=resolved_device)
    elif engine == "faster-whisper":
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "faster-whisper is required for this engine. Install with:  pip install faster-whisper"
            ) from exc
        # Sensible compute_type defaults per device. faster-whisper (CTranslate2)
        # will raise if the chosen type isn't supported on the runtime, so fall
        # back to ``default`` (float32) in that case.
        compute_type = "float16" if resolved_device == "cuda" else "int8"
        if log:
            log(
                f"Loading faster-whisper model '{identifier}' on {resolved_device} "
                f"(compute_type={compute_type})..."
            )

        def _load(ct: str):
            return _load_faster_whisper_with_ssl_fallback(
                WhisperModel, identifier, resolved_device, ct, log=log
            )

        try:
            try:
                model = _load(compute_type)
            except (ValueError, RuntimeError) as exc:
                # Distinguish "compute_type not supported" (retry with default)
                # from the SSL/connection fallback path (already exhausted).
                if _looks_like_ssl_error(exc) or _looks_like_connection_reset(exc):
                    raise
                if log:
                    log(
                        f"faster-whisper: compute_type={compute_type!r} not supported "
                        f"({exc}); retrying with 'default'."
                    )
                model = _load("default")
        except BaseException as exc:
            # Enrich the message when the network is clearly blocked, so
            # the GUI's error dialog points the user at the workaround.
            if _looks_like_connection_reset(exc) and not (model_path or "").strip():
                hint = (
                    "The connection to huggingface.co was refused / closed "
                    "by the network. This usually means a corporate proxy is "
                    "blocking the download. Workarounds:\n"
                    "  1. Download the model on a network that has access, "
                    "then set the GUI's 'Model path' to the local directory.\n"
                    "  2. Or set the HF_ENDPOINT environment variable to a "
                    "reachable HuggingFace mirror.\n"
                    "  3. Or use the 'OpenAI Whisper' engine, whose model "
                    "files download from a different CDN."
                )
                if log:
                    log(hint)
            raise
    else:
        raise RuntimeError(
            f"Unknown transcription engine: {engine!r}. Expected one of: {AVAILABLE_ENGINES}"
        )

    _MODEL_CACHE[key] = model
    return model


@dataclass
class TranscriptionResult:
    words: List[WordSpan]
    detected_language: str
    duration_s: float


def resolve_device(requested: str = "auto", log: Optional[LogCB] = None) -> str:
    """Return ``"cuda"`` or ``"cpu"`` based on the request and torch state."""
    requested = (requested or "auto").lower()
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required. Install with:  pip install torch"
        ) from exc

    if requested == "cpu":
        if log:
            log("Device: CPU (forced by user).")
        return "cpu"

    cuda_ok = bool(torch.cuda.is_available())
    if requested == "cuda":
        if not cuda_ok:
            raise RuntimeError(
                "CUDA was requested but torch.cuda.is_available() is False. "
                "Install a CUDA build of torch or choose CPU/Auto."
            )
        if log:
            name = torch.cuda.get_device_name(0)
            log(f"Device: CUDA ({name}).")
        return "cuda"

    # Auto
    if cuda_ok:
        if log:
            name = torch.cuda.get_device_name(0)
            log(f"Device: CUDA auto-detected ({name}).")
        return "cuda"
    if log:
        log("Device: CPU (no CUDA GPU detected).")
    return "cpu"


def preload_model(
    model_size: str = "medium",
    device: str = "auto",
    engine: str = "openai-whisper",
    model_path: str = "",
    log: Optional[LogCB] = None,
) -> None:
    """Pre-load the transcription model into the module cache.

    Call this once before a batch so the model-load cost (which can be
    5-20 s for medium and even longer for large) isn't rolled into the
    first file's progress bar / ETA.
    """
    resolved_device = resolve_device(device, log=log)
    _get_or_load_model(engine, model_size, resolved_device, log=log, model_path=model_path)


def transcribe(
    wav_path: str,
    model_size: str = "medium",
    device: str = "auto",
    engine: str = "openai-whisper",
    model_path: str = "",
    total_duration_s: float | None = None,
    log: Optional[LogCB] = None,
    progress: Optional[ProgressCB] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
    ffmpeg_dir: str | None = None,
) -> TranscriptionResult:
    """Transcribe ``wav_path`` and return word-level spans.

    ``engine`` selects the backend: ``"openai-whisper"`` (default) or
    ``"faster-whisper"``. Both engines return the same
    :class:`TranscriptionResult` shape so downstream code doesn't care.

    Progress is emitted after each Whisper segment, based on the segment's
    end time relative to ``total_duration_s`` when available.

    ``ffmpeg_dir``, when provided, is prepended to the ``PATH`` environment
    variable for the duration of the call so that Whisper's internal audio
    decoder can find ``ffmpeg`` even when it is not otherwise on PATH.

    The transcription model is cached across calls (see
    ``_get_or_load_model``), so repeated calls with the same
    ``(engine, model_size, device)`` — as happens across a batch of files —
    pay the model-load cost only once.
    """
    engine = _normalize_engine(engine)

    # Silence the harmless Triton fallback warning that openai-whisper emits
    # on systems without the full CUDA Toolkit / Triton (e.g. all Windows
    # setups using the runtime-only torch wheel). Whisper falls back to a
    # pure-PyTorch median filter on the GPU; transcription accuracy is
    # unchanged and only word-timestamp alignment is a bit slower.
    warnings.filterwarnings(
        "ignore",
        message=r".*Failed to launch Triton kernels.*",
        category=UserWarning,
    )

    resolved_device = resolve_device(device, log=log)
    model = _get_or_load_model(engine, model_size, resolved_device, log=log, model_path=model_path)

    if log:
        log(f"Transcribing audio via {engine} (word-level timestamps enabled)...")

    with _augmented_path_for_ffmpeg(ffmpeg_dir, log=log):
        if engine == "faster-whisper":
            result = _transcribe_faster_whisper(
                model=model,
                wav_path=wav_path,
                total_duration_s=total_duration_s,
                progress=progress,
                cancel_check=cancel_check,
                log=log,
            )
        else:
            # openai-whisper's transcribe() streams segments internally; there is
            # no per-segment callback in the stable API. We rely on verbose=False
            # and post-process the segments list. To provide progress feedback
            # during long runs, we monkey-patch its internal tqdm bar.
            result = _transcribe_with_progress(
                model=model,
                wav_path=wav_path,
                total_duration_s=total_duration_s,
                progress=progress,
                cancel_check=cancel_check,
                log=log,
            )

    segments = result.get("segments") or []
    words = flatten_transcript_words(segments)
    detected_language = str(result.get("language", "en"))

    # Final tick.
    if progress and total_duration_s:
        progress(total_duration_s, total_duration_s)

    if log:
        log(
            f"Transcription complete: {len(words)} words, "
            f"{len(segments)} segments, language={detected_language}."
        )

    return TranscriptionResult(
        words=words,
        detected_language=detected_language,
        duration_s=total_duration_s or (words[-1].end if words else 0.0),
    )


class _augmented_path_for_ffmpeg:
    """Context manager that prepends ``ffmpeg_dir`` to os.environ["PATH"].

    Whisper's ``load_audio`` launches ``ffmpeg`` as a subprocess by name only,
    so ffmpeg must be discoverable on PATH. When the user has pointed us at a
    non-PATH folder via the GUI, we splice it in for the duration of the call.
    """

    def __init__(self, ffmpeg_dir: str | Path | None, log: Optional[LogCB] = None) -> None:
        self._ffmpeg_dir = ffmpeg_dir
        self._log = log
        self._old_path: str | None = None

    def __enter__(self) -> "_augmented_path_for_ffmpeg":
        if not self._ffmpeg_dir:
            return self
        p = Path(str(self._ffmpeg_dir).strip().strip('"'))
        # If they pointed at a file, use its parent directory.
        if p.is_file():
            p = p.parent
        # A folder that is not itself the bin dir but contains one:
        if p.is_dir() and not (p / "ffmpeg.exe").exists() and (p / "bin" / "ffmpeg.exe").exists():
            p = p / "bin"
        if not p.is_dir():
            return self
        current = os.environ.get("PATH", "")
        self._old_path = current
        os.environ["PATH"] = f"{p}{os.pathsep}{current}" if current else str(p)
        if self._log:
            self._log(f"Prepended '{p}' to PATH for Whisper audio decoding.")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._old_path is not None:
            os.environ["PATH"] = self._old_path


class _WhisperCancelled(BaseException):
    """Raised from inside our tqdm proxy to unwind whisper's blocking loop.

    Inherits ``BaseException`` (not ``Exception``) so that whisper's own
    ``try/except Exception`` blocks (if any) can't swallow it.
    """


def _transcribe_faster_whisper(
    model,
    wav_path: str,
    total_duration_s: float | None,
    progress: Optional[ProgressCB],
    cancel_check: Optional[Callable[[], bool]],
    log: Optional[LogCB] = None,
) -> dict:
    """Run faster-whisper's ``model.transcribe`` and stream results back.

    Returns a dict shaped like openai-whisper's return value so the downstream
    :func:`flatten_transcript_words` code works unchanged:
    ``{"segments": [...], "language": str}``. Each segment dict has
    ``"start"``, ``"end"``, ``"text"``, and ``"words"``: a list of
    ``{"word": str, "start": float, "end": float}`` items.

    Progress is emitted once per segment (much more frequent than
    openai-whisper's 30 s chunk cadence), and ``cancel_check`` is polled
    between segments \u2014 no monkey-patching needed since faster-whisper
    yields segments to us as they come out of the model.
    """
    segments_gen, info = model.transcribe(
        wav_path,
        language="en",
        word_timestamps=True,
        beam_size=5,
        vad_filter=False,
    )
    detected_language = str(getattr(info, "language", "en") or "en")
    total = float(total_duration_s or getattr(info, "duration", 0.0) or 0.0)

    segments_out: list[dict] = []
    for seg in segments_gen:
        if cancel_check and cancel_check():
            raise RuntimeError("Cancelled during transcription.")

        seg_start = float(getattr(seg, "start", 0.0) or 0.0)
        seg_end = float(getattr(seg, "end", seg_start) or seg_start)
        seg_words_raw = getattr(seg, "words", None) or []
        seg_words: list[dict] = []
        for w in seg_words_raw:
            w_start = getattr(w, "start", None)
            w_end = getattr(w, "end", None)
            seg_words.append(
                {
                    "word": str(getattr(w, "word", "")),
                    "start": float(w_start if w_start is not None else seg_start),
                    "end": float(w_end if w_end is not None else seg_end),
                }
            )
        segments_out.append(
            {
                "start": seg_start,
                "end": seg_end,
                "text": str(getattr(seg, "text", "") or ""),
                "words": seg_words,
            }
        )

        if progress and total > 0:
            progress(min(seg_end, total), total)

    return {"segments": segments_out, "language": detected_language}


def _transcribe_with_progress(
    model,
    wav_path: str,
    total_duration_s: float | None,
    progress: Optional[ProgressCB],
    cancel_check: Optional[Callable[[], bool]],
    log: Optional[LogCB] = None,
) -> dict:
    """Run ``model.transcribe`` and forward its internal ``tqdm`` progress
    to our ``progress`` callback.

    Whisper drives a ``tqdm`` bar in ``whisper.transcribe`` measured in audio
    frames (see whisper/transcribe.py: ``with tqdm.tqdm(total=content_frames)
    as pbar: ... pbar.update(...)``). We monkey-patch ``whisper.transcribe.tqdm``
    with a shim whose ``.tqdm`` attribute yields a proxy that forwards each
    ``update()`` to us. This gives real, accurate progress and ETA.

    Design notes:
      * The proxy clamps monotonically so the bar never goes backwards.
      * A fallback pinger only activates if the real hook produces NO output
        within 10 seconds and never exceeds 20 % of the phase (so the real
        hook, when it finally fires, will always be strictly ahead).

    Between real ticks:
      * Whisper only calls ``pbar.update()`` at the end of each 30-second
        audio chunk it processes, so real ticks can be 15-20 s apart (or
        much more on CPU with the large model). We run a background
        interpolator that, after the second real tick, extrapolates linearly
        between ticks at the observed rate. It caps just short of the next
        expected tick so the real update always overtakes it cleanly.
    """
    import threading
    import time
    import types

    done_flag = threading.Event()
    forwarded_any = threading.Event()
    interp_state = {
        "max_frac": 0.0,           # highest fraction the bar has shown
        "real_frac": 0.0,          # last fraction from a real tick
        "real_ts": 0.0,            # wall time of last real tick
        "prev_real_frac": 0.0,     # fraction at the tick before that
        "prev_real_ts": 0.0,       # wall time of the tick before that
        "have_two_ticks": False,   # true once we can estimate rate
        "logged_first_hook": False,
    }
    interp_lock = threading.Lock()

    # Load the real tqdm class so our proxy can delegate to it and the
    # terminal progress bar continues to render (in addition to our GUI hook).
    try:
        import tqdm as _tqdm_pkg

        _real_tqdm_cls = _tqdm_pkg.tqdm  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover
        _real_tqdm_cls = None

    class _ProxyTqdm:
        """tqdm.tqdm replacement that both forwards to our progress callback
        and drives a real ``tqdm.tqdm`` so whisper's terminal bar still shows.
        """

        def __init__(self, *args, **kwargs) -> None:
            self._total = kwargs.get("total")
            if self._total is None and args:
                for a in args:
                    if isinstance(a, int):
                        self._total = a
                        break
            self._n = 0
            # Delegate to a real tqdm.tqdm so the terminal bar keeps working.
            self._real = None
            if _real_tqdm_cls is not None:
                try:
                    self._real = _real_tqdm_cls(*args, **kwargs)
                except Exception:  # pragma: no cover
                    self._real = None

        def _emit(self) -> None:
            if not progress or not total_duration_s:
                return
            if not self._total or self._total <= 0:
                return
            frac = self._n / self._total
            now = time.time()
            with interp_lock:
                # Record the tick even if it's not larger (helps rate estimate),
                # but only push the bar forward monotonically.
                if interp_state["real_ts"] > 0.0:
                    interp_state["prev_real_frac"] = interp_state["real_frac"]
                    interp_state["prev_real_ts"] = interp_state["real_ts"]
                    interp_state["have_two_ticks"] = True
                interp_state["real_frac"] = frac
                interp_state["real_ts"] = now

                if frac < interp_state["max_frac"]:
                    return  # monotonic clamp
                interp_state["max_frac"] = frac

            frac = max(0.0, min(1.0, frac))
            progress(frac * total_duration_s, total_duration_s)
            if not forwarded_any.is_set():
                forwarded_any.set()
                if log and not interp_state["logged_first_hook"]:
                    interp_state["logged_first_hook"] = True
                    log(f"Whisper progress hook active (first tick at {frac * 100:.1f}%).")

        def update(self, n: int = 1) -> None:
            self._n += int(n)
            # Drive the real terminal bar too.
            if self._real is not None:
                try:
                    self._real.update(int(n))
                except Exception:  # pragma: no cover
                    pass
            self._emit()
            # Cancellation checkpoint: whisper calls pbar.update() at the end
            # of every 30-second audio chunk it processes. That is our one
            # cooperative hook inside its blocking transcribe() call. Raising
            # here unwinds whisper's loop and hands control back to us.
            if cancel_check and cancel_check():
                raise _WhisperCancelled()

        # tqdm context-manager / iterator protocol used by whisper.
        def __enter__(self) -> "_ProxyTqdm":
            if self._real is not None:
                try:
                    self._real.__enter__()
                except Exception:  # pragma: no cover
                    pass
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            if self._real is not None:
                try:
                    self._real.__exit__(exc_type, exc, tb)
                except Exception:  # pragma: no cover
                    pass
            return None

        def close(self) -> None:  # pragma: no cover - not used by whisper
            if self._real is not None:
                try:
                    self._real.close()
                except Exception:
                    pass

        def set_description(self, *a, **kw) -> None:  # pragma: no cover
            if self._real is not None:
                try:
                    self._real.set_description(*a, **kw)
                except Exception:
                    pass

        def set_postfix(self, *a, **kw) -> None:  # pragma: no cover
            if self._real is not None:
                try:
                    self._real.set_postfix(*a, **kw)
                except Exception:
                    pass

    # Last-resort pinger. Only kicks in if the real hook is silent for a full
    # 45 seconds AND caps out at 20 %, so the real hook (once it fires) always
    # overtakes it cleanly with no visible jump backwards. Whisper only calls
    # pbar.update() at the end of each 30-second audio chunk it processes, and
    # a single chunk can easily take 30+ seconds of wall time on CPU with the
    # large model, so we need a generous timeout here.
    def _pinger() -> None:
        for _ in range(450):  # 45 s grace at 0.1 s
            if done_flag.is_set() or forwarded_any.is_set():
                return
            time.sleep(0.1)
        if log:
            log(
                "Whisper progress hook silent after 45 s; showing coarse "
                "fallback estimate up to 20 %."
            )
        start_fake = time.time()
        while not done_flag.is_set() and not forwarded_any.is_set():
            if cancel_check and cancel_check():
                return
            if progress and total_duration_s and total_duration_s > 0:
                elapsed = time.time() - start_fake
                # Very conservative: 1 s of wall clock -> 1 % of the phase,
                # never above 20 %.
                fake_frac = min(0.20, elapsed / 100.0)
                if fake_frac > interp_state["max_frac"]:
                    with interp_lock:
                        interp_state["max_frac"] = fake_frac
                    progress(fake_frac * total_duration_s, total_duration_s)
            time.sleep(1.0)

    # Interpolator: once we have two real ticks, extrapolate linearly at the
    # observed rate to keep the bar moving smoothly (every 250 ms) between
    # real updates. Caps at (real_frac + 0.95 * last_delta_frac) so the next
    # real update always overtakes it cleanly and never causes a visible jump
    # backwards.
    def _interpolator() -> None:
        while not done_flag.is_set():
            time.sleep(0.25)
            if not (progress and total_duration_s and total_duration_s > 0):
                continue
            with interp_lock:
                if not interp_state["have_two_ticks"]:
                    continue
                real_frac = interp_state["real_frac"]
                real_ts = interp_state["real_ts"]
                prev_frac = interp_state["prev_real_frac"]
                prev_ts = interp_state["prev_real_ts"]
                shown = interp_state["max_frac"]

            delta_frac = real_frac - prev_frac
            delta_time = real_ts - prev_ts
            if delta_frac <= 0.0 or delta_time <= 0.0:
                continue
            # Project at 85 % of the observed rate, capped at 90 % of the way
            # to the next expected tick. Together these keep the bar moving
            # smoothly across almost the whole inter-tick interval while
            # leaving a comfortable 10 % headroom so the next real tick
            # visibly overtakes and never causes a jump backwards. Whisper's
            # inter-tick times are not perfectly regular, so we deliberately
            # under-project rather than run right up to the next tick.
            rate = 0.85 * (delta_frac / delta_time)  # fraction per second
            elapsed_since = time.time() - real_ts
            projected = real_frac + rate * elapsed_since
            ceiling = real_frac + 0.90 * delta_frac
            projected = min(projected, ceiling, 0.999)
            if projected > shown + 1e-4:
                with interp_lock:
                    interp_state["max_frac"] = projected
                progress(projected * total_duration_s, total_duration_s)

    pinger = threading.Thread(target=_pinger, daemon=True)
    interpolator = threading.Thread(target=_interpolator, daemon=True)
    if progress and total_duration_s:
        pinger.start()
        interpolator.start()

    # Monkey-patch whisper.transcribe.tqdm for the duration of the call.
    # Whisper does ``import tqdm`` then ``tqdm.tqdm(...)``, so we replace the
    # module-level ``tqdm`` name with a SimpleNamespace whose ``.tqdm`` is our
    # proxy class.
    #
    # IMPORTANT: whisper/__init__.py does ``from .transcribe import transcribe``
    # which REBINDS ``whisper.transcribe`` from the submodule to the function.
    # So ``import whisper.transcribe as X`` gives us the function, not the
    # module, and patching an attribute on the function object does nothing.
    # We must fetch the real module via ``sys.modules`` (populated correctly by
    # importlib) instead.
    _wt = None
    original_tqdm = None
    try:
        import importlib
        import sys as _sys

        importlib.import_module("whisper.transcribe")  # ensure it's imported
        _wt = _sys.modules.get("whisper.transcribe")
        if _wt is None or not hasattr(_wt, "tqdm"):
            raise RuntimeError("whisper.transcribe module or its tqdm symbol not found")
        original_tqdm = _wt.tqdm
        _wt.tqdm = types.SimpleNamespace(tqdm=_ProxyTqdm)  # type: ignore[assignment]
    except Exception as exc:  # pragma: no cover
        if log:
            log(f"Warning: could not install tqdm progress hook: {exc}")
        _wt = None

    try:
        result = model.transcribe(
            wav_path,
            language="en",
            word_timestamps=True,
            verbose=False,
        )
    except _WhisperCancelled:
        if log:
            log("Cancelled during transcription.")
        raise RuntimeError("Cancelled during transcription.")
    finally:
        done_flag.set()
        if _wt is not None and original_tqdm is not None:
            _wt.tqdm = original_tqdm  # type: ignore[assignment]

    if cancel_check and cancel_check():
        raise RuntimeError("Cancelled during transcription.")

    if log and not forwarded_any.is_set():
        log(
            "Warning: Whisper progress hook never fired. The bar may have "
            "jumped from ~0 % straight to done. Check that openai-whisper is "
            "the standard build (its transcribe.py must import ``tqdm`` and "
            "use ``tqdm.tqdm(total=content_frames)``)."
        )

    return result
