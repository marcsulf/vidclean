"""Remote Whisper transcription client for the video censor tool.

Connects to a FastAPI batch service (shakespeare-whisper-batch) over HTTP.
Handles multipart audio upload, async job polling with progress reporting,
result retrieval as WordSpan lists, and cooperative cancellation.

Usage
-----
    from remote_transcribe import RemoteEngineClient, RemoteTranscriptionError

    client = RemoteEngineClient(
        server_url="http://shakespeare.whitmore4792:18120",
        api_key="<key>",
    )
    words = client.transcribe(
        audio_path="/tmp/stream.aac",
        progress_cb=lambda elapsed, total: print(f"{elapsed/total:.0%}"),
        cancel_check=lambda: False,
    )
"""

from __future__ import annotations

import logging
import time
from typing import Callable, List, Optional

import requests  # type: ignore

from censor_timestamps import WordSpan

logger = logging.getLogger(__name__)

# Default polling interval in seconds. The server status endpoint reports
# processed_seconds so the client can compute a real progress fraction.
DEFAULT_POLL_INTERVAL_S = 1.0

# Default timeout for individual HTTP requests (seconds). Generous enough for
# large file uploads on a home LAN but will still fail fast if the server is
# unreachable.
DEFAULT_REQUEST_TIMEOUT = 600  # 10 minutes


class RemoteTranscriptionError(RuntimeError):
    """Raised when remote transcription fails (network, server error, or cancel)."""

    pass


ProgressCB = Callable[[float, float], None]  # (elapsed_s, total_s)


class RemoteEngineClient:
    """HTTP client for the shakespeare-whisper-batch service.

    Parameters
    ----------
    server_url :
        Base URL of the FastAPI batch service. Trailing slashes stripped.
    api_key :
        Bearer token sent as ``Authorization: Bearer <key>``.
    poll_interval_s :
        Seconds between status-poll requests during an active job.
    request_timeout :
        Timeout for individual HTTP calls in seconds.
    """

    def __init__(
        self,
        server_url: str,
        api_key: str,
        poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
        request_timeout: int = DEFAULT_REQUEST_TIMEOUT,
    ) -> None:
        self._server_url = server_url.rstrip("/")
        self._api_key = api_key
        self._poll_interval = max(0.1, poll_interval_s)
        self._timeout = max(5, request_timeout)
        self._session = requests.Session()
        self._session.headers.update({"Authorization": f"Bearer {api_key}"})

    # ------------------------------------------------------------------- API --

    def transcribe(
        self,
        audio_path: str,
        progress_cb: Optional[ProgressCB] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
        log: Optional[Callable[[str], None]] = None,
    ) -> List[WordSpan]:
        """Upload audio, poll for completion, and return word-level timestamps.

        Parameters
        ----------
        audio_path :
            Path to the (compressed) audio file to transcribe.
        progress_cb :
            Optional callback ``(elapsed_s, total_s)`` during transcription.
        cancel_check :
            Returns ``True`` when the user has requested cancellation.
        log :
            Optional callback for informational / error messages.

        Returns
        -------
        list[WordSpan]
            Word-level timestamps matching the transcript.
        """
        if log:
            log(f"Uploading audio to remote server {self._server_url}...")
        job_id, _remote_version = self._upload_and_create_job(audio_path)
        if log:
            log(f"Remote job created: {job_id}")
        words = self._poll_status(
            job_id=job_id,
            progress_cb=progress_cb,
            cancel_check=cancel_check,
            log=log,
        )
        return words

    def get_server_version(self) -> str:
        """Return the server's remote_model_version (or empty string on error)."""
        try:
            resp = self._session.get(
                f"{self._server_url}/api/health",
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json()
                return str(data.get("remote_model_version", ""))
        except Exception as exc:
            logger.warning("Health check failed: %s", exc)
        return ""

    # --------------------------------------------------------------- Internals --

    def _upload_and_create_job(
        self,
        audio_path: str,
    ) -> tuple[str, str]:
        """Upload the audio file and return ``(job_id, remote_model_version)``.

        Raises ``RemoteTranscriptionError`` on failure.
        """
        try:
            with open(audio_path, "rb") as f:
                files = {"file": (audio_path, f)}
                resp = self._session.post(
                    f"{self._server_url}/api/transcribe",
                    files=files,
                    timeout=self._timeout,
                )
        except requests.exceptions.ConnectionError as exc:
            raise RemoteTranscriptionError(
                f"Cannot connect to remote server at {self._server_url}: "
                f"{exc}"
            ) from exc
        except requests.exceptions.Timeout:
            raise RemoteTranscriptionError(
                f"Request timed out connecting to {self._server_url}. "
                "Check network and server status."
            )
        except OSError as exc:
            raise RemoteTranscriptionError(f"Failed to read audio file: {exc}") from exc

        if resp.status_code == 401:
            raise RemoteTranscriptionError(
                f"Authentication failed for {self._server_url}. "
                "Check the API key in settings."
            )
        if resp.status_code != 202:
            try:
                detail = resp.json().get("detail", resp.text)
            except Exception:
                detail = resp.text
            raise RemoteTranscriptionError(
                f"Server returned {resp.status_code}: {detail}"
            )

        data = resp.json()
        job_id: str = data["job_id"]
        remote_version: str = data.get("remote_model_version", "")
        return job_id, remote_version

    def _poll_status(
        self,
        job_id: str,
        progress_cb: Optional[ProgressCB] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
        log: Optional[Callable[[str], None]] = None,
    ) -> List[WordSpan]:
        """Poll /status until the job is completed or errors out."""
        last_processed = 0.0
        total_duration: float | None = None

        while True:
            if cancel_check and cancel_check():
                self._cancel_job(job_id, log=log)
                raise RemoteTranscriptionError("Cancelled")

            try:
                resp = self._session.get(
                    f"{self._server_url}/api/jobs/{job_id}/status",
                    timeout=30,
                )
            except requests.exceptions.ConnectionError as exc:
                raise RemoteTranscriptionError(
                    f"Lost connection to server while polling: {exc}"
                ) from exc
            except requests.exceptions.Timeout:
                raise RemoteTranscriptionError(
                    "Status request timed out. Server may be overloaded."
                )

            if resp.status_code == 401:
                raise RemoteTranscriptionError("Session expired. Check API key.")
            if resp.status_code == 404:
                raise RemoteTranscriptionError(
                    f"Job {job_id} not found on server (may have been cleaned up)."
                )
            if resp.status_code != 200:
                raise RemoteTranscriptionError(
                    f"Unexpected status response {resp.status_code}"
                )

            try:
                data = resp.json()
            except Exception:
                raise RemoteTranscriptionError("Malformed server response (status)")

            status = str(data.get("status", ""))

            if status in ("queued", "processing"):
                processed = float(data.get("processed_seconds", 0.0))
                total = data.get("total_duration_s")
                if total is not None:
                    total_duration = float(total)
                progress_so_far = max(processed, last_processed)
                if total_duration and progress_cb:
                    progress_cb(progress_so_far, total_duration)
                last_processed = progress_so_far
                time.sleep(self._poll_interval)
                continue

            if status == "completed":
                return self._fetch_result(job_id, log=log)

            if status == "cancelled":
                raise RemoteTranscriptionError("Cancelled (server-side).")

            detail = data.get(
                "error",
                f"Job ended in unexpected state '{status}'",
            )
            raise RemoteTranscriptionError(detail)

    def _fetch_result(
        self,
        job_id: str,
        log: Optional[Callable[[str], None]] = None,
    ) -> List[WordSpan]:
        """Fetch the completed result JSON and convert to WordSpan list."""
        if log:
            log("Fetching transcription result...")

        try:
            resp = self._session.get(
                f"{self._server_url}/api/jobs/{job_id}/result",
                timeout=60,
            )
        except requests.exceptions.ConnectionError as exc:
            raise RemoteTranscriptionError(f"Cannot fetch result: {exc}") from exc

        if resp.status_code == 404:
            raise RemoteTranscriptionError(
                f"Result for job {job_id} not found (may have expired)."
            )
        if resp.status_code != 200:
            try:
                detail = resp.json().get("detail", resp.text)
            except Exception:
                detail = resp.text
            raise RemoteTranscriptionError(
                f"Server returned {resp.status_code} when fetching result: {detail}"
            )

        data = resp.json()
        words_data = data.get("words", [])
        words = [
            WordSpan(
                text=str(w["text"]),
                start=float(w["start"]),
                end=float(w["end"]),
            )
            for w in words_data
        ]

        if log:
            log(f"Received {len(words)} word-level timestamps from remote server.")

        return words

    def _cancel_job(
        self,
        job_id: str,
        log: Optional[Callable[[str], None]] = None,
    ) -> None:
        """Send cooperative cancel request to the server (best-effort)."""
        try:
            resp = self._session.post(
                f"{self._server_url}/api/jobs/{job_id}/cancel",
                timeout=10,
            )
            if resp.status_code == 200 and log:
                log("Remote job cancelled.")
            elif resp.status_code == 409 and log:
                log("Remote job already completed; cancel was a no-op.")
            elif log:
                log(
                    f"Cancel request returned {resp.status_code}; "
                    "local cancellation proceeds anyway."
                )
        except Exception as exc:
            logger.warning("Cancel request failed (non-fatal): %s", exc)

    def close(self) -> None:
        """Close the underlying HTTP session."""
        self._session.close()




