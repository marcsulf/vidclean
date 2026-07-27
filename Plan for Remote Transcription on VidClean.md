I’m going to map the current transcription flow first so the plan is grounded in the existing pipeline, then I’ll outline a remote-offload design with clear integration points and tradeoffs.

Searched for regex `faster-whisper|WhisperModel|transcribe|transcription|model_path|engine` (`**/*.py`), 50 results

Searched for files matching `**/video_censor*.py`, 2 matches

Searched for files matching `**/censor_*.py`, 5 matches

I’m mapping the current transcription pipeline and GUI entry points so the plan fits the existing architecture instead of proposing a fresh redesign. Next I’m checking the current session plan memory and the core transcription files to anchor the remote-offload options.

Read [](file:///c%3A/Temp/Python/VidClean/video_censor_gui.py#1-1), lines 1 to 360

Read memory

Read [](file:///c%3A/Temp/Python/VidClean/video_censor_pipeline.py#1-1), lines 1 to 320

Read [](file:///c%3A/Temp/Python/VidClean/censor_transcribe.py#1-1), lines 1 to 320

I have the current architecture in view now: transcription is a single synchronous step inside the pipeline, with engine selection, model path, caching, and progress callbacks already centralized. Next I’ll turn that into a remote-offload plan that slots into the existing transcribe boundary instead of spreading network logic through the GUI and pipeline.

Searched for regex `^def transcribe|^class TranscribeResult|^def _load_model|^def _transcribe_faster_whisper|^def _transcribe_openai_whisper` (`Python` · `VidClean/censor_transcribe.py`), 2 results

Searched for regex `_load_settings|_save_settings|var_output_mode|var_engine|var_model_path|Download NLTK data` (`Python` · `VidClean/video_censor_gui.py`), 20 results

I’ve confirmed the current pipeline already has a single transcribe step and the GUI persists engine/model/output settings, so the remote option can be added as another transcription backend rather than a whole new mode. I’m reading the exact transcribe and settings plumbing now to make the plan specific about where the new remote client/server boundaries should go.

Read [](file:///c%3A/Temp/Python/VidClean/censor_transcribe.py#506-506), lines 506 to 820

Read [](file:///c%3A/Temp/Python/VidClean/video_censor_gui.py#520-520), lines 520 to 780

Read [](file:///c%3A/Temp/Python/VidClean/video_censor_gui.py#980-980), lines 980 to 1060

I’ve confirmed the key seam: the pipeline already isolates transcription behind a single `transcribe()` call, so the remote option can be added there without rewriting the rest of the censoring flow. I’m drafting a plan now with the remote boundary, payload shape, failure behavior, and GUI wiring kept explicit.

Created memory file [](file:///c%3A/Users/fg6trw/AppData/Roaming/Code/User/workspaceStorage/9dcc0b4d734a6e8c3206e234cddde6ce/GitHub.copilot-chat/memory-tool/memories/NjI1MDFmODctMzE5NC00YzA0LWE2N2EtZjFkMDA1ZWNiNDE0/plan.md)

Read [](file:///c%3A/Temp/Python/VidClean/tests/test_video_censor.py#1-1), lines 1 to 260

## Plan: Remote Faster-Whisper Offload

Keep the censoring pipeline local and move only the transcription stage behind a remote HTTP service. The existing choke point is censor_transcribe.py, so the cleanest path is to add a new backend there that returns the same word-span result shape the rest of the app already consumes.

**Steps**
1. Define the remote API contract first: upload audio, start job, poll status, return transcript JSON with word-level timestamps, language, duration, and error details. Make the response shape compatible with the current `TranscriptionResult` flow so video_censor_pipeline.py does not need downstream changes.
2. Add a remote transcription backend in censor_transcribe.py alongside `openai-whisper` and `faster-whisper`. Treat it as another engine choice or a separate `transcription_mode`, but keep the public pipeline call site stable.
3. Implement a small client module for the local app that handles server URL, auth token, upload, polling, timeout, retry, and cancellation. Prefer HTTP JSON plus multipart upload over a heavier RPC stack unless you specifically need live streaming progress.
4. Add a server-side service on the GPU machine that loads faster-whisper once, accepts jobs, and returns normalized transcript data. This should own model download/cache, CUDA setup, and any ffmpeg decoding if you want the server to accept compressed inputs instead of raw WAV.
5. Extend video_censor_pipeline.py and video_censor_gui.py with settings for remote mode, server URL, auth, timeout, and fallback behavior. Keep the current local engine selector so users can still run fully local when needed.
6. Update cache identity in censor_cache.py so transcript reuse does not mix local and remote results unless you explicitly want that. At minimum, include the remote server identity or version in the cache key.
7. Add validation and error handling for server reachability, oversized audio, HTTP failures, auth failures, and partial job failure. If fallback is enabled, degrade to local transcription only when the remote service is unavailable and the user has opted in.
8. Add tests for request serialization, response parsing, cache invalidation, pipeline integration, and GUI persistence. The existing test suite in test_video_censor.py is the right place to mirror the new mode behavior with mocked network calls.

**Relevant files**
- censor_transcribe.py — current transcription boundary and engine dispatch.
- video_censor_pipeline.py — pipeline config and cache wiring.
- video_censor_gui.py — settings UI and persistence.
- censor_cache.py — transcript cache identity.
- test_video_censor.py — unit and integration-style test coverage.

**Verification**
1. Run the unit suite with the remote client fully mocked so the tests stay offline.
2. Add one stub HTTP server test that returns a fixed transcript and verify the downstream mute intervals match the local path.
3. Manually compare one sample file end-to-end against local faster-whisper to confirm output parity and cache behavior.
4. Confirm the GUI persists the new remote settings and that switching between local and remote does not reuse stale cached transcripts.

**Decisions**
- Only transcription moves remote. Matching, interval building, and ffmpeg encoding stay local.
- Use the same downstream transcript shape everywhere to minimize code churn.
- Keep local and remote caches separate unless the remote service can prove stable model identity and compatible output.
- Start with HTTP polling, not streaming, unless you have a strong need for live remote progress updates.

If you want, I can turn this into a more concrete implementation plan with a proposed API schema and UI field list.