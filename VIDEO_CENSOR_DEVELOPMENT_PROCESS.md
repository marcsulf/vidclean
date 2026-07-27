# Video Censor Tool — Development Process

This document captures the full end-to-end development process for the
Tkinter-based video audio censor tool built in this workspace on
**July 7, 2026**. It preserves the actual (or closely approximated) user
prompts in order, describes what was done at each step, and calls out the
technical obstacles that appeared along the way.

> User prompts are reproduced verbatim where captured and paraphrased where
> not — paraphrased versions are marked *(approx.)*.

---

## 1. Project brief

The starting goal (established across the initial exchange):

Build a Tkinter GUI Python tool that:

1. Takes a video file as input.
2. Uses PyTorch (CUDA if available) + OpenAI Whisper to transcribe the
   audio with accurate **word-level** timestamps.
3. Matches transcribed words — including derivatives like plurals, past
   tense, `-ing`, possessive, and compounds — against an editable
   plain-text censor word list.
4. Invokes FFmpeg to produce a censored copy of the video with matched
   words either **muted** (silenced) or **beeped**.
5. Provides a progress bar with real ETA and a live log panel.

### Initial design decisions (from the "1,1,1,3,1,1,1,1,1" answers)

| Question | Choice |
|---|---|
| Transcription engine | `openai-whisper` |
| Derivative matcher | NLTK WordNet lemma + Porter stem hybrid |
| Matching style | Whole-word (Scunthorpe-safe) |
| Censor method | **User-selectable mute or beep** |
| Padding around muted words | 50 ms (later split into 150 ms pre / 50 ms post) |
| Word list format | Plain text, one per line, `#` comments |
| Language | English only |
| Output container | Same as input |
| Package installation | User handles installs |

---

## 2. Chronological prompts and outcomes

Each subsection is a user turn, in order.

### 2.1 Initial implementation

> **User** *(paraphrased opening)*: "Build the video censor tool per the
> spec above."

**Actions taken**:

- Created the seven-file module layout:
  - [censor_wordlist.py](censor_wordlist.py) — WordNet+Porter matcher, wildcard modes.
  - [censor_timestamps.py](censor_timestamps.py) — Interval padding/merging.
  - [censor_transcribe.py](censor_transcribe.py) — Whisper wrapper, device auto-detect.
  - [censor_audio_io.py](censor_audio_io.py) — FFmpeg / FFprobe wrappers.
  - [video_censor_pipeline.py](video_censor_pipeline.py) — Orchestration + progress phases.
  - [video_censor_gui.py](video_censor_gui.py) — Tkinter GUI entry point.
  - [tests/test_video_censor.py](tests/test_video_censor.py) — Unit tests.
- Plus supporting files: [censor_words.txt](censor_words.txt),
  [requirements_video_censor.txt](requirements_video_censor.txt),
  [VIDEO_CENSOR_README.md](VIDEO_CENSOR_README.md).
- All 26 initial tests passed.

**Key design notes**:

- Pipeline broken into weighted phases (probe/extract/transcribe/match/encode)
  for accurate global progress fraction.
- Threading model: worker thread + `queue.Queue` drained by
  `root.after(100, ...)`.
- FFmpeg commands: `volume=enable='between(t,s,e)':volume=0` for mute,
  `aevalsrc` + `amix` for beep. `-c:v copy` to preserve video losslessly.

### 2.2 Torch install failure on Python 3.13

> **User** *(approx.)*: "I'm getting an install error installing torch."

**Actions**: Documented alternative CUDA wheel channels (cu121 → cu124,
cu126) since the requested combination didn't have a Python 3.13 wheel
yet.

### 2.3 Confirm `-ing` derivative support

> **User** *(approx.)*: "Does the derivative matching cover `-ing`
> forms?"

**Answer + verification**: Yes, WordNet + Porter handles verb inflections
including `-ing`. Showed the existing test coverage that already asserts
`damning` matches `damn`.

### 2.4 Wildcard word entries

> **User** *(approx.)*: "Can I add wildcards to the censor list — like
> `*fuck*` to catch every compound, or `fuck*` for prefixes?"

**Actions**:

- Added three wildcard match modes: substring (`*word*`), prefix
  (`word*`), suffix (`*word`).
- Extended [censor_wordlist.py](censor_wordlist.py) `CensorEntry` with a
  `mode` field and `_parse_entry_mode()` helper.
- Added tests. Suite grew to 33 tests, all passing.

### 2.5 FFmpeg folder override

> **User** *(approx.)*: "FFmpeg isn't on my PATH. Let me point the GUI at
> the folder that contains it."

**Actions**:

- Added a "FFmpeg folder (optional)" field with **Browse** and **Detect**
  buttons.
- Threaded a `ffmpeg_dir` parameter through
  [censor_audio_io.py](censor_audio_io.py) `_resolve_binary()`,
  `probe_duration_s()`, `extract_audio_wav()`, `run_censor_encode()`, and
  [censor_transcribe.py](censor_transcribe.py) `transcribe()`.

### 2.6 Rolling log file

> **User** *(approx.)*: "I want a persistent log file. Give me option 2
> from the choices you offered."

**Actions**:

- Added `video_censor.log` in the script folder, appended on every run
  with a session banner (`===== YYYY-MM-DD HH:MM:SS input=... =====`).
- **Open log file** and **Clear log file** buttons in the GUI.

### 2.7 NLTK download failure (corporate SSL)

> **User** *(approx.)*: "NLTK data download fails with a certificate
> error."

**Actions**:

- Added `ensure_nltk_data()` in [censor_wordlist.py](censor_wordlist.py).
- On initial cert failure, falls back to `ssl._create_unverified_context`
  and retries.
- Added a **Download NLTK data** GUI button that runs the fetch on a
  worker thread and reports success or failure via `messagebox`.

### 2.8 Whisper WinError 2 (ffmpeg not found)

> **User** *(approx.)*: "Whisper crashes with `[WinError 2] The system
> cannot find the file specified` even though my ffmpeg is set."

**Root cause**: Whisper's `load_audio` invokes `ffmpeg` by bare name only,
resolved against `os.environ["PATH"]`. Our GUI-configured `ffmpeg_dir`
never made it into that lookup.

**Actions**:

- Added `_augmented_path_for_ffmpeg` context manager in
  [censor_transcribe.py](censor_transcribe.py) that prepends `ffmpeg_dir`
  to `PATH` for the duration of the transcribe call.

### 2.9 Suppress Triton warning

> **User** *(approx.)*: "The Triton kernel warning is noisy. Please
> suppress it (option 2)."

**Actions**: Added a targeted
`warnings.filterwarnings("ignore", message=r".*Failed to launch Triton
kernels.*")` at the top of `transcribe()`.

### 2.10 Progress bar racing ahead

> **User** *(approx.)*: "The progress bar is jumping way too fast — must
> not be reading whisper's real progress."

**Actions (first attempt)**:

- Tried to monkey-patch `whisper.transcribe.tqdm` by importing
  `whisper.transcribe as _wt` and doing `_wt.tqdm = _ProxyTqdm`.
- This appeared to work in isolated tests but did **not** actually
  intercept whisper's real bar (see 2.16 for the real root cause).

### 2.11 Split symmetric padding into asymmetric

> **User** *(approx.)*: "Split the 50 ms padding into a pre-pad and a
> post-pad. Default pre to 150 ms."

**Actions**:

- Renamed `pad_ms` → `pre_pad_ms` (default 150) + `post_pad_ms`
  (default 50).
- Updated [censor_timestamps.py](censor_timestamps.py)
  `build_mute_intervals` signature; kept a legacy `pad_s` fallback.
- Updated tests. Suite: 36 tests passing.

### 2.12 Migrate legacy settings + fix default

> **User** *(approx.)*: "My saved settings still have `pad_ms: 50` — the
> pre-pad default of 150 isn't being applied."

**Actions**:

- Updated the settings backward-compat shim: legacy `pad_ms` is used only
  for `post_pad_ms`; `pre_pad_ms` always defaults to 150.
- Migrated the user's [video_censor_settings.json](video_censor_settings.json).

### 2.13 Transcript caching

> **User** *(paraphrased, close to actual)*: "The long pole in the tent
> for this pipeline is obviously the transcription process. I can foresee
> use cases where I have already run on a given video file, but then I
> want to change the amount of censoring I do (i.e. change the censor
> word list) and then create a new output video, without having to redo
> the transcription. Can we add this functionality?"

**Actions**:

- Created [censor_cache.py](censor_cache.py) with:
  - `CachedTranscript` dataclass, `CACHE_VERSION = 1`.
  - `cache_path_for(input)` → `<input>.transcript.json`.
  - `load_cache()`, `save_cache()`, `clear_cache()` with validity check
    on `file_size`, `mtime`, and `model_size`.
- Pipeline: two phase-weight tables (`_PHASE_WEIGHTS_FULL` vs
  `_PHASE_WEIGHTS_CACHED`); on cache hit it skips extract+transcribe.
- GUI: **"Use cached transcript when available"** checkbox (default on),
  **"Delete cached transcript for input"** button, live status label
  (`Cache: none` / `exists but stale` / `valid, N words`), with
  `trace_add` on the input + model vars so the label refreshes
  automatically.
- Added 6 new cache tests. Suite: **42 tests passing**.
- Completion dialog reports `"Used cached transcript (skipped
  transcription)."` when applicable.

### 2.14 Progress bar still broken

> **User** *(approx.)*: "Progress bar is still not functioning correctly.
> No change in its behavior."

**Actions**:

- Confirmed via an isolated test that the tqdm hook *appeared* to work,
  but was actually still not firing during real whisper runs.
- Added a monotonic clamp inside `_ProxyTqdm.update()`.
- Bumped fallback-pinger cap from 90 % to 20 %.
- Added diagnostic log messages (`"Whisper progress hook active…"` on
  success, `"Whisper progress hook silent after N s…"` on failure).

### 2.15 Whisper's real chunk cadence exposed

> **User** *(paraphrased)*: "Note that from what I am observing on the
> whisper progress bar, it is often going 15 to 20 seconds between
> updates."

**Actions**:

- Added an **interpolator thread** in
  [censor_transcribe.py](censor_transcribe.py) that wakes every 250 ms.
- Once we have two real tqdm ticks, it estimates the rate and linearly
  extrapolates between them, capped at 50 % of the way to the next
  expected tick (so real ticks always overtake cleanly).

### 2.16 The actual root cause — `whisper.transcribe` shadowing

> **User** *(paraphrased)*: "The progress bar function is much worse now.
> After starting transcription, no update for 10 seconds, then it hits
> the fallback log line, races up to 22 % and stops. Meanwhile the
> whisper terminal bar is progressing smoothly at ~32 s per update.
> 10 seconds is way too short before we start faking."

**Root cause discovered**:

`whisper/__init__.py` contains
`from .transcribe import transcribe`, which **rebinds** the attribute
`whisper.transcribe` from the submodule to the function. So

```python
import whisper.transcribe as _wt  # _wt is now a FUNCTION, not a module
_wt.tqdm = shim                    # harmless attribute on a function
```

Whisper's real code inside the submodule kept resolving `tqdm` via its
own module globals and never saw our patch.

**Fixes**:

1. Use `sys.modules["whisper.transcribe"]` (populated by `importlib`) to
   get the *real* module and patch it there.
2. Bumped silent-hook fallback timeout from 10 s → 45 s.
3. Verified with a direct test that patched a `_Marker` sentinel through
   `sys.modules` and confirmed
   `whisper.Whisper.transcribe.__globals__["tqdm"]` picks it up.

Saved the whole gotcha to repository memory
(`/memories/repo/whisper-tqdm-hook.md`).

### 2.17 Cancel button doesn't cancel Whisper

> **User** *(approx.)*: "The Cancel button does not cancel the whisper
> run. I hit the button, and whisper is still chugging along."

**Root cause**: `model.transcribe()` is a single blocking C-level call.
No cancel-flag polling inside it.

**Fix**: The tqdm hook is the one cooperative interrupt point.

- Added `_WhisperCancelled(BaseException)` sentinel.
  - Inherits `BaseException` (not `Exception`) so no stray
    `except Exception` in whisper or torch can swallow it.
- `_ProxyTqdm.update()` now checks `cancel_check()` after emitting and
  raises `_WhisperCancelled`.
- The `try / except _WhisperCancelled` around `model.transcribe()`
  restores whisper's original `tqdm` and re-raises as
  `RuntimeError("Cancelled during transcription.")`.
- Pipeline recognizes the "Cancelled…" prefix and logs
  `"Pipeline cancelled by user (during transcription)."` while
  preserving the wording for GUI detection.
- GUI's cancel check widened from `msg == "cancelled"` to
  `msg.lower().startswith("cancelled")`.

**Caveat noted**: Cancel only takes effect at the end of the *current*
30-second whisper chunk (up to ~30 s of wall clock).

### 2.18 Terminal tqdm bar and ETA drift

> **User** *(paraphrased)*: "Good message: 'Whisper progress hook active
> (first tick at 3.1 %)'. But it races to 15 % after that and stops, and
> the ETA is way off until we hit about 30 %. Also the terminal bar
> vanished."

**Actions**:

- **Terminal bar restored**: `_ProxyTqdm` now also drives a real
  `tqdm.tqdm` instance (via delegation), so whisper's console bar
  reappears in addition to our GUI hook.
- **Rolling-window ETA** in [video_censor_pipeline.py](video_censor_pipeline.py):
  every `_emit()` appends `(now, overall_fraction)` to a small history;
  entries older than **60 s** are dropped (but at least 3 samples always
  kept). ETA is computed from the oldest sample in the window, so the
  fast startup ticks don't distort the long-term rate. Verified with a
  simulation: ETA correctly settled at ~60 s for a fake ~60 s run,
  vs. the old formula's wildly optimistic ~38 s.
- Interpolator ceiling lowered from 95 % to 50 % of the way to the next
  expected tick.

### 2.19 Even smoother interpolation

> **User** *(approx.)*: "The bar surges 1 %, drifts smoothly for 10–15 s,
> then sits idle for the rest of the 30 s gap between whisper ticks. Can
> you make it more continuous?"

**Actions**:

- Interpolator rate reduced to 0.85 × observed rate (was 1.0 ×).
- Ceiling raised to 90 % of the way to the next expected tick (was
  50 %). Motion now covers ~90 % of the inter-tick interval with a
  ~1–2 s idle before the next real tick clearly overtakes.

### 2.20 Batch mode

> **User** *(paraphrased)*: "Now I would like to add a batch mode so that
> I can queue up several video files to run sequentially."

**Actions**:

- Replaced the single Input/Output rows with a **queue Listbox** and
  buttons: **Add files…**, **Remove selected**, **Move up**,
  **Move down**, **Clear queue**.
- Outputs always auto-derived to `<name>_censored.<ext>` next to each
  input.
- Cache status label and delete button operate on the currently selected
  queue item (or the first item if none selected).
- New line above the progress bar: `File 3 of 7:  video.mp4`.
- Added `_batch_thread()` that iterates the queue and posts
  `queue_status`, `file_done`, `batch_done` messages to the log queue.
- Final categorized dialog: succeeded / failed / cancelled / skipped.
- Settings persistence: `queue_paths` list (backward-compat migrates a
  legacy single `input_video` field to a one-item queue).

### 2.21 Layout tweak + split elapsed timer

> **User**: "Make the height of the Input videos queue text box only 5
> lines in the GUI and use the extra space to give to the log output.
> Also, have two different elapsed times: Elapsed time (current file) and
> Elapsed time (batch)."

**Actions**:

- Queue frame changed to `sticky="ew"` with listbox `fill="x"` so it
  stays at 5 lines regardless of window height; reclaimed vertical space
  goes to the log (which has `expand=True`).
- Replaced `var_elapsed` with `var_elapsed_file` + `var_elapsed_batch`.
- Added `_file_start_ts`; batch thread's `queue_status` message resets
  it at the start of each queued file.
- Progress row now shows: `ETA: ... | File: ... | Batch: ...`.

### 2.22 Load Whisper model once per batch

> **User**: "Is it necessary for whisper to load the model before
> transcribing every file? Couldn't it just load once before starting the
> batch?"

**Actions**:

- Added module-level `_MODEL_CACHE` in
  [censor_transcribe.py](censor_transcribe.py), keyed by
  `(model_size, resolved_device)` — single slot so we never hold two
  large models in VRAM.
- New `_get_or_load_model()`, `clear_model_cache()`, and public
  `preload_model()` helper.
- The batch worker calls `preload_model()` **once, before the file loop**
  starts, so the 5–20 s model-load cost is not lumped into file 1's ETA.
- Model changes between batches trigger a clean unload of the previous
  entry (and `torch.cuda.empty_cache()`).

### 2.23 Multi-word phrase entries

> **User**: "Can the censor word list contain two word phrases? For
> example, I don't want to censor Jesus or Christ, but I do want to
> censor 'Jesus Christ'."

**Actions**:

- New match mode: `MODE_PHRASE`. Any entry containing internal whitespace
  is treated as a phrase.
- Each sub-token is a whole-word sub-entry with its own lemma set and
  Porter stem, so `holy cow` also matches `holy cows`.
- New sequence-level API: `WordMatcher.find_matches(tokens)` returns
  `[(start_idx, end_idx_exclusive), ...]`. Phrases tried first (longest
  wins), then single-word entries. Non-overlapping.
- [censor_timestamps.py](censor_timestamps.py)
  `build_mute_intervals` now uses `find_matches`, so a phrase becomes
  one interval spanning all its tokens.
- Punctuation-tolerant (`Jesus,` `Christ!` still matches).
- Added `TestPhraseMatching` with 9 tests. Suite: **51 tests passing**.

### 2.24 Per-entry censor tally in the log

> **User**: "Also, can you report in the log how many words or phrases
> were censored for each video file processed?"

**Actions**:

- Extended matcher with `find_matches_detailed()` returning
  `(start, end, entry_original)` tuples so each match carries the
  originating entry text.
- `find_matches()` now delegates to `find_matches_detailed()`.
- Pipeline: new `_count_matches_detailed()` returns total muted words
  plus a sorted `[(entry_text, count), ...]` list.
- Match-phase log now prints:
  ```
  Censored entries (occurrences):
       3 x  damn
       2 x  *fuck*
       2 x  jesus christ
  ```
  Sorted by descending count, then alphabetically.

### 2.25 Documentation request

> **User**: "Please create a markdown file documenting the full process
> we used today to create this application, including all prompts I used
> (or approximations if original prompts aren't available)."

**Action**: Created the first revision of this file.

### 2.26 Audio-delay (A/V sync) adjustment

> **User**: "The audio in the most recent video output I got is slightly
> (maybe 100 ms or so) ahead of the video. Just enough to be slightly
> annoying. Can we add a delay audio feature that allows the audio to be
> shifted relative to the video. That way I can run it through again,
> and since it is already transcribed it should go fast."

**Actions**:

- New `_audio_delay_filter(audio_delay_ms)` helper in
  [censor_audio_io.py](censor_audio_io.py):
  - Positive → `adelay=N:all=1` (pad silence at the start, pushing audio
    later — used when audio leads video).
  - Negative → `atrim=start=X,asetpts=PTS-STARTPTS` (trim the first |N|
    ms — used when audio lags video).
  - Zero → empty string (caller omits the step).
- `build_mute_filter(intervals, audio_delay_ms=0)` — delay clause is
  appended **after** the mute clauses so the interval timings still
  reference the original audio timeline.
- `build_beep_filter_complex(intervals, beep_freq_hz, audio_delay_ms=0)`
  — delay is applied as a new stage on a `[mixed]` label after `amix`
  so the muted-original and beep-tone streams stay perfectly aligned
  before the shift.
- `MuteJobConfig.audio_delay_ms: int = 0`; `run_censor_encode` threads
  it through and mentions non-zero values in the "Encoding output…" log
  line.
- `PipelineConfig.audio_delay_ms` added in
  [video_censor_pipeline.py](video_censor_pipeline.py) and propagated
  into the `MuteJobConfig`.
- GUI ([video_censor_gui.py](video_censor_gui.py)):
  - New `var_audio_delay_ms = tk.IntVar(value=0)`.
  - New third settings row with an "Audio delay (ms):" spinbox
    (range −2000 to 2000, increment 10) and inline hint text
    ("positive = delay audio (fix audio-ahead-of-video); negative =
    advance audio").
  - Wired into `_build_config_for()`, `_settings_dict()`,
    `_load_settings()`, and the rolling log-file banner.
- Added `TestAudioDelayFilters` with 8 tests covering both filter
  builders for zero, positive, and negative delays plus the
  "no intervals + delay" edge case. Suite: **59 tests passing**.
- README updated with a new item 10 in the settings walkthrough.

**Design note**: The pipeline's transcript cache means the workflow the
user described works exactly as intended — re-running with a different
audio-delay value hits the cache, skips probe/extract/transcribe, and
only the ~10 % encode phase actually runs.

### 2.27 Documentation update

> **User**: "Please also update the development process markdown file with
> the new addition."

**Action**: Added the audio-delay section to this file.

### 2.28 Choice of transcription engine (OpenAI Whisper vs. Faster Whisper)

> **User**: "Can we add support for being able to select between using
> openai-whisper and faster-whisper for transcription through a selection
> box in the GUI?"

**Actions**:

- New public constant `AVAILABLE_ENGINES = ("openai-whisper",
  "faster-whisper")` in [censor_transcribe.py](censor_transcribe.py).
- New `_normalize_engine()` helper accepts common aliases (`"Faster
  Whisper"`, `"faster_whisper"`, `"whisper"`, etc.) and returns the
  canonical id.
- Module-level `_MODEL_CACHE` key changed from `(model_size, device)` to
  `(engine, model_size, device)` so switching engines cleanly unloads the
  previous model instead of holding two large models in memory at once.
- `_get_or_load_model()` now dispatches:
  - `openai-whisper` → `whisper.load_model(model_size, device=...)` (as before).
  - `faster-whisper` → `WhisperModel(model_size, device=...,
    compute_type="float16" if cuda else "int8")` with a fallback to
    `compute_type="default"` if CTranslate2 doesn’t support the chosen
    type on the runtime.
- New `_transcribe_faster_whisper()` adapter:
  - Calls `model.transcribe(wav_path, language="en",
    word_timestamps=True, beam_size=5, vad_filter=False)`.
  - Iterates the streaming segment generator; for each segment it
    normalizes to openai-whisper’s dict shape (`{"segments": [...],
    "language": ...}` with per-segment `{"start", "end", "text",
    "words": [{"word", "start", "end"}, ...]}`) so
    `flatten_transcript_words()` and everything downstream works
    **unchanged**.
  - Emits `progress(segment_end, total)` per segment — much more
    frequent than openai-whisper’s 30-second chunk cadence, so no
    interpolator is needed.
  - Polls `cancel_check()` between segments and raises
    `RuntimeError("Cancelled during transcription.")` so the pipeline’s
    existing cancel-handling code path just works.
- `transcribe()` and `preload_model()` gained an `engine` parameter; the
  public `transcribe()` dispatches to the faster-whisper adapter or the
  existing openai-whisper tqdm-hooked path.
- Cache ([censor_cache.py](censor_cache.py)) now stores an `"engine"`
  field. `load_cache(..., engine=...)` and `save_cache(..., engine=...)`
  accept it as a keyword arg. Backward-compat: cache files predating
  engine selection have no `engine` field and are treated as
  `"openai-whisper"`, so existing users don’t lose their caches.
- Pipeline ([video_censor_pipeline.py](video_censor_pipeline.py)):
  `PipelineConfig.engine`, threaded through `load_cache`, `save_cache`,
  and `transcribe`.
- GUI ([video_censor_gui.py](video_censor_gui.py)):
  - New `var_engine = tk.StringVar(value="OpenAI Whisper")`.
  - New row 3 in the Settings frame: "Transcription engine:" combobox
    with values `("OpenAI Whisper", "Faster Whisper")` plus an inline
    hint mentioning the ~4-8x speedup and the `pip install
    faster-whisper` requirement.
  - New `_canonical_engine()` helper maps the friendly label back to the
    canonical id.
  - Wired into `_build_config_for()`, `_settings_dict()`,
    `_load_settings()` (with backward-compat), `preload_model()` call,
    `_refresh_cache_status()` trace, and the rolling log-file banner.
- [requirements_video_censor.txt](requirements_video_censor.txt) now
  lists `faster-whisper>=1.0.0` as an optional dependency with a comment
  explaining that it enables the second engine.
- Added `TestEngineSelection` with 6 tests: `AVAILABLE_ENGINES`
  presence, `_normalize_engine()` alias handling, cache backward-compat,
  cache engine mismatch invalidates, `_transcribe_faster_whisper()`
  output shape via a fake faster-whisper model, and mid-stream cancel
  handling. Suite: **65 tests passing**.

**Design note**: Because caches are keyed by `(engine, model_size,
device)` and the sidecar records the engine, switching engines simply
re-transcribes. Re-running with the *same* engine (to change padding,
method, audio delay, word list, …) still hits the cache and skips
transcription entirely.

### 2.29 Documentation update

> **User**: Requested that the development-process markdown file be
> updated to reflect the engine-selection work.

**Action**: Added the engine-selection sections to this file.

### 2.30 Corporate-SSL failure downloading faster-whisper model

> **User** *(pasted the error)*:
> ```
> Warning: model preload failed (Got: ConnectError:
> [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed:
> self-signed certificate in certificate chain (_ssl.c:1028)
> An error happened while trying to locate the files on the Hub, and we
> cannot find the appropriate snapshot folder for the specified revision
> on the local disk. Please check your internet connection and try
> again.); will retry per-file.
> ```

**Root cause**: `faster-whisper` downloads model files via
`huggingface_hub`, which in turn uses `requests`. On the user's
corporate network an HTTPS proxy injects a self-signed certificate that
Python's default CA bundle doesn't trust — the same class of failure
we previously hit with NLTK's WordNet download.

**Actions**:

- New `_looks_like_ssl_error(exc)` helper in
  [censor_transcribe.py](censor_transcribe.py) that walks the exception
  chain and returns True on `SSL:`, `certificate verify failed`,
  `self-signed certificate`, `SSLCertVerificationError`, etc.
- New `_disable_hf_ssl_verification()` helper takes a belt-and-suspenders
  approach because `huggingface_hub`'s API surface has moved around:
  1. Monkey-patch `requests.sessions.Session.__init__` so every session
     created after this call has `verify = False`. Works for every
     library that uses `requests`, including hf_hub 1.x.
  2. If `huggingface_hub.configure_http_backend` exists (0.x line),
     install an unverified session factory there too. Harmless when
     absent.
  3. Call `urllib3.disable_warnings(InsecureRequestWarning)` so the
     retry logs stay readable.
  4. Guarded by a module-level `_HF_SSL_BACKEND_DISABLED` flag so we
     don't repeat the patch on every model load in a batch.
- New `_load_faster_whisper_with_ssl_fallback()` wrapper: try the
  normal `WhisperModel(...)` load; on `_looks_like_ssl_error(exc)`,
  install the unverified backend, log a warning explaining what happened
  (and that it's safe if the user trusts the network), and retry
  once. Non-SSL errors propagate on the first attempt so unrelated
  failures aren't accidentally silenced.
- `_get_or_load_model()`'s faster-whisper branch now goes through this
  wrapper, and the `compute_type` fallback path is careful **not** to
  swallow SSL errors: it re-raises them so the user sees the underlying
  message rather than a misleading "compute_type not supported" retry.
- Added 3 new tests in `TestEngineSelection`: `_looks_like_ssl_error`
  positive/negative cases (including chained causes),
  `_load_faster_whisper_with_ssl_fallback` retries once on SSL error and
  succeeds, and it does **not** retry on unrelated errors. Suite:
  **68 tests passing**.

**Verification**: The `huggingface_hub` version in the workspace's venv
is 1.22.0, which does *not* expose `configure_http_backend` — that's
what would have made a simpler fix silently break. The final
implementation still works because the `requests.Session.__init__`
patch is the primary hammer.

**Lesson**: `huggingface_hub`'s HTTP configuration API changed between
0.x and 1.x. Any workaround that relies on `configure_http_backend`
alone is fragile; patching `requests.Session` directly is the more
durable approach and it's also what actually crosses the wire for every
model file download.

### 2.31 Documentation update

> **User**: Requested that the development-process file capture the SSL
> fallback fix.

**Action**: Added the initial SSL-fallback section to this file.

### 2.32 SSL fallback still failing on huggingface_hub 1.x

> **User** *(pasted log)*: The SSL retry printed
> "Retrying with certificate verification DISABLED…" and then
> immediately failed again with the same
> `ConnectError: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify
> failed: self-signed certificate in certificate chain` from
> `huggingface_hub`.

**Root cause**: `huggingface_hub` 1.x no longer uses `requests`. Every
HTTP call goes through an `httpx.Client` created by an internal factory.
The previous fix patched `requests.Session.__init__` — which
`huggingface_hub` 1.x never touches. Confirmed by grepping the installed
package:

```
huggingface_hub\file_download.py: import httpx
huggingface_hub\hf_api.py:        import httpx
huggingface_hub\utils\_http.py:   return httpx.Client(...)
```

`huggingface_hub` 1.x exposes the correct override point:
`huggingface_hub.set_client_factory(factory)` and
`set_async_client_factory(factory)`, both of which internally call
`close_session()` before swapping the factory so the next
`get_session()` builds a fresh client from ours.

**Actions**:

- Rewrote `_disable_hf_ssl_verification()` in
  [censor_transcribe.py](censor_transcribe.py) to install **every**
  override we know about, unconditionally:
  1. **`huggingface_hub.set_client_factory`** (1.x path) — returns an
     `httpx.Client(verify=False, event_hooks=..., follow_redirects=True,
     timeout=None)` matching the library's own defaults so behavior is
     unchanged except for TLS verification.
  2. **`huggingface_hub.set_async_client_factory`** — the async
     equivalent.
  3. **`huggingface_hub.configure_http_backend`** (0.x path) —
     harmless when absent.
  4. Patch `requests.Session.__init__` so any stray `requests`-based
     client also gets `verify=False`.
  5. Set `HF_HUB_DISABLE_SSL_VERIFY=1` and `CURL_CA_BUNDLE=""`
     environment hints.
  6. Silence `urllib3.InsecureRequestWarning` and the
     `Unverified HTTPS request` warning from `httpx`.
- The default event hooks (`hf_request_event_hook`, async pair) are
  imported from `huggingface_hub.utils._http` and reused so the client
  still behaves like the library's real one — minus the TLS check.
- All 68 tests still pass; runtime smoke test confirms
  `set_client_factory` is accepted and `huggingface_hub.get_session()`
  returns a fresh `httpx.Client` built by our factory.

**Lesson (updated)**: Corporate-SSL workarounds must target the HTTP
client the library actually uses. `huggingface_hub` moved from
`requests` to `httpx` between 0.x and 1.x; patching `requests` is now
insufficient. The right hammer is `set_client_factory` when it exists,
and patching multiple HTTP layers as belt-and-suspenders is cheap
insurance against future API churn.

### 2.33 Move engine selector; drop install hint

> **User**: "In the GUI, please move transcription engine selection to
> be the first line in the settings section. Also, remove the note on
> how to install faster-whisper from the GUI."

**Action**: Rearranged the `Settings` `LabelFrame` grid so the
`Transcription engine:` combobox is on row 0, followed by
`Whisper model / Device / Pre-pad / Post-pad / Censor method` on row 1,
transcript-cache controls on row 2, and the `Audio delay (ms):` spinbox
on row 3. The inline hint next to the engine selector now reads
`"'Faster Whisper' uses CTranslate2 for ~4-8x faster transcription"` —
the `pip install faster-whisper` half was removed.

### 2.34 Documentation update

> **User**: Requested that the development-process file record the
> layout change and the second SSL fix.

**Action**: Added sections 2.32 and 2.33 to this file.

### 2.35 Corporate proxy blocks the HuggingFace CDN entirely

> **User** *(pasted log)*: After the httpx-based TLS override took
> effect and printed *"Disabled TLS verification for huggingface_hub for
> this session"*, the download still failed — this time with
> `[WinError 10054] An existing connection was forcibly closed by the
> remote host`.

**Root cause**: The TLS-verify workaround now succeeds, but the
corporate proxy is *actively* refusing to let this workstation reach
`huggingface.co` — it kills the TCP connection after the handshake.
No amount of client-side tweaking can fix that. The user needs a way to
supply the model files out-of-band.

**Actions**:

- New `_looks_like_connection_reset(exc)` helper in
  [censor_transcribe.py](censor_transcribe.py) that recognizes
  `WinError 10054`, `ConnectionResetError`, "Connection refused",
  "Connection aborted", "Remote end closed connection", etc.
- New `model_path` parameter threaded through `_get_or_load_model`,
  `preload_model`, and `transcribe`. When non-empty, it is used
  **verbatim** as the identifier passed to the backend:
  - For faster-whisper it may be a local directory (containing
    `model.bin` / `config.json` / `tokenizer.json` / etc.) or a
    fully-qualified HF repo id like `Systran/faster-whisper-medium`.
  - For openai-whisper it may be a downloaded `.pt` checkpoint.
- Model cache key is now `(engine, model_path or model_size, device)`
  so switching between a size and a path (or between two paths) cleanly
  evicts the previous model.
- Transcript-cache identifier follows the same rule: the pipeline calls
  `load_cache` / `save_cache` with `cfg.model_path.strip() or
  cfg.model_size`. Switching between a well-known size and a local
  path invalidates the cache naturally.
- When the *initial* load raises a connection-reset error and
  `model_path` is empty, `_get_or_load_model` logs a numbered
  workaround hint pointing the user at the new "Model path" GUI field,
  `HF_ENDPOINT`, or switching to openai-whisper (whose model files
  come from a different CDN).
- The `compute_type` fallback in `_get_or_load_model` was updated to
  *also* re-raise on `_looks_like_connection_reset`, so a network
  failure isn't misidentified as a "compute_type not supported" issue.
- GUI ([video_censor_gui.py](video_censor_gui.py)):
  - New `var_model_path = tk.StringVar(value="")`.
  - New row 1 in the Settings grid: "Model path (optional):" text
    entry, **Browse…** button (folder picker), and inline hint
    ("Local model folder or HF repo id (e.g.
    'Systran/faster-whisper-medium'). Overrides Whisper model when
    set."). Every subsequent row shifted down by 1 (model/device/pad on
    row 2, cache on row 3, audio delay on row 4).
  - `_browse_model_path()` method added.
  - Wired into `_build_config_for()`, `_settings_dict()`,
    `_load_settings()`, the `preload_model()` call in the batch
    worker, and the `_refresh_cache_status()` trace.
- Documentation ([VIDEO_CENSOR_README.md](VIDEO_CENSOR_README.md)):
  New item 12 in the settings walkthrough explains the field and its
  intended use case.
- Tests ([tests/test_video_censor.py](tests/test_video_censor.py)):
  - `test_looks_like_connection_reset` — recognizes WinError 10054,
    `ConnectionResetError`, `httpx.ConnectError: Connection refused`;
    does not fire on cert-verify messages or unrelated errors.
  - `test_model_path_overrides_size_in_cache_key` — stubs the
    faster-whisper loader and verifies that (a) `model_path` is passed
    to `WhisperModel` as the identifier, (b) two different paths
    produce two evict-and-reload cycles, and (c) clearing the path
    falls back to `model_size`.
  - Suite now: **70 tests passing**.

**Workflow for the user**: On any network with HuggingFace access, run
something like:

```python
from huggingface_hub import snapshot_download
snapshot_download("Systran/faster-whisper-medium", local_dir=r"C:\models\fw-medium")
```

Copy the resulting folder to the corporate workstation and paste that
path into the new **Model path** field. faster-whisper will load
directly from disk, no HTTPS involved.

**Lesson**: There's a class of failure (proxy-blocked CDNs) that can't
be fixed client-side no matter how many workarounds you install. The
right product move is to expose a way for the user to hand in the
artifact directly and to give a clear, actionable error message when
the automated download fails.

### 2.36 Documentation update

> **User**: Requested a way to work around the proxy blocking the
> HuggingFace CDN, after the SSL override succeeded but the connection
> was still refused.

**Action**: Documented the model-path workaround in this file and in
the README.

### 2.37 Sideloaded model verified — 3.5× speed-up

> **User**: "I can confirm that using a sideloaded medium model file is
> working correctly with faster-whisper, and the model is running about
> 3.5× faster than the medium model on openai-whisper."

**Outcome**: End-to-end verification of the workaround chain that took
five prompts to build up (SSL fallback → engine selector → SSL retry
for httpx-based hf_hub 1.x → model-path override). The user
pre-downloaded the `Systran/faster-whisper-medium` files on an
unrestricted network, copied the folder to the corporate workstation,
pasted the path into the new **Model path** GUI field, and got a 3.5×
speed-up over openai-whisper on the same hardware — matching the low
end of faster-whisper's published GPU benchmarks (4–8×).

**No code changes were needed for this milestone**; it just confirms
the design landed correctly.

**Design decisions validated**:

- `model_path` used verbatim as the identifier passed to the backend
  (no path/repo-id inference needed on our side).
- Cache identifier follows the same rule, so switching between a size
  and a path doesn't accidentally reuse a cache produced by the other
  identifier.
- Single-slot module-level model cache still holds when the identifier
  is a filesystem path — no per-batch reload cost.

### 2.38 Documentation update (this prompt)

> **User** *(this prompt)*: Confirmed the sideloaded model works and
> reported the 3.5× speed-up.

**Action**: This section.

### 2.39 Output audio mode: replace vs. add as additional track

> **User**: "I'd like to give the user the choice of creating a whole
> new output file, or just adding the censored audio stream as an
> additional audio option. If the container type does not support
> multiple audio streams and that option is selected, then change the
> output container type to mp4."

**Design**: One new `PipelineConfig.output_mode` field with two values:

- `"replace"` — legacy behavior: the censored audio replaces the
  original (single audio stream, container unchanged).
- `"add_track"` — the original audio streams are copied through
  unchanged and the censored audio is appended as a new AAC track,
  tagged `title=Censored language=eng`, with the original remaining
  the default track.

**Container retargeting**: A hardcoded set of container extensions
known to not reliably hold multiple AAC audio streams triggers an
automatic swap of the output extension to `.mp4` when `add_track` is
chosen. Initial set: `.webm .ogv .ogg .oga .flv .3gp .gif`. The user
followed up asking to add `.avi` because its multi-audio support is
limited and player compatibility is inconsistent — done.

**Implementation**:

- New public API in [censor_audio_io.py](censor_audio_io.py):
  `OUTPUT_MODE_REPLACE`, `OUTPUT_MODE_ADD_TRACK`,
  `container_supports_multiple_audio`, `adjust_output_path_for_mode`,
  `probe_audio_stream_count`, and `build_censor_ffmpeg_args`
  (extracted from `run_censor_encode` so the argv can be unit-tested
  without launching ffmpeg).
- `MuteJobConfig` and `PipelineConfig` both gained an `output_mode`
  field. `PipelineConfig.resolve()` normalizes bogus values back to
  `"replace"` and calls `adjust_output_path_for_mode` so a `.webm`
  input with `add_track` automatically produces `.mp4` output.
- The `add_track` ffmpeg pattern uses
  `-map 0:v -map 0:a? -map [cens]` plus per-stream options
  `-c:a:N aac -b:a:N 192k -metadata:s:a:N title=Censored
  -metadata:s:a:N language=eng -disposition:a:N 0` where `N` is the
  number of source audio streams (so the new track lands after the
  originals, and the disposition flag ensures the original stays
  default).
- The beep filter builder emits `[aout]` as its output label; in
  `add_track` mode we string-replace that with `[cens]` so the same
  builder can feed either the single-track or additional-track
  encoder command.
- New GUI row in [video_censor_gui.py](video_censor_gui.py) with two
  radios ("Replace audio" default; "Add as additional track (keep
  original)"). Wired into `_build_config_for`, `_settings_dict`,
  `_load_settings`, and the per-file log-file banner.
- 12 new tests in `TestOutputMode` covering: extension-swap rules,
  container-support classification, `PipelineConfig.resolve()`
  behavior (`add_track` swap, `replace` preservation, default,
  invalid-value normalization), replace-mode argv shape, add-track
  argv structure (mute + beep), and rejection of bogus `output_mode`
  values.

**Suite growth**: 70 → 82 tests, all passing.

### 2.40 Compact log panel

> **User**: "can you reduce the number of lines shown in the log
> before scrolling to 6 to make the interface a little more compact?"

**Action**: Changed the `ScrolledText` `height` on `self.txt_log`
from 12 to 6 rows in [video_censor_gui.py](video_censor_gui.py). The
log is virtualized/scrollable so no messages are lost — only the
viewport shrinks. Frees roughly six lines of vertical real estate for
the queue, settings, and word-list panels.

### 2.41 Documentation update (this prompt)

> **User** *(this prompt)*: Requested that the README and PROCESS
> markdown files be brought up to date with the latest changes.

**Action**: Added README entries for the output-mode radio and
container retargeting; documented sections 2.39, 2.40, and this
section here; corrected the stale test-count line in the architecture
summary.

### 2.42 Compact word-list editor

> **User**: "please reduce the number of lines showing in the Censor
> Word List to 4, again for compactness."

**Action**: Changed the `ScrolledText` `height` on `self.txt_wordlist`
from 8 to 4 rows in [video_censor_gui.py](video_censor_gui.py). Same
rationale as 2.40 — the editor is scrollable so no entries are lost,
this just shrinks the viewport to give more vertical room to the
queue and settings panels.

### 2.43 Documentation update (this prompt)

> **User** *(this prompt)*: Requested that the README and PROCESS
> markdown files be brought up to date with the latest changes.

**Action**: Added section 2.42 (word-list height) and this section
here to the PROCESS doc. The README has no user-visible change to
document for these viewport tweaks.

---

## 3. Architecture summary

### 3.1 Module responsibilities

| File | Purpose |
|---|---|
| [video_censor_gui.py](video_censor_gui.py) | Tkinter GUI, batch queue, threading, log/progress event pump |
| [video_censor_pipeline.py](video_censor_pipeline.py) | Orchestrate probe → extract → transcribe → match → encode; rolling-window ETA; per-entry tally; audio-delay pass-through; output-mode (`replace` vs `add_track`) with automatic `.mp4` retargeting for containers that can't hold multiple audio streams |
| [censor_audio_io.py](censor_audio_io.py) | FFmpeg / FFprobe wrappers with progress parsing; mute/beep filter builders with optional audio delay; `build_censor_ffmpeg_args` argv builder for both output modes; `probe_audio_stream_count` |
| [censor_transcribe.py](censor_transcribe.py) | Whisper wrapper: engine dispatch (openai-whisper / faster-whisper), device auto-detect, model cache keyed by `(engine, size, device)`, tqdm hook, PATH injection, interpolator |
| [censor_wordlist.py](censor_wordlist.py) | Word list I/O; matcher with whole-word derivatives, wildcards, phrases; NLTK data download |
| [censor_timestamps.py](censor_timestamps.py) | Interval padding, merging, phrase-aware span building |
| [censor_cache.py](censor_cache.py) | Transcript sidecar cache (JSON at `<input>.transcript.json`); engine-aware |
| [censor_words.txt](censor_words.txt) | Default editable censor list |
| [requirements_video_censor.txt](requirements_video_censor.txt) | Python dependencies |
| [tests/test_video_censor.py](tests/test_video_censor.py) | Unit tests (82 as of the final revision) |

### 3.2 Progress phase weights

Full run (no cache hit):

| Phase | Weight |
|---|---|
| probe | 1 % |
| extract | 4 % |
| transcribe | 85 % |
| match | 2 % |
| encode | 8 % |

Cache-hit run:

| Phase | Weight |
|---|---|
| probe | 2 % |
| match | 8 % |
| encode | 90 % |

### 3.3 Whisper progress hook (final form)

1. Use `importlib.import_module("whisper.transcribe")` then
   `sys.modules["whisper.transcribe"]` to get the **real module** (not
   the shadowed function).
2. Replace `module.tqdm` with
   `types.SimpleNamespace(tqdm=_ProxyTqdm)` so whisper's
   `tqdm.tqdm(total=content_frames)` yields our proxy.
3. The proxy:
   - Delegates every call to a real `tqdm.tqdm` so the terminal bar keeps
     working.
   - Emits progress to our callback with a **monotonic clamp**.
   - Raises `_WhisperCancelled` on cancel.
4. A background **interpolator thread** wakes every 250 ms; once two
   real ticks have been seen it extrapolates linearly at 85 % of the
   observed rate, capped at 90 % of the way to the next expected tick.
5. A **fallback pinger** starts only after 45 s of silence and caps at
   20 % of the phase, so it can never race ahead of the real hook.
6. **ETA** uses a 60 s rolling window of `(wall_time, fraction)` samples
   so early fast ticks don't distort the estimate.

---

## 4. Test summary

**Final test count: 70 (all passing)**.

Grouped by class:

| Class | Tests |
|---|---|
| `TestNormalizeToken` | punctuation stripping, possessives, unicode-ish edges |
| `TestParseWordlistText` | comments, blanks, wildcards, phrases |
| `TestWordlistSaveLoad` | roundtrip, missing file |
| `TestWordMatcher` | Scunthorpe, derivatives, direct match |
| `TestWildcardModes` | substring / prefix / suffix, whole-word safety |
| `TestPhraseMatching` | consecutive-only, case-insensitive, longest wins, mixed with singles, punctuation, interval span |
| `TestBuildMuteIntervals` | always-matcher, none-matcher, padding, clamping, overlap merge |
| `TestFlattenTranscriptWords` | segment/word structure normalization |
| `TestResolveDevice` | CPU / CUDA / Auto under monkey-patched torch |
| `TestTranscriptCache` | missing, roundtrip, mtime/size/model invalidation, clear |
| `TestEngineSelection` | `AVAILABLE_ENGINES`, engine-name normalization, cache backward-compat, cache engine mismatch, faster-whisper adapter shape, faster-whisper mid-stream cancel, SSL-error detection, SSL retry succeeds, non-SSL error is not retried, connection-reset detection, model_path overrides model_size in cache key |
| `TestAudioDelayFilters` | mute + beep filter builders for zero / positive / negative delay, empty-intervals edge case |

---

## 5. Key lessons captured

1. **`from .foo import bar` shadows submodule access.** Never rely on
   `import pkg.submodule as x` after that pattern — use
   `sys.modules["pkg.submodule"]` when you need the real module for
   monkey-patching.
2. **Cooperative cancellation** inside blocking C-level calls needs a
   real Python interception point. For whisper, tqdm's `update()` is the
   only one. Using `BaseException` for the sentinel prevents `except
   Exception` from swallowing it.
3. **Rolling-window ETA** is dramatically better than
   `elapsed / progress` when the underlying rate is non-uniform (whisper
   is fast on the first few chunks, then steady).
4. **Whisper's chunk cadence** (~30 s per real `pbar.update()`) means
   progress-bar interpolation between real ticks is a big UX win.
5. **A single-slot module-level model cache** is the simplest solution
   to "load once per batch" without opening cans of worms about lifetime
   management or thread ownership.
6. **Corporate networks routinely break TLS.** Any pip-adjacent download
   (NLTK, HuggingFace, whisper model download) should fall back to
   `ssl._create_unverified_context` after the first failure.
7. **Prefer patching `requests.Session.__init__` over library-specific
   HTTP-backend APIs** when disabling TLS verification. Library APIs
   (like `huggingface_hub.configure_http_backend`) come and go across
   versions; `requests` sessions do not.

---

## 6. Final feature list (from the README)

- Word-level Whisper transcription with derivative-aware matching
  (WordNet + Porter).
- `*wildcard*`, `word*`, `*word` opt-in wildcard entries.
- **Multi-word phrase entries** (`Jesus Christ`).
- Mute or beep, per run.
- Configurable asymmetric padding (150 ms pre, 50 ms post default).
- **Audio-delay (A/V sync) adjustment** in milliseconds; positive delays
  audio (fixes audio-ahead-of-video), negative advances it. Re-runs hit
  the transcript cache and finish quickly.
- Selectable Whisper model, device auto-detect (CUDA if available).
- **Selectable transcription engine**: OpenAI Whisper (reference) or
  Faster Whisper (CTranslate2, ~4-8x speed-up on the same hardware).
  Transcripts are cached separately per engine.
- Real ETA with 60 s rolling window; smoothly-interpolated progress bar.
- **Batch queue** with per-file and per-batch elapsed timers.
- **Transcript caching** (sidecar JSON) — re-runs with a different word
  list finish in seconds.
- Whisper model loaded **once per batch**.
- Rolling log file with per-run header banners; **per-entry censor
  tally** logged after the match phase.
- Cancel button that cooperatively interrupts even inside whisper's
  blocking transcribe.
- FFmpeg folder override (works around missing PATH entries).
- NLTK data auto-download with corporate-SSL fallback.
- Output video keeps the original video stream (fast, lossless) and
  re-encodes only the audio track.
