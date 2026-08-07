# Video Audio Censor Tool

A Tkinter GUI Python tool that transcribes the audio in a video with
word-level timestamps — using either **OpenAI Whisper** or
**Faster Whisper** (CTranslate2), GPU-accelerated when available —
matches transcribed words against an editable censor list (with automatic
handling of plurals, past-tense, possessives, and other common derivatives),
and produces a copy of the video with those words muted or beeped out.

## Features

- Word-level timestamped transcription with a **selectable engine**:
  - **OpenAI Whisper** — the reference implementation.
  - **Faster Whisper** — CTranslate2 backend, roughly 4–8× faster on the
    same hardware with the same model sizes.

  Both engines are first-class options, share the same GUI settings, and
  produce interchangeable results. Transcripts are cached separately per
  engine.
- Automatic **CUDA** acceleration when available, fallback to CPU
- Editable **censor word list** stored as plain text next to the script
- Derivative-aware matching (lemma + stem hybrid via NLTK)
  - Base form `run` also matches `running`, `ran`, `runs`
  - Base form `damn` also matches `damned`, `damning`, `damns`
- **Whole-word only** by default — will *not* mute `class` because you listed `ass`
- Optional **`*wildcard*`** entries for words where you *do* want every
  compound muted (e.g., `*fuck*` catches `fucktard`, `motherfucker`, etc.)
- Choose **mute** (silence) or **beep** (1 kHz tone) per run
- Configurable asymmetric padding around each muted word
  (defaults: 150 ms **pre**, 50 ms **post**) to catch onset consonants
  without dragging silence into the next word
- **Audio delay** control (±2000 ms) to fix A/V sync issues — shift the
  output audio track forward or backward relative to the video without
  re-transcribing (the cached transcript is reused, so only the ffmpeg
  re-encode runs).
- Selectable Whisper model (`tiny` … `large-v3`, default **medium**) — works
  with both engines.
- **Output audio mode**: replace the original audio (default) *or* keep the
  original and add the censored audio as an additional track. Containers
  that can't hold multiple audio streams (`.webm`, `.ogv`, `.ogg`, `.flv`,
  `.3gp`, `.avi`, `.gif`) are automatically retargeted to `.mp4` when
  add-track mode is chosen.
- Live log panel + progress bar with real ETA
- **Batch queue** — load any number of input videos into a queue and
  process them sequentially with a single Run click. Each file gets its
  own auto-derived output (`<name>_censored.<ext>`) next to the input.
  Per-file status ("File 3 of 7: my_video.mp4") is shown above the
  progress bar, and a final summary reports how many succeeded / failed /
  were cancelled.
- **Transcript caching** — the transcription is saved next to the input
  video as a sidecar JSON file. Re-running against the same input (e.g. to
  try a different censor list, padding, or mute-vs-beep mode) skips the
  slow Whisper step entirely and finishes in seconds.
- Rolling log file (`video_censor.log`) auto-saved next to the script,
  with a per-run header banner and buttons to open/clear it from the GUI
- Cancel button that aborts the pipeline promptly
- Output video keeps the original video stream (fast, lossless) and
  re-encodes only the audio track

## Requirements

1. **Python 3.10+**
2. **ffmpeg / ffprobe** — either on your system PATH, or in a folder you
   point the GUI at via the **FFmpeg folder** field. See
   [Installation § 2](#2-install-ffmpeg--ffprobe) below for download links.
3. **CUDA-capable GPU (optional but recommended)**

## Installation

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install FFmpeg / FFprobe

The tool shells out to `ffmpeg` and `ffprobe` for audio extraction and
for the final mute/beep encode, so both binaries must be reachable.

**Windows** (recommended source):

- **gyan.dev builds** — <https://www.gyan.dev/ffmpeg/builds/>. Grab the
  "release full" 7z (`ffmpeg-release-full.7z`) or the "essentials" build,
  extract it anywhere, and either:
  - add the extracted `bin` folder to your `PATH`, **or**
  - leave it in place and paste the folder into the **FFmpeg folder**
    field in the GUI (the tool also checks a nested `bin\` subfolder).
- **BtbN builds** — <https://github.com/BtbN/FFmpeg-Builds/releases> is
  a good alternative if gyan.dev is unreachable.
- **winget** — `winget install Gyan.FFmpeg` installs to a versioned
  folder and adds it to `PATH` automatically.
- **Chocolatey** — `choco install ffmpeg-full`.

**macOS**: `brew install ffmpeg` (Homebrew).

**Linux**: install both the Tkinter runtime and FFmpeg via your distro.
For Ubuntu/Debian:

```bash
sudo apt update
sudo apt install python3-tk ffmpeg
```

For Fedora:

```bash
sudo dnf install python3-tkinter ffmpeg
```

If you prefer a source-based or alternative Linux package, the key
requirements are the same: `python3-tk` (Tkinter runtime for the GUI) and
`ffmpeg`/`ffprobe` on PATH.

Verify from a shell:

```powershell
ffmpeg -version
ffprobe -version
```

If either command prints "not recognized" / "command not found", the
binary isn't on your `PATH` — use the **FFmpeg folder** field in the GUI
to point at the folder that contains `ffmpeg` and `ffprobe`
instead. The **Detect** button next to that field confirms both binaries
resolve.

### 3. Install PyTorch

You **must** install `torch` before the rest of the requirements, so pip
picks up the CUDA (or CPU) build you want instead of a default from PyPI.

The correct index URL depends on **your Python version** and **your GPU
driver's CUDA version** (check with `nvidia-smi`).

| Your setup | Command |
|---|---|
| GPU, CUDA 12.4 (most modern NVIDIA drivers) | `pip install torch --index-url https://download.pytorch.org/whl/cu124` |
| GPU, CUDA 12.6 (newest drivers) | `pip install torch --index-url https://download.pytorch.org/whl/cu126` |
| GPU, CUDA 12.1 (older drivers, Python 3.12 or below) | `pip install torch --index-url https://download.pytorch.org/whl/cu121` |
| CPU only | `pip install torch --index-url https://download.pytorch.org/whl/cpu` |

> **Python 3.13 users:** the `cu121` index does not publish wheels for
> Python 3.13. Use `cu124` or `cu126` instead.

If pip prints
`ERROR: Could not find a version that satisfies the requirement torch`,
your Python version is not covered by that index URL — pick a different
one from the table above, or use the CPU build.

For an always-current lookup, use the official PyTorch install selector at
<https://pytorch.org/get-started/locally/>.

### 4. Install the remaining requirements

```powershell
pip install -r requirements_video_censor.txt
```

On first run the tool downloads a couple of small NLTK data packages
(`wordnet`, `omw-1.4`) automatically. Whisper will also download its
model weights on first use.

### If the NLTK download fails

On corporate networks or behind SSL-inspecting proxies, the auto-download
may fail with a `Resource 'wordnet' not found` error. Fixes, in order:

1. In the GUI, click **Download NLTK data**. It first tries a normal
   download, then retries with an unverified SSL context (which usually
   defeats corporate TLS interception).
2. From a shell with unrestricted internet access, run:
   ```powershell
   python -m nltk.downloader wordnet omw-1.4
   ```
3. Manual install as a last resort — download the two zip files from
   <https://www.nltk.org/nltk_data/> and extract them so you end up with
   these folders:
   ```
   ~/.nltk/corpora/wordnet/
   ~/.nltk/corpora/omw-1.4/
   ```
   (Windows: `%APPDATA%\nltk_data\corpora\...` also works)

## Usage

```powershell
python video_censor_gui.py
```

1. **Input video** — Browse to select the source file.
2. **Output video** — Auto-fills to `<inputname>_censored.<ext>`; edit if desired.
3. **Censor words file** — path to the editable list; the editor pane below shows its contents.
4. **FFmpeg folder** *(optional)* — leave blank to use PATH. Otherwise pick
   the folder that contains `ffmpeg` and `ffprobe` (a `bin/`
   subfolder inside it is also checked). The **Detect** button verifies the
   setting and reports where the binaries were found.
5. **Whisper model** — default `medium`. Larger = more accurate, slower.
6. **Device** — `Auto` picks CUDA if available.
7. **Censor method** — `Mute` (silence) or `Beep` (1 kHz tone).
8. **Pre-pad (ms)** — extra time added to the leading edge of each muted
   region (default 150 ms). Larger values catch consonants Whisper often
   clips off the start of a word.
9. **Post-pad (ms)** — extra time added to the trailing edge (default 50 ms).
10. **Audio delay (ms)** — shifts the entire output audio track relative to
    the video. Positive values *delay* the audio (pad silence at the start;
    use this when the audio track plays ahead of the video). Negative values
    *advance* the audio (trim the start). Default `0`. If the first run
    shows a small A/V drift, set this and re-run — the cached transcript
    will be reused so only the ffmpeg re-encode runs.
11. **Transcription engine** — choose between:
    - **OpenAI Whisper** *(default)* — the reference implementation.
      Slower but always available.
    - **Faster Whisper** — uses [CTranslate2](https://github.com/OpenNMT/CTranslate2)
      for roughly 4-8x faster transcription with the same model sizes.
      Transcripts are cached separately per engine (a cache produced by
      one engine is not reused by the other).
12. **Model path (optional)** — a local folder containing a downloaded
    Whisper / faster-whisper model, or a HuggingFace repo id such as
    `Systran/faster-whisper-medium`. When set, it overrides the Whisper
    model dropdown as the identifier passed to the transcription
    backend. Use this when your network blocks direct HuggingFace
    downloads: pre-download the model on an unrestricted network, copy
    the folder over, and point this field at it. For faster-whisper the
    folder must contain `model.bin`, `config.json`, `tokenizer.json`,
    etc.; for openai-whisper it can be a downloaded `.pt` checkpoint.
13. **Output audio** — choose how the censored audio track is written:
    - **Replace audio** *(default)* — the original audio is replaced by
      the censored version (single audio stream, unchanged container).
    - **Add as additional track (keep original)** — the original audio
      streams are copied through unchanged, and the censored audio is
      appended as a new AAC track tagged `title=Censored language=eng`.
      The original remains the default track. If the input container
      does not reliably support multiple audio streams (`.webm`,
      `.ogv`, `.ogg`, `.flv`, `.3gp`, `.avi`, `.gif`), the output is
      automatically retargeted to `.mp4`.
14. Press **Run**. Progress bar and log update live.
    Use **Cancel** to abort at any point.

## Censor list format

Plain text, one word per line, `#` starts a comment. Matching is
case-insensitive.

**Match modes** — controlled by `*` wildcards in the entry itself:

| Entry | Matches | Does NOT match |
|---|---|---|
| `damn` (whole word + derivatives) | `damn`, `Damn!`, `damned`, `damning`, `damns` | `goddamn`, `damnation` |
| `*fuck*` (substring anywhere) | `fuck`, `fucked`, `fucking`, `fucktard`, `motherfucker`, `clusterfuck` | (very few — this is the widest mode) |
| `fuck*` (prefix) | `fuck`, `fucker`, `fucktard` | `motherfucker` |
| `*fuck` (suffix) | `fuck`, `motherfuck`, `clusterfuck` | `fucker` |
| `Jesus Christ` (phrase) | consecutive spoken `Jesus Christ` (any case) | bare `Jesus`, bare `Christ`, `Jesus loves Christ` |

- **Phrase entries** — any entry containing two or more whitespace-separated
  words is treated as a phrase. Each sub-word still supports the whole-word
  derivative logic (plurals, tenses, possessives), so `holy cow` also
  matches `holy cows`. Wildcards inside phrases are not supported (write
  the phrase in full, or list each variant).
- Whole-word entries are **Scunthorpe-safe**: listing `ass` will *not*
  affect `class`, `grass`, `pass`, or `brass`.
- Wildcard entries opt out of that safety, so use them deliberately.
  Listing `*ass*` **will** mute `class`, `grass`, `passage`, etc.
- **Wildcard entries do NOT get automatic derivative handling.** They are
  pure lowercased string matches (substring / prefix / suffix on the
  literal core) — NLTK lemmas and Porter stems are only consulted for
  whole-word entries. Concretely: `*fuck` (suffix) matches `fuck`,
  `motherfuck`, `clusterfuck` but *not* `fucked`, `fucking`, or `fucker`,
  because those don't end in the literal string `fuck`. If you want a
  wildcard to catch derivatives too, use the substring form (`*fuck*`
  catches `fuck`, `fucked`, `fucking`, `fucker`, `motherfucker`, etc. via
  substring matching, without needing WordNet).
- Blank lines and `#` comment lines are ignored.
- You only need the base form for whole-word entries; derivatives (plural,
  past tense, `-ing`, possessive) are handled automatically.

## Files

| File | Purpose |
|---|---|
| `video_censor_gui.py` | Tkinter GUI entry point |
| `video_censor_pipeline.py` | Orchestrates extract → transcribe → filter → mute |
| `censor_audio_io.py` | ffmpeg / ffprobe wrappers |
| `censor_transcribe.py` | Whisper wrapper with device auto-detect |
| `censor_wordlist.py` | Word list I/O and derivative-aware matcher |
| `censor_timestamps.py` | Interval padding and merging |
| `censor_cache.py` | Transcript sidecar cache (load / save / validity) |
| `censor_words.txt` | Editable default censor list |
| `requirements_video_censor.txt` | Python dependencies |
| `tests/test_video_censor.py` | Unit tests |

## Limitations / non-goals

- English only.
- No subtitle export.
- Video stream is not re-encoded — output container matches the input,
  except when **Add as additional track** is selected with a container
  that can't hold multiple audio streams (see item 13 above), in which
  case the output is written as `.mp4`.
- Beep/mute is per-word; there is no manual timeline editor.
