"""Tkinter GUI for the video audio censor tool.

Run:  python video_censor_gui.py
"""

from __future__ import annotations

import json
import os
import queue
import sys
import threading
import time
import tkinter as tk
from dataclasses import asdict
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText
from typing import Optional

from censor_audio_io import safe_default_output_path
from censor_transcribe import AVAILABLE_ENGINES, AVAILABLE_MODELS
from video_censor_pipeline import PipelineConfig, run_pipeline


SETTINGS_FILE = "video_censor_settings.json"
# Anchor the default wordlist to the folder containing this script so it
# resolves correctly regardless of the current working directory at launch.
_SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_WORDLIST = str(_SCRIPT_DIR / "censor_words.txt")
LOG_FILE = "video_censor.log"
DEFAULT_WORDLIST_HEADER = (
    "# Censor Words List\n"
    "# One word per line. Lines starting with '#' are comments.\n"
    "# Matching is case-insensitive.\n"
    "#\n"
    "# Match modes:\n"
    "#   word              whole word + derivatives (plurals, -ed, -ing, possessive)\n"
    "#   *word*            substring anywhere (catches compounds; use carefully)\n"
    "#   word*             token starts with word\n"
    "#   *word             token ends with word\n"
    "#   word1 word2 ...   phrase (matches only when these words appear\n"
    "#                     consecutively; each sub-word supports derivatives)\n"
)


def _to_windows_path(p: str) -> str:
    return str(p).replace("/", os.sep)


class VideoCensorGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        root.title("VidClean Audio Censor Tool")
        root.geometry("1116x912")
        root.minsize(900, 700)

        self._log_queue: "queue.Queue[tuple[str, object]]" = queue.Queue()
        self._worker: Optional[threading.Thread] = None
        self._cancel_flag = threading.Event()
        self._run_start_ts: Optional[float] = None
        self._file_start_ts: Optional[float] = None
        self._last_eta: Optional[float] = None

        # Batch state (populated when a queue run starts).
        self._queue_paths: list[str] = []
        self._queue_index: int = 0
        self._queue_results: list[tuple[str, str, Optional[BaseException]]] = []  # (input, output_or_msg, exc)

        # --- Variables --------------------------------------------------------
        self.var_wordlist = tk.StringVar(value=_to_windows_path(DEFAULT_WORDLIST))
        self.var_ffmpeg_dir = tk.StringVar()
        self.var_model = tk.StringVar(value="medium")
        self.var_device = tk.StringVar(value="Auto")
        self.var_engine = tk.StringVar(value="OpenAI Whisper")
        self.var_model_path = tk.StringVar(value="")
        self.var_mode = tk.StringVar(value="mute")
        self.var_pre_pad_ms = tk.IntVar(value=150)
        self.var_post_pad_ms = tk.IntVar(value=50)
        self.var_audio_delay_ms = tk.IntVar(value=0)
        self.var_output_mode = tk.StringVar(value="replace")
        self.var_use_cache = tk.BooleanVar(value=True)
        self.var_status = tk.StringVar(value="Idle.")
        self.var_phase = tk.StringVar(value="")
        self.var_queue_status = tk.StringVar(value="")
        self.var_elapsed_file = tk.StringVar(value="File: 0.0 s")
        self.var_elapsed_batch = tk.StringVar(value="Batch: 0.0 s")
        self.var_eta = tk.StringVar(value="ETA: --")

        self._build_ui()
        self._load_settings()
        self._reload_wordlist_into_editor()
        # Update the cache status label whenever selection or model changes.
        self.var_model.trace_add("write", lambda *_: self._refresh_cache_status())
        self.var_engine.trace_add("write", lambda *_: self._refresh_cache_status())
        self.var_model_path.trace_add("write", lambda *_: self._refresh_cache_status())
        self._refresh_cache_status()
        self.root.after(100, self._drain_log_queue)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------ UI --
    def _build_ui(self) -> None:
        root = self.root

        # --- Batch queue ----------------------------------------------------
        paths = ttk.LabelFrame(root, text="Input videos (batch queue)")
        paths.pack(fill="x", padx=8, pady=(8, 4))

        # Listbox with scrollbar for queued video paths. Fixed height (9 lines)
        # so this frame doesn't grab vertical space at the expense of the log.
        list_frame = ttk.Frame(paths)
        list_frame.grid(row=0, column=0, columnspan=2, sticky="ew", padx=6, pady=(6, 4))
        self.lst_queue = tk.Listbox(
            list_frame,
            height=9,
            selectmode="extended",
            font=("Consolas", 9),
            activestyle="dotbox",
        )
        self.lst_queue.pack(side="left", fill="x", expand=True)
        sb = ttk.Scrollbar(list_frame, orient="vertical", command=self.lst_queue.yview)
        sb.pack(side="right", fill="y")
        self.lst_queue.config(yscrollcommand=sb.set)
        self.lst_queue.bind("<<ListboxSelect>>", lambda _e: self._refresh_cache_status())

        # Right-side buttons for queue management.
        q_btns = ttk.Frame(paths)
        q_btns.grid(row=0, column=2, sticky="n", padx=6, pady=(6, 4))
        ttk.Button(q_btns, text="Add files...", command=self._queue_add_files).pack(fill="x")
        ttk.Button(q_btns, text="Remove selected", command=self._queue_remove_selected).pack(fill="x", pady=(4, 0))
        ttk.Button(q_btns, text="Move up", command=lambda: self._queue_move(-1)).pack(fill="x", pady=(4, 0))
        ttk.Button(q_btns, text="Move down", command=lambda: self._queue_move(1)).pack(fill="x", pady=(4, 0))
        ttk.Button(q_btns, text="Clear queue", command=self._queue_clear).pack(fill="x", pady=(4, 0))

        ttk.Label(paths, text="Censor words file:").grid(row=2, column=0, sticky="w", padx=6, pady=4)
        ent_wl = ttk.Entry(paths, textvariable=self.var_wordlist)
        ent_wl.grid(row=2, column=1, sticky="ew", padx=6, pady=4)
        btn_row = ttk.Frame(paths)
        btn_row.grid(row=2, column=2, padx=6)
        ttk.Button(btn_row, text="Browse...", command=self._browse_wordlist).pack(side="left")
        ttk.Button(btn_row, text="Reload", command=self._reload_wordlist_into_editor).pack(side="left", padx=(4, 0))

        ttk.Label(paths, text="FFmpeg folder (optional):").grid(row=3, column=0, sticky="w", padx=6, pady=4)
        ent_ff = ttk.Entry(paths, textvariable=self.var_ffmpeg_dir)
        ent_ff.grid(row=3, column=1, sticky="ew", padx=6, pady=4)
        ff_btns = ttk.Frame(paths)
        ff_btns.grid(row=3, column=2, padx=6)
        ttk.Button(ff_btns, text="Browse...", command=self._browse_ffmpeg_dir).pack(side="left")
        ttk.Button(ff_btns, text="Detect", command=self._detect_ffmpeg).pack(side="left", padx=(4, 0))

        paths.columnconfigure(1, weight=1)

        # --- Settings row ---------------------------------------------------
        settings = ttk.LabelFrame(root, text="Settings")
        settings.pack(fill="x", padx=8, pady=4)

        # First settings row: transcription engine.
        ttk.Label(settings, text="Transcription engine:").grid(row=0, column=0, sticky="w", padx=6, pady=4)
        cmb_engine = ttk.Combobox(
            settings,
            textvariable=self.var_engine,
            values=("OpenAI Whisper", "Faster Whisper"),
            state="readonly",
            width=16,
        )
        cmb_engine.grid(row=0, column=1, columnspan=2, sticky="w", padx=6)
        ttk.Label(
            settings,
            text="'Faster Whisper' uses CTranslate2 for ~4-8x faster transcription",
            foreground="#555",
        ).grid(row=0, column=3, columnspan=8, sticky="w", padx=6)

        # Second settings row: optional model path / HF repo id override.
        ttk.Label(settings, text="Model path (optional):").grid(row=1, column=0, sticky="w", padx=6, pady=4)
        ent_model_path = ttk.Entry(settings, textvariable=self.var_model_path, width=48)
        ent_model_path.grid(row=1, column=1, columnspan=6, sticky="ew", padx=6)
        ttk.Button(
            settings,
            text="Browse...",
            command=self._browse_model_path,
        ).grid(row=1, column=7, sticky="w", padx=(0, 6))
        ttk.Label(
            settings,
            text="Local model folder. Overrides Whisper model when set.",
            foreground="#555",
        ).grid(row=1, column=8, columnspan=3, sticky="w", padx=6)

        # Third settings row: model / device / padding / censor method.
        ttk.Label(settings, text="Whisper model:").grid(row=2, column=0, sticky="w", padx=6, pady=4)
        cmb_model = ttk.Combobox(
            settings,
            textvariable=self.var_model,
            values=AVAILABLE_MODELS,
            state="readonly",
            width=14,
        )
        cmb_model.grid(row=2, column=1, sticky="w", padx=6)

        ttk.Label(settings, text="Device:").grid(row=2, column=2, sticky="w", padx=6)
        cmb_dev = ttk.Combobox(
            settings,
            textvariable=self.var_device,
            values=("Auto", "CUDA", "CPU"),
            state="readonly",
            width=8,
        )
        cmb_dev.grid(row=2, column=3, sticky="w", padx=6)

        ttk.Label(settings, text="Pre-pad (ms):").grid(row=2, column=4, sticky="w", padx=6)
        spn_pre = ttk.Spinbox(
            settings, from_=0, to=1000, increment=10, textvariable=self.var_pre_pad_ms, width=6
        )
        spn_pre.grid(row=2, column=5, sticky="w", padx=6)

        ttk.Label(settings, text="Post-pad (ms):").grid(row=2, column=6, sticky="w", padx=6)
        spn_post = ttk.Spinbox(
            settings, from_=0, to=1000, increment=10, textvariable=self.var_post_pad_ms, width=6
        )
        spn_post.grid(row=2, column=7, sticky="w", padx=6)

        ttk.Label(settings, text="Censor method:").grid(row=2, column=8, sticky="w", padx=6)
        ttk.Radiobutton(settings, text="Mute", variable=self.var_mode, value="mute").grid(row=2, column=9, sticky="w")
        ttk.Radiobutton(settings, text="Beep", variable=self.var_mode, value="beep").grid(row=2, column=10, sticky="w")

        # Fourth settings row: transcript cache controls.
        ttk.Checkbutton(
            settings,
            text="Use cached transcript when available",
            variable=self.var_use_cache,
        ).grid(row=3, column=0, columnspan=4, sticky="w", padx=6, pady=(2, 4))
        ttk.Button(
            settings,
            text="Delete cached transcript for input",
            command=self._delete_cached_transcript,
        ).grid(row=3, column=4, columnspan=4, sticky="w", padx=6, pady=(2, 4))
        self.lbl_cache_status = ttk.Label(settings, text="", foreground="#555")
        self.lbl_cache_status.grid(row=3, column=8, columnspan=3, sticky="w", padx=6, pady=(2, 4))

        # Fifth settings row: audio delay (shift audio relative to video).
        ttk.Label(settings, text="Audio delay (ms):").grid(row=4, column=0, sticky="w", padx=6, pady=(2, 4))
        spn_delay = ttk.Spinbox(
            settings,
            from_=-2000,
            to=2000,
            increment=10,
            textvariable=self.var_audio_delay_ms,
            width=8,
        )
        spn_delay.grid(row=4, column=1, sticky="w", padx=6, pady=(2, 4))
        ttk.Label(
            settings,
            text="positive = delay audio (fix audio-ahead-of-video); negative = advance audio",
            foreground="#555",
        ).grid(row=4, column=2, columnspan=9, sticky="w", padx=6, pady=(2, 4))

        # Sixth settings row: output mode (replace vs. add as additional track).
        ttk.Label(settings, text="Output audio:").grid(row=5, column=0, sticky="w", padx=6, pady=(2, 4))
        ttk.Radiobutton(
            settings,
            text="Replace audio",
            variable=self.var_output_mode,
            value="replace",
        ).grid(row=5, column=1, columnspan=2, sticky="w", padx=6, pady=(2, 4))
        ttk.Radiobutton(
            settings,
            text="Add as additional track (keep original)",
            variable=self.var_output_mode,
            value="add_track",
        ).grid(row=5, column=3, columnspan=4, sticky="w", padx=6, pady=(2, 4))
        ttk.Label(
            settings,
            text="'Add as additional track' switches container to .mp4 for .webm/.ogv/.flv/.avi inputs.",
            foreground="#555",
        ).grid(row=5, column=7, columnspan=4, sticky="w", padx=6, pady=(2, 4))

        # --- Word list editor -----------------------------------------------
        wl_frame = ttk.LabelFrame(
            root,
            text="Censor Word List  (one per line; '#' comment; wildcards: *word* / word* / *word; multi-word phrases supported)",
        )
        wl_frame.pack(fill="both", expand=False, padx=8, pady=4)

        self.txt_wordlist = ScrolledText(wl_frame, height=4, wrap="none", font=("Consolas", 10))
        self.txt_wordlist.pack(fill="both", expand=True, padx=6, pady=6)

        wl_btns = ttk.Frame(wl_frame)
        wl_btns.pack(fill="x", padx=6, pady=(0, 6))
        ttk.Button(wl_btns, text="Save", command=self._save_wordlist).pack(side="left")
        ttk.Button(wl_btns, text="Revert", command=self._reload_wordlist_into_editor).pack(side="left", padx=(6, 0))
        ttk.Button(wl_btns, text="Save As...", command=self._save_wordlist_as).pack(side="left", padx=(6, 0))
        ttk.Button(wl_btns, text="Download NLTK data", command=self._download_nltk_data).pack(side="left", padx=(12, 0))
        self.lbl_wl_status = ttk.Label(wl_btns, text="")
        self.lbl_wl_status.pack(side="right")

        # --- Action row -----------------------------------------------------
        actions = ttk.Frame(root)
        actions.pack(fill="x", padx=8, pady=6)

        style = ttk.Style()
        try:
            style.configure("Run.TButton", foreground="white", background="#2e7d32", font=("Segoe UI", 12, "bold"))
            style.configure("Cancel.TButton", foreground="white", background="#c62828", font=("Segoe UI", 12, "bold"))
        except tk.TclError:
            pass

        self.btn_run = tk.Button(
            actions,
            text="Run",
            command=self._on_run,
            bg="#2e7d32",
            fg="white",
            activebackground="#1b5e20",
            activeforeground="white",
            font=("Segoe UI", 12, "bold"),
            width=14,
            height=1,
        )
        self.btn_run.pack(side="left")

        self.btn_cancel = tk.Button(
            actions,
            text="Cancel",
            command=self._on_cancel,
            bg="#c62828",
            fg="white",
            activebackground="#8e0000",
            activeforeground="white",
            font=("Segoe UI", 12, "bold"),
            width=14,
            height=1,
            state="disabled",
        )
        self.btn_cancel.pack(side="left", padx=(8, 0))

        # --- Progress -------------------------------------------------------
        progress = ttk.LabelFrame(root, text="Progress")
        progress.pack(fill="x", padx=8, pady=4)

        # Queue status line: "File 2 of 5:  video.mp4"
        q_row = ttk.Frame(progress)
        q_row.pack(fill="x", padx=6, pady=(6, 0))
        ttk.Label(q_row, textvariable=self.var_queue_status, font=("Segoe UI", 9, "bold")).pack(side="left")

        self.pbar = ttk.Progressbar(progress, mode="determinate", maximum=1000)
        self.pbar.pack(fill="x", padx=6, pady=(4, 2))

        row = ttk.Frame(progress)
        row.pack(fill="x", padx=6, pady=(0, 6))
        ttk.Label(row, textvariable=self.var_phase).pack(side="left")
        ttk.Label(row, textvariable=self.var_elapsed_batch).pack(side="right")
        ttk.Label(row, textvariable=self.var_elapsed_file).pack(side="right", padx=(0, 12))
        ttk.Label(row, textvariable=self.var_eta).pack(side="right", padx=(0, 12))

        status_row = ttk.Frame(progress)
        status_row.pack(fill="x", padx=6, pady=(0, 6))
        ttk.Label(status_row, textvariable=self.var_status).pack(side="left")

        # --- Log ------------------------------------------------------------
        log_frame = ttk.LabelFrame(root, text=f"Log  (also appended to {LOG_FILE})")
        log_frame.pack(fill="both", expand=True, padx=8, pady=(4, 8))
        self.txt_log = ScrolledText(log_frame, height=6, wrap="word", font=("Consolas", 9), state="disabled")
        self.txt_log.pack(fill="both", expand=True, padx=6, pady=(6, 0))
        log_btns = ttk.Frame(log_frame)
        log_btns.pack(fill="x", padx=6, pady=(2, 6))
        ttk.Button(log_btns, text="Open log file", command=self._open_log_file).pack(side="left")
        ttk.Button(log_btns, text="Clear log file", command=self._clear_log_file).pack(side="left", padx=(6, 0))

    # ------------------------------------------------------------- Actions --
    def _queue_add_files(self) -> None:
        """Prompt user for one or more input videos and append them to the queue."""
        paths = filedialog.askopenfilenames(
            title="Select input video(s)",
            filetypes=[
                ("Video files", "*.mp4 *.mov *.mkv *.avi *.webm *.m4v *.wmv"),
                ("All files", "*.*"),
            ],
        )
        if not paths:
            return
        existing = set(self._queue_items())
        for p in paths:
            wp = _to_windows_path(p)
            if wp in existing:
                continue
            self.lst_queue.insert("end", wp)
            existing.add(wp)
        self._refresh_cache_status()

    def _queue_remove_selected(self) -> None:
        sel = list(self.lst_queue.curselection())
        for idx in reversed(sel):
            self.lst_queue.delete(idx)
        self._refresh_cache_status()

    def _queue_clear(self) -> None:
        if self.lst_queue.size() == 0:
            return
        if not messagebox.askyesno("Clear queue", "Remove all queued videos?"):
            return
        self.lst_queue.delete(0, "end")
        self._refresh_cache_status()

    def _queue_move(self, direction: int) -> None:
        """Move the selected item up (-1) or down (+1) in the queue."""
        sel = list(self.lst_queue.curselection())
        if len(sel) != 1:
            return
        idx = sel[0]
        new_idx = idx + direction
        if new_idx < 0 or new_idx >= self.lst_queue.size():
            return
        item = self.lst_queue.get(idx)
        self.lst_queue.delete(idx)
        self.lst_queue.insert(new_idx, item)
        self.lst_queue.selection_clear(0, "end")
        self.lst_queue.selection_set(new_idx)
        self.lst_queue.activate(new_idx)

    def _queue_items(self) -> list[str]:
        return [self.lst_queue.get(i) for i in range(self.lst_queue.size())]

    def _selected_or_first_queue_item(self) -> Optional[str]:
        """Return the selected queue item, or the first item if none selected."""
        sel = list(self.lst_queue.curselection())
        if sel:
            return self.lst_queue.get(sel[0])
        if self.lst_queue.size() > 0:
            return self.lst_queue.get(0)
        return None

    def _browse_wordlist(self) -> None:
        # Prefer the folder of the currently selected file; else the script's
        # own directory (where the default censor_words.txt lives).
        current = self.var_wordlist.get().strip()
        if current and Path(current).parent.exists():
            initial_dir = str(Path(current).parent)
        else:
            initial_dir = str(_SCRIPT_DIR)
        path = filedialog.askopenfilename(
            title="Select censor word list",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            initialdir=initial_dir,
            initialfile=Path(current).name if current else "censor_words.txt",
        )
        if path:
            self.var_wordlist.set(_to_windows_path(path))
            self._reload_wordlist_into_editor()

    def _browse_ffmpeg_dir(self) -> None:
        initial = self.var_ffmpeg_dir.get().strip()
        path = filedialog.askdirectory(
            title="Select folder containing ffmpeg.exe and ffprobe.exe",
            initialdir=initial or "",
        )
        if path:
            self.var_ffmpeg_dir.set(_to_windows_path(path))

    def _browse_model_path(self) -> None:
        """Pick a local folder that contains a downloaded Whisper model."""
        initial = self.var_model_path.get().strip()
        path = filedialog.askdirectory(
            title="Select a local folder containing the Whisper / faster-whisper model files",
            initialdir=initial or "",
        )
        if path:
            self.var_model_path.set(_to_windows_path(path))

    def _detect_ffmpeg(self) -> None:
        """Try to locate ffmpeg using the current field or PATH."""
        import shutil

        current = self.var_ffmpeg_dir.get().strip()
        try:
            from censor_audio_io import _resolve_binary  # type: ignore[attr-defined]

            ffmpeg = _resolve_binary("ffmpeg", current or None)
            ffprobe = _resolve_binary("ffprobe", current or None)
        except Exception as exc:  # pragma: no cover - GUI feedback path
            messagebox.showwarning(
                "FFmpeg not found",
                f"Could not locate ffmpeg/ffprobe.\n\n{exc}\n\n"
                "Install ffmpeg from https://www.gyan.dev/ffmpeg/builds/ "
                "and either add its folder to PATH, or set the FFmpeg folder above.",
            )
            return

        # If it came from PATH and no explicit dir is set, tell the user where.
        parent = str(Path(ffmpeg).parent)
        messagebox.showinfo(
            "FFmpeg detected",
            f"ffmpeg:  {_to_windows_path(ffmpeg)}\n"
            f"ffprobe: {_to_windows_path(ffprobe)}",
        )
        if not current:
            # Offer to remember the location so future runs don't re-scan PATH.
            self.var_ffmpeg_dir.set(_to_windows_path(parent))
        _ = shutil  # silence unused-import checker

    def _reload_wordlist_into_editor(self) -> None:
        path = self.var_wordlist.get().strip()
        try:
            if path and Path(path).exists():
                content = Path(path).read_text(encoding="utf-8")
            else:
                content = DEFAULT_WORDLIST_HEADER
            self.txt_wordlist.delete("1.0", "end")
            self.txt_wordlist.insert("1.0", content)
            self.lbl_wl_status.config(text=f"Loaded: {_to_windows_path(path)}" if path else "New list")
        except OSError as exc:
            messagebox.showerror("Load failed", f"Could not read '{path}':\n{exc}")

    def _save_wordlist(self) -> None:
        path = self.var_wordlist.get().strip()
        if not path:
            self._save_wordlist_as()
            return
        try:
            Path(path).write_text(self.txt_wordlist.get("1.0", "end-1c"), encoding="utf-8")
            self.lbl_wl_status.config(text=f"Saved: {_to_windows_path(path)}")
        except OSError as exc:
            messagebox.showerror("Save failed", f"Could not write '{path}':\n{exc}")

    def _save_wordlist_as(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Save censor word list as...",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        )
        if not path:
            return
        self.var_wordlist.set(_to_windows_path(path))
        self._save_wordlist()

    def _download_nltk_data(self) -> None:
        """Trigger a manual NLTK data download in a background thread."""
        if self._worker and self._worker.is_alive():
            messagebox.showwarning(
                "Busy",
                "A censoring job is running. Wait for it to finish or cancel it first.",
            )
            return

        def worker() -> None:
            try:
                from censor_wordlist import ensure_nltk_data

                self._log_queue.put(("log", "Starting NLTK data download..."))
                ensure_nltk_data(log=lambda m: self._log_queue.put(("log", m)))
                self._log_queue.put(("nltk_done", None))
            except Exception as exc:  # noqa: BLE001
                self._log_queue.put(("nltk_error", exc))

        threading.Thread(target=worker, daemon=True).start()

    def _canonical_engine(self) -> str:
        """Map the GUI's human label back to the canonical engine identifier."""
        label = (self.var_engine.get() or "").strip().lower()
        if label in {"faster whisper", "faster-whisper", "faster_whisper"}:
            return "faster-whisper"
        return "openai-whisper"

    def _refresh_cache_status(self) -> None:
        """Update the label next to the cache checkbox based on the currently
        selected queue item (or the first item if none is selected)."""
        if not hasattr(self, "lbl_cache_status"):
            return
        input_video = self._selected_or_first_queue_item()
        if not input_video:
            self.lbl_cache_status.config(text="")
            return
        try:
            from censor_cache import cache_path_for, load_cache

            cp = cache_path_for(input_video)
            if not cp.exists():
                self.lbl_cache_status.config(text=f"Cache: none  ({Path(input_video).name})")
                return
            cache_identifier = self.var_model_path.get().strip() or self.var_model.get()
            cached = load_cache(input_video, cache_identifier, engine=self._canonical_engine())
            if cached is None:
                self.lbl_cache_status.config(
                    text=f"Cache: exists but stale  ({Path(input_video).name})"
                )
            else:
                self.lbl_cache_status.config(
                    text=f"Cache: valid, {len(cached.words)} words  ({Path(input_video).name})"
                )
        except Exception:  # noqa: BLE001
            self.lbl_cache_status.config(text="")

    def _delete_cached_transcript(self) -> None:
        """Delete the transcript sidecar for the currently selected queue item."""
        input_video = self._selected_or_first_queue_item()
        if not input_video:
            messagebox.showinfo("Cache", "No input video selected in queue.")
            return
        try:
            from censor_cache import cache_path_for, clear_cache

            cp = cache_path_for(input_video)
            if not cp.exists():
                messagebox.showinfo(
                    "Cache", f"No cached transcript found at:\n{_to_windows_path(str(cp))}"
                )
                self._refresh_cache_status()
                return
            if not messagebox.askyesno(
                "Delete cached transcript",
                f"Delete this file?\n{_to_windows_path(str(cp))}",
            ):
                return
            if clear_cache(input_video):
                self._append_log(f"Deleted cached transcript: {cp}")
            else:
                messagebox.showerror(
                    "Delete failed", f"Could not delete {_to_windows_path(str(cp))}"
                )
        finally:
            self._refresh_cache_status()

    # ---------------------------------------------------------------- Run --
    def _on_run(self) -> None:
        if self._worker and self._worker.is_alive():
            return

        queue_items = self._queue_items()
        if not queue_items:
            messagebox.showerror(
                "Empty queue",
                "The batch queue is empty. Add one or more input videos first.",
            )
            return

        # Validate each entry up front so we don't fail mid-batch on something
        # obvious like a missing file.
        missing = [p for p in queue_items if not Path(p).exists()]
        if missing:
            msg = "\n".join(_to_windows_path(p) for p in missing[:10])
            if len(missing) > 10:
                msg += f"\n... and {len(missing) - 10} more"
            messagebox.showerror(
                "Missing files",
                f"The following queued files do not exist:\n\n{msg}",
            )
            return

        # Persist word list edits to disk before running so the pipeline picks them up.
        wl_path = self.var_wordlist.get().strip()
        if wl_path:
            try:
                Path(wl_path).write_text(self.txt_wordlist.get("1.0", "end-1c"), encoding="utf-8")
            except OSError as exc:
                messagebox.showerror("Save failed", f"Could not save word list:\n{exc}")
                return
        else:
            messagebox.showerror("Missing word list", "Please choose or create a censor words file.")
            return

        self._save_settings()

        # Snapshot batch state.
        self._queue_paths = list(queue_items)
        self._queue_index = 0
        self._queue_results = []

        # Reset UI state.
        self._log_queue.put(("clear", None))
        self._cancel_flag.clear()
        self.pbar["value"] = 0
        self.var_phase.set("Starting...")
        self.var_status.set(f"Running batch of {len(self._queue_paths)}.")
        self.var_queue_status.set(f"File 1 of {len(self._queue_paths)}:  {Path(self._queue_paths[0]).name}")
        self.var_eta.set("ETA: --")
        self.var_elapsed_file.set("File: 0.0 s")
        self.var_elapsed_batch.set("Batch: 0.0 s")
        self._run_start_ts = time.time()
        self._file_start_ts = self._run_start_ts
        self.btn_run.config(state="disabled")
        self.btn_cancel.config(state="normal")

        self._worker = threading.Thread(target=self._batch_thread, daemon=True)
        self._worker.start()
        self.root.after(200, self._tick_elapsed)

    def _build_config_for(self, input_video: str) -> "PipelineConfig":
        device_map = {"Auto": "auto", "CUDA": "cuda", "CPU": "cpu"}
        output_video = safe_default_output_path(input_video)
        return PipelineConfig(
            input_video=input_video,
            output_video=output_video,
            wordlist_path=self.var_wordlist.get().strip(),
            model_size=self.var_model.get(),
            device=device_map.get(self.var_device.get(), "auto"),
            engine=self._canonical_engine(),
            model_path=self.var_model_path.get().strip(),
            mode=self.var_mode.get(),
            pre_pad_ms=int(self.var_pre_pad_ms.get()),
            post_pad_ms=int(self.var_post_pad_ms.get()),
            audio_delay_ms=int(self.var_audio_delay_ms.get()),
            output_mode=self.var_output_mode.get().strip() or "replace",
            ffmpeg_dir=self.var_ffmpeg_dir.get().strip(),
            use_transcript_cache=bool(self.var_use_cache.get()),
        )

    def _on_cancel(self) -> None:
        if not (self._worker and self._worker.is_alive()):
            return
        self._cancel_flag.set()
        self.var_status.set("Cancelling batch...")
        self.btn_cancel.config(state="disabled")

    def _batch_thread(self) -> None:
        """Iterate through the queue, running the pipeline for each file.
        Sends per-file status to the log queue; sends a final ('batch_done', ...)
        message when the entire queue is finished or cancelled."""

        def log_cb(msg: str) -> None:
            self._log_queue.put(("log", msg))

        def progress_cb(fraction: float, phase: str, eta: Optional[float]) -> None:
            self._log_queue.put(("progress", (fraction, phase, eta)))

        # Pre-load the Whisper model once for the whole batch so we don't pay
        # the load cost (5-20 s for medium; longer for large) for every file
        # and so the first file's ETA isn't distorted by it.
        device_map = {"Auto": "auto", "CUDA": "cuda", "CPU": "cpu"}
        try:
            from censor_transcribe import preload_model

            log_cb("Preloading Whisper model for batch...")
            preload_model(
                model_size=self.var_model.get(),
                device=device_map.get(self.var_device.get(), "auto"),
                engine=self._canonical_engine(),
                model_path=self.var_model_path.get().strip(),
                log=log_cb,
            )
        except Exception as exc:  # noqa: BLE001
            log_cb(f"Warning: model preload failed ({exc}); will retry per-file.")

        total = len(self._queue_paths)
        for i, input_video in enumerate(self._queue_paths, start=1):
            if self._cancel_flag.is_set():
                # Record remaining as skipped.
                for skipped in self._queue_paths[i - 1:]:
                    self._queue_results.append((skipped, "skipped (cancelled)", None))
                break

            self._log_queue.put(("queue_status", (i, total, input_video)))
            self._log_queue.put(("log", ""))
            self._log_queue.put(("log", f"===== File {i} of {total}:  {input_video} ====="))
            # Reset per-file progress to 0 so the bar starts fresh for each file.
            self._log_queue.put(("progress", (0.0, "starting", None)))

            try:
                cfg = self._build_config_for(input_video)
            except Exception as exc:  # noqa: BLE001
                self._queue_results.append((input_video, "failed (config)", exc))
                self._log_queue.put(("log", f"ERROR (config): {exc}"))
                continue

            # Session banner in the rolling log file for this file.
            try:
                with open(LOG_FILE, "a", encoding="utf-8") as f:
                    f.write(
                        f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')}  batch {i}/{total}  "
                        f"input={cfg.input_video}  output={cfg.output_video}  "
                        f"engine={cfg.engine}  model={cfg.model_size}  device={cfg.device}  "
                        f"mode={cfg.mode}  pre_pad_ms={cfg.pre_pad_ms}  post_pad_ms={cfg.post_pad_ms}  "
                        f"audio_delay_ms={cfg.audio_delay_ms}  "
                        f"output_mode={cfg.output_mode} =====\n"
                    )
            except OSError:
                pass

            try:
                result = run_pipeline(
                    cfg, log_cb=log_cb, progress_cb=progress_cb, cancel_flag=self._cancel_flag
                )
                self._queue_results.append((input_video, result.output_video, None))
                self._log_queue.put(("file_done", (input_video, result)))
            except Exception as exc:  # noqa: BLE001
                msg = str(exc) or exc.__class__.__name__
                if msg.lower().startswith("cancelled"):
                    self._queue_results.append((input_video, "cancelled", exc))
                    self._log_queue.put(("log", f"Cancelled: {input_video}"))
                    # Cancel flag is set; loop will record the rest as skipped.
                else:
                    self._queue_results.append((input_video, "failed", exc))
                    self._log_queue.put(("log", f"ERROR: {input_video}: {msg}"))

        self._log_queue.put(("batch_done", list(self._queue_results)))

    # ---------------------------------------------------------------- Log --
    def _drain_log_queue(self) -> None:
        try:
            while True:
                kind, payload = self._log_queue.get_nowait()
                if kind == "log":
                    self._append_log(str(payload))
                elif kind == "clear":
                    self.txt_log.configure(state="normal")
                    self.txt_log.delete("1.0", "end")
                    self.txt_log.configure(state="disabled")
                elif kind == "progress":
                    fraction, phase, eta = payload  # type: ignore[misc]
                    self.pbar["value"] = int(fraction * 1000)
                    self.var_phase.set(f"Phase: {phase} ({fraction * 100:.1f}%)")
                    self._last_eta = eta
                    if eta is None:
                        self.var_eta.set("ETA: --")
                    else:
                        self.var_eta.set(f"ETA: {_format_seconds(eta)}")
                elif kind == "done":
                    self._on_pipeline_done(payload)  # type: ignore[arg-type]
                elif kind == "error":
                    self._on_pipeline_error(payload)  # type: ignore[arg-type]
                elif kind == "queue_status":
                    idx, total, path = payload  # type: ignore[misc]
                    self.var_queue_status.set(
                        f"File {idx} of {total}:  {Path(path).name}"
                    )
                    # Reset per-file timer at the start of each queued file.
                    self._file_start_ts = time.time()
                    self.var_elapsed_file.set("File: 0.0 s")
                elif kind == "file_done":
                    _in, result = payload  # type: ignore[misc]
                    n_int = len(result.intervals)
                    self._append_log(
                        f"Done. {result.words_matched} matched of {result.words_transcribed} words; "
                        f"{n_int} interval(s). Output: {_to_windows_path(result.output_video)}"
                    )
                elif kind == "batch_done":
                    self._on_batch_done(payload)  # type: ignore[arg-type]
                elif kind == "nltk_done":
                    messagebox.showinfo(
                        "NLTK data",
                        "NLTK data downloaded successfully. You can now run the pipeline.",
                    )
                elif kind == "nltk_error":
                    self._append_log(f"ERROR: {payload}")
                    messagebox.showerror("NLTK data download failed", str(payload))
        except queue.Empty:
            pass
        finally:
            self.root.after(100, self._drain_log_queue)

    def _append_log(self, msg: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        line = f"[{stamp}] {msg}"
        self.txt_log.configure(state="normal")
        self.txt_log.insert("end", line + "\n")
        self.txt_log.see("end")
        self.txt_log.configure(state="disabled")
        # Also append to the rolling log file next to the script.
        try:
            date_stamp = time.strftime("%Y-%m-%d")
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(f"[{date_stamp} {stamp}] {msg}\n")
        except OSError:
            # Never let logging failures kill the GUI.
            pass

    def _open_log_file(self) -> None:
        path = Path(LOG_FILE).resolve()
        if not path.exists():
            try:
                path.touch()
            except OSError as exc:
                messagebox.showerror("Open log", f"Could not create '{path}':\n{exc}")
                return
        try:
            if os.name == "nt":
                os.startfile(str(path))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                import subprocess
                subprocess.Popen(["open", str(path)])
            else:
                import subprocess
                subprocess.Popen(["xdg-open", str(path)])
        except OSError as exc:
            messagebox.showerror("Open log", f"Could not open '{path}':\n{exc}")

    def _clear_log_file(self) -> None:
        path = Path(LOG_FILE).resolve()
        if not path.exists():
            return
        if not messagebox.askyesno(
            "Clear log file",
            f"Delete all contents of\n{_to_windows_path(str(path))} ?",
        ):
            return
        try:
            path.write_text("", encoding="utf-8")
            self._append_log("Log file cleared.")
        except OSError as exc:
            messagebox.showerror("Clear log", f"Could not clear '{path}':\n{exc}")

    def _tick_elapsed(self) -> None:
        now = time.time()
        if self._run_start_ts is not None:
            self.var_elapsed_batch.set(f"Batch: {_format_seconds(now - self._run_start_ts)}")
        if self._file_start_ts is not None:
            self.var_elapsed_file.set(f"File: {_format_seconds(now - self._file_start_ts)}")
        if self._worker and self._worker.is_alive():
            self.root.after(500, self._tick_elapsed)

    def _on_batch_done(self, results: list[tuple[str, str, Optional[BaseException]]]) -> None:
        self.pbar["value"] = 1000
        self.var_phase.set("Phase: done (100%)")
        self.var_eta.set("ETA: 0 s")
        self.btn_run.config(state="normal")
        self.btn_cancel.config(state="disabled")
        self._worker = None
        self._run_start_ts = None
        self._file_start_ts = None
        self._refresh_cache_status()

        total = len(results)
        succeeded = [r for r in results if r[2] is None and not r[1].startswith("skipped")]
        cancelled = [r for r in results if r[2] is not None and str(r[2]).lower().startswith("cancelled")]
        failed = [r for r in results if r[2] is not None and not str(r[2]).lower().startswith("cancelled")]
        skipped = [r for r in results if r[2] is None and r[1].startswith("skipped")]

        summary = (
            f"Batch complete: {len(succeeded)} succeeded, "
            f"{len(failed)} failed, "
            f"{len(cancelled)} cancelled, "
            f"{len(skipped)} skipped (of {total})."
        )
        self.var_status.set(summary)
        self.var_queue_status.set("")
        self._append_log("")
        self._append_log(summary)

        # Detailed dialog.
        lines: list[str] = [summary, ""]
        if succeeded:
            lines.append("Succeeded:")
            for inp, out, _ in succeeded:
                lines.append(f"  {Path(inp).name}  ->  {Path(out).name}")
        if failed:
            lines.append("")
            lines.append("Failed:")
            for inp, _out, exc in failed:
                lines.append(f"  {Path(inp).name}: {exc}")
        if cancelled:
            lines.append("")
            lines.append("Cancelled during processing:")
            for inp, _out, _ in cancelled:
                lines.append(f"  {Path(inp).name}")
        if skipped:
            lines.append("")
            lines.append("Skipped (queue cancelled):")
            for inp, _out, _ in skipped:
                lines.append(f"  {Path(inp).name}")

        if failed:
            messagebox.showerror("Batch complete (with errors)", "\n".join(lines))
        elif cancelled or skipped:
            messagebox.showwarning("Batch cancelled", "\n".join(lines))
        else:
            messagebox.showinfo("Batch complete", "\n".join(lines))

    # Legacy per-file completion / error handlers kept for compatibility with
    # any single-file callers that still put ("done", ...) or ("error", ...).
    def _on_pipeline_done(self, result) -> None:  # type: ignore[no-untyped-def]
        self.pbar["value"] = 1000
        self.var_phase.set("Phase: done (100%)")
        self.var_status.set(
            f"Done. {result.words_matched} matched of {result.words_transcribed} words; "
            f"{len(result.intervals)} interval(s)."
        )
        self.var_eta.set("ETA: 0 s")
        self.btn_run.config(state="normal")
        self.btn_cancel.config(state="disabled")
        self._worker = None
        self._run_start_ts = None
        self._file_start_ts = None
        self._refresh_cache_status()
        cache_line = (
            "Used cached transcript (skipped transcription).\n\n"
            if getattr(result, "used_cached_transcript", False)
            else ""
        )
        messagebox.showinfo(
            "Censoring complete",
            f"{cache_line}"
            f"Wrote:\n{_to_windows_path(result.output_video)}\n\n"
            f"Muted intervals: {len(result.intervals)}\n"
            f"Matched words: {result.words_matched} of {result.words_transcribed}",
        )

    def _on_pipeline_error(self, exc: BaseException) -> None:
        self.btn_run.config(state="normal")
        self.btn_cancel.config(state="disabled")
        self._worker = None
        self._run_start_ts = None
        self._file_start_ts = None
        msg = str(exc) or exc.__class__.__name__
        # Detect any of our cancellation variants: pipeline raises
        # RuntimeError("Cancelled") and the transcribe stage raises
        # RuntimeError("Cancelled during transcription.")
        is_cancelled = msg.lower().startswith("cancelled")
        if is_cancelled:
            self.var_status.set("Cancelled.")
            self._append_log(msg)
            return
        self.var_status.set(f"Error: {msg}")
        self._append_log(f"ERROR: {msg}")
        messagebox.showerror("Pipeline failed", msg)

    # ---------------------------------------------------------- Settings --
    def _settings_dict(self) -> dict:
        return {
            "queue_paths": self._queue_items(),
            "wordlist_path": self.var_wordlist.get(),
            "ffmpeg_dir": self.var_ffmpeg_dir.get(),
            "model_size": self.var_model.get(),
            "device": self.var_device.get(),
            "engine": self._canonical_engine(),
            "model_path": self.var_model_path.get().strip(),
            "mode": self.var_mode.get(),
            "pre_pad_ms": int(self.var_pre_pad_ms.get()),
            "post_pad_ms": int(self.var_post_pad_ms.get()),
            "audio_delay_ms": int(self.var_audio_delay_ms.get()),
            "output_mode": self.var_output_mode.get().strip() or "replace",
            "use_transcript_cache": bool(self.var_use_cache.get()),
        }

    def _load_settings(self) -> None:
        p = Path(SETTINGS_FILE)
        if not p.exists():
            return
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        # Backward-compat: older settings had a single input_video field; treat
        # it as a one-item queue if the new queue_paths key is missing.
        queued = data.get("queue_paths")
        if queued is None:
            legacy_in = data.get("input_video", "").strip()
            queued = [legacy_in] if legacy_in else []
        self.lst_queue.delete(0, "end")
        for path in queued:
            if path:
                self.lst_queue.insert("end", _to_windows_path(path))
        # Wordlist path: prefer the stored value when the file still exists;
        # otherwise fall back to the script-relative default so a moved or
        # renamed install still points at a real file on first launch.
        stored_wl = str(data.get("wordlist_path", "") or "").strip()
        if stored_wl and Path(stored_wl).exists():
            wl = stored_wl
        else:
            wl = DEFAULT_WORDLIST
        self.var_wordlist.set(_to_windows_path(wl))
        self.var_ffmpeg_dir.set(_to_windows_path(data.get("ffmpeg_dir", "")))
        self.var_model.set(data.get("model_size", "medium"))
        self.var_device.set(data.get("device", "Auto"))
        # Backward-compat: caches/settings predating engine selection
        # implicitly used openai-whisper.
        stored_engine = str(data.get("engine", "openai-whisper")).lower()
        if stored_engine == "faster-whisper":
            self.var_engine.set("Faster Whisper")
        else:
            self.var_engine.set("OpenAI Whisper")
        self.var_model_path.set(_to_windows_path(str(data.get("model_path", "") or "")))
        self.var_mode.set(data.get("mode", "mute"))
        # Backward-compat: an older settings file only had "pad_ms" (symmetric).
        # Use it for the post-pad default, but keep the new 150 ms pre-pad
        # default so upgrading users see the improved onset padding.
        legacy_pad = data.get("pad_ms")
        try:
            pre = int(data.get("pre_pad_ms", 150))
        except (TypeError, ValueError):
            pre = 150
        try:
            post = int(
                data.get(
                    "post_pad_ms",
                    legacy_pad if legacy_pad is not None else 50,
                )
            )
        except (TypeError, ValueError):
            post = 50
        self.var_pre_pad_ms.set(pre)
        self.var_post_pad_ms.set(post)
        try:
            self.var_audio_delay_ms.set(int(data.get("audio_delay_ms", 0)))
        except (TypeError, ValueError):
            self.var_audio_delay_ms.set(0)
        stored_output_mode = str(data.get("output_mode", "replace")).strip().lower()
        if stored_output_mode not in {"replace", "add_track"}:
            stored_output_mode = "replace"
        self.var_output_mode.set(stored_output_mode)
        self.var_use_cache.set(bool(data.get("use_transcript_cache", True)))

    def _save_settings(self) -> None:
        try:
            Path(SETTINGS_FILE).write_text(
                json.dumps(self._settings_dict(), indent=2),
                encoding="utf-8",
            )
        except OSError:
            pass

    def _on_close(self) -> None:
        if self._worker and self._worker.is_alive():
            if not messagebox.askyesno(
                "Quit",
                "A censoring job is running. Cancel it and exit?",
            ):
                return
            self._cancel_flag.set()
        self._save_settings()
        self.root.destroy()


def _format_seconds(s: float) -> str:
    s = max(0.0, float(s))
    if s < 60:
        return f"{s:.1f} s"
    if s < 3600:
        return f"{int(s // 60)} m {int(s % 60)} s"
    return f"{int(s // 3600)} h {int((s % 3600) // 60)} m"


def main() -> None:
    root = tk.Tk()
    VideoCensorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
