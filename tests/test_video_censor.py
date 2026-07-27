"""Unit tests for the video audio censor tool.

These tests cover only the pure-Python components:

* ``censor_wordlist``: parsing, save/load, and derivative-aware matching
  (tests that require NLTK are gated on NLTK availability).
* ``censor_timestamps``: interval building, padding, merging.
* ``censor_transcribe.resolve_device``: device selection under a monkeypatched
  ``torch.cuda.is_available``.

The ffmpeg / Whisper integration paths are exercised only manually.
"""

from __future__ import annotations

import sys
import os
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

# Make the workspace root importable when running as `python -m unittest`
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from censor_timestamps import WordSpan, build_mute_intervals, flatten_transcript_words  # noqa: E402
from censor_wordlist import (  # noqa: E402
    WordMatcher,
    build_matcher,
    load_wordlist,
    normalize_token,
    parse_wordlist_text,
    save_wordlist,
)


def _nltk_available() -> bool:
    try:
        import nltk  # noqa: F401
        from nltk.stem import PorterStemmer, WordNetLemmatizer  # noqa: F401
        from nltk.corpus import wordnet as wn  # noqa: F401
        # Trigger corpus load; if data is missing we still want to try downloading
        # once in the test environment so subsequent tests work.
        try:
            _ = wn.NOUN
        except LookupError:
            nltk.download("wordnet", quiet=True)
            nltk.download("omw-1.4", quiet=True)
            _ = wn.NOUN
        return True
    except Exception:
        return False


NLTK_OK = _nltk_available()


class TestNormalizeToken(unittest.TestCase):
    def test_lowercases_and_strips_punctuation(self) -> None:
        self.assertEqual(normalize_token("Hello,"), "hello")
        self.assertEqual(normalize_token("  DAMN! "), "damn")

    def test_strips_possessive_s(self) -> None:
        self.assertEqual(normalize_token("dog's"), "dog")
        self.assertEqual(normalize_token("Dogs'"), "dogs")

    def test_returns_empty_for_all_punctuation(self) -> None:
        self.assertEqual(normalize_token("!!!"), "")
        self.assertEqual(normalize_token(""), "")


class TestParseWordlistText(unittest.TestCase):
    def test_ignores_comments_and_blanks(self) -> None:
        text = textwrap.dedent(
            """\
            # a comment
            damn

            HELL
            # another
            crap
            """
        )
        self.assertEqual(parse_wordlist_text(text), ["damn", "hell", "crap"])

    def test_dedupes(self) -> None:
        self.assertEqual(parse_wordlist_text("damn\nDamn\nDAMN"), ["damn"])


class TestWordlistSaveLoad(unittest.TestCase):
    def test_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "wl.txt"
            save_wordlist(p, ["damn", "hell"], header="Test list")
            loaded = load_wordlist(p)
            self.assertEqual(loaded, ["damn", "hell"])

    def test_load_missing_returns_empty(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "nope.txt"
            self.assertEqual(load_wordlist(p), [])


@unittest.skipUnless(NLTK_OK, "NLTK / wordnet not available")
class TestWordMatcher(unittest.TestCase):
    def test_direct_match(self) -> None:
        m = build_matcher(["damn"])
        self.assertTrue(m.matches("damn"))
        self.assertTrue(m.matches("Damn!"))
        self.assertTrue(m.matches("DAMN,"))

    def test_plural_and_past_tense(self) -> None:
        m = build_matcher(["run", "damn"])
        self.assertTrue(m.matches("running"))
        self.assertTrue(m.matches("ran"))
        self.assertTrue(m.matches("runs"))
        self.assertTrue(m.matches("damned"))
        self.assertTrue(m.matches("damning"))

    def test_possessive_form(self) -> None:
        m = build_matcher(["dog"])
        self.assertTrue(m.matches("dog's"))
        self.assertTrue(m.matches("dogs"))

    def test_whole_word_only_scunthorpe(self) -> None:
        """Listing 'ass' must NOT match 'class', 'grass', 'pass', 'brass'."""
        m = build_matcher(["ass"])
        for bad in ("class", "grass", "pass", "brass", "assist", "classes"):
            self.assertFalse(m.matches(bad), f"'{bad}' should not match 'ass'")

    def test_no_match_for_unrelated(self) -> None:
        m = build_matcher(["damn"])
        self.assertFalse(m.matches("hello"))
        self.assertFalse(m.matches("cat"))

    def test_empty_matcher_never_matches(self) -> None:
        m = WordMatcher(entries=[])
        self.assertFalse(m.matches("damn"))
        self.assertFalse(m.matches(""))


class TestWildcardModes(unittest.TestCase):
    """Wildcard modes do not require NLTK -- they use plain string ops."""

    def test_substring_matches_compounds(self) -> None:
        m = build_matcher(["*fuck*"])
        for tok in (
            "fuck",
            "Fucked",
            "fucking",
            "fucktard",
            "motherfucker",
            "clusterfuck",
            "unfuckingbelievable",
        ):
            self.assertTrue(m.matches(tok), f"'{tok}' should match *fuck*")

    def test_substring_ignores_unrelated(self) -> None:
        m = build_matcher(["*fuck*"])
        for tok in ("duck", "puck", "focus", "fun"):
            self.assertFalse(m.matches(tok), f"'{tok}' should NOT match *fuck*")

    def test_prefix_mode(self) -> None:
        m = build_matcher(["fuck*"])
        self.assertTrue(m.matches("fuck"))
        self.assertTrue(m.matches("fucktard"))
        self.assertTrue(m.matches("fucker"))
        self.assertFalse(m.matches("motherfucker"))

    def test_suffix_mode(self) -> None:
        m = build_matcher(["*fuck"])
        self.assertTrue(m.matches("fuck"))
        self.assertTrue(m.matches("motherfuck"))
        self.assertTrue(m.matches("clusterfuck"))
        self.assertFalse(m.matches("fucker"))

    def test_wildcard_preserved_through_parse(self) -> None:
        text = "*fuck*\nfuck*\n*fuck\nplain\n"
        parsed = parse_wordlist_text(text)
        self.assertEqual(parsed, ["*fuck*", "fuck*", "*fuck", "plain"])

    def test_whole_word_still_scunthorpe_safe(self) -> None:
        if not NLTK_OK:
            self.skipTest("NLTK / wordnet not available")
        m = build_matcher(["ass"])
        for bad in ("class", "grass", "passage"):
            self.assertFalse(m.matches(bad))

    def test_mixed_entries(self) -> None:
        if not NLTK_OK:
            self.skipTest("NLTK / wordnet not available")
        m = build_matcher(["damn", "*fuck*"])
        self.assertTrue(m.matches("damn"))
        self.assertTrue(m.matches("motherfucker"))
        self.assertFalse(m.matches("class"))


class TestPhraseMatching(unittest.TestCase):
    """Multi-word phrase entries (e.g. 'Jesus Christ') require NLTK for the
    per-token derivative matching to work, so all tests here are skipped
    when NLTK is unavailable."""

    def setUp(self) -> None:
        if not NLTK_OK:
            self.skipTest("NLTK / wordnet not available")

    def test_phrase_matches_only_when_consecutive(self) -> None:
        m = build_matcher(["jesus christ"])
        # Single words alone do NOT match.
        self.assertFalse(m.matches("Jesus"))
        self.assertFalse(m.matches("Christ"))
        # Consecutive appearance in a sequence does match.
        spans = m.find_matches(["Well", "Jesus", "Christ", "that", "hurt"])
        self.assertEqual(spans, [(1, 3)])

    def test_phrase_not_matched_when_separated(self) -> None:
        m = build_matcher(["jesus christ"])
        spans = m.find_matches(["Jesus", "loves", "Christ"])
        self.assertEqual(spans, [])

    def test_phrase_case_insensitive(self) -> None:
        m = build_matcher(["JESUS Christ"])
        spans = m.find_matches(["jesus", "CHRIST"])
        self.assertEqual(spans, [(0, 2)])

    def test_phrase_with_punctuation_on_tokens(self) -> None:
        m = build_matcher(["jesus christ"])
        # Punctuation attached to a transcribed token is stripped during
        # normalization and should not defeat the phrase match.
        spans = m.find_matches(["Jesus,", "Christ!"])
        self.assertEqual(spans, [(0, 2)])

    def test_longer_phrase_wins_over_shorter(self) -> None:
        m = build_matcher(["holy cow", "holy cow batman"])
        spans = m.find_matches(["Well", "holy", "cow", "batman", "there"])
        self.assertEqual(spans, [(1, 4)])

    def test_phrase_and_singleword_coexist(self) -> None:
        m = build_matcher(["damn", "jesus christ"])
        spans = m.find_matches(["damn", "you", "jesus", "christ"])
        self.assertEqual(spans, [(0, 1), (2, 4)])

    def test_phrase_parsed_from_wordlist_text(self) -> None:
        text = "damn\nJesus Christ\n# comment\n\n"
        parsed = parse_wordlist_text(text)
        self.assertIn("damn", parsed)
        # Phrase is preserved (whitespace collapsed).
        self.assertIn("Jesus Christ", parsed)

    def test_build_mute_intervals_with_phrase(self) -> None:
        m = build_matcher(["jesus christ"])
        words = [
            WordSpan(text="well", start=0.0, end=0.5),
            WordSpan(text="Jesus", start=1.0, end=1.5),
            WordSpan(text="Christ", start=1.6, end=2.2),
            WordSpan(text="ouch", start=3.0, end=3.5),
        ]
        intervals = build_mute_intervals(
            words, m, pre_pad_s=0.0, post_pad_s=0.0, total_duration_s=10.0
        )
        # Should be exactly one interval spanning the two phrase tokens.
        self.assertEqual(len(intervals), 1)
        start, end = intervals[0]
        self.assertAlmostEqual(start, 1.0, places=3)
        self.assertAlmostEqual(end, 2.2, places=3)

    def test_phrase_alone_does_not_censor_single_word(self) -> None:
        """The whole point of phrase mode: censoring 'Jesus Christ' should
        not censor a bare 'Jesus' spoken alone."""
        m = build_matcher(["jesus christ"])
        words = [
            WordSpan(text="Jesus", start=0.0, end=0.5),
            WordSpan(text="loves", start=0.6, end=1.0),
            WordSpan(text="you", start=1.1, end=1.4),
        ]
        intervals = build_mute_intervals(
            words, m, pre_pad_s=0.0, post_pad_s=0.0, total_duration_s=10.0
        )
        self.assertEqual(intervals, [])


class TestBuildMuteIntervals(unittest.TestCase):
    class _AlwaysMatcher:
        def matches(self, token: str) -> bool:  # noqa: D401
            return True

    class _NoneMatcher:
        def matches(self, token: str) -> bool:  # noqa: D401
            return False

    def test_empty_words_returns_empty(self) -> None:
        self.assertEqual(build_mute_intervals([], self._AlwaysMatcher()), [])

    def test_no_matches_returns_empty(self) -> None:
        words = [WordSpan("hello", 0.0, 0.5), WordSpan("world", 0.5, 1.0)]
        self.assertEqual(build_mute_intervals(words, self._NoneMatcher()), [])

    def test_padding_applied(self) -> None:
        words = [WordSpan("damn", 1.0, 1.5)]
        out = build_mute_intervals(words, self._AlwaysMatcher(), pad_s=0.1)
        self.assertEqual(out, [(0.9, 1.6)])

    def test_negative_pad_treated_as_zero(self) -> None:
        words = [WordSpan("damn", 1.0, 1.5)]
        out = build_mute_intervals(words, self._AlwaysMatcher(), pad_s=-0.5)
        self.assertEqual(out, [(1.0, 1.5)])

    def test_clamps_to_duration(self) -> None:
        words = [WordSpan("damn", 9.9, 10.1)]
        out = build_mute_intervals(
            words, self._AlwaysMatcher(), pad_s=0.5, total_duration_s=10.0
        )
        self.assertEqual(out, [(9.4, 10.0)])

    def test_merges_overlapping(self) -> None:
        words = [
            WordSpan("a", 1.00, 1.20),
            WordSpan("b", 1.15, 1.40),
            WordSpan("c", 2.00, 2.10),
        ]
        out = build_mute_intervals(words, self._AlwaysMatcher(), pad_s=0.0)
        self.assertEqual(out, [(1.0, 1.4), (2.0, 2.1)])

    def test_merges_adjacent_after_padding(self) -> None:
        words = [
            WordSpan("a", 1.0, 1.2),
            WordSpan("b", 1.3, 1.5),
        ]
        out = build_mute_intervals(words, self._AlwaysMatcher(), pad_s=0.1)
        # a expands to (0.9, 1.3), b to (1.2, 1.6) -> merged (0.9, 1.6)
        self.assertEqual(len(out), 1)
        self.assertAlmostEqual(out[0][0], 0.9, places=6)
        self.assertAlmostEqual(out[0][1], 1.6, places=6)

    def test_skips_zero_length_words(self) -> None:
        words = [WordSpan("x", 1.0, 1.0), WordSpan("y", 2.0, 2.5)]
        out = build_mute_intervals(words, self._AlwaysMatcher(), pad_s=0.0)
        self.assertEqual(out, [(2.0, 2.5)])

    def test_asymmetric_padding(self) -> None:
        words = [WordSpan("damn", 1.0, 1.5)]
        out = build_mute_intervals(
            words, self._AlwaysMatcher(), pre_pad_s=0.15, post_pad_s=0.05
        )
        self.assertEqual(len(out), 1)
        self.assertAlmostEqual(out[0][0], 0.85, places=6)
        self.assertAlmostEqual(out[0][1], 1.55, places=6)

    def test_pre_pad_clamped_at_zero_start(self) -> None:
        words = [WordSpan("damn", 0.05, 0.30)]
        out = build_mute_intervals(
            words, self._AlwaysMatcher(), pre_pad_s=0.15, post_pad_s=0.05
        )
        self.assertEqual(out[0][0], 0.0)
        self.assertAlmostEqual(out[0][1], 0.35, places=6)

    def test_negative_asymmetric_pads_treated_as_zero(self) -> None:
        words = [WordSpan("damn", 1.0, 1.5)]
        out = build_mute_intervals(
            words, self._AlwaysMatcher(), pre_pad_s=-1.0, post_pad_s=-2.0
        )
        self.assertEqual(out, [(1.0, 1.5)])


class TestFlattenTranscriptWords(unittest.TestCase):
    def test_flattens_segments(self) -> None:
        segments = [
            {
                "words": [
                    {"word": " hello", "start": 0.0, "end": 0.5},
                    {"word": " world", "start": 0.5, "end": 1.0},
                ]
            },
            {"words": [{"word": " !!", "start": 1.0, "end": 1.0}]},  # dropped: zero length
        ]
        out = flatten_transcript_words(segments)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0].text, "hello")
        self.assertEqual(out[1].text, "world")


class TestResolveDevice(unittest.TestCase):
    def _make_fake_torch(self, cuda_available: bool):
        fake_torch = mock.MagicMock()
        fake_torch.cuda.is_available.return_value = cuda_available
        fake_torch.cuda.get_device_name.return_value = "Fake GPU"
        return fake_torch

    def _call_resolve(self, cuda_available: bool, requested: str) -> str:
        fake_torch = self._make_fake_torch(cuda_available)
        with mock.patch.dict(sys.modules, {"torch": fake_torch}):
            if "censor_transcribe" in sys.modules:
                del sys.modules["censor_transcribe"]
            import censor_transcribe
            return censor_transcribe.resolve_device(requested)

    def test_auto_prefers_cuda_when_available(self) -> None:
        self.assertEqual(self._call_resolve(True, "auto"), "cuda")

    def test_auto_falls_back_to_cpu(self) -> None:
        self.assertEqual(self._call_resolve(False, "auto"), "cpu")

    def test_cpu_forced(self) -> None:
        self.assertEqual(self._call_resolve(True, "cpu"), "cpu")

    def test_cuda_requested_but_missing_raises(self) -> None:
        fake_torch = self._make_fake_torch(False)
        with mock.patch.dict(sys.modules, {"torch": fake_torch}):
            if "censor_transcribe" in sys.modules:
                del sys.modules["censor_transcribe"]
            import censor_transcribe
            with self.assertRaises(RuntimeError):
                censor_transcribe.resolve_device("cuda")


class TestTranscriptCache(unittest.TestCase):
    def _write_fake_video(self, folder: Path) -> Path:
        p = folder / "clip.mp4"
        p.write_bytes(b"\x00" * 32)
        return p

    def test_missing_cache_returns_none(self) -> None:
        from censor_cache import load_cache

        with tempfile.TemporaryDirectory() as td:
            vid = self._write_fake_video(Path(td))
            self.assertIsNone(load_cache(vid, "medium"))

    def test_save_then_load_roundtrip(self) -> None:
        from censor_cache import cache_path_for, load_cache, save_cache

        with tempfile.TemporaryDirectory() as td:
            vid = self._write_fake_video(Path(td))
            words = [WordSpan("hello", 0.1, 0.5), WordSpan("world", 0.5, 1.0)]
            save_cache(vid, "medium", words, "en", 12.34)
            self.assertTrue(cache_path_for(vid).exists())
            cached = load_cache(vid, "medium")
            self.assertIsNotNone(cached)
            assert cached is not None
            self.assertEqual(len(cached.words), 2)
            self.assertEqual(cached.words[0].text, "hello")
            self.assertAlmostEqual(cached.duration_s, 12.34, places=3)
            self.assertEqual(cached.detected_language, "en")

    def test_model_mismatch_invalidates_cache(self) -> None:
        from censor_cache import load_cache, save_cache

        with tempfile.TemporaryDirectory() as td:
            vid = self._write_fake_video(Path(td))
            save_cache(vid, "medium", [WordSpan("hi", 0.0, 0.1)], "en", 1.0)
            self.assertIsNone(load_cache(vid, "small"))

    def test_size_change_invalidates_cache(self) -> None:
        from censor_cache import load_cache, save_cache

        with tempfile.TemporaryDirectory() as td:
            vid = self._write_fake_video(Path(td))
            save_cache(vid, "medium", [WordSpan("hi", 0.0, 0.1)], "en", 1.0)
            # Grow the file; size no longer matches.
            with open(vid, "ab") as f:
                f.write(b"\x00" * 16)
            self.assertIsNone(load_cache(vid, "medium"))

    def test_mtime_change_invalidates_cache(self) -> None:
        from censor_cache import load_cache, save_cache

        with tempfile.TemporaryDirectory() as td:
            vid = self._write_fake_video(Path(td))
            save_cache(vid, "medium", [WordSpan("hi", 0.0, 0.1)], "en", 1.0)
            # Bump mtime without changing size.
            st = vid.stat()
            os.utime(vid, (st.st_atime, st.st_mtime + 5.0))
            self.assertIsNone(load_cache(vid, "medium"))

    def test_clear_cache(self) -> None:
        from censor_cache import cache_path_for, clear_cache, save_cache

        with tempfile.TemporaryDirectory() as td:
            vid = self._write_fake_video(Path(td))
            save_cache(vid, "medium", [WordSpan("hi", 0.0, 0.1)], "en", 1.0)
            self.assertTrue(cache_path_for(vid).exists())
            self.assertTrue(clear_cache(vid))
            self.assertFalse(cache_path_for(vid).exists())
            self.assertFalse(clear_cache(vid))  # already gone


class TestEngineSelection(unittest.TestCase):
    """Verify engine plumbing in the cache and the transcribe module."""

    def _write_fake_video(self, folder: Path) -> Path:
        p = folder / "clip.mp4"
        p.write_bytes(b"\x00" * 32)
        return p

    def test_available_engines_constant(self) -> None:
        from censor_transcribe import AVAILABLE_ENGINES

        self.assertIn("openai-whisper", AVAILABLE_ENGINES)
        self.assertIn("faster-whisper", AVAILABLE_ENGINES)

    def test_normalize_engine(self) -> None:
        from censor_transcribe import _normalize_engine

        self.assertEqual(_normalize_engine(None), "openai-whisper")
        self.assertEqual(_normalize_engine(""), "openai-whisper")
        self.assertEqual(_normalize_engine("whisper"), "openai-whisper")
        self.assertEqual(_normalize_engine("openai_whisper"), "openai-whisper")
        self.assertEqual(_normalize_engine("Faster Whisper"), "faster-whisper")
        self.assertEqual(_normalize_engine("faster_whisper"), "faster-whisper")
        self.assertEqual(_normalize_engine("faster-whisper"), "faster-whisper")

    def test_cache_engine_backward_compat(self) -> None:
        """Cache files predating engine selection have no 'engine' key.
        They must load only when the request asks for openai-whisper."""
        import json as _json
        from censor_cache import cache_path_for, load_cache, save_cache

        with tempfile.TemporaryDirectory() as td:
            vid = self._write_fake_video(Path(td))
            save_cache(vid, "medium", [WordSpan("hi", 0.0, 0.1)], "en", 1.0)
            # Rewrite the sidecar without the engine field to simulate an
            # older cache written before engine selection was added.
            cp = cache_path_for(vid)
            data = _json.loads(cp.read_text(encoding="utf-8"))
            data.pop("engine", None)
            cp.write_text(_json.dumps(data), encoding="utf-8")

            # Default engine (openai-whisper) must still find it.
            self.assertIsNotNone(load_cache(vid, "medium"))
            self.assertIsNotNone(
                load_cache(vid, "medium", engine="openai-whisper")
            )
            # But asking for faster-whisper must miss.
            self.assertIsNone(
                load_cache(vid, "medium", engine="faster-whisper")
            )

    def test_cache_engine_mismatch_invalidates(self) -> None:
        """Saving with one engine and loading with another must miss."""
        from censor_cache import load_cache, save_cache

        with tempfile.TemporaryDirectory() as td:
            vid = self._write_fake_video(Path(td))
            save_cache(
                vid,
                "medium",
                [WordSpan("hi", 0.0, 0.1)],
                "en",
                1.0,
                engine="faster-whisper",
            )
            self.assertIsNone(
                load_cache(vid, "medium", engine="openai-whisper")
            )
            self.assertIsNotNone(
                load_cache(vid, "medium", engine="faster-whisper")
            )

    def test_faster_whisper_adapter_shape(self) -> None:
        """The faster-whisper adapter must produce the same segment/word
        dict shape as openai-whisper so flatten_transcript_words works."""
        from censor_transcribe import _transcribe_faster_whisper
        from censor_timestamps import flatten_transcript_words

        # Build a minimal fake faster-whisper model that yields two segments
        # each with two words. Segment/Word attributes match faster-whisper's
        # NamedTuple API surface (.start, .end, .text, .words; word .word).
        class _FakeWord:
            def __init__(self, word: str, start: float, end: float) -> None:
                self.word = word
                self.start = start
                self.end = end

        class _FakeSegment:
            def __init__(self, start: float, end: float, text: str, words: list) -> None:
                self.start = start
                self.end = end
                self.text = text
                self.words = words

        class _FakeInfo:
            language = "en"
            duration = 3.0

        class _FakeModel:
            def transcribe(self, wav_path: str, **kwargs) -> tuple:
                self.last_kwargs = kwargs
                segs = [
                    _FakeSegment(
                        0.0, 1.5, "hello world",
                        [_FakeWord(" hello", 0.0, 0.6), _FakeWord(" world", 0.6, 1.5)],
                    ),
                    _FakeSegment(
                        1.5, 3.0, "foo bar",
                        [_FakeWord(" foo", 1.5, 2.2), _FakeWord(" bar", 2.2, 3.0)],
                    ),
                ]
                return (iter(segs), _FakeInfo())

        progress_calls: list[tuple[float, float]] = []
        result = _transcribe_faster_whisper(
            model=_FakeModel(),
            wav_path="fake.wav",
            total_duration_s=3.0,
            progress=lambda a, b: progress_calls.append((a, b)),
            cancel_check=None,
            log=None,
        )

        self.assertEqual(result["language"], "en")
        segments = result["segments"]
        self.assertEqual(len(segments), 2)
        # Downstream code (flatten_transcript_words) must consume it fine.
        words = flatten_transcript_words(segments)
        self.assertEqual([w.text for w in words], ["hello", "world", "foo", "bar"])
        # Progress emitted once per segment.
        self.assertEqual(len(progress_calls), 2)
        self.assertAlmostEqual(progress_calls[0][0], 1.5, places=3)
        self.assertAlmostEqual(progress_calls[1][0], 3.0, places=3)

    def test_faster_whisper_adapter_cancels_between_segments(self) -> None:
        """The adapter must poll cancel_check between segments and raise
        RuntimeError('Cancelled during transcription.') to unwind."""
        from censor_transcribe import _transcribe_faster_whisper

        class _FakeWord:
            def __init__(self) -> None:
                self.word = " hi"
                self.start = 0.0
                self.end = 0.1

        class _FakeSegment:
            start = 0.0
            end = 0.1
            text = "hi"
            words = [_FakeWord()]

        class _FakeInfo:
            language = "en"
            duration = 0.1

        class _FakeModel:
            def transcribe(self, wav_path: str, **kwargs) -> tuple:
                # Yield forever; the cancel check must break out.
                def gen():
                    while True:
                        yield _FakeSegment()
                return (gen(), _FakeInfo())

        calls = {"n": 0}
        def cancel_check() -> bool:
            calls["n"] += 1
            return calls["n"] > 2  # allow the first two segments through

        with self.assertRaises(RuntimeError) as ctx:
            _transcribe_faster_whisper(
                model=_FakeModel(),
                wav_path="fake.wav",
                total_duration_s=1.0,
                progress=None,
                cancel_check=cancel_check,
                log=None,
            )
        self.assertIn("Cancelled", str(ctx.exception))

    def test_looks_like_ssl_error(self) -> None:
        """The SSL-error detector must recognize common cert-verify messages."""
        from censor_transcribe import _looks_like_ssl_error

        self.assertTrue(_looks_like_ssl_error(Exception(
            "[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: "
            "self-signed certificate in certificate chain (_ssl.c:1028)"
        )))
        self.assertTrue(_looks_like_ssl_error(Exception(
            "SSLCertVerificationError: certificate verify failed"
        )))
        # Chained cause is also inspected.
        try:
            try:
                raise Exception("certificate verify failed")
            except Exception as inner:
                raise RuntimeError("wrapper error") from inner
        except RuntimeError as wrapper:
            self.assertTrue(_looks_like_ssl_error(wrapper))
        # Unrelated errors must NOT match.
        self.assertFalse(_looks_like_ssl_error(ValueError("bad shape")))
        self.assertFalse(_looks_like_ssl_error(RuntimeError("out of memory")))

    def test_faster_whisper_ssl_fallback_retries(self) -> None:
        """The SSL fallback helper must retry on cert errors and succeed
        the second time when huggingface_hub is reconfigured."""
        from censor_transcribe import _load_faster_whisper_with_ssl_fallback

        attempts: list[int] = []

        class _FakeWhisperModel:
            def __init__(self, model_size: str, device: str, compute_type: str) -> None:
                attempts.append(1)
                if len(attempts) == 1:
                    raise RuntimeError(
                        "ConnectError: [SSL: CERTIFICATE_VERIFY_FAILED] "
                        "certificate verify failed: self-signed certificate "
                        "in certificate chain (_ssl.c:1028)"
                    )
                self.size = model_size
                self.device = device
                self.compute_type = compute_type

        model = _load_faster_whisper_with_ssl_fallback(
            _FakeWhisperModel, "tiny", "cpu", "int8", log=None
        )
        self.assertEqual(len(attempts), 2)
        self.assertEqual(model.size, "tiny")  # type: ignore[attr-defined]

    def test_faster_whisper_non_ssl_error_not_retried(self) -> None:
        """Non-SSL errors must propagate on the first attempt."""
        from censor_transcribe import _load_faster_whisper_with_ssl_fallback

        attempts: list[int] = []

        class _FakeWhisperModel:
            def __init__(self, model_size: str, device: str, compute_type: str) -> None:
                attempts.append(1)
                raise RuntimeError("some unrelated error")

        with self.assertRaises(RuntimeError) as ctx:
            _load_faster_whisper_with_ssl_fallback(
                _FakeWhisperModel, "tiny", "cpu", "int8", log=None
            )
        self.assertIn("unrelated", str(ctx.exception))
        self.assertEqual(len(attempts), 1)  # no retry

    def test_looks_like_connection_reset(self) -> None:
        """The connection-reset detector must recognize common variants."""
        from censor_transcribe import _looks_like_connection_reset

        self.assertTrue(_looks_like_connection_reset(Exception(
            "[WinError 10054] An existing connection was forcibly closed by the remote host"
        )))
        self.assertTrue(_looks_like_connection_reset(Exception(
            "ConnectionResetError: connection reset by peer"
        )))
        self.assertTrue(_looks_like_connection_reset(Exception(
            "httpx.ConnectError: Connection refused"
        )))
        self.assertFalse(_looks_like_connection_reset(Exception(
            "certificate verify failed"
        )))
        self.assertFalse(_looks_like_connection_reset(ValueError("bad shape")))

    def test_model_path_overrides_size_in_cache_key(self) -> None:
        """Passing model_path uses it as the identifier for the model cache."""
        from censor_transcribe import (
            _MODEL_CACHE,
            _get_or_load_model,
            clear_model_cache,
        )

        clear_model_cache()

        try:
            # Monkey-patch _load_faster_whisper_with_ssl_fallback to a stub
            # that records what identifier was actually used.
            import censor_transcribe as _ct

            observed: list[str] = []

            def _fake_load(WhisperModel, model_id, resolved_device, compute_type, log=None):
                observed.append(model_id)
                return f"model:{model_id}"

            _orig = _ct._load_faster_whisper_with_ssl_fallback
            _ct._load_faster_whisper_with_ssl_fallback = _fake_load
            try:
                # Stub WhisperModel import inside _get_or_load_model
                import sys as _sys
                import types as _types

                fake_fw = _types.ModuleType("faster_whisper")

                class _FakeWM:
                    def __init__(self, *a, **kw) -> None:
                        pass

                fake_fw.WhisperModel = _FakeWM
                _sys.modules["faster_whisper"] = fake_fw

                # First call: model_path used as the identifier.
                m = _get_or_load_model(
                    "faster-whisper", "medium", "cpu", log=None,
                    model_path=r"C:\models\my-local-model",
                )
                self.assertEqual(m, r"model:C:\models\my-local-model")
                self.assertEqual(observed[-1], r"C:\models\my-local-model")

                # Second call with a different path should evict and reload.
                m2 = _get_or_load_model(
                    "faster-whisper", "medium", "cpu", log=None,
                    model_path=r"C:\models\other",
                )
                self.assertEqual(m2, r"model:C:\models\other")
                self.assertEqual(len(observed), 2)

                # Third call with no path falls back to model_size and re-evicts.
                m3 = _get_or_load_model("faster-whisper", "medium", "cpu", log=None)
                self.assertEqual(m3, "model:medium")
                self.assertEqual(observed[-1], "medium")
            finally:
                _ct._load_faster_whisper_with_ssl_fallback = _orig
        finally:
            clear_model_cache()


class TestAudioDelayFilters(unittest.TestCase):
    """Verify build_mute_filter / build_beep_filter_complex integrate the
    audio_delay_ms shift correctly for zero, positive, and negative values."""

    def test_mute_filter_zero_delay(self) -> None:
        from censor_audio_io import build_mute_filter
        f = build_mute_filter([(1.0, 2.0)], audio_delay_ms=0)
        self.assertIn("volume=enable='between(t,1.000,2.000)':volume=0", f)
        self.assertNotIn("adelay", f)
        self.assertNotIn("atrim", f)

    def test_mute_filter_positive_delay_appended(self) -> None:
        from censor_audio_io import build_mute_filter
        f = build_mute_filter([(1.0, 2.0)], audio_delay_ms=150)
        # Mute clauses come first, delay is the last step.
        self.assertTrue(f.endswith("adelay=150:all=1"), f)

    def test_mute_filter_negative_delay_trims_start(self) -> None:
        from censor_audio_io import build_mute_filter
        f = build_mute_filter([(1.0, 2.0)], audio_delay_ms=-100)
        self.assertIn("atrim=start=0.100", f)
        self.assertIn("asetpts=PTS-STARTPTS", f)

    def test_mute_filter_no_intervals_with_delay(self) -> None:
        from censor_audio_io import build_mute_filter
        # When no mute intervals but delay requested, we still need the shift.
        f = build_mute_filter([], audio_delay_ms=200)
        self.assertEqual(f, "adelay=200:all=1")

    def test_mute_filter_no_intervals_no_delay(self) -> None:
        from censor_audio_io import build_mute_filter
        f = build_mute_filter([], audio_delay_ms=0)
        self.assertEqual(f, "anull")

    def test_beep_filter_zero_delay(self) -> None:
        from censor_audio_io import build_beep_filter_complex
        f, _ = build_beep_filter_complex([(1.0, 2.0)], audio_delay_ms=0)
        self.assertIn("amix=inputs=2:duration=first:normalize=0[aout]", f)
        self.assertNotIn("[mixed]", f)

    def test_beep_filter_positive_delay(self) -> None:
        from censor_audio_io import build_beep_filter_complex
        f, _ = build_beep_filter_complex([(1.0, 2.0)], audio_delay_ms=150)
        # Delay is applied after the amix, on the mixed stream.
        self.assertIn("amix=inputs=2:duration=first:normalize=0[mixed]", f)
        self.assertIn("[mixed]adelay=150:all=1[aout]", f)

    def test_beep_filter_negative_delay(self) -> None:
        from censor_audio_io import build_beep_filter_complex
        f, _ = build_beep_filter_complex([(1.0, 2.0)], audio_delay_ms=-250)
        self.assertIn("[mixed]atrim=start=0.250,asetpts=PTS-STARTPTS[aout]", f)


class TestOutputMode(unittest.TestCase):
    """Verify the replace/add_track output modes, including the container-swap
    rule and the ffmpeg argv structure."""

    def test_adjust_output_path_replace_unchanged(self) -> None:
        from censor_audio_io import adjust_output_path_for_mode
        # Replace mode never touches the extension.
        self.assertEqual(
            Path(adjust_output_path_for_mode("in.webm", "replace")).suffix,
            ".webm",
        )
        self.assertEqual(
            Path(adjust_output_path_for_mode("in.mp4", "replace")).suffix,
            ".mp4",
        )

    def test_adjust_output_path_add_track_keeps_supported_ext(self) -> None:
        from censor_audio_io import adjust_output_path_for_mode
        for ext in (".mp4", ".mkv", ".mov", ".m4v", ".ts"):
            out = adjust_output_path_for_mode(f"in{ext}", "add_track")
            self.assertEqual(Path(out).suffix.lower(), ext, ext)

    def test_adjust_output_path_add_track_swaps_bad_containers(self) -> None:
        from censor_audio_io import adjust_output_path_for_mode
        for bad_ext in (".webm", ".ogv", ".ogg", ".flv", ".3gp", ".avi"):
            out = adjust_output_path_for_mode(f"in{bad_ext}", "add_track")
            self.assertEqual(Path(out).suffix, ".mp4", bad_ext)

    def test_container_supports_multiple_audio(self) -> None:
        from censor_audio_io import container_supports_multiple_audio
        self.assertTrue(container_supports_multiple_audio("x.mp4"))
        self.assertTrue(container_supports_multiple_audio("x.MKV"))
        self.assertFalse(container_supports_multiple_audio("x.webm"))
        self.assertFalse(container_supports_multiple_audio("x.FLV"))
        self.assertFalse(container_supports_multiple_audio("x.avi"))

    def test_pipeline_config_resolve_switches_ext_for_add_track(self) -> None:
        from video_censor_pipeline import PipelineConfig
        cfg = PipelineConfig(
            input_video="in.webm",
            output_video="out.webm",
            output_mode="add_track",
        ).resolve()
        self.assertEqual(Path(cfg.output_video).suffix, ".mp4")
        self.assertEqual(cfg.output_mode, "add_track")

    def test_pipeline_config_resolve_keeps_ext_when_replace(self) -> None:
        from video_censor_pipeline import PipelineConfig
        cfg = PipelineConfig(
            input_video="in.webm",
            output_video="out.webm",
            output_mode="replace",
        ).resolve()
        self.assertEqual(Path(cfg.output_video).suffix, ".webm")
        self.assertEqual(cfg.output_mode, "replace")

    def test_pipeline_config_resolve_default_output_mode(self) -> None:
        from video_censor_pipeline import PipelineConfig
        cfg = PipelineConfig(input_video="in.mp4").resolve()
        self.assertEqual(cfg.output_mode, "replace")

    def test_pipeline_config_resolve_normalizes_bad_output_mode(self) -> None:
        from video_censor_pipeline import PipelineConfig
        cfg = PipelineConfig(input_video="in.mp4", output_mode="bogus").resolve()
        self.assertEqual(cfg.output_mode, "replace")

    def test_build_ffmpeg_args_replace_mute_matches_prior_shape(self) -> None:
        from censor_audio_io import MuteJobConfig, build_censor_ffmpeg_args
        job = MuteJobConfig(
            input_video="in.mp4",
            output_video="out.mp4",
            intervals=[(1.0, 2.0)],
            mode="mute",
            output_mode="replace",
        )
        argv = build_censor_ffmpeg_args(job, "ffmpeg", source_audio_count=1, out_path="out.mp4")
        # Replace mode uses -af and 0:a:0.
        self.assertIn("-af", argv)
        self.assertIn("-map", argv)
        self.assertIn("0:v:0", argv)
        self.assertIn("0:a:0?", argv)
        # No filter_complex, no per-stream codec index in replace mode.
        self.assertNotIn("-filter_complex", argv)
        self.assertNotIn("[cens]", argv)

    def test_build_ffmpeg_args_add_track_mute_maps_original_and_cens(self) -> None:
        from censor_audio_io import MuteJobConfig, build_censor_ffmpeg_args
        job = MuteJobConfig(
            input_video="in.mp4",
            output_video="out.mp4",
            intervals=[(1.0, 2.0)],
            mode="mute",
            output_mode="add_track",
        )
        argv = build_censor_ffmpeg_args(job, "ffmpeg", source_audio_count=2, out_path="out.mp4")
        self.assertIn("-filter_complex", argv)
        fc_idx = argv.index("-filter_complex")
        fc = argv[fc_idx + 1]
        self.assertIn("[0:a:0]", fc)
        self.assertIn("[cens]", fc)
        # Video + all original audio streams are copied.
        self.assertIn("0:v", argv)
        self.assertIn("0:a?", argv)
        self.assertIn("[cens]", argv)
        # Censored track is the 3rd output audio stream (index 2) when the
        # source has 2 audio streams.
        self.assertIn("-c:a:2", argv)
        self.assertIn("aac", argv)
        self.assertIn("-metadata:s:a:2", argv)
        self.assertIn("title=Censored", argv)
        self.assertIn("-disposition:a:2", argv)

    def test_build_ffmpeg_args_add_track_beep_relabels_output(self) -> None:
        from censor_audio_io import MuteJobConfig, build_censor_ffmpeg_args
        job = MuteJobConfig(
            input_video="in.mp4",
            output_video="out.mp4",
            intervals=[(1.0, 2.0)],
            mode="beep",
            output_mode="add_track",
        )
        argv = build_censor_ffmpeg_args(job, "ffmpeg", source_audio_count=1, out_path="out.mp4")
        fc = argv[argv.index("-filter_complex") + 1]
        # The beep builder's original [aout] sink is renamed to [cens] in add_track mode.
        self.assertIn("[cens]", fc)
        self.assertNotIn("[aout]", fc)
        # New track is placed at index 1 when the source has 1 audio stream.
        self.assertIn("-c:a:1", argv)
        self.assertIn("-metadata:s:a:1", argv)

    def test_build_ffmpeg_args_rejects_unknown_output_mode(self) -> None:
        from censor_audio_io import MuteJobConfig, build_censor_ffmpeg_args
        job = MuteJobConfig(
            input_video="in.mp4",
            output_video="out.mp4",
            intervals=[(1.0, 2.0)],
            mode="mute",
            output_mode="nonsense",
        )
        with self.assertRaises(ValueError):
            build_censor_ffmpeg_args(job, "ffmpeg", source_audio_count=1, out_path="out.mp4")


if __name__ == "__main__":
    unittest.main()
