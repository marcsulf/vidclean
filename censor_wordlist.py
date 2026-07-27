"""Word list I/O and derivative-aware matching for the video censor tool.

The censor word list is stored as a plain text file: one word per line,
``#`` starts a comment, blank lines are ignored. Matching is case-insensitive
and whole-word only (compound words are NOT matched unless listed explicitly).

Derivative matching is performed with a hybrid strategy:
    - each entry's WordNet lemmas across POS tags {n, v, a, r} are collected;
    - each entry's Porter stem is collected;
    - a transcribed token matches an entry when either their lemma sets
      intersect OR their Porter stems are equal.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, List, Sequence, Set

# NLTK is imported lazily so the module can still be imported (and its
# non-matching helpers exercised) in environments without NLTK.
_WORDNET_POS: tuple = ()
_LEMMATIZER = None
_STEMMER = None
_NLTK_READY = False
_NLTK_ERROR: str | None = None


def ensure_nltk_data(log: "Callable[[str], None] | None" = None) -> None:
    """Attempt to download the required NLTK data packages, raising a clear
    error if the download fails.

    Safe to call at any time (including from a GUI button). Uses
    ``ssl._create_unverified_context`` as a fallback if the initial download
    fails due to certificate verification, which is the most common failure
    mode on corporate networks.
    """
    import nltk

    required = ("wordnet", "omw-1.4")
    missing: list[str] = []
    for pkg in required:
        try:
            nltk.data.find(f"corpora/{pkg}")
        except LookupError:
            missing.append(pkg)
    if not missing:
        if log:
            log("NLTK data already available (wordnet, omw-1.4).")
        return

    if log:
        log(f"Downloading NLTK data: {', '.join(missing)} ...")

    def _try_download() -> list[str]:
        failures: list[str] = []
        for pkg in missing:
            ok = False
            try:
                ok = bool(nltk.download(pkg, quiet=True, raise_on_error=False))
            except Exception:  # noqa: BLE001
                ok = False
            if not ok:
                failures.append(pkg)
        return failures

    failures = _try_download()
    if failures:
        # SSL fallback: many corporate networks intercept TLS; retry with an
        # unverified SSL context.
        if log:
            log("Initial download failed; retrying with unverified SSL context...")
        import ssl

        original_ctx = getattr(ssl, "_create_default_https_context", None)
        try:
            ssl._create_default_https_context = ssl._create_unverified_context  # type: ignore[attr-defined]
            failures = _try_download()
        finally:
            if original_ctx is not None:
                ssl._create_default_https_context = original_ctx  # type: ignore[attr-defined]

    if failures:
        raise RuntimeError(
            "Could not download NLTK data package(s): "
            + ", ".join(failures)
            + "\n\nManual fix (run from a shell that has internet access):\n"
            "    python -m nltk.downloader wordnet omw-1.4\n\n"
            "If that also fails (e.g. SSL / proxy errors), download the .zip files\n"
            "from https://www.nltk.org/nltk_data/ and extract them into one of:\n"
            "    %APPDATA%\\nltk_data\\corpora\\\n"
            "    C:\\nltk_data\\corpora\\\n"
            "so that e.g. 'wordnet.zip' becomes '<path>\\corpora\\wordnet\\'."
        )
    if log:
        log("NLTK data downloaded successfully.")


def _ensure_nltk_ready() -> None:
    """Lazily initialize NLTK, downloading required data if needed."""
    global _WORDNET_POS, _LEMMATIZER, _STEMMER, _NLTK_READY, _NLTK_ERROR
    if _NLTK_READY:
        return
    try:
        import nltk  # noqa: F401
        from nltk.stem import PorterStemmer, WordNetLemmatizer

        ensure_nltk_data()

        # Import wordnet after data is present so POS constants resolve.
        from nltk.corpus import wordnet as wn

        _WORDNET_POS = (wn.NOUN, wn.VERB, wn.ADJ, wn.ADV)
        _LEMMATIZER = WordNetLemmatizer()
        _STEMMER = PorterStemmer()
        _NLTK_READY = True
    except Exception as exc:  # pragma: no cover - depends on env
        _NLTK_ERROR = str(exc)
        raise RuntimeError(
            "NLTK data (wordnet / omw-1.4) is required for derivative "
            "matching.\n\n"
            f"{exc}"
        ) from exc


# Strip everything except letters and internal apostrophes/hyphens for matching.
_TOKEN_STRIP_RE = re.compile(r"[^A-Za-z'\-]+")


def normalize_token(token: str) -> str:
    """Normalize a token for matching: strip punctuation, lowercase,
    remove possessive ``'s`` / trailing apostrophes."""
    if not token:
        return ""
    cleaned = _TOKEN_STRIP_RE.sub("", token).lower()
    # Drop possessive endings.
    if cleaned.endswith("'s"):
        cleaned = cleaned[:-2]
    cleaned = cleaned.strip("'-")
    return cleaned


def _lemmas_for(word: str) -> Set[str]:
    """Return the set of WordNet lemmas for ``word`` across all POS tags,
    plus the word itself."""
    _ensure_nltk_ready()
    assert _LEMMATIZER is not None
    lemmas: Set[str] = {word}
    for pos in _WORDNET_POS:
        try:
            lemmas.add(_LEMMATIZER.lemmatize(word, pos=pos))
        except Exception:
            # Defensive: unusual inputs can raise inside WordNet.
            pass
    return lemmas


def _stem_for(word: str) -> str:
    _ensure_nltk_ready()
    assert _STEMMER is not None
    return _STEMMER.stem(word)


# Match modes for a single censor entry.
MODE_WHOLE = "whole"          # whole-word + derivatives
MODE_SUBSTRING = "substring"  # core appears anywhere in the token
MODE_PREFIX = "prefix"        # token starts with core
MODE_SUFFIX = "suffix"        # token ends with core
MODE_PHRASE = "phrase"        # multi-word phrase (each sub-token whole-word)


def _parse_entry_mode(raw: str) -> tuple[str, str]:
    """Interpret ``*`` wildcards in a raw entry and return ``(core, mode)``.

    Syntax:
      - ``word``           -> whole-word (with derivatives)
      - ``*word*``         -> substring anywhere
      - ``word*``          -> prefix (token starts with ``word``)
      - ``*word``          -> suffix (token ends with ``word``)
      - ``word1 word2 ...``-> phrase (consecutive whole-word matches)

    Phrases are detected by internal whitespace and always use whole-word
    matching per sub-token. Wildcards inside phrases are not supported and
    will be silently stripped.
    """
    r = raw.strip()
    # Phrase: two or more whitespace-separated tokens (ignoring wildcard
    # decorators — those are meaningless for a phrase entry).
    inner = r.strip("*").strip()
    if any(ch.isspace() for ch in inner):
        return inner, MODE_PHRASE
    lead = r.startswith("*")
    trail = r.endswith("*")
    core = r.strip("*").strip()
    if lead and trail:
        return core, MODE_SUBSTRING
    if trail:
        return core, MODE_PREFIX
    if lead:
        return core, MODE_SUFFIX
    return core, MODE_WHOLE


@dataclass(frozen=True)
class CensorEntry:
    """One entry from the censor word list, with precomputed match features.

    For :data:`MODE_PHRASE` entries, ``tokens`` is a tuple of sub-entries
    (each in :data:`MODE_WHOLE`) representing the consecutive words of the
    phrase. For all other modes ``tokens`` is empty.
    """

    original: str
    normalized: str
    mode: str
    lemmas: frozenset
    stem: str
    tokens: tuple = ()

    @classmethod
    def from_word(cls, word: str) -> "CensorEntry":
        core, mode = _parse_entry_mode(word)

        if mode == MODE_PHRASE:
            # Build a whole-word sub-entry for each token in the phrase.
            parts = core.split()
            sub_entries: List[CensorEntry] = []
            for part in parts:
                norm = normalize_token(part)
                if not norm:
                    continue
                sub_entries.append(
                    cls(
                        original=part,
                        normalized=norm,
                        mode=MODE_WHOLE,
                        lemmas=frozenset(_lemmas_for(norm)),
                        stem=_stem_for(norm),
                    )
                )
            if len(sub_entries) < 2:
                raise ValueError(
                    f"Phrase entry '{word}' must contain at least two valid tokens"
                )
            return cls(
                original=word,
                normalized=" ".join(e.normalized for e in sub_entries),
                mode=MODE_PHRASE,
                lemmas=frozenset(),
                stem="",
                tokens=tuple(sub_entries),
            )

        # For substring/prefix/suffix, use raw lowercased core; do NOT strip
        # apostrophes/hyphens because the user may deliberately include them.
        if mode == MODE_WHOLE:
            norm = normalize_token(core)
            if not norm:
                raise ValueError(f"Censor word '{word}' is empty after normalization")
            return cls(
                original=word,
                normalized=norm,
                mode=mode,
                lemmas=frozenset(_lemmas_for(norm)),
                stem=_stem_for(norm),
            )
        # Wildcard modes
        core_norm = core.lower().strip()
        if not core_norm:
            raise ValueError(f"Censor word '{word}' is empty after normalization")
        return cls(
            original=word,
            normalized=core_norm,
            mode=mode,
            lemmas=frozenset(),
            stem="",
        )


@dataclass
class WordMatcher:
    """Match transcribed tokens against a set of censor entries.

    Whole-word entries match a normalized token when:
      - direct equality with the entry's normalized form,
      - or the entry's lemma set intersects the token's lemma set,
      - or the entry's Porter stem equals the token's Porter stem.

    Wildcard entries match by substring / prefix / suffix on the lowercased,
    de-punctuated token.

    Phrase entries match a consecutive run of tokens where each sub-token
    matches its phrase position by the whole-word rule above.
    """

    entries: List[CensorEntry] = field(default_factory=list)

    def matches(self, token: str) -> bool:
        norm = normalize_token(token)
        if not norm:
            return False

        whole_entries: List[CensorEntry] = []
        for entry in self.entries:
            if entry.mode == MODE_PHRASE:
                continue  # phrases can't match a single isolated token
            if entry.mode == MODE_SUBSTRING:
                if entry.normalized in norm:
                    return True
            elif entry.mode == MODE_PREFIX:
                if norm.startswith(entry.normalized):
                    return True
            elif entry.mode == MODE_SUFFIX:
                if norm.endswith(entry.normalized):
                    return True
            else:
                whole_entries.append(entry)

        # Cheap direct-equality path first for whole-word entries.
        for entry in whole_entries:
            if entry.normalized == norm:
                return True
        if not whole_entries:
            return False
        # Compute derivative features once per token.
        try:
            token_lemmas = _lemmas_for(norm)
            token_stem = _stem_for(norm)
        except RuntimeError:
            # NLTK unavailable — fall back to direct equality only.
            return False
        for entry in whole_entries:
            if entry.lemmas & token_lemmas:
                return True
            if entry.stem == token_stem:
                return True
        return False

    def find_matches(self, tokens: Sequence[str]) -> List[tuple[int, int]]:
        """Return ``(start, end_exclusive)`` index ranges of matches in
        ``tokens``. Convenience wrapper around :meth:`find_matches_detailed`
        that drops the entry information.
        """
        return [(s, e) for s, e, _ in self.find_matches_detailed(tokens)]

    def find_matches_detailed(
        self, tokens: Sequence[str]
    ) -> List[tuple[int, int, str]]:
        """Return ``(start, end_exclusive, entry_original)`` for each match.

        The third element is :attr:`CensorEntry.original`, i.e. the entry
        text as the user wrote it (with any wildcards intact), useful for
        per-entry tallies in logs and reports.

        At each position, phrases are tried first (longest wins). If no
        phrase matches, single-word entries are consulted. Matched ranges
        do not overlap: after a match at position i of length L, the next
        candidate is at position i + L.
        """
        n = len(tokens)
        normed: List[str] = [normalize_token(t) for t in tokens]

        # Per-position derivative feature caches to avoid recomputing.
        lemma_cache: List[Set[str] | None] = [None] * n
        stem_cache: List[str | None] = [None] * n

        def _get_lemmas(i: int) -> Set[str]:
            cached = lemma_cache[i]
            if cached is not None:
                return cached
            if not normed[i]:
                lemma_cache[i] = set()
                return lemma_cache[i]  # type: ignore[return-value]
            try:
                lemma_cache[i] = _lemmas_for(normed[i])
            except Exception:  # noqa: BLE001
                lemma_cache[i] = set()
            return lemma_cache[i]  # type: ignore[return-value]

        def _get_stem(i: int) -> str:
            cached = stem_cache[i]
            if cached is not None:
                return cached
            if not normed[i]:
                stem_cache[i] = ""
                return ""
            try:
                stem_cache[i] = _stem_for(normed[i])
            except Exception:  # noqa: BLE001
                stem_cache[i] = ""
            return stem_cache[i]  # type: ignore[return-value]

        def _whole_matches_at(i: int, entry: CensorEntry) -> bool:
            tok = normed[i]
            if not tok:
                return False
            if entry.normalized == tok:
                return True
            if entry.lemmas and (entry.lemmas & _get_lemmas(i)):
                return True
            if entry.stem and entry.stem == _get_stem(i):
                return True
            return False

        def _phrase_matches_at(i: int, entry: CensorEntry) -> bool:
            if i + len(entry.tokens) > n:
                return False
            for j, sub in enumerate(entry.tokens):
                if not _whole_matches_at(i + j, sub):
                    return False
            return True

        phrase_entries = [e for e in self.entries if e.mode == MODE_PHRASE]
        single_entries = [e for e in self.entries if e.mode != MODE_PHRASE]

        # Sort phrases by descending token count so longest wins by preference.
        phrase_entries.sort(key=lambda e: -len(e.tokens))

        results: List[tuple[int, int, str]] = []
        i = 0
        while i < n:
            if not normed[i]:
                i += 1
                continue
            # Phrases first.
            matched_len = 0
            matched_original = ""
            for entry in phrase_entries:
                if _phrase_matches_at(i, entry):
                    matched_len = len(entry.tokens)
                    matched_original = entry.original
                    break
            if matched_len > 0:
                results.append((i, i + matched_len, matched_original))
                i += matched_len
                continue
            # Single-word.
            hit_entry: CensorEntry | None = None
            for entry in single_entries:
                if entry.mode == MODE_WHOLE:
                    if _whole_matches_at(i, entry):
                        hit_entry = entry
                        break
                elif entry.mode == MODE_SUBSTRING:
                    if entry.normalized in normed[i]:
                        hit_entry = entry
                        break
                elif entry.mode == MODE_PREFIX:
                    if normed[i].startswith(entry.normalized):
                        hit_entry = entry
                        break
                elif entry.mode == MODE_SUFFIX:
                    if normed[i].endswith(entry.normalized):
                        hit_entry = entry
                        break
            if hit_entry is not None:
                results.append((i, i + 1, hit_entry.original))
            i += 1
        return results


def parse_wordlist_text(text: str) -> List[str]:
    """Parse censor list text into raw entry strings, preserving order and
    dropping duplicates, comments, and blanks. Wildcard decorations
    (``*word*``, ``word*``, ``*word``) are preserved. Phrase entries
    (two or more whitespace-separated words) are preserved verbatim."""
    seen: Set[str] = set()
    words: List[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        core, mode = _parse_entry_mode(line)
        if mode == MODE_PHRASE:
            # Normalize whitespace and lowercase for the dedup key, but keep
            # the user's exact phrase text as the entry so it's visible in
            # the editor.
            norm_phrase = " ".join(core.lower().split())
            key = f"phrase:{norm_phrase}"
            entry = " ".join(core.split())  # collapse repeated spaces
            if not norm_phrase or key in seen:
                continue
            seen.add(key)
            words.append(entry)
            continue
        if mode == MODE_WHOLE:
            norm_core = normalize_token(core)
            key = norm_core
            entry = norm_core
        else:
            norm_core = core.lower().strip()
            key = f"{mode}:{norm_core}"
            if mode == MODE_SUBSTRING:
                entry = f"*{norm_core}*"
            elif mode == MODE_PREFIX:
                entry = f"{norm_core}*"
            else:  # suffix
                entry = f"*{norm_core}"
        if not norm_core or key in seen:
            continue
        seen.add(key)
        words.append(entry)
    return words


def load_wordlist(path: str | Path) -> List[str]:
    """Load a plain-text censor word list from ``path``."""
    p = Path(path)
    if not p.exists():
        return []
    return parse_wordlist_text(p.read_text(encoding="utf-8"))


def save_wordlist(path: str | Path, words: Iterable[str], header: str | None = None) -> None:
    """Save a list of words (one per line) to ``path``, optionally prefixed
    with a header comment block."""
    p = Path(path)
    lines: List[str] = []
    if header:
        for hline in header.splitlines():
            if hline.startswith("#") or not hline:
                lines.append(hline)
            else:
                lines.append(f"# {hline}")
        lines.append("")
    for w in words:
        lines.append(w)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_matcher(words: Sequence[str]) -> WordMatcher:
    """Build a :class:`WordMatcher` from an iterable of raw censor words."""
    entries: List[CensorEntry] = []
    for w in words:
        try:
            entries.append(CensorEntry.from_word(w))
        except ValueError:
            continue
    return WordMatcher(entries=entries)
