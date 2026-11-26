"""Word list for the open-vocabulary box: accepts known object names, rejects typos and nonsense."""

from __future__ import annotations

import difflib
import re
from pathlib import Path

from .config import ROOT

VOCAB_FILE = ROOT / "config" / "vocabulary.txt"


def normalize(text: str) -> str:
    text = re.sub(r"[^a-z0-9' \-]", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


class Vocabulary:
    def __init__(self, path: Path = VOCAB_FILE):
        terms: set[str] = set()
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                line = normalize(line.split("#", 1)[0])
                if line:
                    terms.add(line)
        self.terms = sorted(terms)
        self._set = terms

    def __len__(self):
        return len(self.terms)

    def _known(self, word: str) -> str | None:
        """Exact match, or the singular form of a plural ('glasses' stays, 'cups' -> 'cup')."""
        if word in self._set:
            return word
        for suffix, repl in (("ies", "y"), ("es", ""), ("s", "")):
            if word.endswith(suffix) and word[: -len(suffix)] + repl in self._set:
                return word[: -len(suffix)] + repl
        return None

    def check(self, text: str) -> tuple[str | None, list[str]]:
        """Returns (term to search for, None if rejected) and up to 3 'did you mean' suggestions."""
        term = normalize(text)
        if not term:
            return None, []
        if not self._set:  # no word list: accept everything
            return term, []
        known = self._known(term)
        if known:
            return known, []
        words = term.split()
        # a phrase like 'red cup' or 'coffee mug' is fine when it ends in a known object name
        if len(words) > 1 and self._known(words[-1]):
            return term, []
        return None, difflib.get_close_matches(term, self.terms, n=3, cutoff=0.72)
