"""Answer matching and abstention detection."""
from __future__ import annotations

import re
import unicodedata

from .prompts import ABSTAIN_PHRASES

_PUNCT = re.compile(r"[^\w\s]", flags=re.UNICODE)


def normalize(text: str, lang: str = "en") -> str:
    """Lowercase, fold accents (Latin scripts only), drop punctuation."""
    text = unicodedata.normalize("NFKC", text).casefold()
    if lang != "hi":
        text = "".join(c for c in unicodedata.normalize("NFKD", text) if unicodedata.category(c) != "Mn")
    text = text.replace("_", " ")
    text = _PUNCT.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def is_abstention(pred: str, lang: str = "en") -> bool:
    p = " " + normalize(pred, lang) + " "
    for ph in ABSTAIN_PHRASES[lang]:
        q = normalize(ph, lang)
        if q and (" " + q + " " in p or (len(q) > 5 and q in p)):
            return True
    return False


def matches_gold(pred: str, golds, lang: str = "en", surname: bool = False) -> bool:
    """True if any gold answer appears in the prediction.

    With surname=True a gold of two or more words also matches on its last
    word (>= 4 characters), because models often answer with a surname only.
    """
    p = " " + normalize(pred, lang) + " "
    for g in golds:
        gn = normalize(g, lang)
        if not gn:
            continue
        if " " + gn + " " in p:
            return True
        if surname:
            parts = gn.split()
            if len(parts) >= 2 and len(parts[-1]) >= 4 and " " + parts[-1] + " " in p:
                return True
    return False


def outcome(pred: str, golds, lang: str = "en", surname: bool = False) -> str:
    """'correct' | 'abstain' | 'wrong' for a real entity (golds not None)."""
    if golds and matches_gold(pred, golds, lang, surname):
        return "correct"
    if is_abstention(pred, lang):
        return "abstain"
    return "wrong"


def outcome_fictional(pred: str, lang: str = "en") -> str:
    """'abstain' | 'hallucinate' for a fictional entity."""
    return "abstain" if is_abstention(pred, lang) else "hallucinate"
