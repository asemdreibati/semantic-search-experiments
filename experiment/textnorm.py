"""Arabic text normalisation applied identically to documents and queries."""
import re
import unicodedata

_DIACRITICS = re.compile("[ؐ-ًؚ-ٰٟۖ-ۭ]")
_TATWEEL = "ـ"
_BIDI = re.compile("[​-‏‪-‮⁦-⁩]")
_WS = re.compile("[ \t ]+")
_BLANKS = re.compile(r"\n{3,}")


def normalize(text: str) -> str:
    # NFKC maps Arabic presentation forms (U+FB50-U+FEFF) back to base letters.
    text = unicodedata.normalize("NFKC", text)
    text = _DIACRITICS.sub("", text).replace(_TATWEEL, "")
    text = _BIDI.sub("", text)  # OCR emits LRM/RLM marks around numbers
    text = _WS.sub(" ", text)
    return _BLANKS.sub("\n\n", text).strip()
