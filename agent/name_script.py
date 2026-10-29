"""Names across scripts (owner, 2026-10-29).

Why: a name said in Hindi letters ("डॉक्टर् एम पाल") was echoed back inside a Bengali sentence, title and all, and a "did you mean পাল?" was offered for a word the caller had just said. Bengali and Devanagari are
laid out the same way (both follow the ISCII order), so a letter in one script is the letter U+0080 higher or lower in the other: that turns a name from one script into the other without a dictionary, and
lets two spellings in different scripts be compared as the same name. Pure functions, no I/O.
"""

from __future__ import annotations

import difflib
import re
import unicodedata

_TITLES = frozenset(
    "dr doctor doc sir madam mr mrs ms mam maam "
    "ডাক্তার ডাঃ ডা ডক্টর ডাক্তর স্যার স্যর সার সির ম্যাডাম ম্যাম বাবু সাহেব "
    "डॉक्टर डॉ डाक्टर सर सिर मैडम मैम साहब बाबू".split()
)
_DROP = re.compile("[়া-্ৗঁ-ঃ.,'\\-]")
_MAP = str.maketrans({"ণ": "ন", "শ": "স", "ষ": "স", "ৎ": "ত"})


def _deva_to_bn(s: str) -> str:
    return "".join(chr(ord(c) + 0x80) if "ऀ" <= c <= "ॿ" else c for c in s)


def to_script(text: str, lang: str) -> str:
    """`text` with its Devanagari letters turned into Bengali ones (lang "bn") or its Bengali letters into Devanagari ("hi"); other text is returned as it is."""
    if lang == "bn":
        return _deva_to_bn(text)
    if lang == "hi":
        return "".join(chr(ord(c) - 0x80) if "ঀ" <= c <= "৿" else c for c in text)
    return text


def bare(w: str) -> str:
    return unicodedata.normalize("NFC", w or "").strip(".,?!।").replace("्", "").replace("্", "").lower()


def clean_query(name: str, lang: str) -> str:
    """The name a caller said, without its title or honorific and in the reply language's own script ("डॉक्टर् एम पाल" -> "এম পাল" in a Bengali reply)."""
    titles = {bare(t) for t in _TITLES}
    kept = [w for w in (name or "").split() if bare(w) and bare(w) not in titles]
    return to_script(" ".join(kept) if kept else (name or ""), lang)


def name_skeleton(w: str) -> str:
    """A name's consonant skeleton (vowel signs, virama and the sh / n / t look-alikes removed), the same for Bengali and Devanagari spellings; Latin words are lower-cased as they are."""
    w = unicodedata.normalize("NFC", w or "").lower()
    if w.isascii():
        return w
    return _DROP.sub("", _deva_to_bn(w)).translate(_MAP)


def same_given_name(said: str, own: str) -> bool:
    """Is the name SAID the same name as `own`? Spelling variants of one name are (রুপসা ~ রূপা, অনির্বান ~ অনির্বাণ, हक ~ হক); a different name that merely contains it is not
    (গোপাল is not পাল: the other word sits at the END, behind extra letters)."""
    if (said or "").lower() == (own or "").lower():
        return True
    a, b = name_skeleton(said), name_skeleton(own)
    if not a or not b:
        return False
    if a.isascii() and b.isascii():
        return difflib.SequenceMatcher(None, a, b).ratio() >= 0.75
    if a == b:
        return True
    if (a.endswith(b) or b.endswith(a)) and a[0] != b[0]:
        return False
    return difflib.SequenceMatcher(None, a, b).ratio() >= 0.8


def drop_said(suggestions: list[str], query: str) -> list[str]:
    """Suggestions that are NOT just words the caller already said: "did you mean পাল?" after "এম পাল" was not found tells them nothing."""
    titles = {bare(t) for t in _TITLES}
    qtokens = [t for t in (query or "").split() if bare(t) and bare(t) not in titles]
    out = []
    for s in suggestions or []:
        stokens = [t for t in s.split() if bare(t) and bare(t) not in titles]
        if stokens and all(any(same_given_name(q, t) for q in qtokens) for t in stokens):
            continue
        out.append(s)
    return out


# the letter names used for initials ("এম পাল" = M. Pal, "जेड हक" = Z. Haq)
_LETTER_NAMES = {
    "এ": "A", "বি": "B", "সি": "C", "ডি": "D", "ই": "E", "এফ": "F", "জি": "G", "এইচ": "H", "আই": "I", "জে": "J", "কে": "K", "এল": "L", "এম": "M", "এন": "N", "ও": "O", "পি": "P", "কিউ": "Q",
    "আর": "R", "এস": "S", "টি": "T", "ইউ": "U", "ভি": "V", "ডাব্লিউ": "W", "এক্স": "X", "ওয়াই": "Y", "জেড": "Z",
    "ए": "A", "बी": "B", "सी": "C", "डी": "D", "ई": "E", "एफ": "F", "जी": "G", "एच": "H", "आई": "I", "जे": "J", "के": "K", "एल": "L", "एम": "M", "एन": "N", "ओ": "O", "पी": "P", "क्यू": "Q",
    "आर": "R", "एस": "S", "टी": "T", "यू": "U", "वी": "V", "डब्ल्यू": "W", "एक्स": "X", "वाई": "Y", "जेड": "Z",
}


def initial_letter(token: str) -> str | None:
    """The letter an initial stands for: a Bengali / Hindi letter name ("জেড", "जेड") or a single Latin letter ("Z" / "z"), else None."""
    t = bare(token)
    if len(t) == 1 and t.isascii() and t.isalpha():
        return t.upper()
    return _LETTER_NAMES.get(unicodedata.normalize("NFC", (token or "").strip(".,?!।")))
