"""Every lab-test code in the catalogue, spelled out the way callers say it, generated -- not typed test by test.

Callers say "টি এস এইচ", "পি এস এ", "এইচ আই ভি" (Bengali) or "टी एस एच" (Hindi), and a recogniser writes them either apart or run together
("টিএসএইচ"). Hand-listing these for each test missed one after another (CBC, then TSH, then the rest), so the forms are derived from the
code itself: for every test whose short code is letters only (CBC, ESR, TSH, PSA, HIV, ECG, KFT, ...), the spaced and the joined spelling in
Bengali and in Hindi. A new test with a code gets its spelled forms with no further edit. The same letter names are used by the voice agent
(agent/spelled_letters.py) to read a spelled run back into the code; the two tables are kept identical by tests/test_spelled_gate.py.
"""

from __future__ import annotations

BN_LETTER = {
    "A": "এ", "B": "বি", "C": "সি", "D": "ডি", "E": "ই", "F": "এফ", "G": "জি", "H": "এইচ", "I": "আই", "J": "জে", "K": "কে", "L": "এল",
    "M": "এম", "N": "এন", "O": "ও", "P": "পি", "Q": "কিউ", "R": "আর", "S": "এস", "T": "টি", "U": "ইউ", "V": "ভি", "W": "ডাব্লিউ",
    "X": "এক্স", "Y": "ওয়াই", "Z": "জেড",
}  # fmt: skip
HI_LETTER = {
    "A": "ए", "B": "बी", "C": "सी", "D": "डी", "E": "ई", "F": "एफ", "G": "जी", "H": "एच", "I": "आई", "J": "जे", "K": "के", "L": "एल",
    "M": "एम", "N": "एन", "O": "ओ", "P": "पी", "Q": "क्यू", "R": "आर", "S": "एस", "T": "टी", "U": "यू", "V": "वी", "W": "डब्ल्यू",
    "X": "एक्स", "Y": "वाई", "Z": "जेड",
}  # fmt: skip


def _code_like(word: str) -> bool:
    return len(word) >= 2 and word.isalpha() and word.isascii() and word.upper() == word


def code_of(name: str) -> str | None:
    """The letters-only code a test is known by ("Complete Blood Count (CBC)" -> CBC, "HIV Test (ELISA)" -> HIV, "ESR" -> ESR); else None."""
    head, _, rest = name.partition("(")
    words, inner = head.split(), rest.rstrip(") ").strip()
    if words and _code_like(words[0]):
        return words[0]
    if _code_like(inner):
        return inner
    return None


def forms(name: str, table: dict[str, str]) -> list[str]:
    code = code_of(name)
    if not code:
        return []
    letters = [table[c] for c in code if c in table]
    return [" ".join(letters), "".join(letters)] if len(letters) == len(code) else []


def forms_bn(name: str) -> list[str]:
    return forms(name, BN_LETTER)


def forms_hi(name: str) -> list[str]:
    return forms(name, HI_LETTER)
