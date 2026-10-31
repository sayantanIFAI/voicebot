"""A doctor's name that is nobody's: say so, and offer the doctors that come closest (owner, 2026-10-29: "অনীর্বাণ দত্ত").

When no doctor has the name the caller said, the answer says there is no such doctor and names, from the clinic's own list:
  1. the doctors with the same SURNAME ("দত্ত": Dr. Manas Dutta, Dr. Rajarshi Dutta),
  2. the doctors with the same FIRST NAME ("অনির্বাণ": Dr. Anirban Mondal),
  3. the doctors whose whole name sounds closest (the clinic API's did-you-mean list).
Exception, kept from the owner's earlier rule ("for any similarity re-ask"): when nobody has that first name and exactly ONE doctor with that surname sounds close, the reply is only
"আপনি কি <that doctor> বলতে চাইছেন?", without "no such doctor" first.

Pure functions over the catalogue rows (dicts with name, full_name, full_name_bn, full_name_hi); nothing here is a guess about a person that the list does not hold. Names are compared by sound,
across Bengali / Devanagari / Latin spellings (agent/name_script.py).
"""

from __future__ import annotations

from agent import name_script
from agent.dept_talk import _join, spoken_name

_TITLES = frozenset("dr dr. doctor ডাঃ ডাক্তার ডা ডক্টর डॉक्टर डॉ डाक्टर".split())
MAX_PER_GROUP = 4


def _tokens(text: str) -> list[str]:
    return [t.strip(".,?!।") for t in (text or "").replace(".", " ").split() if t.strip(".,?!।") and t.strip(".,?!।").lower() not in _TITLES]


def _own_names(info: dict) -> tuple[list[str], list[str]]:
    """(given-name words, surname words) of a doctor, from every script he is written in."""
    given, sur = [], []
    for key in ("full_name_bn", "full_name_hi", "full_name"):
        toks = _tokens(info.get(key) or "")
        if len(toks) >= 2:
            given += toks[:-1]
            sur.append(toks[-1])
    return given, sur


def _same(a: str, b: str) -> bool:
    return len(name_script.name_skeleton(a)) >= 2 and len(name_script.name_skeleton(b)) >= 2 and name_script.same_given_name(a, b)


def _same_sound(a: str, b: str) -> bool:
    """The two first names have the SAME consonant skeleton (spelling variants of one name); a merely similar name ("রাজশ্রী" / "রাজেশ") does not count here."""
    import difflib

    ka, kb = name_script.name_skeleton(a), name_script.name_skeleton(b)
    # same consonants AND close written forms: "অনীর্বাণ"/"অনির্বাণ" (0.88) are one name, "অমৃতা"/"অমিত" (0.67) are two
    return len(ka) >= 2 and ka == kb and difflib.SequenceMatcher(None, a, b).ratio() >= 0.75


def build(said: str, infos: list[dict], api_suggestions: list[str]) -> dict | None:
    """{"surname": [info...], "given": [info...], "closer": [info...], "short": info|None, "surname_word": str, "given_word": str} or None when nothing is close."""
    toks = _tokens(said)
    if not toks or not infos:
        return None
    sur_word = toks[-1]
    given_word = toks[0] if len(toks) >= 2 else ""
    seen, uniq = set(), []
    for info in infos:
        if id(info) not in seen:
            seen.add(id(info))
            uniq.append(info)
    by_sur = [i for i in uniq if any(_same(sur_word, s) for s in _own_names(i)[1])]
    by_given = [i for i in uniq if given_word and any(_same_sound(given_word, g) for g in _own_names(i)[0]) and i not in by_sur]
    if not given_word:  # a single word: it may be the first name or the surname
        by_given = [i for i in uniq if any(_same_sound(sur_word, g) for g in _own_names(i)[0]) and i not in by_sur]
    sugg = []
    for s in api_suggestions or []:
        for i in uniq:
            if s in (i.get("name"), i.get("full_name")) and i not in sugg:
                sugg.append(i)
    closer = [i for i in sugg if i not in by_sur and i not in by_given]
    # Owner, 2026-10-30 ("ডক্টর অভিজিৎ সেন" was answered "Abhijit Sarkar or Biswajit Sen?"): the order of closeness is the first name (exact, then sounding the same), then the whole name sounding similar; a
    # doctor who shares ONLY the surname is offered when no first name matches at all, never beside a doctor whose first name the caller did say.
    if by_given:
        by_sur = []
    best_same_surname = [i for i in sugg if i in by_sur]  # (empty when a first name matched)
    short = best_same_surname[0] if (not by_given and len(best_same_surname) == 1 and not closer) else None
    if not (by_sur or by_given or closer):
        return None
    return {"surname": by_sur[:MAX_PER_GROUP], "given": by_given[:MAX_PER_GROUP], "closer": closer[:MAX_PER_GROUP], "short": short, "surname_word": sur_word, "given_word": given_word or sur_word}


def offered(groups: dict) -> list[dict]:
    """Every doctor the reply named (kept for the caller's next sentence)."""
    out = []
    for key in ("surname", "given", "closer"):
        out += [i for i in groups.get(key, []) if i not in out]
    if groups.get("short") is not None and groups["short"] not in out:
        out.append(groups["short"])
    return out


def which_offered(text: str, infos: list[dict]) -> list[dict]:
    """The offered doctors the sentence names by SOUND ("মানুষ দর" -> Dr. Manas Dutta): the one(s) with the best score (a first name that sounds the same = 2, the surname = 1); a tie is not a choice."""
    toks = _tokens(text)
    best, hits = 0, []
    for info in infos:
        given, sur = _own_names(info)
        score = (2 if any(_same(t, g) for t in toks for g in given) else 0) + (1 if any(_same(t, s) for t in toks for s in sur) else 0)
        if score > best:
            best, hits = score, [info]
        elif score == best and score > 0:
            hits.append(info)
    return hits if best >= 2 else []


def reply(groups: dict, said_text: str, lang: str, label=None) -> str:
    """The spoken answer for `groups` (see build)."""
    def _named(i: dict) -> str:
        """The doctor's spoken name with his department beside it ("অর্থোপেডিক্স বিভাগের ডাঃ অভিজিৎ সরকার") when the caller of this function can say the department in the call's language (`label`)."""
        n = spoken_name(i, lang)
        dept = label(i.get("department") or "") if label and i.get("department") else ""
        if not dept:
            return n
        return {"bn": f"{dept} বিভাগের {n}", "hi": f"{dept} विभाग के {n}", "en": f"{n} ({dept})"}.get(lang, n)

    if groups.get("short") is not None:
        name = _named(groups["short"])
        return {"bn": f"আপনি কি {name} বলতে চাইছেন?", "hi": f"क्या आप {name} कहना चाह रहे हैं?"}.get(lang, f"Did you mean {name}?")
    names = lambda key: _join([_named(i) for i in groups[key]], lang)  # noqa: E731
    said = name_script.clean_query(said_text, lang) if lang in ("bn", "hi") else said_text
    parts = {
        "bn": [f"'{said}' নামে কেউ বসেন না।"],  # owner, 2026-10-30: "Abhijit Sen bole keu bosen na" -- said first, before anyone is offered
        "hi": [f"'{said}' नाम के कोई डॉक्टर नहीं बैठते।"],
        "en": [f"No doctor named '{said}' sits here."],
    }.get(lang, [f"No doctor named '{said}' sits here."])
    if groups["surname"]:
        w = groups["surname_word"]
        parts.append({"bn": f"{w} নামে আছেন {names('surname')}।", "hi": f"{w} नाम के डॉक्टर हैं {names('surname')}।", "en": f"With the surname {w}: {names('surname')}."}.get(lang, ""))
    if groups["given"]:
        w = groups["given_word"]
        parts.append({"bn": f"{w} নামে আছেন {names('given')}।", "hi": f"{w} नाम के डॉक्टर हैं {names('given')}।", "en": f"With the first name {w}: {names('given')}."}.get(lang, ""))
    if groups["closer"]:
        parts.append({"bn": f"কাছাকাছি নাম: {names('closer')}।", "hi": f"मिलते-जुलते नाम: {names('closer')}।", "en": f"Closest names: {names('closer')}."}.get(lang, ""))
    if len(offered(groups)) == 1:
        parts.append({"bn": "ওনার কথা বলছেন?", "hi": "क्या आप इनकी बात कर रहे हैं?", "en": "Do you mean this doctor?"}.get(lang, ""))
    else:
        parts.append({"bn": "কার কথা বলছেন?", "hi": "आप किसकी बात कर रहे हैं?", "en": "Which one do you mean?"}.get(lang, ""))
    return " ".join(p for p in parts if p)
