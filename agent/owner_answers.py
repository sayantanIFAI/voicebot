"""Answers the owner dictated on 2026-10-18 for questions callers keep asking. Deterministic and multilingual: a cue-matched fixed answer, no model, no
guessing. The facts that change (the branch the agent is configured for) come from the environment; the figures the owner gave (30 minutes to an hour, two
to five minutes per patient) are named constants here and nowhere else.

  * branch identity  -- "is it Sonoscan?  Malda Sonoscan?  Kolkata Sonoscan?"  CLINIC_BRANCH (default Kolkata) decides yes / no.
  * schemes          -- Swasthya Sathi / Ayushman Bharat do not cover the OPD or diagnostics.
  * distance         -- from the bus stand, Sealdah, Howrah, Chitpur: no exact figure is known; Google Maps, 30 minutes to 1 hour, brief the driver, the address.
  * token / arrival  -- the doctor sits at 10, the token is 20: come between 11 and 12 (2 to 5 minutes a patient); a serial number cannot be moved up
                        on request, but whoever pays sooner from the payment link gets an earlier token.
  * visit fee        -- the cue only; the figure comes from the doctors table.
"""

from __future__ import annotations

import os
import re
import unicodedata

CLINIC_BRANCH = os.environ.get("CLINIC_BRANCH", "Kolkata")
MINUTES_PER_PATIENT = (2, 5)  # owner, 2026-10-18: "per patient keep 2 to 5 minutes"
TRAVEL_TIME = "30 minutes to 1 hour"


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFC", s or "").lower().replace("়", "").replace("़", "")
    return " ".join(re.sub(r"[.,;:!?\"'()।॥\-]+", " ", s).split())


def _has(t: str, words) -> bool:
    padded = f" {t} "
    for w in words:
        w = _norm(w) if not w.isascii() else w.strip() if w != w.strip() and w.strip() else w
        if not w:
            continue
        if (f" {w} " in padded) if w.isascii() else (w in t):
            return True
    return False


# ------------------------------------------------------------------------------------------------------------------------------------------ branch

# key -> (English, Bengali, Hindi, forms heard)
PLACES: dict[str, tuple[str, str, str, tuple[str, ...]]] = {
    "kolkata": ("Kolkata", "কলকাতা", "कोलकाता", ("kolkata", "calcutta", "kolkatta", "কলকাতা", "কোলকাতা", "कोलकाता", "कलकत्ता")),
    "malda": ("Malda", "মালদা", "मालदा", ("malda", "maldah", "মালদা", "মালদহ", "मालदा", "मालदह")),
    "balurghat": ("Balurghat", "বালুরঘাট", "बालुरघाट", ("balurghat", "বালুরঘাট", "बालुरघाट")),
    "gangarampur": ("Gangarampur", "গঙ্গারামপুর", "गंगारामपुर", ("gangarampur", "গঙ্গারামপুর", "গংগারামপুর", "गंगारामपुर")),
    "siliguri": ("Siliguri", "শিলিগুড়ি", "सिलीगुड़ी", ("siliguri", "শিলিগুড়ি", "सिलीगुड़ी", "सिलिगुड़ी")),
    "raiganj": ("Raiganj", "রায়গঞ্জ", "रायगंज", ("raiganj", "রায়গঞ্জ", "रायगंज")),
    "cooch behar": ("Cooch Behar", "কোচবিহার", "कूचबिहार", ("cooch behar", "coochbehar", "কোচবিহার", "कूचबिहार")),
    "jalpaiguri": ("Jalpaiguri", "জলপাইগুড়ি", "जलपाईगुड़ी", ("jalpaiguri", "জলপাইগুড়ি", "जलपाईगुड़ी")),
    "berhampore": ("Berhampore", "বহরমপুর", "बहरामपुर", ("berhampore", "baharampur", "বহরমপুর", "बहरामपुर")),
    "durgapur": ("Durgapur", "দুর্গাপুর", "दुर्गापुर", ("durgapur", "দুর্গাপুর", "दुर्गापुर")),
    "asansol": ("Asansol", "আসানসোল", "आसनसोल", ("asansol", "আসানসোল", "आसनसोल")),
    "krishnanagar": ("Krishnanagar", "কৃষ্ণনগর", "कृष्णानगर", ("krishnanagar", "কৃষ্ণনগর", "कृष्णानगर")),
}
_SONOSCAN = ("sonoscan", "sono scan", "sonoscen", "sonosken", "সোনোস্ক্যান", "সোনোস্কেন", "সোনো স্ক্যান", "সোনোস্ক্যান", "सोनोस्कैन", "सोनो स्कैन", "सोनोस्कैन")
_IDENTITY_CUES = ("কি", "কী", "এটা", "ওটা", "is it", "is this", "this is", "are you", "hai", "है", "क्या", "ye", "ইয়ে", "naki", "নাকি", "ki", "এই")
_NOT_IDENTITY = ("price", "rate", "দাম", "রেট", "কত", "book", "বুক", "appointment", "অ্যাপয়েন্টমেন্ট", "test", "টেস্ট", "doctor", "ডাক্তার", "report", "রিপোর্ট")


_BN_SIGNS = re.compile("[ািীুূৃেৈোৌংঁ্]")
_HI_SIGNS = re.compile("[ािीुूृेैोौंँ्]")


def _skeleton(word: str) -> str:
    """The consonants of a word, however its vowels are written: "সোনোস্ক্যান", "সোনস্কান", "সোনোস্কেন" -> "সনসকন"; "sonoscan", "sonoscen" -> "snscn";
    "सोनोस्कैन" -> "सनसकन". A recogniser spells the same name many ways; the consonants hold still."""
    w = unicodedata.normalize("NFC", word or "").lower().replace("়", "").replace("़", "")
    w = w.replace("্য", "").replace("्य", "")  # ya-phala: "স্ক্যান" -> "স্কান"
    w = _BN_SIGNS.sub("", w)
    w = _HI_SIGNS.sub("", w)
    w = re.sub(r"[aeiouy]", "", w) if w.isascii() else w
    return w


_SONOSCAN_SKELETONS = frozenset({"snscn", "সনসকন", "सनसकन"})


def _mentions_sonoscan(t: str) -> bool:
    """"Sonoscan" in any spelling, written as one word or two ("sono scan", "সোনো স্ক্যান")."""
    tokens = t.split()
    for i, tok in enumerate(tokens):
        if _skeleton(tok) in _SONOSCAN_SKELETONS:
            return True
        if i + 1 < len(tokens) and _skeleton(tok + tokens[i + 1]) in _SONOSCAN_SKELETONS:
            return True
    return False


def _place_in(t: str) -> str | None:
    for key, (_en, _bn, _hi, forms) in PLACES.items():
        if _has(t, forms):
            return key
    return None


def branch_identity(text: str, lang: str) -> str | None:
    """"Is it Sonoscan?" / "Malda Sonoscan?" / "কলকাতা সোনোস্ক্যান?" -- answered from the branch this agent is configured for."""
    t = _norm(text)
    if not t or len(t.split()) > 10 or not (_has(t, _SONOSCAN) or _mentions_sonoscan(t)):
        return None
    place = _place_in(t)
    if place is None and (not _has(t, _IDENTITY_CUES) or _has(t, _NOT_IDENTITY)):
        return None
    here_key = _place_in(_norm(CLINIC_BRANCH)) or _norm(CLINIC_BRANCH)
    en_here, bn_here, hi_here = next(((v[0], v[1], v[2]) for k, v in PLACES.items() if k == here_key), (CLINIC_BRANCH, CLINIC_BRANCH, CLINIC_BRANCH))
    if place is None or place == here_key:
        return {
            "bn": f"জি, আপনি সোনোস্ক্যান {bn_here}-তে কথা বলছেন। বলুন, কীভাবে সাহায্য করতে পারি?",
            "hi": f"जी, आप सोनोस्कैन {hi_here} में बात कर रहे हैं। बताइए, मैं क्या मदद कर सकती हूँ?",
            "en": f"Yes, you have reached Sonoscan {en_here}. How can I help you?",
        }.get(lang, "")
    en_p, bn_p, hi_p, _ = PLACES[place]
    return {
        "bn": f"না, এটা সোনোস্ক্যান {bn_here}। {bn_p} শাখার জন্য আপনাকে ওই শাখার নম্বরেই ফোন করতে হবে। {bn_here} নিয়ে কিছু জানার থাকলে বলুন।",
        "hi": f"नहीं, यह सोनोस्कैन {hi_here} है। {hi_p} शाखा के लिए आपको उसी शाखा के नंबर पर फ़ोन करना होगा। {hi_here} के बारे में कुछ पूछना हो तो बताइए।",
        "en": f"No, this is Sonoscan {en_here}. For the {en_p} branch you will need to call that branch's own number. If you need anything about {en_here}, please tell me.",
    }.get(lang, "")


# ----------------------------------------------------------------------------------------------------------------------------------------- schemes

_SCHEME = (
    "স্বাস্থ্যসাথী", "স্বাস্থ্য সাথী", "স্বাস্থ্যসাথি", "swasthya sathi", "swasthyasathi", "swastha sathi", "swasthasathi", "swasthya sathee",
    "আয়ুষ্মান", "আয়ুস্মান", "আয়ুষমান", "ayushman", "ayusman", "ayushmaan", "pmjay", "pm jay", "आयुष्मान", "स्वास्थ्य साथी", "स्वास्थ्यसाथी",
)


def scheme_cover(text: str, lang: str) -> str | None:
    t = _norm(text)
    if not t or not _has(t, _SCHEME):
        return None
    return {
        "bn": "না, স্বাস্থ্যসাথী বা আয়ুষ্মান ভারত কার্ডে আমাদের ওপিডি আর ডায়াগনস্টিকসের খরচ কভার হয় না। ডাক্তার দেখানো আর টেস্টের জন্য নিজেকে পেমেন্ট করতে হবে।",
        "hi": "नहीं, स्वास्थ्य साथी या आयुष्मान भारत कार्ड में हमारे ओपीडी और डायग्नोस्टिक्स का खर्च कवर नहीं होता। डॉक्टर को दिखाने और टेस्ट के लिए खुद भुगतान करना होगा।",
        "en": "No. Swasthya Sathi and Ayushman Bharat cards do not cover the OPD or diagnostics here. You will need to pay yourself for the doctor's visit and for tests.",
    }.get(lang, "")


# --------------------------------------------------------------------------------------------------------------------------------------- distance

_FAR = ("কত দূর", "কতদূর", "দূরত্ব", "কতটা দূর", "দূরে", "কতক্ষণ", "কত সময় লাগ", "কত টাইম", "koto dur", "koto door", "kotodur", "koto dure", "distance", "how far", "how long", "how much time", "kitna door", "kitni door", "कितनी दूर", "कितना दूर", "दूरी", "कितना समय", "कितना टाइम")
_STATIONS: dict[str, tuple[str, str, str, tuple[str, ...]]] = {
    "bus": ("the bus stand", "বাসস্ট্যান্ড", "बस स्टैंड", ("bus stand", "busstand", "bus stop", "বাস স্ট্যান্ড", "বাসস্ট্যান্ড", "বাস স্ট্যাণ্ড", "বাসস্ট্যাণ্ড", "বাস স্টপ", "बस स्टैंड", "बस स्टैण्ड")),
    "sealdah": ("Sealdah station", "শিয়ালদা স্টেশন", "सियालदह स्टेशन", ("sealdah", "sealdaha", "শিয়ালদা", "শিয়ালদহ", "সিয়ালদা", "सियालदह", "सियालदा")),
    "howrah": ("Howrah station", "হাওড়া স্টেশন", "हावड़ा स्टेशन", ("howrah", "haora", "হাওড়া", "हावड़ा", "हावडा")),
    "chitpur": ("Chitpur station", "চিতপুর স্টেশন", "चितपुर स्टेशन", ("chitpur", "chitpore", "চিতপুর", "चितपुर")),
    "kolkata_station": ("Kolkata station", "কলকাতা স্টেশন", "कोलकाता स्टेशन", ("kolkata station", "calcutta station", "কলকাতা স্টেশন", "কোলকাতা স্টেশন", "कोलकाता स्टेशन")),
}
_DRIVER = ("driver", "ড্রাইভার", "ড্রাইভারকে", "ট্যাক্সি", "taxi", "cab", "অটো", "auto", "টোটো", "ड्राइवर", "टैक्सी", "ऑटो")
_ADDRESS_FALLBACK = {
    "bn": "৪৪, সি আই টি রোড, লেডিজ পার্কের কাছে, বেনিয়াপুকুর, কলকাতা",
    "hi": "44, सी आई टी रोड, लेडीज़ पार्क के पास, बेनियापुकुर, कोलकाता",
    "en": "44, C I T Road, near Ladies Park, Beniapukur, Kolkata",
}


def distance_from(text: str) -> str | None:
    """The station / stand the caller asks the distance from, or None. A distance (or travel-time, or driver) word AND one of the places."""
    t = _norm(text)
    if not t or not (_has(t, _FAR) or _has(t, _DRIVER)):
        return None
    # "kolkata station" before the bare "kolkata"; a station is only ever matched whole
    for key in ("kolkata_station", "chitpur", "sealdah", "howrah", "bus"):
        if _has(t, _STATIONS[key][3]):
            return key
    return None


def distance_reply(key: str, lang: str, address: str | None = None) -> str:
    en_p, bn_p, hi_p, _ = _STATIONS[key]
    addr = address or _ADDRESS_FALLBACK.get(lang, _ADDRESS_FALLBACK["en"])
    return {
        "bn": f"{bn_p} থেকে ঠিক কত দূর, সেটা আমার জানা নেই। গুগল ম্যাপ দেখে নিন; প্রতিটা জায়গা থেকেই সাধারণত ৩০ মিনিট থেকে ১ ঘণ্টার মতো লাগে। ড্রাইভারকে পরিষ্কার করে বুঝিয়ে দেবেন। আমাদের ঠিকানা আবার বলছি: {addr}।",
        "hi": f"{hi_p} से ठीक कितनी दूरी है, यह मुझे नहीं पता। गूगल मैप देख लीजिए; हर जगह से आमतौर पर 30 मिनट से 1 घंटा लगता है। ड्राइवर को साफ़ समझा दीजिएगा। हमारा पता फिर से बताती हूँ: {addr}।",
        "en": f"I do not know the exact distance from {en_p}. Please check Google Maps; from each of these places it usually takes about 30 minutes to 1 hour. Please explain clearly to the driver. Our address again: {addr}.",
    }.get(lang, "")


# ------------------------------------------------------------------------------------------------------------------------------ token and arrival

_TOKEN = ("টোকেন", "token", "সিরিয়াল", "serial", "ক্রমিক", "टोकन", "सीरियल", "নম্বর পড়েছে", "number pore")
_ARRIVE = (
    "কখন আসব", "কখন আসতে", "কটায় আসব", "কটার সময় আসব", "কখন যাব", "কখন পৌঁছ", "কটার মধ্যে আসব", "কতটার মধ্যে", "কখন এলে", "কখন গেলে", "কখন ঢুকব", "কখন আসলে", "কখন আসা",
    "when should i come", "when to come", "when to reach", "when do i come", "what time should i come", "what time to come", "when should i reach", "what time to reach",
    "कब आना", "कब आऊँ", "कब आऊं", "कब पहुँच", "कब पहुंच", "कितने बजे आना", "कितने बजे आऊँ", "कब तक आना",
)
_PUSH = (
    "আগে দিন", "আগে করে দিন", "এগিয়ে", "আগে চাই", "আগে হবে", "আগে নম্বর", "আগে সিরিয়াল", "আগে টোকেন", "তাড়াতাড়ি টোকেন", "তাড়াতাড়ি সিরিয়াল", "সিরিয়াল বদল", "সিরিয়াল পাল্টে", "টোকেন বদল", "টোকেন পাল্টে",
    "early token", "earlier token", "earlier serial", "early serial", "push my", "move my token", "move my serial", "change my token", "change my serial", "jump the", "before others", "first token", "first serial",
    "पहले टोकन", "पहले सीरियल", "आगे कर", "जल्दी टोकन", "जल्दी सीरियल", "टोकन बदल", "सीरियल बदल",
)

_BN_NUM = {
    "এক": 1, "দুই": 2, "দু": 2, "তিন": 3, "চার": 4, "পাঁচ": 5, "ছয়": 6, "ছয়": 6, "সাত": 7, "আট": 8, "নয়": 9, "দশ": 10, "এগারো": 11, "বারো": 12, "তেরো": 13, "চোদ্দ": 14,
    "পনেরো": 15, "ষোলো": 16, "সতেরো": 17, "আঠারো": 18, "উনিশ": 19, "বিশ": 20, "একুশ": 21, "বাইশ": 22, "তেইশ": 23, "চব্বিশ": 24, "পঁচিশ": 25, "ছাব্বিশ": 26, "সাতাশ": 27, "আটাশ": 28,
    "ঊনত্রিশ": 29, "উনত্রিশ": 29, "ত্রিশ": 30, "একত্রিশ": 31, "বত্রিশ": 32, "তেত্রিশ": 33, "চৌত্রিশ": 34, "পঁয়ত্রিশ": 35, "ছত্রিশ": 36, "সাঁইত্রিশ": 37, "আটত্রিশ": 38, "ঊনচল্লিশ": 39,
    "চল্লিশ": 40, "একচল্লিশ": 41, "বিয়াল্লিশ": 42, "তেতাল্লিশ": 43, "চুয়াল্লিশ": 44, "পঁয়তাল্লিশ": 45, "ছেচল্লিশ": 46, "সাতচল্লিশ": 47, "আটচল্লিশ": 48, "ঊনপঞ্চাশ": 49, "পঞ্চাশ": 50,
}
_HI_NUM = {
    "एक": 1, "दो": 2, "तीन": 3, "चार": 4, "पाँच": 5, "पांच": 5, "छह": 6, "छः": 6, "सात": 7, "आठ": 8, "नौ": 9, "दस": 10, "ग्यारह": 11, "बारह": 12, "तेरह": 13, "चौदह": 14, "पंद्रह": 15,
    "सोलह": 16, "सत्रह": 17, "अठारह": 18, "उन्नीस": 19, "बीस": 20, "इक्कीस": 21, "बाईस": 22, "तेईस": 23, "चौबीस": 24, "पच्चीस": 25, "तीस": 30, "चालीस": 40, "पचास": 50,
}
_EN_NUM = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
}
_DIG = str.maketrans("০১২৩৪৫৬৭৮৯०१२३४५६७८९", "01234567890123456789")


def token_number(text: str) -> int | None:
    """The token / serial number the caller says ("আমার টোকেন ২০", "token number twenty", "बीस नंबर"). A number of one or two digits, or a number word."""
    t = _norm(text).translate(_DIG)
    m = re.search(r"(?<!\d)(\d{1,3})(?!\d)", t)
    if m:
        n = int(m.group(1))
        return n if 1 <= n <= 300 else None
    for table in (_BN_NUM, _HI_NUM, _EN_NUM):
        for word in sorted(table, key=len, reverse=True):
            w = _norm(word)  # the text is normalised (no nukta), so the key is too
            if (w in t.split()) or (not w.isascii() and w in t):
                return table[word]
    return None


def asks_token_arrival(text: str) -> bool:
    t = _norm(text)
    return bool(t) and _has(t, _ARRIVE) and (_has(t, _TOKEN) or token_number(text) is not None)


def asks_token_push(text: str) -> bool:
    t = _norm(text)
    return bool(t) and _has(t, _TOKEN) and _has(t, _PUSH)


def _hm(hhmm: str) -> int:
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


def arrival_window(sitting_start: str, token: int) -> tuple[int, int]:
    """(first hour, last hour) to come, 24-hour clock. At two to five minutes a patient the token's turn falls between start + 2n and start + 5n minutes;
    the first is rounded to the nearest hour and the last up to the next whole hour: sits at 10, token 20 -> 10:40 to 11:40 -> "11 to 12"."""
    lo = _hm(sitting_start) + MINUTES_PER_PATIENT[0] * token
    hi = _hm(sitting_start) + MINUTES_PER_PATIENT[1] * token
    lo_h = round(lo / 60)
    hi_h = -(-hi // 60)  # ceil
    if hi_h <= lo_h:
        hi_h = lo_h + 1
    return lo_h, hi_h


_BN_HOUR = {1: "একটা", 2: "দুটো", 3: "তিনটে", 4: "চারটে", 5: "পাঁচটা", 6: "ছটা", 7: "সাতটা", 8: "আটটা", 9: "নটা", 10: "দশটা", 11: "এগারোটা", 12: "বারোটা"}
_HI_HOUR = {1: "एक", 2: "दो", 3: "तीन", 4: "चार", 5: "पाँच", 6: "छह", 7: "सात", 8: "आठ", 9: "नौ", 10: "दस", 11: "ग्यारह", 12: "बारह"}


def hour_phrase(h24: int, lang: str) -> str:
    """"সকাল এগারোটা" / "सुबह ग्यारह बजे" / "11 AM" for a whole hour."""
    h12 = h24 % 12 or 12
    if lang == "bn":
        part = "সকাল" if 4 <= h24 < 12 else "দুপুর" if 12 <= h24 < 16 else "বিকেল" if 16 <= h24 < 18 else "সন্ধ্যা" if 18 <= h24 < 20 else "রাত"
        return f"{part} {_BN_HOUR[h12]}"
    if lang == "hi":
        part = "सुबह" if 4 <= h24 < 12 else "दोपहर" if 12 <= h24 < 16 else "शाम" if 16 <= h24 < 20 else "रात"
        return f"{part} {_HI_HOUR[h12]} बजे"
    return f"{h12} {'AM' if h24 < 12 else 'PM'}"


def _bn_case(phrase: str, case: str) -> str:
    """"সকাল দশটা" -> "সকাল দশটায়" (case "loc") / "সকাল দশটার" ("gen"); "চারটে" -> "চারটেয়" / "চারটের"."""
    for tail in ("টা", "টে"):
        if phrase.endswith(tail):
            return phrase[: -len(tail)] + tail + ("য়" if case == "loc" else "র")
    return phrase + ("-এ" if case == "loc" else "-এর")


def token_arrival_reply(token: int | None, sitting_start: str | None, lang: str, doctor: str | None = None) -> str:
    lo_s, hi_s = MINUTES_PER_PATIENT
    if token is None or not sitting_start:
        return {
            "bn": f"রোগী পিছু আমরা দুই থেকে পাঁচ মিনিট ধরি। আপনার টোকেন নম্বর আর ডাক্তার কখন বসেন সেটা বললে আমি বলে দেব কখন আসবেন।",
            "hi": "हम हर मरीज़ पर दो से पाँच मिनट मानते हैं। आपका टोकन नंबर और डॉक्टर कब बैठते हैं, बताइए, तो मैं बता दूँगी कि कब आना है।",
            "en": "We allow two to five minutes per patient. Tell me your token number and when the doctor sits, and I will tell you when to come.",
        }.get(lang, "")
    lo, hi = arrival_window(sitting_start, token)
    start_h = int(sitting_start.split(":")[0])
    who_bn = f"{doctor} " if doctor else "ডাক্তার "
    who_en = f"{doctor} " if doctor else "the doctor "
    return {
        "bn": f"{who_bn}{_bn_case(hour_phrase(start_h, 'bn'), 'loc')} বসেন। রোগী পিছু দুই থেকে পাঁচ মিনিট ধরলে আপনার {token} নম্বর টোকেনের জন্য {hour_phrase(lo, 'bn')} থেকে {_bn_case(hour_phrase(hi, 'bn'), 'gen')} মধ্যে আসুন।",
        "hi": f"डॉक्टर {hour_phrase(start_h, 'hi')} बैठते हैं। हर मरीज़ पर दो से पाँच मिनट मानें तो आपके {token} नंबर टोकन के लिए {hour_phrase(lo, 'hi')} से {hour_phrase(hi, 'hi')} के बीच आइए।",
        "en": f"{who_en.capitalize() if not doctor else doctor + ' '}sits at {hour_phrase(start_h, 'en')}. At two to five minutes a patient, for token {token} please come between {hour_phrase(lo, 'en')} and {hour_phrase(hi, 'en')}.",
    }.get(lang, "")


def token_push_reply(lang: str) -> str:
    return {
        "bn": "সিরিয়াল নম্বর অনুরোধ করে এগিয়ে আনা যায় না। টোকেন পেমেন্টের ক্রমে দেওয়া হয়, তাই পেমেন্ট লিংক থেকে যত তাড়াতাড়ি পেমেন্ট করবেন, তত আগে টোকেন পাবেন।",
        "hi": "सीरियल नंबर कहने पर आगे नहीं किया जा सकता। टोकन भुगतान के क्रम में मिलता है, इसलिए पेमेंट लिंक से जितनी जल्दी भुगतान करेंगे, उतना पहले टोकन मिलेगा।",
        "en": "A serial number cannot be moved up on request. Tokens are given in the order of payment, so the sooner you pay from the payment link, the earlier your token will be.",
    }.get(lang, "")


# ------------------------------------------------------------------------------------------------------------------------------------ visit fee

_VISIT = ("ভিজিট", "বিজিট", "বিজিত", "ভিজিত", "ভিসিট", "ভিজিটিং ফি", "visit", "visiting fee", "consultation fee", "consultation", "fees", "fee ", "ফি ", "ফিস", "কনসালটেশন", "विज़िट", "विजिट", "फीस", "फ़ीस", "कंसल्टेशन")


def asks_visit_fee(text: str) -> bool:
    """A question about what a doctor's visit costs. The figure is read from the doctors table by the caller of this function."""
    t = _norm(text) + " "
    if not t.strip() or "visit" in t and not _has(t, ("fee", "fees", "charge", "charges", "cost", "price", "how much", "কত", "কতো", "क्या", "कितना", "कितने")):
        return False
    return _has(t, _VISIT) and _has(t, ("কত", "কতো", "কত টাকা", "how much", "what is", "what's", "cost", "charge", "charges", "price", "rate", "রেট", "कितना", "कितने", "कितनी", "क्या है", "কী", "কি", "কেমন"))
