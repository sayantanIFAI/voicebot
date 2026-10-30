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

import difflib
import os
import re
import unicodedata

from agent.digit_speech import spoken_digit_groups

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
_IDENTITY_CUES = ("কি", "কী", "এটা", "ওটা", "এটাই", "আপনারা", "আপনি", "is it", "is this", "this is", "are you", "you are", "hai", "है", "क्या", "आप", "ye", "ইয়ে", "naki", "নাকি", "ki", "eta", "এই", "আপনাদের")
_NOT_IDENTITY = (
    "price", "rate", "দাম", "রেট", "কত", "book", "বুক", "appointment", "অ্যাপয়েন্টমেন্ট", "test", "টেস্ট", "doctor", "ডাক্তার", "report", "রিপোর্ট", "cbc", "সিবিসি", "usg", "ইউএসজি",
    "x-ray", "xray", "এক্সরে", "mri", "এমআরআই", "ecg", "ইসিজি", "সময়", "খোলা", "বন্ধ", "কখন", "কবে", "ঠিকানা", "address", "দূর", "টাকা", "रेट", "दाम", "कीमत", "बुकिंग", "डॉक्टर", "रिपोर्ट",
)


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


# "সংস্কান" (the recogniser sometimes drops the first ন into a nasal) is in too; the question still needs an identity cue or a branch name to count
_SONOSCAN_SKELETONS = frozenset({"snscn", "সনসকন", "सनसकन", "সসকন", "ससकन"})


# ---- by SOUND: the consonants of a word as a few sound classes, the same for Bengali, Hindi and Latin spellings. "সোনু স্কান", "সোনস্থান", "সংস্কান", "सोनो स्कैन",
# "sono scan", "sonuscan" all come out as S-N-S-K-N (the stops k/kh/g/t/th are one class: a recogniser hears "স্ক্যান" as "স্থান" as easily as not).
_SOUND_CLASSES = {
    "S": "সশষসসศसशषszSZ", "N": "নণংঙঞनणंङञn", "K": "কখগঘতথটঠক্ষकखगघतथटठkcqgtx", "D": "দধডঢदधडढd", "M": "মमm", "R": "রড়ঢ়रड़ढ़r", "L": "লलl",
    "B": "বভबभbv", "P": "পফपफpf", "J": "জঝচছযजझचछयj",
}
_SOUND_OF = {ch: cls for cls, chars in _SOUND_CLASSES.items() for ch in chars}
_SONOSCAN_SOUND = "SNSKN"


def _sound(word: str) -> str:
    """The sound-class string of `word`: vowels, signs, h and y dropped; repeats collapsed ("sonoscan" -> "SNSKN")."""
    out: list[str] = []
    w = unicodedata.normalize("NFC", word or "").lower().replace("্য", "").replace("्य", "").replace("য়", "").replace("य़", "")  # ya-phala / y: "স্ক্যান" is "স্কান"
    for ch in w:
        cls = _SOUND_OF.get(ch)
        if cls and (not out or out[-1] != cls):
            out.append(cls)
    return "".join(out)


def _mentions_sonoscan(t: str) -> bool:
    """"Sonoscan" in any spelling, however the recogniser breaks it into words ("sono scan", "সোনো স্ক্যান", "সোনু স্কান", "সোন স্ক্যান", "সোনস্থান"): one to three
    neighbouring words whose consonants sound like S-N-S-K-N, allowing one consonant lost or added."""
    tokens = t.split()
    for i in range(len(tokens)):
        for width in (1, 2, 3):
            if i + width > len(tokens):
                break
            if width > 1 and any(len(_sound(w)) > 3 for w in tokens[i : i + width]):
                continue  # a long word in the middle is a different word, not a piece of the name
            # ল and ন are folded together: "শোনো স্কাল" for "সোনোস্ক্যান" (owner, 2026-10-30, "ha eta Sonoscan") is the same name heard with a final l
            snd = _sound("".join(tokens[i : i + width])).replace("L", "N")
            if 4 <= len(snd) <= 6 and difflib.SequenceMatcher(None, snd, _SONOSCAN_SOUND).ratio() >= 0.85:
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
    # "আপনারা কলকাতা সোনোস্ক্যান", "ETA সোনোস্ক্যান", "সোন স্ক্যান", "আপনারা কি সোনু স্কান বলছেন": the clinic's name in a short sentence IS the question "is this Sonoscan?" -- a cue word
    # is only needed when the sentence is long enough to be about something else; any word of another request rules it out.
    if _has(t, _NOT_IDENTITY) and place is None:
        return None
    if place is None and len(t.split()) > 6 and not _has(t, _IDENTITY_CUES):
        return None
    here_key = _place_in(_norm(CLINIC_BRANCH)) or _norm(CLINIC_BRANCH)
    en_here, bn_here, hi_here = next(((v[0], v[1], v[2]) for k, v in PLACES.items() if k == here_key), (CLINIC_BRANCH, CLINIC_BRANCH, CLINIC_BRANCH))
    if place is None or place == here_key:
        return {
            "bn": f"হ্যাঁ, আপনি সোনোস্ক্যান {bn_here}তে কথা বলছেন। বলুন, কীভাবে সাহায্য করতে পারি?",
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
_ROUTE = (
    "কি করে যাব", "কী করে যাব", "কিভাবে যাব", "কীভাবে যাব", "কেমন করে যাব", "কি করে আসব", "কী করে আসব", "কিভাবে আসব", "কীভাবে আসব", "কেমনে যাব", "কেমনে আসব", "যেতে পারব", "আসতে পারব", "যাওয়া যাবে",
    "যাব", "যাবো", "আসব", "আসবো", "পৌঁছ", "পৌছ", "রাস্তা", "রুট", "কোন দিক", "যাওয়ার", "আসার",
    "how to go", "how do i go", "how to come", "how do i come", "how to reach", "how can i reach", "how can i come", "how can i go", "route", "which way", "way to", "get there",
    "kaise jau", "kemne jabo", "kibhabe jabo", "ki kore jabo", "ki kore asbo",
    "कैसे जाऊं", "कैसे जाऊँ", "कैसे जाएं", "कैसे जायें", "कैसे आऊं", "कैसे आऊँ", "कैसे पहुँच", "कैसे पहुंच", "रास्ता", "रूट", "कैसे आना", "कैसे जाना",
)
# "station" as a recogniser writes it: "স্তেতিওন", "স্টেসন", "ইস্টিশন", "stesan"...
_STATION_WORD = (
    "স্টেশন", "স্টেসন", "স্তেশন", "স্তেসন", "স্তেতিওন", "স্টেতিওন", "স্তেতিয়ন", "স্টেটিয়ন", "স্তেশান", "স্টেশান", "ইস্টিশন", "ইষ্টিশন", "ইস্টেশন", "ইস্তিশন", "স্টেসান", "স্তেসান",
    "station", "stesan", "stashan", "istishan", "स्टेशन", "स्टेसन", "इस्टेशन",
)
_STATIONS: dict[str, tuple[str, str, str, tuple[str, ...]]] = {
    "bus": ("the bus stand", "বাসস্ট্যান্ড", "बस स्टैंड", ("bus stand", "busstand", "bus stop", "বাস স্ট্যান্ড", "বাসস্ট্যান্ড", "বাস স্ট্যাণ্ড", "বাসস্ট্যাণ্ড", "বাস স্টপ", "बस स्टैंड", "बस स्टैण्ड")),
    "sealdah": ("Sealdah station", "শিয়ালদা স্টেশন", "सियालदह स्टेशन", ("sealdah", "sealdaha", "sealdoho", "sealdaho", "sealda", "shealdah", "শিয়ালদা", "শিয়ালদহ", "শিয়ালদাহ", "শিয়ালদ", "সিয়ালদা", "সিয়ালদহ", "শেয়ালদা", "শেয়ালদহ", "शियालदह", "शियालदा", "सियालदह", "सियालदा", "सियालदाह")),
    "howrah": ("Howrah station", "হাওড়া স্টেশন", "हावड़ा स्टेशन", ("howrah", "haora", "howra", "haowra", "howrha", "হাওড়া", "হাউড়া", "হাওরা", "হাউরা", "हावड़ा", "हावडा", "हावरा", "हाओड़ा")),
    "chitpur": ("Chitpur station", "চিতপুর স্টেশন", "चितपुर स्टेशन", ("chitpur", "chitpore", "chitpoor", "চিতপুর", "চিৎপুর", "চিত্পুর", "চিতপূর", "चितपुर", "चित्पुर", "चितपूर")),
    "kolkata_station": ("Kolkata station", "কলকাতা স্টেশন", "कोलकाता स्टेशन", ("kolkata station", "calcutta station", "কলকাতা স্টেশন", "কোলকাতা স্টেশন", "कोलकाता स्टेशन")),
    "station": ("your station", "ওই স্টেশন", "उस स्टेशन", ()),
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
    if not t or not (_has(t, _FAR) or _has(t, _DRIVER) or _has(t, _ROUTE)):
        return None
    # "kolkata station" before the bare "kolkata"; a station is only ever matched whole
    for key in ("kolkata_station", "chitpur", "sealdah", "howrah", "bus"):
        if _has(t, _STATIONS[key][3]):
            return key
    # "স্তেতিওন থেকে কি করে যাবো" -- a station, which one not made out: the same answer (Google Maps, tell the driver the address), without naming a station
    if _has(t, _STATION_WORD) and (_has(t, _ROUTE) or _has(t, _FAR)):
        return "station"
    return None


def distance_reply(key: str, lang: str, address: str | None = None) -> str:
    en_p, bn_p, hi_p, _ = _STATIONS[key]
    addr = address or _ADDRESS_FALLBACK.get(lang, _ADDRESS_FALLBACK["en"])
    return {
        "bn": f"{bn_p} থেকে ঠিক কত দূর, সেটা আমার জানা নেই। গুগল ম্যাপ দেখে নিন; প্রতিটা জায়গা থেকেই সাধারণত ৩০ মিনিট থেকে ১ ঘণ্টার মতো লাগে। ড্রাইভারকে পরিষ্কার করে বুঝিয়ে দেবেন। আমাদের ঠিকানা আবার বলছি: {addr}।",
        "hi": f"{hi_p} से ठीक कितनी दूरी है, यह मुझे नहीं पता। गूगल मैप देख लीजिए; हर जगह से आमतौर पर 30 मिनट से 1 घंटा लगता है। ड्राइवर को साफ़ समझा दीजिएगा। हमारा पता फिर से बताती हूँ: {addr}।",
        "en": f"I do not know the exact distance from {en_p}. Please check Google Maps; from each of these places it usually takes about 30 minutes to 1 hour. Please explain clearly to the driver. Our address again: {addr}.",
    }.get(lang, "")


# ------------------------------------------------------------------------------------------------------------------------------------ how do I get there

_HOW = ("কিভাবে", "কীভাবে", "কি করে", "কী করে", "কেমনে", "কেমন করে", "কি ভাবে", "কী ভাবে", "kibhabe", "kivabe", "kemne", "kemon kore", "ki kore", "how", "कैसे", "kaise", "किस तरह")
_GO = (
    "যাব", "যাবো", "আসব", "আসবো", "যেতে", "আসতে", "পৌঁছ", "পৌছ", "যাওয়া", "আসা", "যাই", "আসি", "go", "come", "reach", "get there", "get to", "arrive",
    "जाऊं", "जाऊँ", "आऊं", "आऊँ", "जाना", "आना", "पहुँच", "पहुंच", "jabo", "asbo", "jete", "aste", "pouchbo",
)


def asks_how_to_reach(text: str) -> bool:
    """"আপনাদের ওখানে যাবো কিভাবে" / "how do I come to you" / "कैसे आऊं": a HOW word and a GO word in a short sentence, with no station or place named (those get the
    distance / Google Maps answer). The answer is the clinic's address (owner, 2026-10-25: "this must answer with the address")."""
    t = _norm(text)
    return bool(t) and len(t.split()) <= 16 and _has(t, _HOW) and _has(t, _GO)


# ------------------------------------------------------------------------------------------------------------------------------------ booking by phone

_PHONE_WORDS = ("ফোনে", "ফোন এ", "ফোনেই", "ফোন দিয়ে", "মোবাইলে", "over the phone", "on phone", "by phone", "on the phone", "फोन पर", "फ़ोन पर", "फोन से", "phone e", "phone-e")
_BOOKING_WORDS = ("বুকিং", "বুক", "অ্যাপয়েন্টমেন্ট", "অযাপ্ফোন্টমেন্ট", "নাম লেখা", "বুকিং", "booking", "book", "appointment", "बुकिंग", "बुक", "अपॉइंटमेंट")
_BOOKING_NOTE = {
    "bn": "হ্যাঁ, ফোনেই বুকিং হবে। বুকিংয়ের জন্য চল্লিশ টাকা অনলাইনে পেমেন্ট করতে হবে।",
    "hi": "जी हाँ, बुकिंग फ़ोन पर ही हो जाएगी। बुकिंग के लिए चालीस रुपये ऑनलाइन भुगतान करने होंगे।",
    "en": "Yes, you can book over the phone. The booking needs forty rupees paid online.",
}


def asks_phone_booking(text: str) -> bool:
    """"একটা বুকিং দিয়ে দেন, বুকিং করতে পারবো ফোনে": asking whether the booking can be made over this phone call. Answered once, in the owner's words: yes, by phone, forty rupees online
    (owner, 2026-10-24); the booking then goes on."""
    t = _norm(text)
    return bool(t) and len(t.split()) <= 24 and _has(t, _PHONE_WORDS) and _has(t, _BOOKING_WORDS)


def phone_booking_note(lang: str) -> str:
    return _BOOKING_NOTE.get(lang, _BOOKING_NOTE["en"])


# ------------------------------------------------------------------------------------------------------------------------------------ where in the address

_WHERE = ("কোথায়", "কোথা", "কোথাকার", "কোনখানে", "কোন জায়গায়", "কোন দিকে", "কোনদিকে", "কই", "kothay", "kothai", "where", "कहाँ", "कहां", "किधर", "किस तरफ")
_ADDRESS_PARTS = (
    "লেডিজ", "লাডিজ", "লেডিস", "লাডিস", "ladies", "পার্ক", "park", "বেনিয়া", "বিনিয়া", "বেনিআ", "বিনিআ", "benia", "beniya", "beniapukur", "পুকুর",
    "সি আই টি", "সিআইটি", "সিআই টি", "cit road", "c i t", "लेडीज़", "लेडीज", "पार्क", "बेनिया", "सी आई टी",
)
_DIRECTIONS = (
    "কীভাবে যাব", "কিভাবে যাব", "কীভাবে যাওয়া", "কিভাবে যাওয়া", "কীভাবে যেতে", "কিভাবে যেতে", "কীভাবে আসব", "কিভাবে আসব", "কীভাবে আসতে", "কিভাবে আসতে", "রাস্তা", "যাব কীভাবে", "যাব কিভাবে",
    "कैसे जाए", "कैसे जाएं", "कैसे जाये", "कैसे जाना", "कैसे पहुंच", "कैसे पहुँच", "कैसे आएं", "कैसे आए", "कैसे आना", "कैसे आऊं", "कैसे आऊँ", "रास्ता", "कैसे जा सकते",
    "how do i get", "how do i reach", "how can i reach", "how to reach", "how to get", "how can i get", "directions", "route to", "way to the clinic",
)
_WHERE_REPLY = {
    "bn": "জায়গাটা মুখে ঠিকঠাক বুঝিয়ে বলা একটু কঠিন। আপনি গুগল ম্যাপে দেখে নেবেন। অথবা আপনার যে নম্বরে হোয়াটসঅ্যাপ আছে সেটা বলুন, আমরা লোকেশনটা পাঠিয়ে দিচ্ছি।",
    "hi": "जगह मुँह से ठीक से समझाना थोड़ा मुश्किल है। आप गूगल मैप में देख लीजिए। या आपके जिस नंबर पर व्हाट्सऐप है वह बताइए, हम लोकेशन भेज देते हैं।",
    "en": "It is a little hard to explain the spot by voice. Please look it up on Google Maps. Or tell me the number you have WhatsApp on, and we will send you the location.",
}
_LOCATION_SENT = {
    "bn": "ধন্যবাদ। {digits} এই নম্বরের হোয়াটসঅ্যাপে লোকেশন পাঠিয়ে দেওয়া হবে।",
    "hi": "धन्यवाद। {digits} इस नंबर के व्हाट्सऐप पर लोकेशन भेज दी जाएगी।",
    "en": "Thank you. The location will be sent to the WhatsApp on {digits}.",
}
_LOCATION_AGAIN = {
    "bn": "নম্বরটা পুরোটা শুনতে পাইনি। হোয়াটসঅ্যাপ নম্বরটা, দশটা সংখ্যা, আরেকবার বলবেন?",
    "hi": "नंबर पूरा सुनाई नहीं दिया। व्हाट्सऐप नंबर, दस अंक, एक बार फिर बताइए।",
    "en": "I did not catch the whole number. Please say the ten-digit WhatsApp number once more.",
}


def asks_where_in_address(text: str) -> bool:
    """"বিনিয়া পুকুরটা কোথায়" / "কোথায় এটা লাডিজ পার্ক এর কাছে": a WHERE question about a part of the clinic's address. Voice cannot place a spot like that; the answer is
    Google Maps, or the number on WhatsApp to send the location to (owner, 2026-10-19). A where-word AND a word of the address, in a short sentence."""
    t = _norm(text)
    if not t or len(t.split()) > 14:
        return False
    # Owner, 2026-10-28: "तो इधर हावड़ा से उतर के कैसे जाए" (how do I get there from Howrah) was answered with the clinic's address again. A HOW-TO-GET-THERE question gets the same honest
    # answer as a where-question: it is hard to explain by voice -- Google Maps, or the WhatsApp number to send the location to. No route is ever invented.
    if _has(t, _DIRECTIONS):
        return True
    return _has(t, _WHERE) and _has(t, _ADDRESS_PARTS)


def where_reply(lang: str) -> str:
    return _WHERE_REPLY.get(lang, _WHERE_REPLY["en"])


def location_sent(digits: str, lang: str) -> str:
    return _LOCATION_SENT.get(lang, _LOCATION_SENT["en"]).format(digits=spoken_digit_groups(digits))


def location_again(lang: str) -> str:
    return _LOCATION_AGAIN.get(lang, _LOCATION_AGAIN["en"])


# ------------------------------------------------------------------------------------------------------------------------ billing dues (KCD-lead-9)

# Owner, 2026-10-22: a real number is set per deployment; left blank, the caller is told "our WhatsApp number"
# without reciting a figure that does not exist -- same discipline as the contact_number FAQ answer
# (clinic-api/seed.py), which also never states a number it cannot verify, only describes how one is delivered.
CLINIC_WHATSAPP_NUMBER = os.environ.get("CLINIC_WHATSAPP_NUMBER", "")

_BILLING_DUES = {
    "bn": "পেমেন্টের প্রমাণ{number} হোয়াটসঅ্যাপে পাঠিয়ে দিন, কেউ আপনাকে ফিরে ফোন করে জানিয়ে দেবে।",
    "hi": "पेमेंट का प्रमाण{number} व्हाट्सऐप पर भेज दीजिए, कोई आपको वापस फ़ोन करके बता देगा।",
    "en": "Please send proof of payment{number} on WhatsApp, and someone will call you back about it.",
}
_WITH_NUMBER = {"bn": " আমাদের {n} নম্বরে", "hi": " हमारे {n} नंबर पर", "en": " to our number {n}"}


# KCD-lead-B: "packages" (plural), "health checkup package", "full body package", "master health check", and the Bengali / Hindi words for package
# (a conjunct may be written প্যাকেজ / প্যাকেজ / প্যাকেজের ...) must all be caught -- the whole-word tuple used before refused "packages" because it padded
# "package" with spaces. A bare "package" is enough in this domain (a diagnostic clinic): nothing else a caller says "package" about is on offer here.
_HEALTH_PACKAGE_RE = re.compile(
    r"\bpackages?\b|\bpkgs?\b|\bpackej\b|\bpakej\b|\bpaikej\b|\bpackeg\b|master health ?check|health ?check ?up"
    r"|(?:full|whole|complete|total) body (?:check|checkup|check-up)|(?:full|complete|preventive|annual) health (?:check|checkup|check-up)|body (?:check|checkup|check-up) (?:plan|scheme|offer)"
    r"|প্যাকেজ|প্যাকেট|পেকেজ|পাকেজ|পেকেজ|ফুল বডি চেকআপ|মাস্টার হেলথ|সম্পূর্ণ (?:স্বাস্থ্য|শরীর)(?:ের)? ?(?:পরীক্ষা|চেকআপ)|হেলথ চেকআপ"
    r"|पैकेज|पेकेज|पैकिज|फुल बॉडी चेकअप|फुल बॉडी चैकअप|मास्टर हेल्थ|पूरा (?:हेल्थ )?चेकअप|संपूर्ण (?:स्वास्थ्य|शरीर) (?:जांच|चेकअप)|हेल्थ चेकअप",
    re.I,
)
_HEALTH_PACKAGE_REPLY = {
    "bn": "দুঃখিত, সোনোস্ক্যানে কোনো হেলথ প্যাকেজ বা চেকআপ প্যাকেজ নেই। নির্দিষ্ট কোনো টেস্ট বা ডাক্তারের অ্যাপয়েন্টমেন্ট বুক করতে চান?",
    "hi": "माफ़ कीजिए, सोनोस्कैन में कोई हेल्थ पैकेज या चेकअप पैकेज नहीं है। कोई खास टेस्ट या डॉक्टर का अपॉइंटमेंट बुक करना चाहेंगे?",
    "en": "Sorry, Sonoscan does not offer any health package or checkup package. Would you like to book a specific test or a doctor's appointment instead?",
}


def asks_about_health_package(text: str) -> bool:
    t = _norm(text)
    return bool(t) and _HEALTH_PACKAGE_RE.search(t) is not None


def health_package_reply(lang: str) -> str:
    """Owner, 2026-10-22 (KCD-lead-13): Sonoscan has no health-package offering -- a caller asking about one is
    told so plainly and redirected to a real test or doctor booking, the same "no package offer" framing
    agent/reply_templates.test_rate_reply already documents for an unmatched test name, said BEFORE that generic
    not-found path (deterministic, pre-model, same reason asks_for_a_person runs before intent extraction) so the
    caller is never told "no such test" for something that was never a test name to begin with."""
    return _HEALTH_PACKAGE_REPLY.get(lang, _HEALTH_PACKAGE_REPLY["en"])


# Owner, 2026-10-23: a caller who did not get the booking payment link is told it is a network issue, to wait fifteen minutes, and to call back if it still has
# not come -- nothing is looked up and no promise is made about the link itself.
_PAYMENT_LINK_MISSING = {
    "bn": "নেটওয়ার্কে একটা সমস্যা হচ্ছে। দয়া করে পনেরো মিনিট অপেক্ষা করুন। তারপরও লিংক না পেলে আবার ফোন করবেন, আমরা আপনাকে সাহায্য করব।",
    "hi": "नेटवर्क में कुछ समस्या हो रही है। कृपया पंद्रह मिनट इंतज़ार कीजिए। फिर भी लिंक न मिले तो दोबारा फ़ोन कीजिए, हम आपकी मदद करेंगे।",
    "en": "There is a network issue. Please wait fifteen minutes. If you still do not get the link, please call back again and we will help you.",
}


def payment_link_missing_reply(lang: str) -> str:
    return _PAYMENT_LINK_MISSING.get(lang, _PAYMENT_LINK_MISSING["en"])


def billing_dues_reply(lang: str) -> str:
    """Owner, 2026-10-22 (KCD-lead-9): a caller asking whether they owe money is never read a balance over the
    phone -- they are asked to send payment proof on WhatsApp, and told a human will call back. CLINIC_WHATSAPP_NUMBER
    unset means the sentence still makes sense without reciting a number that was never configured."""
    lang = lang if lang in _BILLING_DUES else "en"
    number = _WITH_NUMBER[lang].format(n=CLINIC_WHATSAPP_NUMBER) if CLINIC_WHATSAPP_NUMBER else ""
    return _BILLING_DUES[lang].format(number=number)


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
    "सोलह": 16, "सत्रह": 17, "अठारह": 18, "उन्नीस": 19, "बीस": 20, "इक्कीस": 21, "बाईस": 22, "तेईस": 23, "चौबीस": 24, "पच्चीस": 25, "छब्बीस": 26, "सत्ताईस": 27, "अट्ठाईस": 28, "उनतीस": 29, "तीस": 30,
    "इकतीस": 31, "बत्तीस": 32, "तैंतीस": 33, "चौंतीस": 34, "पैंतीस": 35, "छत्तीस": 36, "सैंतीस": 37, "अड़तीस": 38, "उनतालीस": 39, "चालीस": 40, "इकतालीस": 41, "बयालीस": 42, "तैंतालीस": 43, "चौवालीस": 44, "पैंतालीस": 45, "छियालीस": 46, "सैंतालीस": 47, "अड़तालीस": 48, "उनचास": 49, "पचास": 50,
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


def spoken_number(text: str) -> int | None:
    """A number said in the sentence -- digits, a Bengali / Hindi number word, or an English one, compound ("forty four") included."""
    t = _norm(text).translate(_DIG)
    m = re.search(r"(?<!\d)(\d{1,4})(?!\d)", t)
    if m:
        return int(m.group(1))
    words = t.split()
    tens = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50}
    for i, w in enumerate(words):
        if w in tens:
            unit = _EN_NUM.get(words[i + 1]) if i + 1 < len(words) else None
            return tens[w] + (unit if unit and unit < 10 else 0)
    return token_number(text)


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

# Owner, 2026-10-27: real calls ask this far more often as "কত টাকা লাগবে দেখাতে" (how much to SEE him) than as
# "ভিজিট/ফি" (the formal word for a visit fee) -- "দেখাতে/দেখানো" (bn), "দিখানে/দিখানা" (hi), "to see/to show" (en)
# are now an alternate way into the same question, not only the formal fee words.
_VISIT = (
    "ভিজিট", "বিজিট", "বিজিত", "ভিজিত", "ভিসিট", "ভিজিটিং ফি", "visit", "visiting fee", "consultation fee",
    "consultation", "fees", "fee ", "ফি ", "ফিস", "ফীস", "কনসালটেশন", "विज़िट", "विजिट", "फीस", "फ़ीस", "कंसल्टेशन",
)
_SEE_DOCTOR = (
    "দেখাতে", "দেখানো", "দেখাতে গেলে", "দেখানোর", "দেখাব", "দেখাবো", "দেখানোতে",
    "dekhate", "dekhabo", "dekhano", "dekhanor",
    "दिखाने", "दिखाना", "दिखाऊं", "दिखाऊंगा", "दिखाने में", "दिखाने को",
    "dikhane", "dikhana", "dikhaane", "dikhaanaa",
    "to see", "to show", "seeing the doctor", "see him", "see her", "see dr",
)
_FEE_CONTEXT = _VISIT + _SEE_DOCTOR
_PRICE_WORD = (
    "কত", "কতো", "কত টাকা", "কততাকা", "কেমন",
    "koto", "kototaka", "kemon",
    "how much", "what is", "what's", "cost", "price", "charge", "charges", "rate",
    "कितना", "कितने", "कितनी", "क्या है", "कितना लगेगा", "कितने का",
    "kitna", "kitne", "kitni", "lagega", "lagbe",
)
# Owner, 2026-10-27 (real call): bare "কি"/"কী" ("what") used to be in the list above -- but they are two of the
# most common words in the language, in constant use with no connection to price at all ("ওটা কি যেন বলে" =
# "whatchamacallit"). "ডাক্তার দেখাবো ... ওটা কি যেন বলে মাথার ডাক্তার আছে" (I want to see a doctor... what's it
# called... is there a head doctor) had no price word in it whatsoever, but _SEE_DOCTOR's "দেখাবো" plus this bare
# "কি" from an unrelated filler phrase were enough together to pass asks_visit_fee, so a plain availability
# question got answered as a fee question instead. "কত"/"কতো" and the other, far more specific phrases already
# above cover the real "how much" question; a bare "what" is too generic a signal to use on its own.


# Owner, 2026-10-28 (real calls). (1) "অন্য একটা নম্বার দিচ্ছি ওটাতে পাঠিয়ে দিবেন পেমেন্ট লিঙ্ক": a payment link goes ONLY to the number the booking was made from -- never to a number the caller
# names -- so the answer is fixed and the caller is told how to get it. (2) "পেমেন্ট লিংক কতক্ষণে আসবে" was answered with the payment METHODS, and in the address step was taken for the address.
# (3) a bare "কত লাগবে" in the middle of a booking was not answered at all.
_LINK = ("লিঙ্ক", "লিংক", "link", "लिंक", "লিনক")
_SENDW = ("পাঠ", "send", "भेज", "resend")
_OTHER_NUMBER = (
    "অন্য একটা নম", "অন্য নম", "অন্য একটি নম", "আরেকটা নম", "আরেকটি নম", "নতুন নম", "অন্য ফোন", "another number", "other number", "different number", "new number", "another phone", "दूसरे नंबर", "दूसरा नंबर",
    "दूसरे नम्बर", "नया नंबर", "अलग नंबर",
)
_LINK_TIME = (
    "কতক্ষণ", "কত সময়", "কখন", "কবে", "কত দেরি", "কতদিন", "how long", "when will", "when do", "how soon", "how much time", "कब", "कितनी देर", "कितने समय", "कितना समय",
)
_LINK_REPLY_OTHER = {
    "bn": "দুঃখিত, পেমেন্ট লিংক শুধু যে নম্বর দিয়ে বুকিং করা হয়েছিল সেই নম্বরেই পাঠানো যায়, অন্য নম্বরে পাঠানো যায় না। লিংকটা যাঁর নম্বরে গেছে, তাঁকে বলুন আপনাকে লিংকটা বা পেমেন্টের নম্বরটা পাঠিয়ে দিতে।",
    "hi": "माफ़ कीजिए, पेमेंट लिंक सिर्फ़ उसी नंबर पर भेजा जा सकता है जिससे बुकिंग हुई थी, किसी दूसरे नंबर पर नहीं। जिनके नंबर पर लिंक गया है, उनसे कहिए कि वे आपको लिंक या पेमेंट का नंबर भेज दें।",
    "en": "Sorry, the payment link can only be sent to the number the booking was made from, never to another number. Please ask the person whose number received the link to forward it, or to send you the payment number.",
}
_LINK_REPLY_TIMING = {
    "bn": "বুকিং শেষ হলে পেমেন্ট লিংকটা আপনার হোয়াটসঅ্যাপে পাঠানো হয়। লিংক পাওয়ার পনেরো মিনিটের মধ্যে পেমেন্ট করতে হবে।",
    "hi": "बुकिंग पूरी होने पर पेमेंट लिंक आपके व्हाट्सऐप पर भेजा जाता है। लिंक मिलने के पंद्रह मिनट के अंदर पेमेंट करना होगा।",
    "en": "When the booking is complete, the payment link is sent to your WhatsApp. The payment has to be made within fifteen minutes of getting the link.",
}
_BARE_COST = (
    "কত লাগবে", "কত লাগে", "কত খরচ", "কত পড়বে", "কত টাকা লাগবে", "কত দিতে হবে", "কত টাকা দিতে", "কত টাকা পড়বে", "কত দেব", "কত দিব", "কত টাকা", "कितना लगेगा", "कितने पैसे", "कितना खर्च",
    "कितना देना", "how much will it", "how much do i", "how much is it", "how much does it cost", "how much to pay",
)


def asks_link_to_other_number(text: str) -> bool:
    t = _norm(text)
    return bool(t) and _has(t, _LINK) and _has(t, _OTHER_NUMBER) and (_has(t, _SENDW) or "দিচ্ছি" in t or "दे रहा" in t or "দিব" in t)


def link_to_other_number_reply(lang: str) -> str:
    return _LINK_REPLY_OTHER.get(lang, _LINK_REPLY_OTHER["en"])


def asks_payment_link_timing(text: str) -> bool:
    t = _norm(text)
    return bool(t) and len(t.split()) <= 14 and _has(t, _LINK) and _has(t, _LINK_TIME) and not _has(t, _OTHER_NUMBER)


def payment_link_timing_reply(lang: str) -> str:
    return _LINK_REPLY_TIMING.get(lang, _LINK_REPLY_TIMING["en"])


# Owner, 2026-10-29: "ওই চল্লিশ টাকা কোন নম্বরে পাঠাতে হবে" ("to which number do I send the forty rupees?") was answered with the helpdesk number. Nobody sends money to a number: the payment link
# goes to the number the booking was made from, and the payment is made through that link. The same for "kun number e 40 taka dite hobe", "किस नंबर पर पैसे भेजने हैं", "which number do I pay to".
_MONEY = ("টাকা", "taka", "tk", "rupee", "rupees", "rs", "পেমেন্ট", "payment", "pay", "ফি", "fee", "fees", "चार्ज", "रुपये", "रुपए", "रुपया", "पैसे", "पैसा", "पेमेंट", "फीस", "চার্জ", "ভিজিট")
_WHICH_NUMBER = (
    "কোন নম্বরে", "কোন নাম্বারে", "কোন নম্বর", "কোন নাম্বার", "কোন নম্বরটায়", "কোন নম্বরটাতে", "কোন একাউন্টে", "কোন অ্যাকাউন্টে", "কোথায় পাঠাতে", "কোথায় পাঠাব", "কোথায় পাঠাবো", "কোথায় দিতে", "কোথায় দেব",
    "কাকে পাঠাতে", "কাকে পাঠাব", "কাকে দিতে", "কার নম্বরে", "কার নাম্বারে", "কোন গুগল পে", "কোন ফোনপে",
    "kon number", "kun number", "kon nombore", "kun nombore", "kon numbare", "kothay pathate", "kothay pathabo", "kothay dite", "kake pathate", "kake dite",
    "which number", "what number", "which account", "where to send", "where do i send", "where should i send", "where do i pay", "where to pay", "to whom", "whom to pay", "whom do i pay",
    "किस नंबर", "किस नम्बर", "कौन से नंबर", "कौन सा नंबर", "किस अकाउंट", "कहाँ भेजना", "कहां भेजना", "कहाँ भेजें", "कहां भेजें", "कहाँ देना", "कहां देना", "किसको भेजना", "किसे भेजना", "किसको देना",
)
_PAY_DEST_REPLY = {
    "bn": "কোনো নম্বরে টাকা পাঠাতে হবে না। যে নম্বর থেকে বুকিং করেছেন, সেই নম্বরেই একটা পেমেন্ট লিংক যাবে। ওই লিংক দিয়েই পেমেন্ট করুন।",
    "hi": "किसी नंबर पर पैसे भेजने की ज़रूरत नहीं है। जिस नंबर से बुकिंग की है, उसी नंबर पर एक पेमेंट लिंक आएगा। उसी लिंक से पेमेंट कीजिए।",
    "en": "You do not send money to any number. A payment link goes to the number you booked from; please pay through that link.",
}


def asks_payment_destination(text: str) -> bool:
    t = _norm(text)
    return bool(t) and len(t.split()) <= 24 and _has(t, _MONEY) and _has(t, _WHICH_NUMBER)


def payment_destination_reply(lang: str) -> str:
    return _PAY_DEST_REPLY.get(lang, _PAY_DEST_REPLY["en"])


def asks_bare_cost(text: str) -> bool:
    """A short "how much?" with no word for what: only meant where a doctor's booking is open (the caller of this function checks that)."""
    t = _norm(text)
    return bool(t) and len(t.split()) <= 6 and _has(t, _BARE_COST)


def asks_visit_fee(text: str) -> bool:
    """A question about what a doctor's visit costs -- the formal "visit/fee" wording, or the far more common "কত
    টাকা লাগবে দেখাতে" (how much to SEE him) phrasing. The figure is read from the doctors table by the caller of
    this function."""
    t = _norm(text) + " "
    if not t.strip():
        return False
    if "visit" in t and not _has(t, ("fee", "fees", "charge", "charges", "cost", "price", "how much", "কত", "কতো", "क्या", "कितना", "कितने")):
        return False
    return _has(t, _FEE_CONTEXT) and _has(t, _PRICE_WORD)
