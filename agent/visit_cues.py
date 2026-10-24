"""Three small caller requests answered by rule, never by the model (sourav, 2026-10-24; owner's items 4-6 of the 2026-10-24 list).

  1. AFTERNOON VISIT  "ami bikaler dike jabo" / "main shaam ko aaunga" / "I will come in the evening":
        "Sure -- but come keeping some time in hand, and check your serial number."
  2. SHOW A REPORT    "ami report dekhabo" / "mujhe doctor ko report dikhana hai, unhone bulaya tha" / "I want to show my report":
        asks whether the visit was 7 days ago or earlier. Earlier (7 days or more) -> a new visit: book and pay to come. Within 7 days -> may come and show it.
  3. NEXT SERIAL      "aj book korle serial koto hobe" / "agar aaj book karun to serial kya milega" / "what will my serial be if I book today":
        the number the clinic's own table would give the next booking (clinic-api /api/v1/bookings/next-serial). The number is the DATABASE's; this module only decides the
        sentence is that question and words the reply.

Pure functions, no I/O: every cue is unit-tested (tests/test_visit_cues.py) in English, Bengali and Hindi -- script, romanized and mixed. Cues are deliberately loose on
spelling (the recogniser writes "bikel" a dozen ways) and strict on MEANING (a time-of-day word alone, a report word alone, a serial word alone never trigger anything).
Phrase lists are provisional, written by a non-native speaker (agent/persona.py REVIEW_STATUS): a native reviewer should extend them.
"""

from __future__ import annotations

import re
import unicodedata

_SPLIT = re.compile(r"[^\wऀ-ॿঀ-৿]+", re.UNICODE)


def _nfc(t: str) -> str:
    return unicodedata.normalize("NFC", t or "").lower()


def _tokens(text: str) -> list[str]:
    return [w for w in _SPLIT.split(_nfc(text)) if w]


def _has_stem(tokens: list[str], stems: tuple[str, ...]) -> bool:
    return any(tok.startswith(stems) for tok in tokens)


def _has_word(tokens: list[str], words: frozenset[str]) -> bool:
    return any(tok in words for tok in tokens)


# ------------------------------------------------------------------------------------------------------------------ shared vocabularies
# going / coming, in every spelling a recogniser or a typist produces (Latin Bengali, Latin Hindi, Bengali script, Devanagari, English)
_GO_COME_LATIN = frozenset(
    {
        "jabo", "jaabo", "jabo.", "jai", "jacchi", "jachi", "jachhi", "jete", "jeye", "jaabo", "asbo", "ashbo", "asi", "ashi", "aschi", "aschi", "aschhi", "ashchi", "aste", "ashte", "esho", "eshe", "ese",
        "pouchabo", "pouchhabo", "pouchhe", "pochabo", "pouche", "ghurbo", "ghurte", "jabe", "ashbe", "asben", "jaben", "ashben", "visit",
        "aaunga", "aunga", "ayunga", "ayungi", "aaunga", "aaungi", "aungi", "aaoonga", "aaonga", "aaungaa", "jaunga", "jaungi", "jaoonga", "jaonga", "aana", "ana", "jana", "aaye", "aaenge", "ayenge", "aayenge", "jayenge", "jaenge",
        "pahunchunga", "pahuchunga", "pohonchunga", "pahunchungi", "aaoonga", "aaogi", "aaoge", "jaaunga", "aa", "sakta", "sakti", "sakte", "chalunga", "chalungi", "aata", "aati", "jata", "jati",
        "come", "coming", "go", "going", "reach", "reaching", "visit", "visiting", "arrive", "arriving", "drop", "stop", "turn", "show", "see",
    }
)  # fmt: skip
_GO_COME_STEMS = (
    "যাব", "যাবো", "যাই", "যাচ্ছি", "যেতে", "যেয়ে", "আসব", "আসবো", "আসি", "আসছি", "আসতে", "এসে", "পৌঁছ", "পৌছ", "ঘুরতে", "আসবে", "যাবেন", "আসবেন",
    "आऊंगा", "आऊँगा", "आउंगा", "आऊंगी", "आउंगी", "जाऊंगा", "जाऊँगा", "जाउंगा", "जाऊंगी", "जाउंगी", "आना", "जाना", "आएंगे", "आयेंगे", "जाएंगे", "जायेंगे", "पहुंच", "पहुँच", "आ सकत", "आऊं", "जाऊं",
    "आता", "आती", "जाता", "जाती", "आएगा", "आयेगा", "जाएगा", "आइए", "आओ", "चलूँगा", "चलूंगा",
)  # fmt: skip

# a time of day that is not a clock time: afternoon / evening / noon (the owner's "bikaler dike")
_AFTERNOON_LATIN_STEMS = (
    "bikel", "bikal", "bikeler", "bikaler", "bikale", "bikele", "bikhel", "bical", "bikael", "bekel", "bhikel", "dupur", "dupor", "dupure", "dupurer", "duphur", "dupuer", "dophar", "dopahar", "dopaher", "dopehar", "dupahar",
    "sondhe", "sondhy", "shondhe", "shondhy", "sondh", "shondh", "sandhya", "sandhe", "sanjh", "shaam", "sham", "shyam", "shaam", "afternoon", "evening", "aftrnoon", "eveninng", "noon", "late", "later", "baad", "pore", "porer",
)  # fmt: skip
_AFTERNOON_SCRIPT_STEMS = (
    "বিকেল", "বিকাল", "বিকেলে", "বিকালে", "বিকেলের", "বিকালের", "দুপুর", "সন্ধ্যা", "সন্ধ্যে", "সন্ধে", "পরে", "পরের", "একটু পরে", "দুপুরে",
    "शाम", "दोपहर", "सायं", "सांय", "तीसरे पहर", "बाद में", "बाद", "शाम को", "दोपहर को", "दोपहर में", "बाद में",
)  # fmt: skip
# words that make a "later" word NOT a time of day by themselves ("pore" alone = "later"): only the clear afternoon / evening / noon words open the cue
_CLEAR_AFTERNOON_STEMS = tuple(
    s
    for s in _AFTERNOON_LATIN_STEMS + _AFTERNOON_SCRIPT_STEMS
    if s not in ("late", "later", "baad", "pore", "porer", "পরে", "পরের", "একটু পরে", "बाद में", "बाद", "noon") and len(s) > 2
)

_QUESTION_ABOUT_SITTING = (
    "kota", "koTa", "kokhon", "kakhon", "koyta", "kitne", "kitna", "kab", "kaun", "kon ", "which", "what time", "when", "kobe", "boshen", "bosen", "bose", "bosbe", "sitting", "chamber", "hours", "timing", "time ke",
    "কটা", "কখন", "কবে", "বসেন", "বসে", "বসবে", "কোন", "কতক্ষণ", "कितने", "कब", "कौन", "बैठ", "समय क्या", "कितना",
)  # fmt: skip


def _is_clock_question(text: str) -> bool:
    t = _nfc(text)
    return any(q in t for q in _QUESTION_ABOUT_SITTING)


# ======================================================================================================================= 1. afternoon visit
def says_will_come_in_the_afternoon(text: str) -> bool:
    """sourav, 2026-10-24: "ami bikaler dike jabo", "main shaam ko aaunga", "I will come in the evening", "dupure jabo", "দুপুরে যাব", "शाम को आऊंगा", "bikel e asbo".
    Why: the clinic's answer to a caller who plans to turn up later in the day is always the same -- fine, but keep time in hand and look at your serial. Flow: the sentence
    must carry a clear afternoon / evening / noon word AND a going-or-coming word (the caller's own plan), and must not be a question about the doctor's hours or time
    (those belong to the sitting-hours / when-to-come answers, which are checked first by the caller of this function)."""
    toks = _tokens(text)
    if not toks or len(toks) > 20:
        return False
    if not _has_stem(toks, _CLEAR_AFTERNOON_STEMS):
        return False
    if not (_has_word(toks, _GO_COME_LATIN) or _has_stem(toks, _GO_COME_STEMS)):
        return False
    if _is_clock_question(text) and "?" in (text or ""):
        return False
    return True


_AFTERNOON_REPLY = {
    "bn": "ঠিক আছে, আসতে পারেন। তবে আপনার সিরিয়াল দেখে, হাতে সময় নিয়ে আসবেন।",
    "hi": "जी ज़रूर, आ सकते हैं। लेकिन अपना सीरियल देखकर, हाथ में समय लेकर आइएगा।",
    "en": "Sure, you can come. But please check your serial number and come keeping some time in hand.",
}


def afternoon_reply(lang: str) -> str:
    return _AFTERNOON_REPLY.get(lang, _AFTERNOON_REPLY["en"])


# ======================================================================================================================= 2. show a report
_REPORT_LATIN = frozenset({"report", "reports", "reportta", "reportti", "reportta", "reporter", "riport", "ripor", "ripot", "reprt", "repot", "rapat", "rapot", "reportgulo", "reportguli", "reportgulo"})
_REPORT_STEMS = ("রিপোর্ট", "রিপোট", "রিপর্ট", "रिपोर्ट", "रिपोर्ट्स", "रपट", "रिपोट")
_SHOW_LATIN_STEMS = (
    "dekhabo", "dekhaboo", "dekhai", "dekhate", "dekhano", "dekhaw", "dekhaite", "dekhachi", "dekhacchi", "dekhachhi", "dekhaben", "dekhabe", "dekha",
    "dikha", "dikhana", "dikhane", "dikhaunga", "dikhaungi", "dikhaoonga", "dikhani", "dikhwana", "dikhwane", "dikhao", "dikhaye", "dikhayenge", "dikhana",
    "show", "showing", "shown", "bhalo kore dekh", "check", "dekhte", "dekhbo", "dekhbe",
)  # fmt: skip
_SHOW_SCRIPT_STEMS = ("দেখাব", "দেখাবো", "দেখাতে", "দেখাই", "দেখানো", "দেখাচ্ছি", "দেখাবেন", "দেখতে", "দেখব", "দেখবো", "दिखाना", "दिखाने", "दिखाऊंगा", "दिखाऊँगा", "दिखाउंगा", "दिखाऊंगी", "दिखानी", "दिखाओ", "दिखाएंगे", "दिखाया", "दिखवाना", "दिखाके", "दिखा", "देखना", "देखने", "देखूंगा", "देखाना")
# "the doctor called me / told me to come": bulaya tha, dakechilen, বলেছিলেন আসতে
_CALLED_LATIN_STEMS = ("bulaya", "bulaye", "bulayi", "bulaaya", "bola", "bolechilen", "bolechen", "bolen", "bolechhilen", "dakechilen", "dakechen", "dakechhilen", "daakechilen", "daka", "dakche", "bulaiche", "boleche", "bolchen", "kaha", "kahaa", "bola tha", "bataya", "bulwaya", "asten", "ashte bolechen", "esho bol", "aane ko")  # fmt: skip
_CALLED_SCRIPT_STEMS = ("বলেছিলেন", "বলেছেন", "ডেকেছিলেন", "ডেকেছেন", "ডাকছেন", "ডাকলেন", "আসতে বলেছ", "বলেছিল", "बुलाया", "बुलाए", "बुलाये", "बुला रहे", "बुला रही", "कहा था", "कहा है", "बोला था", "बोला है", "बताया था", "बताया है", "बुलवाया")  # fmt: skip
_PICKUP_WORDS = ("ready", "collect", "pick", "nite", "nibo", "nebo", "nene", "nena", "lena", "lene", "leni", "milega", "milegi", "paoa", "pabo", "পাব", "নিতে", "নেব", "নিব", "কালেক্ট", "রেডি", "तैयार", "रेडी", "लेने", "लेना", "मिलेगी", "मिलेगा", "कलेक्ट", "download", "email", "whatsapp", "পাঠ", "भेज", "send")  # fmt: skip


def wants_to_show_a_report(text: str) -> bool:
    """sourav, 2026-10-24: "ami report dekhabo", "report dekhate chai", "doctor ke report dekhate hobe, tini bolechilen", "mujhe doctor ka report dikhana hai, unhone bulaya tha",
    "I want to show my report to the doctor", "রিপোর্ট দেখাতে আসব", "रिपोर्ट दिखाने आना है". Why: a report is shown to the doctor in a visit -- a follow-up inside the clinic's window is
    free to come to, later than that is a new, paid consultation, so the first thing to know is how long ago the visit was. Flow: a report word AND either a show word or a
    "the doctor told me to come" word; a sentence that is about collecting / downloading / receiving the report (agent/report_pickup's business) is left alone."""
    toks = _tokens(text)
    if not toks or len(toks) > 30:
        return False
    if not (_has_word(toks, _REPORT_LATIN) or _has_stem(toks, _REPORT_STEMS)):
        return False
    t = _nfc(text)
    if any(w in t for w in _PICKUP_WORDS):
        return False
    shows = any(tok.startswith(_SHOW_LATIN_STEMS) for tok in toks) or _has_stem(toks, _SHOW_SCRIPT_STEMS) or "show" in toks
    called = any(tok.startswith(_CALLED_LATIN_STEMS) for tok in toks) or _has_stem(toks, _CALLED_SCRIPT_STEMS)
    return shows or called


_ASK_VISIT = {
    "bn": "আপনি কি ডাক্তারকে সাত দিন আগে, বা তার আগে দেখিয়েছিলেন?",
    "hi": "क्या आपने डॉक्टर को सात दिन पहले, या उससे पहले दिखाया था?",
    "en": "Did you see the doctor seven days ago, or earlier than that?",
}
_REPEAT_VISIT = {
    "bn": "দুঃখিত, বুঝতে পারিনি। ডাক্তার দেখানোর পর কি সাত দিন বা তার বেশি হয়ে গেছে? হ্যাঁ বা না বলুন।",
    "hi": "माफ़ कीजिए, समझ नहीं पाई। डॉक्टर को दिखाए हुए क्या सात दिन या उससे ज़्यादा हो गए हैं? हाँ या नहीं बताइए।",
    "en": "Sorry, I did not get that. Has it been seven days or more since you saw the doctor? Please say yes or no.",
}
_OLD_VISIT = {  # seven days or more: a new visit
    "bn": "তাহলে ডাক্তারকে রিপোর্ট দেখাতে হলে নতুন করে অ্যাপয়েন্টমেন্ট বুক করে পেমেন্ট করে আসতে হবে। বুক করতে চাইলে ডাক্তারের নামটা বলুন।",
    "hi": "तो डॉक्टर को रिपोर्ट दिखाने के लिए नई अपॉइंटमेंट बुक करके पेमेंट करके आना होगा। बुक करना हो तो डॉक्टर का नाम बताइए।",
    "en": "Then, to show the report to the doctor you will need to book an appointment and pay before you come. If you want to book, tell me the doctor's name.",
}
_RECENT_VISIT = {  # within seven days: may come and show it
    "bn": "তাহলে আপনি রিপোর্ট দেখাতে আসতে পারেন। আপনার সিরিয়াল দেখে, হাতে সময় নিয়ে আসবেন।",
    "hi": "तो आप रिपोर्ट दिखाने आ सकते हैं। अपना सीरियल देखकर, हाथ में समय लेकर आइएगा।",
    "en": "Then you can come and show the report. Please check your serial number and come keeping some time in hand.",
}

SEVEN_DAYS = 7


def ask_visit_reply(lang: str) -> str:
    return _ASK_VISIT.get(lang, _ASK_VISIT["en"])


def repeat_visit_reply(lang: str) -> str:
    return _REPEAT_VISIT.get(lang, _REPEAT_VISIT["en"])


def old_visit_reply(lang: str) -> str:
    return _OLD_VISIT.get(lang, _OLD_VISIT["en"])


def recent_visit_reply(lang: str) -> str:
    return _RECENT_VISIT.get(lang, _RECENT_VISIT["en"])


_BN_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯०१२३४५६७८९", "01234567890123456789")
_NUM_WORDS = {
    # English / romanized / Bengali / Hindi number words for the days a caller might say
    "one": 1, "ek": 1, "এক": 1, "एक": 1, "two": 2, "dui": 2, "do": 2, "দুই": 2, "দুদিন": 2, "दो": 2, "three": 3, "tin": 3, "teen": 3, "তিন": 3, "तीन": 3, "four": 4, "char": 4, "chaar": 4, "চার": 4, "चार": 4,
    "five": 5, "panch": 5, "paanch": 5, "pach": 5, "পাঁচ": 5, "পাচ": 5, "पाँच": 5, "पांच": 5, "six": 6, "chhoy": 6, "choy": 6, "chhe": 6, "che": 6, "ছয়": 6, "ছয়": 6, "छह": 6, "छे": 6, "seven": 7, "sat": 7, "saat": 7,
    "সাত": 7, "सात": 7, "eight": 8, "aat": 8, "aath": 8, "আট": 8, "आठ": 8, "nine": 9, "noy": 9, "nau": 9, "নয়": 9, "नौ": 9, "ten": 10, "dosh": 10, "das": 10, "দশ": 10, "दस": 10,
    "fifteen": 15, "pondro": 15, "panadro": 15, "pandrah": 15, "পনেরো": 15, "পনের": 15, "पंद्रह": 15, "twenty": 20, "kuri": 20, "bis": 20, "বিশ": 20, "बीस": 20,
}  # fmt: skip
_DAY_UNIT = ("din", "dine", "dino", "day", "days", "দিন", "দিনে", "दिन", "दिनों", "roj", "rojh")
_WEEK_UNIT = ("week", "weeks", "saptah", "sopthaho", "soptaho", "shoptaho", "সপ্তাহ", "हफ्ता", "हफ्ते", "हफ़्ते", "सप्ताह", "hafta", "hafte", "haphta")
_MONTH_UNIT = ("month", "months", "mash", "mas", "maas", "mahina", "mahine", "মাস", "মাসখানেক", "মাসে", "महीना", "महीने", "महिना", "साल", "year", "bochor", "bachhor", "বছর", "saal", "sal")
_RECENT_WORDS = (
    "kal", "gotokal", "gatokal", "porshu", "parshu", "porsu", "aj", "aaj", "ajke", "today", "yesterday", "recent", "recently", "few days", "kayek din", "koyek din", "kuch din", "this week", "ei soptaho", "is hafte",
    "কাল", "গতকাল", "পরশু", "আজ", "আজকে", "কয়েক দিন", "কদিন", "এই সপ্তাহ", "कल", "परसों", "परसो", "आज", "कुछ दिन", "इस हफ्ते", "इस हफ़्ते", "दो-तीन दिन", "just now", "abhi",
)  # fmt: skip
_OLD_WORDS = (
    "long ago", "long time", "onek din", "onek age", "bohudin", "bahudin", "kaafi din", "kafi din", "bahut din", "bahut pehle", "bahut pahle", "last month", "last year", "earlier", "before that", "ager", "agey", "purano",
    "অনেক দিন", "অনেকদিন", "বহুদিন", "অনেক আগে", "গত মাস", "গত বছর", "আগের", "बहुत दिन", "बहुत पहले", "काफी दिन", "काफ़ी दिन", "पिछले महीने", "पिछले साल", "पुराना",
)  # fmt: skip


def visit_age_answer(text: str, yes_no: str | None) -> str | None:
    """sourav, 2026-10-24: the answer to "did you see the doctor seven days ago, or earlier?" -> "old" (seven days or more), "recent" (fewer), or None (not understood).
    Why: both a plain yes / no ("ha", "na", "haan", "nahi") and the length of time itself ("3 din age", "ek mash age", "kal", "পরশু", "दो हफ्ते पहले") are natural answers.
    Flow: a number with a day / week / month unit decides first (the caller's own figure beats a yes / no); then clear words ("kal", "onek din age"); then the yes / no
    to the question as it was asked (yes = seven days or earlier = old)."""
    t = _nfc(text).translate(_BN_DIGITS)
    toks = _tokens(t)
    for i, tok in enumerate(toks):
        n = int(tok) if tok.isdigit() else _NUM_WORDS.get(tok)
        if n is None:
            continue
        nxt = " ".join(toks[i + 1 : i + 3])
        if any(nxt.startswith(u) for u in _MONTH_UNIT) or any(u in nxt.split() for u in _MONTH_UNIT):
            return "old"
        if any(nxt.startswith(u) for u in _WEEK_UNIT) or any(u in nxt.split() for u in _WEEK_UNIT):
            return "old" if n >= 1 else None
        if any(nxt.startswith(u) for u in _DAY_UNIT) or any(u in nxt.split() for u in _DAY_UNIT):
            return "old" if n >= SEVEN_DAYS else "recent"
    if any(u in toks for u in _MONTH_UNIT) or any(w in t for w in _OLD_WORDS):
        return "old"
    if any(u in toks for u in _WEEK_UNIT):
        return "old"
    if any(w in toks or (" " in w and w in t) for w in _RECENT_WORDS):
        return "recent"
    if len(toks) > 4:
        return None  # a long sentence with a 'not' somewhere ("I do not remember") is not a plain yes / no
    if yes_no == "yes":
        return "old"
    if yes_no == "no":
        return "recent"
    return None


# ======================================================================================================================= 3. next serial
_SERIAL_WORDS = ("serial", "sirial", "seriyal", "sireal", "siriyal", "serail", "sirial", "token", "টোকেন", "সিরিয়াল", "সিরিয়েল", "সিরিয়াল", "सीरियल", "सिरियल", "सीरीयल", "टोकन", "टोकेन", "number", "নম্বর", "नंबर", "নাম্বার", "नम्बर", "no.")
_REAL_SERIAL_WORDS = _SERIAL_WORDS[:-6]  # "number" alone is not a serial
_BOOK_LATIN = ("book", "booking", "buk", "bukking", "register", "naam", "nam lekh", "likhi", "likhale", "likhaye", "likhwa", "korle", "kore", "korbo", "korley", "karun", "karu", "karoon", "karun to", "karwa", "karaun", "karaoon", "karte", "karne", "nile", "nei", "nebo", "nibo", "dile", "dilei", "dilam", "karenge", "karoge", "karun")  # fmt: skip
_BOOK_SCRIPT = ("বুক", "বুকিং", "করলে", "করি", "নিলে", "নাম লেখ", "লেখালে", "করব", "করবো", "করালে", "बुक", "बुकिंग", "कराऊं", "कराऊँ", "करूं", "करूँ", "करवाऊं", "करवाऊँ", "कराएं", "करेंगे", "करूंगा", "नाम लिखा", "कराने", "कराओ", "करो")  # fmt: skip
_WHAT_WILL_LATIN = ("koto", "kto", "kotto", "konta", "kon ", "kotoh", "kitna", "kitne", "kaunsa", "kaun sa", "kya", "kya milega", "milega", "milegi", "hoga", "hogi", "hobe", "pabo", "pabe", "pai", "will be", "would be", "will i get", "would i get", "what will", "which number", "what number", "what would", "what is the serial if", "how many", "number kya", "kon number", "konnumber")  # fmt: skip
_WHAT_WILL_SCRIPT = ("কত", "কতো", "কত নম্বর", "কোন নম্বর", "কোনটা", "হবে", "পাব", "পাবো", "পাই", "कितना", "कितने", "कौन सा", "कौनसा", "क्या", "क्या मिलेगा", "मिलेगा", "मिलेगी", "होगा", "होगी", "कौन सा नंबर")  # fmt: skip
_TODAY = ("aj", "aaj", "ajke", "ajkei", "ajkei", "ekhon", "ekhuni", "abhi", "abhi ke abhi", "today", "now", "right now", "this moment", "aajj", "আজ", "আজকে", "আজই", "এখন", "এখনই", "आज", "आजही", "अभी", "अभी के अभी")  # fmt: skip
_TOMORROW = ("kal", "kalke", "kaal", "tomorrow", "agamikal", "কাল", "কালকে", "আগামীকাল", "कल", "कल को", "आने वाले कल")  # fmt: skip
_COMPLAINT = ("paini", "pai ni", "pacchi na", "aseni", "ashe ni", "ase ni", "didn't", "did not", "not get", "not received", "nahi aaya", "nahi mila", "nahin aaya", "पाইনি".replace("ই", "ই"), "পাইনি", "আসেনি", "নহী", "नहीं आया", "नहीं मिला", "नही आया", "नही मिला")  # fmt: skip


def asks_next_serial(text: str) -> bool:
    """sourav, 2026-10-24: "aj book korle serial koto hobe?", "ami ekhon booking korle kon number pabo", "if I book today what will my serial number be", "agar aaj book karun to
    serial number kya milega", "আজ বুক করলে সিরিয়াল কত হবে", "आज बुक करूं तो सीरियल कितना होगा". Why: for now serials are simulated and generated with every booking, so the next number is
    a plain fact in the table. Flow: a SERIAL word AND a booking word AND a "what will it be" word; a complaint ("I booked but did not get") is the serial-lookup flow's
    and is never taken here; a plain "what is the serial number?" has no booking word and keeps its old answer (it comes on WhatsApp after payment)."""
    t = _nfc(text)
    toks = _tokens(t)
    if not toks or len(toks) > 30:
        return False
    if any(c in t for c in _COMPLAINT):
        return False
    if not (any(w in t for w in _REAL_SERIAL_WORDS) or any(w in t for w in _NUMBER_QUESTION)):
        return False
    if not (any(w in t for w in _BOOK_LATIN + _BOOK_SCRIPT)):
        return False
    if not any(w in t for w in _WHAT_WILL_LATIN + _WHAT_WILL_SCRIPT):
        return False
    # a HYPOTHETICAL booking ("if I book", "book korle", "agar book karun") or an explicit day -- "how do I get a serial after booking?" is the plain serial answer's
    return any(tok in _TODAY or tok in _TOMORROW for tok in toks) or any(c in t for c in _IF_WORDS)


# "which number will I get" with no serial word at all ("ami ekhon book korle kon number pabo")
_NUMBER_QUESTION = ("kon number", "kon nombor", "koto number", "kototam", "kon nambar", "which number", "what number", "kaun sa number", "kaunsa number", "kitna number", "kitne number", "কোন নম্বর", "কত নম্বর", "কোন নাম্বার", "কত নাম্বার", "कौन सा नंबर", "कौनसा नंबर", "कितना नंबर", "कितने नंबर", "कौन सा नम्बर")  # fmt: skip
_IF_WORDS = ("korle", "korley", "nile", "jodi", "if ", "agar", "karun", "karoon", "karein", "karne par", "karoge", "karenge", "kar lun", "করলে", "নিলে", "যদি", "अगर", "यदि", "करूं", "करूँ", "करने पर", "करवाऊं", "कराऊं", "करें", "कर लूं", "korbo to")  # fmt: skip


def serial_day(text: str) -> str:
    """sourav: "today" (the default) or "tomorrow" -- the day the caller means. Why: "kal book korle" asks about tomorrow's list, not today's."""
    t = _nfc(text)
    toks = _tokens(t)
    if any(tok in _TODAY for tok in toks):
        return "today"
    if any(tok in _TOMORROW for tok in toks):
        return "tomorrow"
    return "today"


def next_serial_reply(doctor: str, serial: int, day: str, lang: str) -> str:
    doctor = (
        (doctor or "")
        .replace("Dr. ", "", 1)
        .replace("Dr ", "", 1)
        .replace("ডাঃ ", "", 1)
        .replace("ডাক্তার ", "", 1)
        .replace("डॉ. ", "", 1)
        .replace("डॉक्टर ", "", 1)
    )
    when = {"today": {"bn": "আজ", "hi": "आज", "en": "today"}, "tomorrow": {"bn": "কাল", "hi": "कल", "en": "tomorrow"}}[day]
    lg = lang if lang in ("bn", "hi") else "en"
    return {
        "bn": f"{when[lg]} ডাক্তার {doctor}-এর কাছে বুক করলে আপনার সিরিয়াল নম্বর হবে {serial}।",
        "hi": f"{when[lg]} डॉक्टर {doctor} के पास बुक करने पर आपका सीरियल नंबर {serial} होगा।",
        "en": f"If you book {when[lg]} with Doctor {doctor}, your serial number will be {serial}.",
    }[lg]


def next_serial_not_sitting_reply(doctor: str, day: str, next_date_phrase: str | None, lang: str) -> str:
    """sourav: no serial is promised for a day the doctor does not sit; the next sitting day (spoken by the caller) is named when the table has one."""
    doctor = (doctor or "").replace("Dr. ", "", 1).replace("Dr ", "", 1).replace("ডাঃ ", "", 1).replace("ডাক্তার ", "", 1).replace("डॉ. ", "", 1).replace("डॉक्टर ", "", 1)
    when = {"today": {"bn": "আজ", "hi": "आज", "en": "today"}, "tomorrow": {"bn": "কাল", "hi": "कल", "en": "tomorrow"}}[day]
    lg = lang if lang in ("bn", "hi") else "en"
    tail = {
        "bn": f" তিনি পরের বার {next_date_phrase} বসবেন।" if next_date_phrase else "",
        "hi": f" वे अगली बार {next_date_phrase} बैठेंगे।" if next_date_phrase else "",
        "en": f" He sits next on {next_date_phrase}." if next_date_phrase else "",
    }[lg]
    return {
        "bn": f"ডাক্তার {doctor} {when[lg]} বসছেন না, তাই সিরিয়াল দিতে পারছি না।{tail}",
        "hi": f"डॉक्टर {doctor} {when[lg]} नहीं बैठ रहे हैं, इसलिए सीरियल नहीं बता सकती।{tail}",
        "en": f"Doctor {doctor} is not sitting {when[lg]}, so I cannot give a serial number.{tail}",
    }[lg]


# ======================================================================================================================= 4. how far ahead can I book
import os  # noqa: E402

BOOKING_WINDOW_DAYS = int(os.environ.get("BOOKING_MAX_ADVANCE_DAYS", "3"))  # the clinic API's own limit (clinic-api/main.py MAX_ADVANCE_DAYS) -- same environment name
_WINDOW_Q = (
    "how many days", "how far", "how early", "how much in advance", "in advance", "advance booking", "advance e", "advance me", "ahead", "koto din age", "kotodin age", "koto din agey", "koto din aage", "kotodin aage",
    "kitne din pehle", "kitne din pahle", "kitne din aage", "kitne din ahead", "kab tak booking", "kab tak book", "kobe porjonto", "koto tarikh porjonto", "age theke", "age e book", "age book", "pehle se book", "pehle booking",
    "next month", "agle mahine", "agle mahine ka", "porer mash", "porer mase", "next mash", "কত দিন আগে", "কতদিন আগে", "কত দিন আগে থেকে", "আগে থেকে বুক", "আগে বুক", "অগ্রিম", "পরের মাসে", "পরের মাস", "কবে পর্যন্ত",
    "कितने दिन पहले", "कितने दिन आगे", "पहले से बुक", "पहले बुक", "अग्रिम", "अगले महीने", "कब तक बुक", "कब तक बुकिंग",
)  # fmt: skip
_WINDOW_BOOK = ("book", "booking", "appointment", "serial", "বুক", "অ্যাপয়েন্টমেন্ট", "সিরিয়াল", "बुक", "बुकिंग", "अपॉइंटमेंट", "naam", "নাম", "नाम", "nite", "নেওয়া", "নিতে", "lena", "lete")  # fmt: skip


def asks_booking_window(text: str) -> bool:
    """sourav, 2026-10-24: "koto din age booking hoy?", "kitne din pehle appointment mil jata hai", "how many days in advance can I book", "next month er booking hobe?", "কত দিন আগে
    থেকে বুক করা যায়", "अगले महीने की बुकिंग हो जाएगी?". Why: the clinic takes appointments only a few days ahead (3); the caller asking how far ahead gets that plain answer. Flow: a
    booking word AND a how-far-ahead phrase (or a whole month away, which is always past the window)."""
    t = _nfc(text)
    toks = _tokens(t)
    if not toks or len(toks) > 25:
        return False
    return any(w in t for w in _WINDOW_BOOK) and any(q in t for q in _WINDOW_Q)


def booking_window_reply(lang: str, days: int | None = None) -> str:
    n = days or BOOKING_WINDOW_DAYS
    words = {"bn": {1: "এক", 2: "দুই", 3: "তিন", 4: "চার", 5: "পাঁচ", 6: "ছয়", 7: "সাত"}, "hi": {1: "एक", 2: "दो", 3: "तीन", 4: "चार", 5: "पाँच", 6: "छह", 7: "सात"}, "en": {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven"}}  # fmt: skip
    lg = lang if lang in ("bn", "hi") else "en"
    d = words[lg].get(n, str(n))
    return {
        "bn": f"অ্যাপয়েন্টমেন্ট শুধু {d} দিন আগে পর্যন্ত নেওয়া যায়। তার পরের তারিখের জন্য তারিখের কাছাকাছি সময়ে আবার ফোন করবেন।",
        "hi": f"अपॉइंटमेंट केवल {d} दिन पहले तक ही ली जाती है। उसके बाद की तारीख़ के लिए तारीख़ के नज़दीक फिर से फ़ोन कीजिए।",
        "en": f"Appointments can be taken only up to {d} days ahead. For a later date, please call us again closer to the day.",
    }[lg]

