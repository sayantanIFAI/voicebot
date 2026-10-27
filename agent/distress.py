"""A caller who is frightened, in pain or at the end of their patience (owner, 2026-10-27).

The call state "distressed" (agent/call_state.py, Appendix C row in agent/speech_policy.py) already changes HOW the agent speaks -- slower, short sentences, one question at a time -- and has an
approved acknowledgement (agent/acknowledgement.py), but nothing could put a call into it. This module is that detector. It reads what the caller SAYS and DOES, never a borrowed emotion model:
public models are trained on acted English speech and are unproven on Kolkata phone audio (a SpeechBrain trial on the pod was removed again the same day).

Signals, strongest first:
  * WORDS (strong -- one turn is enough): "I am scared", "I cannot bear it", "save us", the same in Bengali and Hindi. Medical emergencies stay with agent/emergency.py, which runs first.
  * BEHAVIOUR (soft -- needs two turns out of the last three): the caller repeats themselves; two unanswered turns in a row; a clearly louder, higher voice than their own first turns; soft
    urgency words ("please hurry"). Loudness and pitch are compared with the caller's OWN opening turns, never an absolute number.
The state goes off again after three quiet turns.

REASONED, NOT MEASURED: the word lists were written by a non-native reader and the numbers below come from no recording. A native reader should extend the lists (tests/test_distress.py is the place
to record phrasings), and a listening pass on real calls should tune the numbers. A false trigger costs one kind sentence and slower speech; a miss costs a stressed caller being treated as routine.
"""

from __future__ import annotations

import re
import unicodedata
from collections import deque

# Said plainly, these are distress by themselves.
_STRONG = (
    # Bengali
    "ভয় পাচ্ছি", "ভয় লাগছে", "খুব ভয়", "ভীষণ ভয়", "আর পারছি না", "সহ্য করতে পারছি না", "সহ্য হচ্ছে না", "অসহ্য", "বাঁচান", "বাঁচাও", "খুব বিপদে", "বিপদে পড়েছি", "খুব চিন্তায়",
    "খুব টেনশন", "ভীষণ টেনশন", "মাথা ঠিক নেই", "কান্না পাচ্ছে", "কাঁদছি", "হাত পা কাঁপছে", "দিশেহারা", "কী করব বুঝতে পারছি না", "কি করব বুঝতে পারছি না",
    # Hindi
    "डर लग रहा", "बहुत डर", "घबरा", "सहन नहीं हो रहा", "सहा नहीं जा रहा", "अब और नहीं", "बचाइए", "बचा लो", "बचा लीजिए", "बहुत परेशान", "बहुत टेंशन", "बहुत चिंता", "रो रहा", "रो रही",
    "समझ नहीं आ रहा क्या करूँ", "समझ नहीं आ रहा क्या करूं", "मुसीबत में",
)
_STRONG_EN = re.compile(
    r"\b(i am|i'm|we are|we're) (so |very |really )?(scared|frightened|terrified|panicking|desperate)|"
    r"\b(so|very|really) (scared|frightened|worried|desperate)\b|\bpanic(king)?\b|"
    r"\b(can'?t|cannot|can not) (take|bear|stand|cope with) (it|this|any more|anymore)\b|"
    r"\b(i am|i'm|we are|we're) crying\b|\bsave (us|him|her|me)\b|\bi don'?t know what to do\b|\bi'?m (in )?(a )?(lot of )?trouble\b",
    re.I,
)
# Urgency alone is not distress (people say "quickly please" about a booking), so it only counts together with something else.
_SOFT = (
    "তাড়াতাড়ি করুন", "প্লিজ তাড়াতাড়ি", "দয়া করে তাড়াতাড়ি", "এখুনি", "এক্ষুনি", "খুব জরুরি", "ভীষণ জরুরি", "অনেকক্ষণ ধরে", "কেউ শুনছে না", "কেউ ধরছে না", "বারবার বলছি", "কতবার বলব",
    "जल्दी कीजिए", "प्लीज़ जल्दी", "प्लीज जल्दी", "अभी के अभी", "बहुत ज़रूरी", "बहुत जरूरी", "कोई सुन नहीं रहा", "कितनी बार बोलूँ", "कितनी बार बोलूं", "बार बार बोल रहा", "बार-बार बोल रहा",
)
_SOFT_EN = re.compile(r"\b(please|pls) (hurry|be quick|quickly)\b|\bhurry up\b|\bright now\b|\bvery urgent\b|\bno one (is )?(listening|answering)\b|\bhow many times\b|\bi already (said|told)\b", re.I)

LOUD_DB = 8.0  # louder than the caller's own opening turns by this much ...
HIGH_PITCH = 1.2  # ... and this much higher (ratio of the voice's fundamental)
BASELINE_TURNS = 2
WINDOW = 3
NEEDED = 2  # soft turns out of the last WINDOW
QUIET_TO_CLEAR = 3
REPEAT_SIMILARITY = 0.8

_PUNCT = re.compile(r"[.,;:!?\"'()\[\]{}।॥\-–—_/…]+")


def _norm(text: str) -> str:
    return " ".join(_PUNCT.sub(" ", unicodedata.normalize("NFC", text or "").lower()).split())


def strong_words(text: str) -> bool:
    t = _norm(text)
    return bool(t) and (any(w in t for w in _STRONG) or _STRONG_EN.search(t) is not None)


def soft_words(text: str) -> bool:
    t = _norm(text)
    return bool(t) and (any(w in t for w in _SOFT) or _SOFT_EN.search(t) is not None)


def _similar(a: str, b: str) -> bool:
    """Roughly the same sentence said again (word overlap), for 'the caller is repeating themselves'."""
    wa, wb = set(a.split()), set(b.split())
    if len(wa) < 3 or len(wb) < 3:
        return False
    return len(wa & wb) / max(len(wa), len(wb)) >= REPEAT_SIMILARITY


class DistressEvidence:
    """One per call. observe() reads a turn and returns "on", "off" or None (no change)."""

    def __init__(self) -> None:
        self.active = False
        self._soft: deque[bool] = deque(maxlen=WINDOW)
        self._quiet_run = 0
        self._previous_text = ""
        self._levels: list[float] = []
        self._pitches: list[float] = []
        self.reason = ""

    def _loud_and_high(self, near_end: tuple[float, float] | None) -> bool:
        if not near_end:
            return False
        level, pitch = near_end
        if len(self._levels) < BASELINE_TURNS:
            self._levels.append(level)
            if pitch:
                self._pitches.append(pitch)
            return False
        base_level = sum(self._levels) / len(self._levels)
        base_pitch = sum(self._pitches) / len(self._pitches) if self._pitches else 0.0
        return level - base_level >= LOUD_DB and bool(base_pitch) and bool(pitch) and pitch / base_pitch >= HIGH_PITCH

    def observe(self, text: str, *, unanswered_in_a_row: int = 0, near_end: tuple[float, float] | None = None) -> str | None:
        norm = _norm(text)
        strong = strong_words(text)
        soft_reasons = []
        if soft_words(text):
            soft_reasons.append("urgency words")
        if norm and self._previous_text and _similar(norm, self._previous_text):
            soft_reasons.append("repeating themselves")
        if unanswered_in_a_row >= 2:
            soft_reasons.append("two unanswered turns")
        if self._loud_and_high(near_end):
            soft_reasons.append("louder and higher than their opening turns")
        if norm:
            self._previous_text = norm
        self._soft.append(bool(soft_reasons))
        signal = strong or bool(soft_reasons)
        self._quiet_run = 0 if signal else self._quiet_run + 1
        if not self.active:
            if strong or sum(self._soft) >= NEEDED:
                self.active = True
                self.reason = "distress words" if strong else ", ".join(soft_reasons)
                return "on"
        elif self._quiet_run >= QUIET_TO_CLEAR:
            self.active = False
            self._soft.clear()
            return "off"
        return None


# "connect me to a person": the way out the calm-delivery acknowledgement promises. Only phrases that name talking to / being connected to someone (not the word "staff" alone, which
# ordinary questions use).
_PERSON = (
    "মানুষের সাথে কথা", "মানুষের সঙ্গে কথা", "কারও সাথে কথা বলিয়ে", "কাউকে দিন", "কাউকে ফোনটা দিন", "স্টাফের সাথে কথা", "স্টাফের সাথে যুক্ত", "স্টাফের সঙ্গে কথা", "রিসেপশনে দিন",
    "রিসেপশনিস্টের সাথে", "অপারেটরের সাথে", "জীবন্ত কারও", "আসল মানুষ",
    "किसी इंसान से", "किसी इन्सान से", "किसी से बात करा", "स्टाफ से बात", "स्टाफ से जोड़", "रिसेप्शन पर दीजिए", "ऑपरेटर से बात", "असली इंसान",
)
_PERSON_EN = re.compile(
    r"\b(speak|talk|connect|put|transfer)\b[^.]{0,25}\b(to|with|me to|me with)\b[^.]{0,15}\b(a |the |some )?(person|human|someone|somebody|staff|receptionist|operator|agent)\b|\breal (person|human)\b",
    re.I,
)


def asks_for_person(text: str) -> bool:
    t = _norm(text)
    return bool(t) and (any(w in t for w in _PERSON) or _PERSON_EN.search(t) is not None)
