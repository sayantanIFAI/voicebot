"""Hindi spoken, but the recogniser wrote it in Bengali letters (owner, 2026-10-30: "নিউরোসার্জন কনকন হ্যাঁ উধার" -- Hindi "kaun kaun ... udhar").

The Bengali recogniser writes Hindi sounds with Bengali spellings, so the written script says nothing. What does say it is a Hindi word that Bengali does not have: "উধার" (उधर, over there), "ক্যা" (क्या), "মেরা" (मेरा), "চাহিয়ে" (चाहिए). A word is compared by its consonant SOUND skeleton (agent/name_script.name_skeleton, the same for Bengali, Devanagari and Latin spellings), so any
spelling of the sound counts. Only Hindi words whose skeleton is NOT also a common Bengali word are kept ("कौन" is the same sounds as Bengali "কোন": it proves nothing, so it is not here).

This is a signal, not a verdict: it only decides whether the Hindi recogniser is asked to read the clip as well (agent/main.py), and the Hindi reading must itself be written in Devanagari and agree with itself.
"""

from __future__ import annotations

import re

from agent import name_script

# Hindi words that Bengali does not share (Devanagari; compared by sound). Function words and the commonest request words of a clinic call.
_HINDI = """
उधर इधर यहाँ यहां वहाँ वहां कहाँ कहां क्या कैसे कितना कितने कितनी कौनसा कौनसे कोई कुछ
है हैं हूँ हूं था थी थे होगा होगी होंगे रहा रही रहे सकता सकती सकते चाहिए चाहिये चाहता चाहती
मैं मुझे मुझको मेरा मेरी मेरे हमारा हमारी तुम तुम्हारा आपका आपकी आपके उनका उनकी उनके इनका इनकी
नहीं नही नहीं बताइए बताइये बताओ बोलिए बोलिये बोलो कीजिए कीजिये दीजिए दीजिये करना करो करें करूँ करूं
अभी अब फिर दोबारा बहुत थोड़ा बाद पहले लेकिन मगर क्योंकि इसलिए अच्छा सही जरूर ज़रूर बिलकुल
मिलेगा मिलेगी मिलेंगे बैठते बैठता बैठती बैठेंगे आएंगे जाएंगे
का की के को से में पर और यह ये वो वह तो भी ही
""".split()

_STOP = frozenset({"और"})  # too short / shared once reduced to a skeleton (checked below by length)


def _skeleton_set() -> frozenset[str]:
    out = set()
    for w in _HINDI:
        sk = name_script.name_skeleton(w)
        if len(sk) >= 3:  # one or two consonants collide with half the Bengali dictionary
            out.add(sk)
    return frozenset(out)


_SKELETONS = _skeleton_set()
# Bengali words whose skeleton equals one of the above: never evidence (kept short, found by checking the list against Bengali clinic phrasing)
_BENGALI_COLLISIONS = frozenset(
    name_script.name_skeleton(w)
    for w in "কোন কোনটা কেমন কেমনে কত কতটা আছে আছেন হবে হয় হলো থাকবে বলুন বলেন বলো ঠিক ভালো অনেক আজকে কালকে এখন পরে আবার একটু মানে".split()
)


def hindi_words(text: str) -> int:
    """How many words of a Bengali-script sentence are Hindi words Bengali does not have."""
    n = 0
    for tok in re.findall(r"[ঀ-৿]+", text or ""):
        sk = name_script.name_skeleton(tok)
        if len(sk) >= 3 and sk in _SKELETONS and sk not in _BENGALI_COLLISIONS:
            n += 1
    return n


def looks_hindi(text: str) -> bool:
    """A Bengali-script sentence that carries Hindi-only words: at least one, and at least a fifth of its words."""
    toks = re.findall(r"[ঀ-৿]+", text or "")
    if not toks:
        return False
    k = hindi_words(text)
    return k >= 1 and k * 5 >= len(toks)
