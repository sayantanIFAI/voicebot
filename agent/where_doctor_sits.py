"""Which floor / room does the doctor sit in? (owner, 2026-10-29)

The clinic's tables hold no floor or room, so the agent never names one. It says the counter tells it: "কোন তলায় বসবেন সেটা ক্লিনিকের কাউন্টারে জানিয়ে দেওয়া হবে।" The question is recognised in the three
languages, in their own scripts or in Latin letters, and only when it is about a floor / room / chamber place (a plain "when does he sit?" is not this).
"""

from __future__ import annotations

import re

_FLOOR = re.compile(
    r"তলা|তলায়|তলার|ফ্লোর|ফ্লোরে|"
    r"मंज़िल|मंजिल|मंज़िला|माले|फ्लोर|फ़्लोर|"
    r"\b(?:floor|storey|story|tola|tolay|tolar|manzil|manjil|manjhil)\b",
    re.I,
)
_ROOM = re.compile(
    r"(?:রুম|চেম্বার|ঘর|কক্ষ|কামরা)\s*(?:নম্বর|নাম্বার|নং|কত|কোন|কোথায়|কোথায়)|কোন\s*(?:রুম|চেম্বার|ঘর|কক্ষ)|কোথায়\s*বসেন|কোথায়\s*বসবেন|কোথায়\s*বসছেন|"
    r"(?:कमरा|कमरे|रूम|चेंबर|चैम्बर)\s*(?:नंबर|नम्बर|कौन|कितने|कहाँ|कहां)|कौन\s*(?:सा|से)\s*(?:कमरे|कमरा|रूम|चेंबर)|कहाँ\s*बैठ|कहां\s*बैठ|"
    r"\b(?:room|chamber|cabin)\s*(?:number|no|num)\b|\bwhich\s+(?:room|chamber|cabin)\b|\bwhere\s+(?:will|does|do|is)\b.{0,25}\b(?:sit|sits|sitting|seated)\b|"
    r"\b(?:kon|kon\s*ta|konta|kothay)\s*(?:room|ghor|ghore|chamber)\b|\b(?:room|ghor|chamber)\s*(?:number|no)\b|\bkothay\s*(?:boshe|boshben|bosben|boshbe|bosche|boschen)\b",
    re.I,
)
# a floor word alone ("তলা") is also in other sentences; it counts with a sitting / doctor / which-word beside it
_CONTEXT = re.compile(
    r"কোন|কত|কোথা|বসে|বসবে|বসেন|বসছে|বসছেন|ডাক্তার|ডাঃ|চেম্বার|দেখাব|দেখান|"
    r"कौन|कितन|कहाँ|कहां|बैठ|डॉक्टर|डाक्टर|चेंबर|"
    r"\b(?:which|what|where|sit|sits|sitting|doctor|dr|chamber|kon|konta|kothay|bosh\w*|daktar|doctor)\b",
    re.I,
)

REPLY = {
    "bn": "কোন তলায় বা কোন ঘরে ডাক্তার বসবেন, সেটা ক্লিনিকের কাউন্টারে জানিয়ে দেওয়া হবে।",
    "hi": "डॉक्टर किस मंज़िल या किस कमरे में बैठेंगे, यह क्लिनिक के काउंटर पर बता दिया जाएगा।",
    "en": "Which floor or room the doctor sits in will be told at the clinic counter.",
}


# Owner, 2026-10-31 ("সেন্টারটা কি পাঁচতলা", "ডাক্তাররা কোথায় বসে দোতালাতে" were answered with the clinic's ADDRESS): a floor COUNT or a named floor -- পাঁচতলা, দোতালা, তেতলা, "second floor", दूसरी मंज़िल --
# is talk about the building, which the assistant knows nothing about. It is not answered with the address; the caller is asked what they need, which is a doctor's appointment.
_STATED_FLOOR = re.compile(
    r"(?:এক|দু|দো|তিন|তে|চার|পাঁচ|পাচ|ছয়|সাত|আট|নয়|দশ|[০-৯0-9]+)\s*(?:তলা|তালা|তলায়|তালায়|তলাতে|তালাতে|তলার|তালার)|দোতলা|দোতালা|তেতলা|তেতালা|তিনতলা|চারতলা|পাঁচতলা|ছয়তলা|সাততলা|আটতলা|নয়তলা|দশতলা|"
    r"(?:ground|first|second|third|fourth|fifth|top)\s+floor|\b(?:two|three|four|five|six|seven|eight|nine|ten)[\s-]*stor(?:e)?y(?:ed)?\b|"
    r"(?:पहली|दूसरी|तीसरी|चौथी|पांचवीं|पाँचवीं)\s*मंज़िल|(?:दो|तीन|चार|पांच|पाँच|छह|सात|आठ)\s*मंज़िला|(?:दो|तीन|चार|पांच|पाँच|छह|सात|आठ)\s*मंजिला",
    re.I,
)
HELP = {
    "bn": "বলুন, ডাক্তারের অ্যাপয়েন্টমেন্টের ব্যাপারে আমি আপনাকে কীভাবে সাহায্য করতে পারি?",
    "hi": "बताइए, डॉक्टर के अपॉइंटमेंट के बारे में मैं आपकी कैसे मदद कर सकती हूँ?",
    "en": "Please tell me how I can help you with a doctor's appointment.",
}


def states_a_floor(text: str) -> bool:
    """A numbered / named floor in a short sentence ("পাঁচতলা", "দোতালাতে", "on the second floor")."""
    t = text or ""
    return len(t.split()) <= 25 and _STATED_FLOOR.search(t) is not None


def help_reply(lang: str) -> str:
    return HELP.get(lang, HELP["en"])


def asks_floor(text: str) -> bool:
    t = text or ""
    if len(t.split()) > 25:
        return False
    if _ROOM.search(t):
        return True
    return bool(_FLOOR.search(t) and _CONTEXT.search(t))


def reply(lang: str) -> str:
    return REPLY.get(lang, REPLY["en"])
