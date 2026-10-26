"""The eyes (owner, 2026-10-26).

A caller said "আমার চোখটা অনেক জ্বালা করছে, কোনো ডাক্তার আছে?" (my eye burns -- is there a doctor?) and was told "I could not tell which department", which sounds as if the agent did not
understand. The clinic has no eye department, and that is a plain fact the caller should hear. clinic-api carries an "Ophthalmology" route row (booking_migrate.DEPARTMENT_ROUTE_KEYWORDS) that
only takes effect once a clinic has such a department; THIS module is used only when nothing matched, so a clinic that adds one is never told it has none.
"""

from __future__ import annotations

import re

_EYE = re.compile(
    r"(?<![a-z])(?:eyes?|eyesight|vision|chokh\w*|chokher|ankh\w*|aankh\w*)(?![a-z])|চোখ|চক্ষু|आँख|आंख|आँखों|आंखों|नज़र|नजर",
    re.I,
)


def mentions_eye(text: str) -> bool:
    return bool(text) and _EYE.search(text) is not None


NO_EYE_DEPARTMENT = {
    "bn": "দুঃখিত, আমাদের এখানে চোখের কোনো বিভাগ বা ডাক্তার নেই। অন্য কোনো সমস্যা হলে বলুন, সেই বিভাগের ডাক্তার বলে দিতে পারব।",
    "hi": "माफ़ कीजिए, हमारे यहाँ आँखों का कोई विभाग या डॉक्टर नहीं है। कोई और समस्या हो तो बताइए, उस विभाग के डॉक्टर बता दूँगी।",
    "en": "Sorry, we do not have an eye department or an eye doctor here. If you have another problem, tell me and I can say which doctors see it.",
}
