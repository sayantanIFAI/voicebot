"""Owner, 2026-10-26: a phone or confirmation number read back as one flat, unbroken run of digits has no natural
breath in it. agent/figures.py (KCD-157) already owns the grouping RULE for exactly this ("numbers spoken at a pace
a caller can write down") -- phone_groups() for the chunk boundaries, GROUP_SEPARATOR for the prosodic break every
FastPitch checkpoint here renders as a pause. That module's own speak_grouped() turns each chunk into spoken WORDS
per language (KCD-445's verbalize_en/_hi/_bn pipeline); agent/booking_capture.py's own prompts have a different,
deliberate convention -- the digits stay as literal characters ("9 8 7 6 5"), not spelled-out words, one at a time
-- so this module reuses figures.py's grouping only, not its word-spelling, to keep that convention unchanged.
"""

from __future__ import annotations

from agent.figures import GROUP_SEPARATOR, phone_groups


def spoken_digit_groups(digits) -> str:
    """`digits` ("7079792145", or any iterable of digit characters/substrings) -> "7 0 7 9 7 , 9 2 1 4 5" -- the
    same 5+5 (ten digits) / four-at-a-time (any other length) grouping agent.figures.phone_groups already uses for
    the general verbalizer, with the same standalone-comma pause between groups; only the digits inside each group
    stay literal characters, one at a time, matching how agent/booking_capture.py has always spoken them."""
    flat = "".join(str(d) for d in digits) if digits else ""
    if not flat:
        return ""
    return GROUP_SEPARATOR.join(" ".join(group) for group in phone_groups(flat))
