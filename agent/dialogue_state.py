"""What the conversation is about RIGHT NOW, in one place (owner, 2026-10-31).

The call's state used to be a dozen separate attributes of CallSession (subject_doctor, dept_topic, dept_pending, last_route, name_pick, doctor_check, last_doctor_enquiry, pending_enquiry, the booking ...), each
read by its own rule. This module is the single READ model of them: `main._dialogue_snapshot(session)` fills a plain dict, and everything that needs "what did the assistant just say, what question is open, who is
'he'" reads that dict. New conversational state is added here first.

Two uses:
  * the intent model is told the context (`render`), so a short follow-up ("কে আছে", "কোন ডাক্তার এটা", "হ্যাঁ ঠিক আছে", "ওটা") is understood from what was just said instead of from the words alone;
  * a clarifying question can name the OPEN question ("I had asked ...") instead of a generic "I did not understand" (`open_question_line`).

The context is only for understanding what the caller refers to or answers. It is never a source of facts: prices, times and names still come from the clinic's tables.
"""

from __future__ import annotations

MAX_LAST = 300


def render(snap: dict) -> str:
    """The context block put in the intent model's prompt; empty when there is nothing to say (the first turn)."""
    if not snap:
        return ""
    lines = []
    if snap.get("last_ai"):
        lines.append(f'- The assistant\'s last sentence to the caller: "{snap["last_ai"][:MAX_LAST]}"')
    if snap.get("open_question"):
        lines.append(f"- The question the assistant is waiting for the caller to answer: {snap['open_question']}")
    if snap.get("doctor"):
        lines.append(f"- The doctor being talked about: {snap['doctor']}")
    if snap.get("department"):
        lines.append(f"- The department being talked about: {snap['department']}")
    if snap.get("booking"):
        lines.append(f"- A booking is in progress: {snap['booking']}")
    if not lines:
        return ""
    return (
        "CONVERSATION CONTEXT (use it ONLY to understand what the caller is referring to or answering -- 'he', 'this one', 'yes', 'who is there', 'say it again'. It is never a source of facts, and a slot is still "
        "filled only from the caller's own words):\n" + "\n".join(lines) + "\nIf the utterance is a short follow-up that only makes sense given this context, choose the intent that continues THIS topic; "
        'if the caller asks to hear the last sentence again, the intent is "repeat_last".'
    )


def signature(snap: dict) -> str:
    """A short key of the OPEN QUESTION, for the intent cache: the same words mean different things under a different open question ("yes" to a booking vs "yes" to a doctor check)."""
    if not snap:
        return ""
    oq = (snap.get("open_question") or "")[:40]
    bk = (snap.get("booking") or "")[:30]
    return f"{oq}|{bk}" if (oq or bk) else ""


def open_question_line(snap: dict, lang: str) -> str:
    """"I had asked: <the open question>" in the caller's language -- for a clarification that names what is still waiting for an answer. Empty when no question is open."""
    q = (snap or {}).get("open_question_spoken") or ""
    if not q:
        return ""
    return {"bn": f"আমি জানতে চেয়েছিলাম: {q}", "hi": f"मैंने पूछा था: {q}", "en": f"I had asked: {q}"}.get(lang, f"I had asked: {q}")
