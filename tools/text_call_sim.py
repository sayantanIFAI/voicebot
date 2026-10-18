"""Drive the REAL voice-agent turn logic with typed utterances (no microphone, no recogniser) and print what the agent would say.

Everything after the recogniser is the real code: language gates, the intent model (Ollama), the fast path, the booking flow, the clinic API, the reply templates and
the speaking path. Only three things are replaced: what the recogniser "heard" (the text you give it), the voice (a clip of silence), and the websocket (a recorder).
Run it on the pod, where the models and the clinic API are:

    cd /workspace/kolkata-care-voice-agent
    /workspace/venv/bin/python3 tools/text_call_sim.py conversation.txt          # one utterance per line; a line starting with '#' is a note
    /workspace/venv/bin/python3 tools/text_call_sim.py -c "ডাক্তার বিশ্বজিৎ সেনের অ্যাপয়েন্টমেন্ট নেব" "ওই দিনই করে দিন" "হ্যাঁ"

It exists because a spoken test is only as good as the recogniser's luck: the same sentence is heard differently every time, so a regression can hide for a day. Here the
words are exactly the ones typed, so a flow either works or it does not.
"""

from __future__ import annotations

import asyncio
import io
import os
import sys
import tempfile
import types
import wave

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)
sys.path.insert(0, REPO)


def _silence_wav(seconds: float = 0.3, rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(seconds * rate))
    return buf.getvalue()


class _Recorder:
    """A websocket that remembers what the agent sent."""

    def __init__(self):
        self.said: list[str] = []

    async def send_text(self, data: str):
        import json

        try:
            d = json.loads(data)
        except ValueError:
            return
        if d.get("sender") == "AI":
            self.said.append(d["text"])

    async def send_bytes(self, data: bytes):
        return None

    async def close(self, *a, **k):
        return None


async def run(utterances: list[str], lang: str = "bn") -> list[tuple[str, list[str]]]:
    import main
    from agent.asr import ASRResult
    from agent.semantic_cache import SemanticCache

    main._tools = main.ClinicToolsClient(main.CLINIC_API_BASE)
    main._languages_active = ("bn", "hi", "en")
    main._intent_cache = SemanticCache()
    main._fast_path = await main._load_fast_path()

    clip = _silence_wav()

    async def fake_tts(session, clause, language, fallback_reason):  # noqa: ARG001
        return clip

    main._synthesize_one_clause = fake_tts

    heard: dict[str, str] = {"text": ""}

    async def fake_route(session, wav):  # noqa: ARG001
        text = heard["text"]
        script = "hi" if any("ऀ" <= c <= "ॿ" for c in text) else "en" if text.isascii() else "bn"
        return script, ASRResult(text=text, decoder_used="rnnt", decoder_agreement=1.0), True

    main._route_and_transcribe = fake_route

    ws = _Recorder()
    session = main.CallSession(ws)
    session.lang, session.lang_locked = lang, True
    out = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, text in enumerate(utterances):
            heard["text"] = text
            path = os.path.join(tmp, f"u{i}.wav")
            with open(path, "wb") as f:
                f.write(clip)
            ws.said.clear()
            await main._dispatch_turn(session, path)
            out.append((text, list(ws.said)))
    return out


def main_cli() -> None:
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return
    if args[0] == "-c":
        utterances = args[1:]
    else:
        with open(args[0], encoding="utf-8") as f:
            utterances = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    for text, replies in asyncio.run(run(utterances)):
        print(f"[User] {text}")
        for r in replies:
            print(f"[AI]   {r}")
        print()


if __name__ == "__main__":
    main_cli()
