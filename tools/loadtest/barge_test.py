"""Caller talks over the AI, through the REAL echo-cancelling path (not headphones mode).

The AI's own audio frames are played back into the microphone after a delay (a speaker-to-mic echo), and a recorded caller sentence is added on top while the AI is speaking.
Prints the frames the server sends; passes when a stop_playback arrives soon after the caller starts and the caller's words are recognised.

    python3 barge_test.py ws://localhost:8101/ws/audio question.wav interruption.wav [echo_delay_ms] [echo_gain]
"""

import asyncio
import io
import json
import sys
import time
import wave

import numpy as np
import websockets

RATE = 16000
CHUNK_S = 0.1
TICK = int(RATE * CHUNK_S)


def load(path):
    with wave.open(path) as w:
        sr, ch, n = w.getframerate(), w.getnchannels(), w.getnframes()
        x = np.frombuffer(w.readframes(n), dtype="<i2").astype(np.float32)
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    if sr != RATE:
        x = np.interp(np.arange(int(len(x) * RATE / sr)) / RATE, np.arange(len(x)) / sr, x)
    return x


def decode_wav(b):
    try:
        with wave.open(io.BytesIO(b)) as w:
            sr, ch, n = w.getframerate(), w.getnchannels(), w.getnframes()
            x = np.frombuffer(w.readframes(n), dtype="<i2").astype(np.float32)
        if ch > 1:
            x = x.reshape(-1, ch).mean(axis=1)
        if sr != RATE:
            x = np.interp(np.arange(int(len(x) * RATE / sr)) / RATE, np.arange(len(x)) / sr, x)
        return x
    except Exception:
        return np.zeros(0, np.float32)


async def main(url, question, interruption, delay_ms=150, gain=0.35, headphones=False):
    t0 = time.time()
    echo = np.zeros(RATE * 120, np.float32)  # what the microphone hears of the AI, on the call's timeline
    state = {"play_until": 0.0, "stop_at": [], "texts": [], "caller_started": None, "ai_started": None}
    speech = []

    def say(tag, text):
        print(f"[{time.time() - t0:6.1f}s] [{tag}] {text}", flush=True)

    q, intr = load(question), load(interruption)
    async with websockets.connect(url, max_size=None) as ws:
        await ws.send(json.dumps({"type": "hello", "sampleRate": RATE, "format": "pcm_s16le", "channels": 1, **({"headphones": True} if headphones else {})}))  # without headphones the echo canceller is in the path with the ERLE rule; with it, the phone-line rule

        async def receiver():
            async for m in ws:
                now = time.time()
                if isinstance(m, bytes):
                    x = decode_wav(m)
                    start = max(now, state["play_until"])
                    if state["ai_started"] is None or start > state["play_until"] + 1.0:
                        state["ai_started"] = start
                    state["play_until"] = start + len(x) / RATE
                    at = int((start - t0 + delay_ms / 1000.0) * RATE)
                    if 0 <= at and at + len(x) < len(echo):
                        echo[at : at + len(x)] += x * gain
                else:
                    try:
                        j = json.loads(m)
                    except Exception:
                        continue
                    if j.get("type") == "stop_playback":
                        state["stop_at"].append(now)
                        say("SERVER", "stop_playback")
                    elif "text" in j:
                        state["texts"].append((now, j.get("sender"), j["text"]))
                        say(j.get("sender", "?"), j["text"])

        async def sender():
            n = 0
            nxt = time.time()
            while True:
                chunk = echo[n * TICK : (n + 1) * TICK].copy()
                if speech:
                    buf = speech[0]
                    seg, rest = buf[:TICK], buf[TICK:]
                    if len(rest):
                        speech[0] = rest
                    else:
                        speech.pop(0)
                    chunk[: len(seg)] += seg
                now = time.time()
                if state["play_until"] > now:
                    state["signalled"] = False
                elif state["play_until"] > 0 and now >= state["play_until"] + 0.3 and not state.get("signalled", True):
                    await ws.send(json.dumps({"type": "playback_done"}))
                    state["signalled"] = True
                await ws.send(np.clip(chunk, -32768, 32767).astype("<i2").tobytes())
                n += 1
                nxt += CHUNK_S
                await asyncio.sleep(max(0.0, nxt - time.time()))

        rtask, stask = asyncio.create_task(receiver()), asyncio.create_task(sender())
        await asyncio.sleep(4.0)  # greeting
        say("CALLER", "asks a question")
        speech.append(q)
        await asyncio.sleep(len(q) / RATE + 1.2)
        speech.append(np.zeros(int(RATE * 1.0), np.float32))
        # wait for the AI to be speaking, then talk over it one second in
        for _ in range(200):
            await asyncio.sleep(0.1)
            if state["play_until"] > time.time() + 2.5:
                break
        await asyncio.sleep(1.0)
        state["caller_started"] = time.time()
        say("CALLER", f"talks over the AI ({len(intr) / RATE:.1f}s of speech; the AI has {max(0.0, state['play_until'] - time.time()):.1f}s left to say)")
        speech.append(intr)
        await asyncio.sleep(len(intr) / RATE + 1.0)
        speech.append(np.zeros(int(RATE * 1.5), np.float32))
        await asyncio.sleep(12.0)
        stops = [s - state["caller_started"] for s in state["stop_at"] if s >= state["caller_started"]]
        heard = [t for (ts, snd, t) in state["texts"] if ts >= state["caller_started"] and snd == "User"]
        say("RESULT", f"stop_playback after caller started: {['%.2fs' % s for s in stops] or 'NONE'}; caller's words recognised: {heard or 'NONE'}")
        rtask.cancel()
        stask.cancel()


if __name__ == "__main__":
    a = sys.argv
    asyncio.run(main(a[1], a[2], a[3], float(a[4]) if len(a) > 4 else 150, float(a[5]) if len(a) > 5 else 0.35, len(a) > 6 and a[6] == "hp"))
