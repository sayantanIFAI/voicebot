"""(tools/barge_probe.py, run on the pod: python tools/barge_probe.py /tmp/some_16k_voice.wav) Does the server react when a human voice talks over the agent's greeting?  Streams a real human recording in real time."""
import asyncio
import json
import sys
import time
import wave

import numpy as np
import websockets

SRC = sys.argv[1] if len(sys.argv) > 1 else "/tmp/rec48.wav"
START_AT = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0  # seconds into the recording to start from
LEAD = float(sys.argv[3]) if len(sys.argv) > 3 else 0.6      # seconds after the greeting arrives before we start talking
w = wave.open(SRC)
assert w.getframerate() == 16000
pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)[int(START_AT * 16000):]
CH = 1600  # 100 ms


async def main():
    t0 = time.time()
    log = []
    async with websockets.connect("ws://127.0.0.1:8101/ws/audio", max_size=None) as ws:
        await ws.send(json.dumps({"type": "hello", "sampleRate": 16000, "format": "pcm_s16le"}))
        got_audio_at = None
        talk_started = None

        async def reader():
            nonlocal got_audio_at
            async for m in ws:
                now = time.time() - t0
                if isinstance(m, bytes):
                    kind = f"audio {len(m)} bytes"
                    if got_audio_at is None:
                        got_audio_at = now
                else:
                    kind = "text " + m[:120]
                log.append((now, kind))
                print(f"[{now:5.2f}s] server -> {kind}", flush=True)

        rd = asyncio.create_task(reader())
        while got_audio_at is None and time.time() - t0 < 20:
            await asyncio.sleep(0.05)
        print(f"greeting audio arrived at {got_audio_at:.2f}s; talking over it from +{LEAD}s", flush=True)
        await asyncio.sleep(LEAD)
        talk_started = time.time() - t0
        i = 0
        # stream the human voice in real time (100 ms chunks), never sending playback_done: the agent is "still speaking"
        while i < len(pcm) and time.time() - t0 < talk_started + 9:
            await ws.send(pcm[i: i + CH].tobytes())
            i += CH
            await asyncio.sleep(0.1)
        await asyncio.sleep(1.5)
        rd.cancel()
    stops = [l for l in log if "stop_playback" in l[1]]
    print("\nRESULT:", "STOP_PLAYBACK received at %.2fs (%.2fs after we started talking)" % (stops[0][0], stops[0][0] - talk_started) if stops else "NO stop_playback: the server ignored the voice talking over it")


asyncio.run(main())
