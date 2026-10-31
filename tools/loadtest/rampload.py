"""Realistic ramp: calls start 10 together, then +5 every 10 s until DURATION s; every call plays its files in order, each only after the AI's previous answer has finished.
usage: rampload.py OUT.json [duration_s=120] [initial=10] [add=5] [every_s=10] [files=g1,d1,d2,d3,d4,d5]"""
import asyncio, io, json, os, sys, time, wave
import websockets

URL = os.environ.get("LOAD_URL", "ws://127.0.0.1:8101/ws/audio")
SR, FRAME = 16000, 320
OUT = sys.argv[1]
DURATION = float(sys.argv[2]) if len(sys.argv) > 2 else 120
INITIAL = int(sys.argv[3]) if len(sys.argv) > 3 else 10
ADD = int(sys.argv[4]) if len(sys.argv) > 4 else 5
EVERY = float(sys.argv[5]) if len(sys.argv) > 5 else 10
FILES = (sys.argv[6] if len(sys.argv) > 6 else "g1,d1,d2,d3,d4,d5").split(",")
TURN_TIMEOUT = 45.0
T0 = time.monotonic()
loop_lag = []


def clip_seconds(b):
    try:
        w = wave.open(io.BytesIO(b))
        return w.getnframes() / w.getframerate()
    except Exception:
        return len(b) / 2 / 22050


def rel():
    return round(time.monotonic() - T0, 2)


async def one_call(cid, pcms, results):
    r = {"call": cid, "start_s": rel(), "turns": [], "outcome": None}
    ev = {"play_end": 0.0, "pending_done": False, "clips": 0, "ai": [], "user": [], "last_ai": None, "first_audio": None, "mark": None, "handoff": False}
    ws = None
    try:
        ws = await websockets.connect(URL, max_size=None, open_timeout=30, ping_interval=None)
        await ws.send(json.dumps({"type": "hello", "sampleRate": SR, "format": "pcm_s16le", "channels": 1}))

        async def receiver():
            try:
                async for m in ws:
                    now = time.monotonic()
                    if isinstance(m, (bytes, bytearray)):
                        d = clip_seconds(bytes(m))
                        ev["clips"] += 1
                        ev["play_end"] = max(ev["play_end"], now) + d
                        ev["pending_done"] = True
                        if ev["mark"] is not None:
                            ev["last_ai"] = now
                            if ev["first_audio"] is None:
                                ev["first_audio"] = now
                    else:
                        d = json.loads(m)
                        if d.get("type") == "handoff_human":
                            ev["handoff"] = True
                            continue
                        if d.get("sender") == "_ping" or d.get("type") in ("config", "stop_playback"):
                            continue
                        if d.get("sender") == "AI":
                            ev["ai"].append((now, d.get("text")))
                            if ev["mark"] is not None:
                                ev["last_ai"] = now
                        elif d.get("sender") == "User":
                            ev["user"].append((now, d.get("text")))
            except Exception:
                pass

        async def acker():
            while True:
                await asyncio.sleep(0.02)
                if ev["pending_done"] and time.monotonic() >= ev["play_end"]:
                    ev["pending_done"] = False
                    try:
                        await ws.send(json.dumps({"type": "playback_done"}))
                    except Exception:
                        return

        rt, at = asyncio.create_task(receiver()), asyncio.create_task(acker())
        # greeting
        t_conn = time.monotonic()
        for _ in range(1500):
            await asyncio.sleep(0.02)
            if ev["clips"] and not ev["pending_done"] and time.monotonic() >= ev["play_end"]:
                break
        else:
            r["outcome"] = "no_greeting"
            raise RuntimeError("no greeting")
        r["greeting_s"] = round(time.monotonic() - t_conn, 2)
        gtext = " ".join(t or "" for _t, t in ev["ai"])
        if ev["handoff"] or "staff" in gtext.lower() or "স্টাফ" in gtext:
            r["outcome"] = "refused"
            r["refusal_text"] = gtext[:120]
            raise RuntimeError("refused")
        nxt = time.monotonic() + 0.3
        for name in FILES:
            pcm = pcms[name]
            ev["ai"].clear(); ev["user"].clear(); ev["first_audio"] = None; ev["last_ai"] = None; ev["mark"] = None
            for off in range(0, len(pcm), FRAME * 2):
                chunk = pcm[off:off + FRAME * 2]
                if len(chunk) < FRAME * 2:
                    chunk += b"\x00" * (FRAME * 2 - len(chunk))
                await ws.send(chunk)
                nxt += FRAME / SR
                await asyncio.sleep(max(0.0, nxt - time.monotonic()))
            ev["mark"] = time.monotonic()
            t_end = ev["mark"]
            silence = b"\x00" * (FRAME * 2)
            done = False
            while time.monotonic() - t_end < TURN_TIMEOUT:
                await ws.send(silence)
                nxt += FRAME / SR
                await asyncio.sleep(max(0.0, nxt - time.monotonic()))
                if ev["last_ai"] is not None and not ev["pending_done"] and time.monotonic() > max(ev["play_end"], ev["last_ai"]) + 1.0:
                    done = True
                    break
            first_text = next((t for t, _x in ev["ai"] if t >= t_end), None)
            turn = {
                "file": name, "end_s": round(t_end - T0, 2),
                "ai_text_ms": round((first_text - t_end) * 1000) if first_text else None,
                "ai_audio_ms": round((ev["first_audio"] - t_end) * 1000) if ev["first_audio"] else None,
                "done": done,
                "heard": " / ".join(x or "" for _t, x in ev["user"])[:90],
                "ai": " | ".join(x or "" for _t, x in ev["ai"])[:160],
            }
            r["turns"].append(turn)
            if not done:
                r["outcome"] = "turn_timeout"
                break
            await asyncio.sleep(0.4)
            nxt = time.monotonic()
        else:
            r["outcome"] = "completed"
        await ws.close()
        rt.cancel(); at.cancel()
    except Exception as e:
        if r["outcome"] is None:
            r["outcome"] = f"error:{type(e).__name__}"
        if ws is not None:
            try:
                await ws.close()
            except Exception:
                pass
    r["end_s"] = rel()
    results.append(r)


async def lag_probe():
    while True:
        t = time.monotonic()
        await asyncio.sleep(0.05)
        loop_lag.append(round((time.monotonic() - t - 0.05) * 1000, 1))


async def main():
    pcms = {n: open(f"/tmp/{n}.pcm", "rb").read() for n in FILES}
    results = []
    tasks = []
    lp = asyncio.create_task(lag_probe())
    cid = 0
    launched_at = []
    def launch(n):
        nonlocal cid
        for _ in range(n):
            cid += 1
            tasks.append(asyncio.create_task(one_call(cid, pcms, results)))
        launched_at.append((rel(), n))
    launch(INITIAL)
    nxt = time.monotonic() + EVERY
    while time.monotonic() - T0 < DURATION:
        await asyncio.sleep(max(0.0, nxt - time.monotonic()))
        if time.monotonic() - T0 >= DURATION:
            break
        launch(ADD)
        nxt += EVERY
    stop_launch = rel()
    await asyncio.gather(*tasks)
    lp.cancel()
    results.sort(key=lambda x: x["call"])
    lag = sorted(loop_lag)
    json.dump({"launched": launched_at, "stop_launch_s": stop_launch, "total_s": rel(), "client_lag_ms_p95": lag[int(0.95 * (len(lag) - 1))] if lag else None,
               "client_lag_ms_max": max(lag) if lag else None, "results": results}, open(OUT, "w", encoding="utf-8"), ensure_ascii=False)
    print("done", len(results), "calls", rel(), "s")


asyncio.run(main())
