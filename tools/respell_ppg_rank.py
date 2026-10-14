"""LIMIT, measured 2026-10-15: this scores sound similarity through the Bengali recogniser, which hears Bengali স and শ as the SAME sound, so it cannot see the Hindi-vs-Bengali 'sh' difference a listener hears (a listener's four picks ranked only 4th-7th). Use it to sanity-check, never to override a listener.

Rank candidate spellings by how close the Bengali voice's sounds are to a HUMAN reference recording of the same word.

Speaker-independent: both sides go through the Bengali recogniser's CTC head and are compared as sequences of per-frame sound
(token) probabilities with dynamic time warping, so the voice of the speaker drops out and the SOUND remains.
"""
import json
import sys
import unicodedata
import urllib.request
import wave

import numpy as np

sys.path.insert(0, "/workspace/kolkata-care-voice-agent")
from agent.asr import TurnASR  # noqa: E402

asr = TurnASR(language_id="bn")
TTS = "http://localhost:8002/synthesize"


def posteriors(path):
    asr.model.cur_decoder = "ctc"
    out = asr.model.transcribe([path], batch_size=1, logprobs=True, language_id="bn")
    arr = out[0] if isinstance(out, (list, tuple)) else out
    arr = np.asarray(arr[0] if isinstance(arr, (list, tuple)) else arr)
    if arr.ndim == 3:
        arr = arr[0]
    p = np.exp(arr - arr.max(axis=-1, keepdims=True))
    p /= p.sum(axis=-1, keepdims=True)
    return p  # (T, V)


def trim(p):
    blank = p.shape[1] - 1  # CTC blank is the last class
    speech = np.where(p[:, blank] < 0.9)[0]
    if len(speech) == 0:
        return p
    return p[max(0, speech[0] - 2): speech[-1] + 3]


def dtw(a, b):
    # Bhattacharyya distance between per-frame probability vectors
    sa, sb = np.sqrt(a), np.sqrt(b)
    cost = -np.log(np.clip(sa @ sb.T, 1e-8, 1.0))
    n, m = cost.shape
    acc = np.full((n + 1, m + 1), np.inf)
    acc[0, 0] = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            acc[i, j] = cost[i - 1, j - 1] + min(acc[i - 1, j], acc[i, j - 1], acc[i - 1, j - 1])
    return float(acc[n, m] / (n + m))


def synth(text, name):
    req = urllib.request.Request(TTS, data=json.dumps({"text": text, "lang": "bn", "pauses": False}).encode(), headers={"Content-Type": "application/json"})
    path = f"/tmp/ppg/{name}.wav"
    open(path, "wb").write(urllib.request.urlopen(req, timeout=60).read())
    return path


import os  # noqa: E402

os.makedirs("/tmp/ppg", exist_ok=True)

# the human recording, cut by the pause detector (seconds); দুঃখিত was said as two parts (seg 0 + seg 1)
w = wave.open("/tmp/rec48.wav")
sr = w.getframerate()
x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)


def cut(a, b, name):
    pad = int(0.12 * sr)
    seg = x[max(0, int(a * sr) - pad): min(len(x), int(b * sr) + pad)]
    path = f"/tmp/ppg/human_{name}.wav"
    with wave.open(path, "wb") as o:
        o.setnchannels(1)
        o.setsampwidth(2)
        o.setframerate(sr)
        o.writeframes(seg.tobytes())
    return path


HUMAN = {
    "দুঃখিত": cut(0.74, 2.52, "dukkhito"),
    "কয়েক": cut(4.44, 4.84, "koyek"),
    "নেওয়া": cut(6.28, 6.96, "neoya"),
    "যাওয়ার": cut(8.36, 9.76, "jaoar"),
    "প্রয়োজন": cut(10.48, 11.54, "projojon"),
    "সাড়া": cut(12.70, 13.32, "shara"),
    "সময়টা": cut(14.30, 15.44, "shomoyta"),
}
CANDS = json.load(open("/tmp/ppg_cands.json", encoding="utf-8"))
USER_PICK = {"প্রয়োজন": "প্রোজোন", "সাড়া": "শাঁড়া", "সময়টা": "শমোইটা", "নেওয়া": "নেওয়াা"}

results = {}
for word, hpath in HUMAN.items():
    href = trim(posteriors(hpath))
    rows = []
    for ci, cand in enumerate(dict.fromkeys([word] + CANDS.get(word, []))):
        tp = trim(posteriors(synth(cand, f"{word}_{ci}")))
        rows.append((dtw(href, tp), cand))
    rows.sort()
    results[word] = rows
    plain = next(d for d, c in rows if c == word)
    print(f"\n=== {word}   (plain spelling distance {plain:.3f}; lower = closer to YOUR recording)")
    for rank, (d, c) in enumerate(rows[:8], 1):
        flag = "  <- plain" if c == word else ("  <- YOU PICKED THIS" if USER_PICK.get(word) == c else "")
        print(f"  {rank}. {c:12} {d:.3f}{flag}", flush=True)
    if USER_PICK.get(word):
        pick = USER_PICK[word]
        rank = [c for _, c in rows].index(pick) + 1 if pick in [c for _, c in rows] else None
        print(f"  (your pick ranks #{rank} of {len(rows)} by this measure)")
json.dump({w: [[round(d, 4), c] for d, c in r] for w, r in results.items()}, open("/tmp/ppg/ranking.json", "w"), ensure_ascii=False, indent=1)
print("DONE")
