"""Run on the pod: python tools/respell_listening_pack.py -> /tmp/pack/*.wav. Edit WORDS first. A native listener picks the best number per word; that listener's choice, not any machine score, decides what goes in agent/bn_respell.py.

(tools/respell_listening_pack.py) Build one listening file per word: the voice says each candidate spelling in turn, numbered, so a native ear can pick."""
import io
import json
import os
import urllib.request
import wave

import numpy as np

TTS = "http://localhost:8002/synthesize"
OUT = "/tmp/pack"
os.makedirs(OUT, exist_ok=True)

# word -> candidate spellings. #1 is always today's spelling, so the listener hears the problem first.
WORDS = {
    "dukkhito": ("দুঃখিত", ["দুঃখিত", "দুক্খিত", "দুখ্খিত", "দুখখিত"]),
    "projojon": ("প্রয়োজন", ["প্রয়োজন", "প্রোয়োজন", "প্রোজোন", "প্রয়োজোন", "প্রোইয়োজন"]),
    "jaoar": ("যাওয়ার", ["যাওয়ার", "জাওয়ার", "জাওআর", "জাউয়ার", "জাওয়ার্"]),
    "shara": ("সাড়া", ["সাড়া", "শাড়া", "শারা", "শাঁড়া"]),
    "shomoyta": ("সময়টা", ["সময়টা", "সমোইটা", "শময়টা", "শমোইটা", "শমইটা"]),
    "koyek": ("কয়েক", ["কয়েক", "কোয়েক", "কইয়েক", "কয়্যেক", "কোইয়েক"]),
    "neoya": ("নেওয়া", ["নেওয়া", "নেয়া", "নেআ", "নেওআ", "নেওয়াা"]),
    "hoye": ("হয়ে", ["হয়ে", "হোয়ে", "হোএ", "হইয়ে", "হয়্যে"]),
}
NUMBERS = ["এক", "দুই", "তিন", "চার", "পাঁচ", "ছয়"]


def synth(text):
    req = urllib.request.Request(TTS, data=json.dumps({"text": text, "lang": "bn", "pauses": False}).encode(), headers={"Content-Type": "application/json"})
    raw = urllib.request.urlopen(req, timeout=60).read()
    with wave.open(io.BytesIO(raw)) as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16), w.getframerate()


manifest = {}
for key, (word, cands) in WORDS.items():
    pieces, rate = [], None
    for i, cand in enumerate(cands):
        num, rate = synth(NUMBERS[i])
        wav, _ = synth(cand)
        wav2, _ = synth(cand)  # said twice, so a listener can catch it
        gap = np.zeros(int(rate * 0.35), dtype=np.int16)
        long_gap = np.zeros(int(rate * 0.9), dtype=np.int16)
        pieces += [num, gap, wav, gap, wav2, long_gap]
    out = np.concatenate(pieces)
    path = f"{OUT}/{key}.wav"
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(out.tobytes())
    manifest[key] = {"word": word, "candidates": {str(i + 1): c for i, c in enumerate(cands)}}
    print(key, word, len(cands), "candidates", round(len(out) / rate, 1), "s", flush=True)
json.dump(manifest, open(f"{OUT}/manifest.json", "w"), ensure_ascii=False, indent=1)
print("DONE")
