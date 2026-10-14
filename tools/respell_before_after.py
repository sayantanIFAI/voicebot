"""Run on the pod after changing agent/bn_respell.py: renders each word/sentence as spelled before and as the bot now sends it, for a listener to compare.

Before/after listening files for the sh rule: each word (and three sentences) said first as spelled today, then as the bot now sends it."""
import io
import json
import os
import sys
import urllib.request
import wave

import numpy as np

sys.path.insert(0, "/workspace/kolkata-care-voice-agent")
from agent import bn_respell  # noqa: E402

TTS = "http://localhost:8002/synthesize"
os.makedirs("/tmp/shapack", exist_ok=True)


def synth(text):
    req = urllib.request.Request(TTS, data=json.dumps({"text": text, "lang": "bn", "pauses": False}).encode(), headers={"Content-Type": "application/json"})
    raw = urllib.request.urlopen(req, timeout=60).read()
    with wave.open(io.BytesIO(raw)) as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16), w.getframerate()


def build(name, items, between=0.45, after=1.2):
    pieces, rate, key = [], 22050, []
    for text in items:
        new = bn_respell.apply(text, "bn")
        a, rate = synth(text)
        b, _ = synth(new)
        pieces += [a, np.zeros(int(rate * between), dtype=np.int16), b, np.zeros(int(rate * after), dtype=np.int16)]
        key.append((text, new))
    out = np.concatenate(pieces)
    with wave.open(f"/tmp/shapack/{name}.wav", "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(out.tobytes())
    print(name, round(len(out) / rate, 1), "s")
    for i, (t, n) in enumerate(key, 1):
        print(f"  {i}. {t}  ->  {n}")


build("words_before_then_after", ["সময়", "সাহায্য", "সেটা", "সাথে", "সরাসরি", "সব", "সকাল", "সুবিধা", "সঠিক", "সেন", "সাত", "সাড়ে", "সত্তর", "সৌরভ"])
build(
    "sentences_before_then_after",
    ["রিপোর্ট সাত দিনের মধ্যে পাবেন। কোনো সমস্যা হলে আমাদের সাথে যোগাযোগ করুন।", "আপনি কি সকাল সাড়ে সাতটায় আসতে পারবেন?", "ডাক্তার সেন সোমবার সকাল দশটা থেকে বসেন।", "দুঃখিত, সময়টা ধরে রাখা যায়নি। প্রয়োজন হলে আবার বলুন।"],
    between=0.7,
    after=1.6,
)
print("DONE")
