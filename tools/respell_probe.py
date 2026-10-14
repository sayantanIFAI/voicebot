"""Search for respellings of Bengali words with য় that the Bengali voice says well enough for the recogniser to read back."""
import json
import sys
import unicodedata
import urllib.request

sys.path.insert(0, "/workspace/kolkata-care-voice-agent")
from agent.asr import TurnASR  # noqa: E402

TTS = "http://localhost:8002/synthesize"
YA = "য়"  # য়  (the voice's vocabulary has it as one letter)
nfc = lambda s: unicodedata.normalize("NFC", s or "")  # noqa: E731


def synth(text, name):
    req = urllib.request.Request(TTS, data=json.dumps({"text": text, "lang": "bn"}).encode(), headers={"Content-Type": "application/json"})
    path = f"/tmp/respell/{name}.wav"
    open(path, "wb").write(urllib.request.urlopen(req, timeout=60).read())
    return path


# word -> (hand-written extra candidates, two carrier sentences)
WORDS = {
    "সময়টা": (["সোময়টা", "সমইটা", "সমুয়টা", "শমযটা", "সময়্টা", "সমোইটা", "সমইয়টা"], ["দুঃখিত, {w} ধরে রাখা যায়নি।", "আপনার {w} কখন?"]),
    "কয়েক": (["কইয়েক", "কয়্যেক", "কোয়েক", "কযেক", "কোইয়েক", "কইএক", "কোয়াক"], ["রিপোর্ট {w} ঘণ্টার মধ্যে পাবেন।", "{w} সেকেন্ড লাগবে।"]),
    "নেওয়া": (["নেয়া", "নিয়া", "নেওআ", "নেওয়া", "নেওয়াা", "নেঅয়া", "নেয়্যা"], ["একটু সময় {w} হয়েছে।", "নমুনা {w} হবে।"]),
    "হয়ে": (["হযে", "হইয়ে", "হয়্যে", "হোয়ে", "হই", "হএ", "হয়ে"], ["বুক {w} গেছে।", "দেরি {w} গেছে।"]),
    "যায়": (["যাই", "যায", "জায়", "যাঈ", "যায়্", "জাই"], ["টেস্ট করা {w}।", "নমুনা নেওয়া {w}।"]),
    "দেওয়া": (["দেয়া", "দিয়া", "দেওআ", "দেঅয়া", "দেয়্যা"], ["নমুনা {w} হয়েছে।", "সময় {w} হবে।"]),
}

asr = TurnASR(language_id="bn")
out = {}
for word, (cands, carriers) in WORDS.items():
    ranked = []
    for ci, cand in enumerate(dict.fromkeys([word] + cands)):
        hits = 0
        alone = nfc(asr.transcribe_utterance_sync(synth(cand, f"s3_{word}_{ci}_a")).text)
        a_ok = nfc(word) in alone
        heard = []
        for si, c in enumerate(carriers):
            t = nfc(asr.transcribe_utterance_sync(synth(c.format(w=cand), f"s3_{word}_{ci}_{si}")).text)
            hits += nfc(word) in t
            heard.append(t)
        score = hits + (1 if a_ok else 0)
        ranked.append((score, cand, a_ok, hits, alone, heard))
        print(f"{word:7} try {cand:9} alone={'OK' if a_ok else 'no'} sentences={hits}/{len(carriers)} (alone heard {alone!r})", flush=True)
    ranked.sort(key=lambda r: -r[0])
    out[word] = [{"score": r[0], "spelling": r[1], "alone_ok": r[2], "sentence_hits": r[3], "alone_heard": r[4], "heard": r[5]} for r in ranked]
    best = ranked[0]
    print(f"==> {word}: best spelling {best[1]!r} score {best[0]}/{len(carriers)+1}", flush=True)
json.dump(out, open("/tmp/respell/search_results.json", "w"), ensure_ascii=False, indent=1)
print("DONE", flush=True)
