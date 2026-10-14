# sourav: quick "is the NEW code running?" check. Run from the voicebot-main folder:  python check_new.py
import sys, datetime
sys.path[:0] = [".", "clinic-api", "tests"]
from agent.phrases import PHRASES, greeting_text
from agent.language_switch import detect_language_switch_request as sw
from agent import cart_tally
from agent.fast_path import Catalogue, FastPath
from _clinic_app import clinic_app

ok = lambda c: "PASS" if c else "FAIL  <-- old code still in use"
print("greeting en :", greeting_text("en"), ok("Sonoscan Vaani" in greeting_text("en") and "Hello" not in greeting_text("en")))
print("greeting bn :", greeting_text("bn"), ok("সোনোস্ক্যান বাণী" in greeting_text("bn")))
print("switch bn   :", sw("বেঙ্গলিতে কথা বলুন", "en"), ok(sw("বেঙ্গলিতে কথা বলুন", "en") == "bn"))
print("switch hi-en:", sw("Speak in Bangla", "en"), ok(sw("Speak in Bangla", "en") == "bn"))
print("cart plan   :", cart_tally.plan("remove CBC and tell me the total", ["cbc", "uric acid"]), ok(cart_tally.plan("remove CBC and tell me the total", ["cbc", "uric acid"])))
with clinic_app(sample_patients=False) as (_a, c):
    cat = c.get("/api/v1/catalogue").json()
print("doctors/tests in seed:", len(cat["doctors"]), "/", len(cat["tests"]), ok(len(cat["doctors"]) == 36))
fp = FastPath(Catalogue(cat), today=datetime.date.today())
for lang, q in [("en", "How to get there?"), ("bn", "okhane kivabe jabo"), ("hi", "waha kaise jayenge"),
                ("hi", "उखाने जाबो की भावे"), ("en", "ওখানে কিভাবে যাব"), ("en", "adress batao")]:
    r = fp.resolve(q, lang)
    got = r and (r.intent, r.slots.get("faq_topic"))
    print(f"location [{lang}] {q!r}:", got, ok(got == ("clinic_faq", "location")))
