"""Subprocess runner for tests/test_turn_routing_harness.py: drives the REAL `_dispatch_turn` of BOTH orchestrator entry points (main.py and main_pcm.py)
with typed sentences, and prints one JSON document of what the agent said / did.

Why a subprocess: the voice stack (NeMo, torch ...) is not installed off the pod, so its packages are replaced by stand-ins here -- and that must never leak into the
pytest process, where other tests probe for those packages. Everything after the recogniser is the REAL code (language gates, early routing, fast path, intent
repair, department listing, the reply templates, the speaking path). Replaced: the recogniser's text, the voice (silence), the websocket (a recorder), the intent
model (scripted, so a model mistake can be reproduced exactly) and the clinic API (the REAL clinic-api booking_service over a seeded throw-away database).

    python tests/_turn_harness_runner.py            # prints JSON
"""

from __future__ import annotations

import asyncio
import importlib.abc
import importlib.machinery
import importlib.util
import io
import json
import os
import sys
import tempfile
import wave
from unittest.mock import MagicMock

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)

_STUB = (
    "nemo",
    "torch",
    "torchaudio",
    "speechbrain",
    "librosa",
    "soundfile",
    "silero_vad",
    "omegaconf",
    "lightning",
    "pytorch_lightning",
    "sentencepiece",
    "transformers",
    "ctranslate2",
    "faster_whisper",
    "TTS",
    "onnxruntime",
)


class _Finder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    _busy = False
    _absent: dict[str, bool] = {}

    def find_spec(self, name, path, target=None):
        root = name.split(".")[0]
        if root not in _STUB or self._busy:
            return None
        if (
            root not in self._absent
        ):  # decided once, BEFORE any stand-in for this package exists (a stand-in would make it look installed)
            self._busy = True  # importlib.util.find_spec below walks sys.meta_path, which includes this finder
            try:
                self._absent[root] = importlib.util.find_spec(root) is None
            except (ImportError, ValueError):
                self._absent[root] = True
            finally:
                self._busy = False
        return importlib.machinery.ModuleSpec(name, self, is_package=True) if self._absent[root] else None

    def create_module(self, spec):
        m = MagicMock(name=spec.name)
        m.__path__ = []
        m.__spec__ = spec
        m.__name__ = spec.name
        return m

    def exec_module(self, module):
        return None


sys.meta_path.insert(0, _Finder())

# ---- the real clinic-api business logic over a seeded throw-away database
_fd, _DB = tempfile.mkstemp(suffix=".db")
os.close(_fd)
os.environ["CLINIC_DB_PATH"] = _DB
os.environ.pop("DATABASE_URL", None)
sys.path.insert(0, os.path.join(REPO, "clinic-api"))
import seed as seed_mod  # noqa: E402

seed_mod.seed()
import booking_migrate  # noqa: E402

booking_migrate.migrate_booking_schema()
booking_migrate.finish_booking_schema_setup()
import booking_service as bs  # noqa: E402
import db as db_mod  # noqa: E402

_DBSESSION = db_mod.SessionLocal()

INSURANCE_ANSWER = {
    "bn": "প্রধান বিমা সংস্থাগুলোর ক্যাশলেস সুবিধা আছে।",
    "hi": "प्रमुख बीमा कंपनियों की कैशलेस सुविधा उपलब्ध है।",
    "en": "Cashless facility is available with major insurance companies. Tell us your card's name and the counter will confirm it for you.",
}

FAQ_PAYLOAD = {
    "tests": [],
    "doctors": [],
    "faq_topics": [
        {
            "topic": "insurance",
            "keywords_bn": ["ইনসিওরেন্স", "ইন্সুরেন্স", "বিমা", "ক্যাশলেস", "মেডিক্লেম"],
            "keywords_hi": ["इंश्योरेंस", "बीमा", "कैशलेस", "मेडिक्लेम"],
            "keywords_en": ["insurance", "cashless", "mediclaim", "health insurance"],
        },
        {
            "topic": "hours",
            "keywords_bn": ["কখন খোলা"],
            "keywords_hi": ["कब खुलता"],
            "keywords_en": ["opening hours", "what time do you open"],
        },
        {"topic": "parking", "keywords_bn": ["পার্কিং"], "keywords_hi": ["पार्किंग"], "keywords_en": ["parking"]},
    ],
}


class FakeTools:
    """The clinic API as the agent sees it, answered by the REAL booking_service over the seeded database where a department is concerned."""

    def __init__(self):
        self.calls: list[tuple] = []

    async def get_doctor_availability(self, name, date=None):
        self.calls.append(("get_doctor_availability", name))
        return {"found": False, "query": name}

    async def department_doctors(self, name, lang="bn", sex=None, on_date=None, at_time=None):
        self.calls.append(("department_doctors", name))
        return bs.department_doctors(_DBSESSION, name, lang, sex=sex, on_date=on_date, at_time=at_time)

    async def route_department(self, text, lang="bn"):
        self.calls.append(("route_department", text))
        return bs.route_department(_DBSESSION, text, lang)

    async def get_faq(self, topic, lang="bn"):
        self.calls.append(("get_faq", topic))
        return {"found": True, "topic": topic, "answer": INSURANCE_ANSWER.get(lang, INSURANCE_ANSWER["en"])}

    async def request_callback(self, phone, call_id, window, reason, summary):
        self.calls.append(("request_callback", reason))
        return {"success": True}

    async def log_referral(self, **kw):
        self.calls.append(("log_referral", kw.get("patient_name"), kw.get("phone")))
        return {"success": True}

    def __getattr__(self, name):  # any other clinic call a scenario does not exercise
        async def _unexpected(*a, **k):
            self.calls.append((name,))
            raise RuntimeError(f"unexpected clinic call {name}")

        return _unexpected


def _silence_wav(seconds: float = 0.3) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 4800)
    return buf.getvalue()


class _Recorder:
    def __init__(self):
        self.said: list[str] = []
        self.frames: list[dict] = []

    async def send_text(self, data: str):
        try:
            d = json.loads(data)
        except ValueError:
            return
        if d.get("sender") == "AI":
            self.said.append(d["text"])
        elif d.get("type"):
            self.frames.append(d)

    async def send_json(self, *a, **k):
        return None

    async def send_bytes(self, data: bytes):
        return None

    async def close(self, *a, **k):
        return None


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO, filename))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# What the (scripted) intent model says for a sentence. Reproduces the model's real mistakes on purpose.
SCRIPTED = {
    "There is an orthopedic doctor.": {
        "intent": "doctor_availability",
        "slots": {"doctor_name": "Orthopaedics Doctor"},
    },
    "Orthopaedics Doctor": {"intent": "doctor_availability", "slots": {"doctor_name": "Orthopaedics Doctor"}},
    "My leg hurts.": {"intent": "unclear", "slots": {}},
    "আমার পায়ে ব্যথা।": {"intent": "smalltalk", "slots": {}},
    "My eyes hurt.": {"intent": "unclear", "slots": {}},
    "Do you have any health packages?": {"intent": "clinic_faq", "slots": {"faq_topic": "insurance"}},
    "Do I have any money left to pay?": {"intent": "unclear", "slots": {}},
    "Do you accept cashless insurance?": {"intent": "clinic_faq", "slots": {"faq_topic": "insurance"}},
    "Orthopaedics please.": {"intent": "doctor_availability", "slots": {"doctor_name": "Orthopaedics"}},
    "Dentistry please.": {"intent": "doctor_availability", "slots": {"doctor_name": "Dentistry"}},
    "Amar paye fracture hoyeche.": {"intent": "unclear", "slots": {}},
    "मेरे कान में दर्द है।": {"intent": "smalltalk", "slots": {}},
    "Ami jhapsa dekhchi.": {"intent": "unclear", "slots": {}},
}
_SLOT_KEYS = (
    "test_name",
    "test_names",
    "doctor_name",
    "date",
    "time_slot",
    "new_date",
    "new_time_slot",
    "confirmation_id",
    "patient_name",
    "patient_age",
    "patient_address",
    "phone",
    "contact_phone",
    "relationship",
    "spelled_letters",
    "symptom_description",
    "faq_topic",
)


async def run_entry(mod, utterances: list[tuple[str, str]], with_fast_path: bool) -> list[dict]:
    from agent.asr import ASRResult
    from agent.fast_path import Catalogue, FastPath

    tools = FakeTools()
    mod._tools = tools
    mod._languages_active = ("bn", "hi", "en")
    mod._fast_path = FastPath(Catalogue(FAQ_PAYLOAD)) if with_fast_path else None

    async def _no_reload():
        return None

    mod._maybe_reload_fast_path = _no_reload

    async def _no_spelling(
        text,
    ):  # the spelled-letters normaliser calls the live clinic API / model: not part of what is under test
        return text

    mod._normalise_spelled = _no_spelling
    model_calls: list[str] = []

    async def fake_model(session, text, lang, key):
        model_calls.append(text)
        spec = SCRIPTED.get(text, {"intent": "unclear", "slots": {}})
        return {
            "intent": spec["intent"],
            "slots": {k: spec["slots"].get(k) for k in _SLOT_KEYS},
            "direct_reply_bn": None,
            "secondary_intent": None,
            "secondary_slots": None,
        }

    mod._resolve_intent_uncached = fake_model

    clip = _silence_wav()

    async def fake_tts(session, clause, language, fallback_reason):
        return clip

    mod._synthesize_one_clause = fake_tts
    heard = {"text": "", "lang": "en"}

    async def fake_route(session, wav):
        return heard["lang"], ASRResult(text=heard["text"], decoder_used="rnnt", decoder_agreement=1.0), True

    mod._route_and_transcribe = fake_route
    handoffs: list[str] = []
    real_handoff = mod._handoff_to_human

    async def spy_handoff(session, reason, languages=None, announce=True):
        handoffs.append(reason)
        return await real_handoff(session, reason, languages, announce)

    mod._handoff_to_human = spy_handoff

    out = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, (text, lang) in enumerate(utterances):
            ws = _Recorder()
            session = mod.CallSession(ws)
            session.lang, session.lang_locked = lang, True
            session.phone_for_test = None
            script = (
                "bn"
                if any("ঀ" <= c <= "৿" for c in text)
                else "hi"
                if any("ऀ" <= c <= "ॿ" and c not in "।॥" for c in text)
                else "en"
                if text.isascii()
                else "bn"
            )  # the SCRIPT written; the danda U+0964 is shared, so it decides nothing
            heard["text"], heard["lang"] = text, script
            path = os.path.join(tmp, f"u{i}.wav")
            with open(path, "wb") as f:
                f.write(
                    _silence_wav(max(1.0, 0.7 * len(text.split())))
                )  # long enough for the words: a 0.3 s clip with 7 words is a "jumbled transcript"
            tools.calls.clear()
            model_calls.clear()
            handoffs.clear()
            err = None
            try:
                await mod._dispatch_turn(session, path)
            except Exception as e:  # noqa: BLE001 - reported to the test, never swallowed
                err = f"{type(e).__name__}: {e}"
            out.append(
                {
                    "text": text,
                    "lang": lang,
                    "said": list(ws.said),
                    "handoffs": list(handoffs),
                    "model_called": bool(model_calls),
                    "clinic_calls": [c[0] for c in tools.calls],
                    "error": err,
                }
            )
    return out


def main() -> None:
    cases = json.loads(sys.stdin.read())
    utter = [(c["text"], c["lang"]) for c in cases]
    result = {}
    for name, filename in (("main", "main.py"), ("main_pcm", "main_pcm.py")):
        mod = _load("voicebot_" + name, filename)
        result[name] = asyncio.run(run_entry(mod, utter, with_fast_path=True))
    print("@@RESULT@@" + json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    finally:
        _DBSESSION.close()
        try:
            os.remove(_DB)
        except OSError:
            pass
