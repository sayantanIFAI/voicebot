"""Kolkata Care Diagnostics -- Bengali voice agent, WebSocket orchestrator.

Turn loop, once a caller's utterance is judged complete (agent/vad_stream.py):

  utterance WAV -> ASR (agent/asr.py, IndicConformer)
                -> intent+slots (agent/llm.py, Ollama JSON-mode,
                   fronted by agent/semantic_cache.py)
                -> Spring Boot lookup (agent/tools_client.py) -- ALWAYS
                   live, never cached; see semantic_cache.py's docstring
                -> reply text, TEMPLATED from the API response, never
                   restated by the model (agent/reply_templates.py)
                -> TTS (agent/tts.py) -> WAV bytes back over the socket

Every stage has a named failure path (see _dispatch_turn) so a caller never
gets dead air: ASR-empty, LLM-failure, tool-failure and TTS-failure each
speak a distinct, pre-recorded apology rather than the process hanging or
the socket just going quiet. See README.md "Error handling" for the full
table and the reasoning behind each choice.

HALF-DUPLEX GATE
----------------
The mic is open for the entire call, and the agent's replies play out of
the caller's speaker. With no gate, the agent hears itself: its own
greeting lands in the same buffer the turn detector is watching, so VAD
fires a "the caller finished talking" on the agent's own voice, ASR
transcribes the agent, and processed_until_s advances past audio the
caller never produced. That is a self-sustaining loop, and it is what
made real calls cut the caller off in the first second and then run a
turn behind for the rest of the call.

Browser echoCancellation does not save this. It is built to cancel a
remote WebRTC peer's rendered stream; here the audio is synthesized
locally and played through Web Audio, which the canceller never sees as a
far-end reference.

So the pipeline is explicitly half-duplex, gated from BOTH ends:
  * client mutes the mic track while agent audio is playing (static/
    index.html) -- the track stays live and keeps emitting, so the WebM
    timeline never breaks, it just carries silence;
  * server refuses to run turn detection while `agent_speaking`, then
    resynchronizes processed_until_s past the muted region once playback
    is confirmed finished.

The cost is no barge-in: a caller cannot interrupt the agent mid-sentence.
That is a real limitation, chosen deliberately over the alternative, which
was a system that interrupted ITSELF. Supporting barge-in properly needs
an acoustic echo canceller with the played audio as a reference signal
(WebRTC APM or speex AEC), which is a much larger change.
"""
from __future__ import annotations

import asyncio
import contextlib
import io
import json
import logging
import os
import secrets
import tempfile
import time
import uuid
import wave

import numpy as np
import torchaudio
from fastapi import FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agent.acknowledgement import select_acknowledgement, slot_echo
from agent.admission import AdmissionController, HealthMonitor, http_probe
from agent.audio_quality import assess as assess_audio
from agent.audio_quality import enhance as enhance_audio
from agent.audio_quality import transcript_problem
from agent.asr import TurnASR
from agent.asr_router import ASRRouter, HTTPASREngine, UnroutableLanguageError
from agent.booking_flow import (
    BookingState,
    classify_yes_no,
    correction_acknowledgement,
    effective_phone,
    is_ready_to_confirm,
    mark_awaiting_charge_confirm,
    mark_confirming,
    merge_slots,
    merge_spelling,
    missing_required,
    new_state,
)
from agent.call_state import (
    CallState, apply_caller_state, apply_channel_quality, apply_confidence, apply_language, new_call_state,
)
from agent.channel_quality import CHANNEL_CLEAN_16K, classify_channel
from agent.clause_split import split_into_clauses
from agent.code_switch import mixture_bucket
from agent.confidence_gate import is_low_confidence, should_withhold_factual_answer
from agent.detector_budget import run_within_budget
from agent.detector_budget import snapshot as detector_budget_snapshot
from agent.enquiry_followup import entities_from_turn, resolve_followup_slot
from agent.fast_path import Catalogue, FastPath
from agent.filler import await_with_filler
from agent.latency_metrics import turn_latency_by_language
from agent.language_policy import language_mismatch
from agent.outcome_metrics import (
    audio_issue_buckets, barge_in_interrupts, channel_quality_buckets, code_switch_buckets,
    insufficient_information, language_mismatches, reask_outcomes, senior_detections,
)
from agent.reask_policy import ReaskTracker
from agent.senior_voice import SeniorEvidence, explicit_senior_cue, stated_age_is_senior
from agent.senior_voice import estimate as estimate_senior
from agent.speech_policy import derive_policy, effective_rate, limit_questions
from agent.lang_select import languages_to_verify, pick_candidate
from agent.lang_select import speakable as _speakable
from agent.language_switch import detect_language_switch_request
from agent.lid import (
    SUPPORTED_LANGUAGES,
    ASRLanguageRouter,
    LIDResult,
    SpeechBrainVoxLingua107LID,
)
from agent.llm import ExtractionError, extract_intent
from agent.phrases import HANDOFF_ALL_LANGUAGES, phrase, prewarm_lines
from agent.reply_templates import (
    add_test_reply,
    booking_confirmation_readback,
    booking_reply,
    cancel_reply,
    clinic_faq_reply,
    conflict_reply,
    department_route_reply,
    doctor_availability_reply,
    insufficient_information_reply,
    lookup_reply,
    missing_slot_prompt,
    multi_test_reply,
    multiple_bookings_reply,
    reschedule_reply,
    resend_reply,
    spelling_prompt,
    spelling_readback,
    test_prep_reply,
    test_rate_reply,
)
from agent.semantic_cache import SemanticCache
from agent.semantic_cache import embed as _embed_probe
from agent.speech_norm import contains_critical_figure
from agent.tools_client import ClinicToolsClient, ToolCallError
from agent.tts import FIGURE_SPEECH_SPEED, TTSClient, UnspeakableTextError
from agent.tts_router import TTSRouter
from agent.vad_stream import TurnDetector

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("main")

POLL_INTERVAL_S = 0.5
IDLE_TIMEOUT_S = 90.0
UTTERANCE_PAD_S = 0.15  # small trailing pad so ASR doesn't clip the last phoneme

# A live call was observed closing itself ~26s after the last exchange --
# far short of IDLE_TIMEOUT_S, which only fires at 90s. That gap points to
# an intermediate proxy (RunPod's or an nginx in front of it) closing
# WebSocket connections that go quiet for a while, independent of this
# app's own idle logic. A small periodic heartbeat keeps real traffic
# flowing on the socket so no proxy in between decides it's abandoned.
HEARTBEAT_INTERVAL_S = 15.0

# Backstop for the half-duplex gate. Normally the client reports playback
# finished and the gate lifts immediately; this only fires when that
# message never arrives (JS error, stale cached page, a client that
# predates the control channel). Generous on purpose -- lifting the gate
# early puts the agent back to hearing itself, which is the bug.
PLAYBACK_GUARD_S = 3.0

# On resync, rewind slightly before the buffer's decoded end. The region
# being skipped is muted silence, so rewinding into it costs nothing,
# while NOT rewinding risks clipping the caller's first syllable if the
# WebM decode is running a beat behind real time.
RESYNC_REWIND_S = 0.25

# KCD-461: "a natural filler is spoken when a stage exceeds its
# threshold... silence never exceeds a stated maximum." REASONED, not
# measured against real telephony (no pod to time it against) --
# comfortably above a fast_path/cache hit's near-zero latency and a
# warm LLM call's own "~9s across retries" evidence agent/llm.py's own
# KCD-465 fix cites, short enough that a genuinely slow turn does not
# leave the caller wondering whether the line is still open.
FILLER_THRESHOLD_S = 2.5

# KCD-076: agent/channel_quality.py's FFT-based classification is pure
# CPU arithmetic over one short clip -- REASONED, not measured against
# Appendix B's real per-call budget breakdown (no live pod to time it
# against right now); generous enough that it should never fire in
# practice on hardware this pipeline already assumes, tight enough that
# a genuinely pathological clip cannot silently eat into the turn.
CHANNEL_QUALITY_BUDGET_S = 0.08

# agent/audio_quality.assess + agent/senior_voice.estimate: FFT/autocorrelation
# over one utterance, run through detector_budget so a pathological clip
# cannot delay the turn. REASONED, not measured on the pod: generous for a
# 3-6 s clip on the hardware this pipeline already assumes.
AUDIO_ANALYSIS_BUDGET_S = 0.5

CLINIC_API_BASE = os.environ.get("CLINIC_API_BASE", "http://localhost:8080")

# Languages the live path routes between. "bn" alone reproduces the original
# Bengali-only behaviour (no LID, no Hindi ASR loaded) -- the rollback lever.
# Bengali is always active: it is the clinic's primary language and the
# greeting language.
ACTIVE_LANGUAGES = tuple(dict.fromkeys(
    ["bn"] + [x for x in os.environ.get("VOICE_AGENT_LANGUAGES", "bn,hi,en").replace(" ", "").split(",")
              if x in SUPPORTED_LANGUAGES]))
ENGLISH_ASR_URL = os.environ.get("ENGLISH_ASR_URL", "http://localhost:8003")
TTS_HEALTH_URL = os.environ.get("TTS_HEALTH_URL", "http://localhost:8002/health")
OLLAMA_HEALTH_URL = os.environ.get("OLLAMA_HEALTH_URL", "http://localhost:11434/api/tags")

# Admission control (agent/admission.py). Cap is per PROCESS: with the WebM
# and PCM services both running, each has its own counter.
ADMISSION_MAX_CALLS = int(os.environ.get("ADMISSION_MAX_CALLS", "28"))
ADMISSION_SHED_P95_S = (float(os.environ["ADMISSION_SHED_P95_S"])
                        if os.environ.get("ADMISSION_SHED_P95_S") else None)
ADMISSION_BYPASS_FILE = os.environ.get("ADMISSION_BYPASS_FILE", "/workspace/.ai_bypass")
ADMISSION_ADMIN_TOKEN = os.environ.get("ADMISSION_ADMIN_TOKEN", "")

app = FastAPI()

# KCD-467: "time from process start to ready is measured and bounded."
# Captured at import time -- as close to "process start" as this module
# can observe -- so _startup()'s own duration can be reported once every
# model has answered ready, rather than assumed. See /api/health.
_PROCESS_STARTED_AT = time.monotonic()
_READY_AT: float | None = None   # set once, at the end of _startup()

# ---- process-wide singletons: loaded once, shared by every call ----
_asr: TurnASR | None = None
_turn_detector: TurnDetector | None = None
_tools: ClinicToolsClient | None = None
_tts: TTSClient | None = None
_intent_cache: SemanticCache | None = None
_fast_path: FastPath | None = None
_asr_router: ASRRouter | None = None
_lid: SpeechBrainVoxLingua107LID | None = None
_tts_router: TTSRouter | None = None
_admission: AdmissionController | None = None
_health: HealthMonitor | None = None
_health_task: asyncio.Task | None = None
_languages_active: tuple[str, ...] = ("bn",)


class _UnavailableASR:
    """Registered for a language this build does not serve. Fails loudly on
    use (ASRRouter's contract) instead of silently sending Hindi audio to
    the Bengali model."""

    def __init__(self, language: str):
        self.language = language

    async def transcribe_utterance(self, wav_path: str):
        raise UnroutableLanguageError(f"language {self.language!r} is not active in this build")


_fast_path_retry_at = 0.0


async def _warm_speech_models() -> None:
    """Push one real utterance per language through its ASR, and through LID.

    The first real inference on a CUDA model compiles kernels and allocates
    workspaces; measured on the pod, the first turn of the first call took
    ~23 s (three ASRs plus LID, all cold) against ~2 s once warm. A tone or
    silence does NOT do it -- the decode path only runs on speech -- so the
    warm-up speech is the agent's own greeting, synthesized by the TTS."""
    t0 = time.monotonic()
    for lang in _languages_active:
        path = os.path.join(tempfile.gettempdir(), f"kcd_warm_{lang}_{uuid.uuid4().hex[:6]}.wav")
        try:
            wav = await _tts_router.synthesize(lang, phrase("greeting", lang))
            with open(path, "wb") as f:
                f.write(wav)
            await _asr_router.transcribe(lang, path)
            if _lid is not None:
                await asyncio.to_thread(_lid.identify_path, path)
            if lang == "bn":
                # The turn detector and the audio writer are on every call's
                # first-turn path too, and both are lazy: Silero's first
                # inference and torchaudio's first save (torchcodec, loaded
                # from the network volume) are where the first caller's wait
                # went if they are not exercised here.
                import soundfile as sf
                data, sr = await asyncio.to_thread(sf.read, path, dtype="float32", always_2d=True)
                mono = torchaudio.functional.resample(
                    __import__("torch").from_numpy(data.mean(axis=1)), sr, 16000)
                await asyncio.to_thread(_turn_detector.poll, mono, 16000)
                warm_out = path + ".save.wav"
                await asyncio.to_thread(torchaudio.save, warm_out, mono.unsqueeze(0), 16000)
                with contextlib.suppress(OSError):
                    os.remove(warm_out)
        except Exception as e:  # noqa: BLE001 - warmup is advisory
            logger.warning("speech warmup failed for %s: %s", lang, e)
        finally:
            with contextlib.suppress(OSError):
                os.remove(path)
    logger.info("speech models warm (%.1fs)", time.monotonic() - t0)


async def _load_fast_path() -> FastPath | None:
    """Fetch the catalogue and build the fast path, or None if clinic-api is
    not reachable. Optional by design: without it every turn goes to the LLM,
    which is the behaviour that existed before this path did."""
    import httpx as _hx
    try:
        async with _hx.AsyncClient(timeout=10) as c:
            payload = (await c.get(f"{CLINIC_API_BASE}/api/v1/catalogue")).json()
        fp = FastPath(Catalogue(payload))
        logger.info("fast path ready over %d catalogue rows", len(fp.catalogue))
        return fp
    except Exception as e:  # noqa: BLE001 - degrade to LLM-only, never fail startup
        logger.warning("catalogue unavailable, fast path disabled: %s", e)
        return None


async def _maybe_reload_fast_path() -> None:
    """If the fast path failed to load at boot (clinic-api was not up yet --
    it starts AFTER this service in start_all.sh), retry at most every 30 s.
    Without this a boot-order race silently costs the fast path until the
    next restart: every turn pays an LLM call and nothing says why."""
    global _fast_path, _fast_path_retry_at
    if _fast_path is not None or time.monotonic() < _fast_path_retry_at:
        return
    _fast_path_retry_at = time.monotonic() + 30.0
    _fast_path = await _load_fast_path()


@app.on_event("startup")
async def _startup():
    global _asr, _turn_detector, _tools, _tts, _intent_cache, _fast_path
    global _asr_router, _lid, _tts_router, _admission, _health, _health_task, _languages_active
    import httpx as _httpx

    active = list(ACTIVE_LANGUAGES)
    engines: dict = {}
    logger.info("loading IndicConformer bn...")
    _asr = await asyncio.to_thread(TurnASR)
    engines["bn"] = _asr
    if "hi" in active:
        logger.info("loading IndicConformer hi...")
        engines["hi"] = await asyncio.to_thread(TurnASR, None, "hi")
    if "en" in active:
        engines["en"] = HTTPASREngine(ENGLISH_ASR_URL)
    for lang in SUPPORTED_LANGUAGES:
        engines.setdefault(lang, _UnavailableASR(lang))
    _asr_router = ASRRouter(engines)

    _lid = None
    if len(active) > 1:
        logger.info("loading language ID (SpeechBrain VoxLingua107, CPU)...")
        try:
            lid = SpeechBrainVoxLingua107LID(device="cpu")
            await asyncio.to_thread(lid.load)
            _lid = lid
        except Exception as e:  # noqa: BLE001 - degrade LOUDLY to Bengali-only, never guess a language
            logger.error("language ID unavailable (%s) -- serving Bengali ONLY until fixed", e)
            active = ["bn"]
    _languages_active = tuple(active)
    logger.info("active languages: %s", ",".join(_languages_active))

    logger.info("loading Silero VAD...")
    _turn_detector = await asyncio.to_thread(TurnDetector)
    _tools = ClinicToolsClient(CLINIC_API_BASE)
    _tts = TTSClient()
    _tts_router = TTSRouter({lang: _tts.for_language(lang) for lang in SUPPORTED_LANGUAGES})
    _intent_cache = SemanticCache()

    _admission = AdmissionController(
        max_calls=ADMISSION_MAX_CALLS,
        latency_shed_p95_s=ADMISSION_SHED_P95_S,
        bypass_file=ADMISSION_BYPASS_FILE or None,
    )
    probe_client = _httpx.AsyncClient()
    probes = {
        "tts": http_probe(probe_client, TTS_HEALTH_URL),
        "clinic-api": http_probe(probe_client, f"{CLINIC_API_BASE}/api/health"),
        "ollama": http_probe(probe_client, OLLAMA_HEALTH_URL),
    }
    if "en" in _languages_active:
        probes["english-asr"] = http_probe(probe_client, f"{ENGLISH_ASR_URL}/health")
    _health = HealthMonitor(_admission, probes)
    _health.probe_client = probe_client
    _health_task = asyncio.create_task(_health.run_forever())

    # Pull bge-m3 into VRAM before the first caller needs it. Cold-loading
    # it inside a live turn measured past the client's patience AND past
    # the embed timeout, which silently degraded the cache to exact-match
    # only for the opening minutes of the process -- healthy-looking logs,
    # zero semantic hits. OLLAMA_KEEP_ALIVE=-1 keeps it resident after.
    try:
        await asyncio.to_thread(_embed_probe, "warmup")
        logger.info("embedding model warm")
    except Exception as e:  # noqa: BLE001 - cache is optional, the call is not
        logger.warning("embedding warmup failed, cache starts L1-only: %s", e)

    # Load Qwen into VRAM now. Cold, it measured 74 s for its first intent --
    # that is the first caller's wait, and it would also poison the latency
    # window admission control sheds load on.
    try:
        _, diag = await asyncio.to_thread(extract_intent, "নমস্কার")
        logger.info("intent model warm (%.1fs)", diag["total_time_s"])
    except Exception as e:  # noqa: BLE001 - warmup is advisory
        logger.warning("intent model warmup failed: %s", e)

    # Load the 74-row catalogue once so the fast path can identify a test
    # or doctor locally. Optional: if the clinic API is not up yet, every
    # turn simply goes to the LLM, which is the behaviour that existed
    # before this path did.
    _fast_path = await _load_fast_path()

    logger.info("prewarming TTS...")
    lines = prewarm_lines()
    await _tts.prewarm({lang: lines[lang] for lang in _languages_active})
    await _warm_speech_models()

    global _READY_AT
    _READY_AT = time.monotonic()
    logger.info("startup complete -- ready for calls (%.1fs from process start)",
                _READY_AT - _PROCESS_STARTED_AT)


@app.on_event("shutdown")
async def _shutdown():
    if _health_task:
        _health_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await _health_task
    if _health is not None:
        await _health.probe_client.aclose()
    if _asr_router is not None:
        with contextlib.suppress(Exception):
            await _asr_router.engine_for("en").aclose()
    if _tools:
        await _tools.aclose()
    if _tts:
        await _tts.aclose()


@app.get("/api/health")
async def health():
    # KCD-467: "instances receive traffic only once every model answers
    # a probe" -- ASGI/uvicorn already refuses connections until
    # _startup() (which awaits every model load) returns, so `ready`
    # here is never False for a request that could reach this handler
    # at all; it exists so an orchestrator polling this URL through a
    # proxy that itself started early sees an honest, explicit signal
    # rather than inferring readiness from a 200 alone. `startup_duration_s`
    # is the "measured and bounded" half of the acceptance criterion --
    # the actual bound (a deploy pipeline gating traffic on this field
    # rather than on the process merely existing) is an infra concern
    # outside this repository.
    return {
        "status": "ok",
        "ready": _READY_AT is not None,
        "startup_duration_s": round(_READY_AT - _PROCESS_STARTED_AT, 1) if _READY_AT else None,
        "asr_loaded": _asr is not None,
        "clinic_api_base": CLINIC_API_BASE,
        "languages_active": list(_languages_active),
        "lid_loaded": _lid is not None,
        "admission_open": bool(_admission and _admission.snapshot()["open"]),
    }


@app.get("/api/admission")
async def admission_status():
    """Read-only: safe to leave unauthenticated behind the private network."""
    return _admission.snapshot() if _admission else {"open": False, "closed_reason": "starting"}


class _BypassRequest(BaseModel):
    enabled: bool


@app.post("/api/admission/bypass")
async def admission_bypass(req: _BypassRequest, x_admin_token: str = Header(default="")):
    """The audited AI kill-switch: enabled=true sends every NEW call to the
    human contact centre without a deploy. Disabled entirely (503) unless
    ADMISSION_ADMIN_TOKEN is set -- an unauthenticated switch that can take
    the whole service offline is worse than none."""
    if not ADMISSION_ADMIN_TOKEN:
        raise HTTPException(status_code=503, detail="bypass endpoint disabled: no admin token configured")
    if not secrets.compare_digest(x_admin_token, ADMISSION_ADMIN_TOKEN):
        raise HTTPException(status_code=403, detail="bad admin token")
    _admission.set_bypass(req.enabled, actor="http")
    return _admission.snapshot()


@app.get("/api/stats")
async def stats():
    """Cache effectiveness, for tuning the similarity threshold against
    real traffic rather than against my assumptions about it."""
    return {
        "fast_path": _fast_path.snapshot() if _fast_path else None,
        "intent_cache": _intent_cache.snapshot() if _intent_cache else None,
        "tts_cache": _tts.snapshot() if _tts else None,
        "admission": _admission.snapshot() if _admission else None,
        "insufficient_information": insufficient_information.snapshot(),
        "code_switch_buckets": code_switch_buckets.snapshot(),
        "channel_quality_buckets": channel_quality_buckets.snapshot(),
        "detector_budget": detector_budget_snapshot(),
        "turn_latency_by_language": turn_latency_by_language.snapshot(),
        "barge_in_interrupts": barge_in_interrupts.snapshot(),
        "reask_outcomes": reask_outcomes.snapshot(),
        "audio_issue_buckets": audio_issue_buckets.snapshot(),
        "language_mismatches": language_mismatches.snapshot(),
        "senior_detections": senior_detections.snapshot(),
    }


def _wav_duration_s(wav_bytes: bytes) -> float:
    try:
        with contextlib.closing(wave.open(io.BytesIO(wav_bytes), "rb")) as w:
            return w.getnframes() / float(w.getframerate())
    except Exception:  # noqa: BLE001 - a fallback clip may not be canonical WAV
        return 5.0


def _classify_channel_from_wav_path(path: str) -> str:
    """KCD-075's sync (CPU-bound) half -- run through
    agent.detector_budget.run_within_budget, never called directly, so a
    pathological clip cannot delay the turn past its budget slice."""
    with contextlib.closing(wave.open(path, "rb")) as w:
        raw = w.readframes(w.getnframes())
        sample_rate = w.getframerate()
        sampwidth = w.getsampwidth()
    if sampwidth != 2:
        # This pipeline only ever produces 16-bit PCM clips (torchaudio.save's
        # default); anything else is unexpected input this classifier was not
        # built to read faithfully, so default to the least alarming label
        # rather than mis-parse it into a bogus one.
        return CHANNEL_CLEAN_16K
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0
    return classify_channel(samples, sample_rate)


def _read_wav_mono(path: str) -> tuple[np.ndarray, int]:
    with contextlib.closing(wave.open(path, "rb")) as w:
        raw, sr, width, channels = w.readframes(w.getnframes()), w.getframerate(), w.getsampwidth(), w.getnchannels()
    if width != 2:
        raise ValueError(f"unsupported sample width {width}")
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return (x.reshape(-1, channels).mean(axis=1) if channels > 1 else x), sr


def _analyze_utterance_from_wav_path(path: str) -> dict:
    """agent/audio_quality.assess (why this clip is hard to hear) plus
    agent/senior_voice.estimate (does the speaker sound older), on one
    read of the clip. Sync and CPU-bound: only ever called through
    detector_budget.run_within_budget."""
    samples, sr = _read_wav_mono(path)
    return {"audio": assess_audio(samples, sr), "senior": estimate_senior(samples, sr)}


def _enhance_wav_to_path(path: str) -> str | None:
    """A noise-filtered, level-corrected copy of the clip for a retry, or
    None if it cannot be produced."""
    try:
        samples, sr = _read_wav_mono(path)
        out = enhance_audio(samples, sr)
        enhanced = path + ".enh.wav"
        with contextlib.closing(wave.open(enhanced, "wb")) as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes((np.clip(out, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes())
        return enhanced
    except Exception as e:  # noqa: BLE001 - enhancement is an optional retry, never a failure
        logger.warning("enhance failed: %s", e)
        return None


async def _decode_to_wav(raw_path: str, wav_path: str) -> bool:
    """Re-decode the WHOLE growing webm buffer every poll -- same choice
    server.py makes for the consultation recorder, and for the same reason:
    MediaRecorder only puts the container header on the FIRST chunk, so
    later chunks are not independently decodable, and re-decoding a few
    seconds of audio is cheap next to ASR+LLM+TTS.

    NOTE the cost profile: this is O(call length) on every poll, so the
    work per poll grows for the whole duration of a call. Fine for a
    handful of concurrent bench calls; it is the first thing that has to
    change for real concurrency (see README.md "Scaling")."""
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg", "-y", "-loglevel", "error", "-i", raw_path,
        "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", wav_path,
        stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
    )
    await proc.wait()
    return proc.returncode == 0 and os.path.exists(wav_path) and os.path.getsize(wav_path) > 44


class CallSession:
    """One raw/decoded buffer for the ENTIRE call, not one per turn.

    An earlier version deleted the buffer and started a fresh file after
    every completed turn. That broke on real testing: MediaRecorder only
    puts the WebM container's header in the very first chunk of the whole
    recording session -- every chunk after a mid-call "reset" was being
    appended to a file that could never be decoded, because its one valid
    header lived in the turn-1 file that had already been deleted. Every
    turn after the first silently failed to decode, forever, for the rest
    of the call.

    Fixed the same way voice-to-rx-repo/server.py already had to: keep
    ONE continuous file for the whole call and track `processed_until_s`
    -- a marker for how much of it a prior turn has already consumed.
    Each poll only ever looks at the UNPROCESSED TAIL past that marker.
    """

    def __init__(self, ws: WebSocket):
        self.ws = ws
        self.call_id = uuid.uuid4().hex[:8]
        self.tmpdir = tempfile.mkdtemp(prefix=f"kcd_call_{self.call_id}_")
        self.last_activity = time.time()
        self.dispatch_lock = asyncio.Lock()
        self.processed_until_s = 0.0
        self.utt_seq = 0
        self.last_heartbeat = time.time()
        self.raw_path = os.path.join(self.tmpdir, "call.webm")
        self.wav_path = self.raw_path + ".wav"
        open(self.raw_path, "wb").close()

        # Starts True: the greeting goes out before the caller has said
        # anything, so the gate must already be closed when the first poll
        # tick runs, not opened a moment later by _speak().
        self.agent_speaking = True
        self.speak_deadline = time.time() + PLAYBACK_GUARD_S
        self.resync_pending = False

        # Language is per UTTERANCE (ASRLanguageRouter decides each turn);
        # `lang` is only the last committed one -- the reply language, and
        # the prior the router falls back on when LID is unsure.
        self.lang = "bn"
        self.lang_router = ASRLanguageRouter()
        self.turn_started_at: float | None = None
        self.admission = None

        # Epic E26: the in-progress booking/reschedule/cancel/add-test flow
        # for this call, if any -- see agent/booking_flow.py. One at a time;
        # a caller starting a second booking mid-flow replaces it (their
        # most recent words win, same principle as merge_slots' overwrite
        # rule). No real telephony CallerID exists yet (see HANDOVER.md /
        # CLAUDE.md's "STILL NOT BUILT" notes), so there is no independent
        # "number this call came from" distinct from whatever contact phone
        # the caller states -- booking_service's proxy-authorisation model
        # is real, but is only as strong as a phone-based system without
        # SIP CallerID can be until that lands.
        self.booking: BookingState | None = None

        # KCD-395/KCD-396: what the last answered enquiry turn(s) were
        # about (agent/enquiry_followup.EnquiryEntity), so a bare
        # elliptical follow-up ("and the sample for that?") can resolve
        # without asking the caller to repeat the test/doctor name.
        # Booking turns neither read nor write this -- it is scoped to
        # the stateless enquiry intents only.
        self.last_enquiry_entities: list = []

        # KCD-061: the single, versioned carrier of caller signals
        # (Blueprint 4.5) -- language and confirmation_required are kept
        # current every turn (see _dispatch_turn); every other field
        # stays at its neutral default until a real caller-state
        # detector exists to drive it (see agent/call_state.py's own
        # docstring).
        self.call_state: CallState = new_call_state()

        # Call intelligence (KCD-084/149/155): the empathetic re-ask tracker,
        # the older-caller evidence accumulator, and bookkeeping so an
        # acknowledgement is spoken once per state entry, not every turn.
        self.reask = ReaskTracker()
        self.senior_evidence = SeniorEvidence()
        self.acknowledged_state: str | None = None
        self.senior_lookup_done = False

    @property
    def policy(self):
        """Appendix C's delivery parameters for the caller's CURRENT state
        (agent/speech_policy.py) -- derived, never stored, so it can never
        disagree with call_state."""
        return derive_policy(self.call_state.caller_state, self.call_state.senior)

    def hold_gate_for(self, audio_duration_s: float):
        """Called before each reply goes out. Extends rather than replaces
        the deadline: replies queue on the client, so a second clip starts
        playing only after the first finishes."""
        base = max(self.speak_deadline, time.time()) if self.agent_speaking else time.time()
        self.agent_speaking = True
        self.speak_deadline = base + audio_duration_s + PLAYBACK_GUARD_S

    def release_gate(self):
        """Playback is over. Don't touch processed_until_s here -- the poll
        loop owns the decoded buffer and does the resync on its next tick."""
        self.agent_speaking = False
        self.resync_pending = True

    async def append(self, chunk: bytes):
        self.last_activity = time.time()
        with open(self.raw_path, "ab") as f:
            f.write(chunk)

    async def send_json(self, sender: str, text: str):
        await self.ws.send_text(json.dumps({"sender": sender, "text": text}, ensure_ascii=False))

    async def send_audio(self, wav_bytes: bytes):
        if wav_bytes:
            await self.ws.send_bytes(wav_bytes)

    def cleanup(self):
        import shutil
        with contextlib.suppress(OSError):
            shutil.rmtree(self.tmpdir, ignore_errors=True)


async def _synthesize_one_clause(session: CallSession, clause: str, lang: str,
                                 fallback_reason: str | None) -> bytes:
    # KCD-149/157: the caller's policy rate, and slower still for a price,
    # phone number or reference (the two multiply).
    speed = effective_rate(session.policy,
                           FIGURE_SPEECH_SPEED if contains_critical_figure(clause) else None)
    try:
        return await _tts_router.synthesize(lang, clause, speed=speed)
    except UnspeakableTextError as e:
        # KCD-455: a production occurrence is an alert, not routine
        # noise -- distinct log level and reason from an ordinary TTS
        # infra failure below, even though both fall back to the same
        # pre-recorded apology.
        logger.error("[%s] reply blocked, unspeakable spans %s in %r",
                     session.call_id, e.spans, clause)
        return _tts.fallback_audio(fallback_reason or "tts_failure", lang)
    except Exception as e:  # noqa: BLE001 - TTS is the last mile, must not raise past here
        logger.warning("[%s] TTS failed (%s) -- using fallback audio", session.call_id, e)
        return _tts.fallback_audio(fallback_reason or "tts_failure", lang)


async def _speak(session: CallSession, text: str, lang: str | None = None,
                 fallback_reason: str | None = None) -> float:
    """Speak `text` in `lang` (default: the call's current language).
    Returns the total audio duration in seconds.

    KCD-456: automatically slower for a reply carrying a price, phone
    number or reference ID (agent.speech_norm.contains_critical_figure)
    -- decided here, once, rather than at every call site that happens to
    produce such a reply, so no future reply_templates.py addition can
    forget to ask for it.

    KCD-462: a long reply (agent.clause_split.split_into_clauses) is
    synthesized and sent ONE CLAUSE AT A TIME instead of as one TTS call
    for the whole text -- the first clause reaches the caller as soon as
    its own (much shorter) synthesis finishes, not after the entire
    reply renders. A short reply (the common case) comes back as a
    single-item list and behaves exactly as before -- one TTS call, one
    clip. session.hold_gate_for already extends cumulatively for
    multiple clips in a row (see its own docstring), which is what a
    correction-acknowledgement-then-readback turn already relies on
    elsewhere in this file, so streaming several clips here is the same,
    already-proven pattern, not a new one."""
    lang = lang or session.lang
    policy = session.policy
    if not policy.emergency and policy.questions_per_turn >= 1:
        # KCD-084/149: one question at a time. Only a SURPLUS question is
        # ever dropped, never a statement, so no fact is lost.
        text = limit_questions(text, policy.questions_per_turn)
    if language_mismatch(lang, text):
        # KCD-087: a reply in a different language from the caller's is a
        # defect, made visible rather than discovered on a live call.
        language_mismatches.record("reply", "mismatch", lang)
        logger.error("[%s] reply language mismatch (expected %s): %r", session.call_id, lang, text[:60])
    await session.send_json("AI", text)
    clauses = split_into_clauses(text) or [text]

    total_duration = 0.0
    for i, clause in enumerate(clauses):
        wav = await _synthesize_one_clause(session, clause, lang, fallback_reason)

        # Close the gate BEFORE the bytes leave, never after: the client
        # can start playing the moment they land, and a poll tick that
        # slips in between send and gate is exactly the echo this
        # prevents.
        duration = _wav_duration_s(wav)
        session.hold_gate_for(duration)
        await session.send_audio(wav)
        total_duration += duration

        # "Time to first audio" (admission.py's own docstring) means the
        # FIRST clause, not the last -- recorded once, right after it is
        # sent, so streaming a long reply is measured (and shows up as
        # faster) rather than averaged against the clauses still to come.
        if i == 0 and session.turn_started_at is not None and _admission is not None:
            turn_latency_s = time.monotonic() - session.turn_started_at
            _admission.record_turn_latency(turn_latency_s)
            # KCD-469: the SAME measurement, also broken down per
            # language -- admission's own p50/p95 stay a single
            # aggregate signal by design (see its module docstring), so
            # a language-specific regression could hide inside a
            # healthy-looking aggregate without this.
            turn_latency_by_language.record(lang, turn_latency_s)
            session.turn_started_at = None

    return total_duration


async def _handoff_to_human(session: CallSession, reason: str, languages: tuple[str, ...] | None = None):
    """Tell the caller and the telephony bridge that a person is taking over.

    The `handoff_human` frame is the machine-readable half: the SBC/bridge
    that owns the SIP leg acts on it (this repo has no SIP transfer). The
    spoken notice is the human-readable half, said in every language when
    the caller's is not yet known.
    """
    logger.warning("[%s] handoff to human: %s", session.call_id, reason)
    with contextlib.suppress(Exception):
        await session.ws.send_text(json.dumps({"type": "handoff_human", "reason": reason}))
    total = 0.0
    for lang in (languages or (session.lang,)):
        with contextlib.suppress(Exception):
            total += await _speak(session, phrase("handoff", lang), lang)
    # Let the notice finish playing before the socket closes under it.
    await asyncio.sleep(min(total + 0.5, 20.0))
    with contextlib.suppress(Exception):
        await session.ws.close()


async def _reask_or_handoff(session: CallSession, decision, lang: str,
                            fallback_reason: str | None = None,
                            languages: tuple[str, ...] | None = None) -> None:
    """Ask again, kindly, before routing to a person (agent/reask_policy.py).
    A second consecutive failure also marks the caller "confused", which
    slows the agent and lowers its escalation threshold for the rest of the
    call (Appendix C)."""
    reask_outcomes.record(decision.reason or "unclear", decision.action, lang)
    if decision.mark_confused:
        apply_caller_state(session.call_state, caller_state="confused")
    # `languages` is passed when the caller's language could not be identified:
    # the re-ask and the hand-off notice are then said in EVERY active language,
    # not in whichever one the call happened to default to.
    langs = languages or (lang,)
    if decision.action == "handoff":
        for lg in langs:
            await _speak(session, phrase("reask_final", lg), lg)
        await _handoff_to_human(session, f"unintelligible:{decision.reason}", langs)
        return
    for lg in langs:
        await _speak(session, phrase(decision.phrase_key, lg), lg, fallback_reason=fallback_reason)


async def _retry_on_enhanced_audio(session: CallSession, wav_path: str, lang: str | None, result,
                                   audio_issues: list[str], duration_s: float | None):
    """A failed or doubtful turn on a noisy or faint line gets one more try
    on a noise-filtered, level-corrected copy of the clip. Only when the
    first attempt was actually poor -- filtering can hurt a recogniser that
    coped fine -- and never for cross-talk, which filtering cannot fix."""
    if lang is None or not ({"noisy", "too_quiet"} & set(audio_issues)):
        return result
    poor = (not result.text.strip()
            or transcript_problem(result.text, lang, duration_s, result.decoder_agreement)
            or is_low_confidence(result.decoder_agreement))
    if not poor:
        return result
    enhanced = await asyncio.to_thread(_enhance_wav_to_path, wav_path)
    if not enhanced:
        return result
    try:
        retry = await _asr_router.transcribe(lang, enhanced)
    except Exception as e:  # noqa: BLE001 - the original result stands
        logger.warning("[%s] enhanced-audio retry failed: %s", session.call_id, e)
        return result
    finally:
        with contextlib.suppress(OSError):
            os.remove(enhanced)
    if retry.text.strip() and (not result.text.strip() or retry.decoder_agreement > result.decoder_agreement):
        logger.info("[%s] enhanced-audio retry improved the transcript", session.call_id)
        return retry
    return result


def _update_caller_state(session: CallSession, text: str, lang: str, analysis: dict | None) -> None:
    """KCD-084: the older-caller decision. An explicit request or self-
    description decides at once; the acoustic estimate needs evidence over
    several clips (agent/senior_voice.SeniorEvidence). Once senior, sticky
    for the call."""
    ev = session.senior_evidence
    cue = explicit_senior_cue(text, lang)
    flipped = ev.note_explicit(cue) if cue else (ev.add(analysis["senior"]) if analysis else False)
    if flipped and not session.call_state.senior:
        apply_caller_state(session.call_state, senior=True)
        senior_detections.record(ev.reason or "unknown", "call", lang)
        logger.info("[%s] senior mode on (%s)", session.call_id, ev.reason)


def _note_stated_age(session: CallSession, slots: dict, lang: str) -> None:
    if stated_age_is_senior(slots.get("patient_age"), slots.get("relationship")):
        if session.senior_evidence.note_explicit("stated_age") and not session.call_state.senior:
            apply_caller_state(session.call_state, senior=True)
            senior_detections.record("stated_age", "call", lang)


_ECHOED_FIELDS = ("doctor_name", "date", "time_slot", "patient_name", "phone", "test_name",
                  "new_date", "new_time_slot")


async def _echo_new_slots(session: CallSession, changed: list[str], prior_slots: dict,
                          slots: dict, lang: str) -> None:
    """Senior mode's "one question, listen, confirm, next question": say
    the value just collected back before the next question. Also the moment
    a phone number first arrives, so a returning older caller is recognised
    from the record without having to ask for slower speech again."""
    newly = [f for f in changed if f not in prior_slots]
    if "phone" in newly:
        await _remember_senior(session, slots.get("phone"), lang)
    if not session.policy.confirm_each_slot:
        return
    for field in newly:
        value = slots.get(field)
        if value and field in _ECHOED_FIELDS:
            await _speak(session, slot_echo(str(value), lang), lang)
            return


async def _remember_senior(session: CallSession, phone: str | None, lang: str) -> None:
    if session.senior_lookup_done or not phone:
        return
    session.senior_lookup_done = True
    try:
        if await _tools.get_patient_senior(phone) and session.senior_evidence.note_explicit("remembered"):
            apply_caller_state(session.call_state, senior=True)
            senior_detections.record("remembered", "call", lang)
    except ToolCallError as e:
        logger.warning("[%s] senior lookup failed: %s", session.call_id, e)


async def _persist_senior(session: CallSession, phone: str | None) -> None:
    """KCD-084: once a booking is made, remember the delivery mode against
    the patient -- a boolean only, never a score, an age or any audio."""
    if not (session.call_state.senior and phone):
        return
    try:
        await _tools.set_patient_senior(phone, True)
    except ToolCallError as e:
        logger.warning("[%s] senior persist failed: %s", session.call_id, e)


async def _handle_unverified_write(session: CallSession, lang: str) -> None:
    """KCD-486: a booking/reschedule write whose post-commit verification
    (clinic-api's _verify_appointment_persisted) did not confirm the
    expected values. Speaks a hold notice -- distinct from both a
    confirmation number (which would be an unverified fact) and the
    generic tool_failure apology (which invites a retry, and the write
    likely already happened, so a retry risks a second, duplicate one) --
    then escalates to a human rather than guessing."""
    await _speak(session, phrase("booking_hold_for_verification", lang), lang)
    await _handoff_to_human(session, "unverified_booking_write")


async def _await_with_filler(session: CallSession, awaitable, lang: str,
                             threshold_s: float = FILLER_THRESHOLD_S):
    """KCD-461: thin wrapper over agent.filler.await_with_filler (the
    tested race logic) supplying "speak the pre-warmed please_wait
    phrase" as the timeout callback -- never synthesized fresh
    (TTSClient's cache already holds it from startup prewarm), so it
    costs no synthesis time at exactly the moment the system is already
    running slow."""
    async def _speak_filler():
        await _speak(session, phrase("please_wait", lang), lang)
    return await await_with_filler(awaitable, threshold_s, _speak_filler)


async def _slice_utterance(session: CallSession, start_s: float, end_s: float, seq: int) -> str:
    """Cuts [start_s, end_s+pad] -- both ABSOLUTE call-time offsets -- out
    of the call's decoded WAV into its own small file for ASR."""
    wav, sr = await asyncio.to_thread(torchaudio.load, session.wav_path)
    a = max(0, int(start_s * sr))
    b = min(int((end_s + UTTERANCE_PAD_S) * sr), wav.shape[-1])
    clip_path = f"{session.wav_path}.utt{seq}.wav"
    await asyncio.to_thread(torchaudio.save, clip_path, wav[:, a:b], sr)
    return clip_path


async def _resolve_intent_uncached(session: CallSession, text: str, lang: str, key: str) -> dict:
    """Tier 2 (semantic cache) + tier 3 (LLM), run AFTER fast_path already
    abstained. Split out from _resolve_intent so the filler race below can
    wrap the WHOLE remaining sequence -- see KCD-459 note there."""
    cached, how = await asyncio.to_thread(_intent_cache.get, key)
    if cached is not None:
        logger.info("[%s] intent cache %s hit", session.call_id, how)
        return cached

    data, diag = await asyncio.to_thread(extract_intent, text, 2, lang)
    logger.info("[%s] intent extracted in %.2fs (%d attempt(s))",
                session.call_id, diag["total_time_s"], diag["attempts"])
    await asyncio.to_thread(_intent_cache.put, key, data)
    return data


async def _resolve_intent(session: CallSession, text: str, lang: str = "bn") -> dict:
    """Semantic cache in front of the LLM. A hit skips Ollama entirely --
    the slowest hop in the turn -- but the clinic lookup that follows still
    runs live, so a cached intent can never serve a stale price."""
    # Tier 1: decide it locally if we can. For a fixed catalogue the
    # entity is a string-matching problem with a 0.32 confidence margin,
    # where the embedding route had 0.03 -- see agent/fast_path.py. This
    # returns None whenever it is not sure, which is the common case for
    # anything except a routine price or availability question. Its cues
    # are Bengali, so for Hindi/English it mostly abstains and the turn
    # goes to the LLM -- safe by construction, just not free.
    await _maybe_reload_fast_path()
    if _fast_path is not None:
        hit = await asyncio.to_thread(_fast_path.resolve, text)
        if hit is not None:
            logger.info("[%s] fast path resolved %s (%.2f) -- no LLM call",
                        session.call_id, hit.intent, hit.confidence)
            return hit.as_llm_shape()

    # The cache stores the whole intent object (smalltalk carries reply
    # text in the caller's language), so the language is part of the key.
    key = text if lang == "bn" else f"[{lang}] {text}"

    # KCD-459: the filler race used to wrap ONLY the LLM call, so a slow
    # embedding lookup (agent/semantic_cache.py's L2 tier, a live call to
    # Ollama's embed endpoint) burned into FILLER_THRESHOLD_S's budget
    # silently before the caller ever got the "please hold" filler -- a
    # cold/loaded embed step and a cold LLM call could sum past the
    # threshold with the caller hearing nothing for either half. Wrapping
    # the whole cache-then-LLM sequence means "no more than
    # FILLER_THRESHOLD_S of silence" is an end-to-end guarantee for this
    # tier, not just a promise about the LLM's own share of it.
    return await _await_with_filler(
        session, _resolve_intent_uncached(session, text, lang, key), lang)


# CodeRabbit-flagged, real bug: a failed hold used to ALWAYS clear
# time_slot, regardless of why hold_slot actually failed ("keep
# doctor/date, re-collect just the time" -- true only for a slot-shaped
# failure). For "doctor_not_found" or a date-shaped failure, the ACTUAL
# problem field was never cleared, so missing_required() still saw
# doctor_name/date as present and the SAME bad hold was retried forever
# on every subsequent turn, having only pointlessly re-asked for a time
# that was never the issue.
_HOLD_FAILURE_FIELD = {
    "doctor_not_found": "doctor_name",
    "invalid_date": "date",
    "date_in_past": "date",
    "doctor_not_available_that_day": "date",
}


def _field_to_reclear_on_hold_failure(reason: str | None) -> str:
    return _HOLD_FAILURE_FIELD.get(reason, "time_slot")


async def _answer_enquiry_intent(intent: str, slots: dict, lang: str) -> str | None:
    """Tool call + reply-template for one of the stateless enquiry
    intents (agent.enquiry_followup.ENQUIRY_INTENTS). Returns None when
    the intent's required slot is missing -- the caller decides whether
    that means "ask the caller" (the turn's primary question) or
    "silently skip" (a KCD-395 secondary question), never this function.
    The single implementation per intent here is also what a secondary
    question is answered with, so there is exactly one place each
    intent's tool call and reply template are wired together."""
    if intent == "test_rate":
        if not slots.get("test_name"):
            return None
        result = await _tools.get_test_rate(slots["test_name"])
        return test_rate_reply(slots, result, lang)
    if intent == "doctor_availability":
        if not slots.get("doctor_name"):
            return None
        result = await _tools.get_doctor_availability(slots["doctor_name"], slots.get("date"))
        return doctor_availability_reply(slots, result, lang)
    if intent == "test_prep":
        if not slots.get("test_name"):
            return None
        result = await _tools.get_test_prep(slots["test_name"], lang)
        return test_prep_reply(slots, result, lang)
    if intent == "clinic_faq":
        if not slots.get("faq_topic"):
            return None
        result = await _tools.get_faq(slots["faq_topic"], lang)
        return clinic_faq_reply(slots, result, lang)
    if intent == "department_query":
        symptom = slots.get("symptom_description")
        if not symptom:
            return None
        result = await _tools.route_department(symptom, lang)
        return department_route_reply(result, symptom, lang)
    return None


async def _finish_enquiry_turn(session: CallSession, text: str, lang: str,
                               intent: str, slots: dict, base_reply: str, data: dict) -> None:
    """Shared tail for every stateless enquiry branch (KCD-395/KCD-396):
    answers `secondary_intent` if the model extracted one this turn,
    remembers this turn's entities for the NEXT turn's coreference
    resolution, then speaks the combined reply.

    A secondary lookup that fails (ToolCallError) never discards the
    primary answer the caller already got -- it is reported as "couldn't
    check that" addendum instead, same as this call's own primary
    ToolCallError handling one level up in _dispatch_turn, just scoped
    so a second-question failure cannot blow away a first-question
    success."""
    turn_entities = [(intent, slots)]
    secondary_intent = data.get("secondary_intent")
    if secondary_intent:
        secondary_slots = data.get("secondary_slots") or {}
        secondary_slots, _ = resolve_followup_slot(
            secondary_intent, secondary_slots, text, lang, session.last_enquiry_entities)
        addendum = None
        try:
            reply2 = await _answer_enquiry_intent(secondary_intent, secondary_slots, lang)
        except ToolCallError as e:
            logger.warning("[%s] secondary-question lookup failed (%s) -- keeping the first answer",
                           session.call_id, e)
            reply2, addendum = None, phrase("tool_failure", lang)
        if reply2:
            base_reply = f"{base_reply} {reply2}"
            turn_entities.append((secondary_intent, secondary_slots))
        else:
            # A second question was asked but could not be answered this
            # turn (missing slot, or the lookup itself failed) -- KCD-395
            # requires it be explicitly addressed, never silently dropped.
            base_reply = f"{base_reply} {addendum or phrase('unclear', lang)}"
    session.last_enquiry_entities = entities_from_turn(*turn_entities)
    await _speak(session, base_reply, lang)


async def _route_and_transcribe(session: CallSession, utterance_wav: str):
    """LID -> routing decision -> ASR. Returns (language, ASRResult), or
    (None, None) when the router says a human should take the call."""
    if _lid is None or len(_languages_active) < 2:
        return "bn", await _asr_router.transcribe("bn", utterance_wav)

    try:
        lid = await asyncio.to_thread(_lid.identify_path, utterance_wav)
    except Exception as e:  # noqa: BLE001 - a LID fault must degrade the turn, not kill it
        logger.warning("[%s] LID failed (%s) -- treating as unknown", session.call_id, e)
        lid = LIDResult(language="unknown", confidence=0.0)

    # LID's top label alone is not trusted: Indian-accented English is
    # labelled Hindi at ~0.9, or not found at all (measured; see
    # agent/lang_select.py). Unless LID is decisive, let every active ASR
    # try and keep the one whose decoders agree.
    verify = languages_to_verify(lid.language, lid.scores, _languages_active) if lid.scores else None
    if verify:
        outcomes = await _asr_router.transcribe_many(verify, utterance_wav)
        lang, result = pick_candidate(outcomes)
        logger.info("[%s] LID %s %s not decisive -> ran %s, chose %s (%s)",
                    session.call_id, lid.language, {k: round(v, 2) for k, v in lid.scores.items()},
                    "+".join(l for l, _ in outcomes), lang,
                    ", ".join(f"{l}:{r.decoder_agreement:.2f}" for l, r in outcomes))
        session.lang_router.note_response_language(lang)
        return lang, result

    decision = session.lang_router.route(lid)
    logger.info("[%s] LID %s %.2f -> %s %s (%s)", session.call_id, lid.language, lid.confidence,
                decision.action, decision.language or "", decision.reason)

    if decision.action == "handoff_human":
        return None, None

    if decision.action == "dual_asr" and decision.secondary_language:
        primary, secondary = decision.language, decision.secondary_language
        r1, r2 = await _asr_router.transcribe_dual(primary, secondary, utterance_wav)
        return pick_candidate([(primary, r1), (secondary, r2)])

    return decision.language, await _asr_router.transcribe(decision.language, utterance_wav)


async def _dispatch_turn(session: CallSession, utterance_wav: str):
    """One full turn: LID -> ASR -> intent -> tool -> templated reply -> TTS.
    Serialized per-call via session.dispatch_lock so replies never
    interleave, even if the caller starts talking again immediately."""
    async with session.dispatch_lock:
        session.turn_started_at = time.monotonic()
        channel_quality = session.call_state.channel_quality
        analysis: dict | None = None
        audio_issues: list[str] = []
        try:
            lang, asr_result = await _route_and_transcribe(session, utterance_wav)
            # KCD-075/KCD-076: measured inside its own budget slice, on
            # the SAME clip ASR already read, before the finally below
            # deletes it. A slow or failed classification just keeps
            # last turn's label -- never blocks or fails the turn.
            channel_quality = await run_within_budget(
                "channel_quality", CHANNEL_QUALITY_BUDGET_S, _classify_channel_from_wav_path,
                utterance_wav, previous_value=channel_quality)
            # Why is this clip hard to hear, and does the speaker sound older?
            # A slow or failed analysis just means no verdict this turn.
            analysis = await run_within_budget(
                "audio_quality", AUDIO_ANALYSIS_BUDGET_S, _analyze_utterance_from_wav_path,
                utterance_wav, previous_value=None)
            audio_issues = analysis["audio"].issues if analysis else []
            for issue in audio_issues:
                audio_issue_buckets.record(issue, "seen", lang or session.lang)
            asr_result = await _retry_on_enhanced_audio(
                session, utterance_wav, lang, asr_result, audio_issues,
                analysis["audio"].duration_s if analysis else None)
        except Exception as e:
            logger.exception("[%s] ASR/LID stage failed: %s", session.call_id, e)
            await _speak(session, phrase("llm_failure", session.lang), session.lang,
                         fallback_reason="llm_failure")
            return
        finally:
            with contextlib.suppress(OSError):
                os.remove(utterance_wav)

        if lang is None:
            # Ask again, kindly, before any hand-off -- an unidentifiable
            # utterance is usually a faint, noisy or mumbled one, not a
            # request for a person (agent/reask_policy.py).
            await _reask_or_handoff(
                session, session.reask.decide(language_ambiguous=True, audio_issues=audio_issues),
                session.lang, languages=_languages_active)
            return
        session.lang = lang
        session.lang_router.note_response_language(lang)
        apply_language(session.call_state, lang)
        apply_confidence(session.call_state, is_low_confidence(asr_result.decoder_agreement))
        apply_channel_quality(session.call_state, channel_quality)
        channel_quality_buckets.record(channel_quality, "seen", lang)
        logger.debug("[%s] call_state %s", session.call_id, session.call_state.to_log_dict())

        text = asr_result.text.strip()
        if not text:
            logger.info("[%s] ASR returned empty text", session.call_id)
            await _reask_or_handoff(
                session, session.reask.decide(asr_empty=True, audio_issues=audio_issues), lang,
                fallback_reason="asr_empty")
            return
        # A yes/no to a confirmation is legitimately one short word; the
        # jumbled-transcript checks would misread it as a fragment.
        # The same holds for a dictated phone number, a spelled name or a bare
        # "না" while a booking is collecting slots: all legitimately short or
        # tokenised, so only decoder disagreement is meaningful then.
        in_booking_flow = session.booking is not None and (
            session.booking.stage in ("confirming", "awaiting_charge_confirm")
            or not session.booking.is_stale())
        problem = transcript_problem(
            text, lang, analysis["audio"].duration_s if analysis else None, asr_result.decoder_agreement,
            slot_answer=in_booking_flow)
        if problem:
            logger.info("[%s] jumbled transcript (%s)", session.call_id, problem)
            await _reask_or_handoff(
                session, session.reask.decide(transcript_issue=problem, audio_issues=audio_issues), lang)
            return
        session.reask.note_success()
        await session.send_json("User", text)
        code_switch_buckets.record(mixture_bucket(text), "seen", lang)

        # KCD-084/149/155: caller state -> delivery policy, and the
        # acknowledgement that comes before anything else when the policy
        # calls for one (once per state entry, not every turn).
        _update_caller_state(session, text, lang, analysis)
        ack = select_acknowledgement(session.policy, session.call_state.caller_state, lang)
        if ack and session.acknowledged_state != session.call_state.caller_state:
            session.acknowledged_state = session.call_state.caller_state
            await _speak(session, ack, lang)

        # KCD-438: an explicit "speak in Hindi/Bengali/English" request,
        # detected deterministically (no LLM call, same zero-extra-latency
        # reasoning as the yes/no check below). Acknowledged immediately in
        # the requested language, and biases the language-ID router's
        # ambiguous-turn prior toward it (note_response_language) -- see
        # agent/language_switch.py's docstring for the deliberate scope
        # limit: this does not force every later reply into the requested
        # language regardless of what the caller goes on to actually say.
        switch_target = detect_language_switch_request(text, lang)
        if switch_target:
            session.lang_router.note_response_language(switch_target)
            await _speak(session, phrase("language_switched", switch_target), switch_target)
            return

        # Epic E26: a booking/reschedule/cancel/add-test confirmation
        # already in progress is a closed yes/no question -- answered
        # deterministically (agent/booking_flow.classify_yes_no), with NO
        # LLM round trip, both for the zero-extra-latency requirement and
        # because CLAUDE.md's truth boundary keeps exactly this kind of
        # high-stakes binary decision out of the model's hands.
        if session.booking is not None and session.booking.stage in ("confirming", "awaiting_charge_confirm"):
            await _handle_booking_confirmation_turn(session, text, lang)
            return

        try:
            data = await _resolve_intent(session, text, lang)
        except ExtractionError as e:
            logger.error("[%s] intent extraction failed: %s", session.call_id, e)
            await _speak(session, phrase("llm_failure", lang), lang, fallback_reason="llm_failure")
            return

        intent = data["intent"]
        slots = data["slots"]
        _note_stated_age(session, slots, lang)

        if intent == "smalltalk":
            reply = data.get("direct_reply_bn")
            await _speak(session, reply if _speakable(reply or "", lang)
                         else phrase("smalltalk_default", lang), lang)
            return

        if intent == "unclear":
            # CodeRabbit-flagged, real gap: the "confirming"/
            # "awaiting_charge_confirm" short-circuit above only covers
            # the yes/no step. During ordinary "collecting"-stage slot
            # filling, a bare answer with no sentence structure to key
            # off ("Ravi Das", a bare 10-digit number) is exactly the
            # kind of turn the LLM classifies "unclear" even when it DID
            # correctly extract the slot into `slots` -- and nothing
            # routed it back to the active booking, silently stalling
            # the guided one-slot-per-turn flow on the minimal answers
            # it is meant to collect. Re-attempt this turn as the active
            # action instead of bailing, as long as the state is not
            # stale (is_stale() -- previously never called anywhere,
            # per CodeRabbit -- guards against resurrecting a booking
            # the caller has plainly moved on from).
            if session.booking is not None and session.booking.stage == "collecting" \
                    and not session.booking.is_stale():
                intent = session.booking.action
            else:
                await _speak(session, phrase("unclear", lang), lang)
                return

        # KCD-442/KCD-447: a turn the two ASR decoders disagreed on is not
        # trusted to drive a factual lookup at all -- acting on a misheard
        # test/doctor name would produce a confident, wrong answer about
        # something else, which is worse than asking the caller to repeat
        # themselves. Checked before the tool call, not after: the point
        # is to never RUN the lookup on an unreliable entity, not merely
        # to hedge the reply once it comes back.
        if should_withhold_factual_answer(intent, asr_result.decoder_agreement):
            logger.info("[%s] withholding %s answer: decoder_agreement=%.2f below floor",
                        session.call_id, intent, asr_result.decoder_agreement)
            insufficient_information.record("low_decoder_agreement", intent, lang)
            await _speak(session, insufficient_information_reply(lang), lang)
            return

        try:
            if intent == "test_rate":
                slots, _ = resolve_followup_slot(intent, slots, text, lang, session.last_enquiry_entities)
                reply = await _answer_enquiry_intent(intent, slots, lang)
                if reply is None:
                    await _speak(session, missing_slot_prompt(intent, "test_name", lang), lang)
                    return
                await _finish_enquiry_turn(session, text, lang, intent, slots, reply, data)

            elif intent == "doctor_availability":
                slots, _ = resolve_followup_slot(intent, slots, text, lang, session.last_enquiry_entities)
                reply = await _answer_enquiry_intent(intent, slots, lang)
                if reply is None:
                    await _speak(session, missing_slot_prompt(intent, "doctor_name", lang), lang)
                    return
                await _finish_enquiry_turn(session, text, lang, intent, slots, reply, data)

            elif intent == "test_prep":
                slots, _ = resolve_followup_slot(intent, slots, text, lang, session.last_enquiry_entities)
                reply = await _answer_enquiry_intent(intent, slots, lang)
                if reply is None:
                    await _speak(session, missing_slot_prompt(intent, "test_name", lang), lang)
                    return
                await _finish_enquiry_turn(session, text, lang, intent, slots, reply, data)

            elif intent == "clinic_faq":
                slots, _ = resolve_followup_slot(intent, slots, text, lang, session.last_enquiry_entities)
                reply = await _answer_enquiry_intent(intent, slots, lang)
                if reply is None:
                    await _speak(session, missing_slot_prompt(intent, "faq_topic", lang), lang)
                    return
                await _finish_enquiry_turn(session, text, lang, intent, slots, reply, data)

            elif intent == "book_appointment":
                if session.booking is None or session.booking.action != "book_appointment":
                    session.booking = new_state("book_appointment")
                st = session.booking
                prior_slots = dict(st.slots)
                changed = merge_slots(st, slots)
                ack = correction_acknowledgement(changed, prior_slots, st.slots, lang)
                if ack:
                    await _speak(session, ack, lang)
                else:
                    await _echo_new_slots(session, changed, prior_slots, st.slots, lang)

                # Secure the hold as soon as doctor+date+time are known, even
                # if patient details are still missing -- KCD-376: the
                # concurrency guard must run before the caller has spoken a
                # patient name, not after.
                if st.hold_token is None and st.slots.get("doctor_name") and st.slots.get("date") \
                        and st.slots.get("time_slot"):
                    hold = await _tools.hold_slot(st.slots["doctor_name"], st.slots["date"], st.slots["time_slot"])
                    if not hold.get("success"):
                        await _speak(session, booking_reply(st.slots, hold, lang), lang)
                        st.slots.pop(_field_to_reclear_on_hold_failure(hold.get("reason")), None)
                        return
                    st.hold_token, st.hold_doctor_id = hold["hold_token"], hold["doctor_id"]

                # KCD-368: a caller may spell a name unprompted, or after
                # being offered it below -- either way, letters heard this
                # turn are assembled and used before anything else is
                # asked, so an eager caller is never told to repeat
                # themselves. Confirmed by readback like every other slot.
                if not st.slots.get("patient_name") and slots.get("spelled_letters"):
                    spelled = merge_spelling(st, slots["spelled_letters"])
                    if spelled:
                        st.slots["patient_name"] = spelled.capitalize()
                        await _speak(session, spelling_readback(spelled, lang), lang)
                        return

                missing = missing_required(st)
                if missing:
                    field_name = missing[0]
                    if field_name == "phone" and st.note_retry("phone") >= 2:
                        # KCD-370: a caller who won't give a number still
                        # gets the booking, told plainly what that costs
                        # them, instead of being blocked here forever.
                        st.phone_declined = True
                    elif field_name == "patient_name" and st.note_retry("patient_name") >= 2:
                        # KCD-368: offered proactively after one failed
                        # capture, not only when the caller asks for it.
                        await _speak(session, spelling_prompt(lang), lang)
                        return
                    else:
                        await _speak(session, missing_slot_prompt(intent, field_name, lang), lang)
                        return
                    missing = missing_required(st)
                    if missing:
                        await _speak(session, missing_slot_prompt(intent, missing[0], lang), lang)
                        return

                # Skipped when no real phone exists to check by -- the
                # "not_provided" sentinel must never be used as a lookup
                # key, or two different callers who both declined to give
                # a number would appear to conflict with EACH OTHER.
                if effective_phone(st) != "not_provided" and not st.slots.get("_conflict_checked"):
                    conflict = await _tools.booking_conflict(
                        effective_phone(st), st.slots["date"], st.slots["time_slot"])
                    st.slots["_conflict_checked"] = "1"
                    if conflict.get("conflict"):
                        await _speak(session, conflict_reply(conflict["existing"], lang), lang)
                        return

                if is_ready_to_confirm(st):
                    mark_confirming(st)
                    if st.phone_declined:
                        await _speak(session, phrase("no_confirmation_number", lang), lang)
                    await _speak(session, booking_confirmation_readback(st.slots, "book_appointment", lang), lang)

            elif intent == "book_test":
                if session.booking is None or session.booking.action != "book_test":
                    session.booking = new_state("book_test")
                st = session.booking
                prior_slots = dict(st.slots)
                changed = merge_slots(st, slots)
                ack = correction_acknowledgement(changed, prior_slots, st.slots, lang)
                if ack:
                    await _speak(session, ack, lang)
                else:
                    await _echo_new_slots(session, changed, prior_slots, st.slots, lang)
                st.slots["_test_names_display"] = st.test_names
                missing = missing_required(st)
                if missing:
                    field_name = missing[0]
                    if field_name == "phone" and st.note_retry("phone") >= 2:
                        st.phone_declined = True
                    else:
                        await _speak(session, missing_slot_prompt(intent, field_name, lang), lang)
                        return
                    missing = missing_required(st)
                    if missing:
                        await _speak(session, missing_slot_prompt(intent, missing[0], lang), lang)
                        return
                if is_ready_to_confirm(st):
                    mark_confirming(st)
                    if st.phone_declined:
                        await _speak(session, phrase("no_confirmation_number", lang), lang)
                    await _speak(session, booking_confirmation_readback(st.slots, "book_test", lang), lang)

            elif intent == "reschedule_appointment":
                if session.booking is None or session.booking.action != "reschedule_appointment":
                    session.booking = new_state("reschedule_appointment")
                st = session.booking
                prior_slots = dict(st.slots)
                changed = merge_slots(st, slots)
                ack = correction_acknowledgement(changed, prior_slots, st.slots, lang)
                if ack:
                    await _speak(session, ack, lang)
                else:
                    await _echo_new_slots(session, changed, prior_slots, st.slots, lang)
                if not st.slots.get("confirmation_id") and slots.get("phone"):
                    found = await _tools.lookup_bookings(phone=slots["phone"])
                    bookings = found.get("bookings") or []
                    if len(bookings) > 1:
                        # CodeRabbit-flagged: never silently act on
                        # bookings[0] -- a caller with several bookings
                        # names which one before anything proceeds.
                        await _speak(session, multiple_bookings_reply(bookings, lang), lang)
                        return
                    if bookings:
                        st.slots["confirmation_id"] = bookings[0]["confirmation_id"]
                missing = missing_required(st)
                if missing:
                    await _speak(session, missing_slot_prompt(intent, missing[0], lang), lang)
                    return
                if is_ready_to_confirm(st):
                    mark_confirming(st)
                    await _speak(session, booking_confirmation_readback(st.slots, "reschedule_appointment", lang), lang)

            elif intent == "cancel_appointment":
                if session.booking is None or session.booking.action != "cancel_appointment":
                    session.booking = new_state("cancel_appointment")
                st = session.booking
                prior_slots = dict(st.slots)
                changed = merge_slots(st, slots)
                ack = correction_acknowledgement(changed, prior_slots, st.slots, lang)
                if ack:
                    await _speak(session, ack, lang)
                else:
                    await _echo_new_slots(session, changed, prior_slots, st.slots, lang)
                if not st.slots.get("confirmation_id") and slots.get("phone"):
                    found = await _tools.lookup_bookings(phone=slots["phone"])
                    bookings = found.get("bookings") or []
                    if len(bookings) > 1:
                        await _speak(session, multiple_bookings_reply(bookings, lang), lang)
                        return
                    if bookings:
                        st.slots["confirmation_id"] = bookings[0]["confirmation_id"]
                missing = missing_required(st)
                if missing:
                    await _speak(session, missing_slot_prompt(intent, missing[0], lang), lang)
                    return
                if is_ready_to_confirm(st):
                    mark_confirming(st)
                    await _speak(session, booking_confirmation_readback(st.slots, "cancel_appointment", lang), lang)

            elif intent == "add_test_booking":
                if session.booking is None or session.booking.action != "add_test_booking":
                    session.booking = new_state("add_test_booking")
                st = session.booking
                prior_slots = dict(st.slots)
                changed = merge_slots(st, slots)
                ack = correction_acknowledgement(changed, prior_slots, st.slots, lang)
                if ack:
                    await _speak(session, ack, lang)
                else:
                    await _echo_new_slots(session, changed, prior_slots, st.slots, lang)
                missing = missing_required(st)
                if missing:
                    await _speak(session, missing_slot_prompt(intent, missing[0], lang), lang)
                    return
                if is_ready_to_confirm(st):
                    mark_confirming(st)
                    await _speak(session, booking_confirmation_readback(st.slots, "add_test_booking", lang), lang)

            elif intent == "lookup_booking":
                if not (slots.get("phone") or slots.get("confirmation_id")):
                    await _speak(session, missing_slot_prompt(intent, "phone", lang), lang)
                    return
                result = await _tools.lookup_bookings(
                    phone=slots.get("phone"), confirmation_id=slots.get("confirmation_id"))
                await _speak(session, lookup_reply(result.get("bookings") or [], lang), lang)

            elif intent == "resend_confirmation":
                confirmation_id = slots.get("confirmation_id")
                if not confirmation_id and slots.get("phone"):
                    found = await _tools.lookup_bookings(phone=slots["phone"])
                    bookings = found.get("bookings") or []
                    if len(bookings) > 1:
                        await _speak(session, multiple_bookings_reply(bookings, lang), lang)
                        return
                    if bookings:
                        confirmation_id = bookings[0]["confirmation_id"]
                if not confirmation_id:
                    await _speak(session, missing_slot_prompt("cancel_appointment", "confirmation_id", lang), lang)
                    return
                result = await _tools.resend_confirmation(confirmation_id)
                await _speak(session, resend_reply(result, lang), lang)

            elif intent == "department_query":
                reply = await _answer_enquiry_intent(intent, slots, lang)
                if reply is None:
                    await _speak(session, phrase("unclear", lang), lang)
                    return
                await _finish_enquiry_turn(session, text, lang, intent, slots, reply, data)

        except ToolCallError as e:
            logger.error("[%s] clinic API call failed: %s", session.call_id, e)
            await _speak(session, phrase("tool_failure", lang), lang, fallback_reason="tool_failure")


async def _handle_booking_confirmation_turn(session: CallSession, text: str, lang: str) -> None:
    """The caller's answer to a "shall I confirm this?" readback --
    resolved deterministically (see the classify_yes_no call site in
    _dispatch_turn), never via the LLM.

    "no" means different things by action: for cancel_appointment there is
    nothing to correct, so "no" simply aborts it (KCD-372's spirit -- the
    caller was asked once with the charge stated, a "no" ends it there).
    For every other action, "no" re-opens collection instead of discarding
    the booking, because the far more common reason a caller says "no"
    here is "something's wrong", not "forget the whole thing" -- and
    KCD-367 explicitly asks for a correction to re-enter slot filling
    rather than restart. An unrecognised answer just repeats the question
    rather than guessing which the caller meant."""
    st = session.booking
    assert st is not None
    answer = classify_yes_no(text, lang)

    if answer is None:
        await _speak(session, booking_confirmation_readback(st.slots, st.action, lang), lang)
        return

    if answer == "no":
        if st.action == "cancel_appointment":
            await _speak(session, phrase("cancel_aborted", lang), lang)
            session.booking = None
            return
        st.stage = "collecting"
        st.pending_charge_inr = None
        await _speak(session, missing_slot_prompt(st.action,
                     "doctor_name" if st.action == "book_appointment" else "date", lang), lang)
        return

    # answer == "yes": commit for real.
    try:
        if st.action == "book_appointment":
            phone = effective_phone(st)
            result = await _tools.confirm_booking(
                st.hold_token, st.hold_doctor_id, st.slots["date"], st.slots["time_slot"],
                st.slots["patient_name"], phone, caller_phone=phone,
                patient_age=st.slots.get("patient_age"), relationship=st.slots.get("relationship") or "self",
            )
            if result.get("reason") == "write_unverified":
                session.booking = None
                await _handle_unverified_write(session, lang)
                return
            await _speak(session, booking_reply(st.slots, result, lang), lang)
            if result.get("success"):
                await _persist_senior(session, phone)
            if result.get("success") or result.get("reason") != "hold_expired":
                session.booking = None
            else:
                st.hold_token, st.stage = None, "collecting"   # let a retry re-hold

        elif st.action == "book_test":
            phone = effective_phone(st)
            result = await _tools.book_tests(
                st.test_names, st.slots["date"], st.slots["patient_name"], phone,
                caller_phone=phone, patient_age=st.slots.get("patient_age"),
                relationship=st.slots.get("relationship") or "self",
            )
            await _speak(session, multi_test_reply(result, lang), lang)
            session.booking = None

        elif st.action == "reschedule_appointment":
            result = await _tools.reschedule_appointment(
                st.slots["confirmation_id"], st.slots["new_date"], st.slots["new_time_slot"])
            if result.get("reason") == "write_unverified":
                session.booking = None
                await _handle_unverified_write(session, lang)
                return
            await _speak(session, reschedule_reply(result, lang), lang)
            if result.get("success") or result.get("reason") != "slot_taken":
                session.booking = None
            else:
                st.stage = "collecting"
                st.slots.pop("new_time_slot", None)

        elif st.action == "cancel_appointment":
            confirm_charge = st.stage == "awaiting_charge_confirm"
            result = await _tools.cancel_appointment(st.slots["confirmation_id"], confirm_charge)
            if not result.get("success") and result.get("reason") == "charge_confirmation_required":
                mark_awaiting_charge_confirm(st, result["charge_inr"])
                await _speak(session, cancel_reply(result, lang), lang)
                return
            await _speak(session, cancel_reply(result, lang), lang)
            session.booking = None

        elif st.action == "add_test_booking":
            result = await _tools.add_test_to_booking(st.slots["confirmation_id"], st.slots["test_name"])
            await _speak(session, add_test_reply(result, lang), lang)
            session.booking = None

    except ToolCallError as e:
        logger.error("[%s] clinic API call failed during booking confirmation: %s", session.call_id, e)
        await _speak(session, phrase("tool_failure", lang), lang, fallback_reason="tool_failure")
        session.booking = None


async def _resync_after_playback(session: CallSession) -> bool:
    """Drop everything captured while the agent was talking, by moving
    processed_until_s to the current end of the decoded buffer. That region
    is muted silence from the client's side; skipping it keeps the turn
    detector from ever analysing it, and -- more importantly -- keeps
    processed_until_s anchored to real time instead of drifting a full
    reply behind, which is what made later turns surface late."""
    if not await _decode_to_wav(session.raw_path, session.wav_path):
        return False
    wav, sr = await asyncio.to_thread(torchaudio.load, session.wav_path)
    buffer_end_s = wav.shape[-1] / sr
    session.processed_until_s = max(session.processed_until_s,
                                    buffer_end_s - RESYNC_REWIND_S)
    session.resync_pending = False
    logger.info("[%s] resynced to %.2fs after playback", session.call_id, session.processed_until_s)
    return True


async def _turn_poll_loop(session: CallSession):
    """Runs for the lifetime of the call. Every POLL_INTERVAL_S, re-decodes
    the growing buffer -- ALWAYS from byte 0, since that's the only way
    the WebM container stays valid -- then asks the turn detector "is the
    caller done talking yet?" using only the slice of audio past
    session.processed_until_s (a prior turn's already-consumed audio).
    On yes: slice that utterance out for ASR, hand it to _dispatch_turn as
    a background task (so ingestion of the NEXT turn's audio is never
    blocked by this turn's ASR/LLM/TTS work), and advance the marker.
    """
    while True:
        await asyncio.sleep(POLL_INTERVAL_S)

        if time.time() - session.last_activity > IDLE_TIMEOUT_S:
            logger.info("[%s] idle timeout, closing", session.call_id)
            await _speak(session, phrase("idle_close", session.lang), session.lang)
            with contextlib.suppress(Exception):
                await session.ws.close()
            return

        if time.time() - session.last_heartbeat > HEARTBEAT_INTERVAL_S:
            session.last_heartbeat = time.time()
            with contextlib.suppress(Exception):
                await session.ws.send_text('{"sender":"_ping","text":""}')

        # --- half-duplex gate: never run turn detection on our own voice ---
        if session.agent_speaking:
            if time.time() < session.speak_deadline:
                continue
            logger.warning("[%s] no playback-done from client, releasing gate on deadline",
                           session.call_id)
            session.release_gate()

        if session.resync_pending:
            await _resync_after_playback(session)
            continue

        if not await _decode_to_wav(session.raw_path, session.wav_path):
            continue  # too little data yet to form a valid container -- not an error

        wav, sr = await asyncio.to_thread(torchaudio.load, session.wav_path)
        wav = wav.mean(dim=0) if wav.shape[0] > 1 else wav.squeeze(0)

        tail_start_sample = min(int(session.processed_until_s * sr), wav.shape[-1])
        tail = wav[tail_start_sample:]

        result = await asyncio.to_thread(_turn_detector.poll, tail, sr)
        if result.utterance_end_s is None:
            continue

        absolute_end_s = session.processed_until_s + result.utterance_end_s
        session.utt_seq += 1
        utterance_wav = await _slice_utterance(
            session, session.processed_until_s, absolute_end_s, session.utt_seq,
        )
        session.processed_until_s = absolute_end_s
        asyncio.create_task(_dispatch_turn(session, utterance_wav))


async def _handle_control(session: CallSession, raw: str):
    """Client -> server control channel.

    "playback_done" is the load-bearing half of the echo gate: the server
    cannot otherwise know when the caller's speaker actually stopped.

    "interrupt" (KCD-464) is the manual stop affordance: static/index.html
    shows a button while agent_speaking, and clicking it stops local
    playback, unmutes the mic and sends this immediately, WITHOUT waiting
    for the queued clips to finish. Deliberately NOT acoustic barge-in --
    this file's own module docstring already explains why that needs an
    echo canceller this system does not have (no reference signal for
    audio synthesized locally and played through Web Audio). This is the
    honest, buildable version of "stop the AI, let the caller talk": the
    caller asks for the floor instead of the system trying to detect it
    from the muted mic, which structurally cannot see it."""
    try:
        msg = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("[%s] unparseable control frame: %r", session.call_id, raw[:80])
        return
    if msg.get("type") == "playback_done":
        session.release_gate()
    elif msg.get("type") == "interrupt":
        was_speaking = session.agent_speaking
        session.release_gate()
        if was_speaking:
            logger.info("[%s] caller interrupt -- gate released early", session.call_id)
            barge_in_interrupts.record("manual", session.lang)


@app.websocket("/ws/audio")
async def ws_audio(ws: WebSocket):
    await ws.accept()
    session = CallSession(ws)

    # The door. A call the AI cannot serve goes to a person NOW -- never
    # queued behind the GPU. See agent/admission.py for what closes it.
    admission = _admission.try_admit()
    if not admission.admitted:
        logger.warning("[%s] call refused: %s", session.call_id, admission.reason)
        try:
            await _handoff_to_human(session, admission.reason, languages=HANDOFF_ALL_LANGUAGES)
        finally:
            session.cleanup()
        return
    session.admission = admission
    logger.info("[%s] call started (%d active)", session.call_id, _admission.active_calls)
    poll_task = asyncio.create_task(_turn_poll_loop(session))

    try:
        await _speak(session, phrase("greeting", "bn"), "bn")
        while True:
            message = await ws.receive()
            if message["type"] == "websocket.disconnect":
                break
            if message.get("bytes") is not None:
                await session.append(message["bytes"])
            elif message.get("text"):
                await _handle_control(session, message["text"])
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("[%s] session crashed", session.call_id)
    finally:
        poll_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await poll_task
        _admission.release(admission)
        session.cleanup()
        logger.info("[%s] call ended (%d active)", session.call_id, _admission.active_calls)


app.mount("/", StaticFiles(directory="static", html=True), name="static")
