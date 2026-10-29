"""Kolkata Care Diagnostics -- voice agent on the RAW PCM transport.

GENERATED FILE -- do not edit directly. Produced by
tools/make_pcm_variant.py from main.py; edit main.py and re-run that.

This variant changes only the TRANSPORT (how audio arrives and how the
turn detector reads it):
  * No ffmpeg, no WebM, no temp audio files, no subprocess per poll.
  * Appending is O(chunk) and reading the tail is O(tail), where the WebM
    path was O(call length) EVERY poll -- O(T^2) per call.
  * The sample index IS the timeline, exactly, so processed_until_s
    cannot drift away from real time.
See agent/pcm_buffer.py for the argument and the arithmetic.

----------------------------------------------------------------------


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
import collections
import contextlib
import difflib
import re
import dataclasses
import shutil
import contextvars
import datetime
import io
import json
import logging
import os
import secrets
import sys
import tempfile
import time
import uuid
import wave

import numpy as np
import torchaudio
from agent.pcm_buffer import PcmCallBuffer, SAMPLE_RATE
from fastapi import FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agent import abuse, action_gate, call_end, slot_grouping, topic_flow
from agent import ws_guard  # limits and validation for what a client may send (review 2026-10-28)
from agent import head_pain  # "mathar betha": a headache is asked about once (hurt or only a headache) before a department is named
from agent import where_doctor_sits  # which floor / room a doctor sits in: the counter tells
from agent import entity_confirmation as entity_text
from agent import history_templates as history_text
from agent import messages as agent_messages
from agent import patient_context as patient_policy
from agent import security_check as sec_text
from agent.acknowledgement import select_acknowledgement, slot_echo
from agent.admission import AdmissionController, HealthMonitor, http_probe
from agent.apology import count_apologies, enforce_single_apology
from agent.asr import TurnASR
from agent.asr_router import ASRRouter, HTTPASREngine, UnroutableLanguageError
from agent.attention import attend
from agent.audio_quality import assess as assess_audio
from agent.audio_quality import transcript_problem
from agent.booking_flow import (
    what_to_change_prompt,
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
from agent.call_record import FINISH_DEADLINE_S as CALL_RECORD_DEADLINE_S
from agent.call_record import CallRecorder
from agent.call_score import CallSignals, score_call
from agent.call_state import (
    CallState,
    apply_caller_state,
    apply_channel_quality,
    apply_confidence,
    apply_language,
    new_call_state,
)
from agent.call_summary import build_call_summary
from agent.channel_quality import CHANNEL_CLEAN_16K, classify_channel
from agent.clause_split import split_into_clauses
from agent.code_switch import mixture_bucket
from agent.conditioning import condition as condition_audio
from agent.confidence_gate import VERIFIED, confidence_state, is_low_confidence
from agent import asr_refine
from agent.detector_budget import run_within_budget
from agent.distress import DistressEvidence, asks_for_person
from agent.detector_budget import snapshot as detector_budget_snapshot
from agent.disclosure import disclosure_for
from agent.disclosure import version_label as disclosure_version_label
from agent.emergency import detect_emergency
from agent.endpointing import classify_completeness, is_certain, never_shorter_than
from agent.endpointing import decide as decide_turn_end
from agent.enquiry_followup import entities_from_turn, recent_unique, resolve_followup_slot
from agent import cart_tally  # sourav 2026-10-14: running list of priced tests + remove/total
from agent import test_availability  # real call 2026-10-14: "is the X test done here?" is a question, not a booking
from agent.combined_question import add_implied_second_question  # "price AND fasting?" in one breath: answer both
from agent import fastpath_guards  # refuse a quick answer that cannot be right for the sentence
from agent import slot_proposal  # the machine picks the appointment time and says it
from agent import date_phrase  # sourav, 2026-10-24: ISO dates are spoken as 'আঠারো তারিখ' / 'সামনের মাসের দুই তারিখ' (same month / next month ...)
from agent import age_cues  # sourav, 2026-10-24: the patient's age in the caller's words; 13 and above is not sent to the paediatrician
from agent import clinical_safety, lost_found, off_topic, result_request, unrelated_caller  # ported 2026-10-27: a result's meaning goes to a clinician by rule; a wrong number is closed politely
from agent import clinical_terms, doctor_search, lay_terms, echo_check, imaging_words, multi_question, nato_letters, owner_answers  # owner 2026-10-18: lady-doctor filter, neuro/euro, letters said with words, fixed answers
from agent.catalogue_forms import TITLES as _DOCTOR_TITLES
from agent import body_imaging  # 'buker chhobi korbo': which imaging test for that body part, from the catalogue
from agent import dept_list  # 'who are the doctors in cardio'
from agent import greeting_words  # a hello is answered with the introduction; "okay" is not a hello
from agent import serial_lookup, test_days, visit_cues  # sourav, 2026-10-24: 'paid but no serial on WhatsApp' lookup flow; 'is the CT scan done on Thursday?' from the lab table's days column
from agent import silence_policy  # sourav, 2026-10-24: which turn-detector verdicts mean "nobody is speaking" (the hello / cut-the-call clock)
from agent import test_overview  # what tests do you do? from the catalogue, no model
from agent import translit_frames  # "do you do chest X-ray" written in Bengali/Devanagari letters
from agent import spelled_letters  # a spelled test code ("C B C") the Bengali recogniser wrote as a made-up word
from agent.catalogue_forms import lab_test_code
from agent import multi_test_confirm  # garbled test names in a total: read back once, price after a yes
from agent.slot_grounding import ground_slots, has_test_context, is_plausible_test_name  # a test/doctor name the model made up is dropped before any lookup
from agent.transcript_rules import AliasTable  # which catalogue tests/doctors a sentence actually names
from agent import booking_capture  # sourav 2026-10-16: deterministic phone / name / ordinal-slot capture + safe confirm recovery for the doctor booking
from agent.digit_speech import spoken_digit_groups  # owner, 2026-10-26: a phone/confirmation number spoken with a pause between groups of five, not one flat run
from agent import booking_carryover  # sourav 2026-10-14: 'book him / that day' after a doctor-availability answer
from agent.fast_path import Catalogue, FastPath
from agent.fast_path import _normalize as fast_path_normalize
from agent.filler import await_with_filler
from agent.full_duplex import FullDuplexProcessor, wav_to_pcm16k
from agent.golden_buckets import channel_buckets
from agent.history_intent import detect_history_question
from agent import doctor_enquiry_cues, early_routing, weekday_cues, whatsapp_cues
from agent import referral_cues
from agent import report_pickup
from agent.identity import IdentityState
from agent.lang_select import (
    engines_needed,
    english_overrules_language_id,
    fix_asr_confusables,
    language_of_script,
    languages_to_verify,
    pick_candidate,
)
from agent.lang_select import script_share
from agent.lang_select import speakable as _speakable
from agent.language_policy import language_mismatch
from agent.language_switch import detect_language_switch_request
from agent.latency_metrics import turn_latency_by_language
from agent.lay_terms import (
    ABDOMEN,
    BLOOD,
    HEAD,
    URINE,
    expresses_not_knowing_test_name,
    match_category_answer,
    match_lay_term,
)
from agent.lid import (
    SUPPORTED_LANGUAGES,
    ASRLanguageRouter,
    LIDResult,
    RemoteLID,
    SpeechBrainVoxLingua107LID,
)
from agent.llm import ExtractionError, extract_intent
from agent.near_end import NearEndProfile, level_and_pitch
from agent.outcome_metrics import (
    apology_events,
    audio_issue_buckets,
    barge_in_interrupts,
    channel_quality_buckets,
    code_switch_buckets,
    golden_buckets,
    insufficient_information,
    language_mismatches,
    reask_outcomes,
    senior_detections,
)
from agent.pause_profile import PauseProfile
from agent.persona import is_clean as persona_clean
from agent.phrases import HANDOFF_ALL_LANGUAGES, phrase, prewarm_lines
from agent.playback_gate import PlaybackGate
from agent.reask_policy import DEFAULT_MAX_REASKS, ReaskTracker
from agent.reply_templates import (
    add_test_reply,
    blood_test_list_reply,
    booking_confirmation_readback,
    booking_department_suggestion_reply,
    booking_doctor_sitting_hint,
    booking_general_checkup_first_reply,
    booking_reply,
    cancel_reply,
    category_test_list_reply,
    clinic_faq_reply,
    conflict_reply,
    department_route_reply,
    doctor_weekday_yes_reply,
    doctor_availability_reply,
    insufficient_information_reply,
    lookup_reply,
    missing_slot_prompt,
    multi_test_rate_reply,
    multi_test_reply,
    multiple_bookings_reply,
    reschedule_reply,
    resend_reply,
    shown_time,
    spelling_prompt,
    spelling_readback,
    test_prep_reply,
    test_rate_reply,
)
from agent.security_check import SecurityCheck
from agent.semantic_cache import SemanticCache
from agent.semantic_cache import embed as _embed_probe
from agent.senior_care import KindnessPlanner
from agent.senior_voice import SeniorEvidence, explicit_senior_cue, stated_age_is_senior
from agent.senior_voice import estimate as estimate_senior
from agent.speaker_change import SpeakerChangeDetector, voice_embedding
from agent.speech_norm import contains_critical_figure
from agent.speech_policy import derive_policy, effective_rate, limit_questions
from agent.tools_client import ClinicToolsClient, ToolCallError
from agent import tools_client  # noqa: E402  (API_STATS: the clinic-API calls of a turn)
from agent import human_request  # noqa: E402  (a request for a person / staff / a complaint)
from agent import call_later  # noqa: E402  ("I will give the name / call a little later": thanks, call again later, close)
from agent import doctor_origin  # noqa: E402  (is the doctor from Kolkata?)
from agent import name_script  # noqa: E402
from agent.name_script import name_skeleton  # noqa: E402
from agent import dept_talk  # noqa: E402  (talking about a department's doctors: exists / names / good / when / origin / a garbled first name)
from agent.tts import FIGURE_SPEECH_SPEED, TTSClient, UnspeakableTextError
from agent.tts_router import TTSRouter
from agent.turn_ack import AckTracker, thanks_for
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
RESYNC_REWIND_S = 0.8  # 2026-10-17: was 0.25 and measured from the poll tick; see agent/playback_gate.py

# KCD-049: poll granularity. The fixed 0.5 s poll costs half its period on
# average (0.25 s) on EVERY turn. Instead: an ACTIVE cadence while the caller
# has spoken this turn, and -- the part that removes the tax -- a wake-up at
# exactly the moment the detector says its answer could change
# (TurnDecision.commit_after_s). REASONED, not measured under load on the pod:
# each active poll re-runs Silero over the unprocessed tail, so a shorter
# cadence costs CPU. Set ACTIVE_POLL_INTERVAL_S=0.5 to get the old behaviour.
ACTIVE_POLL_INTERVAL_S = float(os.environ.get("ACTIVE_POLL_INTERVAL_S", "0.2"))
MIN_WAKE_S = 0.02
WAKE_SLACK_S = 0.005

# KCD-048: semantic endpointing (agent/endpointing.py). OFF by default: it
# spends one extra speculative ASR call per candidate turn end, and shortens the
# silence needed after a "finished" transcript, which changes how the agent
# feels to talk to. That needs an A/B on the pod before it is anyone's default.
SEMANTIC_ENDPOINTING = os.environ.get("SEMANTIC_ENDPOINTING", "1") == "1"
# Real call 2026-10-14: "ठीक है मेरे को एक चीज बोलिए इधर" and "ঠিক আছে আমাকে বলুন তো এই ক্ব টেস্ট" were cut off at the
# baseline pause and the rest of the sentence was lost. Semantic endpointing (agent/endpointing.py) can wait LONGER when
# the words so far are plainly unfinished. It is now on, but it may only ever add patience: a transcript that reads
# finished no longer shortens the wait below the baseline (that shortcut cut people off), and an unfinished one waits
# ENDPOINT_INCOMPLETE_S. Set SEMANTIC_ENDPOINTING=0 to return to the fixed baseline.
# Owner, 2026-10-18: a caller may pause up to 3.5 seconds in the middle of what they are saying (a number said in groups, a name remembered): that long is waited for
# when the sentence is not finished.
ENDPOINT_INCOMPLETE_S = float(os.environ.get("ENDPOINT_INCOMPLETE_S", "3.5"))
# Owner, 2026-10-25: "for a single call it must be super fast, 0.5 seconds always". The agent used to wait a full 3 s of quiet after EVERY turn before it began to answer. Now: a bare yes/no or a
# whole phone number is answered after ENDPOINT_CERTAIN_S of quiet; a sentence that reads finished after ENDPOINT_COMPLETE_S; anything else after TURN_SILENCE_CONFIRM_S; an unfinished one still waits
# ENDPOINT_INCOMPLETE_S, so a caller who pauses mid-number or mid-sentence is not cut off.
ENDPOINT_CERTAIN_S = float(os.environ.get("ENDPOINT_CERTAIN_S", "0.45"))
ENDPOINT_COMPLETE_S = float(os.environ.get("ENDPOINT_COMPLETE_S", "1.2"))
SPECULATIVE_ASR_TIMEOUT_S = 1.5
MAX_SPECULATIONS_PER_TURN = 3
SPECULATION_MATCH_TOLERANCE_S = 0.25

# Trailing audio never trusted as "confirmed silence". 0.3 s covers the decode
# lag of a growing WebM; the PCM variant reads exact sample positions and
# overrides this default (tools/make_pcm_variant.py).
TURN_TAIL_GUARD_S = float(os.environ.get("TURN_TAIL_GUARD_S", "0.1"))

# owner, 2026-10-24: agent/vad_stream.TurnDetector's own silence_confirm_s (1.0s) was already raised once from
# 0.8s after real testing showed THAT was cutting callers off mid-sentence; live calls since then still cut
# callers off before they had finished a longer, slower sentence. Owner's explicit instruction: at least 3
# seconds. Kept as its own env-tunable (same pattern as TURN_TAIL_GUARD_S above) rather than hardcoded, since
# this is a reasoned, not measured, number -- easy to dial back if 3s makes the agent feel slow to respond.
TURN_SILENCE_CONFIRM_S = float(os.environ.get("TURN_SILENCE_CONFIRM_S", "2.0"))

# KCD-051/052: acoustic echo cancellation + barge-in on the PCM transport. OFF by
# default: the canceller and detector are validated on SYNTHETIC echo only and
# need the pod's real handset audio (tools/echo_eval.py) before anyone relies on
# them. Off, the half-duplex gate behaves exactly as before.
AEC_BARGE_IN = os.environ.get("AEC_BARGE_IN", "0") == "1"
# How far before the detected onset the caller's utterance is taken to start
# (the detector needs ~0.1 s of evidence, so speech began before it fired).
BARGE_IN_PREROLL_S = 0.35

# KCD-513: "always" opens every substantive reply with "Thank you for telling me" (the operator's
# wording, editable from the database); "varied" is the short rotating acknowledgement.
ACK_MODE = os.environ.get("ACK_MODE", "varied")  # owner, 2026-10-26: short rotating acknowledgements ("আচ্ছা", "ঠিক আছে"), not a single repeated "Thank you"
# KCD-495: history is spoken only after the security questions pass. "off" restores the older behaviour
# of offering a person instead of asking (for a deployment that verifies some other way).
SECURITY_QUESTIONS = os.environ.get("SECURITY_QUESTIONS", "on")
# KCD-047: learn each caller's own pause length as the call goes and wait accordingly.
ADAPTIVE_PAUSE = os.environ.get("ADAPTIVE_PAUSE", "on")
# KCD-053/055: attenuate what is far below the caller's own level and not their voice.
NEAR_END_ATTENTION = os.environ.get("NEAR_END_ATTENTION", "on")

# A test the caller named must match the catalogue at least this well before it is used to look up
# their history; below it the agent ASKS which test (KCD-500: never guess). REASONED.
HISTORY_TEST_MATCH_MIN = 0.85

# KCD-055/057: condition the audio the recogniser hears -- noise suppression (only when
# the clip is noisy, and guarded so it cannot eat the speech) and level normalisation
# with a limiter (agent/conditioning.py). "off" (the default), "level" or "full".
# OFF by default: the proxies it is validated on are synthetic, and whether it helps
# the real recognisers is exactly what tools/conditioning_eval.py measures on a pod.
# The RETRY path (a poor first result on a noisy or faint line) always uses it.
CONDITION_INPUT = os.environ.get("CONDITION_INPUT", "off")
# Owner, 2026-10-27: noise suppression is NOT applied to every call (it is validated on synthetic noise only and can hurt a recogniser that coped). It is switched on for ONE call, from the
# next turn, only after that caller has been asked this many times to move somewhere quiet / say it again because the line was noisy (agent/reask_policy.py "reask_noisy").
# Even then it only acts on a clip that is measurably noisy ("auto" mode of agent/conditioning.py). 0 turns the rule off.
NOISE_SUPPRESS_AFTER_ASKS = int(os.environ.get("NOISE_SUPPRESS_AFTER_ASKS", "3"))

# KCD-461: "a natural filler is spoken when a stage exceeds its
# threshold... silence never exceeds a stated maximum." REASONED, not
# measured against real telephony (no pod to time it against) --
# comfortably above a fast_path/cache hit's near-zero latency and a
# warm LLM call's own "~9s across retries" evidence agent/llm.py's own
# KCD-465 fix cites, short enough that a genuinely slow turn does not
# leave the caller wondering whether the line is still open.
# OWNER'S INSTRUCTION, 2026-10-10 (supersedes the 0.7 s instruction below): a holding phrase only when the answer
# is taking MORE THAN 1.0 s, counted from the moment the caller stopped speaking (so recognition time counts); an
# answer that comes sooner has no filler at all. Raised from 0.7 s because a fast-path MISS (anything routed to
# the semantic cache / LLM, which is the only path that can ever reach this filler at all -- a fast-path HIT
# returns before this code runs) was crossing 0.7 s often enough to feel like "every turn", not an occasional
# slow one.
# Owner, 2026-10-24: the spoken "আচ্ছা, বলছি।" holding phrase and the bare "ধন্যবাদ।" after an answer are removed ("remove"). Both come back with FILLER_ENABLED=1 / THANKS_ENABLED=1.
FILLER_ENABLED = os.environ.get("FILLER_ENABLED", "0") == "1"
THANKS_ENABLED = os.environ.get("THANKS_ENABLED", "0") == "1"
FILLER_THRESHOLD_S = float(os.environ.get("FILLER_THRESHOLD_S", "1.0"))  # was 2.5 s of silence, then 0.9 s, then 0.7 s
# ...but the semantic cache is given this long to answer before the model is waited on at all, so a hit is never
# announced by a filler that the answer was about to make pointless.
FILLER_MIN_WAIT_S = float(os.environ.get("FILLER_MIN_WAIT_S", "0.25"))

# Owner, 2026-10-27: "a natural pause before answering" -- a short, once-per-turn beat before the
# FIRST clip of a turn's reply (_speak, below), the way a person briefly takes in what was said
# before responding. Deliberately tiny next to every other wait in this file (FILLER_THRESHOLD_S
# alone is 1.0 s) so it is felt as a beat, not a delay; never applied to a filler utterance, which
# exists specifically to cover a SLOW turn and must not be slowed down further itself.
NATURAL_PAUSE_ENABLED = os.environ.get("NATURAL_PAUSE_ENABLED", "1") == "1"
NATURAL_PAUSE_S = float(os.environ.get("NATURAL_PAUSE_S", "0.15"))

# OWNER'S INSTRUCTION, 2026-09-25: when the caller says nothing for this long after the agent finished speaking, ask
# once whether there is anything else (the call ends if not); if there is still nothing after SILENCE_CLOSE_S, end it.
SILENCE_PROMPT_S = float(os.environ.get("SILENCE_PROMPT_S", "5"))
SILENCE_CLOSE_S = float(os.environ.get("SILENCE_CLOSE_S", "5"))
BOOKING_CLOSE_FACTOR = float(os.environ.get("BOOKING_CLOSE_FACTOR", "4"))  # REASONED: a caller looking for a phone number or an address needs more than 5 s
SILENCE_HELLOS = int(os.environ.get("SILENCE_HELLOS", "3"))  # "hello?" this many times to a caller who has said nothing at all, then the call is cut

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

# agent/asr_refine.py: off | shadow | on. In "shadow" the layer only LOGS what it would do with the second decoder's text; "on" is not built yet and behaves as "shadow" (docs/adr/0004).
ASR_REFINE = asr_refine.parse_mode(os.environ.get("ASR_REFINE", "off"))

CLINIC_API_BASE = os.environ.get("CLINIC_API_BASE", "http://localhost:8080")

# Languages the live path routes between. "bn" alone reproduces the original
# Bengali-only behaviour (no LID, no Hindi ASR loaded) -- the rollback lever.
# Bengali is always active: it is the clinic's primary language and the
# greeting language.
ACTIVE_LANGUAGES = tuple(
    dict.fromkeys(
        ["bn"]
        + [
            x
            for x in os.environ.get("VOICE_AGENT_LANGUAGES", "bn,hi,en").replace(" ", "").split(",")
            if x in SUPPORTED_LANGUAGES
        ]
    )
)
ENGLISH_ASR_URL = os.environ.get("ENGLISH_ASR_URL", "http://localhost:8003")
# Bengali / Hindi recognition as its own service with several worker processes (indic_asr_server.py); empty = load the two models inside this process as before.
INDIC_ASR_URL = os.environ.get("INDIC_ASR_URL", "")
TTS_HEALTH_URL = os.environ.get("TTS_HEALTH_URL", "http://localhost:8002/health")
OLLAMA_HEALTH_URL = os.environ.get("OLLAMA_HEALTH_URL", "http://localhost:11434/api/tags")

# Admission control (agent/admission.py). Cap is per PROCESS: with the WebM
# and PCM services both running, each has its own counter.
ADMISSION_MAX_CALLS = int(os.environ.get("ADMISSION_MAX_CALLS", "36"))
ADMISSION_SHED_P95_S = float(os.environ["ADMISSION_SHED_P95_S"]) if os.environ.get("ADMISSION_SHED_P95_S") else None
ADMISSION_BYPASS_FILE = os.environ.get("ADMISSION_BYPASS_FILE", "/workspace/.ai_bypass")
ADMISSION_ADMIN_TOKEN = os.environ.get("ADMISSION_ADMIN_TOKEN", "")

app = FastAPI()

# KCD-467: "time from process start to ready is measured and bounded."
# Captured at import time -- as close to "process start" as this module
# can observe -- so _startup()'s own duration can be reported once every
# model has answered ready, rather than assumed. See /api/health.
_PROCESS_STARTED_AT = time.monotonic()
_READY_AT: float | None = None  # set once, at the end of _startup()

# ---- process-wide singletons: loaded once, shared by every call ----
_asr: TurnASR | None = None
_turn_detector: TurnDetector | None = None
_tools: ClinicToolsClient | None = None
_tts: TTSClient | None = None
_intent_cache: SemanticCache | None = None
_fast_path: FastPath | None = None
_answer_warm_task = None
_availability_task = None
_asr_router: ASRRouter | None = None
_lid: SpeechBrainVoxLingua107LID | None = None
_tts_router: TTSRouter | None = None
_admission: AdmissionController | None = None
_health: HealthMonitor | None = None
_health_task: asyncio.Task | None = None
_ingress_task: asyncio.Task | None = None
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
                mono = torchaudio.functional.resample(__import__("torch").from_numpy(data.mean(axis=1)), sr, 16000)
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


async def _fetch_catalogue() -> dict:
    """GET /api/v1/catalogue with the service token; raises on any failure or on a body that is not a catalogue."""
    import httpx as _hx

    token = os.environ.get("CLINIC_API_TOKEN", "")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    async with _hx.AsyncClient(timeout=10) as c:
        resp = await c.get(f"{CLINIC_API_BASE}/api/v1/catalogue", headers=headers)
    resp.raise_for_status()
    payload = resp.json()
    if not isinstance(payload, dict) or "tests" not in payload:
        raise ValueError("clinic-api /api/v1/catalogue did not return a catalogue")
    return payload


# The lay-term fast path (agent/lay_terms.py, below) is a deterministic, rare gate -- not every turn -- but still
# reads the catalogue's sample_type, which the FastPath.Catalogue index does not keep. Cached separately, refetched
# at most every CATALOGUE_CACHE_TTL_S, rather than hitting the network on every matching turn.
CATALOGUE_CACHE_TTL_S = 300.0
_catalogue_cache: dict = {"payload": None, "at": 0.0}


async def _cached_catalogue() -> dict:
    now = time.monotonic()
    if _catalogue_cache["payload"] is None or now - _catalogue_cache["at"] > CATALOGUE_CACHE_TTL_S:
        _catalogue_cache["payload"] = await _fetch_catalogue()
        _catalogue_cache["at"] = now
    return _catalogue_cache["payload"]


# ---- the answers every caller asks for, rendered ahead of time -------------------------------------------------
ANSWER_WARM_INTERVAL_S = float(os.environ.get("ANSWER_WARM_INTERVAL_S", "600"))
_answers_warm: dict = {"rounds": 0, "pinned_clips": 0, "last_round_s": None, "last_error": None}


AVAILABILITY_PRELOAD_INTERVAL_S = float(os.environ.get("AVAILABILITY_PRELOAD_INTERVAL_S", "60"))


async def _availability_preload_loop() -> None:
    """Owner, 2026-10-29: no database hit in a turn for a doctor's availability today or tomorrow. Every AVAILABILITY_PRELOAD_INTERVAL_S seconds (less than the entries' life) every doctor of the catalogue is asked about
    for today, tomorrow and "next sitting" and kept (agent/tools_client.preload_availability); the catalogue itself is re-read, so a new doctor is covered within a minute. Never lets a failure stop the loop."""
    await asyncio.sleep(5)
    while True:
        try:
            if _tools is not None:
                cat = await _cached_catalogue()
                shorts, fulls = [], []
                for d in cat.get("doctors") or []:
                    if d.get("name"):
                        shorts.append(d["name"])
                    if d.get("full_name"):
                        fulls.append(d["full_name"])
                today = slot_proposal.clinic_now().date()
                two_days = [today.isoformat(), (today + datetime.timedelta(days=1)).isoformat()]
                # Owner, 2026-10-29: the first version asked ~550 lookups every 20 s (27 a second: the clinic API's whole capacity) and starved the callers' own lookups. Now ONE request a minute
                # (every doctor, short and whole name, today / tomorrow / next sitting); the single lookups below are the fallback when the clinic API has no bulk answer.
                try:
                    await _tools.preload_availability_bulk(two_days + [None])
                except ToolCallError as e:
                    logger.warning("bulk availability preload failed (%s): falling back to single lookups", e)
                    await _tools.preload_availability(fulls, two_days + [None])
                    await _tools.preload_availability(shorts, two_days)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 - advisory: a turn simply asks the clinic API
            logger.warning("availability preload skipped: %s", e)
        await asyncio.sleep(AVAILABILITY_PRELOAD_INTERVAL_S)


async def _warm_answers_once() -> int:
    """Render, and pin in the TTS cache, the spoken audio of every test price, every test-preparation answer and every
    clinic FAQ answer, in every active language -- exactly the clauses a live turn would synthesise, so the next caller
    who asks gets them from the cache with no synthesis. The TEXT is still built from live clinic data every time
    (the truth boundary is unchanged: nothing here stores a price as a fact); only the rendered AUDIO is kept, keyed
    by the exact text, so a changed price is simply a different clip. Each round re-reads the catalogue, renders only
    clips it does not already hold, and lets clips for answers that no longer exist become evictable."""
    if _tools is None or _tts is None:
        return 0
    t0 = time.monotonic()
    cat = await _fetch_catalogue()
    keep: set[str] = set()
    rate = derive_policy("neutral")
    for lang in _languages_active:
        replies: list[str] = []
        for row in cat.get("tests", []):
            name = row["name"]
            try:
                replies.append(test_rate_reply({"test_name": name}, await _tools.get_test_rate(name), lang))
                replies.append(test_prep_reply({"test_name": name}, await _tools.get_test_prep(name, lang), lang))
            except ToolCallError as e:
                logger.warning("answer warm-up: lookup for %r failed (%s)", name, e)
        for f in cat.get("faq_topics", []):
            try:
                replies.append(
                    clinic_faq_reply({"faq_topic": f["topic"]}, await _tools.get_faq(f["topic"], lang), lang)
                )
            except ToolCallError as e:
                logger.warning("answer warm-up: FAQ %r failed (%s)", f["topic"], e)
        for reply in replies:
            for clause in split_into_clauses(reply) or [reply]:  # as _finish_enquiry_turn does
                speed = effective_rate(rate, FIGURE_SPEECH_SPEED if contains_critical_figure(clause) else None, lang=lang)
                try:
                    await _tts.synthesize(clause, lang, speed=speed, pin=keep)
                except Exception as e:  # noqa: BLE001 - an unspeakable or failed clip is skipped, never fatal
                    logger.debug("answer warm-up: skipped %r (%s)", clause[:30], e)
                await asyncio.sleep(0)  # never hog the loop while calls are live
    n = _tts.retain_pinned(keep)
    _answers_warm.update(
        rounds=_answers_warm["rounds"] + 1,
        pinned_clips=n,
        last_round_s=round(time.monotonic() - t0, 1),
        last_error=None,
    )
    logger.info("answer warm-up: %d clips pinned in %.1fs", n, time.monotonic() - t0)
    return n


async def _answer_warm_loop() -> None:
    while True:
        try:
            await _warm_answers_once()
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 - advisory: a failure only means the next caller pays a synthesis
            _answers_warm["last_error"] = str(e)
            logger.warning("answer warm-up failed: %s", e)
        await asyncio.sleep(ANSWER_WARM_INTERVAL_S)


async def _load_fast_path() -> FastPath | None:
    """Fetch the catalogue and build the fast path, or None if clinic-api is
    not reachable. Optional by design: without it every turn goes to the LLM,
    which is the behaviour that existed before this path did."""
    global _fast_path_stamp
    try:
        payload = await _fetch_catalogue()
        fp = FastPath(Catalogue(payload))
        _fast_path_stamp = json.dumps([payload.get("doctors"), payload.get("tests")], sort_keys=True, default=str)  # what the quick path was built from (see _maybe_reload_fast_path)
        logger.info("fast path ready over %d catalogue rows", len(fp.catalogue))
        return fp
    except Exception as e:  # noqa: BLE001 - degrade to LLM-only, never fail startup
        logger.warning("catalogue unavailable, fast path disabled: %s", e)
        return None


_fast_path_stamp: str | None = None


async def _maybe_reload_fast_path() -> None:
    """If the fast path failed to load at boot (clinic-api was not up yet --
    it starts AFTER this service in start_all.sh), retry at most every 30 s.
    Without this a boot-order race silently costs the fast path until the
    next restart: every turn pays an LLM call and nothing says why."""
    global _fast_path, _fast_path_retry_at, _fast_path_stamp
    if _fast_path is not None:
        # Owner, 2026-10-29: a doctor added after the agents started ("Dr. Ziaul Haq") was invisible to the quick path until the next restart. Every 60 seconds the clinic's list is read again
        # (cheap: it is cached for a few seconds) and the quick path is rebuilt when the doctors or tests changed.
        if time.monotonic() >= _fast_path_retry_at:
            _fast_path_retry_at = time.monotonic() + 60.0
            try:
                payload = await _cached_catalogue()
                stamp = json.dumps([payload.get("doctors"), payload.get("tests")], sort_keys=True, default=str)
                if _fast_path_stamp is not None and stamp != _fast_path_stamp:
                    fp = await asyncio.to_thread(lambda: FastPath(Catalogue(payload)))
                    _fast_path = fp
                    logger.info("fast path rebuilt: the clinic's doctors or tests changed")
                _fast_path_stamp = stamp
            except Exception as e:  # noqa: BLE001 - a refresh that fails changes nothing
                logger.warning("fast path refresh skipped: %s", e)
        return
    if time.monotonic() < _fast_path_retry_at:
        return
    _fast_path_retry_at = time.monotonic() + 30.0
    _fast_path = await _load_fast_path()


# How long the startup warm-up may take to get the intent model into VRAM. NOT the per-turn deadline
# (agent.llm.DEFAULT_DEADLINE_S, 12 s, KCD-465): a cold 7B load measured 47-74 s on this pod, so a warm-up under
# the turn deadline times out, the client disconnects, and Ollama abandons the load -- the model never becomes
# resident and the FIRST CALLER pays the cold start against a 12 s deadline (seen on the live pod: "intent model
# warmup failed ... 12.0s deadline" in both entrypoints' logs). REASONED headroom above the measured 74 s.
INTENT_WARMUP_DEADLINE_S = float(os.environ.get("INTENT_WARMUP_DEADLINE_S", "300"))


async def _warm_intent_model() -> bool:
    """Load the intent model now, with a deadline that fits a cold load. Advisory: a failure is logged and
    startup continues (calls then pay the cold start, which is the behaviour without a warm-up)."""
    try:
        _, diag = await asyncio.to_thread(extract_intent, "নমস্কার", 2, "bn", INTENT_WARMUP_DEADLINE_S)
        logger.info("intent model warm (%.1fs)", diag["total_time_s"])
        return True
    except Exception as e:  # noqa: BLE001 - warmup is advisory
        logger.warning("intent model warmup failed: %s", e)
        return False


@app.on_event("startup")
async def _startup():
    global _asr, _turn_detector, _tools, _tts, _intent_cache, _fast_path
    global _asr_router, _lid, _tts_router, _admission, _health, _health_task, _languages_active
    import httpx as _httpx

    # httpcore asks "is trio / sniffio installed?" on EVERY request; when a module is missing Python rescans sys.path (stat calls on the network volume) each time -- ~0.3 of the loop's time under
    # load (profile 2026-10-25). A None entry in sys.modules makes the question answer "no" at once.
    for _mod in ("trio", "sniffio"):
        try:
            __import__(_mod)
        except ImportError:
            sys.modules[_mod] = None
    with contextlib.suppress(Exception):  # the CPU threads one process may use for tensor maths: the default is every core, which makes the processes on this machine fight
        import torch as _torch

        _torch.set_num_threads(int(os.environ.get("AGENT_TORCH_THREADS", "6")))
    active = list(ACTIVE_LANGUAGES)
    engines: dict = {}
    if INDIC_ASR_URL:
        logger.info("Bengali / Hindi recognition: the indic ASR service at %s", INDIC_ASR_URL)
        engines["bn"] = HTTPASREngine(INDIC_ASR_URL, timeout_s=40.0, lang="bn")
        _asr = engines["bn"]
        if "hi" in active:
            engines["hi"] = HTTPASREngine(INDIC_ASR_URL, timeout_s=40.0, lang="hi")
    else:
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
            if INDIC_ASR_URL:  # identified by the service's several worker processes, side by side (agent/lid.RemoteLID)
                lid = RemoteLID(INDIC_ASR_URL)
            else:
                lid = SpeechBrainVoxLingua107LID(device=os.environ.get("LID_DEVICE", "cuda"))
            await asyncio.to_thread(lid.load)
            _lid = lid
        except Exception as e:  # noqa: BLE001 - degrade LOUDLY to Bengali-only, never guess a language
            logger.error("language ID unavailable (%s) -- serving Bengali ONLY until fixed", e)
            active = ["bn"]
    _languages_active = tuple(active)
    logger.info("active languages: %s", ",".join(_languages_active))

    logger.info("loading Silero VAD...")
    _turn_detector = await asyncio.to_thread(
        TurnDetector, silence_confirm_s=TURN_SILENCE_CONFIRM_S, tail_guard_s=TURN_TAIL_GUARD_S
    )
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
    if INDIC_ASR_URL:
        probes["indic-asr"] = http_probe(probe_client, f"{INDIC_ASR_URL.rstrip('/')}/health")
    _health = HealthMonitor(_admission, probes)
    _health.probe_client = probe_client
    _health_task = asyncio.create_task(_health.run_forever())
    global _ingress_task
    _ingress_task = asyncio.create_task(_ingress_watch())

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
    await _warm_intent_model()

    # Load the 74-row catalogue once so the fast path can identify a test
    # or doctor locally. Optional: if the clinic API is not up yet, every
    # turn simply goes to the LLM, which is the behaviour that existed
    # before this path did.
    _fast_path = await _load_fast_path()

    logger.info("prewarming TTS...")
    lines = prewarm_lines()
    await _tts.prewarm({lang: lines[lang] + [thanks_for(lang)] for lang in _languages_active})
    await _warm_speech_models()

    global _READY_AT, _answer_warm_task, _availability_task
    _READY_AT = time.monotonic()
    _answer_warm_task = asyncio.create_task(_answer_warm_loop())  # background: startup does not wait for it
    _availability_task = asyncio.create_task(_availability_preload_loop())  # background: every doctor's availability for today, tomorrow and "next sitting", always in memory
    logger.info("startup complete -- ready for calls (%.1fs from process start)", _READY_AT - _PROCESS_STARTED_AT)


@app.on_event("shutdown")
async def _shutdown():
    if _availability_task is not None:
        _availability_task.cancel()
    if _answer_warm_task is not None:
        _answer_warm_task.cancel()
    if _ingress_task:
        _ingress_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await _ingress_task
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
        "ingress": _ingress_snapshot(),
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
        "answer_warmup": _answers_warm,
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
        "golden_buckets": golden_buckets.snapshot(),
        "apology_events": apology_events.snapshot(),
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
    level_dbfs, f0_hz = level_and_pitch(samples, sr)
    return {
        "audio": assess_audio(samples, sr),
        "senior": estimate_senior(samples, sr),
        "near_end": (level_dbfs, f0_hz),
        "voice": voice_embedding(samples, sr),
    }


def _enhance_wav_to_path(path: str) -> str | None:
    """A noise-filtered, level-corrected copy of the clip for a retry, or
    None if it cannot be produced."""
    try:
        samples, sr = _read_wav_mono(path)
        out, _report = condition_audio(samples, sr, suppress="always")
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


def _attend_wav_to_path(path: str, level_dbfs: float, f0_hz: float) -> str | None:
    """A copy of the clip with the room attenuated (agent/attention.py), or None if nothing was
    changed or it could not be produced. Never raises."""
    try:
        samples, sr = _read_wav_mono(path)
        out, report = attend(samples, level_dbfs, f0_hz, sr)
        if not report.applied:
            return None
        cleaned = path + ".att.wav"
        with contextlib.closing(wave.open(cleaned, "wb")) as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes((np.clip(out, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes())
        golden_buckets.record("attention", "applied", "any")
        return cleaned
    except Exception as e:  # noqa: BLE001 - attention is an optional cleanup, never a failure
        logger.warning("attention failed: %s", e)
        return None


def _condition_wav_to_path(path: str, mode: str) -> str | None:
    """A conditioned COPY of the clip for the recogniser (KCD-055/057), or None. The
    original is left alone: audio analysis, senior and near-end judgements are made on
    what the caller actually sounded like, not on what was done to help the ASR."""
    try:
        samples, sr = _read_wav_mono(path)
        out, _report = condition_audio(samples, sr, suppress="never" if mode == "level" else "auto")
        conditioned = path + ".cond.wav"
        with contextlib.closing(wave.open(conditioned, "wb")) as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes((np.clip(out, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes())
        return conditioned
    except Exception as e:  # noqa: BLE001 - conditioning is an enhancement, never a failure
        logger.warning("input conditioning failed: %s", e)
        return None


class CallSession:
    """One PCM buffer for the ENTIRE call, and a marker for how much of it
    has been consumed.

    The continuous-buffer design is inherited from the WebM version, where
    it was forced: MediaRecorder puts the container header only in the
    first chunk, so resetting the buffer mid-call produced audio that
    could never be decoded again. That constraint is GONE here -- raw PCM
    has no header and any byte range is independently valid.

    It is kept anyway, because the second reason for it was always the
    better one: `processed_until_s` gives every turn an absolute,
    monotonic position on one call-long timeline. Silero's segment
    boundaries move as more audio arrives, so a turn detector run against
    a buffer that keeps restarting drops any utterance straddling the
    seam. Each poll therefore looks only at the UNPROCESSED TAIL.

    What PCM changes is the cost and the accuracy of that lookup: the tail
    is a slice rather than a full re-decode, and sample index maps to
    wall-clock exactly, so the timeline cannot drift.
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
        # Ingress (blueprint 2026-10-29, section 4): the socket reader only counts and queues; a separate consumer feeds the audio path. Counters make a backlog visible.
        self.audio_q: asyncio.Queue | None = None
        self.rx_bytes = 0
        self.consumed_bytes = 0
        self.audio_overruns = 0
        self.audio = PcmCallBuffer()
        self.declared_rate: int | None = None
        if AEC_BARGE_IN:
            # KCD-051/052: echo cancellation + acoustic barge-in. Raw PCM only.
            self.duplex = FullDuplexProcessor.for_phone_line(SAMPLE_RATE) if getattr(ws, 'phone_line', False) else FullDuplexProcessor(sr=SAMPLE_RATE)

        # Starts True: the greeting goes out before the caller has said
        # anything, so the gate must already be closed when the first poll
        # tick runs, not opened a moment later by _speak().
        self.gate = PlaybackGate(PLAYBACK_GUARD_S)

        # KCD-049/048: when the poll loop next wakes, and the speculative
        # transcript state for the candidate turn end currently in view.
        self.next_wake_s = POLL_INTERVAL_S
        self.spec_end_abs: float | None = None
        self.spec_verdict: str | None = None
        self.spec_count = 0

        # KCD-052/163: an interrupt bumps speak_epoch; the turn in flight keeps
        # the epoch it started under, so everything it has yet to say is
        # discarded rather than played over the caller.
        self.speak_epoch = 0
        self.turn_epoch = 0
        # KCD-051: set by the PCM variant when AEC_BARGE_IN is on.
        # (the PCM variant assigns it earlier in __init__ when AEC_BARGE_IN is on)
        self.duplex: FullDuplexProcessor | None = getattr(self, "duplex", None)

        # `lang` is the call's REPLY language -- sticky by design (KCD-lay-terms, a live call 2026-09-28): a caller
        # who started in Bengali and later says one English or Hindi word/name is not asking to be answered in a
        # different language, and re-flipping mid-call, re-playing the disclosure notice each time, was
        # disorienting on a real call. Recognition (which ASR engine transcribes THIS turn) still follows
        # per-utterance LID every turn, via ASRLanguageRouter/_route_and_transcribe -- only the REPLY language is
        # locked. It changes only for: the first turn (nothing is locked yet), an explicit request ("speak in
        # Hindi" -- agent/language_switch.py), or a detected different voice after identity verification
        # (agent/speaker_change.py revokes verification on a voice change; see _check_speaker_change) -- see
        # _resolve_reply_language below, which is the one place this decision is made.
        self.lang = "bn"
        self.lang_locked = False
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
        # sourav, 2026-10-14: every test a price was FOUND for this call, in the order asked (names as the caller said
        # them). last_enquiry_entities above holds ONE test, so "remove that and tell the total" had nothing to work on.
        # Only names live here -- prices are always re-asked of the clinic API (agent/cart_tally.py).
        self.priced_tests: list[str] = []
        # sourav, 2026-10-14: the doctor (and date) the last FOUND doctor-availability answer was about, so "book him / that day" has
        # something to point at (agent/booking_carryover.py). {"doctor_name", "date", "turn"} or None.
        self.last_doctor_enquiry: dict | None = None

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
        self.dept_pending = None  # a question asked by agent/doctor_search.py and waiting for its answer
        self.last_spoken = ""  # the last thing said to the caller (not the holding filler)
        self.last_route = None  # the department just suggested for a symptom (so a push-back is answered with the same suggestion)
        self.pending_prescription = None  # (test, turn): "do you have the prescription?" was just asked
        self.location_pending = None  # turn number: "tell me the WhatsApp number and we send the location" was just said
        self.caller_numbers: list[str] = []  # numbers the caller has SAID in this call, newest last (used to call back an unresolved call)
        self.unclear_streak = 0  # consecutive turns the agent could not answer
        self.senior_evidence = SeniorEvidence()
        self.acknowledged_state: str | None = None
        self.distress = DistressEvidence()
        self.person_requests = 0
        self.senior_lookup_done = False

        # KCD-513/514: acknowledgements are suppressed on repeat turns; KCD-155's distress
        # acknowledgement, if spoken this turn, already opened it.
        self.acks = AckTracker(mode=ACK_MODE)
        self.turn_acknowledged = False
        # Owner, 2026-10-27: "a natural pause before answering" -- a short, once-per-turn beat
        # before the FIRST clip of a turn's reply, the same place/timing this file already tracks
        # turn_acknowledged at. Never for every _speak() call in a turn (a turn often speaks more
        # than once -- an ack, a note, the answer), only the first.
        self.turn_spoken = False
        # The agent's last utterance asked something (it contained a question mark), and whether THIS turn is the
        # caller's answer to it: only an answer to our question is thanked ("Thank you."), never a question of theirs.
        self.awaiting_answer = False
        self.answering = False
        # Silence handling (SILENCE_PROMPT_S): when the quiet started, how many times we have asked, and whether the
        # caller's next words answer "anything else?".
        self.silence_since: float | None = None
        self.silence_prompts = 0
        self.silence_hellos = 0  # how many times a caller who has not spoken yet has been called "hello"
        self.awaiting_close_answer = False
        self.abuse_count = 0  # abusive turns so far this call (agent/abuse.py)
        # The implicit happiness score (agent/call_score.py): counters updated as the call goes, scored once at hang-up.
        self.signals = CallSignals()
        self.reply_noted = False  # this turn's time-to-first-audio has been counted
        self.speaking_filler = False  # the "achha bolchi" filler is being spoken: not the answer
        self.lang_history: list[str] = []  # the language of each accepted turn, to see a flip that is flipped back
        # KCD-353: the disclosure was spoken in the greeting (Bengali); once more, in the
        # caller's own language, the first time it differs.
        self.disclosed_langs: set[str] = {"bn"}
        # OWNER'S INSTRUCTION, 2026-10-10: the reply language is locked by design (see `lang` above) and only
        # moves on an explicit "speak in Hindi" request -- a caller who just keeps answering in a different
        # language than the call is locked to, without ever asking, stays stuck in the wrong one. After two
        # turns IN A ROW in the same other language (session.lang_history, already tracked for the flip-detection
        # signal below), ask ONCE whether to switch, in the language the caller has actually been using -- not
        # before: a single stray word in another language (a name, a number) must not trigger this.
        # `lang_switch_last_offered` makes the ask fire at most once per candidate language per call: a caller
        # who says "no" (or ignores it and keeps talking) is not asked again for that same language, so this
        # never nags. Confirming a switch does NOT replay the disclosure in the new language (see the offer's
        # own handler) -- the owner's instruction is explicit that the opening introduction must not repeat.
        self.awaiting_language_switch_offer = False
        self.pending_language_switch: str | None = None
        # OWNER'S INSTRUCTION, 2026-10-10: the turn that TRIGGERED the offer is not answered when the offer
        # fires (it is replaced by the question "should I switch?", see the trigger next to lang_history) --
        # stored here so that confirming the switch can answer it in the NEW language instead of leaving the
        # caller to repeat themselves. A live call (2026-09-25-ish testing): "ये रिपोर्ट कैसे मिलेगा..." triggered
        # the offer, "हाँ" confirmed it, and the caller had to ask the report question all over again before it
        # was ever answered.
        self.pending_language_switch_text: str | None = None
        # The SAME turn's recognition result, not re-derived -- _answer_content_turn's confidence/action-gate
        # check (CLAUDE.md's truth boundary: withhold a factual answer the recogniser's own decoders disagreed
        # on) reads decoder_agreement off this. The question is being answered a turn later than it was asked,
        # but how confidently it was HEARD does not change because of that -- it must not be re-guessed.
        self.pending_language_switch_asr_result = None
        self.lang_switch_last_offered: str | None = None
        # KCD-053: who this call's near-end speaker is (level, pitch), to tell them from a
        # bystander. KCD-054/495: what we actually know about who is on the call, and the
        # voice reference that lets a change of speaker revoke a verification.
        self.near_end = NearEndProfile()
        self.identity = IdentityState()
        self.speaker = SpeakerChangeDetector()

        # KCD-501: the call's record, written as the call goes and closed at hang-up.
        self.recorder: CallRecorder | None = None
        self.outcome = "completed"
        self.turn_count = 0
        # sourav, 2026-10-24: the 'I paid but never got my serial number' conversation (agent/serial_lookup.new_flow), None when not under way
        self.serial_flow: dict | None = None
        # sourav, 2026-10-24: 'I want to show my report' asked 'was the visit seven days ago or earlier?' -- the turn it was asked on (None = not waiting for the answer); and the
        # 'which doctor?' asked for a 'what will my serial be if I book today' question: (turn, "today" | "tomorrow")
        self.report_visit_asked: dict | None = None
        self.pending_serial: tuple[int, str] | None = None
        # KCD-494/495/498: a question about the caller's own history is a small state machine --
        # find the record (an OPEN question when the number is shared), then, only if a
        # verification has passed, answer from retrieved fields.
        self.history_state: str | None = None  # None | need_phone | need_name | need_test
        self.pending_history = None  # the HistoryQuestion being answered
        self.history_attempts = 0
        self.awaiting_handoff_offer = False
        # A factual lookup held back until the caller confirms what the recogniser (with nothing to
        # check it against) heard: agent/entity_confirmation.py.
        self.pending_entity: entity_text.PendingEntity | None = None
        # KCD-495/497: the security questions in progress, and the senior-care planner (KCD-512).
        self.security: SecurityCheck | None = None
        self.kindness = KindnessPlanner()
        # KCD-496: an unfinished booking from an earlier call, offered after verification.
        self.pending_draft: dict | None = None
        self.awaiting_draft_answer = False
        # KCD-104: a booking set aside while the caller did a different task, and whether we are waiting for their
        # yes or no to "shall I go back to it".
        self.suspended = None
        self.awaiting_resume = False
        self.pending_enquiry = None  # (intent, turn) when we just asked "which test / which doctor"
        # KCD-lead-1: the turn we asked "patient's name, number, which doctor/department" after an inter-
        # center referral call (agent/referral_cues.py); the ANSWER is read on the next turn regardless of
        # what intent the model guesses for it, same discipline as pending_enquiry above.
        self.pending_referral: int | None = None
        self.pending_weekdays: int | None = None  # the turn we asked "which doctor?" for a which-days question (agent/doctor_enquiry_cues.py)
        self.pending_hours: int | None = None  # the turn we asked "which doctor?" for a from-and-to-hours question
        self.pending_day: tuple[int, str] | None = None  # (turn, ISO date) we asked "which doctor?" for "robibar ki doc x bosbe" (agent/weekday_cues.py)
        self.pending_when: int | None = None  # the turn we asked "which doctor?" for a when-to-come question
        self.referral_notes: str = ""
        # OWNER'S INSTRUCTION, 2026-10-12: set when the caller said they do not know the test's name at all and
        # was just asked "is it blood, urine, or imaging?" -- the NEXT turn is read as the answer to that
        # question (agent.lay_terms.match_category_answer), not as a fresh question; see the dispatch site.
        self.awaiting_test_category = False
        self.awaiting_clinician_offer = False  # "a doctor will explain it, shall I connect one?" was asked (agent/clinical_safety.py); the next yes / no answers it
        # THREE LANGUAGES, kept apart (review 2026-10-28). `lang` (below) is the PREFERRED REPLY language: it moves only on the first decisive turn, an accepted explicit request, or the
        # two guarded early-call / weak-lock corrections. `detected_input_language` is what the caller spoke THIS turn (set every turn, never a preference). `tts_output_language` is the
        # language of the last line actually synthesised. A detected difference alone never moves the preference; the mismatch is counted so it can be reviewed.
        self.detected_input_language: str | None = None
        self.tts_output_language: str | None = None
        self.language_mismatch_turns = 0
        self.lang_lock_weak = False  # the language was locked from a doubtful first read (decoders disagreeing): the first clear sentence in the other script may correct it
        self.last_enquiry = None  # {"text", "turn"}: the last read-only enquiry answered, replayed in a language the caller then asks for ("বাংলায় বলুন এটা")
        self.lang_explicit = False  # the caller ASKED for a language ("speak English"): a later clear sentence in another script never moves the call on its own
        self.noisy_asks = 0  # how many times this caller was asked, because of noise on the line, to find a quiet place / repeat
        self.noise_suppress = False  # set after NOISE_SUPPRESS_AFTER_ASKS such asks: this call's audio is then noise-filtered (when noisy) before recognition
        self.offtopic_streak = 0  # consecutive turns about nothing to do with Sonoscan (agent/off_topic.py); the third closes the call
        self.offtopic_last_turn = -10  # the turn number of the last such turn, to tell "in a row" from "once, long ago"
        self.unrelated_streak = 0  # consecutive turns that were a wrong number (agent/unrelated_caller.py); the second closes the call
        self.awaiting_blood_test: int | None = None  # the turn "which blood test?" was asked (owner, 2026-10-26); the next sentence is its answer
        self.last_enquiry_turn = 0  # turn_count when the last test/doctor answer was given (agent/enquiry_followup.py)
        self.no_at_confirm = False  # the caller said "no, ..." at the confirmation step and it was not a bare no
        # KCD-499: stored preferences, offered after the answer and applied only if the caller says yes.
        self.pending_pref = None
        self.awaiting_pref_answer = False
        self.confirmed_preferences: dict = {}
        # KCD-047: this caller's own pause rhythm, and the evidence for "that cut was too early".
        self.pause = PauseProfile()
        self.last_commit_end_abs: float | None = None
        self.last_confirm_s = 0.0
        self.agent_spoke_since_commit = False
        self.commit_checked = True

    @property
    def policy(self):
        """Appendix C's delivery parameters for the caller's CURRENT state
        (agent/speech_policy.py) -- derived, never stored, so it can never
        disagree with call_state."""
        return derive_policy(self.call_state.caller_state, self.call_state.senior)

    # The gate itself lives in agent/playback_gate.py (KCD-050) so its rules can be
    # tested without a pod; these keep the names the rest of this file uses.
    @property
    def agent_speaking(self) -> bool:
        return self.gate.speaking

    @property
    def speak_deadline(self) -> float:
        return self.gate.deadline

    @property
    def resync_pending(self) -> bool:
        return self.gate.resync_pending

    @resync_pending.setter
    def resync_pending(self, value: bool) -> None:
        self.gate.resync_pending = value

    def hold_gate_for(self, audio_duration_s: float):
        """Called before each reply goes out. Extends rather than replaces
        the deadline: replies queue on the client, so a second clip starts
        playing only after the first finishes."""
        self.gate.hold(audio_duration_s)

    def release_gate(self):
        """Playback is over. Don't touch processed_until_s here -- the poll
        loop owns the decoded buffer and does the resync on its next tick."""
        self.gate.release()

    def register_agent_audio(self, wav_bytes: bytes, voice: str | None = None) -> None:
        """KCD-051: hand the clip about to be sent to the echo canceller as its
        far-end reference. A no-op unless the PCM variant enabled AEC."""
        if self.duplex is None or not wav_bytes:
            return
        try:
            self.duplex.place_reference(wav_to_pcm16k(wav_bytes), voice=voice)
        except Exception as e:  # noqa: BLE001 - AEC is an enhancement, never a reason to drop a reply
            logger.warning("[%s] could not register agent audio for echo cancellation: %s", self.call_id, e)

    def take_wake_delay(self) -> float:
        """Seconds to sleep before the next poll; the schedule resets to the
        idle cadence unless the last decision said otherwise."""
        delay, self.next_wake_s = self.next_wake_s, POLL_INTERVAL_S
        return delay

    def reset_speculation(self) -> None:
        self.spec_end_abs, self.spec_verdict, self.spec_count = None, None, 0

    async def append(self, chunk: bytes):
        self.last_activity = time.time()
        if self.duplex is None:
            self.audio.append(chunk)
            return
        # The buffer the turn detector reads holds the ECHO-CANCELLED signal,
        # so the agent's own voice is not what it hears (agent/full_duplex.py).
        x = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
        # the echo-cancelling maths (FFTs) runs on a worker thread, not on the event loop that serves every call (load profile 2026-10-25)
        result = await asyncio.to_thread(self.duplex.process, x)
        if result.cleaned.size:
            self.audio.append((np.clip(result.cleaned, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes())
        if result.barge_in is not None:
            await _handle_barge_in(self, result.barge_in)

    async def send_json(self, sender: str, text: str):
        await self.ws.send_text(json.dumps({"sender": sender, "text": text}, ensure_ascii=False))

    async def send_audio(self, wav_bytes: bytes):
        if wav_bytes:
            await self.ws.send_bytes(wav_bytes)

    def cleanup(self):
        import shutil

        with contextlib.suppress(OSError):
            shutil.rmtree(self.tmpdir, ignore_errors=True)
        with contextlib.suppress(Exception):
            _turn_detector.release(self.call_id)  # the call's streaming voice-activity state


# Owner, 2026-10-27: "lifted for questions, softer for apologies" -- light per-clause pitch/
# energy hints sent to tts_server.py's own (unverified -- see that file's SynthesizeRequest.
# pitch_scale/energy_scale note) hasattr-guarded attributes. Gentle multipliers either way, and
# env-overridable without a redeploy, same convention as every other tuned constant in this file.
QUESTION_PITCH_SCALE = float(os.environ.get("QUESTION_PITCH_SCALE", "1.08"))
APOLOGY_ENERGY_SCALE = float(os.environ.get("APOLOGY_ENERGY_SCALE", "0.9"))


async def _synthesize_one_clause(session: CallSession, clause: str, lang: str, fallback_reason: str | None) -> bytes:
    # KCD-149/157: the caller's policy rate, and slower still for a price,
    # phone number or reference (the two multiply).
    speed = effective_rate(session.policy, FIGURE_SPEECH_SPEED if contains_critical_figure(clause) else None, lang=lang)
    prosody = {
        "pitch_scale": QUESTION_PITCH_SCALE if ("?" in clause or "？" in clause) else None,
        "energy_scale": APOLOGY_ENERGY_SCALE if count_apologies(clause, lang) > 0 else None,
    }
    prosody = {k: v for k, v in prosody.items() if v is not None}
    try:
        return await _tts_router.synthesize(lang, clause, speed=speed, prosody=prosody or None)
    except UnspeakableTextError as e:
        # KCD-455: a production occurrence is an alert, not routine
        # noise -- distinct log level and reason from an ordinary TTS
        # infra failure below, even though both fall back to the same
        # pre-recorded apology.
        logger.error("[%s] reply blocked, unspeakable spans %s in %r", session.call_id, e.spans, clause)
        return _tts.fallback_audio(fallback_reason or "tts_failure", lang)
    except Exception as e:  # noqa: BLE001 - TTS is the last mile, must not raise past here
        logger.warning("[%s] TTS failed (%s) -- using fallback audio", session.call_id, e)
        return _tts.fallback_audio(fallback_reason or "tts_failure", lang)


async def _speak(session: CallSession, text: str, lang: str | None = None, fallback_reason: str | None = None) -> float:
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
    session.tts_output_language = lang
    if fallback_reason in ("llm_failure", "tool_failure"):
        _note(session, "system_failures")
        _rec(session, "reask", {"kind": "system_failure", "reason": fallback_reason, "lang": lang})
    session.agent_spoke_since_commit = True
    if session.speak_epoch != session.turn_epoch:
        # The caller interrupted the turn this reply belongs to: it is stale, and
        # playing it over them is exactly what barge-in exists to stop.
        logger.info("[%s] reply discarded after interrupt", session.call_id)
        return 0.0
    if not session.speaking_filler and not session.turn_spoken:
        # The first real utterance of this turn (never a filler, which exists to cover a slow
        # turn and must reach the caller as fast as possible, not after an extra wait of its own).
        session.turn_spoken = True
        if NATURAL_PAUSE_ENABLED and NATURAL_PAUSE_S > 0:
            await asyncio.sleep(NATURAL_PAUSE_S)
    if "?" in text or "\uff1f" in text:
        session.awaiting_answer = True
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
    # KCD-514: at most one apology per reply. Templates are written to that rule; this catches
    # the ones assembled at run time (a secondary answer appended to a primary one).
    text, stacked = enforce_single_apology(text, lang)
    if stacked:
        apology_events.record("stacked_removed", "reply", lang)
    # sourav, 2026-10-24 (owner): a date is SAID by what is nearest in time -- "18th" -> "আঠারো তারিখ" (same month: no month, no date format); "2 September" from October ->
    # "সামনের মাসের দুই তারিখ"; further away: the month's name (agent/date_phrase.py). Every ISO date in a reply is reworded here, once, just before speaking.
    text = date_phrase.speak_dates(text, lang, slot_proposal.clinic_now().date())
    await session.send_json("AI", text)
    if not session.speaking_filler:
        session.last_spoken = text  # the caller may ask about a word of it ("চুয়াল্লিশ বললেন?")
    clauses = split_into_clauses(text) or [text]

    total_duration = 0.0
    for i, clause in enumerate(clauses):
        if session.speak_epoch != session.turn_epoch:
            break  # interrupted mid-reply: the rest is discarded, not sent
        # A live call (2026-09-28): the filler ("please hold") is itself spoken through this same
        # function while the REAL answer is still being resolved (see _await_with_filler). Marking
        # and logging it here used to consume and clear session.marks (_mark is first-occurrence-only,
        # _log_timing clears the list) before the real answer's own "reply"/"tts"/"send" marks could
        # ever be recorded -- so on exactly the slow turns worth measuring, the turn-timing log went
        # dark right after the filler and the actual delay (the thing a caller waited through) was
        # never logged at all. The filler's own latency is not interesting (it is pre-cached, near-
        # instant); skip marking it so the marks recorded so far (lid/prep/asr) survive to be logged
        # against the real reply that follows.
        if not session.speaking_filler:
            _mark(session, "reply")
        wav = await _synthesize_one_clause(session, clause, lang, fallback_reason)
        if not session.speaking_filler:
            _mark(session, "tts")
        if session.speak_epoch != session.turn_epoch:
            break  # interrupted while this clause was being synthesised

        # Close the gate BEFORE the bytes leave, never after: the client
        # can start playing the moment they land, and a poll tick that
        # slips in between send and gate is exactly the echo this
        # prevents.
        duration = _wav_duration_s(wav)
        session.hold_gate_for(duration)
        session.register_agent_audio(wav, voice=lang)
        await session.send_audio(wav)
        total_duration += duration
        if i == 0:
            if not session.speaking_filler:
                _mark(session, "send")
                _log_timing(session)
            if not session.reply_noted and not session.speaking_filler and session.turn_started_at is not None:
                session.reply_noted = True
                session.signals.note_reply(time.monotonic() - session.turn_started_at)

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


# ---- 4. the clinic cancelled the booking (a doctor not consulting): told with sympathy and an apology ---------------------------------------
_CLINIC_CANCELLED = {
    "bn": "আন্তরিকভাবে দুঃখিত। ডাক্তার {doctor} {date} তারিখে বসতে পারছেন না, তাই আপনার অ্যাপয়েন্টমেন্টটা বাতিল করতে হয়েছে। অসুবিধার জন্য আমরা ক্ষমাপ্রার্থী, আর এর জন্য আপনার কোনো টাকা কাটা হবে না। চাইলে এখনই নতুন একটা তারিখে বুক করে দিতে পারি, কোন দিন সুবিধা হবে?",
    "hi": "हमें बहुत खेद है। डॉक्टर {doctor} {date} को नहीं बैठ पाएँगे, इसलिए आपकी अपॉइंटमेंट रद्द करनी पड़ी। असुविधा के लिए हम क्षमा चाहते हैं, और इसके लिए आपसे कोई शुल्क नहीं लिया जाएगा। चाहें तो अभी नई तारीख पर बुक कर दूँ, कौन सा दिन ठीक रहेगा?",
    "en": "We are so sorry. Dr {doctor} cannot sit on {date}, so we had to cancel your appointment. We sincerely apologise for the trouble, and you will not be charged for it. If you like I can book a new date for you right now -- which day suits you?",
}


async def _clinic_cancelled_reply(session: CallSession, lang: str, phone: str | None = None, confirmation_id: str | None = None) -> str | None:
    """If the booking the caller is asking about was cancelled BY THE CLINIC (a doctor on leave), the sympathetic explanation; else None (the normal
    "no booking found"). The doctor is remembered so "book him on another day" carries over."""
    if _tools is None or not (phone or confirmation_id):
        return None
    try:
        found = await _tools.clinic_cancelled(phone=phone, confirmation_id=confirmation_id)
    except Exception as e:  # noqa: BLE001 - advisory
        logger.warning("[%s] clinic-cancelled lookup failed: %s", session.call_id, e)
        return None
    rows = found.get("bookings") or []
    if not rows:
        return None
    b = rows[0]
    doctor = {"bn": b.get("doctor_name_bn"), "hi": b.get("doctor_name_hi")}.get(lang) or b.get("doctor_name") or ""
    session.last_doctor_enquiry = {"doctor_name": b.get("doctor_name") or doctor, "date": None, "turn": session.turn_count}
    return _CLINIC_CANCELLED.get(lang, _CLINIC_CANCELLED["en"]).format(doctor=doctor, date=b.get("date", ""))


# ---- 7. "do you have the prescription?" answered ---------------------------------------------------------------------------------------------
_RX_NEGATIVE = ("নাই", "নেই", "নেয়", "নাহ", "ছিল না", "নহে", "নয়", "नहीं", "नही", "नाहीं", "nahi", "nai", "dont have", "don't have", "do not have", "not have", "no ", "without", "haven't", "have not")
_RX_POSITIVE = ("আছে", "আছে", "অছে", "है", "हाँ", "हां", "hai", "have", "yes", "হ্যাঁ", "হ্যা", "নিয়ে আসব", "আনব", "लाऊंगा", "लाऊँगा")


async def _answer_prescription(session: CallSession, text: str, lang: str) -> bool:
    """"Do you have the prescription?" answered -- also when the agent only SAID "bring the prescription" and the caller replies about it. The word is found by sound
    (agent/clinical_terms.py): "প্রেস্কৃপ্তিন" is "prescription" when the agent has just been talking about prescriptions (a lower bar in that context), while "পরিষ্কার" is not.
    No prescription -> the doctor's WhatsApp message to the clinic's number; yes -> bring it."""
    pend = getattr(session, "pending_prescription", None)
    asked = bool(pend) and session.turn_count - pend[1] <= 2
    last = getattr(session, "last_spoken", "") or ""
    mentioned = clinical_terms.mentions("prescription", text, context=last)
    if not (asked or (mentioned and clinical_terms.mentions("prescription", last))):
        return False
    low = " " + " ".join(text.lower().split()) + " "
    answer = booking_capture.confirm_answer(text, lang)
    negative = answer == "no" or any(c in low for c in _RX_NEGATIVE)
    positive = answer == "yes" or any(c in low for c in _RX_POSITIVE)
    if not (mentioned or answer is not None):
        return False
    if negative:
        answer = "no"
    elif positive:
        answer = "yes"
    else:
        return False
    session.pending_prescription = None
    if answer == "yes":
        await _speak(session, _with_resume(session, phrase("rx_yes", lang), lang), lang)
        return True
    await _speak(session, _with_resume(session, phrase("rx_no", lang), lang), lang)
    await _queue_prescription_callback(session, "prescription_missing", text, lang)
    return True


# ---- 12. a push-back on a department suggestion is answered with the same suggestion ------------------------------------------------------------
_PUSHBACK = (
    "অন্য ডাক্তার", "অন্য বিভাগ", "না আমি", "আমি চাই না", "চাই না", "লাগবে না", "দেখাব না", "মানছি না", "পছন্দ না", "ওটা না", "ওনার কাছে যাব না", "কার্ডিওলজিস্ট চাই", "স্পেশালিস্ট চাই",
    "दूसरा डॉक्टर", "दूसरे डॉक्टर", "नहीं मुझे", "नहीं चाहिए", "नहीं दिखाना", "मंजूर नहीं", "स्पेशलिस्ट चाहिए",
    "other doctor", "another doctor", "different doctor", "i don't want", "i do not want", "dont want", "don't want", "not a surgeon", "no i want", "i want a specialist", "i want another",
)


async def _answer_route_pushback(session: CallSession, text: str, lang: str) -> bool:
    """"I do not want a general surgeon for a hernia": the clinic's routing does not change because the caller is unhappy with it (owner, 2026-10-18).
    The same department is suggested again, kindly, with who is there."""
    route = getattr(session, "last_route", None)
    if not route or session.turn_count - route["turn"] > 3:
        return False
    low = " ".join(text.lower().split())
    if not any(c in low for c in _PUSHBACK):
        return False
    dept = _dept_label(route["dept"], lang)
    names = ", ".join(route.get("names") or [])
    tail = {
        "bn": f" {names} আছেন। কার কাছে বুক করব?" if names else " কার কাছে বুক করব?",
        "hi": f" {names} हैं। किसके पास बुक करूँ?" if names else " किसके पास बुक करूँ?",
        "en": f" {names} are there. Whom shall I book?" if names else " Whom shall I book?",
    }[lang if lang in ("bn", "hi") else "en"]
    head = {
        "bn": f"আমি বুঝতে পারছি, তবে এই সমস্যার জন্য {dept} বিভাগের ডাক্তারকেই দেখাতে হয়, তাই আমি ওটাই বলব।",
        "hi": f"मैं समझती हूँ, लेकिन इस समस्या के लिए {dept} विभाग के डॉक्टर को ही दिखाना होता है, इसलिए मैं वही बताऊँगी।",
        "en": f"I understand, but for this problem the right doctor to see is in {dept}, so that is what I must suggest.",
    }[lang if lang in ("bn", "hi") else "en"]
    session.last_route = {**route, "turn": session.turn_count}
    await _speak(session, head + tail, lang)
    return True


# ---- 11. human routing, and the numbers kept for a call back ---------------------------------------------------------------------------------------
HUMAN_ROUTE_URL = os.environ.get("HUMAN_ROUTE_URL", "")  # placeholder: where a live-agent transfer is requested once the telephony / contact-centre system exists


def _known_phone(session: CallSession) -> str | None:
    """The best number to call this caller back on: the verified one, else the one booked with, else the newest number said in the call."""
    st = session.booking
    return (
        session.identity.phone
        or (st.slots.get("phone") or st.slots.get("contact_phone") if st is not None else None)
        or (session.caller_numbers[-1] if session.caller_numbers else None)
    )


def _note_caller_number(session: CallSession, phone: str | None) -> None:
    n = booking_capture.normalise_phone(phone or "") if phone else None
    if n and (not session.caller_numbers or session.caller_numbers[-1] != n):
        session.caller_numbers.append(n)


async def _route_to_human_placeholder(session: CallSession, reason: str, phone: str | None) -> None:
    """THE PLACEHOLDER for routing a caller to a person. With HUMAN_ROUTE_URL unset it only logs; set it to the contact-centre / telephony endpoint and it receives
    {"call_id", "reason", "phone", "language"} by POST. (The `handoff_human` websocket frame to the bridge is sent separately, as before.)"""
    logger.info("[%s] human routing requested (%s) phone=%s", session.call_id, reason, "known" if phone else "unknown")
    if not HUMAN_ROUTE_URL:
        return
    try:
        import httpx

        async with httpx.AsyncClient(timeout=4.0) as c:
            await c.post(HUMAN_ROUTE_URL, json={"call_id": session.call_id, "reason": reason, "phone": phone, "language": session.lang})
    except Exception as e:  # noqa: BLE001 - routing is best effort here; the spoken notice and the callback below still stand
        logger.warning("[%s] human routing hook failed: %s", session.call_id, e)


async def _queue_unresolved_callback(session: CallSession) -> None:
    """A call that ended with the caller not served -- cut off, unheard, or a question the agent could not answer -- keeps the number in the callback table so
    a person can ring back. Only with a number the caller actually gave (there is no caller-ID yet); nothing is queued, or said, without one."""
    if _tools is None or session.outcome in ("handed_off", "emergency", "abusive_caller"):
        return
    st = session.booking
    reason = None
    if st is not None and st.stage in ("collecting", "confirming", "awaiting_charge_confirm"):
        reason = "unresolved:booking_not_finished"
    elif session.unclear_streak > 0 or session.reask.consecutive > 0:
        reason = "unresolved:not_understood"
    if not reason:
        return
    phone = _known_phone(session)
    if not phone:
        return
    try:
        summary = build_call_summary(reason, lang=session.lang, phone=phone, call_id=session.call_id)
        await asyncio.wait_for(_tools.request_callback(phone, session.call_id, "asap", reason, summary), 4.0)
        logger.info("[%s] call back queued (%s)", session.call_id, reason)
    except Exception as e:  # noqa: BLE001
        logger.warning("[%s] could not queue the call back (%s): %s", session.call_id, reason, e)



async def _handoff_to_human(session: CallSession, reason: str, languages: tuple[str, ...] | None = None, announce: bool = True):
    """Tell the caller and the telephony bridge that a person is taking over.

    The `handoff_human` frame is the machine-readable half: the SBC/bridge
    that owns the SIP leg acts on it (this repo has no SIP transfer). The
    spoken notice is the human-readable half, said in every language when
    the caller's is not yet known.

    Also QUEUES a callback (clinic-api/enquiry_service.request_callback) whenever a phone is already known
    (session.identity.phone) -- this is what makes a handoff (including the reask policy's own 3rd-failure handoff,
    agent/reask_policy.DEFAULT_MAX_REASKS) an actual actionable queue entry for a human, not only a spoken notice and
    a websocket frame nothing durable is kept of. Without a phone there is nothing to call back on (no automatic
    caller-ID exists in this stack), so nothing is queued -- silently, never a spoken complaint about it.
    """
    logger.warning("[%s] handoff to human: %s", session.call_id, reason)
    _note(
        session,
        "emergency"
        if reason == "emergency"
        else "asked_for_person"
        if reason in ("caller_requested", "history_requested_staff")
        else "system_handoff",
    )
    session.outcome = "handed_off"
    _rec(session, "escalation", reason)
    phone = _known_phone(session)
    await _route_to_human_placeholder(session, reason, phone)
    if phone:
        summary = build_call_summary(f"handoff:{reason}", lang=session.lang, phone=phone, call_id=session.call_id)
        with contextlib.suppress(ToolCallError):
            await _tools.request_callback(phone, session.call_id, "asap", f"handoff:{reason}", summary)
    with contextlib.suppress(Exception):
        await session.ws.send_text(json.dumps({"type": "handoff_human", "reason": reason}))
    total = 0.0
    for lang in (languages or (session.lang,)) if announce else ():  # announce=False: the caller was already told why (phrase "handoff_better")
        with contextlib.suppress(Exception):
            total += await _speak(session, phrase("handoff", lang), lang)
    # Let the notice finish playing before the socket closes under it.
    await asyncio.sleep(min(total + 0.5, 20.0))
    with contextlib.suppress(Exception):
        await session.ws.close()


async def _reask_or_handoff(
    session: CallSession,
    decision,
    lang: str,
    fallback_reason: str | None = None,
    languages: tuple[str, ...] | None = None,
) -> None:
    """Ask again, kindly, before routing to a person (agent/reask_policy.py).
    A second consecutive failure also marks the caller "confused", which
    slows the agent and lowers its escalation threshold for the rest of the
    call (Appendix C)."""
    if session.offtopic_streak > 0 and session.turn_count - session.offtopic_last_turn <= 1:
        # Owner, 2026-10-27: right after the caller was told once that this is not something Sonoscan can help with, "sorry, I did not understand, say again" only keeps the talk going: thank-you and goodbye
        logger.info("[%s] unclear turn straight after an off-topic strike: closing the call politely", session.call_id)
        session.outcome = "off_topic"
        await _end_call(session, lang, "offtopic_final")
        return
    reask_outcomes.record(decision.reason or "unclear", decision.action, lang)
    _note(session, "reasks")
    _rec(session, "reask", {"kind": "reask", "reason": decision.reason or "unclear", "action": decision.action, "lang": lang})
    if decision.action == "reask" and decision.reason == "noisy" and NOISE_SUPPRESS_AFTER_ASKS > 0:
        session.noisy_asks += 1
        if session.noisy_asks >= NOISE_SUPPRESS_AFTER_ASKS and not session.noise_suppress:
            session.noise_suppress = True
            logger.info("[%s] asked %d times about noise: noise suppression is on for the rest of this call", session.call_id, session.noisy_asks)
    if decision.mark_confused:
        apply_caller_state(session.call_state, caller_state="confused")
    # `languages` is passed when the caller's language could not be identified:
    # the re-ask and the hand-off notice are then said in EVERY active language,
    # not in whichever one the call happened to default to.
    langs = languages or (lang,)
    if decision.action == "handoff":
        for lg in langs:
            await _speak(session, phrase("reask_final", lg) + " " + phrase("handoff_better", lg), lg)
        await _handoff_to_human(session, f"unintelligible:{decision.reason}", langs, announce=False)
        return
    for lg in langs:
        text = phrase(decision.phrase_key, lg)
        patience = session.kindness.patience(lg)
        await _speak(session, f"{text} {patience}" if patience else text, lg, fallback_reason=fallback_reason)


async def _retry_on_enhanced_audio(
    session: CallSession, wav_path: str, lang: str | None, result, audio_issues: list[str], duration_s: float | None
):
    """A failed or doubtful turn on a noisy or faint line gets one more try
    on a noise-filtered, level-corrected copy of the clip. Only when the
    first attempt was actually poor -- filtering can hurt a recogniser that
    coped fine -- and never for cross-talk, which filtering cannot fix."""
    if lang is None or not ({"noisy", "too_quiet"} & set(audio_issues)):
        return result
    poor = (
        not result.text.strip()
        or transcript_problem(result.text, lang, duration_s, result.decoder_agreement)
        or is_low_confidence(result.decoder_agreement)
    )
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


def _note_distress(session: CallSession, text: str, lang: str, analysis: dict | None) -> None:
    """Moves a call between neutral and distressed only (an emergency, from words, or a confused caller is never overwritten). In distress a failed turn hands over to a person one try sooner."""
    change = session.distress.observe(text, unanswered_in_a_row=session.reask.consecutive, near_end=(analysis or {}).get("near_end"))
    state = session.call_state.caller_state
    if change == "on" and state == "neutral":
        apply_caller_state(session.call_state, "distressed")
        session.reask.max_reasks = 1
        logger.info("[%s] caller sounds distressed (%s): calm delivery on", session.call_id, session.distress.reason)
    elif change == "off" and state == "distressed":
        apply_caller_state(session.call_state, "neutral")
        session.reask.max_reasks = DEFAULT_MAX_REASKS
        session.acknowledged_state = None  # a later upset is acknowledged afresh
        logger.info("[%s] caller is calm again: normal delivery", session.call_id)


async def _acknowledge_if_due(session: CallSession, lang: str) -> None:
    ack = select_acknowledgement(session.policy, session.call_state.caller_state, lang)
    if ack and session.acknowledged_state != session.call_state.caller_state:
        session.acknowledged_state = session.call_state.caller_state
        session.turn_acknowledged = True
        await _speak(session, ack, lang)


def _note_stated_age(session: CallSession, slots: dict, lang: str) -> None:
    if stated_age_is_senior(slots.get("patient_age"), slots.get("relationship")):
        if session.senior_evidence.note_explicit("stated_age") and not session.call_state.senior:
            apply_caller_state(session.call_state, senior=True)
            senior_detections.record("stated_age", "call", lang)


# owner, 2026-10-24: the day and the time are NOT echoed ("ঠিক আছে, চব্বিশ তারিখ" -- an echo of a day the doctor then turned out not to sit on); the doctor's sitting is what is told next
_ECHOED_FIELDS = ("doctor_name", "patient_name", "phone", "test_name", "new_date", "new_time_slot")


async def _thank(session: CallSession, lang: str) -> None:
    """A bare "Thank you." -- once per turn, and only when this turn is the caller's answer to a question we asked."""
    if not THANKS_ENABLED or not session.answering or getattr(session, "thanked_turn", None) == session.turn_count:
        return
    session.thanked_turn = session.turn_count
    await _speak(session, thanks_for(lang), lang)


async def _thank_for_answer(session: CallSession, st, changed: list[str], prior_slots: dict, lang: str) -> None:
    """Booking flows: thank the caller when they have just given a detail we asked for. Not on the first request
    ("I want an appointment with Dr Sen" answers the greeting, and is not a reply to a question about a detail)."""
    if (prior_slots or st.asked_fields) and any(f not in prior_slots for f in changed):
        await _thank(session, lang)


async def _echo_new_slots(session: CallSession, changed: list[str], prior_slots: dict, slots: dict, lang: str) -> None:
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
        if field in ("phone", "contact_phone", "patient_name", "patient_age", "patient_address"):
            continue  # these are READ BACK for a yes in their own turn: echoing them first says them twice (real call 2026-10-18)
        if value and field in _ECHOED_FIELDS and _readable_in(str(value), lang):  # evidence pack 2026-10-28: "ঠিক আছে, Dr. Abhijit Sarkar।" was blocked as unspeakable -- the caller heard nothing
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


_last_messages_error_logged = False


async def _refresh_messages() -> None:
    """Pull the operator-editable wording from the database (KCD-353/500/513). Bounded and silent on
    failure: the built-in text is spoken whenever the API cannot be reached, so an outage never leaves the
    agent without words."""
    global _last_messages_error_logged
    if _tools is None or not agent_messages.stale():
        return
    try:
        agent_messages.load(await asyncio.wait_for(_tools.agent_messages(), 1.5))
        _last_messages_error_logged = False
    except Exception as e:  # noqa: BLE001
        if not _last_messages_error_logged:
            logger.warning("agent messages not refreshed (%s): built-in wording is used", e)
            _last_messages_error_logged = True


def _note(session: CallSession, name: str, n: int = 1) -> None:
    """Count one signal for the call's implicit happiness score. Never raises: a score is never a reason to fail a turn."""
    try:
        session.signals.note(name, n)
    except Exception as e:  # noqa: BLE001
        logger.warning("[%s] score signal %s failed: %s", session.call_id, name, e)


def _write_satisfaction(session: CallSession) -> None:
    """At hang-up: score the call from its signals and queue the result with the call record. Counts and flags only."""
    st = session.booking
    if st is not None and topic_flow.worth_suspending(st):
        session.signals.left_mid_task = True  # a booking was still half-done when the line dropped
    result = score_call(session.signals)
    logger.info(
        "[%s] call score: %s (%s) %s",
        session.call_id,
        result.score,
        result.band,
        ", ".join(f"{n} {p:+d}" for n, p in result.reasons) or result.reason,
    )
    _rec(session, "satisfaction", result.to_payload(session.signals))


def _rec(session: CallSession, method: str, *args) -> None:
    """Queue one call-record event and start delivering it. Never raises and never waits: a
    record that cannot be written is retried and, at hang-up, reported (agent/call_record.py) --
    it is never a reason to slow or fail a turn."""
    rec = session.recorder
    if rec is None:
        return
    try:
        getattr(rec, method)(*args)
        asyncio.ensure_future(rec.flush())
    except Exception as e:  # noqa: BLE001
        logger.warning("[%s] call record event %s failed: %s", session.call_id, method, e)


def _record_action(session: CallSession, name: str, result: dict, slots: dict) -> None:
    if result.get("success"):
        _note(session, "task_completed")
        _rec(session, "action", name, result.get("confirmation_id"))
        _rec(session, "confirmed", slots)


async def _write_call_event(call_id: str, seq: int, kind: str, payload: dict, caller_phone: str | None) -> dict:
    if _tools is None:
        raise ToolCallError("clinic tools not ready")
    return await _tools.write_call_event(call_id, seq, kind, payload, caller_phone)


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


async def _await_with_filler(session: CallSession, awaitable, lang: str, threshold_s: float = FILLER_THRESHOLD_S):
    """KCD-461: thin wrapper over agent.filler.await_with_filler (the
    tested race logic) supplying "speak the pre-warmed please_wait
    phrase" as the timeout callback -- never synthesized fresh
    (TTSClient's cache already holds it from startup prewarm), so it
    costs no synthesis time at exactly the moment the system is already
    running slow."""

    if not FILLER_ENABLED:
        return await awaitable

    async def _speak_filler():
        session.speaking_filler = True  # the holding phrase is not the answer: do not time it as one
        try:
            await _speak(session, phrase("please_wait", lang), lang)
        finally:
            session.speaking_filler = False

    # The threshold is counted from when the caller stopped speaking, not from when this stage began: recognition
    # already used some of it. It never drops below FILLER_MIN_WAIT_S, so a semantic-cache hit is not announced.
    started = session.turn_started_at
    if started is not None:
        threshold_s = max(threshold_s - (time.monotonic() - started), FILLER_MIN_WAIT_S)
    return await await_with_filler(awaitable, threshold_s, _speak_filler)


async def _slice_utterance(session: CallSession, start_s: float, end_s: float, seq: int) -> str:
    """Cuts [start_s, end_s+pad] -- both ABSOLUTE call-time offsets -- out
    of the call's decoded WAV into its own small file for ASR."""
    sr = session.audio.sample_rate
    clip = session.audio.slice_tensor(start_s, end_s + UTTERANCE_PAD_S)
    clip_path = os.path.join(session.tmpdir, f"utt{seq}.wav")

    def _write():
        wav = clip.unsqueeze(0)
        out_sr = sr
        if sr != SAMPLE_RATE:
            # Only reachable when the browser refused a 16kHz AudioContext.
            # ASR expects 16k, so convert here rather than letting it
            # silently transcribe pitch-shifted audio.
            wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
            out_sr = SAMPLE_RATE
        torchaudio.save(clip_path, wav, out_sr)

    await asyncio.to_thread(_write)
    return clip_path


# KCD-101: the whole wall-clock a caller may wait for "what do they want" (cache lookup + model, all attempts)
# before the apology is spoken. Equal to the model's own default deadline (agent/llm.py, 12 s, REASONED, not
# measured on real telephony). The holding phrase is spoken once FILLER_THRESHOLD_S has passed without an answer.
INTENT_BUDGET_S = float(os.environ.get("INTENT_TURN_BUDGET_S", "12"))
# The semantic-cache embedding lookup gets this head start on the model; after that both run together.
CACHE_HEAD_START_S = float(os.environ.get("INTENT_CACHE_HEAD_START_S", "0.15"))
# ...and it may never hold the turn for more than this (kept as a ceiling on the head start).
CACHE_LOOKUP_MAX_S = float(os.environ.get("INTENT_CACHE_MAX_S", "1.5"))
# Below this much budget left, asking the model is pointless: go straight to the apology.
MIN_MODEL_BUDGET_S = 0.5


async def _resolve_intent_uncached(session: CallSession, text: str, lang: str, key: str) -> dict:
    """Tier 2 (semantic cache) + tier 3 (LLM), run AFTER fast_path already
    abstained. Split out from _resolve_intent so the filler race below can
    wrap the WHOLE remaining sequence -- see KCD-459 note there."""
    # KCD-101: ONE wall-clock budget covers the cache lookup AND every model attempt. The model call already had
    # its own deadline (KCD-465), but the embedding lookup in front of it did not count against it, so a slow
    # cache plus a slow model could keep a caller waiting past the budget. A slow cache is treated as a miss.
    start = time.monotonic()

    def left() -> float:
        return INTENT_BUDGET_S - (time.monotonic() - start)

    # The cache lookup (an embedding call) and the model run TOGETHER, not one after the other. The lookup gets a
    # short head start; if it has not answered by then the model starts anyway and the lookup keeps going. A hit ends
    # the wait at once and the model's answer is dropped; a miss costs the model turn nothing. Measured on the pod, the
    # serial version spent up to 1.5 s waiting for a lookup that then missed, before the 1.3 s model call even began.
    lookup = asyncio.ensure_future(asyncio.to_thread(_intent_cache.get, key))
    lookup.add_done_callback(lambda f: f.cancelled() or f.exception())  # a failed lookup is a miss, never an error
    try:
        cached, how = await asyncio.wait_for(
            asyncio.shield(lookup), timeout=max(0.02, min(CACHE_HEAD_START_S, CACHE_LOOKUP_MAX_S, left()))
        )
    except TimeoutError:
        cached, how = None, "pending"
    except Exception as e:  # noqa: BLE001 - the semantic cache is optional
        logger.warning("[%s] intent cache lookup failed (%s): treated as a miss", session.call_id, e)
        cached, how = None, "error"
    # sourav, 2026-10-14: a sentence that names SEVERAL tests is never answered from the saved-answer cache. An entry
    # holds one test's intent, and a similar-sounding sentence ("uric acid, lipid profile, CBC total") was served the
    # saved "Uric Acid" answer, silently dropping the other two. Skipping the hit sends it to the model as a miss.
    if cached is not None and _fast_path is not None:
        try:
            if _fast_path.looks_multi_test(text, lang):
                logger.info("[%s] intent cache %s hit skipped: sentence names several tests", session.call_id, how)
                cached = None
        except Exception:  # noqa: BLE001 - this guard must never take a turn down
            pass
    if cached is not None:
        logger.info("[%s] intent cache %s hit", session.call_id, how)
        return cached

    budget_left = left()
    if budget_left < MIN_MODEL_BUDGET_S:
        raise ExtractionError(f"intent budget of {INTENT_BUDGET_S}s used up before the model was asked")
    # the extractor bounds itself by `budget_left`; the wait_for is the backstop for a thread that does not
    model = asyncio.ensure_future(
        asyncio.wait_for(asyncio.to_thread(extract_intent, text, 2, lang, budget_left), timeout=budget_left + 1.0)
    )
    model.add_done_callback(lambda f: f.cancelled() or f.exception())
    if not lookup.done():
        await asyncio.wait({lookup, model}, return_when=asyncio.FIRST_COMPLETED)
    if lookup.done() and not lookup.cancelled() and lookup.exception() is None and not model.done():
        hit, how = lookup.result()
        if hit is not None:
            model.cancel()  # its thread finishes on its own, bounded by its deadline; the answer is dropped
            logger.info("[%s] intent cache %s hit (while the model was running)", session.call_id, how)
            return hit
    try:
        data, diag = await model
    except TimeoutError:
        raise ExtractionError(f"intent extraction did not complete within the {INTENT_BUDGET_S}s budget") from None
    logger.info(
        "[%s] intent extracted in %.2fs (%d attempt(s))", session.call_id, diag["total_time_s"], diag["attempts"]
    )
    # Owner, 2026-10-27 (real call, pod log): saving the answer to the semantic cache needs a bge-m3 embedding, and on one turn that embedding took 5.8 s (the Ollama log shows it), which
    # the caller waited for although the save only helps LATER turns. It now runs in the background: a failure or a slow embedding costs nothing on this turn.
    saved = asyncio.ensure_future(asyncio.to_thread(_intent_cache.put, key, data))
    saved.add_done_callback(lambda f: f.cancelled() or f.exception())
    return data


async def _resolve_intent(session: CallSession, text: str, lang: str = "bn") -> dict:
    """Semantic cache in front of the LLM. A hit skips Ollama entirely --
    the slowest hop in the turn -- but the clinic lookup that follows still
    runs live, so a cached intent can never serve a stale price."""
    # Tier 1: decide it locally if we can. For a fixed catalogue the
    # entity is a string-matching problem with a 0.32 confidence margin,
    # where the embedding route had 0.03 -- see agent/fast_path.py. This
    # returns None whenever it is not sure, which is the common case for
    # anything except a routine price or availability question. It has a
    # cue table per language (agent/fast_path_cues.py, KCD-095) and abstains
    # for a language it has none for, so the turn goes to the LLM -- safe by
    # construction, just not free.
    await _maybe_reload_fast_path()
    text = await _normalise_spelled(text)  # letters said one by one ("টি এস এইচ") become the code they spell ("TSH") before anything reads the sentence
    # Real call 2026-10-17: `lang` is the call's sticky REPLY language, not the language of THIS sentence. A Hindi sentence in a call
    # whose reply language was Bengali was matched against the Bengali cue table and answered with the clinic's address. The shortcut,
    # the cache and the model are all given the language the sentence is actually written in; `lang` stays what the reply is spoken in.
    text_lang = language_of_script(text, lang)
    if _fast_path is not None:
        gap = session.turn_count - session.last_enquiry_turn if session.last_enquiry_entities else None
        topic = recent_unique(session.last_enquiry_entities, "test_name", gap)
        hit = await asyncio.to_thread(_fast_path.resolve, text, text_lang, topic)
        if hit is not None:
            refusal = fastpath_guards.veto(hit.intent, hit.slots, text, lambda: len(_fast_path.named_tests(text, text_lang)))
            if refusal:
                logger.info("[%s] fast path %s refused (%s) -- the model decides", session.call_id, hit.intent, refusal)
                hit = None
        if hit is not None and hit.intent == "clinic_faq" and _doctor_booking_open(session) and booking_carryover.continues_booking(text):
            # Real call 2026-10-17: "ওই দিনই এগারোটার সময় করুন" (book it that day at eleven) matched the opening-hours FAQ on the word "সময়" while a doctor
            # booking was waiting for its day/time. A day, a time or a "do it" said to an open doctor booking is the booking's answer, not an FAQ.
            logger.info("[%s] fast path clinic_faq refused: the open doctor booking is being answered", session.call_id)
            return {
                "intent": "book_appointment",
                "slots": {k: None for k in (
                    "test_name", "test_names", "doctor_name", "date", "time_slot", "new_date", "new_time_slot", "confirmation_id",
                    "patient_name", "patient_age", "phone", "contact_phone", "relationship", "spelled_letters", "symptom_description", "faq_topic")},
                "direct_reply_bn": None,
                "secondary_intent": None,
                "secondary_slots": None,
            }
        if hit is not None:
            logger.info("[%s] fast path resolved %s (%.2f) -- no LLM call", session.call_id, hit.intent, hit.confidence)
            return hit.as_llm_shape()

    # Owner, 2026-10-29: "কালকে বসবে" / "আজ আছেন?" -- a SHORT sitting question about a doctor named in the sentence or just discussed, with no booking, number or test in it, is read by rule: the model took ~650 ms
    # to say what the words already say. (The doctor is never a fuzzy guess: _strict_enquiry_doctor.)
    if (
        session.booking is None
        and len(text.split()) <= 30  # (was 6, then 12 -- a real sentence is never skipped for its length: "বলছি কি এক ডাক্তার ধন বাবুর চেম্বারে বসছে" has eight)
        and _SITS_VERB.search(text)
        and booking_capture.read_digits(text)[1] < 2
        and not booking_carryover.mentions_change_or_cancel(text)
        and not booking_carryover.asks_to_book(text)
        and not (_fast_path is not None and _fast_path.named_tests(text, text_lang))
    ):
        _qdoc = _strict_enquiry_doctor(session, text, lang)
        _unk = None if _qdoc else dept_talk.unknown_doctor_name(text)
        if _unk:
            logger.info("[%s] a sitting question about a doctor not on our list (%r) -- answered by rule", session.call_id, _unk)
            return {
                "intent": "doctor_availability",
                "slots": {k: (_unk if k == "doctor_name" else None) for k in (
                    "test_name", "test_names", "doctor_name", "date", "time_slot", "new_date", "new_time_slot", "confirmation_id",
                    "patient_name", "patient_age", "phone", "contact_phone", "relationship", "spelled_letters", "symptom_description", "faq_topic")},
                "direct_reply_bn": None,
                "secondary_intent": None,
                "secondary_slots": None,
                "unknown_doctor": _unk,
            }
        if _qdoc:
            logger.info("[%s] sitting question read by rule -- no model call", session.call_id)
            _said = booking_capture.read_when(text, False, want_day=True).date
            return {
                "intent": "doctor_availability",
                "slots": {k: (_qdoc if k == "doctor_name" else (_said if k == "date" else None)) for k in (
                    "test_name", "test_names", "doctor_name", "date", "time_slot", "new_date", "new_time_slot", "confirmation_id",
                    "patient_name", "patient_age", "phone", "contact_phone", "relationship", "spelled_letters", "symptom_description", "faq_topic")},
                "direct_reply_bn": None,
                "secondary_intent": None,
                "secondary_slots": None,
                "from_context": not _text_names_a_doctor(text, lang),
            }

    # "do you do chest X-ray" written by the Bengali/Hindi recogniser in its own letters is read by rule, not left to the model, which
    # called it "unclear" (real call 2026-10-17). The name after the frame is a candidate for the catalogue lookup that follows.
    asked = translit_frames.match(text) or spelled_letters.bare_code_request(text)
    if asked is None and _fast_path is not None:
        # Real call 2026-10-17: "আপনাদের ওখানে চিস্টিক্স রে হয়" (chest X-ray, misheard) was answered with the payment methods. When the words in the
        # "do you have X?" frame are close to a test the clinic does (catalogue similarity >= 0.75; ordinary clinic words score <= 0.67), X is a test
        # candidate: the lookup answers an exact name or asks "did you mean ...?".
        x = test_availability.have_frame_name(text)
        if x:
            hit = await asyncio.to_thread(_fast_path.catalogue.match, x, "test", text_lang, 0.75)
            if hit and hit[0]:
                asked = x
    if asked is not None:
        logger.info("[%s] test named by rule (frame / spelled code) -> test_rate for %r", session.call_id, asked)
        return {
            "intent": "test_rate",
            "slots": {k: (asked if k == "test_name" else None) for k in (
                "test_name", "test_names", "doctor_name", "date", "time_slot", "new_date", "new_time_slot", "confirmation_id",
                "patient_name", "patient_age", "phone", "contact_phone", "relationship", "spelled_letters", "symptom_description", "faq_topic")},
            "direct_reply_bn": None,
            "secondary_intent": None,
            "secondary_slots": None,
        }

    # The cache stores the whole intent object (smalltalk carries reply
    # text in the caller's language), so the language is part of the key.
    key = text if text_lang == "bn" else f"[{text_lang}] {text}"

    # KCD-459: the filler race used to wrap ONLY the LLM call, so a slow
    # embedding lookup (agent/semantic_cache.py's L2 tier, a live call to
    # Ollama's embed endpoint) burned into FILLER_THRESHOLD_S's budget
    # silently before the caller ever got the "please hold" filler -- a
    # cold/loaded embed step and a cold LLM call could sum past the
    # threshold with the caller hearing nothing for either half. Wrapping
    # the whole cache-then-LLM sequence means "no more than
    # FILLER_THRESHOLD_S of silence" is an end-to-end guarantee for this
    # tier, not just a promise about the LLM's own share of it.
    # sourav, 2026-10-16: a doctor booking request ("I want to book Dr Sen", "Dr Sen ka appointment book karna hai", "ami Dr Sen ke dekhabo") must
    # not die because the model failed or did not understand: the doctor is recognised from the clinic's own catalogue
    # (FastPath.doctor_booking_hint) and the turn enters the normal booking flow. The model is still the first choice; this only fills its gap.
    try:
        data = await _await_with_filler(session, _resolve_intent_uncached(session, text, text_lang, key), lang)
    except ExtractionError:
        hint = _doctor_booking_hint(text, text_lang)
        if hint is None:
            raise
        return hint
    if isinstance(data, dict) and (
        data.get("intent") == "unclear"
        or (
            data.get("intent") in ("lookup_booking", "smalltalk", "clinic_faq", "reschedule_appointment", "cancel_appointment")
            and booking_carryover.asks_to_book(text)
            and not booking_carryover.mentions_change_or_cancel(text)
        )
    ):
        hint = _doctor_booking_hint(text, text_lang)
        if hint is not None:
            return hint
        if isinstance(data, dict) and data.get("intent") == "unclear":
            hint = _doctor_presence_hint(text)
            if hint is not None:
                return hint
    return _restore_said_given_name(_without_generic_doctor_name(_with_whole_doctor_name(data, text, text_lang, session)), text)


def _restore_said_given_name(data, text: str):
    """Owner, 2026-10-29 (hh2: "রাজেশ্বরী দত্ত ... আছেন" was answered as Dr. Manas Dutta; third time this class): the model dropped the FIRST name and kept only the surname ("দত্ত"), so the clinic was asked
    about the surname alone and read out the one doctor it found. When the sentence carries a name-like word right before the surname the model kept, the doctor asked about is "<that word> <surname>" --
    the clinic then matches the whole name (or says there is no such doctor and offers the closest ones); a surname alone is never turned into one doctor's answer. Nothing is guessed: only words
    the caller said are put back."""
    if _fast_path is None or not isinstance(data, dict) or data.get("intent") not in ("doctor_availability", "book_appointment"):
        return data
    slots = data.get("slots")
    name = (slots or {}).get("doctor_name") if isinstance(slots, dict) else None
    if not name or not isinstance(name, str) or len(name.split()) != 1:
        return data
    toks = [t.strip(".,?!।") for t in (text or "").split()]
    not_given = _fast_path.catalogue._NOT_A_GIVEN_NAME  # noqa: SLF001
    for i, t in enumerate(toks):
        if i == 0 or not (t == name or (t.startswith(name) and len(t) - len(name) <= 2)):
            continue
        prev = toks[i - 1]
        if (
            len(prev) > 3 and prev.isalpha() and prev.lower() not in not_given
            and not _sentence_asks_for_a_doctor(prev) and not dept_talk.dept_in_text(prev)
            and not booking_capture.read_when(prev, False, want_day=True).date
        ):
            return {**data, "slots": {**slots, "doctor_name": f"{prev} {name}"}}
    return data


_PRESENCE_WORDS = frozenset("আছে আছেন আছো बैठते बैठती है हैं available".split())
_DOCTOR_TITLES = frozenset("ডাক্তার ডাঃ ডা ডাক্তর ডক্টর डॉक्टर डॉ dr dr. doctor".split())
_PRESENCE_STOP = frozenset("কে কেউ কি কী কোন কোনো কিছু আর এখানে ওখানে সব আমার আমি আমাকে আমাদের আপনার আপনি তার তাঁর সব सब कौन कोई क्या यहाँ मेरा मुझे मेरी".split())


def _doctor_presence_hint(text: str) -> dict | None:
    """Real call 2026-10-28: "শুভদীপ মন্ডল আছে" (is there a Shubhadip Mondal?) was read "unclear" by the model and answered "I did not understand". A short sentence that is only a person's name
    followed by "is there / does he sit" (optionally after "Dr") is a doctor-availability question: the clinic's own list answers it, including "no doctor of that name, did you mean ...?"."""
    words = [w.strip(".,?!।") for w in text.split()]
    words = [w for w in words if w]
    if len(words) < 2 or not words[-1].lower() in _PRESENCE_WORDS:
        return None
    head = words[:-1]
    titled = bool(head) and head[0].lower() in _DOCTOR_TITLES
    name = [w for w in head if w.lower() not in _DOCTOR_TITLES]
    if not name or len(name) > 3 or (len(name) < 2 and not titled):
        return None
    if off_topic.has_health_word(" ".join(name)):
        return None  # "আমার জ্বর আছে" is a symptom, not a doctor
    if any(w in _PRESENCE_STOP or any(ch.isdigit() for ch in w) or not re.fullmatch(r"[\wঀ-৿ऀ-ॿ]+", w) for w in name):
        return None
    logger.info("a bare 'name + is there' is a doctor question: %r", " ".join(name))
    return {
        "intent": "doctor_availability",
        "slots": {k: (" ".join(name) if k == "doctor_name" else None) for k in (
            "test_name", "test_names", "doctor_name", "date", "time_slot", "new_date", "new_time_slot", "confirmation_id",
            "patient_name", "patient_age", "phone", "contact_phone", "relationship", "spelled_letters", "symptom_description", "faq_topic")},
        "direct_reply_bn": None,
        "secondary_intent": None,
        "secondary_slots": None,
    }


_SITS_VERB = re.compile(r"বস(?:বে|বেন|ছে|ছেন|ে|েন|তেন)|আসবেন|থাকবেন|থাকছেন|बैठ(?:ेंगे|ते|ती|ेगा|ेगी)|\bsits?\b|\bsitting\b")


_WHEN_SITS = ("কবে", "কখন", "কোন দিন", "কোন কোন", "বসেন", "বসে", "বসবে", "বসব", "বসছে", "বসছেন", "বসতেন", "কটায়", "कब", "कौन से दिन", "बैठते", "बैठती", "when", "which day", "what time", "schedule")


def _asks_when_sits(text: str) -> bool:
    return any(w in text.lower() for w in _WHEN_SITS)


_NO_BOOKING_FOR_NUMBER = {
    "bn": "{n} নম্বরে কোনো বুকিং পেলাম না। নম্বরটা আরেকবার বলবেন, বা কনফার্মেশন নম্বরটা?",
    "hi": "{n} नंबर पर कोई बुकिंग नहीं मिली। नंबर दोबारा बताइए, या कन्फ़र्मेशन नंबर?",
    "en": "I could not find a booking for {n}. Could you say the number again, or the confirmation number?",
}


def _done_here(text: str | None, lang: str, found: list[str]) -> str:
    """Owner, 2026-10-18: asked whether the tests are done here, the answer STARTS with the yes naming them ("হ্যাঁ, লিপিড প্রোফাইল আর সিবিসি আমাদের এখানে হয়।") and only then the
    rates. `found` = the spoken names of the tests that ARE in the clinic's list; a name that is not found is never part of the yes. Only when the caller did ask whether they are done."""
    if not found or not text or not test_availability.mentions_done_here(text):
        return ""
    sep = {"bn": " আর ", "hi": " और "}.get(lang, " and ")
    names = sep.join(found) if len(found) <= 3 else sep.join(found[:3])
    return {
        "bn": f"হ্যাঁ, {names} আমাদের এখানে হয়। ",
        "hi": f"हाँ, {names} हमारे यहाँ होता है। ",
        "en": f"Yes, we do {names} here. ",
    }[lang if lang in ("bn", "hi") else "en"]


def _found_names(results: list[tuple[str, dict]], lang: str) -> list[str]:
    out = []
    for requested, r in results:
        if r.get("found"):
            out.append({"hi": r.get("test_name_hi"), "en": r.get("test_name")}.get(lang) or r.get("test_name_bn") or requested)
    return out


_SCAN_WORDS = ("ইউএসজি", "यूएसजी", "usg", "আলট্রাসাউন্ড", "আল্ট্রাসাউন্ড", "সোনোগ্রাফি", "अल्ट्रासाउंड", "सोनोग्राफी", "ultrasound", "sonography", "ইউ এস জি")
_SCAN_Q = {"bn": "আপনি কোনটার কথা বলছেন?", "hi": "आप किसकी बात कर रहे हैं?", "en": "Which one do you mean?"}


def _is_scan_name(name: str | None) -> bool:
    low = (name or "").lower()
    return any(w in low for w in _SCAN_WORDS)


async def _usg_list(lang: str) -> str:
    """The scan tests (USG ...) in the catalogue, spoken in the caller's language: "A, B"."""
    try:
        cat = await _cached_catalogue()
        key = {"bn": "aliases_bn", "hi": "aliases_hi"}.get(lang)
        out = []
        for t in cat.get("tests", []):
            if "usg" not in str(t.get("name") or "").lower():
                continue
            aliases = [a for a in (t.get(key) or []) if a and _is_scan_name(a)] if key else []
            out.append(max(aliases, key=lambda a: len(a.split())) if aliases else str(t["name"]))
        return ", ".join(out)
    except Exception:  # noqa: BLE001 - advisory
        return ""


async def _scan_unlisted_reply(lang: str) -> str:
    """A scan the caller named is not in the table (or is only "USG"): say which scans we DO have and ask which one they mean; if theirs is not among them, a colleague takes over."""
    names = await _usg_list(lang)
    ask = _SCAN_Q.get(lang, _SCAN_Q["en"])
    if lang == "bn":
        head = f"ইউএসজির মধ্যে আমাদের এখানে এই টেস্টগুলো হয়: {names}। " if names else ""
        return f"{head}{ask} এগুলোর মধ্যে আপনারটা না থাকলে বলুন, আমি আমাদের কর্মীর কাছে লাইনটা দিয়ে দিচ্ছি।"
    if lang == "hi":
        head = f"अल्ट्रासाउंड में हमारे यहाँ ये टेस्ट होते हैं: {names}। " if names else ""
        return f"{head}{ask} इनमें आपका वाला न हो तो बताइए, मैं हमारे कर्मचारी को लाइन दे देती हूँ।"
    head = f"For ultrasound we do these tests here: {names}. " if names else ""
    return f"{head}{ask} If yours is not among them, tell me and I will pass you to a colleague."


async def _multi_reply(results: list, lang: str) -> str:
    """multi_test_rate_reply, except that a scan the table does not hold is never reported as "no such test": the scans we do have are listed and the caller is asked which one."""
    scan = [n for n, r in results if not r.get("found") and _is_scan_name(n)]
    if not scan:
        return multi_test_rate_reply(results, lang)
    rest = [(n, r) for n, r in results if n not in scan]
    head = multi_test_rate_reply(rest, lang) + " " if rest and any(r.get("found") for _n, r in rest) else ""
    return head + await _scan_unlisted_reply(lang)


async def _answer_scan_followup(session: CallSession, text: str, lang: str) -> bool:
    """After "which scan do you mean? if yours is not among them I will pass you to a colleague": a "no / another" answer is the handoff."""
    last = getattr(session, "last_spoken", "") or ""
    if _SCAN_Q.get(lang, "") not in last and not any(q in last for q in _SCAN_Q.values()):
        return False
    if booking_capture.confirm_answer(text, lang) != "no" and not any(w in text.lower() for w in ("অন্য", "আলাদা", "নেই", "এগুলো না", "other", "another", "none", "दूसरा", "अलग", "इनमें नहीं")):
        return False
    await _handoff_to_human(session, "unlisted_scan", (lang,))
    return True


_BOOKING_CUE = re.compile(r"নাম লেখা|বুকিং|বুক |অ্যাপয়েন্টমেন্ট|অযাপ্ফোন্টমেন্ট|ডাক্তার দেখা|appointment|\bbook|बुक|अपॉइंटमेंट|नाम लिखव", re.I)
_MULTI_KINDS = ("clinic_faq", "test_rate", "test_prep", "doctor_availability")
_FOLLOW_BACK = {
    "bn": "আপনার আরেকটা প্রশ্নও ছিল, সেটা ঠিক বুঝতে পারিনি। একটু আবার বলবেন?",
    "hi": "आपका एक और सवाल भी था, वह मैं ठीक से समझ नहीं पाई। ज़रा फिर से बताइए?",
    "en": "You had another question too, which I did not quite catch. Could you say it again?",
}
# owner, 2026-10-26: when the part that was not understood is about a doctor ("ডাক্তার দেখাতে ডেকেছে ... কটার সময় আসব"), the useful question is WHICH doctor, not "say it again".
_FOLLOW_BACK_DOCTOR = {
    "bn": "কোন ডাক্তারের কথা বলছেন? একটু নামটা বলবেন?",
    "hi": "आप किस डॉक्टर की बात कर रहे हैं? ज़रा नाम बताइए?",
    "en": "Which doctor do you mean? Could you tell me the name?",
}


async def _answer_multi_questions(session: CallSession, text: str, lang: str) -> bool:
    """Two or three questions in one breath ("নাম লেখাবো ... আপনাদের অ্যাড্রেস কোথায়", "CBC price and where are you"): the sentence is cut into its questions (agent/multi_question.py),
    each is recognised by the same routine-question recogniser that answers a single question, and EVERY one is answered, in order. A part that is not a routine question is either the
    booking (it goes on in this same turn, after the answers) or is asked about again -- never silently dropped (owner, 2026-10-25). True when the turn was fully handled here."""
    if _fast_path is None or _tools is None:
        return False
    parts = multi_question.split_questions(text)
    if len(parts) < 2:
        return False
    text_lang = language_of_script(text, lang)
    answered: list[str] = []
    rest: list[str] = []
    seen: set[str] = set()
    for part in parts:
        hit = None
        with contextlib.suppress(Exception):
            hit = await asyncio.to_thread(_fast_path.resolve, part, text_lang, None)
        if hit is not None:
            with contextlib.suppress(Exception):
                if fastpath_guards.veto(hit.intent, hit.slots, part, lambda p=part: len(_fast_path.named_tests(p, text_lang))):
                    hit = None
        if hit is None or hit.intent not in _MULTI_KINDS:
            rest.append(part)
            continue
        key = f"{hit.intent}:{sorted((hit.slots or {}).items())}"
        if key in seen:
            continue
        seen.add(key)
        _turn_text.set(part)
        try:
            reply = await _answer_enquiry_intent(hit.intent, dict(hit.slots or {}), lang)
        except Exception as e:  # noqa: BLE001 - one failed lookup must not lose the other answers
            logger.warning("[%s] multi-question part %r failed: %s", session.call_id, part, e)
            reply = None
        if reply:
            answered.append(reply)
        else:
            rest.append(part)
    if not answered or (len(answered) < 2 and not rest):
        return False  # nothing recognised, or just one question: the normal path
    spoken = " ".join(answered)
    # a doctor is talked about but not named ("আমাকে বারোটার সময় ডাক্তার দেখাতে ডেকেছে"): ask which doctor
    doctor_unnamed = False
    for p in rest:
        if doctor_search._has(doctor_search._norm(p), doctor_search._DOCTOR_WORD):  # noqa: SLF001
            named = None
            with contextlib.suppress(Exception):
                named = _fast_path._named_doctor(fast_path_normalize(p), language_of_script(p, lang))  # noqa: SLF001
            if named is None:
                doctor_unnamed = True
    if doctor_unnamed and session.booking is None:
        await _speak(session, f"{spoken} {_FOLLOW_BACK_DOCTOR.get(lang, _FOLLOW_BACK_DOCTOR['en'])}", lang)
        return True
    booking_part = any(_BOOKING_CUE.search(p) for p in rest)
    if booking_part:
        await _speak(session, spoken, lang)
        return False  # the booking goes on in this same turn and asks its next question
    if rest:
        spoken = f"{spoken} {_FOLLOW_BACK.get(lang, _FOLLOW_BACK['en'])}"
    await _speak(session, _with_resume(session, spoken, lang) if session.booking is not None else spoken, lang)
    return True


_SITTING_WORDS = ("বসছেন", "বসবেন", "বসেন", "बैठ", "sits", "sitting")


async def _answer_when_repeat(session: CallSession, text: str, lang: str) -> bool:
    """"কবে আছে, নেক্স্ট কবে পাব, পরে কবে বসছে" right after the agent said when somebody sits: that IS the question it just answered -- said again (owner, 2026-10-24: it was "sorry,
    I did not understand"). Not when the caller names a doctor (that is a fresh availability question) or when no sitting was just told."""
    last = getattr(session, "last_spoken", "") or ""
    if not last or not _asks_when(text) or len(text.split()) > 16 or not any(w in last for w in _SITTING_WORDS):
        return False
    if any(w in last for w in ("বসেন না", "বসছেন না", "বসবেন না", "नहीं बैठ", "Nobody", "does not sit")):
        return False  # what was just said is that nobody sits: repeating it answers nothing
    if booking_capture.confirm_answer(text, lang) is not None:
        return False
    low = text.lower()
    if any(w in low for w in ("রিপোর্ট", "report", "টেস্ট", "test", "रिपोर्ट", "টাকা", "পেমেন্ট", "payment", "সিরিয়াল", "serial", "token")):
        return False  # "when do I get the report" is not "when does the doctor sit"
    if _fast_path is not None:
        with contextlib.suppress(Exception):
            if _fast_path._named_doctor(fast_path_normalize(text), language_of_script(text, lang), strong=True) is not None:  # noqa: SLF001
                return False
    await _speak(session, last, lang)
    return True


async def _answer_after_date(session: CallSession, text: str, lang: str) -> bool:
    """"একুশ তারিখের পরের কোনো দিন, কালকে হবে না": a booking is open with a doctor, and the caller rules a day out and asks for a later one. The doctor's next sitting AFTER
    that day is found and told (agent/slot_proposal.py), exactly as when the caller names a day -- never read as a question about lab tests."""
    st = session.booking
    if st is None or st.action != "book_appointment" or not st.slots.get("doctor_name"):
        return False
    today = slot_proposal.clinic_now().date()
    after = booking_capture.after_date(text, today)
    if not after:
        return False
    import datetime as _dt

    target = (_dt.date.fromisoformat(after) + _dt.timedelta(days=1)).isoformat()
    for k in ("_time_ok", "_sits_told", "time_slot", "_time_explicit", "_day_full_from"):
        st.slots.pop(k, None)
    st.hold_token, st.hold_doctor_id = None, None
    st.slots["date"] = target
    if await _propose_slot(session, st, lang):
        return True
    await _speak(session, slot_proposal.book_that_day_question(target, lang), lang)
    return True


async def _answer_where_in_address(session: CallSession, text: str, lang: str) -> bool:
    """"বিনিয়া পুকুরটা কোথায়" -- a where-question about a piece of the address. Voice cannot place it: Google Maps, or the WhatsApp number to send the location to.
    The number, when said next, is read and the location is queued to it (placeholder provider, like every message here)."""
    pend = getattr(session, "location_pending", None)
    if pend is not None and session.turn_count - pend <= 3:
        digits = booking_capture.read_digits(text)[0]
        if len(digits) >= 10:
            number = booking_capture.phone_from_digits(digits[-10:] if len(digits) > 10 else digits) or digits[:10]
            session.location_pending = None
            session.caller_numbers.append(number)
            await _speak(session, owner_answers.location_sent(number, lang), lang)
            phone = number
            try:
                summary = build_call_summary("send_location_whatsapp", lang=lang, transcript=text, phone=phone, call_id=session.call_id)
                await asyncio.wait_for(_tools.request_callback(phone, session.call_id, "asap", "send_location_whatsapp", summary), 4.0)
            except Exception as e:  # noqa: BLE001 - advisory: the caller was told what to expect, a failed queue must not break the call
                logger.warning("[%s] could not queue the location message: %s", session.call_id, e)
            return True
        if digits and not owner_answers.asks_where_in_address(text):
            await _speak(session, owner_answers.location_again(lang), lang)
            return True
    if owner_answers.asks_where_in_address(text):
        session.location_pending = session.turn_count
        await _speak(session, owner_answers.where_reply(lang), lang)
        return True
    return False


async def _any_test_needs_prescription(slots: dict) -> bool:
    """Is any test named in `slots` marked "prescription required" in the clinic's lab-test table? A failed lookup is simply "no": the normal counter message stands."""
    names = [n for n in list((slots or {}).get("test_names") or []) + [(slots or {}).get("test_name")] if n][:5]
    for name in names:
        try:
            if (await _tools.get_test_rate(name)).get("prescription_required"):
                return True
        except Exception:  # noqa: BLE001 - advisory
            continue
    return False


def _readback_pending(session: CallSession) -> bool:
    b = session.booking
    return b is not None and b.action == "book_appointment" and bool(b.pending_name or b.slots.get("_phone_pending"))


def _whatsapp_confirm_pending(session: CallSession) -> bool:
    """True while "does this number have WhatsApp?" is outstanding and not yet answered. Real calls, 2026-10-26: a caller
    reassuring the agent ("WhatsApp ache, ei number e WhatsApp ache") often runs 6+ words long and in a script the LID
    reads as decisive -- long enough to trip the "clear sentence overrides the language lock" rule below even though it
    is still just the yes/no answer to this one booking question, never a real request to switch languages."""
    b = session.booking
    return b is not None and b.action == "book_appointment" and bool(b.slots.get("_wa_asked")) and not b.slots.get("_wa_done")


def _doctor_booking_open(session: CallSession) -> bool:
    """A doctor booking is being filled in, or a doctor was just asked about."""
    b = session.booking
    if b is not None and b.action == "book_appointment" and b.stage == "collecting" and not b.is_stale():
        return True
    return booking_carryover.recent(session.last_doctor_enquiry, session.turn_count)


def _with_whole_doctor_name(data, text: str, lang: str, session: "CallSession | None" = None):
    """Real call 2026-10-17: "ডাক্তার সেন এর অ্যাপয়েন্টমেন্ট নেবো আমি বিশ্বজিৎ সেন" -- the model took the first \"Sen\" as the doctor and the clinic asked
    \"Arindam or Biswajit?\". When the caller SAID one doctor's whole name anywhere in the sentence, that is the doctor: the slot is set to it. A
    booking request for that doctor enters the booking flow. Never raises."""
    if _fast_path is None or not isinstance(data, dict) or not isinstance(data.get("slots"), dict):
        return data
    b = session.booking if session is not None else None
    if b is not None and b.action == "book_appointment" and b.slots.get("doctor_name") and not b.slots.get("patient_name") and not _sentence_asks_for_a_doctor(text):
        # Real call 2026-10-26: the doctor was already confirmed (Dr Joydeep Sanyal) and the caller then gave the PATIENT's
        # name, "সায়ন্তন চৌধুরী" -- whose surname matched a different real doctor's surname ("...Chowdhury") at the 0.9
        # fuzzy-match bar below and silently swapped the booking's doctor. Once a doctor is confirmed, naming the patient
        # must never re-run doctor matching -- that question is settled; only the name is still being asked.
        return data
    if data.get("intent") not in ("department_query", "doctor_availability", "book_appointment", "unclear", "lookup_booking", "reschedule_appointment", "cancel_appointment"):
        return data
    try:
        name, form, score = _fast_path.catalogue.match(fast_path_normalize(text), "doctorfull", lang, 0.9)
    except Exception:  # noqa: BLE001 - a repair must never take a turn down
        logger.warning("whole doctor name check failed", exc_info=True)
        return data
    if not name:
        return data
    whole = form or name
    if data["slots"].get("doctor_name") == whole:
        return data
    # KCD-lead-3: "department_query" and "doctor_availability" are the two intents a real call (see the
    # book_appointment re-route's own comment further down, for the mid-booking case) has shown the per-
    # utterance classifier confuse on a terse answer. Outside that re-route's mid-booking scope, a confidently
    # matched WHOLE doctor name settles it the same deterministic way this function already settles an
    # ambiguous surname: naming a real doctor was never a department question.
    if data.get("intent") == "department_query":
        return {**data, "intent": "doctor_availability", "slots": {**data["slots"], "doctor_name": whole}}
    if data.get("intent") == "doctor_availability":
        hint = _doctor_booking_hint(text, lang)  # None for a schedule question, so only a booking request changes intent
        if hint is not None:
            return hint
    if data.get("intent") == "doctor_availability" and booking_carryover.continues_booking(text) and not _asks_when_sits(text):
        return {**data, "intent": "book_appointment", "slots": {**data["slots"], "doctor_name": whole}}  # "…নেব আমি বিশ্বজিৎ সেন" is a booking, with the whole name said
    if data.get("intent") in ("doctor_availability", "book_appointment"):
        data = {**data, "slots": {**data["slots"], "doctor_name": whole}}
    return data


_NOT_A_NAME_ANSWER_INTENTS = ("test_rate", "test_prep", "clinic_faq", "department_query", "smalltalk", "unclear", "doctor_availability", "lookup_booking")


def _name_answer_repair(session: CallSession, text: str, data):
    """Owner, 2026-10-26: asked "your name please?", a caller said "रोहिमान शेख" and the model read it as a lab test ("no test of that name -- did you mean the urine test?"). When a booking is
    waiting for the patient's name and the sentence is, by the plain rules (agent/booking_capture.candidate_name: one to four words, letters only, no question, no booking words), a
    name, it is the name -- whatever intent the model guessed. Only the intents that can never be an answer to this question are overridden. Never raises."""
    st = session.booking
    if st is None or not isinstance(data, dict) or not isinstance(data.get("slots"), dict):
        return data
    if data.get("intent") not in _NOT_A_NAME_ANSWER_INTENTS or data["slots"].get("patient_name"):
        return data
    if data.get("intent") in ("doctor_availability", "lookup_booking") and _sentence_asks_for_a_doctor(text):
        return data  # owner, 2026-10-27: only a sentence with NO doctor word is a bare name ("সায়ন দাস" -- whose surname is also a doctor's)
    try:
        if "patient_name" not in st.asked_fields or st.slots.get("patient_name") or st.pending_name or st.name_confirmed:
            return data
        name = booking_capture.candidate_name(text)
    except Exception:  # noqa: BLE001 - a repair must never take a turn down
        return data
    if not name:
        return data
    slots = {**data["slots"], "patient_name": name}
    if data.get("intent") == "doctor_availability":
        slots["doctor_name"] = None  # the surname that matched a doctor was the patient's own
    return {**data, "intent": "book_appointment", "slots": slots}


def _readable_in(value: str, lang: str) -> bool:
    """Can the call's voice say `value`? The Bengali voice reads Bengali script, the Hindi voice Devanagari (Latin letters are dropped by both: agent/pronunciation.py); English reads Latin."""
    if lang == "bn":
        return re.search(r"[ঀ-৿]", value or "") is not None
    if lang == "hi":
        return re.search(r"[ऀ-ॿ]", value or "") is not None
    return re.search(r"[A-Za-z]", value or "") is not None


def _sentence_asks_for_a_doctor(text: str) -> bool:
    """The caller's sentence itself talks about a doctor ("বাচ্চাদের ডাক্তার কে আছে", "डॉक्टर मंडल", "ড মণ্ডল"): a plain patient name has no such word."""
    return doctor_search._has(doctor_search._norm(text), doctor_search._DOCTOR_WORD)  # noqa: SLF001


def _lock_doctor_name_during_name_capture(session: CallSession, data, text: str = ""):
    """Owner, 2026-10-26: real calls -- "সায়ন্তন চৌধুরী" and "সায়ন দাস", both patients, each matching a real doctor's
    surname -- silently swapped the booking's already-confirmed doctor while the patient's NAME was being taken.
    _with_whole_doctor_name already blocks the fast-path catalogue match from doing this, but the model's own slot
    output can say a different doctor_name too, through any path that produces `data` (a fast-path hit, a
    department-search hit, or the full model call) -- this runs once, after all of them, and is the one lock every
    path goes through. While a doctor is confirmed and the patient's name is not yet captured, doctor_name cannot
    change, no matter what produced this turn's `data`; patient_name and everything else the turn found is kept.
    Never raises."""
    st = session.booking
    if st is None or not isinstance(data, dict) or not isinstance(data.get("slots"), dict):
        return data
    confirmed = st.slots.get("doctor_name") if st.action == "book_appointment" else None
    if not confirmed or st.slots.get("patient_name"):
        return data
    if _sentence_asks_for_a_doctor(text):
        # Owner, 2026-10-27 (real call): "বাচ্চাদের ডাক্তার কে আছে" in the middle of a booking kept answering with the doctor already chosen (a General Medicine doctor) because this lock
        # froze him. The lock exists for a PATIENT'S NAME that happens to match a doctor's surname; a sentence that says "doctor" is the caller changing the doctor on purpose.
        return data
    if data["slots"].get("doctor_name") in (None, "", confirmed):
        return data
    return {**data, "slots": {**data["slots"], "doctor_name": confirmed}}


def _without_generic_doctor_name(data):
    """Owner, 2026-10-26: "डॉक्टर् दिखाना था" ("I wanted to see a doctor") reached the clinic as a doctor NAMED "डॉक्टर्". The word "doctor" alone, in any script, names nobody:
    the slot is emptied and the turn asks which doctor (or what the problem is) as it does for any missing name. Never raises."""
    if not isinstance(data, dict) or not isinstance(data.get("slots"), dict):
        return data
    name = data["slots"].get("doctor_name")
    if name and isinstance(name, str) and doctor_search.is_generic_doctor_phrase(name):
        return {**data, "slots": {**data["slots"], "doctor_name": None}}
    return data


def _doctor_booking_hint(text: str, lang: str) -> dict | None:
    """sourav, 2026-10-16: the fast path's booking hint as the dict shape the rest of the turn expects, or None. Never raises: a hint
    that cannot be made is simply no hint."""
    if _fast_path is None:
        return None
    try:
        hit = _fast_path.doctor_booking_hint(text, lang)
    except Exception:  # noqa: BLE001 - a fallback must never take a turn down; the original model error / unclear stands
        logger.warning("doctor booking hint failed", exc_info=True)
        return None
    return hit.as_llm_shape() if hit is not None else None


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
    "doctor_ambiguous": "doctor_name",
    "invalid_date": "date",
    "date_in_past": "date",
    "too_far_ahead": "date",  # sourav, 2026-10-24: a date past the booking window is cleared so a nearer day can be given
    "book_on_the_day": "date",  # owner, 2026-10-29: a doctor booked only on the day he sits -- the date asked for is cleared
    "doctor_not_available_that_day": "date",
}


def _field_to_reclear_on_hold_failure(reason: str | None) -> str:
    return _HOLD_FAILURE_FIELD.get(reason, "time_slot")


# A not-found answer that carries exactly ONE sound-alike suggestion ("did you mean Dr. Mukherjee?") is
# remembered for the turn, so a "yes" runs the lookup with that name -- otherwise the question would lead
# nowhere. Set by the lookups below, read by _finish_enquiry_turn.
_suggested: contextvars.ContextVar[dict | None] = contextvars.ContextVar("suggested", default=None)
# sourav, 2026-10-14: names a price was found for during THIS turn's _answer_enquiry_intent calls; _finish_enquiry_turn
# moves them onto session.priced_tests. A ContextVar (like _suggested) so the answer function keeps its signature.
_priced_names: contextvars.ContextVar[list | None] = contextvars.ContextVar("priced_names", default=None)
# sourav, 2026-10-14: set by _answer_enquiry_intent when a doctor-availability lookup FOUND the doctor; _finish_enquiry_turn stores it.
_doctor_enquiry_found: contextvars.ContextVar[dict | None] = contextvars.ContextVar("doctor_enquiry_found", default=None)


_doctor_from_context: contextvars.ContextVar[bool] = contextvars.ContextVar("doctor_from_context", default=False)  # this turn's doctor is the one just discussed, not one named in the sentence


def _pronoun_reply(slots: dict, result: dict, reply: str, lang: str) -> str:
    """Owner, 2026-10-29: "কালকে বসবে" about the doctor just discussed is answered "উনি আগামীকাল, ত্রিশ তারিখে বসবেন। সময়: 10:00-13:00।" -- no "yes", no name repeated, no "are you asking about ...?"."""
    from agent.reply_templates import _spoken_doctor_name

    pronoun = {"bn": "উনি", "hi": "वे", "en": "They"}.get(lang, "They")
    if lang == "hi":
        name = f"डॉक्टर {result['doctor_name_hi']}" if result.get("doctor_name_hi") else None
    elif lang == "bn":
        name = _spoken_doctor_name(slots, result)
    else:
        full = result.get("doctor_name") or ""
        name = f"Doctor {full.replace('Dr.', '').replace('Dr', '').split()[-1]}" if full.split() else None
    body = reply
    for lead in ("হ্যাঁ, ", "जी हाँ, ", "हाँ, ", "Yes, "):
        if body.startswith(lead):
            body = body[len(lead):]
            break
    if name and name in body:
        body = body.replace(name, pronoun, 1)
    return body


def _dept_possessive(label: str, lang: str) -> str:
    """"ডার্মাটোলজির" / "डर्मेटोलॉजी के" / "Dermatology's" -- the department said before a doctor's name."""
    if lang == "bn":
        return label + ("র" if label[-1:] in "ািীেোুূঅআইঈউঊএঐওঔ" else "ের")
    if lang == "hi":
        return f"{label} के"
    return f"{label}'s"


def _confirm_surname_match(slots: dict, result: dict, reply: str, lang: str) -> str:
    """Owner, 2026-10-29: only a SURNAME was said ("ডাক্তার আর চৌধুরীকে কালকে পাওয়া যাবে") and one doctor has it. The answer names the doctor's department before the name and asks whether that
    is who is meant -- "আপনি কি ডার্মাটোলজির ডাঃ রাজীব চৌধুরীর কথা বলছেন? উনি আগামীকাল ... বসবেন।" -- instead of a bare answer about someone who may not be the one asked about (and never "which doctor?" after
    already answering). The schedule is the clinic's own; only the framing is added."""
    from agent.reply_templates import _spoken_doctor_name

    dept = _fast_path.catalogue.doctor_department(result.get("doctor_name")) if _fast_path is not None else None
    if lang == "hi":
        name = f"डॉक्टर {result['doctor_name_hi']}" if result.get("doctor_name_hi") else "डॉक्टर"
    elif lang == "bn":
        name = _spoken_doctor_name(slots, result)
    else:
        full = result.get("doctor_name") or ""
        name = f"Doctor {full.replace('Dr.', '').replace('Dr', '').split()[-1]}" if full.split() else "the doctor"
    full = f"{_dept_possessive(_dept_label(dept, lang), lang)} {name}" if dept else name
    pronoun = {"bn": "উনি", "hi": "वे", "en": "they"}.get(lang, "they")
    body = reply.replace(name, pronoun, 1) if name and name in reply else reply
    ask = {
        "bn": f"আপনি কি {full}-এর কথা বলছেন? ",
        "hi": f"क्या आप {full} की बात कर रहे हैं? ",
        "en": f"Are you asking about {full}? ",
    }.get(lang, f"Are you asking about {full}? ")
    for lead in ("হ্যাঁ, ", "हाँ, ", "Yes, "):  # the "yes" belonged to the answer, not to the new question
        if body.startswith(lead):
            body = body[len(lead):]
            break
    return ask + body
# The caller's words for this turn, so a total can be checked against how many tests they SAID they meant ("all three together").
_turn_text: contextvars.ContextVar[str | None] = contextvars.ContextVar("turn_text", default=None)
# owner 2026-10-18: the test whose price was just given and which needs a prescription (the question is asked after the price), and the department just suggested
_rx_asked: contextvars.ContextVar[str | None] = contextvars.ContextVar("rx_asked", default=None)
_route_found: contextvars.ContextVar[dict | None] = contextvars.ContextVar("route_found", default=None)
# a department question nobody answers today: the next sitting that was told and offered (set in _answer_enquiry_intent, kept as dept_pending by _finish_enquiry_turn)
_next_sit_offer: contextvars.ContextVar[dict | None] = contextvars.ContextVar("next_sit_offer", default=None)


def _note_suggestion(intent: str, slot: str, slots: dict, result: dict) -> None:
    names = result.get("did_you_mean") or []
    # Owner, 2026-10-29: a suggestion that is only a word the caller already said is not SPOKEN any more ("did you mean পাল?" after "এম পাল"), so it must not wait as an open question either --
    # the next sentence ("…জেড হাক ডক্টর আছে") was taken as the yes to it and answered about Dr. Pal.
    bn = result.get("did_you_mean_bn") or []
    if names and len(bn) == len(names):
        from agent import name_script

        query = name_script.clean_query(str(slots.get(slot) or ""), "bn")
        names = [n for n, b in zip(names, bn) if name_script.drop_said([b], query)]
    if not result.get("found") and result.get("needs_confirmation") and not result.get("ambiguous") and len(names) == 1:
        _suggested.set({"intent": intent, "slot": slot, "slots": dict(slots), "name": names[0]})


_alias_table_cache: tuple[int, AliasTable] | None = None
_refiner_cache: tuple[int, asr_refine.AsrRefiner] | None = None


async def _asr_refine_shadow(session: CallSession, asr_result, lang: str) -> None:
    """Shadow only: log what agent/asr_refine.py would do with the other decoder's text. Changes nothing the caller hears, and can never fail or hold up a turn. The log line carries the
    decision and the catalogue entry (clinic data), never the caller's words."""
    global _refiner_cache
    alternatives = getattr(asr_result, "alternatives", None) or []
    if not alternatives:
        return
    try:
        cat = await _cached_catalogue()
        if _refiner_cache is None or _refiner_cache[0] != id(cat):
            _refiner_cache = (id(cat), asr_refine.AsrRefiner(cat))
        d = _refiner_cache[1].decide(asr_result.text, alternatives, lang)
        if d.action != "none":
            logger.info("[%s] ASR_REFINE shadow turn=%d lang=%s decoder=%s agreement=%.2f -> %s %s %s", session.call_id, session.turn_count, lang,
                        getattr(asr_result, "decoder_used", "?"), float(getattr(asr_result, "decoder_agreement", 0.0) or 0.0), d.action, d.kind, d.entry)
    except Exception as e:  # noqa: BLE001 - a shadow measurement is never a reason to fail a turn
        logger.debug("ASR_REFINE shadow skipped: %s", e)


async def _catalogue_mentions(text: str, lang: str) -> tuple[bool, bool, tuple[str, ...]]:
    """(names a test, names a doctor, doctor candidates): does the catalogue know a test / doctor in this sentence
    by any spoken form, and -- when a doctor mention was AMBIGUOUS (a surname/alias two or more real doctors share,
    e.g. two Sens) -- the real doctors it could be, from the catalogue itself. Real calls, 2026-10-27: "ডাক্তার সেন"
    and "ডক্টর ঋষা" were each answered with a confident, different doctor's real schedule; `doctor_candidates` is
    what lets agent/slot_grounding.ground_slots refuse a model's doctor_name claim that is not even one of the
    catalogue's own candidates for what was actually said (see that module's docstring for the full rule). When it
    cannot tell (no catalogue, any error) it says True for both booleans and no candidates, so nothing is ever
    dropped on a guess."""
    global _alias_table_cache
    try:
        cat = await _cached_catalogue()
        if _alias_table_cache is None or _alias_table_cache[0] != id(cat):
            _alias_table_cache = (id(cat), AliasTable(cat))
        entities = _alias_table_cache[1].entities(text, lang)
        kinds = {e.kind for e in entities}
        candidates = tuple(dict.fromkeys(c for e in entities if e.kind == "doctor" and e.value is None for c in e.candidates))
        return "test" in kinds, "doctor" in kinds, candidates
    except Exception:  # noqa: BLE001 - grounding is a safety net, never a reason to fail a turn
        # Owner, 2026-10-29 (CLAUDE.md "Doctor Nobody"): this used to say True for both booleans on ANY error ("nothing
        # is ever dropped on a guess") -- but that fails OPEN: a catalogue hiccup on this one turn fully disabled the
        # doctor-name hallucination check for it, the one turn it is needed most. Failing closed (nothing confirmed)
        # means ground_slots falls back to its literal-substring check instead -- a caller re-asked once on a rare
        # transient error is a far smaller harm than a wrong real doctor's real schedule stated with confidence.
        return False, False, ()


async def _chamber_window(session: CallSession, text: str):
    """sourav, 2026-10-16: the doctor's chamber hours on the booking date as [(start_min, end_min)], or None when not known yet.
    WHY: a bare "ten" in a booking must be read against the hours the doctor actually sits (10-12 -> 10:00), not guessed as 22:00 by the
    model. Only asked while a doctor booking is collecting, the doctor and the day are known, and the time is still missing. The clinic
    API stays the judge of every slot -- this only chooses between 10 am and 10 pm. A failed lookup just means 'unknown' (clinic day)."""
    st = session.booking
    if not booking_capture._active(st) or st.slots.get("time_slot") or not st.slots.get("doctor_name"):
        return None
    when = None if st.slots.get("date") else booking_capture.read_when(text, False, want_day=True).date
    day = st.slots.get("date") or when
    if not day:
        return None
    cache = getattr(st, "_chamber_cache", None)
    if cache is None:
        cache = {}
        try:
            st._chamber_cache = cache
        except AttributeError:
            pass
    key = (st.slots["doctor_name"], day)
    if key not in cache:
        try:
            avail = await _tools.get_doctor_availability(st.slots["doctor_name"], day)
        except ToolCallError as e:
            logger.warning("[%s] chamber window lookup failed (%s)", session.call_id, e)
            return None
        cache[key] = (
            booking_capture.parse_window(avail.get("chamber_hours")) if avail.get("found") and avail.get("available") is not False else None
        )
    return cache[key]


async def _ground_model_slots(session: CallSession, text: str, lang: str, data: dict, trust_age: bool = False) -> dict:
    """Real call 2026-10-14: "reply in english" came back from the intent model as test_names=["CBC", "ECG"], and the
    agent priced both and added a total nobody asked for. A test or doctor name the sentence gives no support for is
    dropped here (agent/slot_grounding.py has the exact rule). `trust_age`: this turn's slots came from booking_
    capture.py's own deterministic read-back/confirm flow, not a model guess -- see ground_slots' own docstring.
    Returns `data` with cleaned slots; never mutates the dict it is given (it may be the semantic cache's own copy)."""
    names_test, names_doctor, doctor_candidates = await _catalogue_mentions(text, lang)
    cleaned = dict(data)
    total_dropped = 0
    for key in ("slots", "secondary_slots"):
        value = data.get(key)
        if isinstance(value, dict) and value:
            new, dropped = ground_slots(
                text, value, names_a_test=names_test, names_a_doctor=names_doctor, doctor_candidates=doctor_candidates, trust_age=trust_age
            )
            if dropped:
                cleaned[key] = new
                total_dropped += len(dropped)
    if total_dropped:
        logger.info(
            "[%s] dropped %d name(s) the model produced that the caller never said", session.call_id, total_dropped
        )
    cleaned = await _augment_test_slots(text, cleaned)
    return _repair_day_slots(session, text, cleaned)


async def _augment_test_slots(text: str, data: dict) -> dict:
    """Test codes in the sentence (CBC, TMT, "DSH" for TSH, HbA1c) are added to the test names the model found -- or to none at all, when it found none. The model returns one name of
    three, or an empty list, on a garbled mixed-script sentence; the codes written in Latin letters are right there to read (owner, 2026-10-19: "never understood any test name")."""
    try:
        if data.get("intent") not in ("test_rate", "book_test", "test_prep", "test_availability"):
            return data
        slots = data.get("slots")
        if not isinstance(slots, dict):
            return data
        codes = await _catalogue_codes()
        found = spelled_letters.latin_test_tokens(text, codes)
        if not found:
            return data
        names = [n for n in list(slots.get("test_names") or []) + ([slots["test_name"]] if slots.get("test_name") else []) if n]
        low = {n.strip().lower() for n in names}
        extra = [c for c in found if c.lower() not in low and not any(c.lower() == str(k).lower() for k in low)]
        if not extra:
            return data
        merged = names + extra
        new = {**slots, "test_names": merged}
        if len(merged) > 1:
            new.pop("test_name", None)
        return {**data, "slots": new}
    except Exception as e:  # noqa: BLE001 - advisory
        logger.warning("test-code augmentation skipped: %s", e)
        return data


_DAY_WORDS = booking_capture._TODAY | booking_capture._TOMORROW | booking_capture._DAY_AFTER


def _repair_day_slots(session: CallSession, text: str, data: dict) -> dict:
    """Real call 2026-10-19: "কখন বসবেন পরশু উনি" (when will he sit the day after tomorrow) was heard as a doctor called "পরশু", and "একুশ তারিখ" (the 21st) gave no date at all, so the
    answer was for TODAY. A day word is never a doctor's name -- the doctor the call was just talking about stays, and the day word is the date; "<number> তারিখ" is a date."""
    slots = data.get("slots")
    if not isinstance(slots, dict):
        return data
    today = slot_proposal.clinic_now().date()
    words = booking_capture._words(text)
    new = dict(slots)
    name = str(new.get("doctor_name") or "").strip().lower()
    if name and (name in _DAY_WORDS or all(w in _DAY_WORDS for w in name.split())):
        new.pop("doctor_name", None)
        ctx = getattr(session, "last_doctor_enquiry", None)
        booked = session.booking.slots.get("doctor_name") if session.booking is not None else None
        known = booked or ((ctx or {}).get("doctor_name") if booking_carryover.recent(ctx, session.turn_count) else None)
        if known:
            new["doctor_name"] = known
    date, _used = booking_capture.day_from_words(words, today)
    if date:
        new["date"] = date
    if new == slots:
        return data
    return {**data, "slots": new}


async def _answer_test_availability(slots: dict, lang: str) -> tuple[str, dict]:
    """"Is the lipid profile test done here?": YES (from the clinic's own table) and an invitation to ask more. The lookup
    comes first -- "yes" is only said for a test the catalogue found -- and the reply carries no price, no preparation
    and no time. A test the table does not hold gets the ordinary not-found / did-you-mean answer, unchanged."""
    names = [n for n in (slots.get("test_names") or []) if n]
    single = slots.get("test_name")
    if single and single not in names:
        names.insert(0, single)
    results = [(n, await _tools.get_test_rate(n)) for n in names[:3]]
    found = [n for n, r in results if r.get("found")]
    if not found:
        name, result = results[0]
        _note_suggestion("test_rate", "test_name", {"test_name": name}, result)
        return test_rate_reply({"test_name": name}, result, lang), {"test_name": name}
    # Real call 2026-10-17: "কিডনি ফাঙ্কটিয়ান টেস্ট হয়" was answered "…কিডনি ফাঙ্কটিয়ান টেস্ট হয়" -- the caller's misheard words repeated back. The answer
    # names the test the way the clinic does (its own Bengali/Hindi name from the table), the way the price answer already does.
    key = {"bn": "test_name_bn", "hi": "test_name_hi"}.get(lang, "test_name")
    spoken = [r.get(key) or r.get("test_name") or n for n, r in results if r.get("found")]
    return test_availability.availability_reply(spoken, lang), {"test_name": found[0]}


def _not_a_test_sentence(result: dict, lang: str) -> str | None:
    """Owner, 2026-10-27: "কিছু না খেটে দিনকাল" (nonsense) was answered "there is no test called 'কিছু না খেটে দিনকাল'" -- the model had put the WHOLE sentence in test_name. When the catalogue finds nothing, offers
    nothing close, and the caller's sentence has no test, price or report word in it at all, the caller did not ask about a test: say the line was not caught instead of inventing a test question."""
    text = _turn_text.get()
    if not text or len(text.split()) < 3 or result.get("found") or result.get("ambiguous"):  # a bare one- or two-word answer to "which test?" is left to the "no such test" reply
        return None
    if result.get("did_you_mean_bn") or result.get("did_you_mean") or has_test_context(text):
        return None
    return phrase("reask_generic", lang)


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
        # Real call 2026-10-13: "uric acid ar cbc test total koto hobe" (more than one test named, a
        # combined TOTAL asked for) got only the last-named test's single price, with no total at all --
        # the LLM already extracts every test named in a turn into "test_names" for book_test (agent/
        # llm.py); doing the same real per-test lookup here, one call each, and summing what was actually
        # found (never an estimate) is the fix. A single named test keeps the existing path below,
        # unchanged, including its own suggestion-tracking for a one-name "did you mean" follow-up.
        # Real call 2026-10-13: three tests named in one noisy/garbled turn ("লিপিড প্রোফাইল সী বী সী
        # টেস্ট এবং ... জামটা কৌতো") came back from the model with ONE of the three put in the singular
        # "test_name" and the other two in the plural "test_names" -- reproduced live against the pod's
        # own model (not a one-off: it is NOT consistent across identical calls either, model sampling
        # variance). Reading only "test_names" when it is a list of 2+ silently dropped whichever test the
        # model happened to put in "test_name" instead -- no price, no "not found" either, it simply never
        # appeared. Merging both fields here, deduplicated, means no test named this turn can be dropped
        # regardless of which of the two fields the model chose to put it in.
        test_names = list(slots.get("test_names") or [])
        single = slots.get("test_name")
        if single and single not in test_names:
            test_names.append(single)
        test_names = [n for n in test_names if is_plausible_test_name(n)]  # never look up "র" or "ওই" as a test
        # Real call 2026-10-17: "all three together" but the recogniser lost the CBC entirely. When the caller said how many tests they meant
        # and fewer were understood, say what was caught and ask for all the names again; nothing is priced from a partial list.
        meant = multi_test_confirm.expected_count(_turn_text.get() or "")
        if meant and test_names and len(test_names) < meant:
            _short = multi_test_confirm.shortfall([(n, await _tools.get_test_rate(n)) for n in test_names], meant, lang)
            if _short:
                return _short
        if len(test_names) > 1:
            results = [(name, await _tools.get_test_rate(name)) for name in test_names]
            meant_short = multi_test_confirm.shortfall(results, meant, lang)
            if meant_short:
                return meant_short
            # Real call 2026-10-17: names the recogniser garbled ("ক্রিকাসিড", "থাইরাইড টেস্ট") have sound-alikes the clinic knows. They are
            # READ BACK once, and priced only after a yes (a near match suggests, it never resolves).
            confirm = multi_test_confirm.plan(results, lang)
            if confirm is not None and _suggested.get() is None:
                question, canonical = confirm
                _suggested.set({"intent": "test_rate", "slot": "test_names", "slots": {}, "name": canonical})
                return question
            _priced_names.set((_priced_names.get() or []) + [n for n, r in results if r.get("found")])
            return _done_here(_turn_text.get(), lang, _found_names(results, lang)) + await _multi_reply(results, lang)
        if not slots.get("test_name"):
            return None
        result = await _tools.get_test_rate(slots["test_name"])
        if _not_a_test_sentence(result, lang) is not None:
            return _not_a_test_sentence(result, lang)
        _note_suggestion(intent, "test_name", slots, result)
        if result.get("found"):
            _priced_names.set((_priced_names.get() or []) + [slots["test_name"]])
        if not result.get("found") and _is_scan_name(slots["test_name"]):
            return await _scan_unlisted_reply(lang)
        reply = _done_here(_turn_text.get(), lang, _found_names([(slots["test_name"], result)], lang)) + test_rate_reply(slots, result, lang)
        if result.get("found") and result.get("prescription_required"):
            _rx_asked.set(slots["test_name"])  # owner 2026-10-18: a test that needs a prescription -> ask whether the caller has one
            reply = f"{reply} {phrase('rx_ask', lang)}"
        return reply
    if intent == "doctor_availability":
        if not slots.get("doctor_name"):
            return None
        result = await _tools.get_doctor_availability(slots["doctor_name"], slots.get("date"))
        _note_suggestion(intent, "doctor_name", slots, result)
        _named_surname_only = (
            bool(result.get("found"))
            and _fast_path is not None
            and _text_names_a_doctor(_turn_text.get() or "", lang)  # the doctor is in THIS sentence (not carried over from the last turn: that one is just answered)
            and not _fast_path.catalogue.given_name_said(_turn_text.get() or "", result.get("doctor_name"))
        )
        if not result.get("found") and not result.get("ambiguous"):
            # KCD-lead-A: "Orthopaedics Doctor" is a department the model put in the doctor slot -- not "no such doctor". Any department in the
            # clinic's database is found by its own name or short form; the answer is its real doctors (never invented), exactly as department_query's.
            # Owner, 2026-10-29: the two lookups below were asked one after the other; they do not depend on each other, so they are asked together (the second is dropped when the first answers).
            _dept_task = asyncio.ensure_future(_tools.department_doctors(slots["doctor_name"], lang))
            _lay_task = asyncio.ensure_future(_route_dept(str(slots["doctor_name"]), lang))
            try:
                dept = await _dept_task
            except ToolCallError:
                dept = {}
            if dept.get("matched"):
                _lay_task.cancel()
                _route_found.set({"dept": dept["department_name"], "names": dept.get("doctors_today") or dept.get("doctors") or []})
                return department_route_reply(_dept_labelled(dept, lang), "", lang)
            # owner, 2026-10-26: "ইন্ট ডাক্তার" (the recogniser's "ENT doctor") is a lay doctor phrase, not a name: the clinic's lay-phrase routing finds its department
            try:
                lay = await _lay_task
            except ToolCallError:
                lay = {}
            if lay.get("matched"):
                _route_found.set({"dept": lay["department_name"], "names": lay.get("doctors_today") or lay.get("doctors") or []})
                return department_route_reply(_dept_labelled(lay, lang), "", lang)
        if result.get("found"):
            # sourav, 2026-10-14: if the doctor does NOT sit on the asked day the agent offered the next sitting day, so "book him that day"
            # means THAT day, not the day he is off.
            _de_date = slots.get("date") or result.get("date")
            if result.get("available") is False and result.get("next_available_date"):
                _de_date = result["next_available_date"]
            _doctor_enquiry_found.set({"doctor_name": slots["doctor_name"], "date": _de_date, "window": result.get("chamber_hours")})
        _reply = doctor_availability_reply(slots, result, lang)
        if result.get("found") and _named_surname_only:
            _reply = _confirm_surname_match(slots, result, _reply, lang)
        elif result.get("found") and _doctor_from_context.get():
            _reply = _pronoun_reply(slots, result, _reply, lang)
        return _reply
    if intent == "test_prep":
        if not slots.get("test_name"):
            return None
        result = await _tools.get_test_prep(slots["test_name"], lang)
        if _not_a_test_sentence(result, lang) is not None:
            return _not_a_test_sentence(result, lang)
        _note_suggestion(intent, "test_name", slots, result)
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
        result = await _route_dept(symptom, lang)
        if result.get("matched"):
            _route_found.set({"dept": result["department_name"], "names": result.get("doctors_today") or result.get("doctors") or []})
            if not result.get("doctors_today") and (result.get("doctors") or []):
                # Owner, 2026-10-24: nobody of the department sits today -- say who sits NEXT and when, and offer the appointment, instead of "tell me whom you mean"
                offer = await _next_sitting_offer(result["department_name"], _dept_label(result["department_name"], lang), lang)
                if offer is not None:
                    _next_sit_offer.set(offer)
                    return offer["say"]
        if result.get("matched") and result.get("doctors_today"):
            # Owner, 2026-10-28: "আজ পেডিয়াট্রিক্স বিভাগে অনির্বাণ মন্ডল আছেন" -- and WHEN he sits (the hours from the schedule), not only that he is there
            try:
                today_iso = slot_proposal.clinic_now().date().isoformat()
                sitting = (await _tools.department_doctors(result["department_name"], lang, on_date=today_iso)).get("sitting") or []
            except ToolCallError:
                sitting = []
            if sitting:
                return doctor_search.sitting_list_reply(sitting, _when_phrase(today_iso, None, lang), _dept_label(result["department_name"], lang), lang)
        return department_route_reply(_dept_labelled(result, lang), symptom, lang)
    return None


def _mentions_abdomen(test_name: str) -> bool:
    return "abdomen" in test_name.lower()


async def _queue_prescription_callback(session: CallSession, reason: str, text: str, lang: str) -> None:
    """Every time the caller is asked to send a prescription over WhatsApp, a callback is queued with a factual
    summary (agent/call_summary.py) so the human colleague who opens the photo has the call's context next to it --
    KCD-lay-terms. Only queued when a phone is actually known (session.identity.phone): there is no automatic
    caller-ID in this stack (see _handoff_to_human's own docstring), so without one there is nothing to call back on,
    and a tool failure here must never interrupt the call -- the caller was already told what to do."""
    phone = session.identity.phone
    if not phone:
        return
    summary = build_call_summary(reason, lang=lang, transcript=text, phone=phone, call_id=session.call_id)
    try:
        await _tools.request_callback(phone, session.call_id, "asap", reason, summary)
    except ToolCallError as e:
        logger.warning("[%s] could not queue prescription callback (%s): %s", session.call_id, reason, e)


async def _handle_lay_term(session: CallSession, category: str, text: str, lang: str, ask_first: bool = True) -> None:
    """A rural lay term for a test CATEGORY (agent/lay_terms.py) -- "রক্ত পরীক্ষা", "পেটের ছবি", "মাথার ছবি" -- answered
    from the REAL catalogue, never a hardcoded test name. Deterministic, no model call. See agent/lay_terms.py's
    module docstring for why HEAD never names a specific test: the catalogue has none today, and naming one that
    does not exist would be exactly the fabricated fact CLAUDE.md's truth boundary exists to prevent."""
    reason = f"lay_term_{category}"
    # Owner, 2026-10-17: "peter chhobi" / "mathar chhobi" are ASKED back from the catalogue ("do you mean the abdomen ultrasound?") rather than answered with a price
    if category in (ABDOMEN, HEAD) and await _answer_body_imaging(session, text, lang):
        return
    if category == BLOOD and ask_first:
        # owner, 2026-10-26: reading out every blood test is too long and loses the caller; ask which one, and send a caller who does not know to the prescription photo
        session.awaiting_blood_test = session.turn_count
        await _speak(session, phrase("ask_which_blood_test", lang), lang)
        return
    if category == BLOOD:
        cat = await _cached_catalogue()
        names = sorted({t["name"] for t in cat.get("tests", []) if t.get("sample_type") == "Blood"})
        reply = blood_test_list_reply(names, lang)
    elif category == ABDOMEN:
        cat = await _cached_catalogue()
        matches = sorted(
            t["name"]
            for t in cat.get("tests", [])
            if t.get("sample_type") == "Imaging" and _mentions_abdomen(t["name"])
        )
        if matches:
            result = await _tools.get_test_rate(matches[0])
            reply = test_rate_reply({"test_name": matches[0]}, result, lang)
        else:  # the catalogue changed and no longer has an abdomen imaging test: honest, not a guess
            reply = phrase("head_imaging_not_listed", lang)
    else:  # HEAD
        reply = phrase("head_imaging_not_listed", lang)
        reply = f"{reply} {await _imaging_listing(lang)}".strip()
    await _speak(session, f"{reply} {phrase('send_prescription_whatsapp', lang)}", lang)
    await _queue_prescription_callback(session, reason, text, lang)


async def _handle_test_category_answer(session: CallSession, category: str, text: str, lang: str) -> None:
    """OWNER'S INSTRUCTION, 2026-10-12: the caller's answer to "is it blood, urine, or imaging?" (asked when
    they said they do not know the test's name at all -- see the dispatch site next to
    agent.lay_terms.expresses_not_knowing_test_name). BLOOD/ABDOMEN/HEAD are handled EXACTLY as they already
    are for a direct lay-term mention (_handle_lay_term above) -- reused outright, not reimplemented, so there
    is only ever one place that decides what each of those three says. URINE/IMAGING are new: same "list the
    real catalogue, never invent a name" discipline, generalised (agent.reply_templates.category_test_list_reply)
    rather than duplicating blood_test_list_reply for each one."""
    if category in (BLOOD, ABDOMEN, HEAD):
        await _handle_lay_term(session, category, text, lang, ask_first=False)  # they already said they do not know the name: the list, not another question
        return
    cat = await _cached_catalogue()
    sample_type = "Urine" if category == URINE else "Imaging"
    names = sorted({t["name"] for t in cat.get("tests", []) if t.get("sample_type") == sample_type})
    reply = category_test_list_reply(names, category, lang)
    await _speak(session, f"{reply} {phrase('send_prescription_whatsapp', lang)}", lang)
    await _queue_prescription_callback(session, f"lay_term_{category}", text, lang)


def _bare_doctor_name(name: str) -> str:
    return " ".join(w for w in str(name or "").split() if w.lower().strip(".") not in _DOCTOR_TITLES).strip()


async def _sits_start(st, date_iso: str, info: dict) -> str | None:
    """When the doctor sits on `date_iso` (start of the chamber hours, "HH:MM"), kept on the booking for the arrival answers. From the answer already in hand
    for that day, else one more lookup for the other day."""
    hours = info.get("chamber_hours") if info.get("date") == date_iso else None
    if not hours:
        with contextlib.suppress(Exception):
            hours = (await _tools.free_slots(st.slots["doctor_name"], date_iso)).get("chamber_hours")
    start = hours.split("-")[0] if isinstance(hours, str) and "-" in hours else None
    if start:
        st.slots["_sits_start"] = start
    return start


async def _propose_slot(session: CallSession, st, lang: str) -> bool:
    """Choose the time for the caller from the doctor's free slots and say it (agent/slot_proposal.py). True when a time was set and spoken.
    A time the caller already gave stays if it is a real slot of that day; one that is not (\"12:00\" for \"between ten and twelve\", the closing time) is
    replaced. Any failure leaves the booking exactly as it was (the old question then applies)."""
    try:
        info = await _tools.free_slots(st.slots["doctor_name"], st.slots["date"])
    except Exception as e:  # noqa: BLE001 - the time proposal is an aid; any failure leaves the booking exactly as it was
        logger.warning("[%s] free_slots failed (%s)", session.call_id, e)
        return False
    if not info.get("found"):
        return False
    given = st.slots.get("time_slot")
    if not info.get("sits") or not info.get("valid_slots"):
        # Real call 2026-10-17: the date had come out as TODAY, a day this doctor does not sit, so no time was proposed and the caller was asked for one four
        # times. The doctor's next sitting day is offered instead, with its first free slot -- the caller is never asked for a time.
        nxt = info.get("next") or {}
        if not (nxt.get("date") and nxt.get("time_slot")):
            if info.get("on_leave") or not info.get("valid_slots"):
                # nothing at all to offer for this doctor: say so and ask the caller to call again about the next date (owner, 2026-10-18)
                session.booking = None
                await _speak(session, slot_proposal.call_again(lang), lang)
                return True
            return False
        key = {"bn": "doctor_name_bn", "hi": "doctor_name_hi"}.get(lang, "doctor_name")
        doctor = _bare_doctor_name(info.get(key) or info.get("doctor_name") or st.slots["doctor_name"])
        asked_day = st.slots["date"]
        st.slots["date"], st.slots["time_slot"] = nxt["date"], nxt["time_slot"]
        st.slots["_time_ok"] = "pending"
        st.slots["_sits_told"] = "1"
        sits_next = await _sits_start(st, nxt["date"], info)
        await _speak(session, slot_proposal.sits_statement(doctor, nxt["date"], sits_next or nxt["time_slot"], lang, moved=True, asked_date=asked_day), lang)
        return True
    if given and given in (info.get("free_slots") or []):
        st.slots["_time_explicit"] = "1"  # the caller named this time: it is the one told back
        st.slots["_time_ok"] = "yes"
        return False  # the caller named a time that is free: the normal hold secures it
    slot = slot_proposal.pick(info.get("free_slots") or [], st.slots["date"], skip=st.slots.get("_time_skip") or ())
    new_date = None
    if not slot:
        nxt = info.get("next") or {}
        if not (nxt.get("date") and nxt.get("time_slot")):
            # nothing free on this day and no later day with a slot: the caller is asked to call again about the next date (owner, 2026-10-18)
            session.booking = None
            await _speak(session, slot_proposal.call_again(lang), lang)
            return True
        # nothing left on that day (hours over / all taken): say so and offer the next day that has a slot
        slot, new_date = nxt["time_slot"], nxt["date"]
    key = {"bn": "doctor_name_bn", "hi": "doctor_name_hi"}.get(lang, "doctor_name")
    doctor = _bare_doctor_name(info.get(key) or info.get("doctor_name") or st.slots["doctor_name"])
    st.slots["time_slot"] = slot
    sits = await _sits_start(st, new_date or st.slots["date"], info)
    if new_date:
        # the day had to move (nothing left on the asked day): that IS a question for the caller
        st.slots["_time_ok"] = "pending"
        st.slots["_sits_told"] = "1"
        # KCD-lead-6: remember the day that was actually full (not "doesn't sit" -- `info.get("sits")` was True
        # here), so that if the caller declines this moved day and wants THAT day anyway, department
        # alternatives can be offered for it instead of silently re-deriving the same moved day again.
        st.slots["_day_full_from"] = st.slots["date"]
        st.slots["date"] = new_date
        await _speak(session, f"{slot_proposal.none_left(lang)} {slot_proposal.sits_statement(doctor, new_date, sits or slot, lang)}", lang)
        return True
    # Owner, 2026-10-18: the day is the caller's own and the doctor sits that day. Nothing is asked about the time here: the slot is held quietly, the name, the number,
    # the age and the address are asked one at a time, and the day / hour / "come with one to two hours in hand" is the last question (agent/slot_proposal.arrival_question).
    # Real call 2026-10-18: "কালকে ডাক্তার বিশ্বজিৎ সেনের অ্যাপয়েন্টমেন্ট লাগবে, কালকে কি বসছেন উনি" -- asked in one breath whether he sits, and the AI went straight to
    # "your name?". The caller is told FIRST whether the doctor sits that day (or when he next sits) and asked whether to book it -- unless they were already told in this call.
    ctx = getattr(session, "last_doctor_enquiry", None)
    told = st.slots.get("_sits_told") or (
        booking_carryover.recent(ctx, getattr(session, "turn_count", 0)) and (ctx or {}).get("date") == st.slots["date"]
    )
    if not told:
        st.slots["_sits_told"] = "1"
        st.slots["_time_ok"] = "pending"
        await _speak(session, slot_proposal.sits_statement(doctor, st.slots["date"], sits or slot, lang), lang)
        return True
    st.slots["_time_ok"] = "yes"
    return False


async def _offer_department_alternatives(session: CallSession, st, lang: str, date_iso: str, doctor_name: str) -> bool:
    """KCD-lead-6: the caller's own doctor was full on `date_iso` and the caller wants that day anyway -- find
    other doctors in the same department who actually have a free slot that day and offer them. True (and
    spoken) only when at least one real alternative was found; False leaves the booking untouched so the
    EXISTING "no slot, ask for another day" behaviour applies exactly as before this function existed."""
    try:
        info = await _tools.department_alternatives(doctor_name, date_iso, lang)
    except ToolCallError as e:  # noqa: BLE001 - this is an aid; any failure falls back to the old behaviour
        logger.warning("[%s] department_alternatives failed (%s)", session.call_id, e)
        return False
    alternatives = (info or {}).get("alternatives") or []
    if not alternatives:
        return False
    doctor = _bare_doctor_name(doctor_name)
    names = [_bare_doctor_name(a["name"]) for a in alternatives if a.get("name")]
    if not names:
        return False
    for k in ("_time_ok", "_sits_told", "time_slot", "_time_explicit", "_day_full_from"):
        st.slots.pop(k, None)
    st.hold_token, st.hold_doctor_id = None, None
    st.slots["date"] = date_iso
    st.slots["doctor_name"] = None  # the caller names one of the offered doctors next; the usual doctor_name pipeline picks it up
    await _speak(session, slot_proposal.alternative_doctors_statement(doctor, names, lang), lang)
    return True


_THESE = ("এইগুলোই", "এগুলোই", "এইগুলো", "এগুলো", "ওইগুলো", "ওগুলো", "এই টেস্ট", "এই টেস্টগুলো", "এইটাই", "এটাই", "सब यही", "यही", "ये ही", "यही सब", "ये सब", "वही सब",
          "these", "those", "these ones", "that's right", "thats right", "exactly", "all of them")


async def _resume_after_unclear(session: CallSession, text: str, lang: str) -> bool:
    """The turn made no sense on its own, but a question was just asked / tests were just named. Use the call's context:
    (1) "এইগুলোই" / "these ones" right after a total of named tests = yes, give that total;
    (2) any other unclear reply while a yes/no question is open = ask the SAME question again (kept for two misses)."""
    lapsed = getattr(session, "lapsed_pending", None)
    multi = getattr(session, "last_multi", None)
    t = text or ""
    low = t.lower()
    if multi and session.turn_count - multi[1] <= 4 and any(w in low for w in _THESE):
        try:
            names = multi[0]
            results = [(n, await _tools.get_test_rate(n)) for n in names]
        except ToolCallError:
            return False
        session.last_multi = None
        session.lapsed_pending = None
        _priced_names.set([n for n, r in results if r.get("found")])
        await _speak(session, await _multi_reply(results, lang), lang)
        return True
    if lapsed and session.turn_count - lapsed[1] <= 2:
        pending, _turn = lapsed
        asked = getattr(session, "pending_question", None)
        if asked and session.turn_count - asked[1] <= 4 and getattr(session, "reasked_pending", 0) < 2:
            session.reasked_pending = getattr(session, "reasked_pending", 0) + 1
            session.pending_entity = pending  # still waiting for the yes
            session.lapsed_pending = (pending, session.turn_count)
            await _speak(session, asked[0], lang)
            return True
    return False


async def _answer_body_imaging(session: CallSession, text: str, lang: str) -> bool:
    """"buker chhobi korbo" / "haater chhobi tulbo" (agent/body_imaging.py): the caller names a body part, not a test. Ask, from the catalogue's own
    imaging tests for that part, which one they mean -- or say plainly that none is listed. Never names a scan the clinic lacks."""
    region = body_imaging.detect(text)
    if region is None:
        return False
    try:
        cat = await _cached_catalogue()
        names = body_imaging.candidates(region, cat.get("tests", []))
        key = {"bn": "aliases_bn", "hi": "aliases_hi"}.get(lang)
        tests = {t["name"]: t for t in cat.get("tests", [])}
        spoken = []
        for n in names:
            alias = next((a for a in (tests[n].get(key) or []) if a), None) if key else None
            spoken.append(alias or n.split("(")[0].strip())
    except Exception as e:  # noqa: BLE001 - an aid; the normal path takes over
        logger.warning("body imaging skipped: %s", e)
        return False
    if not names and region == "head":
        return False  # no head imaging test exists: the existing lay-term answer (not on our list + prescription route) stands
    if names and body_imaging.asks_price(text):
        # Owner, 2026-10-18: "পেটের ছবি তুলব, কোন টেস্ট মনে নাই, কত টাকা পড়বে": the tests there are for that part, each with its rate, THEN the prescription line
        items = []
        try:
            for n, sp in zip(names[:4], spoken):
                r = await _tools.get_test_rate(n)
                if r.get("found"):
                    items.append((sp, r.get("rate_inr")))
        except ToolCallError:
            items = []
        if items:
            await _speak(session, f"{body_imaging.listing(region, items, lang)} {phrase('send_prescription_whatsapp', lang)}", lang)
            await _queue_prescription_callback(session, f"body_imaging_{region}", text, lang)
            return True
    question = body_imaging.question(region, spoken, lang)
    if names:
        # a yes (or the caller naming one of them) goes on to that test's price, through the usual pending-confirmation replay
        data = {"intent": "test_rate", "slots": {"test_name": names[0]}, "direct_reply_bn": None, "secondary_intent": None, "secondary_slots": None}
        if len(names) == 1:
            session.pending_entity = entity_text.PendingEntity("test_rate", "test_name", names[0], data)
        else:
            session.pending_enquiry = ("test_rate", session.turn_count)
        await _speak(session, _with_resume(session, question, lang), lang)
    else:
        # no test of its own for that part: say what pictures the clinic DOES do, then the prescription route
        await _speak(session, f"{question} {await _imaging_listing(lang)}{phrase('send_prescription_whatsapp', lang)}", lang)
        await _queue_prescription_callback(session, f"body_imaging_{region}", text, lang)
    return True


async def _imaging_listing(lang: str) -> str:
    """"The picture tests we do here are: ..." from the catalogue's own Imaging tests, with a trailing space; empty when there are none or the catalogue is unavailable."""
    try:
        cat = await _cached_catalogue()
        key = {"bn": "aliases_bn", "hi": "aliases_hi"}.get(lang)
        spoken = []
        for t in cat.get("tests", []):
            if t.get("sample_type") != "Imaging":
                continue
            alias = next((a for a in (t.get(key) or []) if a), None) if key else None
            spoken.append((alias or t["name"].split("(")[0].strip(), None))
        return body_imaging.listing(None, spoken[:6], lang) + " " if spoken else ""
    except Exception:  # noqa: BLE001 - advisory
        return ""


# ===================================================================================================================================
# Owner's 2026-10-18 additions: answers the owner dictated, finding a doctor by sex / department / time, relationship -> sex, unanswered calls.
# ===================================================================================================================================

_relationship_cache: dict = {"rules": None, "at": 0.0}


async def _relationship_rules() -> list[dict]:
    """The relationship words and the sex each implies (clinic-api relationship_rules), cached for ten minutes. Empty when the clinic cannot be asked."""
    now = time.monotonic()
    if _relationship_cache["rules"] is None or now - _relationship_cache["at"] > 600:
        try:
            _relationship_cache["rules"] = (await _tools.relationships()).get("rules") or []
        except Exception as e:  # noqa: BLE001 - advisory
            logger.warning("relationship rules unavailable: %s", e)
            _relationship_cache["rules"] = _relationship_cache["rules"] or []
        _relationship_cache["at"] = now
    return _relationship_cache["rules"]


_FOR_CUES = ("জন্য", "এর", "র জন্য", "কে ", "for", "के लिए", "का ", "की ", "ke liye", "er jonno", "এর জন্য")


def _sex_of_relationship(rules: list[dict], text: str, require_for: bool = True) -> tuple[str | None, str | None]:
    """(relationship, sex) when the sentence says who the patient is ("for my mother", "আমার মায়ের জন্য", "मेरी माँ के लिए"). sex is "" for a relationship that
    does not say one (child, friend). (None, None) when no relationship word is there. Whole words only."""
    t = " " + " ".join(re.sub(r"[.,;:!?\"'()।॥]+", " ", (text or "").lower()).split()) + " "
    if require_for and not any(c in t for c in _FOR_CUES) and "my " not in t and "আমার" not in t and "मेरी" not in t and "मेरे" not in t:
        return None, None
    best = (None, None, 0)
    for r in rules:
        for kw in [*r.get("keywords_bn", []), *r.get("keywords_hi", []), *r.get("keywords_en", [])]:
            k = kw.lower()
            hit = (f" {k} " in t) if k.isascii() else any(tok == k or tok.startswith(k) for tok in t.split())
            if hit and len(k) > best[2]:
                best = (r["relationship"], r["sex"], len(k))
    return (best[0], best[1]) if best[0] else (None, None)


async def _note_patient_relationship(session: CallSession, st, text: str, slots: dict) -> None:
    """A booking made "for my mother" is for a woman: the relationship the caller said (model slot or the sentence) is looked up in the clinic's
    relationship table and the patient's sex kept on the booking. Never overwrites a sex already known."""
    if st.slots.get("patient_sex"):
        return
    rules = await _relationship_rules()
    if not rules:
        return
    rel_text = (slots or {}).get("relationship") or st.slots.get("relationship")
    rel, sex = _sex_of_relationship(rules, rel_text, require_for=False) if rel_text else (None, None)
    if rel is None:
        rel, sex = _sex_of_relationship(rules, text)
    if rel:
        st.slots.setdefault("relationship", rel)
        if sex:
            st.slots["patient_sex"] = sex
            logger.info("[%s] booking for the %s -> patient sex %s", session.call_id, rel, sex)


_NUM_IN_TEXT = re.compile(r"\d")


def _sitting_start_and_doctor(session: CallSession, text: str, lang: str) -> tuple[str | None, str | None]:
    """(when the doctor sits, from; the doctor's name when speakable). The hours come from the doctor the caller was just told about, from the booking
    being made, or from a time the caller says in the sentence ("the doctor sits at 10")."""
    start = None
    doctor = None
    ctx = session.last_doctor_enquiry or {}
    window = ctx.get("window")
    if isinstance(window, str) and "-" in window:
        start = window.split("-")[0].strip()
    if ctx.get("doctor_name") and not re.search(r"[A-Za-z]", str(ctx["doctor_name"])):
        doctor = str(ctx["doctor_name"])
    b = session.booking
    if not start and b is not None and b.slots.get("_sits_start"):
        start = b.slots["_sits_start"]
    said = booking_capture.read_when(text, True).time
    if said and (not start or _NUM_IN_TEXT.search(text) or True):
        start = said if not (window and start) else start
    return start, doctor


async def _visit_fee_reply(session: CallSession, text: str, lang: str) -> str | None:
    """"ডাক্তারের ভিজিট কত?" -- the figure is read from the doctors table. The doctor is the one named in the sentence, else the one just discussed."""
    name = None
    if _fast_path is not None:
        with contextlib.suppress(Exception):
            doc = _fast_path._named_doctor(fast_path_normalize(text), language_of_script(text, lang), strong=False)  # noqa: SLF001
            if doc is not None:
                name = doc[1] or doc[0]
    if not name:
        ctx = session.last_doctor_enquiry or {}
        name = ctx.get("doctor_name") or (session.booking.slots.get("doctor_name") if session.booking is not None else None)
    if not name:
        return {
            "bn": "কোন ডাক্তারের ভিজিট জানতে চান?",
            "hi": "किस डॉक्टर की विज़िट फ़ीस जानना चाहते हैं?",
            "en": "Which doctor's visit fee would you like to know?",
        }.get(lang, "")
    try:
        result = await _tools.doctor_visit(name)
    except ToolCallError:
        return None
    if result.get("ambiguous"):
        return None
    if not result.get("found"):
        return None
    who = {"bn": result.get("doctor_name_bn"), "hi": result.get("doctor_name_hi"), "en": result.get("doctor_name")}.get(lang) or ""
    fee = result.get("visit_fee_inr")
    if fee is None:
        return None
    return {
        "bn": f"{who} এর ভিজিট {fee} টাকা।",
        "hi": f"{who} की विज़िट फ़ीस {fee} रुपये है।",
        "en": f"{who}'s visit fee is {fee} rupees.",
    }.get(lang)


async def _answer_owner_faq(session: CallSession, text: str, lang: str) -> bool:
    """Which branch this is, whether Swasthya Sathi / Ayushman Bharat cover the visit, how far from the stations, when to come with a token number, a
    serial that cannot be pushed up, and what a doctor's visit costs (agent/owner_answers.py) -- fixed or table-read answers, no model."""
    try:
        spoken = None
        if owner_answers.asks_payment_destination(text):
            session.pending_enquiry = None
            spoken = owner_answers.payment_destination_reply(lang)
        if not spoken and owner_answers.asks_link_to_other_number(text):
            session.pending_enquiry = None  # the "confirmation number?" question that was open is no longer the point
            spoken = owner_answers.link_to_other_number_reply(lang)
        if not spoken and owner_answers.asks_payment_link_timing(text):
            spoken = owner_answers.payment_link_timing_reply(lang)
        if not spoken and session.booking is not None and getattr(session.booking, "action", "") == "book_appointment" and owner_answers.asks_bare_cost(text):
            spoken = await _visit_fee_reply(session, text, lang)  # "কত লাগবে" in the middle of a doctor booking: that doctor's visit fee
        spoken = spoken or owner_answers.branch_identity(text, lang) or owner_answers.scheme_cover(text, lang)
        if not spoken and owner_answers.asks_token_push(text):
            spoken = owner_answers.token_push_reply(lang)
        if not spoken:
            key = owner_answers.distance_from(text)
            if key:
                address = None
                with contextlib.suppress(Exception):
                    address = (await _tools.get_faq("location", lang)).get("answer")
                spoken = owner_answers.distance_reply(key, lang, address)
        if not spoken and owner_answers.asks_how_to_reach(text):
            # "আপনাদের ওখানে যাবো কিভাবে": the clinic's address, from the clinic's own FAQ row (the same one "where are you" gets)
            address = None
            with contextlib.suppress(Exception):
                address = (await _tools.get_faq("location", lang)).get("answer")
            spoken = address or owner_answers._ADDRESS_FALLBACK.get(lang, owner_answers._ADDRESS_FALLBACK["en"])  # noqa: SLF001
        if not spoken and owner_answers.asks_token_arrival(text):
            start, doctor = _sitting_start_and_doctor(session, text, lang)
            spoken = owner_answers.token_arrival_reply(owner_answers.token_number(text), start, lang, doctor)
        if not spoken and owner_answers.asks_visit_fee(text):
            spoken = await _visit_fee_reply(session, text, lang)
    except Exception as e:  # noqa: BLE001 - these are aids; the turn goes on to the normal path
        logger.warning("[%s] owner answers skipped: %s", session.call_id, e)
        return False
    if not spoken:
        return False
    await _speak(session, _with_resume(session, spoken, lang), lang)
    return True


_DEPT_LABEL = {
    "General Medicine": ("জেনারেল মেডিসিন", "जनरल मेडिसिन", "General Medicine"),
    "Cardiology": ("কার্ডিওলজি", "कार्डियोलॉजी", "Cardiology"),
    "Gynaecology & Obstetrics": ("গাইনোকোলজি", "गायनेकोलॉजी", "Gynaecology"),
    "Orthopaedics": ("অর্থোপেডিক্স", "ऑर्थोपेडिक्स", "Orthopaedics"),
    "ENT": ("ই এন টি", "ई एन टी", "ENT"),
    "Dermatology": ("ডার্মাটোলজি", "डर्मेटोलॉजी", "Dermatology"),
    "Paediatrics": ("পেডিয়াট্রিক্স", "पीडियाट्रिक्स", "Paediatrics"),
    "Diabetology & Endocrinology": ("ডায়াবেটিস ও এন্ডোক্রিনোলজি", "डायबिटीज़ और एंडोक्रिनोलॉजी", "Diabetes and Endocrinology"),
    "Dentistry": ("ডেন্টিস্ট্রি", "डेंटिस्ट्री", "Dentistry"),
    "General Surgery": ("জেনারেল সার্জারি", "जनरल सर्जरी", "General Surgery"),
    "Neurology": ("নিউরোলজি", "न्यूरोलॉजी", "Neurology"),
    "Neurosurgery": ("নিউরোসার্জারি", "न्यूरोसर्जरी", "Neurosurgery"),
    "Urology": ("ইউরোলজি", "यूरोलॉजी", "Urology"),
    "Gastroenterology": ("গ্যাস্ট্রোএন্টারোলজি", "गैस्ट्रोएंटेरोलॉजी", "Gastroenterology"),
}


def _dept_labelled(result: dict, lang: str) -> dict:
    """A department-routing result with every department name in the call's own language: the Bengali and Hindi voices drop English letters, and a reply holding "Orthopaedics"
    is blocked whole (real call 2026-10-23). The ambiguous answer's candidates and their doctor lists are labelled too."""
    out = dict(result)
    if result.get("department_name"):
        out["department_name"] = _dept_label(result["department_name"], lang)
    if result.get("candidates"):
        out["candidates"] = [_dept_label(c, lang) for c in result["candidates"]]
    if result.get("doctors_by_department"):
        out["doctors_by_department"] = {_dept_label(k, lang): v for k, v in result["doctors_by_department"].items()}
    return out


def _dept_label(name: str, lang: str) -> str:
    row = _DEPT_LABEL.get(name)
    if not row:
        return name
    return row[{"bn": 0, "hi": 1}.get(lang, 2)]


_WHEN_WORDS = {
    "today": {"bn": "আজ", "hi": "आज", "en": "today"},
    "tomorrow": {"bn": "আগামীকাল", "hi": "कल", "en": "tomorrow"},
}


def _when_phrase(date_iso: str | None, time_hhmm: str | None, lang: str) -> str:
    """"আগামীকাল সকাল এগারোটায়" for the day and time a caller asked for; used in the "who sits then?" answers."""
    parts = []
    if date_iso:
        today = datetime.date.today()
        d = datetime.date.fromisoformat(date_iso)
        if d == today:
            parts.append(_WHEN_WORDS["today"][lang])
        elif d == today + datetime.timedelta(days=1):
            parts.append(_WHEN_WORDS["tomorrow"][lang])
        else:
            parts.append(d.isoformat())
    if time_hhmm:
        h = int(time_hhmm.split(":")[0])
        ph = owner_answers.hour_phrase(h, lang)
        parts.append(owner_answers._bn_case(ph, "loc") if lang == "bn" else ph)  # noqa: SLF001
    return " ".join(parts)


def _resolve_doctor_for_booking(item: dict) -> str:
    return item.get("full_name") or item.get("name") or ""


_WHEN_ASK = ("কবে", "কখন", "kobe", "kokhon", "when", "कब", "कितने दिन")
_NOBODY_LEAD = {
    "bn": "{w} {d} বিভাগের কেউ বসেন না।",
    "hi": "{w} {d} विभाग का कोई डॉक्टर नहीं बैठता।",
    "en": "Nobody from {d} sits {w}.",
}


def _nobody_lead(lang: str, when: str, label: str) -> str:
    """"আজ জেনারেল সার্জারি বিভাগের কেউ বসেন না।" -- the department is NAMED (it used to say "that department")."""
    d = label if label and label != when else {"bn": "ওই", "hi": "उस", "en": "that department"}.get(lang, "that")
    return _NOBODY_LEAD.get(lang, _NOBODY_LEAD["en"]).format(w=when or label, d=d)


def _asks_when(text: str) -> bool:
    low = (text or "").lower()
    return len(low.split()) <= 16 and any(w in low for w in _WHEN_ASK)


async def _next_dept_sitting(text: str, lang: str, sex: str | None, after_iso: str) -> tuple[str, list] | None:
    """The first day within the next week, after `after_iso`, on which anybody of the department the caller named sits: (iso date, the doctors sitting). The clinic's own table, asked day by
    day -- nothing is guessed. None when nobody sits in that week or the clinic API is unavailable."""
    start = datetime.date.fromisoformat(after_iso)
    for i in range(1, 8):
        iso = (start + datetime.timedelta(days=i)).isoformat()
        try:
            r = await _tools.department_doctors(text, lang, sex=sex, on_date=iso)
        except ToolCallError:
            return None
        if r.get("matched") and r.get("sitting"):
            return iso, r["sitting"]
    return None


async def _route_reply(session: CallSession, routed: dict, symptom: str, lang: str) -> str:
    """department_route_reply, except that when nobody of the department sits today the next sitting is told and offered (see _next_sitting_offer) and kept as the open question."""
    if routed.get("matched") and not routed.get("doctors_today") and routed.get("doctors"):
        offer = await _next_sitting_offer(routed["department_name"], _dept_label(routed["department_name"], lang), lang)
        if offer is not None:
            session.dept_pending = {**offer["pending"], "turn": session.turn_count, "say": offer["say"]}
            return offer["say"]
    reply = department_route_reply(_dept_labelled(routed, lang), symptom, lang)
    if routed.get("matched") and len(routed.get("doctors_today") or []) >= 1:
        with contextlib.suppress(ToolCallError):
            today_iso = slot_proposal.clinic_now().date().isoformat()
            sitting = (await _tools.department_doctors(routed["department_name"], lang, on_date=today_iso)).get("sitting") or []
            if len(sitting) >= 2:
                session.dept_pending = {"kind": "pick", "doctors": sitting, "date": today_iso, "time": None, "turn": session.turn_count, "say": reply}
            elif len(sitting) == 1:
                session.dept_pending = {"kind": "confirm_one", "doctor": sitting[0], "date": today_iso, "time": sitting[0].get("start"), "turn": session.turn_count, "say": reply}
    return reply


async def _say_nobody_then_next(session: CallSession, text: str, lang: str, sex: str | None, on_date: str, when_phrase: str, label: str) -> bool:
    """"Nobody from <department> sits <day>. Dr X sits <next day> at <time>. Shall I book that day?" -- or, for several, which of them; the open question is kept (dept_pending). False when
    nobody sits in the following week (the caller then hears the plain 'nobody sits')."""
    nxt = await _next_dept_sitting(text, lang, sex, on_date)
    if not nxt:
        return False
    iso, sit = nxt
    lead = _nobody_lead(lang, when_phrase, label)
    if len(sit) == 1:
        item = sit[0]
        say = f"{lead} {slot_proposal.sits_statement(item['name'], iso, item.get('start') or '', lang)}"
        session.dept_pending = {"kind": "confirm_one", "doctor": item, "date": iso, "time": item.get("start"), "turn": session.turn_count, "say": say}
    else:
        say = f"{lead} {doctor_search.sitting_pick_question([x['name'] for x in sit], _when_phrase(iso, None, lang), lang)}"
        session.dept_pending = {"kind": "pick", "doctors": sit, "date": iso, "time": None, "turn": session.turn_count, "say": say}
    await _speak(session, say, lang)
    return True


async def _next_sitting_offer(dept: str, label: str, lang: str) -> dict | None:
    """"Nobody of <department> sits today. Dr X sits <day> at <time>. Shall I book that day?" (or "Dr X or Dr Y sit <day>; whom?"), from the clinic's own table, with the pending
    question to match. None when nobody sits in the next week."""
    today = slot_proposal.clinic_now().date().isoformat()
    nxt = await _next_dept_sitting(dept, lang, None, today)
    if not nxt:
        return None
    iso, sit = nxt
    lead = {
        "bn": f"আজ {label} বিভাগের কেউ বসছেন না।",
        "hi": f"आज {label} विभाग का कोई डॉक्टर नहीं बैठ रहा।",
        "en": f"Nobody from {label} sits today.",
    }.get(lang, f"Nobody from {label} sits today.")
    if len(sit) == 1:
        item = sit[0]
        say = f"{lead} {slot_proposal.sits_statement(item['name'], iso, item.get('start') or '', lang)}"
        return {"say": say, "pending": {"kind": "confirm_one", "doctor": item, "date": iso, "time": item.get("start")}}
    say = f"{lead} {doctor_search.sitting_pick_question([s['name'] for s in sit], _when_phrase(iso, None, lang), lang)}"
    return {"say": say, "pending": {"kind": "pick", "doctors": sit, "date": iso, "time": None}}


# Owner, 2026-10-28: "until the user specifically changes context, it must not be lost by the algorithm". The department just discussed and the doctors just offered stay the subject for the rest of the call;
# they are replaced only when the caller names another department or doctor (a new route result below) -- never by the number of turns that passed.
_PICK_MEMORY_TURNS = 10**6


async def _dept_search_turn(session: CallSession, text: str, lang: str) -> dict | bool:
    """Finding the doctor from how the caller asks (agent/doctor_search.py):
      * the answer to "neurologist or neurosurgeon?" / "which of these doctors?" asked a moment ago;
      * "neuro" / "euro" said alone: ask which, never guess;
      * "a lady gynaecologist": the department's doctors of that sex only;
      * "whoever is in X department, tomorrow at 11, book me": one doctor -> booked into the normal flow; several -> asked which.
    Returns True when the turn was answered here, a dict (an intent result for the booking flow) when a single doctor was found, False to carry on."""
    if _tools is None:
        return False
    # 1. the answer to a question asked here
    pend = getattr(session, "dept_pending", None)
    if pend and session.turn_count - pend["turn"] <= 3 and pend.get("say") and pend["kind"] in ("confirm_one", "pick") and _asks_when(text) and booking_capture.confirm_answer(text, lang) is None:
        # Owner, 2026-10-28: "কালকে কবে বসবে বা পরশু" names ANOTHER day than the one just told: the answer is for that day (the doctor's own availability, with his next sitting when he is off), never today's line again
        if pend["kind"] == "confirm_one":
            _other = booking_capture.read_when(text, False, want_day=True).date
            if _other and _other != pend.get("date"):
                session.dept_pending = None
                return {"intent": "doctor_availability", "slots": {"doctor_name": _resolve_doctor_for_booking(pend["doctor"]), "date": _other}}
        # "কবে আছে, নেক্স্ট কবে পাব, পরে কবে বসছে" after the next sitting was just told: that IS the answer to "when" -- said again, and the question stays open (owner, 2026-10-24)
        pend["turn"] = session.turn_count
        await _speak(session, pend["say"], lang)
        return True
    if pend and session.turn_count - pend["turn"] <= (_PICK_MEMORY_TURNS if pend["kind"] == "pick" else 3):
        session.dept_pending = None
        if pend["kind"] == "confirm":
            chosen = doctor_search.pick_from(text, pend["options"])
            if chosen is not None:
                try:
                    result = await _tools.department_doctors(chosen["dept"], lang, sex=pend.get("sex"))
                except ToolCallError:
                    return False
                if result.get("matched"):
                    await _speak(session, _with_resume(session, (await _route_reply(session, result, "", lang)), lang), lang)
                    return True
            return False
        if pend["kind"] == "confirm_one":
            _names_him = doctor_search.choose_by_name(text, [[pend["doctor"].get("name", ""), pend["doctor"].get("full_name", "")]]) == 0 and len(text.split()) <= 6  # saying the offered doctor's name IS the yes
            if booking_capture.confirm_answer(text, lang) == "yes" or booking_carryover.commits_to_booking(text) or _names_him:  # owner, 2026-10-29: "একটা নাম লেখা যাবে" IS the yes
                item = pend["doctor"]
                session.last_doctor_enquiry = {
                    "doctor_name": _resolve_doctor_for_booking(item), "date": pend["date"],
                    "window": f"{item.get('start')}-{item.get('end')}" if item.get("start") else None, "turn": session.turn_count,
                }  # the sitting was just told: the booking does not tell it again
                return {"intent": "book_appointment", "slots": {"doctor_name": _resolve_doctor_for_booking(item), "date": pend["date"], "time_slot": pend.get("time")}}
            return False
        if pend["kind"] == "pick":
            # owner, 2026-10-26: the choice is read with the Bengali "of" ending on the surname ("সঞ্জয় গুহার"), and an unrelated sentence in between ("কোন ডাক্তার ভালো?") does not
            # make the agent forget the doctors it just offered: the question stays open for several turns (it used to be dropped by the first sentence that did not choose).
            picked = doctor_search.choose_by_name(text, [[d.get("name", ""), d.get("full_name", "")] for d in pend["doctors"]])
            hits = [pend["doctors"][picked]] if picked is not None else []
            if len(hits) != 1:  # "the first one" / "the second one"
                hits = []
                for idx, words in doctor_search._ORDINALS:  # noqa: SLF001
                    if idx < len(pend["doctors"]) and doctor_search._has(doctor_search._norm(text), words) and len(text.split()) <= 4:  # noqa: SLF001
                        hits = [pend["doctors"][idx]]
            if len(hits) != 1:
                session.dept_pending = pend  # still open: the list is remembered for the next sentences
            if len(hits) == 1:
                session.last_doctor_enquiry = {  # the list just said when each of them sits: the booking does not say it again
                    "doctor_name": _resolve_doctor_for_booking(hits[0]), "date": pend.get("date"),
                    "window": f"{hits[0].get('start')}-{hits[0].get('end')}" if hits[0].get("start") else None, "turn": session.turn_count,
                }
                return {
                    "intent": "book_appointment",
                    "slots": {"doctor_name": _resolve_doctor_for_booking(hits[0]), "date": pend.get("date"), "time_slot": pend.get("time")},
                }
            return False
    # 1b. the doctors a department answer just named, one of them chosen ("তাহলে সঞ্জয় গুহার অ্যাপয়েন্টমেন্ট দিন")
    route = getattr(session, "last_route", None)
    if route and session.booking is None and route.get("names") and session.turn_count - route["turn"] <= _PICK_MEMORY_TURNS:
        chosen = doctor_search.choose_by_name(text, [[n] for n in route["names"]])
        if chosen is not None and (booking_carryover.asks_to_book(text) or len(text.split()) <= 6 or _asks_when(text)):
            slots = {"doctor_name": route["names"][chosen], "date": booking_capture.read_when(text, False, want_day=True).date}
            session.last_route = None
            return {"intent": "doctor_availability" if _asks_when(text) and not booking_carryover.asks_to_book(text) else "book_appointment", "slots": slots}
    # 2. neuro / euro: ask which
    try:
        cat = await _cached_catalogue()
        existing = {d.get("department") or d.get("department_name") for d in (cat.get("doctors") or [])} - {None}
    except Exception:  # noqa: BLE001
        existing = None
    if session.booking is None:
        fam = doctor_search.confusable(text, existing or None)
        if fam:
            session.dept_pending = {"kind": "confirm", "options": fam["options"], "turn": session.turn_count, "sex": doctor_search.wanted_sex(text)}
            await _speak(session, doctor_search.question(fam["options"], lang), lang)
            return True
    # 3. a lady / gentleman doctor, and 4. a department at a day and time
    sex = doctor_search.wanted_sex(text)
    when = booking_capture.read_when(text, False, want_day=True)
    wants_when = bool(when.date or when.time)
    wants_book = booking_carryover.asks_to_book(text) or bool(re.search(r"\b(book|appointment)\b|বুক|অ্যাপয়েন্টমেন্ট|अपॉइंटमेंट|बुक", text, re.I))
    if wants_when and not wants_book and not sex and len(text.split()) <= 40:  # was 12: a real caller's sentence ("... একটা জেনারেল মেডিসিন এর একটা ডাক্তার দেখাতে চাই আজকে কাকে পাবো") has 19 words; the department lookup below decides whether a department was really named
        wants_book = True  # owner, 2026-10-28: "आज के कार्डियोलॉजी ..." / "আজ কার্ডিওলজি..." names a department and a day: who sits then (the department lookup below decides whether a department was really named)
    if not sex and not (wants_when and wants_book):
        return False
    if _fast_path is not None:
        with contextlib.suppress(Exception):
            if _fast_path._named_doctor(fast_path_normalize(text), language_of_script(text, lang)) is not None:  # noqa: SLF001
                return False  # a doctor is named: the normal path
    on_date = when.date if wants_when and wants_book else None
    at_time = when.time if wants_when and wants_book else None
    try:
        result = await _tools.department_doctors(text, lang, sex=sex, on_date=on_date, at_time=at_time)
    except ToolCallError:
        return False
    if not result.get("matched"):
        return False
    label = _dept_label(result["department_name"], lang)
    if on_date or at_time:
        sitting = result.get("sitting") or []
        _part = doctor_search.asked_part_of_day(text)
        if _part:
            sitting = [s for s in sitting if doctor_search.part_of_day_of(s.get("start")) == _part]  # owner, 2026-10-28: "আজকে কে বসছে বিকালের দিকে"
        phrase_when = _when_phrase(on_date, at_time, lang)
        if len(sitting) == 1:
            item = sitting[0]
            say = slot_proposal.sits_statement(item["name"], on_date, at_time or item.get("start") or "", lang)
            session.dept_pending = {"kind": "confirm_one", "doctor": item, "date": on_date, "time": at_time, "turn": session.turn_count, "say": say}
            await _speak(session, say, lang)
            return True
        if len(sitting) > 1:
            say = doctor_search.sitting_pick_question([s["name"] for s in sitting], phrase_when, lang)
            session.dept_pending = {"kind": "pick", "doctors": sitting, "date": on_date, "time": at_time, "turn": session.turn_count, "say": say}
            await _speak(session, say, lang)
            return True
        others = []
        if at_time and on_date:
            with contextlib.suppress(ToolCallError):
                again = await _tools.department_doctors(text, lang, sex=sex, on_date=on_date)
                others = [s["name"] for s in again.get("sitting") or []]
        if not others and on_date:
            # Owner, 2026-10-24: when nobody sits on the asked day the agent says WHO sits next and WHEN, and offers the appointment -- it does not ask the caller to name another day.
            nxt = await _next_dept_sitting(text, lang, sex, on_date)
            if nxt:
                iso, sit = nxt
                lead = _nobody_lead(lang, phrase_when, label)
                if len(sit) == 1:
                    item = sit[0]
                    say = f"{lead} {slot_proposal.sits_statement(item['name'], iso, item.get('start') or '', lang)}"
                    session.dept_pending = {"kind": "confirm_one", "doctor": item, "date": iso, "time": item.get("start"), "turn": session.turn_count, "say": say}
                else:
                    say = f"{lead} {doctor_search.sitting_pick_question([s['name'] for s in sit], _when_phrase(iso, None, lang), lang)}"
                    session.dept_pending = {"kind": "pick", "doctors": sit, "date": iso, "time": None, "turn": session.turn_count, "say": say}
                await _speak(session, say, lang)
                return True
        await _speak(session, doctor_search.nobody_sits(phrase_when or label, others, lang), lang)
        return True
    if sex:
        await _speak(session, _with_resume(session, doctor_search.filtered_reply(result, sex, lang, label), lang), lang)
        return True
    return False



# ---- the caller checks a word of the answer: "চুয়াল্লিশ বললেন?" ("did you say forty-four?") -----------------------------------------------------------
_CHECK_CUES = (
    "বললেন", "বলেছেন", "বলছেন", "বললে", "বলেছিলেন", "বলল", "মানে", "বলেছিলে",
    "did you say", "you said", "you mean", "do you mean", "is it", "was it",
    "क्या कहा", "आपने कहा", "कहा आपने", "कहा", "मतलब", "बोला", "बोले",
)
# Owner, 2026-10-27: "কত বললেন" / "kitna bola" -- asking generically what AMOUNT was just said, no guessed number
# of their own (that's spoken_number's job, just above). Combined with _CHECK_CUES already matching ("বললেন"/"did
# you say"), this is specifically "what figure did you give me", not the many other things "কত" alone could mean.
_GENERIC_AMOUNT_CUES = (
    "কত", "কতো", "কত টাকা", "টাকা",
    "koto", "kototaka", "kitna", "kitne", "kitni",
    "कितना", "कितने", "कितनी", "कितना था", "कितने का",
    "how much", "what amount", "what was the amount", "what price",
)
_BN_DIGIT_WORD = ["শূন্য", "এক", "দুই", "তিন", "চার", "পাঁচ", "ছয়", "সাত", "আট", "নয়"]
_HI_DIGIT_WORD = ["शून्य", "एक", "दो", "तीन", "चार", "पाँच", "छह", "सात", "आठ", "नौ"]
_EN_DIGIT_WORD = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
_ANY_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯०१२३४५६७८९", "01234567890123456789")


def _digit_words(n: int, lang: str) -> str:
    table = {"bn": _BN_DIGIT_WORD, "hi": _HI_DIGIT_WORD}.get(lang, _EN_DIGIT_WORD)
    return " ".join(table[int(d)] for d in str(n))


async def _answer_echo(session: CallSession, text: str, lang: str) -> bool:
    """The caller asks back about a WORD of what was just said ("বেনিয়া পুকুর কেয়া?" after the address): found by sound, across scripts (agent/echo_check.py), and answered
    "yes, that is right" with the sentence again. Nothing is claimed for a word that is not there: then the turn goes on as a new question."""
    found = echo_check.find(text, getattr(session, "last_spoken", "") or "")
    if not found:
        return False
    words, sentence = found
    spoken = {
        "bn": f"হ্যাঁ, ঠিক ধরেছেন, {words}। আবার বলছি: {sentence}",
        "hi": f"हाँ, आपने सही सुना, {words}। फिर से बताती हूँ: {sentence}",
        "en": f"Yes, that is right: {words}. Once more: {sentence}",
    }[lang if lang in ("bn", "hi") else "en"]
    await _speak(session, spoken, lang)
    return True


_WEEKDAY_NAMES = {
    "bn": ["সোমবার", "মঙ্গলবার", "বুধবার", "বৃহস্পতিবার", "শুক্রবার", "শনিবার", "রবিবার"],
    "hi": ["सोमवार", "मंगलवार", "बुधवार", "गुरुवार", "शुक्रवार", "शनिवार", "रविवार"],
    "en": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
}
_DATE_WEEKDAY_CUES = ("মানে", "মানে কি", "নাকি", "না ", "কি ", "কী", "तो", "मतलब", "या", "क्या", "means", "mean", "or", "is it", "isn't", "not", "?")


async def _answer_date_weekday(session: CallSession, text: str, lang: str) -> bool:
    """Owner, 2026-10-29 ("একত্রিশ তারিখ মানে তো শনিবার বুধবার বলবেন"): a day of the month and a weekday in one short question is a CALENDAR question, answered from the calendar -- never "I did not say 31".
    "The 31st is a Saturday" (yes / no, and what it really is); and, when a doctor was just discussed, whether he sits on the weekday the caller named, from the schedule table (the same reply as any
    weekday question). Nothing is said when no real date can be made of it."""
    n = owner_answers.spoken_number(text)
    wds = [w for w in dict.fromkeys(weekday_cues.weekday_of(tok) for tok in text.split()) if w is not None]  # every weekday named, in order ("শনিবার বুধবার")
    if n is None or not (1 <= n <= 31) or not wds or len(text.split()) > 14:
        return False
    low = " ".join(text.lower().split())
    if not any(c in low for c in _DATE_WEEKDAY_CUES):
        return False
    today = slot_proposal.clinic_now().date()
    ctx = session.last_doctor_enquiry or {}
    date = None
    try:
        if ctx.get("date") and datetime.date.fromisoformat(ctx["date"]).day == n:
            date = datetime.date.fromisoformat(ctx["date"])
    except ValueError:
        date = None
    if date is None:
        for i in range(0, 62):
            d = today + datetime.timedelta(days=i)
            if d.day == n:
                date = d
                break
    if date is None:
        return False
    lg = lang if lang in ("bn", "hi") else "en"
    names = _WEEKDAY_NAMES[lg]
    actual = names[date.weekday()]
    wd = date.weekday() if date.weekday() in wds else wds[0]
    said = names[wd]
    if date.weekday() == wd:
        line = {"bn": f"হ্যাঁ, {date.isoformat()} তারিখ {actual}।", "hi": f"हाँ, {date.isoformat()} को {actual} है।", "en": f"Yes, {date.isoformat()} is a {actual}."}[lg]
    else:
        line = {"bn": f"{date.isoformat()} তারিখ {actual}, {said} নয়।", "hi": f"{date.isoformat()} को {actual} है, {said} नहीं।", "en": f"{date.isoformat()} is a {actual}, not a {said}."}[lg]
    # and the doctor just discussed on the weekday named, when there is one
    doctor = ctx.get("doctor_name") or (session.booking.slots.get("doctor_name") if session.booking is not None else None)
    extra = ""
    other_wds = [w for w in wds if w != date.weekday()]
    if doctor and _tools is not None and other_wds:
        _w = other_wds[0]
        wd_date = (today + datetime.timedelta(days=(_w - today.weekday()) % 7 or 7)).isoformat()  # the coming such weekday
        if wd_date:
            try:
                info = await _tools.get_doctor_availability(doctor, wd_date)
                extra = " " + (doctor_weekday_yes_reply({"doctor_name": doctor, "date": wd_date}, info, lang, wd_date) or doctor_availability_reply({"doctor_name": doctor, "date": wd_date}, info, lang))
                session.last_doctor_enquiry = {"doctor_name": doctor, "date": info.get("next_available_date") if info.get("available") is False and info.get("next_available_date") else wd_date, "turn": session.turn_count}
            except ToolCallError:
                extra = ""
    await _speak(session, _with_resume(session, line + extra, lang), lang)
    return True


async def _answer_word_check(session: CallSession, text: str, lang: str) -> bool:
    """The caller wants to be sure of a word in what was just said ("চুয়াল্লিশ বললেন?", "did you say forty-four?"). A number they name is looked for in the last
    thing the agent said: if it is there, it is confirmed digit by digit and the sentence is said again; if it is not, they are told plainly it was not said and
    what was. Only for a short sentence that asks (a checking word) about a number -- never a guess about anything else."""
    if await _answer_date_weekday(session, text, lang):
        return True
    last = getattr(session, "last_spoken", "") or ""
    low = " ".join(text.lower().split())
    if not last or len(low.split()) > 10:
        return False
    # Digits being said (a phone number answered in pieces: "বাকিটা হচ্ছে ফাইভ সিক্স…") are an answer, never a word asked back -- real call 2026-10-23: the digits were thrown away twice
    booking = session.booking
    if booking is not None and booking.stage == "collecting" and (booking.phone_buffer or ("phone" in booking.asked_fields and not booking.slots.get("phone"))):
        return False
    if booking_capture.read_digits(text)[1] >= 2:
        return False
    if not any(c in low for c in _CHECK_CUES):
        return await _answer_echo(session, text, lang)  # no "did you say": a word asked back by sound ("বেনিয়া পুকুর কেয়া?")
    n = owner_answers.spoken_number(text)
    if n is None:
        # Owner, 2026-10-27: "আচ্ছা কত বললেন?" (okay, what did you say?) -- a GENERIC repeat of an amount, no guessed
        # number to check (unlike "did you say 500?", handled below). Cut the number straight out of what was just
        # said and read that sentence back, rather than falling through to the word-sound echo (which has nothing
        # to match here and would simply fail the turn).
        if any(c in low for c in _GENERIC_AMOUNT_CUES):
            said = re.sub(r"(?<=\d),(?=\d)", "", last.translate(_ANY_DIGITS))
            m = re.search(r"(?<!\d)\d{1,6}(?!\d)", said)
            if m:
                sentences = [s.strip() for s in re.split(r"(?<=[।.?!])\s+", last) if s.strip()]
                holder = next(
                    (s for s in sentences if m.group(0) in re.sub(r"(?<=\d),(?=\d)", "", s.translate(_ANY_DIGITS))),
                    last,
                )
                spoken = {
                    "bn": f"আমি বলেছিলাম: {holder}",
                    "hi": f"मैंने कहा था: {holder}",
                    "en": f"I said: {holder}",
                }[lang if lang in ("bn", "hi") else "en"]
                await _speak(session, spoken, lang)
                return True
        return await _answer_echo(session, text, lang)
    said = re.sub(r"(?<=\d),(?=\d)", "", last.translate(_ANY_DIGITS))
    present = re.search(rf"(?<!\d){n}(?!\d)", said) is not None
    sentences = [s.strip() for s in re.split(r"(?<=[।.?!])\s+", last) if s.strip()]
    holder = next((s for s in sentences if re.search(rf"(?<!\d){n}(?!\d)", re.sub(r"(?<=\d),(?=\d)", "", s.translate(_ANY_DIGITS)))), last)
    digits = _digit_words(n, lang)
    if present:
        spoken = {
            "bn": f"হ্যাঁ, ঠিক ধরেছেন। {n} মানে {digits}। আবার বলছি: {holder}",
            "hi": f"जी, आपने सही सुना। {n} यानी {digits}। फिर से बताती हूँ: {holder}",
            "en": f"Yes, that is right: {n} is {digits}. Once more: {holder}",
        }[lang if lang in ("bn", "hi") else "en"]
    else:
        spoken = {
            "bn": f"না, আমি {n} বলিনি। আমি বলেছিলাম: {last}",
            "hi": f"नहीं, मैंने {n} नहीं कहा। मैंने कहा था: {last}",
            "en": f"No, I did not say {n}. What I said was: {last}",
        }[lang if lang in ("bn", "hi") else "en"]
    await _speak(session, spoken, lang)
    return True


async def _answer_doctor_origin(session: CallSession, text: str, lang: str) -> bool:
    """Owner, 2026-10-29: "is the doctor from Kolkata?" -- answered from the doctor table's "From Kolkata?" column, one fixed sentence, about the doctor named in the sentence or just discussed.
    With no doctor to speak of, the caller is asked which one. False (nothing said) when the catalogue is not loaded."""
    if _fast_path is None:
        return False
    cat = _fast_path.catalogue
    named, canonical = False, None
    with contextlib.suppress(Exception):
        doc = _fast_path._named_doctor(fast_path_normalize(text), language_of_script(text, lang), strong=True)  # noqa: SLF001
        if doc is not None:
            named, canonical = True, doc[0]
    if canonical is None:
        ctx = session.last_doctor_enquiry or {}
        said = ctx.get("doctor_name") or (session.booking.slots.get("doctor_name") if session.booking is not None else None)
        if said:
            if cat.doctor_info(said) is not None:
                canonical = said
            else:
                with contextlib.suppress(Exception):
                    hit = cat.match(fast_path_normalize(str(said)), "doctor", language_of_script(str(said), lang), 0.9)
                    canonical = hit[0] if hit and hit[0] else None
    info = cat.doctor_info(canonical) if canonical else None
    if info is None:
        await _speak(session, missing_slot_prompt("doctor_availability", "doctor_name", lang), lang)
        return True
    if named:
        who = {"bn": f"ডাঃ {info.get('full_name_bn') or ''}".strip(), "hi": f"डॉक्टर {info.get('full_name_hi') or ''}".strip(), "en": "the doctor"}.get(lang, "the doctor")
    else:
        who = {"bn": "উনি", "hi": "वे", "en": "they"}.get(lang, "they")
    await _speak(session, _with_resume(session, doctor_origin.reply(info.get("from_kolkata"), who, lang), lang), lang)
    return True


def _text_names_a_doctor(text: str, lang: str) -> bool:
    """Does the sentence itself carry a doctor of the clinic's list (by name or surname)? False when the fast path is off."""
    if _fast_path is None:
        return False
    with contextlib.suppress(Exception):
        return _fast_path._named_doctor(fast_path_normalize(text), language_of_script(text, lang), strong=True) is not None  # noqa: SLF001  (strong: fuzzy noise such as "কালকে" ~ a surname is not a doctor)
    return False


def _strict_enquiry_doctor(session: CallSession, text: str, lang: str) -> str | None:
    """The doctor a sitting question is about, without any fuzzy guess: one NAMED in the sentence (with a doctor word, or whole), else the one just discussed, else the one in the open booking."""
    if _fast_path is not None:
        with contextlib.suppress(Exception):
            doc = _fast_path._named_doctor(fast_path_normalize(text), language_of_script(text, lang), strong=True)  # noqa: SLF001
            if doc is not None:
                return doc[1] or doc[0]
    ctx = session.last_doctor_enquiry or {}
    return ctx.get("doctor_name") or (session.booking.slots.get("doctor_name") if session.booking is not None else None)


def _enquiry_doctor(session: CallSession, text: str, lang: str) -> str | None:
    """The doctor an enquiry is about: the one NAMED in the sentence, else the one just discussed, else the one in the open booking."""
    if _fast_path is not None:
        with contextlib.suppress(Exception):
            doc = _fast_path._named_doctor(fast_path_normalize(text), language_of_script(text, lang), strong=False)  # noqa: SLF001
            if doc is not None:
                return doc[1] or doc[0]
    ctx = session.last_doctor_enquiry or {}
    return ctx.get("doctor_name") or (session.booking.slots.get("doctor_name") if session.booking is not None else None)


async def _general_medicine_today(lang: str) -> list[str]:
    """Who sits in General Medicine today, from the clinic's own table (empty on any failure -- the question is then simply asked)."""
    try:
        r = await _tools.department_doctors("General Medicine", lang, on_date=slot_proposal.clinic_now().date().isoformat())
    except ToolCallError:
        return []
    return [d["name"] for d in (r.get("sitting") or []) if d.get("name")]


async def _route_dept(text: str, lang: str, session: CallSession | None = None) -> dict:
    """sourav, 2026-10-24: clinic API department routing, with the patient's age when it is known -- from the sentence ("amar 15 bochorer chheler ...") or from the open booking's
    age answer -- so a patient of 13 or more is never sent to the paediatrician (clinic-api/lay_departments.py). The age is passed ONLY when known, so every earlier call
    (and every test double that takes just text and lang) behaves exactly as before."""
    age = age_cues.age_in(text)
    if age is None and session is not None and session.booking is not None:
        with contextlib.suppress(TypeError, ValueError):
            age = int(session.booking.slots.get("patient_age"))
    if age is None:
        return await _tools.route_department(text, lang)
    return await _tools.route_department(text, lang, age=age)


async def _redirect_teen_from_paediatrics(session: CallSession, st, lang: str) -> bool:
    """sourav, 2026-10-24 (owner): a patient of 13 YEARS OR MORE is not seen by the paediatrician -- the normal (General Medicine) doctor is. Checked once per booking, as soon as the
    age is known and a doctor is chosen; if that doctor is in Paediatrics the booking moves to a General Medicine doctor who sits on the chosen day (or, with no day yet, the first
    General Medicine doctor), the caller is told why, and the time is proposed again for the new doctor (the old hold simply lapses, exactly as for any doctor correction).
    Anything unknown (no age, no such lookup, the clinic API down, nobody sits) leaves the booking exactly as it was. Returns True when it spoke the next question."""
    if st.action != "book_appointment" or st.slots.get("_peds_age_checked") or _tools is None:
        return False
    try:
        age = int(st.slots.get("patient_age"))
    except (TypeError, ValueError):
        return False
    name = st.slots.get("doctor_name")
    if not name or not age_cues.is_teen_or_older(age):
        return False
    st.slots["_peds_age_checked"] = "1"
    try:
        dept = (await _tools.doctor_department(name)).get("department_name")
        if dept != "Paediatrics":
            return False
        date = st.slots.get("date")
        gm = await _tools.department_doctors("General Medicine", lang, on_date=date) if date else await _tools.department_doctors("General Medicine", lang)
    except (ToolCallError, AttributeError) as e:
        logger.warning("[%s] paediatric age rule skipped (%s)", session.call_id, e)
        return False
    pool = [d.get("name") for d in (gm.get("sitting") or []) if d.get("name")] if date else []
    pool = pool or [n for n in (gm.get("doctors_today") or gm.get("doctors") or []) if n]
    if not pool:
        return False
    new_doctor = pool[0].removeprefix("Dr. ")
    st.slots["doctor_name"] = new_doctor
    for k in ("_time_ok", "_sits_told", "time_slot", "_time_explicit", "_day_full_from"):
        st.slots.pop(k, None)
    st.hold_token, st.hold_doctor_id = None, None
    await _speak(session, age_cues.redirect_reply(_bare_doctor_name(new_doctor), lang), lang)
    if st.slots.get("date") and await _propose_slot(session, st, lang):
        return True
    return False


async def _answer_good_doctor(session: CallSession, text: str, lang: str) -> None:
    """"valo doctor ke ache?" (owner, 2026-10-23): no doctor is ever recommended -- the problem or department is asked for, and General Medicine's doctors
    of the day are named from the clinic's table. When the problem IS in the sentence ("কোমরে ব্যথা হচ্ছে কোন ডাক্তার দেখালে ভালো হয়") it is not asked again:
    the department the clinic's own routing table gives for it, and that department's doctors, are named."""
    try:
        routed = await _route_dept(text, lang, session)
    except ToolCallError:
        routed = {}
    if routed.get("matched"):
        _route_found.set({"dept": routed["department_name"], "names": routed.get("doctors_today") or routed.get("doctors") or []})
        spoken = (await _route_reply(session, routed, text, lang))
    else:
        recent = getattr(session, "last_route", None)
        sticky = None
        if recent and recent.get("dept") and session.turn_count - recent["turn"] <= _PICK_MEMORY_TURNS:
            # Owner, 2026-10-28: "কে ভালো" right after the cardiology doctors were named is about THEM, not General Medicine's list
            with contextlib.suppress(ToolCallError):
                today_iso = slot_proposal.clinic_now().date().isoformat()
                res = await _tools.department_doctors(recent["dept"], lang, on_date=today_iso)
                if res.get("matched") and res.get("doctors"):
                    sticky = {"matched": True, "department_name": recent["dept"], "doctors": res.get("doctors") or [], "doctors_today": [d["name"] for d in res.get("sitting") or [] if d.get("name")]}
        if sticky is not None:
            lead = {"bn": "কোন ডাক্তার ভালো সেটা আমি বলতে পারি না, আমাদের সব ডাক্তারই ভালো। ", "hi": "कौन सा डॉक्टर बेहतर है, यह मैं नहीं बता सकती; हमारे सभी डॉक्टर अच्छे हैं। ", "en": "I cannot say which doctor is better; all our doctors are good. "}[lang if lang in ("bn", "hi") else "en"]
            spoken = lead + department_route_reply(_dept_labelled(sticky, lang), "", lang)
        else:
            spoken = doctor_enquiry_cues.good_doctor_reply(await _general_medicine_today(lang), None, lang)
    await _speak(session, _with_resume(session, spoken, lang), lang)


async def _speak_weekdays(session: CallSession, name: str, lang: str) -> None:
    try:
        info = await _tools.doctor_weekdays(name)
    except ToolCallError as e:
        logger.warning("[%s] doctor_weekdays failed (%s)", session.call_id, e)
        await _speak(session, phrase("tool_failure", lang), lang, fallback_reason="tool_failure")
        return
    if not info.get("found"):
        await _speak(session, doctor_availability_reply({"doctor_name": name}, info, lang), lang)
        return
    key = {"bn": "doctor_name_bn", "hi": "doctor_name_hi"}.get(lang, "doctor_name")
    doctor = _bare_doctor_name(info.get(key) or info.get("doctor_name") or name)
    hours = set((info.get("hours") or {}).values())
    spoken = doctor_enquiry_cues.weekdays_reply(doctor, info.get("weekdays") or [], hours.pop() if len(hours) == 1 else None, lang)
    await _speak(session, _with_resume(session, spoken, lang), lang)


async def _answer_doctor_weekdays(session: CallSession, text: str, lang: str) -> None:
    """"kon kon din boshen?" (owner, 2026-10-23): the doctor's weekdays -- "সোম বুধ শুক্র বারে বসেন" -- never dates. No doctor named or just discussed:
    asked which one, and the next turn's answer is read here (session.pending_weekdays)."""
    name = _enquiry_doctor(session, text, lang)
    if not name:
        session.pending_weekdays = session.turn_count
        await _speak(session, missing_slot_prompt("doctor_availability", "doctor_name", lang), lang)
        return
    await _speak_weekdays(session, name, lang)


def _names_a_test(text: str, lang: str) -> bool:
    """True when the sentence names a lab test (it then belongs to the test flow, which has no doctor's hours and books at the counter)."""
    if _fast_path is None:
        return False
    with contextlib.suppress(Exception):
        return bool(_fast_path.named_tests(text, language_of_script(text, lang)))
    return False


_GENERIC_ADDRESS_WORDS = frozenset(
    "আমার আমাদের বাড়ি বাড়ী বাসা ঘর ঠিকানা এখানে ওখানে কাছে পাশে মোড় রোড রাস্তা পাড়া গ্রাম শহর এলাকা বাজার "
    "मेरा मेरे घर मकान पता यहाँ यहां वहाँ पास रोड गली मोहल्ला गाँव शहर इलाका "
    "my home house address near road lane village town area here there".split()
)


async def _check_address_answer(session: CallSession, text: str) -> None:
    """Owner, 2026-10-27: a short answer to "what is the address?" ("বালদা") is looked up in the West Bengal place tables (clinic-api geo tables: all 23 districts, police stations,
    places). When it is NOT a written-form match but sounds like a known place ("মালদা"), the hint `_addr_suggest` makes agent/booking_capture._detail_turn ask "did you say মালদা?" instead
    of storing a mishearing. No hint when it matches exactly, matches nothing, or the lookup fails: the answer is then taken as said, as before."""
    st = session.booking
    if st is None or _tools is None or st.slots.get("_addr_confirm") or st.slots.get("_ps_pending"):
        return
    if "patient_address" not in st.asked_fields or st.slots.get("patient_address") or st.slots.get("_address_done"):
        return
    st.slots.pop("_addr_suggest", None)
    if not text or len(text.split()) > 3:
        return
    if all(w.strip(" ,.।").lower() in _GENERIC_ADDRESS_WORDS for w in text.split()):
        return  # "আমার বাড়ি" (my house) is not a place to look up
    try:
        found = await _tools.geo_match(text)
    except ToolCallError as e:
        logger.warning("[%s] address lookup failed (%s)", session.call_id, e)
        return
    if not found.get("matched"):
        return
    # only a district or a police station is suggested/accepted here: a house-level place name sounds like half the words in the language, and a wrong "did you say ...?" (or a
    # wrong auto-accept) is noise. A smaller, unmatched place still falls through to the ordinary police-station question below, unchanged.
    top = next((c for c in found.get("candidates") or [] if c.get("kind") in ("district", "police_station") and c.get("score", 0) >= 0.55), None)
    if top is None:
        return
    spoken = top.get(f"name_{session.lang}") or ""
    if not re.search(r"[ঀ-৿ऀ-ॿ]", spoken):
        return  # a Bengali / Hindi voice cannot say Latin letters: no suggestion rather than a blocked reply
    if found.get("exact"):
        # Real call, 2026-10-29: a caller naming a real district or police station BY ITS WRITTEN FORM had the ordinary
        # "which police station?" question asked of them anyway (agent/booking_capture._lacks_police_station has no
        # idea the geo tables already confirmed a real place) -- stalled a real booking for several turns. An exact,
        # written-form match is address enough on its own, same discipline CLAUDE.md already applies to a doctor name.
        # Read and cleared by agent/booking_capture._detail_turn, same hint-handoff pattern as `_addr_suggest` below.
        st.slots["_addr_exact"] = spoken
        return
    st.slots["_addr_suggest"] = spoken


async def _check_police_station_answer(session: CallSession, text: str) -> None:
    """Item 19: while the police station is being asked for, the answer (and the address it belongs to) is looked up in the clinic's geography tables. Leaves two
    hints for agent/booking_capture._detail_turn: `_ps_match` -- the table's name for the station the caller said, when it is a sure match -- and `_ps_place` -- the
    station of a place the address names, when the tables give exactly one. No hint, and nothing changes. The tables only ever supply a written name."""
    st = session.booking
    if st is None or not st.slots.get("_ps_pending") or _tools is None:
        return
    st.slots.pop("_ps_match", None)
    st.slots.pop("_ps_place", None)
    try:
        said = await _tools.geo_match(text)
        home = await _tools.geo_match(st.slots["_ps_pending"])
    except ToolCallError as e:
        logger.warning("[%s] geography lookup failed (%s)", session.call_id, e)
        return
    def speakable(found: dict) -> str:
        # the station as the call's own voice can say it: a Bengali / Hindi call gets the Bengali / Hindi spelling (those voices cannot say English letters)
        if session.lang in ("bn", "hi"):
            return found.get(f"police_station_{session.lang}") or found["police_station"]
        return found["police_station"]

    if said.get("matched") and said.get("confident") and said.get("police_station"):
        st.slots["_ps_match"] = speakable(said)
    if home.get("exact") and home.get("police_station") and any(c.get("kind") == "place" for c in home.get("candidates") or []):
        st.slots["_ps_place"] = speakable(home)


async def _speak_day_availability(session: CallSession, name: str, date_iso: str, lang: str) -> None:
    """The doctor on one real date (the weekday the caller named, resolved on the clinic's calendar): sitting then, or off with his next sitting day -- the
    schedule table's own answer, the same wording as any availability question."""
    try:
        info = await _tools.get_doctor_availability(name, date_iso)
    except ToolCallError as e:
        logger.warning("[%s] weekday availability lookup failed (%s)", session.call_id, e)
        await _speak(session, phrase("tool_failure", lang), lang, fallback_reason="tool_failure")
        return
    session.last_doctor_enquiry = {"doctor_name": name, "date": date_iso, "turn": session.turn_count}
    slots = {"doctor_name": name, "date": date_iso}
    # a yes to "does he sit on <weekday>?" is just that (weekday, date, doctor); his hours and, when he does not sit, his next sitting come from the ordinary reply
    reply = doctor_weekday_yes_reply(slots, info, lang, date_iso) or doctor_availability_reply(slots, info, lang)
    await _speak(session, _with_resume(session, reply, lang), lang)


async def _answer_day_availability(session: CallSession, text: str, lang: str) -> bool:
    """"samner robibar doctor x ke pabo?" / "ei robibar" (owner, 2026-10-23, item 25). The weekday becomes the date of the coming such day and that doctor's
    availability then is told. No doctor named or just discussed: asked which one (session.pending_day). False (nothing said) for a lab test, or a bare weekday
    inside an open booking, which is the booking's own date answer."""
    if _names_a_test(text, lang):
        return False
    date_iso = weekday_cues.resolve_date(text, slot_proposal.clinic_now().date())
    if not date_iso:
        return False
    named_here = False
    if _fast_path is not None:
        with contextlib.suppress(Exception):
            # sourav, 2026-10-24: strong=True here, not the usual strong=False -- this check decides whether to let a
            # weekday question DERAIL an already-open booking, so it must not fire on fast_path.py's own documented
            # fuzzy noise ("कल्याण" scored 0.8 against "आपका पता क्या है"). Real call: "सोमवार छब्बीस तारीख" (no doctor
            # named) fuzzy-matched "सोमवार" to doctor Kalyan Haldar, which both passed this guard and then overwrote
            # the booking's real doctor (Devashish Das) when `name` below was computed with the same weak call.
            named_here = _fast_path._named_doctor(fast_path_normalize(text), language_of_script(text, lang), strong=True) is not None  # noqa: SLF001
    if session.booking is not None and session.booking.stage == "collecting" and not named_here:
        return False
    name = _enquiry_doctor(session, text, lang)
    if not name:
        session.pending_day = (session.turn_count, date_iso)
        await _speak(session, missing_slot_prompt("doctor_availability", "doctor_name", lang), lang)
        return True
    await _speak_day_availability(session, name, date_iso, lang)
    return True


async def _speak_sitting_hours(session: CallSession, name: str, lang: str) -> bool:
    """The doctor's sitting hours FROM and TO, on the day told (else his next sitting day), from the schedule table. False (nothing said) when they cannot be told."""
    ctx = session.last_doctor_enquiry or {}
    date_iso = ctx.get("date") or (session.booking.slots.get("date") if session.booking is not None else None)
    try:
        info = await _tools.get_doctor_availability(name, date_iso)
        if info.get("found") and not info.get("available") and info.get("next_available_date"):
            info = await _tools.get_doctor_availability(name, info["next_available_date"])  # the day he is off: his next sitting day
    except ToolCallError as e:
        logger.warning("[%s] sitting-hours lookup failed (%s)", session.call_id, e)
        return False
    hours = info.get("chamber_hours")
    if not (info.get("found") and info.get("available") and isinstance(hours, str) and "-" in hours):
        return False
    start, end = (x.strip() for x in hours.split("-", 1))
    key = {"bn": "doctor_name_bn", "hi": "doctor_name_hi"}.get(lang, "doctor_name")
    doctor = _bare_doctor_name(info.get(key) or info.get("doctor_name") or name)
    spoken = doctor_enquiry_cues.sitting_hours_reply(doctor, info.get("date") or date_iso or slot_proposal.clinic_now().date().isoformat(), start, end, lang)
    await _speak(session, _with_resume(session, spoken, lang), lang)
    return True


async def _answer_sitting_hours(session: CallSession, text: str, lang: str) -> bool:
    """"doctor x kota theke kota bose?" (owner, 2026-10-23): the doctor's hours, from and to. No doctor named, just discussed or in the booking: asked which one
    (the answer is read on the next turn, session.pending_hours). A lab test named: left to the normal path."""
    if _names_a_test(text, lang):
        return False
    name = _enquiry_doctor(session, text, lang)
    if not name:
        session.pending_hours = session.turn_count
        await _speak(session, missing_slot_prompt("doctor_availability", "doctor_name", lang), lang)
        return True
    return await _speak_sitting_hours(session, name, lang)


async def _answer_come_to_clinic(session: CallSession, text: str, lang: str) -> bool:
    """"clinic e giye book korte parbo?" (owner, 2026-10-23): yes -- but if the booking slots are full the same date may not be available."""
    if _names_a_test(text, lang):
        return False
    await _speak(session, _with_resume(session, doctor_enquiry_cues.come_to_clinic_reply(lang), lang), lang)
    return True


async def _answer_write_name(session: CallSession, text: str, lang: str) -> bool:
    """"bolchi ekta naam lekhanor chhilo" (owner, 2026-10-23): a name to be written = a serial to be taken, with no doctor said: asked back which doctor. The
    answer is read as a booking's doctor (session.pending_enquiry). A doctor / test / day already in the sentence, or a booking already open, is the booking
    flow's business: nothing is said here."""
    if _names_a_test(text, lang):
        return False
    booking = session.booking
    _offer = getattr(session, "dept_pending", None)
    if booking is None and _offer and _offer.get("kind") == "confirm_one" and session.turn_count - _offer.get("turn", -99) <= 3:
        return False  # the doctor and the day were JUST offered: "write my name" is the yes (see _dept_search_turn), not a new question
    # Owner, 2026-10-27: "দাঁতে ব্যথা করছে একটা ডাক্তার দেখাবো নাম লেখান" -- a symptom beside the request to write a name. The doctors of that department (and who sits next) are the answer; "which doctor?"
    # makes the caller say the symptom again.
    if booking is None and _tools is not None and off_topic.has_health_word(text):
        try:
            routed = await _route_dept(text, lang, session)
        except ToolCallError:
            routed = {}
        if routed.get("matched"):
            await _speak(session, _with_resume(session, await _route_reply(session, routed, text, lang), lang), lang)
            return True
    if booking is not None and booking.stage in ("confirming", "awaiting_charge_confirm"):
        return False  # a yes / no to the read-back is being waited for
    when = booking_capture.read_when(text, False, want_day=True)
    if when.date or when.time:
        return False
    if booking is not None and booking.stage == "collecting":
        # a booking is under way: "I will give a name" is the answer to its next question, whichever that is -- asked, not dropped to the model (which finds nothing in it)
        missing = missing_required(booking)
        if not missing:
            return False
        booking.asked_fields.add(missing[0])
        await _speak(session, missing_slot_prompt("book_appointment", missing[0], lang), lang)
        return True
    # Owner, 2026-10-29: a doctor was JUST discussed ("Dr X sits tomorrow, the 30th") and the caller says "write my name": that is the booking of THAT doctor on THAT day -- the booking flow
    # (booking_carryover) takes the doctor and the day from what was just told; "which doctor?" is only for a call where no doctor has come up.
    if booking is None and booking_carryover.recent(session.last_doctor_enquiry, session.turn_count):
        return False
    # no booking yet and no doctor in the air -- the doctor is asked for (the answer starts the booking)
    session.pending_enquiry = ("book_appointment", session.turn_count)
    ask = missing_slot_prompt("book_appointment", "doctor_name", lang)
    if doctor_enquiry_cues.asks_can_book_name(text, lang):
        # sourav, 2026-10-24: "naam ta booking kora jabe?" is a yes/no question -- answered "Yes, ..." first; the doctor's name comes next, then the date questions.
        ask = doctor_enquiry_cues.yes_prefix(lang) + ask
    await _speak(session, ask, lang)
    return True


async def _answer_test_days(session: CallSession, text: str, lang: str) -> bool:
    """sourav, 2026-10-24: "is the CT scan of the brain done on Thursday?" / "which days is the TMT done?" -- answered from the lab table's days column (clinic-api
    /api/v1/tests/availability), never from the model. Why: the owner added a days column so a test done "every day except Thursday" is told that way. Flow: the sentence
    must be a days question (agent/test_days.asks_test_days) AND name a lab test the catalogue knows; anything else (no test named, the clinic API down, test not found) returns
    False with nothing said, and the old path answers exactly as before. The test is spoken by the clinic's own name for it, in the call's language."""
    if _fast_path is None or _tools is None or not test_days.asks_test_days(text):
        return False
    try:
        names = _fast_path.named_tests(text, language_of_script(text, lang))
    except Exception:  # noqa: BLE001 - a catalogue hiccup must not cost the caller an answer: the old path takes the sentence
        return False
    if not names:
        return False
    try:
        r = await _tools.get_test_days(names[0])
    except ToolCallError as e:
        logger.warning("[%s] test-days lookup failed (%s)", session.call_id, e)
        return False
    if not r.get("found") or r.get("ambiguous"):
        return False
    key = {"bn": "test_name_bn", "hi": "test_name_hi"}.get(lang, "test_name")
    spoken_name = r.get(key) or r.get("test_name") or names[0]
    days = r.get("available_days") or []
    await _speak(session, _with_resume(session, test_days.days_reply(spoken_name, days, lang, test_days.named_weekday(text)), lang), lang)
    return True


def _booking_is_open(session: CallSession) -> bool:
    """sourav, 2026-10-24: a booking is being collected or confirmed (its next answer belongs to the booking, so the side requests below leave the turn alone)."""
    b = session.booking
    return b is not None and b.stage in ("collecting", "confirming", "awaiting_charge_confirm")


async def _report_visit_turn(session: CallSession, text: str, lang: str) -> bool:
    """sourav, 2026-10-24: the caller's answer to 'did you see the doctor seven days ago, or earlier?' (asked because they want to show a report). Why: the owner's rule -- a
    visit seven days ago or earlier means a new booking and payment to come; a fresh visit may simply come and show the report. Flow: visit_cues.visit_age_answer reads yes / no
    or the length of time itself; 'old' -> book-and-pay (and the next sentence naming a doctor starts the booking); 'recent' -> may come; not understood -> asked once more, then
    the turn goes on to the ordinary path (returns False). Always clears the question when it is answered or given up."""
    flow = session.report_visit_asked
    if flow is None:
        return False
    answer = visit_cues.visit_age_answer(text, classify_yes_no(text, lang))
    if answer is None:
        flow["tries"] += 1
        if flow["tries"] >= 2:
            session.report_visit_asked = None
            return False
        await _speak(session, visit_cues.repeat_visit_reply(lang), lang)
        return True
    session.report_visit_asked = None
    if answer == "old":
        session.pending_enquiry = ("book_appointment", session.turn_count)
        await _speak(session, visit_cues.old_visit_reply(lang), lang)
    else:
        await _speak(session, visit_cues.recent_visit_reply(lang), lang)
    return True


async def _answer_next_serial(session: CallSession, text: str, lang: str) -> bool:
    """sourav, 2026-10-24: "aj book korle serial koto hobe?" -- the serial number the next booking would get, from the clinic's table (serials are simulated, generated with each
    booking). Which doctor is the sentence's, the one just discussed or the open booking's; none of them -> asked, and the answer is read on the next turn (session.pending_serial).
    Returns True when the turn was answered or the doctor was asked for."""
    day = visit_cues.serial_day(text)
    name = _enquiry_doctor(session, text, lang)
    if not name:
        session.pending_serial = (session.turn_count, day)
        await _speak(session, missing_slot_prompt("doctor_availability", "doctor_name", lang), lang)
        return True
    return await _speak_next_serial(session, name, day, lang)


async def _speak_next_serial(session: CallSession, name: str, day: str, lang: str) -> bool:
    on = slot_proposal.clinic_now().date() + datetime.timedelta(days=1 if day == "tomorrow" else 0)
    try:
        r = await _tools.next_serial(name, on.isoformat())
    except ToolCallError as e:
        logger.warning("[%s] next-serial lookup failed (%s)", session.call_id, e)
        await _speak(session, phrase("tool_failure", lang), lang, fallback_reason="tool_failure")
        return True
    if not r.get("found"):
        session.pending_serial = (session.turn_count, day)
        await _speak(session, missing_slot_prompt("doctor_availability", "doctor_name", lang), lang)
        return True
    key = {"bn": "doctor_name_bn", "hi": "doctor_name_hi"}.get(lang, "doctor_name")
    doctor = _bare_doctor_name(r.get(key) or r.get("doctor_name") or name)
    session.last_doctor_enquiry = {"doctor_name": r.get("doctor_name") or name, "date": r.get("date"), "turn": session.turn_count}
    if r.get("available") and r.get("next_serial") is not None:
        spoken = visit_cues.next_serial_reply(doctor, int(r["next_serial"]), day, lang, chamber_hours=r.get("chamber_hours"))
    else:
        nxt = r.get("next_available_date")
        spoken = visit_cues.next_serial_not_sitting_reply(doctor, day, slot_proposal.when_phrase(nxt, lang) if nxt else None, lang)
    await _speak(session, _with_resume(session, spoken, lang), lang)
    return True


async def _serial_flow_turn(session: CallSession, text: str, lang: str, answer_hint: str | None) -> None:
    """sourav, 2026-10-24: one caller turn inside the 'paid but no serial number' conversation. Why: the owner wants the agent to confirm the mobile number, look the booking
    up in the database (payment is simulated there) and tell the caller what the database holds -- the serial is the database's, never composed. Flow: serial_lookup.advance
    reads the number / the yes-no; "confirm" reads the number back; "lookup" asks the clinic API; a paid booking's serial is spoken and queued to WhatsApp again (said as
    'sending', never 'sent': the channel is a placeholder); an unpaid booking is explained and a callback is queued; nothing found goes back to asking for another number, up to
    serial_lookup.MAX_TRIES. The flow is cleared whenever it ends, so a later 'ok' means goodbye again."""
    flow = session.serial_flow
    if flow is None:
        return
    action, payload = serial_lookup.advance(flow, text, answer_hint)
    if action == "confirm":
        await _speak(session, serial_lookup.confirm_number(payload, lang), lang)
        return
    if action == "repeat":
        await _speak(session, serial_lookup.confirm_number(payload, lang), lang)
        return
    if action == "ask_again":
        await _speak(session, serial_lookup.ask_again(lang), lang)
        return
    if action == "give_up":
        session.serial_flow = None
        await _speak(session, serial_lookup.give_up(lang), lang)
        return
    # action == "lookup": the caller confirmed the number
    phone = payload
    _note_caller_number(session, phone)
    try:
        result = await asyncio.wait_for(_tools.lookup_serial(phone), 6.0)
    except (ToolCallError, asyncio.TimeoutError) as e:
        logger.warning("[%s] serial lookup failed (%s)", session.call_id, e)
        session.serial_flow = None
        await _speak(session, serial_lookup.check_failed(lang), lang)
        return
    bookings = result.get("bookings") or []
    parts = [serial_lookup.result_reply(bookings, lang)]
    paid = [b for b in bookings if b.get("paid") and b.get("serial_number") is not None]
    if paid:
        try:
            sent = await asyncio.wait_for(_tools.resend_serial(phone, paid[0]["confirmation_id"]), 6.0)
            if sent.get("success"):
                parts.append(serial_lookup.resend_note(sent.get("sent_to_last4"), lang))
        except (ToolCallError, asyncio.TimeoutError) as e:
            logger.warning("[%s] serial resend failed (%s)", session.call_id, e)  # the serial was still told; no claim about WhatsApp is made
    elif bookings:
        try:
            summary = build_call_summary("paid_but_no_serial", lang=lang, phone=phone, call_id=session.call_id)
            queued = await asyncio.wait_for(_tools.request_callback(phone, session.call_id, "asap", "paid_but_no_serial", summary), 4.0)
            if isinstance(queued, dict) and queued.get("success", True):
                parts.append(serial_lookup.unpaid_followup(lang))
        except Exception as e:  # noqa: BLE001 - the callback is a courtesy; the explanation above stands without it
            logger.warning("[%s] could not queue the paid-but-no-serial call back: %s", session.call_id, e)
    if not bookings:
        flow["tries"] += 1
        if flow["tries"] < serial_lookup.MAX_TRIES:
            flow.update(stage="ask", phone=None, buf="")  # another number may be the one the booking was made with
            await _speak(session, " ".join(parts), lang)
            return
    session.serial_flow = None
    await _speak(session, " ".join(parts), lang)


async def _answer_when_to_come(session: CallSession, text: str, lang: str) -> bool:
    """"kokhon asbo?" (owner, 2026-10-23): the doctor's sitting START time and "keep one to two hours in hand" -- the sentence a booking already ends with
    (slot_proposal.conclusion). The doctor is the one named, just discussed, or in the open booking; the day is the one told, else his next sitting. True when
    answered; False (nothing said) when no doctor's hours can be told."""
    if _fast_path is not None:
        with contextlib.suppress(Exception):
            if _fast_path.named_tests(text, language_of_script(text, lang)):
                return False  # a lab test is named: it has no doctor's hours to tell, the normal path answers it
    name = _enquiry_doctor(session, text, lang)
    if not name:
        # No doctor yet: ask which one (never let the sentence fall to the fuzzy FAQ matcher, which answered "what time do I need to reach" with the insurance text);
        # the answer is read on the next turn (session.pending_when).
        session.pending_when = session.turn_count
        await _speak(session, missing_slot_prompt("doctor_availability", "doctor_name", lang), lang)
        return True
    booking = session.booking
    if booking is not None and not booking.is_stale() and booking.slots.get("doctor_name") == name:
        booked_date, booked_time = booking.slots.get("date"), shown_time(booking.slots)
        if booked_date and booked_time:
            # Owner, 2026-10-27: real call -- "আমার অ্যাপয়েন্টমেন্ট ছিল, কখন বললেন?" (I had an appointment, what
            # time did you say) asked again later in the SAME booking, after other questions came between. The
            # caller's own slot may not be the doctor's bare sitting start (they could have asked for a later
            # time themselves, shown_time/agent/reply_templates.py already carries that distinction) -- re-deriving
            # a fresh, generic "doctor sits at X" from the clinic API here would contradict what was actually
            # agreed for THIS booking, so the booking's own slots (the one place holding that truth) are read
            # back verbatim instead of calling get_doctor_availability again.
            spoken = slot_proposal.conclusion(_bare_doctor_name(name), booked_date, booked_time, lang)
            await _speak(session, _with_resume(session, spoken, lang), lang)
            return True
    ctx = session.last_doctor_enquiry or {}
    date_iso = ctx.get("date") or (session.booking.slots.get("date") if session.booking is not None else None)
    try:
        info = await _tools.get_doctor_availability(name, date_iso)
        if info.get("found") and not info.get("available") and info.get("next_available_date"):
            info = await _tools.get_doctor_availability(name, info["next_available_date"])  # the day he is off: his next sitting day
    except ToolCallError as e:
        logger.warning("[%s] when-to-come lookup failed (%s)", session.call_id, e)
        return False
    hours = info.get("chamber_hours")
    if not (info.get("found") and info.get("available") and isinstance(hours, str) and "-" in hours):
        return False
    key = {"bn": "doctor_name_bn", "hi": "doctor_name_hi"}.get(lang, "doctor_name")
    doctor = _bare_doctor_name(info.get(key) or info.get("doctor_name") or name)
    spoken = slot_proposal.conclusion(doctor, info.get("date") or date_iso or slot_proposal.clinic_now().date().isoformat(), hours.split("-")[0].strip(), lang)
    await _speak(session, _with_resume(session, spoken, lang), lang)
    return True


async def _answer_next_if_absent(session: CallSession, text: str, lang: str) -> None:
    """"aj doctor na pele next kobe?" (owner, 2026-10-23). A doctor NAMED (or just discussed) is gone with: his next sitting date plus the other doctors of
    his department who are available today. No doctor: the problem is asked for -- or, when it is in the sentence, its department's doctors are named --
    with General Medicine's doctors of the day as the way forward."""
    name = _enquiry_doctor(session, text, lang)
    if not name:
        try:
            routed = await _route_dept(text, lang, session)
        except ToolCallError:
            routed = {}
        if routed.get("matched"):
            _route_found.set({"dept": routed["department_name"], "names": routed.get("doctors_today") or routed.get("doctors") or []})
            spoken = (await _route_reply(session, routed, text, lang))
        else:
            session.pending_enquiry = ("department_query", session.turn_count)
            spoken = missing_slot_prompt("department_query", "symptom_description", lang) + doctor_enquiry_cues.general_medicine_line(
                await _general_medicine_today(lang), None, lang
            )
        await _speak(session, _with_resume(session, spoken, lang), lang)
        return
    today = slot_proposal.clinic_now().date()
    try:
        tomorrow = await _tools.get_doctor_availability(name, (today + datetime.timedelta(days=1)).isoformat())
        others = await _tools.department_alternatives(name, today.isoformat(), lang)
    except ToolCallError as e:
        logger.warning("[%s] next-if-absent lookup failed (%s)", session.call_id, e)
        await _speak(session, phrase("tool_failure", lang), lang, fallback_reason="tool_failure")
        return
    if not tomorrow.get("found"):
        await _speak(session, doctor_availability_reply({"doctor_name": name}, tomorrow, lang), lang)
        return
    key = {"bn": "doctor_name_bn", "hi": "doctor_name_hi"}.get(lang, "doctor_name")
    doctor = _bare_doctor_name(tomorrow.get(key) or tomorrow.get("doctor_name") or name)
    next_date = tomorrow.get("date") if tomorrow.get("available") else tomorrow.get("next_available_date")
    spoken = doctor_enquiry_cues.next_if_absent_reply(doctor, next_date, [a["name"] for a in (others.get("alternatives") or []) if a.get("name")], lang)
    await _speak(session, _with_resume(session, spoken, lang), lang)


async def _answer_department_doctors(session: CallSession, text: str, lang: str) -> bool:
    """"who are the doctors in cardio" (agent/dept_list.py): the department's short form is resolved by the clinic API; the answer is the real list,
    who sits today first, and the caller chooses. No advice, no test, no medicine."""
    lists_doctors = dept_list.asks_doctor_list(text)
    if not lists_doctors and not dept_list.mentions_doctor_request(text):
        return False
    if not lists_doctors and _fast_path is not None:
        # KCD-lead-A: "There is an orthopedic doctor." names a SPECIALTY, not a doctor -- but a sentence that names a real doctor stays on the normal path.
        with contextlib.suppress(Exception):
            if _fast_path._named_doctor(fast_path_normalize(text), language_of_script(text, lang)) is not None:  # noqa: SLF001
                return False
    # Real call 2026-10-18: "কার্ডিওলজি বিভাগে কালকে কোন কোন ডাক্তার বসছে" (who sits TOMORROW) was answered with the list and "nobody sits today". A day or an hour in the
    # question is the day / hour asked about: the doctors who really sit then, with their hours.
    when = booking_capture.read_when(text, False, want_day=True)
    on_date, at_time = (when.date or None), (when.time or None)
    try:
        result = await _tools.department_doctors(text, lang, on_date=on_date, at_time=at_time)
    except ToolCallError:
        return False
    if not result.get("matched"):
        return False
    if on_date or at_time:
        sitting = result.get("sitting") or []
        label = _dept_label(result["department_name"], lang)
        if not sitting and on_date:
            # Owner, 2026-10-25: "is there a neurologist today?" with nobody sitting -- the FIRST sentence says who sits next, when, and asks to book. Never "tell me another day".
            if await _say_nobody_then_next(session, text, lang, None, on_date, _when_phrase(on_date, at_time, lang) or label, label):
                return True
        if len(sitting) == 1:
            # one doctor: a yes books HIM (the same pending question as "Dr X sits tomorrow at 11, shall I book it?")
            session.dept_pending = {"kind": "confirm_one", "doctor": sitting[0], "date": on_date, "time": at_time or sitting[0].get("start"), "turn": session.turn_count}
        elif sitting:
            session.dept_pending = {"kind": "pick", "doctors": sitting, "date": on_date, "time": at_time, "turn": session.turn_count}
        await _speak(session, _with_resume(session, doctor_search.sitting_list_reply(sitting, _when_phrase(on_date, at_time, lang), label, lang), lang), lang)
        return True
    await _speak(session, _with_resume(session, (await _route_reply(session, result, text, lang)), lang), lang)
    return True


async def _answer_test_overview(session: CallSession, text: str, lang: str) -> bool:
    """"What tests do you do?" (agent/test_overview.py): the kinds of tests the clinic offers, from the catalogue, no model. Only when the question
    is about the whole range and names no particular test. True when answered."""
    if not test_overview.asks_which_tests(text):
        return False
    try:
        if _fast_path is not None and _fast_path.named_tests(text, language_of_script(text, lang)):
            return False  # a test is named: that is an availability / price question, handled elsewhere
        cat = await _cached_catalogue()
        spoken = test_overview.reply(cat.get("tests", []), lang)
    except Exception as e:  # noqa: BLE001 - an overview is an aid; the turn goes on to the normal path
        logger.warning("test overview skipped: %s", e)
        return False
    if not spoken:
        return False
    session.pending_enquiry = ("test_rate", session.turn_count)  # the next test name gets its price
    await _speak(session, _with_resume(session, spoken, lang), lang)
    return True


async def _answer_named_total(session: CallSession, text: str, lang: str) -> bool:
    """sourav, 2026-10-16: "<test A> and <test B> total koto hobe" -- two or more tests NAMED in the sentence plus a total word. Answered here, with
    no model: fast_path.named_tests finds the tests, the clinic API prices each one, multi_test_rate_reply adds them up (truth boundary: no
    price comes from anywhere but the clinic). It runs even in the middle of a booking (a price question does not touch the booking; the
    "back to your booking" line is appended). Returns False -- turn untouched -- when unsure or when the clinic lookup fails."""
    fp = _fast_path
    if fp is None or _tools is None:
        return False
    total, remove = cart_tally.wants_total_or_remove(text)
    if not total or remove:
        return False
    try:
        names = fp.named_tests(text, lang)
    except Exception:  # noqa: BLE001
        return False
    if len(names) < 2:
        return False
    try:
        results = [(n, await _tools.get_test_rate(n)) for n in names]
    except ToolCallError as e:
        logger.warning("[%s] named-total lookup failed (%s) -- leaving the turn to the normal path", session.call_id, e)
        return False
    # Real call 2026-10-17: "...three together, total" with the CBC lost by the recogniser was priced as a total of the TWO tests heard.
    # When the caller said how many tests they meant and fewer were understood, say what was caught and ask for all the names again.
    short = multi_test_confirm.shortfall(results, multi_test_confirm.expected_count(text), lang)
    if short:
        await _speak(session, _with_resume(session, short, lang), lang)
        return True
    for n in names:
        if n.strip().lower() not in [x.strip().lower() for x in session.priced_tests]:
            session.priced_tests.append(n)
    prefix = _done_here(text, lang, _found_names(results, lang))  # "Yes, we do <the tests> here." first, when that was the question
    await _speak(session, _with_resume(session, prefix + await _multi_reply(results, lang), lang), lang)
    return True


async def _answer_from_priced_list(session: CallSession, text: str, lang: str) -> bool:
    """sourav, 2026-10-14: True when this turn was a "remove" and/or "total" over session.priced_tests and has been
    answered. The list holds NAMES only; every price spoken is a fresh clinic-API lookup summed by
    multi_test_rate_reply, so a stale or invented figure is impossible. Returns False (turn untouched) when unsure."""
    total, remove = cart_tally.wants_total_or_remove(text)
    if not (total or remove) or _tools is None:
        return False
    fp = _fast_path
    canon = None
    several = False
    if fp is not None:
        def canon(s: str) -> str | None:  # noqa: E306 - the catalogue test a phrase names, in this call's language
            try:
                from agent.fast_path import COMMIT_FLOOR as _floor, _normalize as _fpn

                name, _form, score = fp.catalogue.match(_fpn(s), "test", lang, _floor)
            except Exception:  # noqa: BLE001
                return None
            return name if name and score >= _floor else None
        try:
            several = fp.looks_multi_test(text, lang)
        except Exception:  # noqa: BLE001
            several = False
    if several and not remove:
        return False  # sourav, 2026-10-16: a sentence naming SEVERAL tests is a fresh question (answered by _answer_named_total / the model), never "the total of the old list"
    plan = cart_tally.plan(text, session.priced_tests, canon, several)
    if plan is None:
        return False
    gone = [session.priced_tests[i] for i in plan.remove]
    kept = [n for i, n in enumerate(session.priced_tests) if i not in set(plan.remove)]
    try:
        results = [(n, await _tools.get_test_rate(n)) for n in kept]
    except ToolCallError as e:
        logger.warning("[%s] priced-list lookup failed (%s) -- leaving the turn to the normal path", session.call_id, e)
        return False
    session.priced_tests = kept
    parts = [cart_tally.removed_text(gone, lang)] if gone else []
    parts.append(multi_test_rate_reply(results, lang) if kept else cart_tally.empty_text(lang))
    await _speak(session, " ".join(parts), lang)
    return True


async def _finish_enquiry_turn(
    session: CallSession, text: str, lang: str, intent: str, slots: dict, base_reply: str, data: dict
) -> None:
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
    session.last_enquiry = {"text": text, "turn": session.turn_count}  # a read-only enquiry: it may be replayed in another language on request
    turn_entities = [(intent, slots)]
    secondary_intent = data.get("secondary_intent")
    # Owner, 2026-10-29: the same kind of question again with no name in it ("the doctor" asked twice in one sentence) is not a second question: it was answered by the first. Not asked back.
    if secondary_intent and secondary_intent == intent and not any((data.get("secondary_slots") or {}).get(k) for k in ("doctor_name", "test_name", "faq_topic", "symptom_description", "test_names")):
        secondary_intent = None
    if secondary_intent:
        secondary_slots = data.get("secondary_slots") or {}
        secondary_slots, _ = resolve_followup_slot(
            secondary_intent, secondary_slots, text, lang, session.last_enquiry_entities
        )
        addendum = None
        try:
            reply2 = await _answer_enquiry_intent(secondary_intent, secondary_slots, lang)
        except ToolCallError as e:
            logger.warning("[%s] secondary-question lookup failed (%s) -- keeping the first answer", session.call_id, e)
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
    session.last_enquiry_turn = session.turn_count
    # sourav, 2026-10-14: remember every test priced this turn (once each) for "remove X" / "total" later.
    for _n in _priced_names.get() or []:
        if _n and _n.strip().lower() not in [x.strip().lower() for x in session.priced_tests]:
            session.priced_tests.append(_n)
    _priced_names.set(None)
    _rx = _rx_asked.get()
    if _rx:
        session.pending_prescription = (_rx, session.turn_count)
        _rx_asked.set(None)
    _rf = _route_found.get()
    if _rf:
        _prev = getattr(session, "last_route", None)
        _pend = getattr(session, "dept_pending", None)
        if _prev and _prev.get("dept") != _rf.get("dept") and _pend and _pend.get("kind") == "pick" and _pend.get("turn", 0) < session.turn_count:
            session.dept_pending = None  # another department was named: the old department's doctors are no longer the subject
        if getattr(session, "last_agreement", 1.0) >= 0.6:  # a garbled turn does not move the department being discussed
            session.last_route = {**_rf, "turn": session.turn_count}
        _route_found.set(None)
        _pend2 = getattr(session, "dept_pending", None)
        if _rf.get("dept") and (_pend2 is None or _pend2.get("turn", 0) < session.turn_count):
            with contextlib.suppress(ToolCallError):
                _today_iso = slot_proposal.clinic_now().date().isoformat()
                _sit = (await _tools.department_doctors(_rf["dept"], lang, on_date=_today_iso)).get("sitting") or []
                # owner, 2026-10-28: "ওনার সাথে অ্যাপয়েন্টমেন্ট করতে চান?" -- the caller's "হ্যাঁ করে দিন" must find the offer it answers
                if len(_sit) == 1:
                    session.dept_pending = {"kind": "confirm_one", "doctor": _sit[0], "date": _today_iso, "time": _sit[0].get("start"), "turn": session.turn_count, "say": base_reply}
                elif len(_sit) >= 2:
                    session.dept_pending = {"kind": "pick", "doctors": _sit, "date": _today_iso, "time": None, "turn": session.turn_count, "say": base_reply}
    _nso = _next_sit_offer.get()
    if _nso:
        session.dept_pending = {**_nso["pending"], "turn": session.turn_count, "say": _nso["say"]}
        _next_sit_offer.set(None)
    _de = _doctor_enquiry_found.get()
    if _de:
        session.last_doctor_enquiry = {**_de, "turn": session.turn_count}
    _doctor_enquiry_found.set(None)
    suggestion, _ = _suggested.get(), _suggested.set(None)
    if suggestion is not None and session.pending_entity is None:
        replay = {
            **data,
            "intent": suggestion["intent"],
            "secondary_intent": None,
            "slots": {**suggestion["slots"], suggestion["slot"]: suggestion["name"]},
        }
        session.pending_entity = entity_text.PendingEntity(
            suggestion["intent"], suggestion["slot"], suggestion["name"], replay
        )
        session.pending_question = (base_reply, session.turn_count)  # what was just asked, so a garbled reply can be met with the SAME question
        if isinstance(suggestion["name"], list):
            session.last_multi = (list(suggestion["name"]), session.turn_count)  # the tests of a total, for "these ones" / "এইগুলোই"
    base_reply = _with_resume(session, base_reply, lang)  # KCD-104: back to the booking, if one is open
    # An answer to the caller's own question starts with the answer (no "thank you for telling me"); KCD-513: a
    # short varied acknowledgement before it when due, so the reply doesn't open mid-sentence; KCD-512: a
    # senior's warm closing after.
    base_reply, _ = session.acks.decorate(base_reply, lang, already_acknowledged=session.turn_acknowledged)
    base_reply = session.kindness.decorate(base_reply, lang)
    _note(session, "answers_given")
    await _speak(session, base_reply, lang)


def _mark(session: CallSession, name: str) -> None:
    """Record when a stage of THIS turn finished (first occurrence only). Read by _log_timing at first audio."""
    marks = getattr(session, "marks", None)
    if marks is not None and not marks:
        session.api0 = (tools_client.API_STATS["calls"], tools_client.API_STATS["ms"])  # the clinic-API calls of this turn are counted from here
    if marks is not None and all(n != name for n, _t in marks):
        marks.append((name, time.monotonic()))


def _log_timing(session: CallSession) -> None:
    """One line per turn: where the time from "the turn started" to "the first audio left" went. Latency was being
    felt and reported without a breakdown; this is the breakdown, per stage, in the log."""
    marks, session.marks = getattr(session, "marks", None), None
    if not marks or len(marks) < 2:
        return
    parts, prev = [], marks[0][1]
    for name, t in marks[1:]:
        parts.append(f"{name} {int((t - prev) * 1000)} ms")
        prev = t
    api0 = getattr(session, "api0", None) or (tools_client.API_STATS["calls"], tools_client.API_STATS["ms"])
    api = f" | clinic API {tools_client.API_STATS['calls'] - api0[0]} calls {int(tools_client.API_STATS['ms'] - api0[1])} ms"
    logger.info(
        "[%s] turn timing: %s | total %d ms%s",
        session.call_id,
        " | ".join(parts),
        int((marks[-1][1] - marks[0][1]) * 1000),
        api,
    )


UTT_RING_DIR = os.environ.get("UTT_RING_DIR", "/tmp/utt_ring")
UTT_RING_KEEP = int(os.environ.get("UTT_RING_KEEP", "40"))


def _keep_recent_utterance(session: CallSession, wav_path: str) -> None:
    """A debug aid: the last few utterances sent to recognition are copied to /tmp/utt_ring (container disk, newest UTT_RING_KEEP kept) so a
    report like "I said five sentences and only the last was captured" can be answered from the audio itself. Never raises."""
    if UTT_RING_KEEP <= 0:
        return
    try:
        os.makedirs(UTT_RING_DIR, exist_ok=True)
        shutil.copyfile(wav_path, os.path.join(UTT_RING_DIR, f"{int(time.time())}_{session.call_id}.wav"))
        kept = sorted(f for f in os.listdir(UTT_RING_DIR) if f.endswith(".wav"))
        for old in kept[:-UTT_RING_KEEP]:
            os.remove(os.path.join(UTT_RING_DIR, old))
    except OSError:
        pass


async def _catalogue_codes() -> dict[str, str]:
    """{UPPER-CASE letters-only code: canonical test name} for every catalogue test that has one (CBC, ESR, TSH, PSA, HIV, ECG, ...). A code
    two tests share names neither."""
    cat = await _cached_catalogue()
    by_code: dict[str, set[str]] = {}
    for t in cat.get("tests", []):
        code = lab_test_code(str(t["name"]))
        if code.isalpha() and code.isascii() and code.upper() == code and len(code) >= 2:
            by_code.setdefault(code, set()).add(str(t["name"]))
    return {c: next(iter(names)) for c, names in by_code.items() if len(names) == 1}


_WHICH_TEST = {
    "bn": "কোন টেস্টটার কথা জানতে চান? নামটা একটু ধীরে বলবেন?",
    "hi": "किस टेस्ट के बारे में जानना है? नाम ज़रा धीरे बताइए।",
    "en": "Which test do you mean? Could you say its name slowly?",
}


def _which_test_prompt(lang: str) -> str:
    return _WHICH_TEST.get(lang, _WHICH_TEST["en"])


async def _normalise_spelled(text: str) -> str:
    """"টি এস এইচ টেস্ট কত" -> "TSH টেস্ট কত": letters pronounced separately are read back as the test code they spell, for every test in the
    catalogue (agent/spelled_letters.normalise_spelled_codes). Never fails the turn."""
    text = nato_letters.collapse(text)  # "S for September, A for Alpha" -> "S A"
    try:
        codes = await _catalogue_codes()
        new_text, found = spelled_letters.normalise_spelled_codes(text, codes)
        new_text, sounded = spelled_letters.sound_codes(new_text, codes)  # a code said fast and written as one made-up word ("উসগ")
        found = list(found) + sounded
        if found or new_text != text:
            logger.info("spelled letters read as code(s) %s: %r -> %r", found or "(near)", text, new_text)
        return new_text
    except Exception as e:  # noqa: BLE001 - advisory
        logger.warning("spelled-letter normalisation skipped: %s", e)
        return text


async def _with_spelled_tests(session: CallSession, lang: str, result, english_text: str):
    """A test code said letter by letter that the Bengali/Hindi recogniser wrote as a made-up word ("সি বি সি" -> "ক্বক"): the English
    recogniser (already run on this turn) keeps the letters apart. If it heard a catalogue code the Bengali/Hindi text does not name,
    that code is appended to the text the model reads (agent/spelled_letters.py). Never fails the turn."""
    try:
        if not english_text or not result.text:
            return result
        logger.info("[%s] English recogniser heard %r beside the %s text %r", session.call_id, english_text, lang, result.text)
        cat = await _cached_catalogue()
        by_code: dict[str, set[str]] = {}
        for t in cat.get("tests", []):
            code = lab_test_code(str(t["name"]))
            if code.isalnum() and code.upper() == code and any(c.isalpha() for c in code):
                by_code.setdefault(code, set()).add(str(t["name"]))
        codes = {c: next(iter(names)) for c, names in by_code.items() if len(names) == 1}  # a code two tests share names neither
        already = set(_fast_path.named_tests(result.text, lang)) if _fast_path is not None else set()
        # "T S H" heard beside a named Thyroid Profile (T3 T4 TSH) is that same test, not a fourth one
        add = [
            cand for cand, name, _exact in spelled_letters.match_codes(english_text, codes)
            if name not in already and not any(cand.upper() in a.upper() for a in already)
        ]
        if not add:
            return result
        logger.info("[%s] spelled test code(s) %s heard by the English recogniser (%r) but not in the %s text (%r)",
                    session.call_id, add, english_text, lang, result.text)
        return dataclasses.replace(result, text=f"{result.text} {' '.join(add)}")
    except Exception as e:  # noqa: BLE001 - advisory: the turn proceeds on the Bengali/Hindi text alone
        logger.warning("[%s] spelled-code rescue skipped: %s", session.call_id, e)
        return result


def _keep_established_bengali(session: CallSession, lang: str, result, candidates) -> tuple[str, object]:
    """Owner, 2026-10-28 ("to ekta naam likhe din" was heard as the Hindi "तो आपका नाम लेकर रही हो"): in a call that is ALREADY Bengali, a turn picked as Hindi stays Bengali unless Hindi clearly beats
    Bengali's own decoder agreement (by 0.15) -- language ID is unreliable on telephone-quality Bengali and the Hindi decoders can agree 1.00 on words never said. Used on every route that can pick
    Hindi (the single-recogniser path has the same rule inline). `candidates` = [(lang, ASRResult), ...] that actually ran."""
    if lang != "hi" or (session.lang_locked and session.lang == "hi") or session.lang_explicit:
        return lang, result
    bn = next((r for lg, r in candidates if lg == "bn"), None)
    if bn is None:
        return lang, result
    bn_agree, hi_agree = float(bn.decoder_agreement or 0.0), float(getattr(result, "decoder_agreement", 0.0) or 0.0)
    logger.info("[%s] Hindi picked in a Bengali call: Hindi %.2f / Bengali %.2f decoder agreement", session.call_id, hi_agree, bn_agree)
    if bn_agree >= 0.75 and bn_agree >= hi_agree - 0.15 and getattr(bn, "text", "").strip():
        logger.info("[%s] the call is Bengali and Bengali's reading stands: Bengali", session.call_id)
        return "bn", bn
    return lang, result


async def _route_and_transcribe(session: CallSession, utterance_wav: str):
    """LID -> routing decision -> ASR. Returns (language, ASRResult, decisive), or
    (None, None, True) when the router says a human should take the call.

    `decisive` (real call 2026-10-12): False only for the tie-break fallback below, when LID itself could not
    settle on a language and this turn's language was picked by decoder agreement among what was tried. On a
    short, noisy or badly-timed first utterance that is often just whichever engine hallucinated the most
    confident-sounding garbage ("hano parking a chair", chosen over Bengali's own 1.00 decoder agreement on a
    later, clearly Bengali turn) -- not a real signal of what language the caller speaks. The caller
    (_resolve_reply_language) uses this to avoid LOCKING the whole call's reply language off a guess like that."""
    if _lid is None or len(_languages_active) < 2:
        return "bn", await _asr_router.transcribe("bn", utterance_wav), True

    # Language ID (CPU, ~0.19 s) and recognition used to run one after the other. They now overlap: the recognisers
    # that are almost always needed -- the language this call has been in, and English (cheap, and never ruled out by
    # language ID) -- start at the SAME moment as language ID, and any other one is started the moment it says so.
    # A recogniser nobody needs is simply not run (agent/lang_select.engines_needed): Hindi's ~0.6 s of GPU time was
    # being spent on every Bengali turn for a result that could never win.
    prior = session.lang_router.previous_language or "bn"
    tasks: dict[str, asyncio.Future] = {}
    took: dict[str, float] = {}

    def start(lang: str) -> None:
        if lang in tasks or lang not in _languages_active:
            return

        async def run(lang=lang):
            t0 = time.perf_counter()
            try:
                return await _asr_router.transcribe(lang, utterance_wav)
            finally:
                took[lang] = (time.perf_counter() - t0) * 1000

        tasks[lang] = asyncio.ensure_future(run())
        tasks[lang].add_done_callback(lambda f: f.cancelled() or f.exception())  # an unused failure is not an error

    for lang in dict.fromkeys((prior, "en")):
        start(lang)
    try:
        lid = await asyncio.to_thread(_lid.identify_path, utterance_wav)
    except Exception as e:  # noqa: BLE001 - a LID fault must degrade the turn, not kill it
        logger.warning("[%s] LID failed (%s) -- treating as unknown", session.call_id, e)
        lid = LIDResult(language="unknown", confidence=0.0)
    _mark(session, "lid")

    async def outcomes_for(langs: list[str]) -> list[tuple[str, object]]:
        for lang in langs:
            start(lang)
        done = await asyncio.gather(*[tasks[lang] for lang in langs if lang in tasks], return_exceptions=True)
        good, order = [], [lang for lang in langs if lang in tasks]
        for lang, out in zip(order, done):
            if isinstance(out, BaseException):
                logger.warning("[%s] ASR %s failed: %s", session.call_id, lang, out)
            else:
                good.append((lang, out))
        if not good:
            raise next(o for o in done if isinstance(o, BaseException))
        logger.info(
            "[%s] ASR timings: %s", session.call_id, " | ".join(f"{lg} {int(took.get(lg, 0))} ms" for lg, _ in good)
        )
        return good

    # LID's top label alone is not trusted: Indian-accented English is labelled Hindi at ~0.9, or not found at all
    # (measured; see agent/lang_select.py). Unless LID is decisive, run the recognisers that can still matter and keep
    # the one whose decoders agree AND whom language ID believes.
    verify = languages_to_verify(lid.language, lid.scores, _languages_active) if lid.scores else None
    if verify:
        need = engines_needed(lid.language, lid.scores, _languages_active, prior)
        outcomes = await outcomes_for(need)
        lang, result = pick_candidate(outcomes, lid.scores, session.lang_router.previous_language)
        lang, result = _keep_established_bengali(session, lang, result, outcomes)
        logger.info(
            "[%s] LID %s %s not decisive -> ran %s, chose %s (%s)",
            session.call_id,
            lid.language,
            {k: round(v, 2) for k, v in lid.scores.items()},
            "+".join(lg for lg, _ in outcomes),
            lang,
            ", ".join(f"{lg}:{r.decoder_agreement:.2f}" for lg, r in outcomes),
        )
        session.lang_router.note_response_language(lang)
        # the spelled-code rescue (English recogniser keeps "C B C" apart) on THIS path too: it was missing here, and this is the path a
        # sentence with spelled letters in it takes, because language ID is unsure of it (real call 2026-10-17: 5 turns, nothing captured)
        english = next((r for lg, r in outcomes if lg == "en"), None)
        if lang in ("bn", "hi") and english is not None:
            result = await _with_spelled_tests(session, lang, result, english.text)
        # Real call 2026-10-17: language ID wobbles on accented English (Hindi 0.97 / English 0.03 is typical), so this branch ran,
        # picked English correctly, and returned "not decisive" -- which left the call's reply language unset, so an English question
        # got a Bengali answer. A CLEAN English sentence from the English recogniser (the same bar as the overrule below) is decisive.
        # A garbled first utterance that merely tie-breaks to English is still not (real call 2026-10-12).
        return lang, result, (lang == "en" and english_overrules_language_id(result))

    decision = session.lang_router.route(lid)
    logger.info(
        "[%s] LID %s %.2f -> %s %s (%s)",
        session.call_id,
        lid.language,
        lid.confidence,
        decision.action,
        decision.language or "",
        decision.reason,
    )

    if decision.action == "handoff_human":
        return None, None, True

    if decision.action == "dual_asr" and decision.secondary_language:
        primary, secondary = decision.language, decision.secondary_language
        both = await outcomes_for([primary, secondary])
        if len(both) == 2:
            lang, result = pick_candidate(both, lid.scores, session.lang_router.previous_language)
            lang, result = _keep_established_bengali(session, lang, result, both)
        else:
            lang, result = both[0]
        # the same spelled-code rescue as the single-language path below: the English recogniser keeps "C B C" apart
        english = next((r for lg, r in both if lg == "en"), None)
        if lang in ("bn", "hi") and english is not None:
            result = await _with_spelled_tests(session, lang, result, english.text)
        return lang, result, True

    chosen, result = decision.language, (await outcomes_for([decision.language]))[0][1]
    # Owner, 2026-10-29: the English recogniser is only waited for when it could change the answer. Language ID sure of the language it picked (>= 0.95, English below 0.05), the decoders agreeing
    # (>= 0.95) and a real sentence (>= 3 words) leave nothing for it to overrule; that wait cost ~100 ms a turn. Accented English that language ID labels Hindi does not reach this rule: its Hindi
    # decoders do not agree that well.
    _sure = (
        chosen in ("bn", "hi")
        and lid.language == chosen
        and lid.confidence >= 0.95
        and (lid.scores or {}).get("en", 0.0) < 0.05
        and float(getattr(result, "decoder_agreement", 0.0) or 0.0) >= 0.95
        and len((getattr(result, "text", "") or "").split()) >= 3
    )
    if _sure:
        logger.info("[%s] English recogniser not waited for (language ID %s %.2f, decoders agree %.2f)", session.call_id, chosen, lid.confidence, result.decoder_agreement)
    if chosen in ("bn", "hi") and "en" in _languages_active and tasks.get("en") is not None and not _sure:
        # The English recogniser was started at the same moment as language ID (see the top of this function), so its answer is
        # already here or nearly. A confident but wrong language ID must not hide it (agent/lang_select.py has the evidence).
        try:
            en_result = await asyncio.wait_for(asyncio.shield(tasks["en"]), timeout=1.5)
        except Exception:  # noqa: BLE001 - the English recogniser is optional here; its absence changes nothing
            en_result = None
        if en_result is not None and english_overrules_language_id(en_result):
            logger.info(
                "[%s] English recogniser overruled a confident LID %s (%d words, decoder agreement %.2f)",
                session.call_id, chosen, len(en_result.text.split()), en_result.decoder_agreement,
            )
            session.lang_router.note_response_language("en")
            return "en", en_result, True
        if en_result is not None:
            result = await _with_spelled_tests(session, chosen, result, en_result.text)
    agree = float(getattr(result, "decoder_agreement", 1.0) or 0.0)
    if chosen == "hi" and "bn" in _languages_active and prior == "bn" and tasks.get("bn") is not None and agree >= LOW_AGREEMENT_FOR_LOCK:
        # Live replay, 2026-10-28: two Bengali clips were labelled Hindi by language ID at 0.98 and 0.997 (telephone-quality speech). The Hindi decoders still agreed (0.81, 0.84), so the check
        # below (agreement under 0.5) never ran, and the Bengali call was answered in Hindi. The Bengali recogniser was already running (it is the call's own language, started with language ID),
        # so comparing costs nothing: where it agrees with ITSELF better than Hindi does (0.97 against 0.81; 0.88 against 0.84), the speech is Bengali. A call that is not in Bengali is not touched.
        try:
            bn_early = await asyncio.wait_for(asyncio.shield(tasks["bn"]), timeout=1.5)
        except Exception:  # noqa: BLE001 - optional check
            bn_early = None
        if bn_early is not None:
            bn_agree = float(bn_early.decoder_agreement or 0.0)
            # Owner, 2026-10-28 ("to ekta naam likhe din" was heard as Hindi "तो आपका नाम लेकर रही हो"): language ID is not reliable on telephone-quality Bengali, and the Hindi decoders can agree
            # 1.00 on words that were never said. In a call that is ALREADY Bengali, a single Hindi-labelled turn must show clear evidence to count: Hindi has to beat Bengali's own agreement by 0.15.
            # A call not yet settled on Bengali keeps the stricter rule (Bengali better by 0.03).
            established = not (session.lang_locked and session.lang == "hi")  # owner, 2026-10-29: Bengali until the call is really Hindi -- an unlocked call counts as Bengali too
            margin = -0.15 if established else 0.03
            logger.info("[%s] Hindi %.2f / Bengali %.2f decoder agreement (established Bengali call: %s)", session.call_id, agree, bn_agree, established)
            if bn_agree >= 0.75 and bn_agree >= agree + margin:
                logger.info("[%s] language ID said Hindi but the Bengali reading stands (%.2f against %.2f): Bengali", session.call_id, bn_agree, agree)
                session.lang_router.note_response_language("bn")
                return "bn", bn_early, True
    if chosen == "hi" and "bn" in _languages_active and agree < LOW_AGREEMENT_FOR_LOCK:
        # Owner, 2026-10-28: Bengali speech that language ID called Hindi at 0.99 came back as Devanagari nonsense ("पुलच हम य शगरटकताता", decoder agreement 0.30), and that guess then answered the whole
        # call in Hindi. A Hindi read the two decoders do not agree on is checked against Bengali's; the better-agreeing one is used, and neither locks the call's language.
        try:
            bn_result = (await outcomes_for(["bn"]))[0][1]
        except Exception:  # noqa: BLE001 - the check is optional
            bn_result = None
        if bn_result is not None and float(bn_result.decoder_agreement or 0.0) > agree:
            logger.info("[%s] doubtful Hindi read (agreement %.2f) beaten by Bengali (%.2f): Bengali", session.call_id, agree, bn_result.decoder_agreement)
            session.lang_router.note_response_language("bn")
            return "bn", bn_result, float(bn_result.decoder_agreement or 0.0) >= LOW_AGREEMENT_FOR_LOCK
    return chosen, result, (agree >= LOW_AGREEMENT_FOR_LOCK) if chosen in ("bn", "hi") else True


async def _dispatch_turn(session: CallSession, utterance_wav: str):
    """One full turn: LID -> ASR -> intent -> tool -> templated reply -> TTS.
    Serialized per-call via session.dispatch_lock so replies never
    interleave, even if the caller starts talking again immediately."""
    async with session.dispatch_lock:
        session.turn_epoch = session.speak_epoch  # this turn is current; an interrupt from now on makes it stale
        session.acks.next_turn()
        session.turn_acknowledged = False
        session.turn_spoken = False
        session.reply_noted = False
        prior_lang = session.lang  # the language the call was in BEFORE this turn's recognition (see _keep_language)
        # An answer to a question we asked: the last thing we said had a question mark, or a flow that asks in commands
        # ("please tell me your full name") is waiting on this very turn.
        session.answering = (
            session.awaiting_answer
            or session.history_state in ("need_phone", "need_name")
            or (session.security is not None and not session.security.finished)
        )
        session.awaiting_answer = False
        session.silence_since = None
        session.turn_started_at = time.monotonic()
        session.marks = [("start", session.turn_started_at)]
        channel_quality = session.call_state.channel_quality
        analysis: dict | None = None
        audio_issues: list[str] = []
        asr_wav = utterance_wav
        temp_wavs: list[str] = []
        try:
            if NEAR_END_ATTENTION != "off" and session.near_end.established:
                attended = await asyncio.to_thread(
                    _attend_wav_to_path, utterance_wav, session.near_end.level_dbfs, session.near_end.f0_hz or 0.0
                )
                if attended:
                    temp_wavs.append(attended)
                    asr_wav = attended
            if CONDITION_INPUT != "off" or session.noise_suppress:
                conditioned = await asyncio.to_thread(_condition_wav_to_path, asr_wav, CONDITION_INPUT if CONDITION_INPUT != "off" else "full")
                if conditioned:
                    temp_wavs.append(conditioned)
                    asr_wav = conditioned
            _mark(session, "prep")
            _keep_recent_utterance(session, utterance_wav)
            lang, asr_result, lang_decisive = await _route_and_transcribe(session, asr_wav)
            logger.info("[%s] heard (%s, %s, agreement %.2f): %r", session.call_id, lang, getattr(asr_result, "decoder_used", "?"),
                        float(getattr(asr_result, "decoder_agreement", 0.0) or 0.0), getattr(asr_result, "text", ""))
            _mark(session, "asr")
            # KCD-075/KCD-076: measured inside its own budget slice, on
            # the SAME clip ASR already read, before the finally below
            # deletes it. A slow or failed classification just keeps
            # last turn's label -- never blocks or fails the turn.
            channel_quality = await run_within_budget(
                "channel_quality",
                CHANNEL_QUALITY_BUDGET_S,
                _classify_channel_from_wav_path,
                utterance_wav,
                previous_value=channel_quality,
            )
            # Why is this clip hard to hear, and does the speaker sound older?
            # A slow or failed analysis just means no verdict this turn.
            analysis = await run_within_budget(
                "audio_quality",
                AUDIO_ANALYSIS_BUDGET_S,
                _analyze_utterance_from_wav_path,
                utterance_wav,
                previous_value=None,
            )
            audio_issues = analysis["audio"].issues if analysis else []
            for issue in audio_issues:
                audio_issue_buckets.record(issue, "seen", lang or session.lang)
            # Appendix F channel buckets, on the same vocabulary as the offline evaluation:
            # the live rate of cross_talk / noisy / narrowband turns (KCD-053/055).
            golden_buckets.record("turn", "analysed", lang or session.lang)
            for bucket in channel_buckets(audio_issues, channel_quality):
                golden_buckets.record(bucket, "turn", lang or session.lang)
            asr_result = await _retry_on_enhanced_audio(
                session,
                utterance_wav,
                lang,
                asr_result,
                audio_issues,
                analysis["audio"].duration_s if analysis else None,
            )
        except Exception as e:
            logger.exception("[%s] ASR/LID stage failed: %s", session.call_id, e)
            await _speak(session, phrase("llm_failure", session.lang), session.lang, fallback_reason="llm_failure")
            return
        finally:
            with contextlib.suppress(OSError):
                os.remove(utterance_wav)
            for extra in temp_wavs:
                with contextlib.suppress(OSError):
                    os.remove(extra)

        # KCD-053: a much quieter utterance in a DIFFERENT voice is someone else in the room,
        # not the caller. Its WORDS are never used -- the turn does not open on them. The same
        # voice, merely quieter, is still the caller and goes on to the empathetic re-ask.
        if analysis and analysis.get("near_end"):
            level_dbfs, f0_hz = analysis["near_end"]
            verdict = session.near_end.judge(level_dbfs, f0_hz)
            if verdict.background:
                session.near_end.reject()
                audio_issue_buckets.record("background_utterance", "dropped", lang or session.lang)
                logger.info("[%s] background utterance dropped (%s)", session.call_id, verdict.reason)
                if asr_result is not None and await _name_rescue(session, (asr_result.text or "").strip(), session.lang):
                    return
                # Owner, 2026-10-26: this used to just `return` -- real calls then sat in total silence, which
                # sounds exactly like the agent having stopped, not like it noticed someone else was talking. The
                # content is still discarded (the anti-hijack protection above is unchanged); the caller now always
                # hears something, the same gentle re-ask a cross-talk turn gets anywhere else, never dead air.
                await _reask_or_handoff(session, session.reask.decide(asr_empty=True, audio_issues=["crosstalk"]), session.lang)
                return
            session.near_end.accept(level_dbfs, f0_hz)

        if lang is None:
            # Ask again, kindly, before any hand-off -- an unidentifiable
            # utterance is usually a faint, noisy or mumbled one, not a
            # request for a person (agent/reask_policy.py).
            await _reask_or_handoff(
                session,
                session.reask.decide(language_ambiguous=True, audio_issues=audio_issues),
                session.lang,
                languages=_languages_active,
            )
            return
        detected_lang = lang  # what the CALLER actually spoke this turn -- kept for analytics; never overridden
        session.detected_input_language = detected_lang
        if session.lang_locked and detected_lang != session.lang:
            session.language_mismatch_turns += 1
        _rec(session, "language", detected_lang)
        lang = _resolve_reply_language(
            session, lang, lang_decisive, float(getattr(asr_result, "decoder_agreement", 1.0) or 0.0), asr_result.text
        )  # what the AGENT replies in -- sticky; see the function
        _heard = (asr_result.text or "").strip()
        _digits_said, _digit_hits, _digit_total = booking_capture.read_digits(_heard)
        _short_booking_answer = (
            (_booking_open(session) or _readback_pending(session))
            and len(_heard.split()) < 6
            and not detect_language_switch_request(_heard, detected_lang)
        ) or (
            # Owner, 2026-10-26: "does this number have WhatsApp?" often gets a reassuring run-on answer
            # ("whatsapp ache, ei number e whatsapp ache") that is 6+ words long -- still just this one
            # question's yes/no, never a real language switch, so the word-count limit above must not apply here.
            _whatsapp_confirm_pending(session)
            and not detect_language_switch_request(_heard, detected_lang)
        )
        if lang != prior_lang and _short_booking_answer:
            # Owner, 2026-10-24 (three real calls): a name said in Devanagari ("हा असना रबी बी" = Hasna Bibi), a garbled yes ("अच्छे अच्छे है") and a police station
            # ("english rajah" = English Bazar) each moved the whole call to Hindi or English in the middle of a booking. The short answers of a booking say nothing about the
            # language; only an explicit request, or a long clear sentence, can move it.
            lang = _keep_language(session, prior_lang)
            detected_lang = lang
        if len(_digits_said) >= 3 and _digit_hits * 2 >= _digit_total:
            # Real call 2026-10-19: a Bengali caller said the phone number's digit words and the recogniser wrote them in Devanagari ("नाइन थ्री फोर ...");
            # the reply language jumped to Hindi for the rest of the call. Digit words say nothing about the language: it stays what it was.
            lang = _keep_language(session, prior_lang)
            detected_lang = lang
        # Owner, 2026-10-27: removed the "one confident, clear, 4+ word sentence in the other script instantly
        # relocks the call" shortcut that used to live here. Only two things may ever move the reply language once
        # a call is running: two turns in a row in the other language (asked, then switched on the caller's yes --
        # the block below, and _continue_language_switch_offer) or an explicit request (detect_language_switch_
        # request). This shortcut was a third path firing on ANY ordinary, confidently-heard sentence in the other
        # script, never mind whether it named a language or asked for anything -- exactly the unwanted mid-call
        # flip the owner reported (2026-10-26 voice note). The two-turns path already recovers a genuinely
        # mislocked call (its original motivation, a bad first-turn lock), just with a confirmation step instead of
        # an instant, unasked one.
        # Owner, 2026-10-27: the one flip that IS wanted: a call sitting in ENGLISH that nobody asked for (the first turn's language ID was wrong) while the caller speaks a clear sentence in
        # Bengali or Hindi -- four words or more, nearly all in that script, decoders agreeing -- is answered in the caller's language. A call where English was requested is never moved.
        if (
            lang == "en"
            and detected_lang in ("bn", "hi")
            and not session.lang_explicit
            and not _short_booking_answer
            and len(_heard.split()) >= 4
            and script_share(_heard, detected_lang) >= 0.85
            and asr_result.decoder_agreement >= 0.85
        ):
            logger.info("[%s] an unrequested English call and a clear %s sentence: the reply language moves to %s", session.call_id, detected_lang, detected_lang)
            _lock_reply_language(session, detected_lang)
            session.lang_router.note_response_language(detected_lang)
            lang = detected_lang
        if (
            session.lang_lock_weak
            and lang != detected_lang
            and detected_lang in ("bn", "hi")
            and not session.lang_explicit
            and not _short_booking_answer
            and len(_heard.split()) >= 4
            and script_share(_heard, detected_lang) >= 0.85
            and asr_result.decoder_agreement >= 0.85
        ):
            logger.info("[%s] the call was locked from a doubtful read and a clear %s sentence followed: the reply language moves to %s", session.call_id, detected_lang, detected_lang)
            _lock_reply_language(session, detected_lang)
            session.lang_lock_weak = False
            session.lang_router.note_response_language(detected_lang)
            lang = detected_lang
        confidence = confidence_state(getattr(asr_result, "decoder_used", None), asr_result.decoder_agreement)
        apply_confidence(session.call_state, confidence != VERIFIED)
        apply_channel_quality(session.call_state, channel_quality)
        channel_quality_buckets.record(channel_quality, "seen", lang)
        logger.debug("[%s] call_state %s", session.call_id, session.call_state.to_log_dict())

        if ASR_REFINE != "off":
            await _asr_refine_shadow(session, asr_result, detected_lang or lang)
        text = asr_result.text.strip()
        session.last_agreement = float(getattr(asr_result, "decoder_agreement", 1.0) or 0.0)  # how sure the recognisers were about THIS turn (state is not changed by a guess)
        if detected_lang == "en":
            text = fix_asr_confusables(text)
        if text:
            text = imaging_words.normalise(text)  # "উসগ" is USG, "থাইরাইড গ্লিন্ট" is the thyroid gland: one spelling for the whole pipeline (real call 2026-10-19)
            text = await _normalise_spelled(text)  # letters said one by one ("টি এস এইচ") are the code they spell ("TSH") for the whole turn
        if not text:
            logger.info("[%s] ASR returned empty text", session.call_id)
            lang = _keep_language(session, prior_lang)
            await _reask_or_handoff(
                session,
                session.reask.decide(asr_empty=True, audio_issues=audio_issues),
                lang,
                fallback_reason="asr_empty",
            )
            return
        # A possible medical emergency outranks everything: it is checked on EVERY turn, in every
        # language, at any ASR confidence, before any re-ask, model call or booking step, and it
        # abandons whatever was in progress (agent/emergency.py).
        if detect_emergency(text):
            logger.warning("[%s] possible emergency in the transcript", session.call_id)
            await session.send_json("User", text)
            session.turn_count += 1
            session.booking, session.pending_entity, session.history_state = None, None, None
            _note(session, "emergency")
            apply_caller_state(session.call_state, "emergency")
            await _speak(session, phrase("emergency_notice", lang), lang)
            await _handoff_to_human(session, "emergency", (lang,))
            session.outcome = "emergency"
            return
        # Swearing at the agent: a calm fixed boundary in the caller's own language, never the model (agent/abuse.py);
        # the third time in one call, a courteous close.
        if abuse.is_abusive(text):
            session.abuse_count += 1
            _note(session, "abuse_turns")
            session.signals.turns += 1
            reply_lang = _abuse_reply_language(text, lang, prior_lang)
            _keep_language(session, reply_lang)
            logger.info("[%s] abusive turn %d", session.call_id, session.abuse_count)
            await session.send_json("User", text)
            session.turn_count += 1
            # Owner, 2026-10-27: told ONCE, and the next non-clinic turn (more abuse, an insult, small talk, "I did not understand") ends the call with a thank-you; the old third-strike rule let it go on.
            in_a_row = session.offtopic_streak > 0 and session.turn_count - session.offtopic_last_turn <= 1
            session.offtopic_streak = session.offtopic_streak + 1 if in_a_row else 1
            session.offtopic_last_turn = session.turn_count
            if abuse.closes_the_call(session.abuse_count) or session.offtopic_streak >= off_topic.CLOSE_AT:
                session.outcome = "abusive_caller"
                await _end_call(session, reply_lang, "offtopic_final")
            else:
                await _speak(session, phrase(abuse.response_key(session.abuse_count), reply_lang), reply_lang)
                session.awaiting_answer = False
            return
        # Owner, 2026-10-27: a frightened or worn-out caller (agent/distress.py) hears the approved acknowledgement FIRST, whichever route the turn then takes, and the calm delivery from then on.
        # An emergency and abuse have both returned above.
        _note_distress(session, text, lang, analysis)
        await _acknowledge_if_due(session, lang)
        # A yes/no to a confirmation is legitimately one short word; the
        # jumbled-transcript checks would misread it as a fragment.
        # The same holds for a dictated phone number, a spelled name or a bare
        # "না" while a booking is collecting slots: all legitimately short or
        # tokenised, so only decoder disagreement is meaningful then.
        in_booking_flow = session.booking is not None and (
            session.booking.stage in ("confirming", "awaiting_charge_confirm") or not session.booking.is_stale()
        )
        problem = transcript_problem(
            text,
            detected_lang,  # the script this text was actually RECOGNISED in, not the (now sticky) reply language
            analysis["audio"].duration_s if analysis else None,
            asr_result.decoder_agreement,
            slot_answer=in_booking_flow
            or session.pending_entity is not None
            or session.awaiting_handoff_offer
            or session.history_state is not None
            or session.awaiting_draft_answer
            or session.awaiting_pref_answer
            or session.awaiting_resume
            or session.awaiting_close_answer
            # Real call 2026-10-14: "Would you like me to continue in English?" -- the caller said a clear "yes" and it
            # was logged "jumbled transcript (fragment)" and thrown away before the offer handler ever saw it. A bare
            # yes/no is the whole point of that question.
            or session.awaiting_language_switch_offer
            # Owner, 2026-10-29: "ও দিপ শুভজিৎ দিপ" (a doctor's first name, said as the caller can) was thrown away as "repetitive"; while a department's doctors are the subject, a name is an answer
            or ((getattr(session, "last_route", None) is not None or getattr(session, "dept_pending", None) is not None) and float(asr_result.decoder_agreement or 0.0) >= 0.5),
        )  # KCD-104; "anything else?" -> a bare no: a bare yes or no to "shall I go back to it" is an answer
        if problem:
            logger.info("[%s] jumbled transcript (%s)", session.call_id, problem)
            lang = _keep_language(session, prior_lang)
            if await _name_rescue(session, text, lang):
                return
            await _reask_or_handoff(
                session, session.reask.decide(transcript_issue=problem, audio_issues=audio_issues), lang
            )
            return
        session.reask.note_success()
        session.signals.turns += 1
        # The CALLER's language history (for the flip-detection signal below and call_score.py's language_flips) --
        # what was actually SAID each turn, not the (now sticky) reply language, which barely varies by design.
        session.lang_history.append(detected_lang)
        if (
            len(session.lang_history) >= 3
            and session.lang_history[-1] == session.lang_history[-3] != session.lang_history[-2]
        ):
            _note(session, "language_flips")  # A, B, A: the bot moved the call to B and the caller went back
        if _caller_thanked(text, detected_lang):  # detecting a thank-you in what was actually said, not the reply lang
            _note(session, "caller_thanked")
        session.silence_prompts = 0
        await session.send_json("User", text)
        session.turn_count += 1

        # KCD-438: an explicit "speak in Hindi/Bengali/English" request, detected deterministically (no LLM
        # call) against the SCRIPT this utterance was actually said in (`detected_lang`) -- NOT `lang`
        # (_resolve_reply_language's STICKY result), which once the call is locked no longer reflects what
        # this turn actually said and would check the wrong language's phrase table entirely. Checked, and
        # if matched LOCKED, before the two-consecutive-turns offer just below: real call 2026-10-13, a
        # caller who had already said "speak in Hindi" was still asked "shall I speak in Hindi?", and that
        # offer's own confirmation step then mis-fired on an unrelated filler reply (see
        # _continue_language_switch_offer) -- an explicit request must never be made to wait for a second
        # turn or a separate yes/no. `lang` is updated here too, so the offer's own trigger condition
        # (`detected_lang != session.lang`, now false) and every check below this point -- the disclosure-
        # once line, a pending security/draft/history continuation -- already see the caller's requested
        # language. The actual "okay, switching" acknowledgement is spoken further down (KCD-438, unchanged
        # position), once it is clear nothing else already consumed this turn.
        switch_target = detect_language_switch_request(text, detected_lang)
        if switch_target and session.booking is not None and session.booking.slots.get("_ps_pending") and booking_capture.says_english_bazar(text):
            switch_target = None  # Owner, 2026-10-27: asked "which police station?", "english was a" is English Bazar, not a request to switch to English
        if switch_target:
            session.lang_explicit = True  # set only once the request is accepted (review 2026-10-28: it used to be set before a place-name false positive was rejected)
        if switch_target:
            _lock_reply_language(session, switch_target)
            session.lang_router.note_response_language(switch_target)
            lang = switch_target

        # OWNER'S INSTRUCTION, 2026-10-10: exactly two turns in a row in the same OTHER language -- not one
        # (a stray name/number), not already offered this call for this language (lang_switch_last_offered) --
        # asks once, in that language, instead of answering this turn's content; the answer is read by
        # _continue_language_switch_offer on the turn after. Checked before anything else can consume this turn.
        if (
            not session.awaiting_language_switch_offer
            and not _readback_pending(session)  # a name/number read-back is waiting for its yes/no: a garbled short answer is not a language change
            and not _short_booking_answer  # nor the short answers of a booking in progress (real call 2026-10-24: "english rajah" for English Bazar, then a garbled yes)
            and not switch_target  # Real call 2026-10-17: an EXPLICIT "speak Bengali" said in Hindi was answered with "shall I speak Hindi?" (the offer fired on top of the request)
            and detected_lang != session.lang
            and detected_lang in _languages_active
            and len(session.lang_history) >= 2
            and session.lang_history[-1] == session.lang_history[-2] == detected_lang
            and session.lang_switch_last_offered != detected_lang
        ):
            session.pending_language_switch = detected_lang
            session.pending_language_switch_text = text
            session.pending_language_switch_asr_result = asr_result
            session.awaiting_language_switch_offer = True
            session.lang_switch_last_offered = detected_lang
            await _speak(session, phrase("offer_switch", detected_lang), detected_lang)
            return

        after_prompt, session.awaiting_close_answer = session.awaiting_close_answer, False
        answer = classify_yes_no(text, lang) if after_prompt else None
        # OWNER'S INSTRUCTION, 2026-10-10 (real call): "वो मेरे पास कैश तो नहीं है मैं ऑनलाइन पाय्मेंट कर दूंगा चलेगा"
        # (a real question about paying online, "नहीं" only inside "कैश तो नहीं है" = "don't have cash") was heard as
        # the answer to "anything else?", and classify_yes_no -- built to read a SHORT yes/no reply -- found "नहीं"
        # as a token anywhere in the sentence and returned "no", ending the call before the question was ever
        # answered. A bare "no" ends the call; a LONG sentence that merely contains a negation word, with real
        # content around it, is a missed turn's worth of content, not a goodbye -- same "bare word vs. a real
        # sentence" distinction agent/booking_flow.py already draws for a bare "cancel" (see its own docstring).
        # SHORT_NO_WORDS: an ordinary short decline ("नहीं", "না না ধন্যবাদ", "no thanks") is at or under this many
        # words; the caller's real question above was 11. Not a measured threshold -- reasoned from this one
        # real miss, the same discipline this codebase already applies to its other reasoned-not-measured floors.
        bare_no = answer == "no" and len(text.split()) <= 5
        if call_end.wants_to_end(text, lang, after_prompt=after_prompt) or bare_no:
            _note(session, "ended_by_caller")
            await _end_call(session, lang, "silence_goodbye")
            return
        # Owner, 2026-10-23 (item 6): "achha thik ache" / "ok" / "thik hai" and nothing more, with nothing open, is a goodbye -- said and the call cut at once.
        if call_end.is_closing_acknowledgement(text, lang) and (after_prompt or _nothing_is_open(session)):
            _note(session, "ended_by_caller")
            await _end_call(session, lang, "silence_goodbye")
            return
        # Same "bare word vs. a real sentence" distinction as bare_no just above, mirrored for "yes": a short "yes" /
        # "হ্যাঁ" alone just invites the caller to go on, but a longer sentence that also NAMES what they want
        # ("হ্যাঁ হ্যাঁ জানতে চাই বলছি যে আমার একটা বুকের ডাক্তারের অ্যাপয়েন্টমেন্ট লাগবে...") is the real request
        # itself -- real call 2026-10-24: it was answered with only "yes, go on", throwing away the request and
        # making the caller repeat the whole thing. Falls through to the normal turn-handling below instead.
        if answer == "yes" and len(text.split()) <= 5:
            await _speak(session, phrase("silence_go_on", lang), lang)
            return
        code_switch_buckets.record(mixture_bucket(text), "seen", lang)
        if await _clinical_and_unrelated_guards(session, text, lang):
            return

        # The requests that must never reach the intent model, checked in ONE fixed order (agent/early_routing.py -- shared with main_pcm.py so the two
        # entry points cannot drift apart): a person (KCD-353), the doctor himself, a complaint, money still to pay, a health package, a report, another
        # center's referral. Deterministic, before any model call, before intent extraction.
        # sourav, 2026-10-24: (1) a caller in the middle of the 'find my serial number' conversation, (2) a caller who starts it ('I paid but the serial did not come on WhatsApp'),
        # (3) 'is the CT scan done on Thursday?' from the lab table. Each is checked BEFORE early_routing.classify because classify would send (2) to the generic 'after you pay
        # you will get your serial on WhatsApp' reply and (3) to the availability/price paths; each returns False to fall through to the old path when it does not apply.
        if session.serial_flow is not None:
            await _serial_flow_turn(session, text, lang, answer_hint=classify_yes_no(text, lang))
            return
        if serial_lookup.reports_serial_not_received(text, lang) and _tools is not None:
            session.serial_flow = serial_lookup.new_flow(session.turn_count)
            await _speak(session, serial_lookup.ask_number(lang), lang)
            return
        if await _answer_test_days(session, text, lang):
            return
        # sourav, 2026-10-24: the report-visit question under way; a report to be shown; the serial a booking made now would get; 'I will come in the afternoon'.
        if session.report_visit_asked is not None and await _report_visit_turn(session, text, lang):
            return
        if _tools is not None and visit_cues.wants_to_show_a_report(text) and not _booking_is_open(session):
            session.report_visit_asked = {"turn": session.turn_count, "tries": 0}
            await _speak(session, visit_cues.ask_visit_reply(lang), lang)
            return
        if _tools is not None and visit_cues.asks_next_serial(text) and await _answer_next_serial(session, text, lang):
            return
        if visit_cues.asks_booking_window(text) and not _booking_is_open(session) and early_routing.classify(text, lang) is None:
            await _speak(session, visit_cues.booking_window_reply(lang), lang)  # sourav, 2026-10-24: 'how many days ahead can I book?' -- the same 3-day rule the clinic API enforces
            return
        if _nothing_is_open(session) and visit_cues.says_will_come_in_the_afternoon(text) and early_routing.classify(text, lang) is None:
            await _speak(session, visit_cues.afternoon_reply(lang), lang)
            return
        if _tools is not None and doctor_origin.asks_doctor_origin(text, has_context=booking_carryover.recent(session.last_doctor_enquiry, session.turn_count)) and await _answer_doctor_origin(session, text, lang):
            return
        early = early_routing.classify(text, lang)
        if early == early_routing.SITTING_HOURS and not await _answer_sitting_hours(session, text, lang):
            early = None  # a lab test is named, or no doctor's hours can be told: the normal path
        if early == early_routing.COME_TO_CLINIC and not await _answer_come_to_clinic(session, text, lang):
            early = None  # a lab test is booked at the counter anyway (lab_test_counter_only): the normal path says so
        if early == early_routing.WRITE_NAME and not await _answer_write_name(session, text, lang):
            early = None  # a doctor, a test or a day is already in the sentence (or a booking is open): the booking flow asks for what is missing
        if early == early_routing.DAY_AVAILABILITY and not await _answer_day_availability(session, text, lang):
            early = None  # a lab test is named, or the weekday is the booking's own date answer: the normal path
        if early == early_routing.WHEN_TO_COME and not await _answer_when_to_come(session, text, lang):
            early = None  # no doctor whose hours can be told (a lab test, say): the turn goes on to the normal path
        if early is not None:
            logger.info("[%s] early route: %s", session.call_id, early)
            if early in early_routing.HANDOFF_REASON:
                await _handoff_to_human(session, early_routing.HANDOFF_REASON[early], (lang,))
            elif early == early_routing.DUES:
                # KCD-lead-9: never a balance read over the phone -- proof on WhatsApp, a human calls back. A callback is queued only when a phone is
                # already known (no caller-ID in this stack, same rule _handoff_to_human follows).
                await _speak(session, owner_answers.billing_dues_reply(lang), lang)
                phone = _known_phone(session)
                if phone:
                    try:
                        summary = build_call_summary("billing_dues_proof_requested", lang=lang, phone=phone, call_id=session.call_id)
                        await asyncio.wait_for(_tools.request_callback(phone, session.call_id, "asap", "billing_dues_proof_requested", summary), 4.0)
                    except Exception as e:  # noqa: BLE001 - advisory: the caller was already told what to do, a failed queue must not break the call
                        logger.warning("[%s] could not queue the billing-dues callback: %s", session.call_id, e)
            elif early == early_routing.HEALTH_PACKAGE:
                await _speak(session, owner_answers.health_package_reply(lang), lang)
            elif early == early_routing.REPORT_PICKUP:
                # KCD-lead-8: never a guessed status. "today" in the caller's own words -> a human; otherwise call back on the visit day.
                if report_pickup.mentions_today(text, lang):
                    await _handoff_to_human(session, "report_status_on_visit_day", (lang,))
                else:
                    await _speak(session, report_pickup.report_pickup_defer_reply(lang), lang)
            elif early == early_routing.CANNOT_PAY_ONLINE:
                await _speak(session, doctor_enquiry_cues.cannot_pay_online_reply(lang), lang)
            elif early == early_routing.PAY_NOW:
                # sourav, 2026-10-24: "do we have to pay now?" -- owner's answer: not paying now is allowed, the caller can come and book at the clinic later, but slots
                # are not guaranteed. Fixed text (no model); the open booking, if any, is offered back after it.
                await _speak(session, _with_resume(session, doctor_enquiry_cues.pay_now_reply(lang), lang), lang)
            elif early == early_routing.LINK_MISSING:
                await _speak(session, owner_answers.payment_link_missing_reply(lang), lang)
            elif early == early_routing.GOOD_DOCTOR:
                await _answer_good_doctor(session, text, lang)
            elif early == early_routing.QUALITY_REASSURANCE:
                await _speak(session, _with_resume(session, doctor_enquiry_cues.quality_reassurance_reply(lang), lang), lang)
            elif early == early_routing.DOCTOR_WEEKDAYS:
                await _answer_doctor_weekdays(session, text, lang)
            elif early == early_routing.NEXT_IF_ABSENT:
                await _answer_next_if_absent(session, text, lang)
            elif early == early_routing.SERIAL_NUMBER:
                await _speak(session, _with_resume(session, doctor_enquiry_cues.serial_number_reply(lang), lang), lang)
            elif early == early_routing.REFERRAL:
                # KCD-lead-1: another center passing a patient's number. One combined question, answered on the NEXT turn (session.pending_referral).
                session.pending_referral = session.turn_count
                session.referral_notes = text
                await _speak(session, referral_cues.referral_ask_reply(lang), lang)
            return

        # KCD-353: the disclosure is in the greeting (Bengali); a caller who answers in another language
        # hears it once, in theirs -- and ONLY once per call, total, same as a Bengali-only call hears it
        # exactly once. OWNER'S INSTRUCTION, 2026-10-10: a call whose very FIRST utterance locks the reply
        # language to Hindi/English (the common case; see _resolve_reply_language) was hearing the Bengali
        # greeting's disclosure AND THEN this one again, back to back, on that first real turn -- a Bengali-
        # only call never has that second line. `disclosed_langs - {"bn"}` empty means no OTHER-language
        # disclosure has fired yet THIS CALL; once one has, this never fires again regardless of further
        # switches, matching what a Bengali-only call already experiences.
        # OWNER'S INSTRUCTION, 2026-10-14 (sourav): changing the language must NOT repeat the greeting/identity
        # sentence -- the call simply carries on with the context it already has. So the once-only spoken
        # disclosure that used to be said here is gone; the language is still recorded as seen
        # (session.disclosed_langs) so nothing else treats it as new. (disclosure_for() is still used for text channels.)
        if lang != "bn" and not (session.disclosed_langs - {"bn"}):
            session.disclosed_langs.add(lang)

        # KCD-054: after verification, a different voice revokes it. The agent cannot know who
        # is speaking -- only that the voice differs from the one that was verified -- so it
        # says so plainly and stops disclosing personal details until verification is repeated.
        await _check_speaker_change(session, analysis, lang)

        # KCD-494/495/498/500: a question about the caller's OWN history, or an answer to a step of
        # one already under way, is handled here, deterministically, before any model call.
        if session.security is not None and not session.security.finished:
            if await _continue_security_flow(session, text, lang, analysis):
                return
        if session.awaiting_draft_answer and await _continue_draft_offer(session, text, lang):
            return
        if session.awaiting_pref_answer and await _continue_preference_offer(session, text, lang):
            return
        if session.awaiting_language_switch_offer and await _continue_language_switch_offer(session, text, lang):
            return
        if session.awaiting_handoff_offer or session.history_state:
            if await _continue_history_flow(session, text, lang):
                return
        history_question = detect_history_question(text, lang)
        if history_question is not None:
            session.pending_history, session.history_attempts = history_question, 0
            await _history_step(session, lang)
            return

        # KCD-084/149/155: caller state -> delivery policy, and the
        # acknowledgement that comes before anything else when the policy
        # calls for one (once per state entry, not every turn).
        _update_caller_state(session, text, lang, analysis)
        ack = select_acknowledgement(session.policy, session.call_state.caller_state, lang)
        if ack and session.acknowledged_state != session.call_state.caller_state:
            session.acknowledged_state = session.call_state.caller_state
            session.turn_acknowledged = True
            await _speak(session, ack, lang)

        # KCD-438, continued from the top of this function: the explicit switch request detected and
        # LOCKED up there is acknowledged here, now that nothing earlier (a pending security/draft/history
        # continuation) has already consumed this turn with its own reply.
        if switch_target:
            _prev_enquiry = session.last_enquiry
            if _prev_enquiry and session.turn_count - _prev_enquiry["turn"] <= 2:
                # Owner, 2026-10-28: after "বাংলায় বলুন এটা" the answer just given is said again in Bengali (a read-only enquiry only; nothing is ever re-done that writes)
                session.last_enquiry = None
                await _answer_content_turn(session, _prev_enquiry["text"], switch_target, asr_result)
                return
            await _speak(session, phrase("language_switched", switch_target), switch_target)
            return

        # Epic E26: a booking/reschedule/cancel/add-test confirmation
        # already in progress is a closed yes/no question -- answered
        # deterministically (agent/booking_flow.classify_yes_no), with NO
        # LLM round trip, both for the zero-extra-latency requirement and
        # because CLAUDE.md's truth boundary keeps exactly this kind of
        # high-stakes binary decision out of the model's hands.
        if session.awaiting_resume and await _continue_resume_offer(session, text, lang):
            return
        if session.booking is not None and session.booking.stage in ("confirming", "awaiting_charge_confirm"):
            # KCD-104: only a yes or a no is decided here. Anything else -- a question, a correction, another
            # task -- used to be answered by reading the confirmation out again; it is now a fresh turn, and the
            # pending confirmation stays pending (see _with_resume and the "unclear" branch below).
            session.no_at_confirm = False
            if await _handle_booking_confirmation_turn(session, text, lang) is not False:
                await _after_task_finished(session, lang)
                return

        await _answer_content_turn(session, text, lang, asr_result)
        return


async def _answer_content_turn(session: CallSession, text: str, lang: str, asr_result) -> None:
    """Resolve this utterance's content and speak the answer -- the SAME logic a normal turn uses
    after every early-return special case (booking confirmation, pending entity, disclosure, etc.) has
    already been checked. Extracted (2026-10-10) so a confirmed language switch can answer the turn
    that TRIGGERED the offer, in the NEW language, instead of discarding it -- see
    _continue_language_switch_offer. Callable only from inside session.dispatch_lock: both call sites
    already hold it, and this function does not acquire it itself (asyncio.Lock is not reentrant).

    `asr_result` is an explicit parameter, not read off `session`: the confidence/action-gate check below
    needs the recognition result for the TEXT actually being answered, which for a replayed turn (the
    language-switch offer's own call site) is an EARLIER turn's result, not this turn's."""
    text = whatsapp_cues.canonicalise(text)  # owner, 2026-10-23: "botsapp", "whatsap", "ওয়াটসঅ্যাপ", "वाट्सऐप" are all WhatsApp for every cue below
    confirmed_entity = False
    capture_plan = booking_capture.TurnPlan()  # sourav 2026-10-16: nothing decided deterministically unless the booking turn below says so
    chamber_window = None  # sourav 2026-10-16: the doctor's chamber hours on the booking date, when known (reads a bare "ten" as 10:00)
    pending, session.pending_entity = session.pending_entity, None
    if pending is not None:
        answer = classify_yes_no(text, lang)
        if answer in ("yes", "no"):
            session.reasked_pending = 0
        if answer == "yes":
            data, confirmed_entity = pending.data, True
        elif answer == "no" and len(text.split()) <= 5:
            # same "bare word vs. a real sentence" distinction already drawn for the "anything else?" close
            # question a little below (SHORT_NO_WORDS=5, reasoned there from a real call) -- a short decline
            # ("না", "না না", "नहीं") gets the re-ask; a longer sentence that merely contains a negation word but
            # also NAMES what the caller actually meant ("না আমি ইউরিন রুটিন পরীক্ষার কথা বলছি এটার দাম কত বলবেন",
            # correcting a wrong test-name suggestion) falls through to the fresh-turn path below instead of
            # discarding the correction and re-asking the same wrong question -- real call 2026-10-24.
            await _speak(session, entity_text.reask(lang), lang)
            return
        # Neither yes nor no (or a "no" with real content of its own, just above): this turn is handled as a fresh one, but the question is
        # REMEMBERED. If the turn turns out to be nothing (a garbled "eh eh you look"), the same question is asked again instead of "sorry, I
        # did not understand" -- real call 2026-10-17.
        session.lapsed_pending = (pending, session.turn_count)

    if not confirmed_entity:
        # OWNER'S INSTRUCTION, 2026-10-12: the answer to "is it blood, urine, or imaging?" (asked below, in the
        # "does not know the name at all" branch) is read here, FIRST -- before anything else gets a chance to
        # treat this turn as a fresh question. A category answer that doesn't match any of the three, or is
        # unclear, goes straight to the safety-check message: no guessing past this point, ever.
        if session.awaiting_test_category:
            session.awaiting_test_category = False
            category = match_category_answer(text)
            if category is not None:
                await _handle_test_category_answer(session, category, text, lang)
            else:
                await _speak(session, phrase("test_unknown_safety_check", lang), lang)
            return
        # KCD-lay-terms: a rural lay term for a test category ("রক্ত পরীক্ষা", "পেটের ছবি", "মাথার ছবি"), matched
        # deterministically before any model call -- but never while a booking is already actively in progress
        # (session.booking is not None), so a lay-term-shaped word inside an unrelated answer never hijacks it.
        # owner, 2026-10-26: the answer to "which blood test?": not knowing sends the caller to the prescription photo; asking for the list reads it; a test named goes on as a normal turn
        if session.awaiting_blood_test is not None:
            asked_turn, session.awaiting_blood_test = session.awaiting_blood_test, None
            if session.turn_count - asked_turn <= 2 and session.booking is None:
                named_now = []
                with contextlib.suppress(Exception):
                    named_now = _fast_path.named_tests(text, language_of_script(text, lang)) if _fast_path is not None else []
                if not named_now and lay_terms.wants_the_list(text):
                    await _handle_lay_term(session, BLOOD, text, lang, ask_first=False)
                    return
                if not named_now and lay_terms.says_unsure_which_test(text):
                    await _speak(session, phrase("send_prescription_whatsapp", lang), lang)
                    await _queue_prescription_callback(session, "lay_term_blood", text, lang)
                    return
        lay_term = None if session.booking is not None else match_lay_term(text)
        if lay_term == BLOOD and _fast_path is not None:
            with contextlib.suppress(Exception):
                if _fast_path.named_tests(text, language_of_script(text, lang)):
                    lay_term = None  # "CBC blood test ka rate" names a test: that is a price question, not "which blood test?"
        if lay_term is not None:
            await _handle_lay_term(session, lay_term, text, lang)
            return
        # OWNER'S INSTRUCTION, 2026-10-12: "wo test ka naam toh nhi pta" -- the caller was just asked which test
        # (session.pending_enquiry, set when test_rate/test_prep had no test_name) and says they do not know its
        # name at all, not even a category word match_lay_term above would have already caught. Scoped tightly
        # to that exact context (a recent test_rate/test_prep ask) so an unrelated "I don't know" elsewhere in
        # the call is never mistaken for this.
        pending_peek = session.pending_enquiry
        if (
            pending_peek is not None
            and pending_peek[0] in ("test_rate", "test_prep")
            and session.turn_count - pending_peek[1] <= 1
            and expresses_not_knowing_test_name(text)
        ):
            session.awaiting_test_category = True
            await _speak(session, phrase("ask_test_category", lang), lang)
            return
        # sourav, 2026-10-14: "remove X", "remove that one", "what is the total" over the tests priced earlier in this
        # call -- decided here, before the model (which only ever sees this one sentence), and answered from live
        # lookups. Never while a booking is in progress; anything it is not sure of falls through unchanged.
        if session.booking is None and session.priced_tests and await _answer_from_priced_list(session, text, lang):
            return
        if await _answer_multi_questions(session, text, lang):
            return  # several questions in one breath: each answered (agent/multi_question.py)
        if owner_answers.asks_phone_booking(text) and not getattr(session, "phone_booking_told", False):
            session.phone_booking_told = True  # said once; the booking carries on in the same turn
            await _speak(session, owner_answers.phone_booking_note(lang), lang)
        if await _answer_when_repeat(session, text, lang):
            return
        if await _answer_after_date(session, text, lang):
            return  # "any day after the 21st" while a doctor is being booked (owner, 2026-10-19)
        if await _answer_where_in_address(session, text, lang):
            return  # a where-question about the address (or the number for the location that follows it) is not an echo of the address (owner, 2026-10-19)
        if await _answer_scan_followup(session, text, lang):
            return
        if owner_answers.branch_identity(text, lang) and await _answer_owner_faq(session, text, lang):
            return  # "আপনারা কি সোনু স্কান বলছেন" is a question about the clinic's name, answered as such -- not an echo of the greeting (owner, 2026-10-19)
        if await _answer_word_check(session, text, lang):
            return
        if await _answer_prescription(session, text, lang):
            return
        if await _answer_route_pushback(session, text, lang):
            return
        if await _answer_named_total(session, text, lang):
            return
        if await _answer_owner_faq(session, text, lang):
            return
        if session.booking is None and await _answer_test_overview(session, text, lang):
            return
        if session.booking is None and await _answer_body_imaging(session, text, lang):
            return
        if where_doctor_sits.asks_floor(text) and not _booking_open(session):  # owner, 2026-10-29: which floor does the doctor sit on -> the counter tells
            await _speak(session, where_doctor_sits.reply(lang), lang)
            return
        if session.booking is None and await _head_pain_turn(session, text, lang):
            return
        if session.booking is None and await _answer_department_doctors(session, text, lang):
            return
        # sourav, 2026-10-16: what the caller's OWN words settle during a doctor booking (phone digits said in pieces, "double nine",
        # Hindi/Bengali digit words; a yes to a name read back; "the first one" after offered slots) is decided here in code, before
        # the model -- the real call never reached the read-back because phone and name depended on the model alone.
        # agent/booking_capture.py holds the rules; this only speaks what they say.
        dept_found = None
        _asks_who_sits = (
            session.booking is not None
            and not (session.booking.slots or {}).get("doctor_name")
            and re.search(r"কে (?:আছে|আছেন|বসে|বসেন|বসছে|বসছেন)|কারা|কোন কোন ডাক্তার|\bwho (?:is|sits|are)\b|कौन|which doctors", text, re.I) is not None
        )
        if session.booking is None or getattr(session, "dept_pending", None) or _asks_who_sits:
            dept_found = await _dept_search_turn(session, text, lang)
            if dept_found is True:
                return
            dept_found = dept_found if isinstance(dept_found, dict) else None
        chamber_window = await _chamber_window(session, text)
        await _check_police_station_answer(session, text)
        await _check_address_answer(session, text)
        capture_plan = booking_capture.before_model(session.booking, text, lang, window=chamber_window)
        if capture_plan.speak:
            await _speak(session, capture_plan.speak, lang)
            return
        if capture_plan.skip_model:
            data = {"intent": "book_appointment", "slots": capture_plan.slots or {}}
        else:
            try:
                data = dept_found if dept_found is not None else await _resolve_intent(session, text, lang)
            except ExtractionError as e:
                logger.error("[%s] intent extraction failed: %s", session.call_id, e)
                # sourav, 2026-10-16: a model failure in the middle of a booking asks the next booking question instead of
                # "a little problem" (which is what the live call heard on every garbled turn).
                if await _recover_booking_turn(session, text, lang):
                    return
                await _speak(session, phrase("llm_failure", lang), lang, fallback_reason="llm_failure")
                return

    data = _name_answer_repair(session, text, data)
    data = _lock_doctor_name_during_name_capture(session, data, text)
    if not confirmed_entity:
        data = await _ground_model_slots(session, text, lang, data, trust_age=capture_plan.skip_model)
        # Real call 2026-10-14: "সিবিসি টেস্টের দাম কত ফাস্টিং করতে হবে" got only the price; the fasting half was dropped.
        data = add_implied_second_question(text, data)
    intent = data["intent"]
    slots = data["slots"]
    _turn_text.set(text)  # the caller's own wording, for answers that need it (agent replies that frame who was meant)
    _doctor_from_context.set(bool(data.get("from_context")))
    # Owner, 2026-10-29: "আজ ডাক্তার ডন মাহমুদ চেম্বারে বসছে?" -- a name that is not one of our doctors is answered "no doctor by that name sits at Sonoscan", not "which doctor?" (checked against the clinic first)
    if data.get("unknown_doctor") and _tools is not None:
        try:
            _chk = await _tools.get_doctor_availability(data["unknown_doctor"], None)
        except ToolCallError:
            _chk = {"found": True}  # cannot tell: leave it to the ordinary path
        if not _chk.get("found") and not _chk.get("ambiguous"):
            if _fast_path is not None:
                _seen, _everyone = set(), []
                for _info in _fast_path.catalogue._doctor_info.values():  # noqa: SLF001
                    if id(_info) not in _seen:
                        _seen.add(id(_info))
                        _everyone.append(_info)
                _close = dept_talk.close_doctors(data["unknown_doctor"], _everyone, floor=0.45)
                if _close:
                    session.doctor_check = {"info": _close[0], "turn": session.turn_count, "date": booking_capture.read_when(text, False, want_day=True).date}
                    await _speak(session, dept_talk.confirm_name(dept_talk.spoken_name(_close[0], lang), lang), lang)
                    return
            await _speak(session, dept_talk.no_such_doctor(name_script.clean_query(data["unknown_doctor"], lang), lang), lang)
            return
    pending, session.pending_enquiry = session.pending_enquiry, None
    if intent == "unclear" and pending is not None and session.turn_count - pending[1] <= 1:
        intent = pending[0]  # "the same test", a bare name: the answer to the question we just asked
    # Owner, 2026-10-29: a commitment ("তো একটা নাম লিখে দিন") right after a doctor was discussed is the booking of that doctor, never "off topic" (the gate is loose and state-aware).
    _committing = session.booking is None and booking_carryover.recent(session.last_doctor_enquiry, session.turn_count) and booking_carryover.continues_booking(text)
    if intent in ("smalltalk", "unclear") and not _committing and await _offtopic_by_verdict(session, text, lang, data, asr_result):
        return
    # Owner, 2026-10-29 ("কালকে বসবে" -- will he sit tomorrow? -- was answered "your confirmation number, or the number you booked from?"): a sitting question with no number or confirmation id in it is about a DOCTOR,
    # never a lookup of a booking. The doctor is the one just discussed or in the open booking; with none, the caller is asked which doctor (the answer continues the question).
    if (
        (
            intent in ("lookup_booking", "unclear", "smalltalk", "reschedule_appointment", "cancel_appointment")  # (a reschedule / cancel reading is kept only when the words themselves say change or cancel -- see below)
            or (intent == "doctor_availability" and not _text_names_a_doctor(text, lang))
        )
        and session.booking is None
        and _asks_when_sits(text)
        and not (slots or {}).get("phone")
        and not (slots or {}).get("confirmation_id")
        and booking_capture.read_digits(text)[1] < 2
        and not booking_carryover.mentions_change_or_cancel(text)
    ):
        logger.info("[%s] %s re-read as a doctor sitting question", session.call_id, intent)
        _said_day = booking_capture.read_when(text, False, want_day=True).date
        intent = "doctor_availability"
        _doctor_from_context.set(not _text_names_a_doctor(text, lang))  # no doctor in the sentence itself: the one just discussed (answered with "উনি …")
        slots = {"doctor_name": _strict_enquiry_doctor(session, text, lang), "date": _said_day or (slots or {}).get("date")}  # the doctor NAMED or just discussed -- never the model's guess (it wrote the whole sentence as a name)
        data = {**data, "intent": intent, "slots": slots}

    # KCD-lead-1: the answer to referral_cues.referral_ask_reply's combined question -- read from THIS turn's
    # slots regardless of what intent the model guessed for it (the model has no "referral" intent to guess;
    # whatever patient_name/phone/doctor_name/symptom_description it extracted is exactly what is needed).
    # Logged once, with whatever was actually said -- blank fields included, never asked twice.
    pending_referral, session.pending_referral = session.pending_referral, None
    if pending_referral is not None and session.turn_count - pending_referral <= 2:
        notes = " | ".join(p for p in (session.referral_notes, text) if p)
        try:
            await _tools.log_referral(
                patient_name=slots.get("patient_name") or "",
                phone=slots.get("phone") or "",
                doctor_name=slots.get("doctor_name") or "",
                symptom_description=slots.get("symptom_description") or "",
                notes=notes,
            )
            await _speak(session, referral_cues.referral_logged_reply(lang), lang)
        except ToolCallError as e:
            logger.warning("[%s] could not log the referral: %s", session.call_id, e)
            await _speak(session, phrase("tool_failure", lang), lang, fallback_reason="tool_failure")
        return

    # The answer to "which doctor?" asked for a which-days question (_answer_doctor_weekdays): the doctor the model found, else the one named in the sentence.
    pending_weekdays, session.pending_weekdays = session.pending_weekdays, None
    if pending_weekdays is not None and session.turn_count - pending_weekdays <= 2:
        weekdays_of = slots.get("doctor_name") or _enquiry_doctor(session, text, lang)
        if weekdays_of:
            await _speak_weekdays(session, weekdays_of, lang)
            return

    # The answer to "which doctor?" asked for a 'what will my serial be if I book' question (_answer_next_serial).
    pending_serial, session.pending_serial = session.pending_serial, None
    if pending_serial is not None and session.turn_count - pending_serial[0] <= 2:
        serial_doctor = slots.get("doctor_name") or _enquiry_doctor(session, text, lang)
        if serial_doctor and await _speak_next_serial(session, serial_doctor, pending_serial[1], lang):
            return

    # The answer to "which doctor?" asked for a from-and-to-hours question (_answer_sitting_hours).
    pending_hours, session.pending_hours = session.pending_hours, None
    if pending_hours is not None and session.turn_count - pending_hours <= 2:
        hours_doctor = slots.get("doctor_name") or _enquiry_doctor(session, text, lang)
        if hours_doctor:
            session.last_doctor_enquiry = {"doctor_name": hours_doctor, "date": None, "turn": session.turn_count}
            if await _speak_sitting_hours(session, hours_doctor, lang):
                return

    # The answer to "which doctor?" asked for a weekday question (_answer_day_availability).
    pending_day, session.pending_day = session.pending_day, None
    if pending_day is not None and session.turn_count - pending_day[0] <= 2:
        day_doctor = slots.get("doctor_name") or _enquiry_doctor(session, text, lang)
        if day_doctor:
            await _speak_day_availability(session, day_doctor, pending_day[1], lang)
            return

    # The answer to "which doctor?" asked for a when-to-come question (_answer_when_to_come).
    pending_when, session.pending_when = session.pending_when, None
    if pending_when is not None and session.turn_count - pending_when <= 2:
        when_doctor = slots.get("doctor_name") or _enquiry_doctor(session, text, lang)
        if when_doctor:
            session.last_doctor_enquiry = {"doctor_name": when_doctor, "date": None, "turn": session.turn_count}
            if await _answer_when_to_come(session, when_doctor, lang):
                return

    # KCD-lead-A: a symptom said in everyday words ("my leg hurts", "আমার কানে ব্যথা", "Amar daante byatha") that the model could not place is a
    # department question, not "unclear". Asked of the clinic's own routing table only for turns the model gave up on, outside a booking; no match
    # leaves the turn exactly as it was. (No department for eyes exists in the clinic's data: that stays an honest "could not tell which department".)
    if intent in ("unclear", "smalltalk") and session.booking is None and _tools is not None and 2 <= len(text.split()) <= 20 and pending is None:
        try:
            routed = await _route_dept(text, lang, session)
        except ToolCallError:
            routed = {}
        if routed.get("matched"):
            intent = "department_query"
            slots = {**slots, "symptom_description": text}
            data = {**data, "intent": intent, "slots": slots}

    # Real call 2026-10-17: after "Dr Sen does not sit that day; next: 18 October", "ঠিক আছে ওই দিন আমার একটা বুকিং দিন" was read as a LOOKUP of an
    # existing booking ("from which number did you book?") and the phone digits then fell to the greeting. A request to book, straight after
    # a doctor was asked about, is a booking: the doctor and the day come from what was just said (booking_carryover).
    if (
        intent in ("lookup_booking", "unclear", "smalltalk", "clinic_faq", "book_test", "reschedule_appointment", "cancel_appointment")
        and session.booking is None
        and not (intent in ("reschedule_appointment", "cancel_appointment") and booking_carryover.mentions_change_or_cancel(text))
        and not (intent == "book_test" and _fast_path is not None and _fast_path.named_tests(text, language_of_script(text, lang)))
        and (booking_carryover.recent(session.last_doctor_enquiry, session.turn_count) or intent in ("lookup_booking", "unclear", "reschedule_appointment", "cancel_appointment"))
        and (
            booking_carryover.asks_to_book(text)
            or (booking_carryover.recent(session.last_doctor_enquiry, session.turn_count) and booking_carryover.continues_booking(text))
        )
    ):
        logger.info("[%s] %s re-read as a booking request after a doctor enquiry", session.call_id, intent)
        # keep what the model did read of the booking (the doctor, a date, a time); only the intent was wrong
        slots = {k: v for k, v in (slots or {}).items() if k in ("doctor_name", "date", "time_slot", "patient_name", "phone") and v}
        intent = "book_appointment"
    if (
        intent in ("book_test", "add_test_booking", "unclear", "smalltalk", "clinic_faq", "lookup_booking")
        and session.booking is not None
        and session.booking.action in ("cancel_appointment", "reschedule_appointment")
        and session.booking.stage == "collecting"
        and not session.booking.is_stale()
        and not confirmed_entity
        and not (_fast_path is not None and _fast_path.named_tests(text, language_of_script(text, lang)))
    ):
        # Real call 2026-10-17: "which number did you book from?" was answered with ten digits and the model called it a lab-test booking; the cancellation
        # could never be done. A phone number said while a cancel/reschedule waits for one is that booking's number.
        number = booking_capture.normalise_phone(booking_capture.read_digits(text)[0])
        if number:
            logger.info("[%s] %s read as the number for the open %s", session.call_id, intent, session.booking.action)
            intent, slots = session.booking.action, {"phone": number}
    elif (
        intent in ("cancel_appointment", "reschedule_appointment")
        and session.booking is not None
        and session.booking.action == intent
        and session.booking.stage == "collecting"
        and not session.booking.is_stale()
        and not confirmed_entity
        and not (slots or {}).get("confirmation_id")
    ):
        # Real call 2026-10-17: the model labelled the intent right but wrote the phone as 9976543210 for the spoken 9876543210 ("নাইট" -> 9). The digits the code
        # reads from the caller's own words are the number; the model's copy of them is not trusted.
        number = booking_capture.normalise_phone(booking_capture.read_digits(text)[0])
        if number and number != (slots or {}).get("phone"):
            logger.info("[%s] phone slot %s replaced by the digits heard: %s", session.call_id, (slots or {}).get("phone"), number)
            slots = {**(slots or {}), "phone": number}
    if (
        intent in ("book_test", "lookup_booking", "reschedule_appointment", "cancel_appointment", "add_test_booking")
        and session.booking is not None
        and session.booking.action == "book_appointment"
        and session.booking.stage == "collecting"
        and not session.booking.is_stale()
        and not confirmed_entity
        and not (intent in ("reschedule_appointment", "cancel_appointment") and booking_carryover.mentions_change_or_cancel(text))
        and not (intent == "book_test" and _fast_path is not None and _fast_path.named_tests(text, language_of_script(text, lang)))
        and booking_carryover.continues_booking(text)
    ):
        # Real call 2026-10-17: a doctor booking was open and asked for the day; "এই সেদিনেরই করুন" was read as a lab-test booking and answered with
        # the counter message. An answer to the open doctor booking is never a lab-test request when no test is named.
        logger.info("[%s] %s read as the open doctor booking's answer", session.call_id, intent)
        slots = dict(slots or {})
        slots["date"] = slots.get("date") or slots.get("new_date")  # a "reschedule" reading put the day where a new day goes
        slots["time_slot"] = slots.get("time_slot") or slots.get("new_time_slot")
        slots = {k: v for k, v in slots.items() if k in ("doctor_name", "date", "time_slot", "patient_name", "phone") and v}
        intent = "book_appointment"
    _mark(session, "intent")
    _rec(session, "intent", intent)
    if intent != "unclear":
        session.unclear_streak = 0
    _note_caller_number(session, (slots or {}).get("phone") or (slots or {}).get("contact_phone") or capture_plan.phone)
    _note_stated_age(session, slots, lang)
    if capture_plan.phone and session.booking is not None and intent != "book_appointment" and not confirmed_entity:
        # sourav, 2026-10-16: a complete 10-digit number was heard in a sentence the model labelled something else ("no, my number is
        # ...", a correction at the read-back). The digits are the caller's own; they go into the open booking.
        intent, slots = "book_appointment", {**(slots or {}), "phone": capture_plan.phone}
    if (
        capture_plan.digits
        and not confirmed_entity
        and intent in ("lookup_booking", "cancel_appointment", "reschedule_appointment", "resend_confirmation", "add_test_booking")
        and session.booking is not None
        and session.booking.action == "book_appointment"
        and session.booking.stage == "collecting"
        and not session.booking.is_stale()
    ):
        # sourav, 2026-10-16 (real Hindi call): "the patient's name is Saurabh and the number is ..." was labelled lookup_booking, so the
        # agent read the digits back ("Did you say 458789?") and answered "no upcoming appointment found" -- the booking was hijacked by a
        # lookup. While a booking waits for the patient's details, digits heard in answer to the phone question belong to the booking.
        intent = "book_appointment"

    # Real call 2026-10-12, three turns from one call: "वो तो नहीं पता मतलब क्या सिंप्टम्स वगैरह वो नहीं मालूम"
    # ("I don't know, I mean I don't know the symptoms") -> classified "department_query"; a bare "डॉक्टर सेन"
    # meant to correct a garbled doctor name -> classified "doctor_availability" (answered as a one-off
    # enquiry that never writes back into the booking, so the bogus name was NEVER overwritten and only
    # surfaced as "no doctor found" at the very end); a bare "अः ग्यारह बजे" ("eleven o'clock") answering
    # "which time?" -> classified "smalltalk", drawing the generic greeting filler. All three are the SAME
    # root cause: the per-utterance classifier has no notion of "we are mid-booking, waiting specifically for
    # X" and reads a terse, context-dependent answer on its own unrelated terms. Checked here, BEFORE the
    # smalltalk short-circuit just below (which would otherwise return first) and before any of these three
    # intents' own standalone handlers further down -- re-routing into book_appointment lets the SAME
    # merge_slots() that handles a direct answer pick up whatever this turn actually said (a doctor name, or
    # nothing at all for genuine smalltalk, which just re-asks the same pending question -- never a worse
    # outcome than a random reply that ignores where the caller actually is). Same discipline as the
    # "unclear" re-route a little further down, extended to the other intents real calls have now shown take
    # the exact same shortcut around it.
    if (
        intent in ("department_query", "doctor_availability", "smalltalk")
        and session.booking is not None
        and session.booking.action == "book_appointment"
        and session.booking.stage == "collecting"
        and not session.booking.is_stale()
        # sourav, 2026-10-16 (real call): once the booking HAS its doctor, "when did Dr. Arindam Sen sit down?" is a genuine question about a
        # doctor's days, not an answer to "which doctor?". It was swallowed into the booking ("Got it, changed to doctor ...") and never
        # answered. Only re-route while the booking is still waiting for the doctor; otherwise answer it (the "back to your booking" line follows).
        and (
            intent != "doctor_availability"
            or "doctor_name" in missing_required(session.booking)
            # sourav, 2026-10-16 (real Hindi call): a bare "कल" (tomorrow) after the doctor's sitting time was labelled doctor_availability
            # with NO doctor, and the agent asked "which doctor?". A day or time said with no doctor named is the booking's answer.
            or (not (slots or {}).get("doctor_name") and (capture_plan.date or capture_plan.time))
        )
    ):
        intent = "book_appointment"

    # Real call 2026-10-12: "ten o'clock" / "दस बजे" answering "what time do you want?" has no booking verb
    # in it and reads exactly like the generic "what are your hours?" FAQ -- misclassified "clinic_faq", it
    # spoke the hours disclaimer and re-asked the same time question, which from the caller's side looks
    # exactly like the time they just gave was rejected. Narrowly re-routed only when the booking is
    # actively waiting on time_slot specifically, so a genuine hours question asked at any other point in
    # the call is answered normally, not swallowed into the booking.
    if (
        intent == "clinic_faq"
        and session.booking is not None
        and session.booking.action == "book_appointment"
        and session.booking.stage == "collecting"
        and not session.booking.is_stale()
        and "time_slot" in session.booking.asked_fields
        and "time_slot" in missing_required(session.booking)
    ):
        intent = "book_appointment"

    if intent == "smalltalk":
        reply = data.get("direct_reply_bn")
        # The ONLY model-composed text a caller ever hears, so it passes the persona
        # (register, no hedging, no reassurance or advice, sentence length) or is replaced.
        # KCD-500: and it may not claim anything about THIS caller's past -- history is rendered
        # from retrieved fields by agent/history_templates.py, never composed.
        # Real call 2026-10-17: for the garbled "হস্ট হয় ওখানে" the model wrote "আপনার সব্বাস্তু! কিভাবে আমাদের সাহায্য করতে পারি?" -- nonsense that still
        # passed the register and length checks. A 7B model does not write reliable Bengali or Hindi, so only ENGLISH smalltalk may be model-composed;
        # Bengali and Hindi get the clinic's own wording.
        chosen = (
            reply
            if (
                lang == "en"
                and _speakable(reply or "", lang)
                and persona_clean(reply or "", lang)
                and not history_text.mentions_personal_history(reply or "", lang)
            )
            else phrase("smalltalk_default", lang)
        )
        if chosen == phrase("smalltalk_default", lang) and not greeting_words.is_greeting(text):
            if off_topic.how_are_you(text) and not unrelated_caller.has_hospital_cue(text, lang, _catalogue_entity_in):
                chosen = phrase("how_are_you_reply", lang)  # owner, 2026-10-28: "আপনি ভালো আছেন?" got "হ্যাঁ, বলুন" (yes, go on)
            elif (
                len(text.split()) >= 3
                and _free_for_offtopic(session)
                and classify_yes_no(text, lang) is None  # a yes / no is always an answer, never small talk
                and not call_end.is_closing_acknowledgement(text, lang)
                and not off_topic.has_health_word(text)
                and not unrelated_caller.has_hospital_cue(text, lang, _catalogue_entity_in)
            ):
                # Owner, 2026-10-28: real talk that has nothing to do with the clinic ("আমি আপনাকে ভালোবাসি") is not "yes, go on": told once, the next such turn ends the call
                await _handle_offtopic(session, lang, text)
                return
            else:
                chosen = phrase("silence_go_on", lang)  # "okay" / "ঠিক আছে" is not hello: do not introduce ourselves again, just say go on
        await _speak(session, _with_resume(session, chosen, lang), lang)
        return

    if (
        intent == "unclear"
        and session.booking is not None
        and session.booking.stage in ("confirming", "awaiting_charge_confirm")
    ):
        if session.no_at_confirm:
            await _reopen_after_no(session, session.booking, lang)  # a no with nothing to change in it
        else:
            await _speak(
                session, booking_confirmation_readback(session.booking.slots, session.booking.action, lang), lang
            )
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
        if session.booking is not None and session.booking.stage == "collecting" and not session.booking.is_stale():
            intent = session.booking.action
        elif await _resume_after_unclear(session, text, lang):
            return
        else:
            session.unclear_streak += 1
            if session.unclear_streak >= 3:
                # owner 2026-10-18: after three tries the AI could not answer, it says it is routing the caller to a person to help better
                await _speak(session, phrase("handoff_better", lang), lang)
                await _handoff_to_human(session, "unanswered_x3", (lang,), announce=False)
                return
            await _speak(session, phrase("unclear", lang), lang)
            return

    # KCD-442/KCD-447: a turn the two ASR decoders disagreed on is not
    # trusted to drive a factual lookup at all -- acting on a misheard
    # test/doctor name would produce a confident, wrong answer about
    # something else, which is worse than asking the caller to repeat
    # themselves. Checked before the tool call, not after: the point
    # is to never RUN the lookup on an unreliable entity, not merely
    # to hedge the reply once it comes back.
    #
    # One gate decides this (agent/action_gate.py), not two separate confidence checks -- it is proven
    # (tests/test_action_gate.py) to give exactly the same answer should_withhold_factual_answer() and
    # needs_entity_readback() gave here before; the entity itself is not yet resolved at this point in the
    # turn (the lookup has not run), so it is not passed -- exactly as those two functions never took it either.
    decoder_used = getattr(asr_result, "decoder_used", None)
    # A "yes"/"ठीक"/"ঠিক" to a name or number just read back, or to the proposed time, is settled by the words themselves: the caller heard the very thing it
    # confirms. The recognisers disagreeing on a one-word reply (one wrote three words around it) is no reason to ask for it again (real call 2026-10-18).
    answering_open_question = capture_plan.readback_answer or (
        session.booking is not None
        and session.booking.slots.get("_time_ok") == "pending"
        and classify_yes_no(text, lang) == "yes"
    )
    gate_verdict = (
        action_gate.decide(intent, decoder_used=decoder_used, decoder_agreement=asr_result.decoder_agreement)
        if not confirmed_entity and not answering_open_question
        else None
    )
    if gate_verdict is not None and gate_verdict.action == action_gate.REPEAT:
        logger.info(
            "[%s] withholding %s answer: decoder_agreement=%.2f below floor",
            session.call_id,
            intent,
            asr_result.decoder_agreement,
        )
        insufficient_information.record("low_decoder_agreement", intent, lang)
        # A live call (2026-09-28): the model DID extract a name ("Roy") even though the two decoders
        # disagreed on the sentence as a whole -- the caller heard a blank "sorry, I didn't catch the name,
        # which doctor?" when what they actually needed was "do you mean Roy?", a yes/no that is faster to
        # answer and confirms EXACTLY the uncertain part instead of asking them to repeat everything. This is
        # the same read-back entity_text already does for UNAVAILABLE confidence (see the READ_BACK branch
        # below) -- extended here to LOW confidence too, but ONLY when the model actually named something;
        # nothing is invented, and the lookup still only runs after an explicit yes.
        entity = entity_text.entity_to_confirm(intent, slots)
        if entity is not None and not _readable_in(entity[1], lang):
            entity = None  # owner, 2026-10-27: a garbled name in another script ("औरम सेन" in a Bengali call) cannot be said back by the voice: ask for the name instead
        if entity is not None:
            slot, value = entity
            session.pending_entity = entity_text.PendingEntity(intent, slot, value, data)
            insufficient_information.record("low_decoder_agreement_readback", intent, lang)
            await _speak(session, entity_text.confirm_question(slot, value, lang), lang)
            return
        # Nothing at all was extracted: the caller said WHAT they want ("the price of ...") but not clearly
        # enough to guess a name from. Ask for the name only, instead of "say that again" for the whole sentence.
        name_slot = {"test_rate": "test_name", "test_prep": "test_name", "doctor_availability": "doctor_name"}.get(
            intent
        )
        if name_slot:
            session.pending_enquiry = (intent, session.turn_count)
            await _speak(
                session, f"{phrase('name_not_caught', lang)} {missing_slot_prompt(intent, name_slot, lang)}", lang
            )
        else:
            await _speak(session, insufficient_information_reply(lang), lang)
        return
    # Only one decoder produced this text, so nothing vouches for it: read the entity back and
    # run the lookup only after a yes. Never let "no confidence figure" mean "trusted".
    if gate_verdict is not None and gate_verdict.action == action_gate.READ_BACK:
        entity = entity_text.entity_to_confirm(intent, slots)
        if entity is not None:
            slot, value = entity
            session.pending_entity = entity_text.PendingEntity(intent, slot, value, data)
            insufficient_information.record("single_decoder_readback", intent, lang)
            await _speak(session, entity_text.confirm_question(slot, value, lang), lang)
            return

    try:
        if (
            intent in ("book_test", "test_rate", "test_prep")  # test_prep: the model hears "sugar test হয়" as a fasting question (real call 2026-10-17)
            and (slots.get("test_name") or slots.get("test_names"))
            and test_availability.asks_if_test_is_available(text, lang)
        ):
            # Real call 2026-10-14: "লিপিড প্রোফাইল টেস্ট হয়" / "लिपिड प्रोफाइल टेस्ट होता है" was classified book_test and
            # answered "bookings and payment are at the counter". It is a yes/no question about the clinic: answer it,
            # then ask what else the caller wants to know. A caller who actually asks to BOOK still gets the counter reply.
            reply, follow_slots = await _answer_test_availability(slots, lang)
            await _finish_enquiry_turn(session, text, lang, "test_rate", follow_slots, reply, data)

        elif intent == "test_rate":
            slots, _ = _followup(session, intent, slots, text, lang)
            _turn_text.set(text)
            reply = await _answer_enquiry_intent(intent, slots, lang)
            if reply is None:
                # Owner, 2026-10-27: a real call asked "ডাক্তার কে দেখাতে কত টাকা লাগে" (how much to see the doctor) --
                # a bare price question with no test name in it -- and the model still classified it test_rate,
                # answered with "which lab test's rate do you want to know". agent/owner_answers.asks_visit_fee
                # already catches this phrasing earlier in the turn (_answer_owner_faq, before the model is ever
                # asked), so this should no longer be reached for it -- this is the safety net for whichever
                # phrasing that cue list has not been taught yet: a price question with NO test name, that also
                # names or implies a doctor, is never defaulted to "which test", it is answered as a visit-fee
                # question (or asks which doctor, same as the fee question would on its own).
                if not (slots.get("test_name") or slots.get("test_names")) and owner_answers.asks_visit_fee(text):
                    fee_reply = await _visit_fee_reply(session, text, lang)
                    if fee_reply:
                        await _finish_enquiry_turn(session, text, lang, intent, slots, fee_reply, data)
                        return
                session.pending_enquiry = (intent, session.turn_count)
                await _speak(session, missing_slot_prompt(intent, "test_name", lang), lang)
                return
            await _finish_enquiry_turn(session, text, lang, intent, slots, reply, data)

        elif intent == "doctor_availability":
            slots, _ = _followup(session, intent, slots, text, lang)
            reply = await _answer_enquiry_intent(intent, slots, lang)
            if reply is None:
                session.pending_enquiry = (intent, session.turn_count)
                await _speak(session, missing_slot_prompt(intent, "doctor_name", lang), lang)
                return
            await _finish_enquiry_turn(session, text, lang, intent, slots, reply, data)

        elif intent == "test_prep":
            slots, _ = _followup(session, intent, slots, text, lang)
            _turn_text.set(text)
            reply = await _answer_enquiry_intent(intent, slots, lang)
            if reply is None:
                if test_availability.asks_if_test_is_available(text, lang):
                    # "...টেস্ট হয় ওখানে" with the name lost: not a question about preparation. Ask which test, and carry on as a price/availability
                    # enquiry for whatever name comes next.
                    session.pending_enquiry = ("test_rate", session.turn_count)
                    await _speak(session, _which_test_prompt(lang), lang)
                    return
                session.pending_enquiry = (intent, session.turn_count)
                await _speak(session, missing_slot_prompt(intent, "test_name", lang), lang)
                return
            await _finish_enquiry_turn(session, text, lang, intent, slots, reply, data)

        elif intent == "clinic_faq":
            slots, _ = _followup(session, intent, slots, text, lang)
            reply = await _answer_enquiry_intent(intent, slots, lang)
            if reply is None:
                session.pending_enquiry = (intent, session.turn_count)
                await _speak(session, missing_slot_prompt(intent, "faq_topic", lang), lang)
                return
            await _finish_enquiry_turn(session, text, lang, intent, slots, reply, data)

        elif intent == "book_appointment":
            # A day in the past is never a booking day (real call 2026-10-18: "কালকেই" was read as yesterday, 2026-10-17): it is dropped, and the day already
            # held (or asked for) stands.
            if (slots or {}).get("date") and slots["date"] < slot_proposal.clinic_now().date().isoformat():
                logger.info("[%s] a past day (%s) is not a booking day: ignored", session.call_id, slots["date"])
                slots = {k: v for k, v in slots.items() if k != "date"}
            # The caller's own day word is the day (owner, 2026-10-18: "না, পরশু করুন" after "he sits tomorrow" was booked for TOMORROW, the day already told): it beats
            # the model's reading and the day carried over from the last answer. In a short sentence, or when no day is held yet.
            _held_day = session.booking.slots.get("date") if session.booking is not None else None
            if not _held_day or len(text.split()) <= 6:
                _said_day = booking_capture.read_when(text, False, want_day=True).date
                if _said_day and _said_day >= slot_proposal.clinic_now().date().isoformat():
                    slots = {**(slots or {}), "date": _said_day}
            # Owner, 2026-10-29 ("তাহলে বুধবারে করে দিন উনার"): a weekday named for the doctor just discussed is first ANSWERED -- does she sit on that weekday? if not, when does she -- from the schedule
            # table; the booking then goes on from that answer. It used to be replaced silently by another date ("ঠিক আছে, মজুমদার", then the 31st).
            if session.booking is None and _tools is not None and weekday_cues.weekday_of(text) is not None:
                _wd_doctor = (slots or {}).get("doctor_name") or (session.last_doctor_enquiry or {}).get("doctor_name")
                _wd_date = weekday_cues.resolve_date(text, slot_proposal.clinic_now().date())
                if _wd_doctor and _wd_date:
                    try:
                        _wd_info = await _tools.get_doctor_availability(_wd_doctor, _wd_date)
                    except ToolCallError:
                        _wd_info = None
                    if _wd_info and _wd_info.get("found") and _wd_info.get("available") is False:
                        await _speak_day_availability(session, _wd_doctor, _wd_date, lang)
                        return
            # sourav, 2026-10-14: "book him / for that day" right after a doctor-availability answer -> that doctor (and date).
            slots = booking_carryover.carry_over(
                session.last_doctor_enquiry,
                slots,
                text,
                session.booking.slots if session.booking is not None else None,
                session.turn_count,
                time_said=bool(capture_plan.time),
            )
            st = _enter_task(session, "book_appointment")
            # sourav, 2026-10-16: the model's phone must be a real 10-digit mobile number and its patient name must be in the caller's
            # words (else it is read back for a yes first) -- see agent/booking_capture.after_model. Both used to be stored unchecked.
            slots, ask_back = booking_capture.after_model(st, text, lang, slots, capture_plan, window=chamber_window)
            prior_slots = dict(st.slots)
            changed = merge_slots(st, slots)
            if not st.slots.get("doctor_name") and _fast_path is not None:
                # Owner, 2026-10-27 (real call): "মন্ডল আছে ডাক্তার মন্ডল" (Mondal is there, doctor
                # Mondal) named a real doctor by bare surname -- the model's own slot extraction
                # came back empty, and book_appointment had no catalogue fallback for a bare
                # surname the way enquiry contexts already do (_enquiry_doctor/_named_doctor): the
                # booking asked "which doctor and day" as if nothing had been said. Same catalogue
                # lookup those contexts already use, applied here too.
                with contextlib.suppress(Exception):
                    doc = _fast_path._named_doctor(fast_path_normalize(text), language_of_script(text, lang))  # noqa: SLF001
                    if doc is not None:
                        st.slots["doctor_name"] = doc[1] or doc[0]
                        changed.append("doctor_name")
            # KCD-lead-7: an address can also arrive here -- volunteered in the same sentence as other
            # details, or extracted by the model out of the order the clinic would have asked for it -- not
            # only through agent/booking_capture._detail_turn's own direct "what's the address?" question and
            # answer (which is the only place that already checks for a missing police station). Either way,
            # the police station must still be asked for: stashing it the same way _detail_turn does (`_ps_
            # pending`) means that function's own pre-model resumption check picks up the caller's very next
            # turn correctly -- its gate (patient_name and phone already known) already holds by this point in
            # a booking, so nothing else needs to change for the resumption to work.
            if "patient_address" in changed and booking_capture._lacks_police_station(st.slots["patient_address"]):
                st.slots["_ps_pending"] = st.slots.pop("patient_address")
                await _speak(session, booking_capture.prompt("ps_ask", lang), lang)
                return
            await _note_patient_relationship(session, st, text, slots)
            if st.slots.get("_time_ok") == "pending" and not ask_back:
                # The time was proposed as a question. A different time ("এগারোটার সময় করুন") is taken and confirmed; a no moves to the next free slot; anything
                # else -- a yes, or the name and number straight away -- accepts it. Only then are the name and number asked.
                verdict = classify_yes_no(text, lang)
                # KCD-lead-6: set only when this pending question is offering a day the caller's doctor was MOVED
                # to because the originally-asked day was full (never for "doctor doesn't sit at all" -- that
                # case never sets this). Read before anything below can pop it off st.slots.
                day_full_from = st.slots.get("_day_full_from")
                full_doctor_name = st.slots.get("doctor_name")
                if "time_slot" in changed:
                    st.hold_token, st.hold_doctor_id = None, None
                    st.slots["_time_explicit"] = "1"  # the caller chose this time: it is the one told back
                    await _speak(session, slot_proposal.reconfirm(st.slots["time_slot"], lang), lang)
                    return
                if "date" in changed:
                    if day_full_from and st.slots.get("date") == day_full_from and full_doctor_name:
                        # the caller re-named the EXACT day already found full for this doctor: offering the
                        # same moved day again would just repeat what was already declined.
                        if await _offer_department_alternatives(session, st, lang, day_full_from, full_doctor_name):
                            return
                    # another day named ("না, পরশু করুন"): the doctor's sitting that day is told in its turn
                    for k in ("_time_ok", "_sits_told", "time_slot", "_time_explicit", "_day_full_from"):
                        st.slots.pop(k, None)
                    st.hold_token, st.hold_doctor_id = None, None
                    if await _propose_slot(session, st, lang):
                        return
                elif verdict == "no":
                    if day_full_from and full_doctor_name and await _offer_department_alternatives(
                        session, st, lang, day_full_from, full_doctor_name
                    ):
                        return
                    # not that day: the day is asked again (the sitting of whichever day comes next is told then)
                    for k in ("_time_ok", "_sits_told", "time_slot", "_time_explicit", "date", "_day_full_from"):
                        st.slots.pop(k, None)
                    st.hold_token, st.hold_doctor_id = None, None
                else:
                    st.slots["_time_ok"] = "yes"
            if ask_back:
                await _speak(session, ask_back, lang)
                return
            if "date" in changed and prior_slots.get("date") and st.slots.get("doctor_name") and st.slots.get("_time_ok") != "pending":
                # The caller CHANGED the day ("আজকে পাওয়া যাবে?"). What they hear is whether the doctor sits THAT day (or "not today; next on ..."), not an
                # acknowledgement of a day the doctor does not sit (real call 2026-10-18).
                for k in ("_time_ok", "_sits_told", "time_slot", "_time_explicit", "_day_full_from"):
                    st.slots.pop(k, None)
                st.hold_token, st.hold_doctor_id = None, None
                if await _propose_slot(session, st, lang):
                    return
            if st.hold_token and {"doctor_name", "date", "time_slot"} & set(changed):
                # KCD-104: the caller changed WHICH slot they want; the hold is on the old one. Drop it so
                # the new doctor/date/time is held below (the old hold expires on its own).
                st.hold_token, st.hold_doctor_id = None, None
            ack = correction_acknowledgement(changed, prior_slots, st.slots, lang)
            if ack:
                await _speak(session, ack, lang)
            else:
                await _thank_for_answer(session, st, changed, prior_slots, lang)
                await _echo_new_slots(session, changed, prior_slots, st.slots, lang)

            # KCD-dentistry, real-call bug 2026-10-12: the caller may have named a DEPARTMENT/specialty
            # ("ENT doctor", "I need a dentist"), not a real doctor -- the LLM has nowhere else to put that
            # but doctor_name (it is never allowed to guess a department from a symptom itself). Checked only
            # the turn doctor_name is freshly given, before it is ever treated as a real doctor's name and
            # silently carried all the way to the confirmation readback.
            if "doctor_name" in changed:
                try:
                    dept = await _tools.department_doctors(st.slots["doctor_name"], lang)
                except ToolCallError as e:
                    logger.warning("[%s] department_doctors failed (%s)", session.call_id, e)
                    dept = {"matched": False}
                if dept.get("matched"):
                    doctors = dept.get("doctors") or []
                    if doctors:
                        st.slots["doctor_name"] = doctors[0].removeprefix("Dr. ")
                        await _speak(
                            session, booking_department_suggestion_reply(_dept_label(dept["department_name"], lang), doctors, lang), lang
                        )

            # KCD-dentistry: the caller does not know a doctor's name but described a problem instead --
            # resolve it to a real department/doctor (or a General Medicine checkup) before asking for a
            # doctor's name the caller has already said they do not have.
            # Owner, 2026-10-27: "তোমার বাবার খুব দাঁতে ব্যথা তো কাকে দেখাবো নাম লেখাবো" -- a symptom and a booking request with no doctor named. The answer is the doctors of that department (and who sits
            # next), at the FIRST question; asking "which doctor?" first makes the caller say the symptom again. Once per booking, and only when the clinic's routing recognises the symptom.
            if not st.slots.get("doctor_name") and not st.slots.get("symptom_description") and not st.slots.get("_symptom_routed") and off_topic.has_health_word(text):
                st.slots["_symptom_routed"] = "1"
                try:
                    routed_now = await _route_dept(text, lang, session)
                except ToolCallError:
                    routed_now = {}
                if routed_now.get("matched"):
                    await _speak(session, await _route_reply(session, routed_now, text, lang), lang)
                    return
            _had_doctor = bool(st.slots.get("doctor_name"))
            if not st.slots.get("doctor_name") and st.slots.get("symptom_description"):
                await _resolve_doctor_from_symptom(session, st, lang)
            # owner, 2026-10-24: a doctor chosen FOR the caller from the symptom is followed by that doctor's next sitting and an offer to book it, not by "which day?"
            _routed_now = not _had_doctor and bool(st.slots.get("doctor_name"))

            # Owner's instruction, 2026-10-12: naming a real doctor should say when that doctor actually sits
            # before asking which day the caller wants -- otherwise they can easily name a day the doctor is
            # not in and have to be corrected and asked again. Gated on `"doctor_name" in changed` -- the
            # caller actively named a doctor or department THIS turn (covers the department-redirect above
            # too, which still overwrites st.slots this same turn) -- never on the symptom/GP-default path
            # just above, which already asks its own "which day" as part of its own reply; speaking this one
            # too in the same turn would stack two different "which day" questions back to back.
            #
            # Real call 2026-10-12: a garbled first-turn name ("Doctor Shene" from mangled ASR) was never
            # validated here at all -- missing_required only checks for EMPTINESS, so the bogus name sat in
            # st.slots untouched through five more turns (the caller correctly saying "Dr. Sen" later only
            # answered a one-off standalone availability question, never overwriting it -- see the
            # doctor_availability re-route below) and only surfaced as "no doctor found" at the very end,
            # after the caller had already given a date, a time and their patient details for nothing. Using
            # the SAME availability lookup both ways: found -> the sitting-time hint; not found (or
            # genuinely ambiguous) -> doctor_availability_reply's own "not found"/"did you mean" messaging,
            # clearing the slot so the caller is asked again NOW, not after the rest of the booking is moot.
            if ("doctor_name" in changed or _routed_now) and st.slots.get("doctor_name") and not st.slots.get("date"):
                try:
                    avail = await _tools.get_doctor_availability(st.slots["doctor_name"], None)
                except ToolCallError as e:
                    logger.warning("[%s] get_doctor_availability failed (%s)", session.call_id, e)
                    avail = None
                if avail is not None:
                    if avail.get("found"):
                        st.asked_fields.add("date")
                        if avail.get("date"):  # "that day" in the caller's next sentence points at THIS date (real call 2026-10-17)
                            session.last_doctor_enquiry = {"doctor_name": st.slots["doctor_name"], "date": avail["date"], "window": avail.get("chamber_hours"), "turn": session.turn_count}
                            if avail.get("available"):
                                st.slots["_offered_date"] = avail["date"]  # the hint ends "shall I book that day?": a yes sets this date (agent/booking_capture.before_model)
                        await _speak(session, booking_doctor_sitting_hint(avail, lang), lang)
                        return
                    not_found_reply = doctor_availability_reply(st.slots, avail, lang)
                    st.slots["doctor_name"] = None
                    await _speak(session, not_found_reply, lang)
                    return

            # sourav, 2026-10-24: a patient of 13 or more is moved from the paediatrician to the normal doctor (once per booking).
            if await _redirect_teen_from_paediatrics(session, st, lang):
                return

            # The caller is not asked for a time (agent/slot_proposal.py): the machine knows the sitting hours, what is booked and the clock.
            if (
                st.slots.get("doctor_name")
                and st.slots.get("date")
                and st.hold_token is None
                and st.slots.get("_time_ok") != "yes"
                and not st.slots.get("_time_ok_seen")
                and await _propose_slot(session, st, lang)
            ):
                return  # the time is proposed as a question; the caller's answer comes before the name and number

            # Secure the hold as soon as doctor+date+time are known, even
            # if patient details are still missing -- KCD-376: the
            # concurrency guard must run before the caller has spoken a
            # patient name, not after.
            if (
                st.hold_token is None
                and st.slots.get("doctor_name")
                and st.slots.get("date")
                and st.slots.get("time_slot")
            ):
                hold = await _tools.hold_slot(st.slots["doctor_name"], st.slots["date"], st.slots["time_slot"])
                if not hold.get("success"):
                    await _speak(session, booking_reply(st.slots, hold, lang), lang)
                    st.offered_options = booking_capture.options_from_hold(hold, st.slots)  # sourav 2026-10-16: "the first one" = a slot the clinic offered
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
                # Real call 2026-10-18: a number given in pieces (nine digits, one short) counted as a second miss and the phone was GIVEN UP on -- the booking went on
                # to the age without a number. A turn that carried digits is progress, not a refusal: only turns with no digits count.
                if field_name == "phone" and not capture_plan.digits and st.note_retry("phone") >= 2:
                    # KCD-370: a caller who won't give a number still
                    # gets the booking, told plainly what that costs
                    # them, instead of being blocked here forever.
                    st.phone_declined = True
                # Owner, 2026-10-17: the caller is never asked to SPELL their name letter by letter -- a heard name is read back for a yes
                # (booking_capture), and a name not heard is simply asked again.
                elif field_name == "doctor_name" and st.note_retry("doctor_name") >= 2:
                    # KCD-dentistry, real-call bug 2026-10-12: a caller who says they do not know the doctor
                    # (and gives no symptom either, or route_department already tried and failed above) was
                    # asked "which doctor?" forever. Same retry-then-degrade discipline as phone/patient_name
                    # above -- asked normally once, defaulted on the second miss.
                    await _default_to_general_medicine(session, st, lang)
                else:
                    await _speak(session, _next_question(session, st, intent, missing, lang), lang)
                    return
                missing = missing_required(st)
                if missing:
                    await _speak(session, _next_question(session, st, intent, missing, lang), lang)
                    return

            # Skipped when no real phone exists to check by -- the
            # "not_provided" sentinel must never be used as a lookup
            # key, or two different callers who both declined to give
            # a number would appear to conflict with EACH OTHER.
            if effective_phone(st) != "not_provided" and not st.slots.get("_conflict_checked"):
                try:
                    conflict = await _tools.booking_conflict(effective_phone(st), st.slots["date"], st.slots["time_slot"])
                except ToolCallError as e:
                    # Owner, 2026-10-29: the duplicate-booking check is a courtesy; when the clinic API is too slow to answer it the booking goes on (the hold and the confirm still check the slot)
                    logger.warning("[%s] duplicate-booking check skipped: %s", session.call_id, e)
                    conflict = {"conflict": False}
                st.slots["_conflict_checked"] = "1"
                if conflict.get("conflict"):
                    await _speak(session, conflict_reply(conflict["existing"], lang), lang)
                    return

            if (
                is_ready_to_confirm(st)
                and effective_phone(st) != "not_provided"
                and st.slots.get("_phone_ok") != effective_phone(st)
            ):
                # the number is read back for a yes, like the name, before the whole booking is read out
                st.slots["_phone_pending"] = effective_phone(st)
                await _speak(session, booking_capture.readback(st, lang, "phone", spoken_digit_groups(effective_phone(st))), lang)
                return
            extra = [f for f in ("patient_age", "patient_address") if not st.slots.get(f) and not st.slots.get("_age_done" if f == "patient_age" else "_address_done")]
            if is_ready_to_confirm(st) and extra:
                # owner, 2026-10-18: the age, then the address -- one question each, after the name and the number are confirmed (a read-back may already have asked it:
                # a bare "yes" to that leaves it unanswered, so it is asked again, until it is answered or let go)
                st.asked_fields.add(extra[0])
                await _speak(session, booking_capture.prompt("ask_age" if extra[0] == "patient_age" else "ask_address", lang), lang)
                return
            if is_ready_to_confirm(st) and booking_capture.whatsapp_due(st):
                # owner, 2026-10-23: the payment link and the serial number go to WhatsApp -- asked once whether this number has it, before the closing read-back
                # Owner, 2026-10-25: the address / police station just given is acknowledged AT ONCE, then the WhatsApp question (without the number said again)
                _heard_now = "" if st.slots.get("_address_heard") else booking_capture.address_heard(st.slots, lang)
                if _heard_now:
                    st.slots["_address_heard"] = "1"
                await _speak(session, f"{_heard_now}{booking_capture.ask_whatsapp(st, lang)}", lang)
                return
            if is_ready_to_confirm(st):
                mark_confirming(st)
                if st.phone_declined:
                    await _speak(session, phrase("no_confirmation_number", lang), lang)
                await _speak(session, booking_confirmation_readback(st.slots, "book_appointment", lang), lang)

        elif intent == "book_test":
            # A lab test is paid and booked at the counter, never over the call (clinic-api/booking_service.
            # send_payment_link's docstring says the same from the other side: a lab test's confirmation_id is
            # never found there). No booking task is entered -- there is nothing to collect slots for.
            counter = phrase("lab_test_counter_only", lang)
            if await _any_test_needs_prescription(slots):
                counter = f"{counter} {phrase('rx_ask', lang)}"  # the table says a prescription is needed: ask whether the caller has it (owner, 2026-10-18)
                session.pending_prescription = ("tests", session.turn_count)
            await _speak(session, counter, lang)

        elif intent == "reschedule_appointment":
            st = _enter_task(session, "reschedule_appointment")
            prior_slots = dict(st.slots)
            changed = merge_slots(st, slots)
            ack = correction_acknowledgement(changed, prior_slots, st.slots, lang)
            if ack:
                await _speak(session, ack, lang)
            else:
                await _thank_for_answer(session, st, changed, prior_slots, lang)
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
                else:
                    st.slots.pop("phone", None)  # a number with no booking is not kept; the next one is read fresh
                    await _speak(session, _NO_BOOKING_FOR_NUMBER.get(lang, _NO_BOOKING_FOR_NUMBER["en"]).format(n=spoken_digit_groups(slots["phone"])), lang)
                    return
            missing = missing_required(st)
            if missing:
                await _speak(session, missing_slot_prompt(intent, missing[0], lang), lang)
                return
            if is_ready_to_confirm(st):
                mark_confirming(st)
                await _speak(session, booking_confirmation_readback(st.slots, "reschedule_appointment", lang), lang)

        elif intent == "cancel_appointment":
            st = _enter_task(session, "cancel_appointment")
            prior_slots = dict(st.slots)
            changed = merge_slots(st, slots)
            ack = correction_acknowledgement(changed, prior_slots, st.slots, lang)
            if ack:
                await _speak(session, ack, lang)
            else:
                await _thank_for_answer(session, st, changed, prior_slots, lang)
                await _echo_new_slots(session, changed, prior_slots, st.slots, lang)
            if not st.slots.get("confirmation_id") and slots.get("phone"):
                found = await _tools.lookup_bookings(phone=slots["phone"])
                bookings = found.get("bookings") or []
                if len(bookings) > 1:
                    await _speak(session, multiple_bookings_reply(bookings, lang), lang)
                    return
                if bookings:
                    st.slots["confirmation_id"] = bookings[0]["confirmation_id"]
                else:
                    st.slots.pop("phone", None)  # a number with no booking is not kept; the next one is read fresh
                    await _speak(session, _NO_BOOKING_FOR_NUMBER.get(lang, _NO_BOOKING_FOR_NUMBER["en"]).format(n=spoken_digit_groups(slots["phone"])), lang)
                    return
            missing = missing_required(st)
            if missing:
                await _speak(session, missing_slot_prompt(intent, missing[0], lang), lang)
                return
            if is_ready_to_confirm(st):
                mark_confirming(st)
                await _speak(session, booking_confirmation_readback(st.slots, "cancel_appointment", lang), lang)

        elif intent == "add_test_booking":
            # Same counter-only rule as book_test above -- adding a test to an existing booking is still a lab
            # test booking.
            await _speak(session, phrase("lab_test_counter_only", lang), lang)

        elif intent == "lookup_booking":
            if not (slots.get("phone") or slots.get("confirmation_id")):
                # Real call 2026-10-12: a bare, content-free utterance mid-booking (e.g. just a patient
                # name) can get misclassified as "lookup_booking" instead of continuing the open
                # book_appointment task -- caught here before it asks the wrong "what number did you
                # book with" question and silently drops the booking already in progress.
                open_task = session.booking
                if open_task is not None and open_task.stage == "collecting" and open_task.action != "lookup_booking":
                    missing = missing_required(open_task)
                    await _speak(session, _next_question(session, open_task, open_task.action, missing, lang), lang)
                    return
                await _speak(session, missing_slot_prompt(intent, "phone", lang), lang)
                return
            result = await _tools.lookup_bookings(
                phone=slots.get("phone"), confirmation_id=slots.get("confirmation_id")
            )
            if not (result.get("bookings") or []):
                sorry = await _clinic_cancelled_reply(session, lang, slots.get("phone"), slots.get("confirmation_id"))
                if sorry:
                    await _speak(session, sorry, lang)
                    return
            await _speak(session, _with_resume(session, lookup_reply(result.get("bookings") or [], lang), lang), lang)

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
            await _speak(session, _with_resume(session, resend_reply(result, lang), lang), lang)

        elif intent == "department_query":
            reply = await _answer_enquiry_intent(intent, slots, lang)
            if reply is None:
                # KCD-lead-3: a real "which department/symptom?" question, same courtesy and the same
                # pending_enquiry follow-up mechanism doctor_availability's missing-slot path already gets,
                # instead of a flat "sorry, unclear" that gives the caller nothing to answer.
                session.pending_enquiry = (intent, session.turn_count)
                await _speak(session, missing_slot_prompt(intent, "symptom_description", lang), lang)
                return
            await _finish_enquiry_turn(session, text, lang, intent, slots, reply, data)

    except ToolCallError as e:
        logger.error("[%s] clinic API call failed: %s", session.call_id, e)
        await _speak(session, phrase("tool_failure", lang), lang, fallback_reason="tool_failure")


async def _recover_booking_turn(session: CallSession, text: str, lang: str) -> bool:
    """sourav, 2026-10-16: the intent model failed (timeout, malformed or empty output) on a turn of an open doctor booking. Rather than
    the generic "little problem", read back a name that was just asked for, or ask the SAME next booking question again. Gives up (False)
    after three misses on the same booking so a broken model cannot loop the caller forever; the normal failure reply then plays."""
    st = session.booking
    if not booking_capture._active(st):
        return False
    question = booking_capture.recover_name_without_model(st, text, lang)
    if question is None:
        missing = missing_required(st)
        if not missing or st.note_retry("model_failure") > 3:
            return False
        question = _next_question(session, st, "book_appointment", missing, lang)
    await _speak(session, question, lang)
    return True


async def _verify_booking_after_error(session: CallSession, st, phone: str) -> dict | None:
    """sourav, 2026-10-16: confirm_booking raised (timeout / network) so we do NOT know whether the booking was created. Ask the clinic
    for this caller's bookings and look for the same date and time: a hit is the real booking (and its real confirmation id) and is
    treated as the success it is; no hit means nothing was booked. Returns the booking as a success dict, {} when none exists, or None
    when even that check failed (the outcome stays unknown and nothing is claimed)."""
    try:
        found = await _tools.lookup_bookings(phone=phone)
    except ToolCallError as e:
        logger.warning("[%s] could not verify the booking after a confirm error (%s)", session.call_id, e)
        return None
    for b in found.get("bookings") or []:
        if b.get("date") == st.slots.get("date") and b.get("time_slot") == st.slots.get("time_slot"):
            return {
                "success": True,
                "confirmation_id": b.get("confirmation_id"),
                "doctor_name": b.get("doctor_name"),
                "date": b.get("date"),
                "time_slot": b.get("time_slot"),
            }
    return {}


async def _reopen_after_no(session: CallSession, st, lang: str) -> None:
    """The caller refused the confirmation and gave nothing to change: collection reopens, the details captured
    stay, and the first question is asked again (KCD-367)."""
    st.stage = "collecting"
    st.pending_charge_inr = None
    # sourav, 2026-10-14: for a doctor booking whose details are all captured, ask WHAT to change instead of restarting at "which doctor?"
    if st.action == "book_appointment" and not missing_required(st):
        await _speak(session, what_to_change_prompt(lang), lang)
        return
    await _speak(
        session,
        missing_slot_prompt(st.action, "doctor_name" if st.action == "book_appointment" else "date", lang),
        lang,
    )


def _followup(session: CallSession, intent: str, slots: dict, text: str, lang: str):
    """The enquiry slots, with the topic of the last few turns filled in when the caller pointed at it or simply
    did not name a different one (agent/enquiry_followup.py). Deterministic; the model decides nothing here."""
    gap = session.turn_count - session.last_enquiry_turn if session.last_enquiry_entities else None
    return resolve_followup_slot(intent, slots, text, lang, session.last_enquiry_entities, gap)


_EXTRA_QUESTION = {
    "patient_age": {"bn": "রোগীর বয়স কত?", "hi": "मरीज़ की उम्र क्या है?", "en": "What is the patient's age?"},
    "patient_address": {"bn": "ঠিকানাটা বলবেন?", "hi": "पता बताइए?", "en": "Could you tell me the address?"},
}

_NAME_ONLY = {"bn": "আপনার নামটা বলবেন প্লিজ?", "hi": "अपना नाम बताइए, प्लीज़?", "en": "Could you tell me your name, please?"}


def _next_question(session: CallSession, st, intent: str, missing: list[str], lang: str) -> str:
    """KCD-103: the next question, one field or a group. The caller-state table (`questions_per_turn`) always wins
    over grouping, and a field already asked is asked alone (agent/slot_grouping.py)."""
    if "time_slot" in missing and st.action == "book_appointment":
        # Owner, 2026-10-19 (fixed twice before): the caller is NEVER asked what time to come. The time is the doctor's sitting, told by the agent. If the doctor and the day are known the
        # question is whether to book that day (the sitting time is then said with it); if one is not known, that one is asked instead.
        missing = [m for m in missing if m != "time_slot"]
        if not missing and st.slots.get("doctor_name") and st.slots.get("date"):
            return slot_proposal.book_that_day_question(st.slots["date"], lang)
        if not missing:
            return missing_slot_prompt(st.action, "date", lang)
    fields = slot_grouping.next_fields(st.action, missing, session.policy.questions_per_turn, st.asked_fields)
    grouped = slot_grouping.grouped_prompt(fields, lang) if len(fields) > 1 else None
    if grouped is None:
        fields = fields[:1]
    st.asked_fields.update(fields)
    if fields[0] == "phone" and st.phone_buffer:
        # a number heard in part: say what was heard and ask for the rest, not the opening question again
        return booking_capture.prompt("phone_partial_long", lang, n=len(st.phone_buffer), digits=spoken_digit_groups(st.phone_buffer))
    if not grouped and fields[0] == "patient_name" and (st.slots.get("phone") or getattr(st, "phone_declined", False)):
        return _NAME_ONLY.get(lang, _NAME_ONLY["en"])  # the number is already known: ask only for the name
    return grouped or missing_slot_prompt(intent, fields[0], lang)


def _resume_tail(session: CallSession, reply: str, lang: str) -> str:
    """KCD-104: what brings the caller back to a booking left open while they asked something else -- the next
    question while details are still being collected, "shall I confirm it" at the confirmation step. Empty when
    there is no booking worth returning to, when the policy allows no question, or when the reply already asks one."""
    st = session.booking
    if not topic_flow.worth_suspending(st) or not topic_flow.resume_allowed(session.policy.questions_per_turn, reply):
        return ""
    if st.stage in ("confirming", "awaiting_charge_confirm"):
        return topic_flow.confirm_resume_line(lang)
    missing = missing_required(st)
    if not missing:
        return ""
    return topic_flow.resume_line(_next_question(session, st, st.action, missing, lang), lang)


def _with_resume(session: CallSession, reply: str, lang: str) -> str:
    tail = _resume_tail(session, reply, lang)
    return f"{reply} {tail}" if tail else reply


def _enter_task(session: CallSession, action: str):
    """The booking state for `action`. A DIFFERENT task started while one is half-done sets the old one aside
    instead of replacing it (KCD-104); it is offered back once the new task is finished."""
    current = session.booking
    if current is not None and current.action == action:
        return current
    if topic_flow.worth_suspending(current):
        session.suspended = current
    session.booking = new_state(action)
    return session.booking


async def _default_to_general_medicine(session: CallSession, st, lang: str) -> None:
    """KCD-dentistry, owner's instruction 2026-10-12: neither a usable doctor name nor a symptom -- the GP
    decides and refers onward, rather than asking the caller to choose a specialist they cannot judge. Shared
    by _resolve_doctor_from_symptom (symptom given but unmatched/ambiguous) and the doctor_name retry fallback
    below (caller explicitly does not know, or simply never gives one after being asked twice)."""
    try:
        gp = await _tools.department_doctors("General Medicine", lang)
    except ToolCallError as e:
        logger.warning("[%s] department_doctors failed (%s)", session.call_id, e)
        gp = {}
    gp_doctors = gp.get("doctors") or []
    if gp_doctors:
        # booking_confirmation_readback's own English template prepends "Dr." itself (the same convention as
        # a caller-given name: slots["doctor_name"] is always the BARE name clinic-api's fuzzy matching
        # resolves, never pre-formatted) -- storing the "Dr. " prefix too would read back "Dr. Dr. X". A no-op
        # for bn/hi: the spoken-name form (full_name_bn/full_name_hi) never carries a "Dr." prefix to begin
        # with, and _find_doctor already matches on those fields, so the booking lookup still resolves it.
        st.slots["doctor_name"] = gp_doctors[0].removeprefix("Dr. ")
    await _speak(session, booking_general_checkup_first_reply(gp_doctors, lang), lang)


async def _resolve_doctor_from_symptom(session: CallSession, st, lang: str) -> None:
    """KCD-dentistry, owner's instruction 2026-10-12: the caller wants an appointment but does not know which
    doctor to see, and gave a symptom instead ("mera kaan mein dard hai") -- resolve it via clinic-api's real,
    clinician-approved department routing (never a model guess at which doctor or department) and name a real
    doctor, so the booking continues straight to the date instead of re-asking for a name the caller does not
    have. Genuinely ambiguous (route_department matched two departments) or unmatched (the symptom matched
    nothing) both default to a General Medicine checkup (_default_to_general_medicine).

    `symptom_description` is consumed (popped) once here either way -- it must never be re-resolved on a later
    turn once a doctor is picked, and must never leak into booking_confirmation_readback's slot dump."""
    symptom = st.slots.pop("symptom_description", None)
    if not symptom:
        return
    try:
        result = await _route_dept(symptom, lang, session)
    except ToolCallError as e:
        logger.warning("[%s] route_department failed (%s) -- defaulting to General Medicine", session.call_id, e)
        result = {"matched": False}
    if result.get("matched"):
        doctors = result.get("doctors") or []
        if doctors:
            # See _default_to_general_medicine's own comment: never store the "Dr. " prefix twice.
            st.slots["doctor_name"] = doctors[0].removeprefix("Dr. ")
            await _speak(session, booking_department_suggestion_reply(_dept_label(result["department_name"], lang), doctors, lang), lang)
            return
    await _default_to_general_medicine(session, st, lang)


async def _after_task_finished(session: CallSession, lang: str) -> None:
    """A task just ended (confirmed, refused or aborted). If another was set aside, offer -- never assume -- to
    go back to it."""
    if session.booking is not None or session.suspended is None:
        return
    if not topic_flow.worth_suspending(session.suspended):
        session.suspended = None
        return
    session.awaiting_resume = True
    await _speak(session, topic_flow.offer_resume(session.suspended.action, lang), lang)


async def _continue_resume_offer(session: CallSession, text: str, lang: str) -> bool:
    """The caller's answer to "shall I go back to it?". A yes restores the booking with everything it held; a
    no drops it; anything else lapses the offer and is handled as a fresh turn."""
    session.awaiting_resume = False
    st, session.suspended = session.suspended, None
    if st is None:
        return False
    answer = classify_yes_no(text, lang)
    if answer == "no":
        await _say_history(session, [history_text.ok_anything_else(lang)], lang)
        return True
    if answer != "yes":
        return False
    st.touch()
    session.booking = st
    if st.stage in ("confirming", "awaiting_charge_confirm"):
        await _speak(session, booking_confirmation_readback(st.slots, st.action, lang), lang)
        return True
    missing = missing_required(st)
    if missing:
        await _speak(session, _next_question(session, st, st.action, missing, lang), lang)
    else:
        mark_confirming(st)
        await _speak(session, booking_confirmation_readback(st.slots, st.action, lang), lang)
    return True


async def _handle_booking_confirmation_turn(session: CallSession, text: str, lang: str) -> bool | None:
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
        if booking_capture.is_hedge(text):
            # sourav, 2026-10-16: "hmm" / "maybe" / "not sure" at the read-back is neither a yes nor a no -- ask again, commit nothing.
            # sourav, 2026-10-16 (final verification): and the details are read back AGAIN, so the caller hears exactly what a "yes" would confirm.
            await _speak(
                session,
                booking_capture.prompt("confirm_unsure", lang) + " " + booking_confirmation_readback(st.slots, st.action, lang),
                lang,
            )
            return
        return False  # KCD-104: not a yes or a no -- the caller handles it as a fresh turn

    if answer == "no":
        _note(session, "corrections")
        if st.action == "cancel_appointment":
            await _speak(session, phrase("cancel_aborted", lang), lang)
            session.booking = None
            return
        if len(text.split()) > 2:
            # KCD-104: "no, make it eleven" is a correction, not just a refusal. Only a BARE no reopens the
            # booking by asking again; anything longer is handled as a fresh turn (the correction is applied and
            # read back; a question is answered; a bare no from the extractor's point of view reopens below).
            session.no_at_confirm = True
            return False
        await _reopen_after_no(session, st, lang)
        return

    # answer == "yes": commit for real.
    try:
        if st.action == "book_appointment":
            phone = effective_phone(st)
            try:
                result = await _tools.confirm_booking(
                    st.hold_token,
                    st.hold_doctor_id,
                    st.slots["date"],
                    st.slots["time_slot"],
                    st.slots["patient_name"],
                    phone,
                    caller_phone=phone,
                    patient_age=st.slots.get("patient_age"),
                    relationship=st.slots.get("relationship") or "self",
                    patient_address=st.slots.get("patient_address") or "",
                    patient_sex=st.slots.get("patient_sex") or "",
                )
            except ToolCallError as e:
                # sourav, 2026-10-16: the request may have reached the clinic even though the answer did not come back. Never say
                # "booked" without proof, never throw the booking away, never book twice: check by the caller's bookings first.
                logger.error("[%s] confirm_booking raised (%s); verifying before saying anything", session.call_id, e)
                result = await _verify_booking_after_error(session, st, phone)
                if result is None:
                    await _speak(session, booking_capture.prompt("confirm_unverified", lang), lang)
                    return
                if not result:
                    await _speak(session, booking_capture.prompt("confirm_retry", lang), lang)
                    return  # the booking stays at the read-back: a "yes" tries the same hold again
            if result.get("reason") == "write_unverified":
                session.booking = None
                await _handle_unverified_write(session, lang)
                return
            await _speak(session, booking_reply(st.slots, result, lang), lang)
            if result.get("success"):
                _record_action(session, "booking_confirmed", result, st.slots)
                await _persist_senior(session, phone)
                session.booking = None
                # Only a DOCTOR appointment is paid and booked over the call at all; having reached here, this one
                # just was. The agent calls the clinic API to send a payment link, then ends the call -- it never
                # takes or states a payment itself (CLAUDE.md's truth boundary keeps money out of its hands the same
                # way it keeps every other fact out of the model's). A failure sending the link is logged, never
                # spoken over the confirmation the caller already heard: the appointment IS still confirmed either
                # way, and the closing line says the link was sent regardless -- if it genuinely failed to queue,
                # that is a clinic-api-side follow-up, not something the caller can act on by being told mid-call.
                confirmation_id = result.get("confirmation_id")
                if confirmation_id:
                    with contextlib.suppress(ToolCallError):
                        await _tools.send_payment_link(confirmation_id)
                    # KCD-lead-2: a separate, warm/friendly WhatsApp message confirming the appointment itself
                    # (doctor, date, time, booking ID) -- distinct from send_payment_link's own message above,
                    # which is about the payment hold. A failure here is logged only, same discipline as the
                    # payment link: the appointment is confirmed either way, and the caller already heard so.
                    with contextlib.suppress(ToolCallError):
                        await _tools.send_booking_confirmation_whatsapp(confirmation_id)
                session.suspended = None  # nothing is offered back after the socket closes below
                await _end_call(session, lang, "payment_link_sent")
                return
            if result.get("reason") != "hold_expired":
                session.booking = None
            else:
                st.hold_token, st.stage = None, "collecting"  # let a retry re-hold

        elif st.action == "book_test":
            phone = effective_phone(st)
            result = await _tools.book_tests(
                st.test_names,
                st.slots["date"],
                st.slots["patient_name"],
                phone,
                caller_phone=phone,
                patient_age=st.slots.get("patient_age"),
                relationship=st.slots.get("relationship") or "self",
            )
            await _speak(session, multi_test_reply(result, lang), lang)
            _record_action(session, "tests_booked", result, st.slots)
            session.booking = None

        elif st.action == "reschedule_appointment":
            result = await _tools.reschedule_appointment(
                st.slots["confirmation_id"], st.slots["new_date"], st.slots["new_time_slot"], caller_phone=st.slots.get("phone"), call_id=session.call_id
            )
            if await _change_refused(session, st, result, lang):
                return
            if result.get("reason") == "write_unverified":
                session.booking = None
                await _handle_unverified_write(session, lang)
                return
            await _speak(session, reschedule_reply(result, lang), lang)
            _record_action(session, "appointment_rescheduled", result, st.slots)
            if result.get("success") or result.get("reason") != "slot_taken":
                session.booking = None
            else:
                st.stage = "collecting"
                st.slots.pop("new_time_slot", None)

        elif st.action == "cancel_appointment":
            confirm_charge = st.stage == "awaiting_charge_confirm"
            result = await _tools.cancel_appointment(st.slots["confirmation_id"], confirm_charge, caller_phone=st.slots.get("phone"), call_id=session.call_id)
            if await _change_refused(session, st, result, lang):
                return
            if not result.get("success") and result.get("reason") == "charge_confirmation_required":
                mark_awaiting_charge_confirm(st, result["charge_inr"])
                await _speak(session, cancel_reply(result, lang), lang)
                return
            await _speak(session, cancel_reply(result, lang), lang)
            _record_action(session, "appointment_cancelled", result, st.slots)
            session.booking = None

        elif st.action == "add_test_booking":
            result = await _tools.add_test_to_booking(st.slots["confirmation_id"], st.slots["test_name"])
            await _speak(session, add_test_reply(result, lang), lang)
            _record_action(session, "test_added", result, st.slots)
            session.booking = None

    except ToolCallError as e:
        logger.error("[%s] clinic API call failed during booking confirmation: %s", session.call_id, e)
        await _speak(session, phrase("tool_failure", lang), lang, fallback_reason="tool_failure")
        session.booking = None


_CHANGE_REFUSED = {
    "not_authorized": {
        "bn": "এই নম্বরটা এই বুকিংয়ের সাথে মেলেনি, তাই আমি এটা বদলাতে পারছি না। বুকিংয়ের সময় যে নম্বর দিয়েছিলেন সেটা আরেকবার বলবেন?",
        "hi": "यह नंबर इस बुकिंग से मेल नहीं खाया, इसलिए मैं इसे बदल नहीं सकती। बुकिंग के समय जो नंबर दिया था वह फिर से बताइए?",
        "en": "That number does not match this booking, so I cannot change it. Could you tell me the number you gave when you booked?",
    },
    "too_many_attempts": {
        "bn": "নিরাপত্তার কারণে এই বুকিংটা এখন আমি বদলাতে পারব না। অনুগ্রহ করে কাউন্টারে এসে বা আমাদের স্টাফের সাথে কথা বলুন।",
        "hi": "सुरक्षा के कारण मैं अभी यह बुकिंग नहीं बदल सकती। कृपया काउंटर पर आइए या हमारे स्टाफ़ से बात कीजिए।",
        "en": "For security I cannot change this booking right now. Please come to the counter or speak to our staff.",
    },
}


async def _change_refused(session: CallSession, st, result: dict, lang: str) -> bool:
    """A cancel / reschedule the clinic refused because the number does not belong to the booking (P0 review 2026-10-28). True when this turn is dealt with: the booking task stays open
    for another try at the number (three misses, or the clinic's own lock, end it), nothing is changed, and nothing about the booking is said to the caller."""
    reason = result.get("reason")
    if result.get("success") or reason not in ("phone_required", "not_authorized", "too_many_attempts"):
        return False
    if reason == "too_many_attempts" or (reason == "not_authorized" and st.note_retry("change_denied") >= 3):
        await _speak(session, _CHANGE_REFUSED["too_many_attempts"].get(lang, _CHANGE_REFUSED["too_many_attempts"]["en"]), lang)
        session.booking = None
        return True
    st.slots.pop("phone", None)
    st.stage = "collecting"
    if reason == "phone_required":
        await _speak(session, missing_slot_prompt(st.action, "phone", lang), lang)
    else:
        await _speak(session, _CHANGE_REFUSED["not_authorized"].get(lang, _CHANGE_REFUSED["not_authorized"]["en"]), lang)
    return True


def _take_gate_release(session: CallSession):
    """Where in the audio the client said playback finished (once), or None."""
    at, session.gate_release_at_s = getattr(session, "gate_release_at_s", None), None
    return at


async def _resync_after_playback(session: CallSession) -> bool:
    """Drop everything captured while the agent was talking, by moving
    processed_until_s to the current end of the decoded buffer. That region
    is muted silence from the client's side; skipping it keeps the turn
    detector from ever analysing it, and -- more importantly -- keeps
    processed_until_s anchored to real time instead of drifting a full
    reply behind, which is what made later turns surface late."""
    buffer_end_s = session.audio.duration_s
    session.processed_until_s = session.gate.resync_target(buffer_end_s, session.processed_until_s, _take_gate_release(session))
    session.resync_pending = False
    logger.info("[%s] resynced to %.2fs after playback", session.call_id, session.processed_until_s)
    return True


def _next_wake_delay(decision) -> float:
    """KCD-049: sleep until the moment the answer could change, not until the
    next tick of a fixed timer. While the caller has spoken this turn the cadence
    is ACTIVE_POLL_INTERVAL_S; when the detector says "commit in x seconds if
    silence continues", wake exactly then (plus a hair, so the audio that
    settles it has landed)."""
    base = ACTIVE_POLL_INTERVAL_S if decision.had_any_speech else POLL_INTERVAL_S
    if decision.commit_after_s is None:
        return base
    return min(base, max(MIN_WAKE_S, decision.commit_after_s + WAKE_SLACK_S))


async def _speculative_completeness(session: CallSession, speech_end_abs: float) -> str | None:
    """KCD-048: transcribe what has been said so far, in the call's current
    language only (no LID, no router state touched), and say whether it reads as
    finished. Any failure is simply "no verdict" -- the baseline threshold then
    applies, so this can only ever shorten a wait it is confident about."""
    session.spec_count += 1
    wav = None
    try:
        session.utt_seq += 1
        wav = await _slice_utterance(session, session.processed_until_s, speech_end_abs, session.utt_seq)
        asr = await asyncio.wait_for(_asr_router.transcribe(session.lang, wav), SPECULATIVE_ASR_TIMEOUT_S)
        # an emergency phrase in what has been said so far ends the wait at once: the turn is treated as
        # finished so the notice is not delayed by a silence timer (agent/emergency.py)
        verdict = "complete" if detect_emergency(asr.text) else ("certain" if is_certain(asr.text, session.lang) else classify_completeness(asr.text, session.lang))
    except Exception as e:  # noqa: BLE001 - speculation is advisory
        logger.info("[%s] speculative transcription unavailable: %s", session.call_id, e)
        return None
    finally:
        if wav:
            with contextlib.suppress(OSError):
                os.remove(wav)
    session.spec_end_abs, session.spec_verdict = speech_end_abs, verdict
    return verdict


async def _decide_turn(session: CallSession, tail, sr: int):
    """Where is the speech (Silero), then has the caller finished
    (agent/endpointing.decide, pure and tested off-pod). With SEMANTIC_ENDPOINTING
    a candidate turn end may trigger one speculative transcription, whose
    completeness verdict is reused for as long as the speech end has not moved."""
    spans, duration_s = await asyncio.to_thread(_turn_detector.spans, tail, sr, session.call_id, session.processed_until_s)
    cfg = session.pause.config(_turn_detector.config) if ADAPTIVE_PAUSE != "off" else _turn_detector.config
    if SEMANTIC_ENDPOINTING:  # add patience for an unfinished sentence; never shorten the wait for a finished-looking one
        # Real calls 2026-10-16 and 2026-10-17: a caller was cut off mid-sentence again. The per-caller adaptive pause learns from
        # the pauses INSIDE a caller's sentences, so a fast speaker is given as little as 0.6 s (pause_profile.SILENCE_MIN_S) -- and a
        # thinking pause ("tell me ... the CBC") is longer than that. The adaptive pause may now only LENGTHEN the baseline wait.
        cfg = never_shorter_than(cfg, _turn_detector.config, ENDPOINT_INCOMPLETE_S)
        # ...and then the two quick cases (see ENDPOINT_CERTAIN_S): they only apply once a speculative transcript has said the words read as a finished answer
        cfg = dataclasses.replace(cfg, complete_confirm_s=ENDPOINT_COMPLETE_S, certain_confirm_s=ENDPOINT_CERTAIN_S)
    if spans and not session.commit_checked and session.last_commit_end_abs is not None:
        # speech resumed after the last turn was cut: was that cut too early for THIS caller?
        resumed_after = max(0.0, float(spans[0]["start"]) - session.last_confirm_s)
        session.pause.note_turn_cut(resumed_after, session.agent_spoke_since_commit)
        session.commit_checked = True
    speech_end_abs = session.processed_until_s + float(spans[-1]["end"]) if spans else None
    completeness = None
    if (
        SEMANTIC_ENDPOINTING
        and speech_end_abs is not None
        and session.spec_end_abs is not None
        and abs(session.spec_end_abs - speech_end_abs) < SPECULATION_MATCH_TOLERANCE_S
    ):
        completeness = session.spec_verdict
    decision = decide_turn_end(spans, duration_s, cfg, completeness=completeness, semantic=SEMANTIC_ENDPOINTING)
    if decision.speculate and speech_end_abs is not None and session.spec_count < MAX_SPECULATIONS_PER_TURN and not _busy_for_speculation():
        verdict = await _speculative_completeness(session, speech_end_abs)
        if verdict is not None:
            decision = decide_turn_end(spans, duration_s, cfg, completeness=verdict, semantic=True)
    if decision.utterance_end_s is not None:
        session.pause.observe_utterance(spans, upto_s=decision.utterance_end_s)
        session.last_confirm_s = decision.confirm_s or cfg.silence_confirm_s
        session.last_commit_end_abs = session.processed_until_s + float(decision.utterance_end_s)
        session.agent_spoke_since_commit = False
        session.commit_checked = False
    return decision


async def _interrupt_playback(session: CallSession, reason: str, resume_from_s: float | None = None) -> None:
    """The caller has taken the floor -- by clicking (manual) or by talking over
    the agent (acoustic barge-in). Stop what is playing on the client, discard
    what the interrupted turn has not yet said, open the gate.

    `resume_from_s`, when given, is where the caller's utterance begins on the
    call timeline: the marker is set there and the usual post-playback resync
    (which would skip past their first words) is cancelled."""
    was_speaking = session.agent_speaking
    if was_speaking:
        _note(session, "barge_ins")  # the caller talked over the agent
    session.speak_epoch += 1
    session.release_gate()
    if session.duplex is not None:
        session.duplex.cancel_pending()
    if reason != "manual":
        # A manual interrupt comes FROM the client, which has already stopped
        # its own playback; only a server-detected one needs to tell it to.
        with contextlib.suppress(Exception):
            await session.ws.send_text(json.dumps({"type": "stop_playback"}))
    if resume_from_s is not None:
        session.processed_until_s = max(session.processed_until_s, resume_from_s)
        session.resync_pending = False
    if was_speaking:
        logger.info("[%s] caller interrupt (%s) -- playback stopped, gate released", session.call_id, reason)
        barge_in_interrupts.record(reason, session.lang)


def _mark_verified(session: CallSession, method: str, patient_ref: str, analysis: dict | None) -> None:
    """Record that a check passed (an OTP, a confirmed date of birth -- whatever the clinic
    decides counts) and remember THIS voice as the verified one (KCD-054). The mechanism that
    verifies lives outside this file (KCD-203); it calls this when it succeeds."""
    session.identity.verify(method, patient_ref)
    session.speaker.reset()
    emb = (analysis or {}).get("voice")
    if emb is not None:
        level = ((analysis or {}).get("near_end") or (None,))[0]
        session.speaker.enroll_embedding(emb, level)


async def _check_speaker_change(session: CallSession, analysis: dict | None, lang: str) -> None:
    if not session.identity.is_verified or not session.speaker.enrolled or not analysis:
        return
    emb = analysis.get("voice")
    if emb is None:
        return  # too little voiced speech: abstain, never guess
    level = (analysis.get("near_end") or (None,))[0]
    result = session.speaker.observe_embedding(emb, level)
    if result.verdict == "changed" and session.identity.revoke("speaker_changed"):
        logger.warning("[%s] voice changed after verification -- verification revoked", session.call_id)
        audio_issue_buckets.record("speaker_change", "verification_revoked", lang)
        await _speak(session, phrase("reverify_notice", lang), lang)


async def _say_history(session: CallSession, statements: list[tuple[str, str]], lang: str, warm: bool = False) -> None:
    """Speak history statements and record their provenance ids (KCD-500/501). `warm` marks a real answer
    (not a question): it gets the thanks, and a senior's kind closing (KCD-512/513)."""
    _rec(session, "history", [sid for sid, _ in statements])
    text = " ".join(t for _, t in statements)
    if warm:
        text, _ = session.acks.decorate(text, lang, already_acknowledged=session.turn_acknowledged)
        text = session.kindness.decorate(text, lang)
    await _speak(session, text, lang)


def _digits(value) -> str:
    return "".join(ch for ch in str(value or "") if ch.isdigit())


def _caller_phone(session: CallSession) -> str:
    return session.identity.phone or "not_provided"


async def _history_step(session: CallSession, lang: str) -> None:
    """Where the history question stands: find the record, verify who is asking, answer."""
    ident = session.identity
    if ident.patient_ref is None:
        session.history_state = "need_phone"
        await _say_history(session, [history_text.ask_phone(lang)], lang)
        return
    decision = patient_policy.history_gate(ident, ident.patient_ref, lang)
    if decision.allowed:
        await _answer_history(session, lang)
        return
    if SECURITY_QUESTIONS != "off":
        await _start_security_check(session, ident.patient_ref, lang)
        return
    # NOTHING beyond the existence of a record is spoken before verification. The refusal offers a
    # person; the caller's yes/no is handled at the top of the next turn.
    session.history_state, session.pending_history, session.awaiting_handoff_offer = None, None, True
    await _say_history(session, [decision.statement], lang)


async def _start_security_check(session: CallSession, patient_ref: str | None, lang: str) -> None:
    """Begin the security questions (KCD-495). With no record found yet (`patient_ref` None) the same
    answers first LOCATE the record (KCD-497), then verify it."""
    session.security = SecurityCheck(patient_ref)
    session.history_state = "security"
    text = session.security.next_question(lang)
    if patient_ref is None:
        text = f"{history_text.no_record(lang)[1]} {text}"  # the operator's "I cannot find it" wording first
    await _say_history(session, [("security_question", text)], lang)


async def _continue_security_flow(session: CallSession, text: str, lang: str, analysis: dict | None) -> bool:
    """One caller answer to a security question. True if this turn belonged to the check."""
    chk = session.security
    if chk is None or chk.finished:
        return False
    if not chk.absorb(text):
        if chk.gave_up_reading:
            session.security, session.history_state = None, None
            await _handoff_to_human(session, "verification_unclear", (lang,))
            return True
        await _say_history(
            session,
            [("security_not_understood", f"{sec_text.did_not_understand(lang)} {chk.next_question(lang)}")],
            lang,
        )
        return True
    await _thank(session, lang)
    if not chk.ready_to_submit():
        await _say_history(session, [("security_question", chk.next_question(lang))], lang)
        return True
    try:
        if chk.finding:
            found = (
                await _tools.find_patient(session.call_id, chk.find_payload())
                if chk.find_payload()
                else {"status": "insufficient"}
            )
            if found.get("status") == "single":
                chk.patient_ref = str(found["patient_ref"])
                session.identity.claim(chk.patient_ref, session.identity.phone)
            elif found.get("status") == "insufficient":
                await _say_history(session, [("security_question", chk.next_question(lang))], lang)
                return True
            else:
                # none or ambiguous: never say which. The operator's "I cannot find that" wording, then ask again.
                stop = chk.record_find_miss()
                if stop:
                    session.security, session.history_state = None, None
                    await _say_history(session, [history_text.cannot_see(lang)], lang)
                    await _handoff_to_human(session, "history_record_not_found", (lang,))
                    return True
                await _say_history(
                    session, [history_text.cannot_see(lang), ("security_question", chk.next_question(lang))], lang
                )
                return True
        result = await _tools.verify_patient(
            session.call_id, int(chk.patient_ref), chk.payload(), session.identity.phone
        )
    except ToolCallError as e:
        logger.error("[%s] verification failed to run: %s", session.call_id, e)
        session.security, session.history_state = None, None
        await _speak(session, phrase("tool_failure", lang), lang, fallback_reason="tool_failure")
        return True
    chk.record_result(result)
    if chk.verified:
        await _on_verified(session, chk, analysis, lang)
        return True
    if chk.finished:
        session.security, session.history_state = None, None
        await _say_history(session, [("security_failed", sec_text.failed_text(lang))], lang)
        await _handoff_to_human(session, "verification_failed", (lang,))
        return True
    await _say_history(
        session, [("security_not_matched", f"{sec_text.not_matched(lang)} {chk.next_question(lang)}")], lang
    )
    return True


async def _on_verified(session: CallSession, chk: SecurityCheck, analysis: dict | None, lang: str) -> None:
    """The server said yes (KCD-495): remember this voice, switch to senior care if the registered date
    of birth says so (KCD-512), offer to continue anything unfinished (KCD-496), then answer."""
    ref = chk.patient_ref
    _mark_verified(session, "security_questions", str(ref), analysis)
    _rec(session, "patient", int(ref))
    session.security, session.history_state = None, None
    parts = [("security_verified", sec_text.verified_text(lang))]
    if session.kindness.activate(chk.age_years):
        apply_caller_state(session.call_state, senior=True)  # slower, one question at a time (Appendix C)
        opening = session.kindness.opening(lang)
        if opening:
            parts.append(("senior_opening", opening))
    await _say_history(session, parts, lang)
    await _offer_unfinished(session, lang)
    if session.pending_history is not None:
        await _answer_history(session, lang)
    if not session.awaiting_draft_answer:
        await _offer_preferences(session, lang)


def _draft_offer_text(draft: dict, lang: str) -> str:
    what = {
        "book_appointment": {"bn": "একটা অ্যাপয়েন্টমেন্ট", "hi": "एक अपॉइंटमेंट", "en": "an appointment"},
        "book_test": {"bn": "একটা টেস্ট", "hi": "एक टेस्ट", "en": "a test"},
        "reschedule_appointment": {
            "bn": "অ্যাপয়েন্টমেন্টের সময় বদল",
            "hi": "अपॉइंटमेंट का समय बदलना",
            "en": "changing an appointment",
        },
        "cancel_appointment": {"bn": "অ্যাপয়েন্টমেন্ট বাতিল", "hi": "अपॉइंटमेंट रद्द करना", "en": "cancelling an appointment"},
        "add_test_booking": {"bn": "একটা টেস্ট যোগ করা", "hi": "एक टेस्ट जोड़ना", "en": "adding a test"},
    }
    label = (what.get(draft.get("action")) or what["book_appointment"]).get(lang) or ""
    return {
        "bn": f"আগের বার আপনি {label} করছিলেন, কাজটা শেষ হয়নি। সেটা কি চালিয়ে যাব, নাকি নতুন করে শুরু করব?",
        "hi": f"पिछली बार आप {label} कर रहे थे, वह पूरा नहीं हुआ। क्या उसे जारी रखूँ, या नए सिरे से शुरू करूँ?",
        "en": f"Last time you were doing {label}, and it was not finished. Shall I continue it, or start fresh?",
    }.get(lang, "")


async def _offer_unfinished(session: CallSession, lang: str) -> None:
    """KCD-496: after verification, offer -- never assume -- to continue an unfinished booking."""
    ident = session.identity
    if session.booking is not None or _tools is None:
        return
    try:
        ctx = await _tools.continuity(_caller_phone(session), session.call_id, int(ident.patient_ref))
    except ToolCallError:
        return
    draft = ctx.get("draft")
    if not draft:
        return
    try:
        info = json.loads(draft.get("slots_json") or "{}")
    except ValueError:
        return
    if not info.get("action"):
        return
    session.pending_draft, session.awaiting_draft_answer = info, True
    await _say_history(session, [("continuity_offer", _draft_offer_text(info, lang))], lang)


async def _offer_preferences(session: CallSession, lang: str) -> None:
    """KCD-499: the patient's stored preferences are OFFERED for confirmation (the address is never read
    aloud), and a stored language bias never overrides what the caller actually says."""
    ident = session.identity
    if _tools is None or ident.patient_ref is None:
        return
    try:
        prefs = await _tools.get_preferences(int(ident.patient_ref), _caller_phone(session))
    except ToolCallError:
        return
    offer = patient_policy.preference_offer(prefs.get("preferences") or {}, ident, ident.patient_ref, lang)
    if offer is None:
        return
    session.pending_pref, session.awaiting_pref_answer = offer, True
    await _speak(session, offer.text, lang)


async def _continue_preference_offer(session: CallSession, text: str, lang: str) -> bool:
    if not session.awaiting_pref_answer or session.pending_pref is None:
        return False
    session.awaiting_pref_answer = False
    offer, session.pending_pref = session.pending_pref, None
    answer = classify_yes_no(text, lang)
    if answer == "yes":
        session.confirmed_preferences = dict(offer.fields)
        _rec(session, "confirmed", dict(offer.fields))
        if offer.fields.get("accessibility_mode") == "slower":
            apply_caller_state(session.call_state, senior=True)  # the slower, one-question-at-a-time delivery
        await _speak(session, history_text.ok_anything_else(lang)[1], lang)
        return True
    if answer == "no":
        await _speak(session, history_text.ok_anything_else(lang)[1], lang)
        return True
    return False


# What a caller's "yes" / "no" looks like to the recogniser that happened to hear it. The offer is asked in one language
# but a one-word reply is language-identified on half a second of audio, so the same "yes" can come back in Latin,
# Bengali or Devanagari letters. Only used for THIS question, where a yes/no is the only thing expected.
_SPOKEN_YES = frozenset("yes yeah yep yup ok okay sure ইয়েস ইয়া ইয়াহ ওকে জি জী হ্যাঁ হ্যা হা যস यस येस यैस यप ओके जी हाँ हां हा".split())
_SPOKEN_NO = frozenset("no nope nah নো না নয় নাহ नो नहीं नही ना".split())


def _answer_to_switch_offer(text: str, target: str, lang: str) -> str | None:
    """"yes" / "no" / None for the answer to "shall I continue in <language>?", in any of the three languages' words."""
    for table in dict.fromkeys((target, lang, "en", "hi", "bn")):
        answer = classify_yes_no(text, table)
        if answer:
            return answer
    words = [w for w in re.split(r"[\s,.!?।|]+", (text or "").lower()) if w]
    if words and len(words) <= 3:
        if any(w in _SPOKEN_NO for w in words):
            return "no"
        if any(w in _SPOKEN_YES for w in words):
            return "yes"
    return None


async def _continue_language_switch_offer(session: CallSession, text: str, lang: str) -> bool:
    """The caller's answer to "would you like me to continue in <language>?" -- asked by the two-in-a-row
    check next to session.lang_history. `classify_yes_no` is tried in the OFFERED language first (that is what
    the question was asked in, and what a caller answering naturally replies in), then the call's current
    language as a fallback (a caller can simply keep answering in the language the call is locked to).

    2026-10-12, real-call bug: the question that triggered the offer used to be answered ONLY on a clear
    "yes" -- a "no", or any reply that is not a clean yes/no (a caller who simply repeats or rephrases their
    original question instead of literally saying "yes"), silently dropped it, leaving the caller to repeat
    themselves with no better luck the second time (confirmed live: the rephrase itself then went unanswered
    too). The offer's outcome decides the SPOKEN LANGUAGE only; whether the question gets answered must not
    depend on it, so it is now answered either way -- in the new language on "yes", in whatever language this
    turn is already in otherwise."""
    if not session.awaiting_language_switch_offer or session.pending_language_switch is None:
        return False
    session.awaiting_language_switch_offer = False
    target, session.pending_language_switch = session.pending_language_switch, None
    pending_text, session.pending_language_switch_text = session.pending_language_switch_text, None
    pending_asr, session.pending_language_switch_asr_result = session.pending_language_switch_asr_result, None
    answer = _answer_to_switch_offer(text, target, lang)
    # Owner, 2026-10-27 (real call): Bengali speech was written in Devanagari twice, so "shall I speak Hindi?" was asked; the caller then said a clear Bengali sentence ("সর্দি কাশি জ্বরের
    # কোন ডাক্তার আছে আজকে"), which is neither yes nor no, and the code below read any reply that is not a clean yes / no as consent -- the call moved to Hindi, the OLD garbled sentence was
    # answered and the new one dropped. A clear sentence (three words or more, nearly all in the script of the language the call is already in) is the caller continuing in that
    # language: that is a "no", and the sentence is a fresh turn answered as it stands. A filler sound ("ah ah") has no such sentence, so the rule below it still applies to it.
    fresh_sentence = False
    if answer is None and target != lang and len(text.split()) >= 3 and script_share(text, lang) >= 0.85:
        answer, fresh_sentence = "no", True
    # Owner, 2026-10-27 (real call): the offer was "shall I speak Bengali?" and the caller went on in Bengali ("…ওটা কি রিসুল্ট এসছে বলেন না"); "বলেন না" is "please tell me", but the word না read as
    # a no, so the call stayed in English. A clear sentence (four words or more) in the OFFERED language is the caller speaking it: that is the yes, whatever small words it holds.
    if answer in (None, "no") and not fresh_sentence and target != lang and len(text.split()) >= 4 and script_share(text, target) >= 0.85:
        answer = "yes"
    # An explicit request in this reply ("বাংলায় বলুন", "speak English") names the language to use, whatever was offered, whatever the yes/no reads as.
    explicit = next((r for l in (lang, "bn", "hi", "en") for r in [detect_language_switch_request(text, l)] if r), None)
    if explicit:
        target, answer = explicit, "yes"
        session.lang_explicit = True
    reply_lang = lang
    if answer == "yes":
        _lock_reply_language(session, target)
        session.lang_router.note_response_language(target)
        # OWNER'S INSTRUCTION, 2026-10-10: a confirmed switch never repeats the opening disclosure/intro --
        # pre-mark the new language as already disclosed so the check a few lines above this never fires for it.
        session.disclosed_langs.add(target)
        await _speak(session, phrase("language_switched", target), target)
        reply_lang = target
    elif answer == "no":
        # Real call 2026-10-12: a "no", or any reply that isn't a clean yes/no, used to leave
        # session.lang_locked False, letting the very next turn's decisive LID silently relock the whole
        # call. An explicit "no" IS reliable evidence of what language the caller is speaking (it is a
        # real word, correctly language-ID'd) -- lock the call to it so that trap cannot fire later.
        if not session.lang_locked:
            _lock_reply_language(session, lang)
    else:
        # Real call 2026-10-13: the offer only fires after the caller has ALREADY spoken two consecutive
        # turns in `target` -- stronger, repeated evidence than this one reply. A filler/hesitation sound
        # ("ah ah") that classify_yes_no cannot read as yes OR no is NOT a "no", and must never be resolved
        # by locking to `lang` -- this turn's OWN language-ID, which is unreliable precisely because the
        # utterance had no real linguistic content to ID in the first place. Treating "unrecognised" the
        # same as "declined" (the 2026-10-12 fix's mistake) is how a caller who had just asked twice to
        # speak Hindi ended up permanently locked to Bengali instead. Read as not having declined what the
        # caller already twice demonstrated, and complete the switch.
        _lock_reply_language(session, target)
        session.lang_router.note_response_language(target)
        session.disclosed_langs.add(target)
        await _speak(session, phrase("language_switched", target), target)
        reply_lang = target
    # OWNER'S INSTRUCTION, 2026-10-10: answer the question that TRIGGERED the offer, now resolved, instead of
    # leaving the caller to repeat it -- that question was set aside, not answered, when the offer itself was
    # asked (see the two-in-a-row check next to session.lang_history). This now runs on "no" and on an unclear
    # reply too, not only "yes" -- see the docstring above for why.
    # sourav, 2026-10-14: the stashed sentence can itself BE the language-switch request ("Speak in Bangla" was said
    # twice, so the offer was asked). Replaying it as an ordinary question sent it to the model, which answered
    # "what's wrong with you? I can talk to you" right after the switch was confirmed. A switch request is already
    # acted on (above), so it is not replayed -- the call just continues in the new language with its context.
    if fresh_sentence and answer == "no":
        return False  # this turn's own sentence is answered next, as a fresh turn, in the language the call stays in
    if pending_text and answer != "no" and any(detect_language_switch_request(pending_text, l) for l in ("bn", "hi", "en")):
        return True
    if pending_text and pending_asr is not None:
        await _answer_content_turn(session, pending_text, reply_lang, pending_asr)
        return True
    # No question was actually stashed (should not happen -- the offer always stashes one). "yes" and the
    # now-completed-switch "unclear" case already spoke a reply above and must not fall through to a second
    # one for the same turn; only a bare "no" (which speaks nothing here) falls through as fresh content.
    return answer != "no"


async def _continue_draft_offer(session: CallSession, text: str, lang: str) -> bool:
    """The caller's answer to "continue it, or start fresh?"."""
    if not session.awaiting_draft_answer or session.pending_draft is None:
        return False
    session.awaiting_draft_answer = False
    info, session.pending_draft = session.pending_draft, None
    answer = classify_yes_no(text, lang)
    if answer == "yes":
        st = new_state(info["action"])
        merge_slots(st, info.get("slots") or {})
        for t in info.get("test_names") or []:
            if t not in st.test_names:
                st.test_names.append(t)
        session.booking = st
        missing = missing_required(st)
        if missing:
            await _speak(session, missing_slot_prompt(info["action"], missing[0], lang), lang)
        else:
            await _speak(session, phrase("unclear", lang), lang)
        return True
    if answer == "no":
        await _say_history(session, [history_text.ok_anything_else(lang)], lang)
        return True
    return False


async def _answer_history(session: CallSession, lang: str) -> None:
    hq, ident = session.pending_history, session.identity
    now = datetime.datetime.now()
    phone = _caller_phone(session)
    try:
        timeline = await _tools.patient_timeline(int(ident.patient_ref), phone, session.call_id)
        if hq.kind == "last_test":
            catalogue = _fast_path.catalogue if _fast_path is not None else None
            name, _form, score = (
                catalogue.match(hq.test_hint or "", "test", lang, HISTORY_TEST_MATCH_MIN)
                if catalogue
                else (None, None, 0.0)
            )
            if not name or score < HISTORY_TEST_MATCH_MIN:
                session.history_state = "need_test"
                await _say_history(session, [history_text.ask_which_test(lang)], lang)
                return
            status = await _tools.patient_test_status(int(ident.patient_ref), name, phone, session.call_id)
            answer = patient_policy.answer_last_test(
                timeline, name, status if status.get("success") else None, lang, now
            )
        elif hq.kind == "appointments":
            answer = patient_policy.answer_appointments(timeline, lang, now)
        elif hq.kind == "medicines":
            answer = patient_policy.answer_medicines(timeline, lang, now)
        else:
            answer = patient_policy.answer_recent_tests(timeline, lang, now)
    except ToolCallError as e:
        logger.error("[%s] history lookup failed: %s", session.call_id, e)
        session.history_state, session.pending_history = None, None
        await _speak(session, phrase("tool_failure", lang), lang, fallback_reason="tool_failure")
        return
    session.history_state, session.pending_history = None, None
    await _say_history(session, answer.statements, lang, warm=True)


async def _continue_history_flow(session: CallSession, text: str, lang: str) -> bool:
    """True if this turn belonged to the history flow (and has been handled)."""
    if session.awaiting_handoff_offer:
        session.awaiting_handoff_offer = False
        answer = classify_yes_no(text, lang)
        if answer == "yes":
            await _handoff_to_human(session, "history_requested_staff", (lang,))
            return True
        if answer == "no":
            await _say_history(session, [history_text.ok_anything_else(lang)], lang)
            return True
        return False  # neither: the offer lapses and the turn is handled normally
    state = session.history_state
    if state == "need_test":
        session.pending_history.test_hint, session.history_state = text, None
        await _history_step(session, lang)
        return True
    if state not in ("need_phone", "need_name"):
        return False
    try:
        data = await _resolve_intent(session, text, lang)
    except ExtractionError:
        data = {"slots": {}}
    slots = data.get("slots") or {}
    session.history_attempts += 1
    try:
        if state == "need_phone":
            phone = _digits(slots.get("phone") or slots.get("contact_phone"))
            if len(phone) < 10:
                if session.history_attempts >= 2:
                    session.history_state = None
                    await _handoff_to_human(session, "history_phone_unclear", (lang,))
                else:
                    await _say_history(session, [history_text.ask_phone(lang)], lang)
                return True
            await _thank(session, lang)
            result = await _tools.identify_patient(phone)
            outcome = patient_policy.identify_outcome(result)
            session.identity.phone = session.identity.phone or phone
            if outcome == "new":
                # nobody on this number: the caller's own details can locate the record (KCD-497)
                if SECURITY_QUESTIONS != "off":
                    await _start_security_check(session, None, lang)
                else:
                    session.history_state, session.pending_history = None, None
                    await _say_history(session, [history_text.no_record(lang)], lang)
            elif outcome == "single":
                session.identity.claim(str(result["patient_ref"]), phone)
                _rec(session, "patient", result["patient_ref"])
                session.history_state = None
                await _history_step(session, lang)
            else:
                session.history_state, session.history_attempts = "need_name", 0
                await _speak(session, patient_policy.resolve_question(lang, 0), lang)
            return True
        # state == "need_name": an open question, answered with a NAME (never chosen from a list)
        spoken = slots.get("patient_name") or text
        result = await _tools.resolve_patient(session.identity.phone or "", spoken, slots.get("patient_age"))
        if result.get("status") == "single" and result.get("basis") == "exact":
            session.identity.claim(str(result["patient_ref"]), session.identity.phone)
            _rec(session, "patient", result["patient_ref"])
            session.history_state = None
            await _history_step(session, lang)
        elif session.history_attempts >= 2:
            session.history_state, session.pending_history = None, None
            await _handoff_to_human(session, "history_patient_unclear", (lang,))
        else:
            # none, ambiguous, or found only by SOUND: ask again, never assume
            await _speak(session, patient_policy.resolve_question(lang, 1), lang)
        return True
    except ToolCallError as e:
        logger.error("[%s] patient lookup failed: %s", session.call_id, e)
        session.history_state, session.pending_history = None, None
        await _speak(session, phrase("tool_failure", lang), lang, fallback_reason="tool_failure")
        return True


async def _save_unfinished_draft(session: CallSession) -> None:
    """KCD-496: a booking that was being collected when the call ended is saved, so the next call can
    offer to continue it. Bounded, never raises."""
    st = session.booking
    if st is None or st.stage != "collecting" or not (st.slots or st.test_names) or _tools is None:
        return
    phone = st.slots.get("phone") or st.slots.get("contact_phone") or session.identity.phone
    if not phone:
        return
    try:
        body = json.dumps({"action": st.action, "slots": st.slots, "test_names": st.test_names})
        await asyncio.wait_for(_tools.save_draft_booking(phone, session.call_id, body), 3.0)
    except Exception as e:  # noqa: BLE001
        logger.warning("[%s] could not save the unfinished booking: %s", session.call_id, e)


def _caller_thanked(text: str, lang: str) -> bool:
    from agent import fast_path_cues as _cues

    table = _cues.table_for(lang)
    return bool(table and table.any(_cues.normalise_cue(text), table.thanks))


def _wholly_hindi(text: str, agreement: float) -> bool:
    """Owner, 2026-10-29 ("it must answer in Bengali until all words are Hindi"): a call is moved to Hindi only by a sentence that is Hindi from end to end -- at least three words, all in Devanagari
    (digits and a Latin letter or two aside), and read with agreeing decoders. A Bengali caller whose sentence the Hindi recogniser merely wrote in Devanagari does not qualify."""
    from agent.lang_select import script_share

    words = [w for w in (text or "").split() if w.strip(".,?!।")]
    return len(words) >= 3 and script_share(text, "hi") >= 0.95 and agreement >= 0.9


def _resolve_reply_language(session: CallSession, detected_lang: str, decisive: bool = True, agreement: float = 1.0, text: str = "") -> str:
    """The language to REPLY in this turn (KCD-lay-terms, a live call 2026-09-28) -- locked to the first language
    established for the call, not re-derived from this turn's own language ID. `detected_lang` is what THIS turn's
    LID/ASR-verification actually settled on (used to select the ASR engine already, by the time this runs) --
    that recognition is unaffected; only which language the agent ANSWERS in is locked.

    Unlocks (moves session.lang) only for: the first turn (nothing locked yet -- CallSession.lang_locked), or an
    explicit request, handled where it is detected (agent/language_switch.detect_language_switch_request's call
    site calls _lock_reply_language directly, since that decision is made later in the same turn than this
    function runs). A detected different voice after identity verification revokes verification and asks to
    re-verify (agent/speaker_change.py) but does not, on its own, reopen the language lock -- a new voice
    re-verifying is free to explicitly ask for a different language like any caller can.

    `decisive` (real call 2026-10-12): a garbled, noisy first utterance ("hano parking a chair") made LID give
    up and fall back to decoder-agreement tie-breaking, which landed on "en" -- then, because it was the FIRST
    turn, that guess LOCKED the entire rest of the call to English, so a crystal-clear Bengali question two
    turns later (LID confidence 1.00) still got an English reply. An inconclusive guess must never be the one
    thing that decides a whole call's language: on a not-decisive first turn, this leaves the lock OPEN
    (session.lang stays at its "bn" default for just this turn) so the next turn -- decisive or not -- still
    gets a fair first shot at locking it, instead of a coin-flip from noise becoming permanent."""
    if not session.lang_locked and decisive and detected_lang == "hi" and not session.lang_explicit and not _wholly_hindi(text, agreement):
        decisive = False  # not Hindi from end to end: the call stays Bengali this turn and nothing is locked
    # Owner, 2026-10-29 ("বডি ফুলে গেছে" answered in Bengali, then the same speaker's next sentence written by the Hindi recogniser as "बोलती है हमारा बॉडी तो फूलेगे छे…" and answered in Hindi): once the
    # call has been answered in Bengali, a single Hindi-LOOKING turn never moves it. Only an explicit request, or the two-turn offer, can.
    if not session.lang_locked and decisive and detected_lang == "hi" and not session.lang_explicit and session.turn_count >= 1:
        decisive = False
    if not session.lang_locked and decisive:
        _lock_reply_language(session, detected_lang)
        session.lang_lock_weak = agreement < 0.6
    session.lang_router.note_response_language(session.lang)
    apply_language(session.call_state, session.lang)
    return session.lang


def _lock_reply_language(session: CallSession, lang: str) -> None:
    session.lang, session.lang_locked = lang, True


def _keep_language(session: CallSession, prior_lang: str) -> str:
    """A turn we could not read (empty, jumbled) says nothing about which language the caller speaks: the call stays in the
    language it was in, and is answered in it. Before this, a swear word or a mumble that language ID happened to label
    Hindi moved a Bengali call to Hindi for the re-ask."""
    session.lang = prior_lang
    session.lang_router.note_response_language(prior_lang)
    apply_language(session.call_state, prior_lang)
    return prior_lang


def _abuse_reply_language(text: str, lang: str, prior_lang: str) -> str:
    """The language to answer abuse in: the script it was written in (Bengali or Devanagari), else the call's own."""
    from agent.lang_select import script_share

    for candidate in ("bn", "hi"):
        if script_share(text, candidate) >= 0.5:
            return candidate
    return prior_lang if prior_lang in _languages_active else lang


# below this decoder agreement a recognition is too doubtful to decide which language the whole call is in
LOW_AGREEMENT_FOR_LOCK = 0.5

_GENERIC_PROMPT_KEYS = ("offtopic_redirect", "how_are_you_reply", "smalltalk_default", "silence_go_on", "ask_problem")


def _last_line_generic(session: CallSession) -> bool:
    """True when the last thing the agent said was a generic prompt (the greeting, a redirect, "how can I help"), whose answer can be anything. A SPECIFIC question ("do you want an appointment with Dr X?",
    "which doctor?") is not generic: the caller's next words are its answer."""
    last = (session.last_spoken or "").strip()
    if not last:
        return True
    for lg in ("bn", "hi", "en"):
        if any(last == phrase(k, lg).strip() for k in _GENERIC_PROMPT_KEYS):
            return True
        if phrase("greeting", lg).strip()[:15] in last:
            return True
    return False


def _free_for_offtopic(session: CallSession) -> bool:
    """The ONLY gate for declining a sentence as "nothing to do with the clinic" (owner, 2026-10-28: "make sure the out-of-topic path never breaks a patient's correct path"). Nothing may be open: no booking, no
    flow, no remembered offer of a doctor waiting for its yes / no, and the last thing said must not have been a specific question."""
    if session.booking is not None or not _nothing_is_open(session, ignore_answering=True):
        return False
    if getattr(session, "dept_pending", None) is not None:
        return False
    if session.answering and not _last_line_generic(session):
        return False
    return True


def _nothing_is_open(session: CallSession, ignore_answering: bool = False) -> bool:
    """True when no question is waiting for an answer and no flow is under way, so a bare "okay" / "thik ache" is a goodbye and not a yes to something (owner,
    2026-10-23, item 6). The last thing said ended in a question, a booking being collected or confirmed, an offer, a pending "which doctor/test?" -- any of them
    makes the same words an answer, and the call goes on."""
    b = session.booking
    return not (
        (session.answering and not ignore_answering)  # ignore_answering: the greeting's own "how can I help?" (and the agent's plain redirects) end in a question but open no flow
        or (b is not None and b.stage in ("collecting", "confirming", "awaiting_charge_confirm"))
        or session.suspended is not None
        or session.pending_entity is not None
        or session.history_state is not None
        or session.report_visit_asked is not None  # sourav, 2026-10-24: 'was your visit 7 days ago?' is waiting for its yes / no
        or session.pending_serial is not None
        or session.serial_flow is not None  # sourav, 2026-10-24: a number is being asked for / read back: 'ok' / 'yes' is the answer, not a goodbye
        or (session.security is not None and not session.security.finished)
        or session.awaiting_handoff_offer
        or session.awaiting_draft_answer
        or session.awaiting_pref_answer
        or session.awaiting_resume
        or session.awaiting_language_switch_offer
        or session.awaiting_test_category
        or session.awaiting_close_answer
        or session.pending_enquiry is not None
        or session.pending_referral is not None
        or session.pending_weekdays is not None
        or session.pending_hours is not None
        or session.pending_day is not None
        or session.pending_when is not None
        or session.pending_prescription is not None
    )


async def _end_call(session: CallSession, lang: str, key: str) -> None:
    """Say the closing line, let it play, then close the connection."""
    session.turn_epoch = session.speak_epoch
    duration = await _speak(session, phrase(key, lang), lang)
    await asyncio.sleep(min(duration, 8.0) + 0.3)
    with contextlib.suppress(Exception):
        await session.ws.close()


def _catalogue_entity_in(norm_text: str, lang: str) -> bool:
    """The clinic's side of the hospital veto for agent/unrelated_caller.py: does the sentence name a test, an FAQ topic, a full doctor name or a doctor's
    surname/alias from the live catalogue? True means "this is a hospital caller". Any failure answers True (an error must never turn a patient away)."""
    if _fast_path is None:
        return False
    try:
        from agent.fast_path import COMMIT_FLOOR as _cf, FAQ_COMMIT_FLOOR as _ff

        catalogue = _fast_path.catalogue
        for lg in dict.fromkeys((lang, "en")):
            for kind, floor in (("test", _cf), ("faq", _ff), ("doctorfull", 0.9)):
                key, _form, score = catalogue.match(norm_text, kind, lg, floor)
                if key and score >= floor:
                    return True
        return _fast_path._named_doctor(fast_path_normalize(norm_text), language_of_script(norm_text, lang)) is not None  # noqa: SLF001
    except Exception:  # noqa: BLE001
        return True


async def _clinical_and_unrelated_guards(session: CallSession, text: str, lang: str) -> bool:
    """Two rule guards that run before the fast path, the cache and the model. True when the turn was answered here.

    1. A caller asking what a result MEANS ("is my report dangerous?", "am I dying?") is never answered with a judgement and never left to the model: a fixed sentence says a doctor
       will explain it and offers the clinician (agent/clinical_safety.py, ported from the sibling repository 2026-10-27; its lab-value rule is narrowed here so a time, a price or
       a test's own digits are not read as a result). The yes / no that follows is read first on the next turn: yes connects a person, no goes on.
    2. A clear WRONG NUMBER ("sorry, wrong number") when nothing else is open: a kind sentence the first time, a polite close the second consecutive time (agent/unrelated_caller.py).
       Only the wrong-number family is acted on. Its 'a name being asked for' family is NOT used: in Bengali it also matches ordinary answers ("thik ache", "whatsapp ache")."""
    if session.awaiting_clinician_offer:
        session.awaiting_clinician_offer = False
        offer = classify_yes_no(text, lang)
        if offer in ("yes", "no"):
            if offer == "yes":
                await _handoff_to_human(session, "clinical_interpretation", (lang,))
            else:
                await _speak(session, phrase("silence_go_on", lang), lang)
            return True
    # "connect me to a person": always possible, and the way out the calm-delivery acknowledgement promises (agent/distress.py)
    if asks_for_person(text) or human_request.asks_for_a_person(text, lang) or human_request.wants_to_complain(text, lang):
        # Owner, 2026-10-29 (live call, three refusals in a row): a request for a person, for staff or a complaint is connected AT ONCE, on the first time of asking. (The 2026-10-28 "I will help you,
        # ask again" step is removed; it left callers pleading.) Only an open clinician offer, above, takes the same words as its yes first.
        session.person_requests += 1
        logger.info("[%s] the caller asked for a person / staff / lodged a complaint (request %d): connecting", session.call_id, session.person_requests)
        await _handoff_to_human(session, "complaint" if human_request.wants_to_complain(text, lang) else "caller_requested", (lang,))
        return True
    # Owner, 2026-10-29: "I will give the name / the day and time / call a little later" -- thanks, "please call again later", and the call is closed
    if call_later.asks_to_call_later(text, lang):
        logger.info("[%s] the caller will call / give the details later: closing politely", session.call_id)
        await _end_call(session, lang, "call_later_final")
        return True
    if clinical_safety.is_clinical_interpretation(text):
        logger.info("[%s] a request to interpret a result: a clinician is offered", session.call_id)
        session.awaiting_clinician_offer = True
        await _speak(session, phrase("clinical_interpretation", lang), lang)
        return True
    # Owner, 2026-10-27: lost and found -- something left behind, or someone to be found: the agent knows nothing about it and never guesses; the call goes to a person (the human-routing placeholder)
    if session.booking is None and lost_found.asks_lost_and_found(text):
        logger.info("[%s] lost and found: routing to a person", session.call_id)
        await _handoff_to_human(session, "lost_and_found", (lang,))
        return True
    # Owner, 2026-10-29: the department being discussed (exists? names? good? when? from Kolkata? a garbled first name) -- see _dept_talk_turn
    if await _dept_talk_turn(session, text, lang):
        return True
    # Owner, 2026-10-27: questions about the doctors JUST offered ("who is better among them, who sits in the morning") keep the offered doctors as the subject
    if await _answer_about_offered_doctors(session, text, lang):
        return True
    # Owner, 2026-10-27 (live test): the FIRST sentence after the greeting is the commonest place for "which doctors do you have?", a wrong number or small talk, and it was never reached: the greeting
    # ends in a question, so the turn counted as "the answer to a question" and every guard below stood down. Only a real open flow (a booking, a pending choice ...) holds them back now.
    free = _free_for_offtopic(session)
    # a wrong place dialled ("is this a girls' school or a police station?"): said directly and the call is closed -- no second chance asked
    if free and off_topic.wrong_place(text) and not unrelated_caller.has_hospital_cue(text, lang, _catalogue_entity_in):
        session.outcome = "unrelated_caller"
        logger.info("[%s] wrong place dialled: told so and closing", session.call_id)
        await _end_call(session, lang, "wrong_place_final")
        return True
    # an insult at the clinic or the agent, asking for nothing: a strike (told once; the next non-clinic turn closes the call)
    if free and off_topic.hostile(text):
        await _handle_offtopic(session, lang)
        return True
    # "what was MY result?" -- never read out over the phone (agent/result_request.py)
    if session.booking is None and result_request.asks_for_own_result(text):
        await _speak(session, phrase("result_counter_only", lang), lang)
        return True
    # "which doctors do you have?" -- a few real examples and the question of whom to book
    if free and doctor_search.asks_which_doctors(text) and await _answer_which_doctors(session, text, lang):
        return True
    kind = None
    if free:
        kind = unrelated_caller.classify(text, lang, _catalogue_entity_in)
    if kind != unrelated_caller.WRONG_NUMBER:
        session.unrelated_streak = 0
        # an explicit chit-chat / idiom phrase ("ঘুরতে যাবে", "মাথায় বাজ পড়েছে"), with no hospital cue anywhere in the sentence (agent/off_topic.py)
        if free and off_topic.chit_chat(text) and not unrelated_caller.has_hospital_cue(text, lang, _catalogue_entity_in):
            await _handle_offtopic(session, lang, text)
            return True
        if session.outcome in ("unrelated_caller", "off_topic"):
            session.outcome = "completed"  # the caller went on to ask for something real
        return False
    session.unrelated_streak += 1
    session.outcome = "unrelated_caller"
    if session.unrelated_streak >= unrelated_caller.CLOSE_AT:
        logger.info("[%s] wrong number, second consecutive turn: closing politely", session.call_id)
        await _end_call(session, lang, "unrelated_final")
    else:
        logger.info("[%s] wrong number: one more chance", session.call_id)
        await _speak(session, phrase("unrelated_first", lang), lang)
    return True


def _dept_infos(dept: str) -> list[dict]:
    """The catalogue rows of a department's doctors, once each, in the clinic's order."""
    if _fast_path is None:
        return []
    seen, out = set(), []
    for info in _fast_path.catalogue._doctor_info.values():  # noqa: SLF001
        if info.get("department") == dept and id(info) not in seen:
            seen.add(id(info))
            out.append(info)
    return out


def _set_dept_state(session: CallSession, dept: str, infos: list[dict], lang: str, say: str = "") -> None:
    """The department and its doctors stay the subject of the next sentences (names, good, when, origin, a first name, a choice) until another department or a doctor is named."""
    names = [dept_talk.spoken_name(i, lang) for i in infos]
    session.last_route = {"dept": dept, "names": names, "turn": session.turn_count}
    session.dept_pending = {
        "kind": "pick",
        "doctors": [{"name": n, "full_name": i.get("full_name") or i.get("name"), "start": None, "end": None} for n, i in zip(names, infos)],
        "date": None, "time": None, "turn": session.turn_count, "say": say,
    }


def _close_pair(text: str, close: list[dict]) -> bool:
    """True when the second doctor sounds almost as much like the sentence as the first (within 0.08): both are offered."""
    toks = [t for t in dept_talk._tokens(text) if t not in dept_talk._TITLES]  # noqa: SLF001
    windows = [name_skeleton("".join(toks[i : i + n])) for n in (1, 2, 3) for i in range(len(toks) - n + 1)]

    def best(info: dict) -> float:
        parts = [p for p in (info.get("full_name_bn") or "").split()]
        c = name_skeleton("".join(parts[:-1])) if len(parts) >= 2 else ""
        return max((difflib.SequenceMatcher(None, w, c).ratio() for w in windows if w), default=0.0) if c else 0.0

    return len(close) > 1 and abs(best(close[0]) - best(close[1])) < 0.08


async def _head_pain_turn(session: CallSession, text: str, lang: str) -> bool:
    """Owner, 2026-10-29 ("mathar betha doctor"): a headache is asked about ONCE -- "only a headache, or did you hurt your head?" -- and the answer decides the department: hurt -> General Medicine,
    a plain headache -> Neurology. From there it is the ordinary department answer (who sits, shall I book?), so a yes goes on to the booking. Nothing here states a fact about a person: the doctors
    and sittings are the clinic's own rows. Words: agent/head_pain.py."""
    if _tools is None or not (text or "").strip() or len(text.split()) > 20:
        return False
    asked = getattr(session, "head_check", None)
    if asked and session.turn_count - asked["turn"] <= 2:
        session.head_check = None
        if head_pain.hurt(text):
            dept = "General Medicine"
        elif head_pain.a_pain_answer(text) or head_pain.asks_about_head_pain(text) or booking_capture.confirm_answer(text, lang) == "yes":
            dept = "Neurology"
        else:
            return False  # not an answer to the question: the sentence is handled as any other
        try:
            routed = await _tools.department_doctors(dept, lang)
        except ToolCallError:
            return False
        if not routed.get("matched"):
            return False
        reply = await _route_reply(session, routed, "", lang)
        session.dept_topic = dept
        session.last_route = {"dept": dept, "names": routed.get("doctors_today") or routed.get("doctors") or [], "turn": session.turn_count}
        await _speak(session, _with_resume(session, reply, lang), lang)
        return True
    if getattr(session, "last_doctor_enquiry", None) and booking_carryover.recent(session.last_doctor_enquiry, session.turn_count):
        return False  # a doctor is already the subject
    if head_pain.asks_about_head_pain(text) and not dept_talk.dept_in_text(text):
        session.head_check = {"turn": session.turn_count}
        await _speak(session, head_pain.question(lang), lang)
        return True
    return False


def _names_offered_doctor(session: CallSession, text: str) -> bool:
    """Owner, 2026-10-29 ("सुतापा चटर्जी" answering "which of them?" was re-asked as "are you asking about Dr. Sutapa?"): when the sentence names ONE of the doctors just offered (any script), it is the
    answer to the offer and the booking goes on -- the name is not asked about again."""
    pend = getattr(session, "dept_pending", None)
    if not pend or len((text or "").split()) > 8:
        return False
    if pend.get("kind") == "pick":
        docs = pend.get("doctors") or []
    elif pend.get("kind") == "confirm_one" and pend.get("doctor"):
        docs = [pend["doctor"]]
    else:
        return False
    return bool(docs) and doctor_search.choose_by_name(text, [[d.get("name", ""), d.get("full_name", "")] for d in docs]) is not None


async def _name_rescue(session: CallSession, text: str, lang: str) -> bool:
    """Owner, 2026-10-29 ("it is fine to ask again, but if you hear a name, check the table and ask"): a turn that would be dropped or re-asked in general words ("someone else is talking", "say that again") but
    whose words sound like one of OUR doctors is answered with "আপনি কি ডাঃ X-এর কথা জিজ্ঞেস করছেন?". The department being discussed is searched first (looser), then every doctor (stricter). Never an answer
    about the doctor -- only the question; the yes that follows is read by _dept_talk_turn."""
    if _fast_path is None or not (text or "").strip() or len(text.split()) > 14 or _names_offered_doctor(session, text):
        return False
    topic = getattr(session, "dept_topic", None) or (getattr(session, "last_route", None) or {}).get("dept")
    pools = []
    if topic:
        pools.append((_dept_infos(topic), 0.5))
    seen, everyone = set(), []
    for info in _fast_path.catalogue._doctor_info.values():  # noqa: SLF001
        if id(info) not in seen:
            seen.add(id(info))
            everyone.append(info)
    pools.append((everyone, 0.62))
    for infos, floor in pools:
        close = dept_talk.close_doctors(text, infos, floor=floor)
        if close:
            first = dept_talk.spoken_name(close[0], lang)
            say = dept_talk.confirm_name(first, lang)
            session.doctor_check = {"info": close[0], "turn": session.turn_count}
            await _speak(session, say, lang)
            return True
    return False


async def _weekly_lines(infos: list[dict], lang: str) -> list[str]:
    """Each doctor's sitting days and hours, from the clinic's schedule table (the weekly pattern, never a date)."""
    async def one(info: dict):
        try:
            wd = await _tools.doctor_weekdays(info.get("full_name") or info.get("name"))
        except ToolCallError:
            return None
        if not wd.get("found") or not wd.get("weekdays"):
            return None
        hours = set((wd.get("hours") or {}).values())
        return dept_talk.spoken_name(info, lang), list(wd["weekdays"]), (next(iter(hours)) if len(hours) == 1 else None)

    got = [g for g in await asyncio.gather(*[one(i) for i in infos]) if g]
    # doctors who sit on the same days at the same hours are said together ("ডাঃ A এবং ডাঃ B সোম বুধ শুক্র বারে সকাল দশটা থেকে দুপুর বারোটা পর্যন্ত বসেন।")
    groups: dict[tuple, list[str]] = {}
    for name, days, hours in got:
        groups.setdefault((tuple(days), hours), []).append(name)
    lines = []
    for (days, hours), who in groups.items():
        line = doctor_enquiry_cues.weekdays_reply(dept_talk._join(who, lang), list(days), None, lang)  # noqa: SLF001
        if hours:
            line = line.rstrip("।. ") + {"bn": ", ", "hi": ", ", "en": ", "}[lang if lang in ("bn", "hi") else "en"] + dept_talk.hours_words(hours, lang) + ("।" if lang in ("bn", "hi") else ".")
        lines.append(line)
    return lines


async def _dept_talk_turn(session: CallSession, text: str, lang: str) -> bool:
    """Owner, 2026-10-29 (live call): "অর্থপেটি ডাক্তার ... কেন বসে" -> "হ্যাঁ, বসেন।"; "ডাক্তারের নাম কি আছে" -> the count and all the department's doctors; "কলকাতা ডাক্তার এদিকের মালদারে" -> each doctor's yes / no;
    "ও ভালো কোনটা হবে কখন বসবে" -> all are good, every doctor's days and hours, who to book; a first name said as the caller can ("শুভজিৎ দিপ") -> the one doctor it sounds like. Every department, all three
    languages; the department stays the subject. Nothing here is a guess about a person: the doctors are the clinic's own rows, and a name is matched only among the department's doctors."""
    if _fast_path is None or _tools is None or _booking_open(session) or len(text.split()) > 30 or _names_offered_doctor(session, text):
        return False
    # the answer to "are you asking about Dr. X?" asked a moment ago
    chk = getattr(session, "doctor_check", None)
    if chk and session.turn_count - chk["turn"] <= 2:
        session.doctor_check = None
        answer = booking_capture.confirm_answer(text, lang) if len(text.split()) <= 4 else None  # a longer sentence is a new question, not a "yes"
        if answer == "yes":
            info = chk["info"]
            session.last_doctor_enquiry = {"doctor_name": info.get("name"), "date": None, "turn": session.turn_count}
            if chk.get("date"):  # the day asked about ("…বসবে আজকে") is answered for that day
                await _speak_day_availability(session, info.get("name"), chk["date"], lang)
                return True
            lines = await _weekly_lines([info], lang)
            say = " ".join(lines + [dept_talk.ask_which(lang, 1)]) if lines else dept_talk.ask_which(lang, 1)
            await _speak(session, say, lang)
            return True
        if answer == "no":
            await _speak(session, missing_slot_prompt("doctor_availability", "doctor_name", lang), lang)
            return True
    mention = dept_talk.dept_in_text(text)
    if mention:
        session.dept_topic = mention  # named by the caller: this is the subject until another department is named
    topic = mention or getattr(session, "dept_topic", None) or (getattr(session, "last_route", None) or {}).get("dept")
    if not topic:
        return False
    infos = _dept_infos(topic)
    if not infos:
        return False
    label = _dept_label(topic, lang)
    names = [dept_talk.spoken_name(i, lang) for i in infos]
    asks_names, asks_good, asks_when, asks_origin = dept_talk.asks_names(text), dept_talk.asks_good(text), dept_talk.asks_when(text), dept_talk.asks_origin(text)
    doctor_in_mind = booking_carryover.recent(session.last_doctor_enquiry, session.turn_count)

    # does a <department> doctor sit here?  -> "yes, they sit." and nothing else
    if mention and dept_talk.yes_no_existence(text) and not asks_names:
        say = dept_talk.yes_sits(lang)
        _set_dept_state(session, topic, infos, lang, say)
        await _speak(session, say, lang)
        return True
    # are these doctors from Kolkata?  -> each of the department's doctors
    if asks_origin and not _text_names_a_doctor(text, lang):
        values = [(i.get("from_kolkata") or "").strip().lower() for i in infos]
        if values and all(v == "yes" for v in values):
            say = dept_talk.origin_all_yes(_dept_possessive(label, lang), label, lang)
        else:
            say = dept_talk.origin_reply([(n, i.get("from_kolkata") or "") for n, i in zip(names, infos)], lang)
        _set_dept_state(session, topic, infos, lang, say)
        await _speak(session, say, lang)
        return True
    # what are the doctors' names?  -> the count first, then every name
    if asks_names and not asks_when and not asks_good:
        say = dept_talk.names_reply(label, _dept_possessive(label, lang), names, lang)
        _set_dept_state(session, topic, infos, lang, say)
        await _speak(session, say, lang)
        return True
    # who is good / when do they sit  -> all are good; every doctor's days and hours; who to book
    if (asks_good and asks_when) or (asks_when and not doctor_in_mind and not _text_names_a_doctor(text, lang) and len(text.split()) <= 8 and (mention is None)):
        lines = await _weekly_lines(infos, lang)
        if lines:
            say = " ".join(([dept_talk.all_are_good(lang)] if asks_good else []) + lines + [dept_talk.ask_which(lang, len(lines))])
            _set_dept_state(session, topic, infos, lang, say)
            await _speak(session, say, lang)
            return True
    # the caller checks one or two of the doctors just told ("শুভদীপ আর তপন কি?"): the SAME names again
    told = getattr(session, "last_route", None) or {}
    if told.get("names") and not (asks_names or asks_good or asks_when or asks_origin) and len(text.split()) <= 20 and not mention and not booking_carryover.asks_to_book(text) and not booking_carryover.commits_to_booking(text):
        pool = [i for i in infos if dept_talk.spoken_name(i, lang) in told["names"]] or infos
        hits = dept_talk.named_doctors(text, pool)
        if hits and len(hits) < len(pool):
            say = dept_talk.repeat_names([dept_talk.spoken_name(i, lang) for i in hits], lang)
            await _speak(session, say, lang)
            return True
    # a first name said in a garbled way, among the department's own doctors  -> only that doctor's name
    if not (asks_names or asks_good or asks_when or asks_origin) and len(text.split()) <= 14 and not mention:
        close = dept_talk.close_doctors(text, infos)
        if close:
            # Owner, 2026-10-29: "ও দিপ শুভজিৎ দিপ" -- a name said as the caller can: the doctor it sounds like is ASKED about ("আপনি কি ডাঃ শুভদীপ মন্ডলের কথা জিজ্ঞেস করছেন?"), never answered about unasked;
            # two that sound alike are offered as "A, or B?"
            first = dept_talk.spoken_name(close[0], lang)
            say = dept_talk.confirm_either(first, dept_talk.spoken_name(close[1], lang), lang) if len(close) > 1 and dept_talk.close_doctors(text, [close[0], close[1]], floor=0.45) == close and _close_pair(text, close) else dept_talk.confirm_name(first, lang)
            session.doctor_check = {"info": close[0], "turn": session.turn_count}
            await _speak(session, say, lang)
            return True
    return False


async def _answer_about_offered_doctors(session: CallSession, text: str, lang: str) -> bool:
    """The doctors a department answer just offered ("tomorrow these doctors sit: A and B. Whom shall I book?") stay the subject while the caller asks about THEM: "who is better among them", "who sits
    in the morning" (owner, 2026-10-27; it used to answer "sorry, say again" and then "tell me your problem", losing the whole conversation). All our doctors are good (the approved answer); the
    morning / evening question is answered from the real sitting hours, and when exactly one doctor fits, the question is whether to book him. A doctor named in the sentence is the caller's
    choice and is left to _dept_search_turn."""
    pend = getattr(session, "dept_pending", None)
    if not pend or pend.get("kind") != "pick" or session.turn_count - pend["turn"] > _PICK_MEMORY_TURNS:
        return False
    items = pend.get("doctors") or []
    if len(items) < 2 or doctor_search.choose_by_name(text, [[d.get("name", ""), d.get("full_name", "")] for d in items]) is not None:
        return False
    part = doctor_search.asked_part_of_day(text)
    good = doctor_search.asks_who_is_good(text) and re.search(r"(?:কে|কোন|কার|কাকে|কারা|who|which|कौन|किस|किसे)", text, re.I) is not None
    if not part and not good:
        return False
    say, one = doctor_search.offered_followup_reply(items, part, good, lang)
    if one is not None:
        session.dept_pending = {"kind": "confirm_one", "doctor": one, "date": pend.get("date"), "time": pend.get("time") or one.get("start"), "turn": session.turn_count, "say": say}
    else:
        pend["turn"] = session.turn_count  # the list stays open
    await _speak(session, say, lang)
    return True


async def _department_week_reply(route: dict, lang: str) -> tuple[str, list] | None:
    """One department's doctors, each with the weekdays they sit (the weekly pattern from the clinic API, never a date), then the question of whom to book. Doctors without a stored schedule are left out."""
    lines = []
    items = []
    for name in (route.get("doctors") or [])[:4]:
        try:
            wd = await _tools.doctor_weekdays(name)
        except ToolCallError:
            continue
        if not wd.get("found") or not wd.get("weekdays"):
            continue
        hours = set((wd.get("hours") or {}).values())
        titled = name if name.startswith(("ডাঃ", "ডাক্তার", "ডক্টর", "डॉ", "Dr")) else {"bn": "ডাঃ ", "hi": "डॉ. "}.get(lang, "Dr. ") + name
        lines.append(doctor_enquiry_cues.weekdays_reply(titled, list(wd["weekdays"]), next(iter(hours)) if len(hours) == 1 else None, lang))
        first = sorted(hours)[0] if hours else ""
        start, _, end = first.partition("-")
        items.append({"name": name, "full_name": name, "start": start.strip() or None, "end": end.strip() or None})
    if not lines:
        return None
    dept = _dept_label(route["department_name"], lang)
    intro = {"bn": f"{dept} বিভাগে", "hi": f"{dept} विभाग में", "en": f"In {dept}:"}[lang if lang in ("bn", "hi") else "en"]
    # owner, 2026-10-28: with ONE doctor the question is not "which doctor"
    ask = {
        "bn": "কোন ডাক্তারের সাথে অ্যাপয়েন্টমেন্ট করতে চান?" if len(lines) > 1 else "অ্যাপয়েন্টমেন্ট করতে চান?",
        "hi": "किस डॉक्टर के साथ अपॉइंटमेंट लेना चाहेंगे?" if len(lines) > 1 else "अपॉइंटमेंट लेना चाहेंगे?",
        "en": "Which doctor would you like to book?" if len(lines) > 1 else "Would you like to book an appointment?",
    }[lang if lang in ("bn", "hi") else "en"]
    return f"{intro} {' '.join(lines)} {ask}", items


async def _answer_which_doctors(session: CallSession, text: str, lang: str) -> bool:
    """"Which doctors do you have?" / "কোন কোন ডাক্তার বসে" / "क्या क्या डॉक्टर हैं" (owner, 2026-10-27): every department has doctors -- a few REAL ones from the clinic's own tables as examples, then
    the question of whom to book. Never a name the tables do not hold; if the lookup fails the answer simply has no examples."""
    if _tools is None:
        return False
    # Owner, 2026-10-28: "বুক এর ডাক্তার ... কোন কোন ডাক্তার বসে এই সপ্তাহে" names a department ("chest" = heart): only THAT department's doctors, each with the weekdays they sit
    try:
        route = await _route_dept(text, lang, session)
    except ToolCallError:
        route = {}
    if route.get("matched") and route.get("doctors"):
        weekly = await _department_week_reply(route, lang)
        if weekly:
            reply, items = weekly
            # the doctors just offered stay the subject ("কে ভালো এর মধ্যে", "who sits in the morning"), and the department stays the subject of the next questions
            if len(items) == 1:
                session.dept_pending = {"kind": "confirm_one", "doctor": items[0], "date": None, "time": None, "turn": session.turn_count, "say": reply}
            else:
                session.dept_pending = {"kind": "pick", "doctors": items, "date": None, "time": None, "turn": session.turn_count, "say": reply}
            session.last_route = {"dept": route["department_name"], "names": [i["name"] for i in items], "turn": session.turn_count}
            await _speak(session, _with_resume(session, reply, lang), lang)
            return True
    examples: list[tuple[str, str]] = []
    for dept in ("General Medicine", "Cardiology", "Orthopaedics"):
        try:
            res = await _tools.department_doctors(dept, lang)
        except ToolCallError:
            continue
        names = res.get("doctors") or []
        if res.get("matched") and names:
            examples.append((_dept_label(dept, lang), names[0]))
    await _speak(session, _with_resume(session, doctor_search.overview_reply(examples, lang), lang), lang)
    return True


async def _handle_offtopic(session: CallSession, lang: str, text: str | None = None) -> None:
    """Talk about nothing to do with Sonoscan: the first two times in a row, a polite refusal and what the agent CAN help with; the third closes the call with a thank-you (the owner's
    rule, 2026-10-27). "In a row" = the previous such turn was the turn before this one; any real turn in between starts the count again. Nothing is asked or stored."""
    if text:
        lang = language_of_script(text, lang)  # owner, 2026-10-28: a Bengali sentence got the Hindi refusal because the call's language was still Hindi
    in_a_row = session.offtopic_streak > 0 and session.turn_count - session.offtopic_last_turn <= 1
    session.offtopic_streak = session.offtopic_streak + 1 if in_a_row else 1
    session.offtopic_last_turn = session.turn_count
    session.outcome = "off_topic"
    if session.offtopic_streak >= off_topic.CLOSE_AT:
        logger.info("[%s] off-topic talk, third turn in a row: closing the call politely", session.call_id)
        await _end_call(session, lang, "offtopic_final")
        return
    logger.info("[%s] off-topic talk (%d): redirected to what the clinic can help with", session.call_id, session.offtopic_streak)
    await _speak(session, phrase("offtopic_redirect", lang), lang)
    # the redirect ends in a question, which would make the caller's next sentence count as "the answer to a question" and keep the second strike from being recognised: nothing is open
    session.awaiting_answer = False


_CLINIC_BUSINESS = re.compile(
    r"স্টাফ|কর্মী|ম্যানেজার|অফিসার|কাউন্টার|রিসেপশন|হেল্প|কম্প্লে|কমপ্লে|অভিযোগ|লিঙ্ক|লিংক|হোয়াটস|পেমেন্ট|রিপোর্ট|বিল|টাকা|ডাক্তার|ডক্টর|বুকিং|অ্যাপয়েন্টমেন্ট|অ্যাপ্ল|টেস্ট|ফি\b|রশিদ|"
    r"स्टाफ|मैनेजर|काउंटर|रिसेप्शन|शिकायत|लिंक|व्हाट्सऐप|पेमेंट|रिपोर्ट|बिल|पैसे|डॉक्टर|बुकिंग|अपॉइंटमेंट|टेस्ट|"
    r"\b(?:staff|manager|counter|reception\w*|help\w*|complain\w*|link|whatsapp|payment|report|bill|doctor|booking|appointment|test|fee|receipt)\b",
    re.I,
)


async def _offtopic_by_verdict(session: CallSession, text: str, lang: str, data: dict, asr_result) -> bool:
    """Second layer: the intent model called the sentence small talk or unclear AND nothing in it points at the clinic (no hospital cue, no catalogue name, no symptom or body part, no
    slot the model filled) AND it is a real sentence (four words or more) heard with good agreement between the decoders. Only then is it off-topic. True when answered here.
    Everything doubtful -- a short or garbled sentence, a health word, an open booking or question -- keeps the old path (the model's reply, or "say that again")."""
    if not _free_for_offtopic(session):
        return False
    if len(text.split()) < 4 or getattr(asr_result, "decoder_agreement", 0.0) < 0.85 or off_topic.has_health_word(text):
        return False
    # Owner, 2026-10-29: "out of context must be very loose". A request for staff / a person / a complaint / lost property, or any word of the clinic's own business (staff, link, report, bill, payment, WhatsApp,
    # doctor, booking ...) is never "off topic": the model's verdict is not even asked.
    if human_request.asks_for_a_person(text, lang) or human_request.wants_to_complain(text, lang) or lost_found.asks_lost_and_found(text) or _CLINIC_BUSINESS.search(text):
        return False
    verdict = unrelated_caller.llm_verdict(text, lang, data.get("slots"), _catalogue_entity_in)
    if verdict != unrelated_caller.VERDICT_UNRELATED:
        return False
    await _handle_offtopic(session, lang)
    return True


def _booking_open(session: CallSession) -> bool:
    """A booking is being collected or read back: a question of its own is waiting for the caller."""
    b = session.booking
    return b is not None and b.stage in ("collecting", "confirming", "awaiting_charge_confirm")


def _silence_clock(session: "CallSession") -> float:
    """The clock the quiet is measured on. On the PCM path it is the call's AUDIO time (seconds of audio actually received and processed): if the event loop stalls, or audio stops arriving,
    wall-clock seconds pass but audio seconds do not, so a stall can no longer read as a caller who stopped talking (blueprint 2026-10-29, section 4.2). Browser WebM sessions, which have no sample
    clock, keep the wall clock. Watchdogs and timeouts elsewhere stay on the wall clock."""
    audio = getattr(session, "audio", None)
    if audio is not None and hasattr(audio, "duration_s"):
        return float(audio.duration_s)
    return time.monotonic()


async def _handle_silence(session: CallSession, quiet: bool) -> bool:
    """The caller has said nothing since the agent stopped speaking. After SILENCE_PROMPT_S ask once whether there is
    anything else (one question; the call ends if not); if still nothing SILENCE_CLOSE_S later, end the call.
    True when the call was ended. `quiet` is False whenever there is speech, or the agent is mid-turn."""
    if not quiet or session.dispatch_lock.locked():
        session.silence_since = None
        return False
    if _ingress_backlog_ms(session) > INGRESS_SETTLE_MS:
        session.silence_since = None  # audio is still queued: the quiet seen so far may be a stall, not the caller
        return False
    now = _silence_clock(session)
    if session.agent_speaking and time.monotonic() < session.speak_deadline:
        # Owner, 2026-10-24: "THERE WAS NO SILENCE OR PAUSE, STILL THE AI TOLD BELOW WITHIN MILLISECONDS" -- the clock ran while the question was still playing. It starts when the
        # playback is over.
        session.silence_since = None
        return False
    if session.silence_since is None:
        session.silence_since = now
        return False
    waited = now - session.silence_since
    if session.turn_count == 0:
        # Owner, 2026-10-23: a caller who has said NOTHING since the greeting is called "hello" up to SILENCE_HELLOS times; unanswered after the last, the call is cut
        # at once (no "anything else?" -- there was no conversation). A caller who HAS spoken keeps the existing prompt-then-close below, unchanged.
        if session.silence_hellos < SILENCE_HELLOS and waited >= SILENCE_PROMPT_S:
            session.silence_hellos += 1
            session.silence_since = None
            session.turn_epoch = session.speak_epoch
            logger.info("[%s] no word from the caller yet -- hello %d of %d", session.call_id, session.silence_hellos, SILENCE_HELLOS)
            await _speak(session, phrase("hello_check", session.lang), session.lang)
            return False
        if session.silence_hellos >= SILENCE_HELLOS and waited >= SILENCE_PROMPT_S:
            logger.info("[%s] no answer to %d hellos -- cutting the call", session.call_id, SILENCE_HELLOS)
            session.outcome = "silence_timeout"
            _note(session, "silence_timeout")
            with contextlib.suppress(Exception):
                await session.ws.close()
            return True
        return False
    if session.silence_prompts == 0 and waited >= SILENCE_PROMPT_S:
        session.silence_prompts = 1
        _note(session, "silence_prompts")
        session.silence_since = None
        if _booking_open(session):
            # Real call 2026-10-23: "anything else? if not I will end the call" was said while the agent was waiting for the phone number; "nothing else" then ended the call and the
            # booking was lost. With a booking question open the caller is told to take their time instead -- no "anything else?", so a later "no" is not read as a goodbye.
            session.turn_epoch = session.speak_epoch
            logger.info("[%s] %.1fs of silence inside a booking -- waiting for the answer", session.call_id, waited)
            await _speak(session, phrase("booking_wait", session.lang), session.lang)
            return False
        session.awaiting_close_answer = True
        session.turn_epoch = session.speak_epoch
        logger.info("[%s] %.1fs of silence -- asking whether there is anything else", session.call_id, waited)
        await _speak(session, phrase("silence_prompt", session.lang), session.lang)
        return False
    if session.silence_prompts >= 1 and waited >= SILENCE_CLOSE_S * (BOOKING_CLOSE_FACTOR if _booking_open(session) else 1):
        logger.info("[%s] still silent after the prompt -- ending the call", session.call_id)
        session.outcome = "silence_timeout"
        _note(session, "silence_timeout")
        await _end_call(session, session.lang, "idle_close")
        return True
    return False


async def _handle_barge_in(session: CallSession, event) -> None:
    sr = session.audio.sample_rate if hasattr(session, "audio") else 16000
    await _interrupt_playback(session, "acoustic", resume_from_s=max(0.0, event.at_sample / sr - BARGE_IN_PREROLL_S))


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
        await asyncio.sleep(session.take_wake_delay())

        if time.time() - session.last_activity > IDLE_TIMEOUT_S:
            logger.info("[%s] idle timeout, closing", session.call_id)
            session.turn_epoch = session.speak_epoch
            await _speak(session, phrase("idle_close", session.lang), session.lang)
            with contextlib.suppress(Exception):
                await session.ws.close()
            return

        if time.time() - session.last_heartbeat > HEARTBEAT_INTERVAL_S:
            session.last_heartbeat = time.time()
            with contextlib.suppress(Exception):
                await session.ws.send_text('{"sender":"_ping","text":""}')

        # --- half-duplex gate: never run turn detection on our own voice ---
        backstops_before = session.gate.backstop_releases
        if session.gate.blocked():
            session.silence_since = None  # the agent is talking: the quiet starts when it stops
            continue
        if session.gate.backstop_releases != backstops_before:
            logger.warning("[%s] no playback-done from client, released gate on deadline", session.call_id)

        if session.resync_pending:
            await _resync_after_playback(session)
            continue

        if _ingress_backlog_ms(session) > INGRESS_SETTLE_MS:
            continue  # audio that came before any possible turn end is still queued: deciding now would read the silence of a stall

        sr = session.audio.sample_rate
        tail = session.audio.tail_tensor(session.processed_until_s)
        if tail.numel() < int(0.2 * sr):
            continue  # not enough new audio to judge yet -- not an error

        result = await _decide_turn(session, tail, sr)
        session.next_wake_s = _next_wake_delay(result)
        if result.utterance_end_s is None:
            # sourav, 2026-10-24: a click, breath or line hiss is "transient" (shorter than a word), not the caller speaking -- it used to reset the hello clock, so a caller who
            # said nothing was never called "hello" (agent/silence_policy.py).
            if await _handle_silence(session, silence_policy.line_is_quiet(result.reason)):
                return
            continue
        session.reset_speculation()

        absolute_end_s = session.processed_until_s + result.utterance_end_s
        session.utt_seq += 1
        utterance_wav = await _slice_utterance(
            session,
            session.processed_until_s,
            absolute_end_s,
            session.utt_seq,
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
    msg = ws_guard.parse_control(raw)  # an object with a known type, or None; an oversized frame raises ws_guard.Rejected
    if msg is None:
        logger.warning("[%s] ignored control frame: %r", session.call_id, str(raw)[:80])
        return
    if msg.get("type") == "playback_done":
        # remember WHERE in the audio the agent finished, so the caller's first words (spoken as it finished) are not skipped by the resync
        session.gate_release_at_s = getattr(getattr(session, "audio", None), "duration_s", None)
        session.release_gate()
    elif msg.get("type") == "hello":
        # The browser may refuse the 16kHz AudioContext we ask for. Trust
        # what the client reports over what we requested: a wrong assumed
        # rate would not fail loudly, it would just make every timestamp
        # and every transcript quietly wrong.
        rate = SAMPLE_RATE
        if msg.get("sampleRate") is not None:
            rate = ws_guard.valid_sample_rate(msg.get("sampleRate"))
            if rate is None:  # 0, 10**9, a string, a boolean: not a rate this agent can resample from
                raise ws_guard.Rejected(ws_guard.CLOSE_BAD_DATA, "unsupported sample rate")
        session.declared_rate = rate
        session.audio.sample_rate = rate
        if rate != SAMPLE_RATE and session.duplex is not None:
            logger.warning("[%s] client at %dHz: echo cancellation needs 16kHz, disabled for this call",
                           session.call_id, rate)
            session.duplex = None
        if rate != SAMPLE_RATE:
            logger.warning("[%s] client capturing at %dHz, not %dHz -- resampling per utterance",
                           session.call_id, rate, SAMPLE_RATE)
        logger.info("[%s] transport: %s @ %dHz", session.call_id,
                    msg.get("format", "pcm_s16le"), rate)
        if msg.get("headphones") and session.duplex is not None:
            # the test page's "I am using headphones" box: no speaker-to-microphone echo, so talking over the agent is accepted without the echo-canceller rule (agent/full_duplex.py)
            session.duplex.enable_echo_free()
            logger.info("[%s] headphones mode: talking over the agent is accepted", session.call_id)
    elif msg.get("type") == "interrupt":
        await _interrupt_playback(session, "manual")


# ---- ingress: the socket reader never waits for the audio path --------------------------------------------------------------------------------------------
# Owner/blueprint 2026-10-29: at 30 calls the receive loop was blocked behind the greeting and then behind per-chunk echo/VAD work, so speech arrived late or was missed. The reader now only
# counts bytes and puts them on a queue; one consumer task per call feeds session.append. A backlog is measured in audio milliseconds; past INGRESS_MAX_S the call is handed to a person
# rather than silently losing speech. A turn is never finalised while frames before its boundary are still queued (see _turn_poll_loop).
INGRESS_MAX_S = float(os.environ.get("INGRESS_MAX_S", "10"))
INGRESS_SETTLE_MS = float(os.environ.get("INGRESS_SETTLE_MS", "60"))
INGRESS_BATCH_MAX = int(os.environ.get("INGRESS_BATCH_MAX", "40"))  # at most this many queued chunks are joined into one append (about 0.8 s of 20 ms frames)
_INGRESS: dict[str, "CallSession"] = {}  # call_id -> session, for the /api/health counters


def _ingress_backlog_ms(session: "CallSession") -> float:
    """Audio received but not yet consumed, in milliseconds. 0 on the browser WebM path (compressed bytes carry no duration)."""
    if not hasattr(session, "audio"):
        return 0.0
    rate = getattr(session, "declared_rate", None) or 16000
    return max(0, session.rx_bytes - session.consumed_bytes) / (2.0 * rate / 1000.0)


def _ingress_snapshot() -> dict:
    sessions = list(_INGRESS.values())
    backlog = [_ingress_backlog_ms(s) for s in sessions]
    return {
        "calls": len(sessions),
        "max_backlog_ms": round(max(backlog), 1) if backlog else 0.0,
        "total_overruns": sum(s.audio_overruns for s in sessions),
        "event_loop_lag_ms_p95": round(sorted(_loop_lag_ms)[int(0.95 * (len(_loop_lag_ms) - 1))], 1) if _loop_lag_ms else 0.0,
        "event_loop_lag_ms_max": round(max(_loop_lag_ms), 1) if _loop_lag_ms else 0.0,
        "rx_bytes": sum(s.rx_bytes for s in sessions),
        "consumed_bytes": sum(s.consumed_bytes for s in sessions),
    }


INGRESS_SHED_MS = float(os.environ.get("INGRESS_SHED_MS", "500"))  # a backlog above this for two seconds closes the door to NEW calls
INGRESS_SHED_LAG_MS = float(os.environ.get("INGRESS_SHED_LAG_MS", "400"))  # so does an event loop that sleeps this much longer than asked
SPECULATE_MAX_CALLS = int(os.environ.get("SPECULATE_MAX_CALLS", "10"))  # above this many live calls the extra "has the caller finished?" transcriptions are not made
_loop_lag_ms: collections.deque = collections.deque(maxlen=60)


def _busy_for_speculation() -> bool:
    return bool(_admission is not None and _admission.active_calls > SPECULATE_MAX_CALLS)


async def _ingress_watch() -> None:
    """Load-based shedding (blueprint 2026-10-29, sections 2 and 4): once a second, look at how far behind the audio path is and how late the event loop wakes. When either stays bad for two
    seconds the door closes to NEW calls (they go to a person at once); calls in progress carry on. It re-opens after five calm seconds."""
    bad = good = 0
    while True:
        t0 = time.monotonic()
        await asyncio.sleep(1.0)
        lag_ms = max(0.0, (time.monotonic() - t0 - 1.0) * 1000.0)
        _loop_lag_ms.append(lag_ms)
        snap = _ingress_snapshot()
        if snap["max_backlog_ms"] > INGRESS_SHED_MS or lag_ms > INGRESS_SHED_LAG_MS:
            bad, good = bad + 1, 0
        else:
            good, bad = good + 1, 0
        if _admission is not None:
            if bad >= 2:
                _admission.set_backend_health("ingress", False, f"backlog {snap['max_backlog_ms']} ms, loop lag {lag_ms:.0f} ms")
            elif good >= 5:
                _admission.set_backend_health("ingress", True)


async def _audio_consumer(session: "CallSession") -> None:
    """Feeds the queued audio to session.append, in order. Its own task, so a slow echo/VAD step delays only this call's processing, never the socket read."""
    while True:
        chunk = await session.audio_q.get()
        # Catching up: whatever has queued meanwhile is processed in ONE call. The echo canceller runs on a worker thread per call of `append`; one hop per 20 ms chunk per call is what a 24-call load
        # could not keep up with (measured on the pod 2026-10-29: a 5.7 s backlog, answers at 12 s). While the audio path is keeping up the queue holds one chunk and nothing changes.
        taken = 1
        while taken < INGRESS_BATCH_MAX and not session.audio_q.empty():
            chunk += session.audio_q.get_nowait()
            taken += 1
        try:
            await session.append(chunk)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - one bad chunk must not end the call
            logger.exception("[%s] audio consumer failed on a chunk", session.call_id)
        finally:
            session.consumed_bytes += len(chunk)


async def _greet(session: "CallSession") -> None:
    """The greeting, spoken in the background so incoming audio is read from the first byte."""
    try:
        await _speak(session, phrase("greeting", "bn"), "bn")
    except asyncio.CancelledError:
        raise
    except Exception:  # noqa: BLE001
        logger.exception("[%s] greeting failed", session.call_id)


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
    await _refresh_messages()
    session.recorder = CallRecorder(session.call_id, _write_call_event)
    _rec(session, "start", "voice", disclosure_version_label())
    session.audio_q = asyncio.Queue()
    _INGRESS[session.call_id] = session
    consumer_task = asyncio.create_task(_audio_consumer(session))
    poll_task = asyncio.create_task(_turn_poll_loop(session))
    greeting_task = None

    try:
        if session.duplex is not None:
            # Tell the client to keep the microphone LIVE during playback and to
            # honour stop_playback: without this it mutes capture as before and
            # the canceller simply sees silence (safe, just no barge-in).
            await session.ws.send_text(json.dumps({"type": "config", "aec": True}))
        greeting_task = asyncio.create_task(_greet(session))  # in the background: the reader below starts at once
        limits = ws_guard.CallLimits(time.monotonic())
        while True:
            message = await ws.receive()
            if message["type"] == "websocket.disconnect":
                break
            if message.get("bytes") is not None:
                limits.check_audio(len(message["bytes"]), time.monotonic())
                session.rx_bytes += len(message["bytes"])
                session.audio_q.put_nowait(message["bytes"])
                if _ingress_backlog_ms(session) > INGRESS_MAX_S * 1000.0:
                    # the audio path cannot keep up: say so and hand the caller to a person, never silently lose speech
                    session.audio_overruns += 1
                    logger.error("[%s] audio_overrun: %.1f s of audio waiting -- handing the call to a person", session.call_id, _ingress_backlog_ms(session) / 1000.0)
                    session.outcome = "audio_overrun"
                    await _handoff_to_human(session, "audio_overrun")
                    break
            elif message.get("text"):
                limits.check_time(time.monotonic())
                await _handle_control(session, message["text"])
    except WebSocketDisconnect:
        pass
    except ws_guard.Rejected as e:
        logger.warning("[%s] call closed by the socket guard: %s", session.call_id, e.reason)
        with contextlib.suppress(Exception):
            await ws.close(code=e.code)
    except Exception:
        logger.exception("[%s] session crashed", session.call_id)
    finally:
        _admission.release(admission)  # the slot is free the moment the caller hangs up, not after the call record is written (that took seconds, and the next burst was refused for "capacity")
        _INGRESS.pop(session.call_id, None)
        for _t in (greeting_task, consumer_task):
            if _t is not None:
                _t.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await _t
        poll_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await poll_task
        if session.recorder is not None:
            # KCD-501: everything completed was already written as it happened; this adds the outcome
            # and drains anything still queued, within the story's thirty seconds. A miss is logged.
            outcome = session.outcome if session.turn_count else "abandoned"
            with contextlib.suppress(Exception):
                _write_satisfaction(session)
            try:
                left = await asyncio.wait_for(session.recorder.finish(outcome), CALL_RECORD_DEADLINE_S + 3.0)
                if left:
                    logger.error("[%s] call record: %d events could not be written", session.call_id, left)
            except Exception as e:  # noqa: BLE001
                logger.error("[%s] call record could not be closed: %s", session.call_id, e)
        with contextlib.suppress(Exception):
            await _queue_unresolved_callback(session)
        await _save_unfinished_draft(session)
        _admission.release(admission)
        session.cleanup()
        logger.info("[%s] call ended (%d active)", session.call_id, _admission.active_calls)


# sourav, 2026-10-21: Twilio phone bridge -- see agent/twilio_bridge.py (registered before the static mount below).
try:
    from agent import twilio_bridge as _twilio_bridge

    _twilio_bridge.install(app, ws_audio)
except Exception:  # never let telephony block the agent from starting
    logger.exception("Twilio bridge not installed")

app.mount("/", StaticFiles(directory="static/pcm", html=True), name="static")
