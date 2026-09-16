"""AI4Bharat TTS -- FastAPI wrapper around coqui-tts's Synthesizer.

Matches the contract agent/tts.py's TTSClient expects: POST /synthesize
with {"text": ..., "lang": "bn"} -> raw WAV bytes.

Pilot single-L4 architecture (docs/adr/0001-pilot-single-l4-architecture.md):
one process serves all THREE languages -- bn, hi, en -- because AI4Bharat's
own Indic-TTS v1-checkpoints-release ships an English FastPitch+HiFi-GAN
checkpoint in the exact same coqui-tts format as Bengali and Hindi
(https://github.com/AI4Bharat/Indic-TTS/releases/tag/v1-checkpoints-release,
en.zip). That is simpler than the ADR's original assumption of a second,
NVIDIA-NeMo-based English TTS path, and this file supersedes that
assumption -- no second TTS MODEL FAMILY is needed.

It still needs its own PROCESS and VENV (/workspace/tts_venv), separate
from the orchestrator/ASR venv -- confirmed the hard way, not assumed:
coqui-tts needs transformers>=4.57 + huggingface_hub>=0.34, while the
AI4Bharat NeMo fork used for Bengali/Hindi ASR imports huggingface_hub's
ModelFilter, removed in huggingface_hub 0.24. No single version of either
package satisfies both consumers at once, so this cannot share an
environment with agent/asr.py's bn/hi models no matter how the two
libraries' own version constraints are pinned. See deploy/start_all.sh's
_run_tts.sh for the resulting process layout.

All three Synthesizers are loaded ONCE at module import (not per-request)
-- model load takes several seconds and holds real GPU memory, so this
must be a long-lived process,
not spawned per call.

WHY THIS DOES MORE THAN CALL synthesizer.tts()
----------------------------------------------
A single FastPitch pass over a whole multi-clause reply is what makes this
voice sound like a machine, and it is fixable without changing models:

* No breaths. FastPitch renders one flat prosodic contour across an entire
  utterance. Real speakers stop between clauses. Splitting on Bengali
  sentence and clause boundaries and inserting real silence is the single
  largest naturalness gain available here.
* Ragged padding. Each pass emits its own leading/trailing near-silence of
  arbitrary length, so naive concatenation produces gaps that are too long
  in some places and absent in others. Each chunk is trimmed, then padded
  by an amount chosen for the punctuation that ended it.
* Rushed delivery. length_scale 1.0 is noticeably fast for a service line
  a caller is trying to write a price down from. Slightly above 1 reads as
  measured rather than sluggish.
* Inconsistent level. Peak varies per utterance, which on a phone sounds
  like the speaker keeps moving. Normalizing to a fixed peak fixes it.

Every one of these is tunable per-request (see SynthesizeRequest) so the
settings can be A/B'd against a real handset without a redeploy.
"""
import io
import os
import re

# The AI4Bharat checkpoint's speaker manager resolves a RELATIVE path
# ("models/v1/bn/fastpitch/speakers.pth") baked in at save time, against
# whatever the process's CWD happens to be -- not the checkpoint's own
# location. Pin CWD explicitly so this doesn't depend on how/where this
# script gets launched from.
CHECKPOINTS_ROOT = os.environ.get("TTS_CHECKPOINTS_ROOT", "/workspace/tts_checkpoints")
os.chdir(CHECKPOINTS_ROOT)

import numpy as np
import soundfile as sf
from fastapi import FastAPI
from fastapi.responses import Response
from pydantic import BaseModel
from TTS.utils.synthesizer import Synthesizer

app = FastAPI()

SUPPORTED_LANGUAGES = ("bn", "hi", "en")


def _load_synthesizer(lang: str) -> Synthesizer:
    ckpt = f"{CHECKPOINTS_ROOT}/{lang}"
    return Synthesizer(
        tts_checkpoint=f"{ckpt}/fastpitch/best_model.pth",
        tts_config_path=f"{ckpt}/fastpitch/config.json",
        tts_speakers_file=f"{ckpt}/fastpitch/speakers.pth",
        vocoder_checkpoint=f"{ckpt}/hifigan/best_model.pth",
        vocoder_config=f"{ckpt}/hifigan/config.json",
        use_cuda=True,
    )


# All three loaded and resident on the GPU together -- small footprint,
# and never unloaded/reloaded on a language switch (docs/adr/0001 section
# on why all six speech models stay warm: a model-load stall mid-call is
# worse than the fixed VRAM cost of keeping them all in memory).
SYNTHESIZERS: dict[str, Synthesizer] = {lang: _load_synthesizer(lang) for lang in SUPPORTED_LANGUAGES}

SAMPLE_RATES: dict[str, int] = {
    lang: (synth.output_sample_rate or 22050) for lang, synth in SYNTHESIZERS.items()
}
DEFAULT_SPEAKER = os.environ.get("TTS_SPEAKER", "female")

# >1 slows delivery. 1.08 measured as the point where the line stops
# sounding hurried without starting to drag -- tune via /synthesize's
# `speed` override before changing this default.
DEFAULT_LENGTH_SCALE = float(os.environ.get("TTS_LENGTH_SCALE", "1.08"))

# Silence inserted AFTER a chunk, by the punctuation that ended it.
PAUSE_S = {"sentence": 0.28, "clause": 0.14, "none": 0.06}

TARGET_PEAK = 0.89          # ~-1 dBFS; loud without clipping on phone speakers
TRIM_THRESHOLD = 0.012      # below this is padding, not speech

# Bengali sentence enders (danda + Latin punctuation, since clinic data
# mixes both) and clause separators.
_RE_SENTENCE = re.compile(r"([^।?!\n]+[।?!\n]?)")
_RE_CLAUSE = re.compile(r"([^,;:]+[,;:]?)")

MAX_CHUNK_CHARS = 90        # long single clauses still get a breath


def _split_for_prosody(text: str) -> list[tuple[str, str]]:
    """-> [(chunk_text, pause_kind)]. Sentences first, then clauses inside
    any sentence long enough that a listener would expect a breath."""
    out: list[tuple[str, str]] = []
    for raw_sentence in _RE_SENTENCE.findall(text):
        sentence = raw_sentence.strip()
        if not sentence:
            continue
        if len(sentence) <= MAX_CHUNK_CHARS:
            out.append((sentence, "sentence"))
            continue
        clauses = [c.strip() for c in _RE_CLAUSE.findall(sentence) if c.strip()]
        for i, clause in enumerate(clauses):
            out.append((clause, "sentence" if i == len(clauses) - 1 else "clause"))
    if not out:
        out = [(text.strip(), "sentence")]
    return out


def _trim_silence(wav: np.ndarray) -> np.ndarray:
    loud = np.where(np.abs(wav) > TRIM_THRESHOLD)[0]
    if loud.size == 0:
        return wav[:0]
    return wav[loud[0]:loud[-1] + 1]


def _normalize_peak(wav: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(wav))) if wav.size else 0.0
    return wav * (TARGET_PEAK / peak) if peak > 1e-6 else wav


def _render(lang: str, text: str, speaker: str, speed: float, pauses: bool) -> tuple[np.ndarray, int]:
    synth = SYNTHESIZERS[lang]
    sample_rate = SAMPLE_RATES[lang]
    if hasattr(synth.tts_model, "length_scale"):
        synth.tts_model.length_scale = speed

    chunks = _split_for_prosody(text) if pauses else [(text, "sentence")]
    pieces: list[np.ndarray] = []

    for chunk_text, pause_kind in chunks:
        # split_sentences=False: this module already decided the chunking,
        # and letting Coqui re-split would reintroduce the ragged joins.
        raw = synth.tts(chunk_text, speaker_name=speaker, split_sentences=False)
        wav = _trim_silence(np.asarray(raw, dtype=np.float32))
        if wav.size == 0:
            continue
        pieces.append(wav)
        if pauses:
            pieces.append(np.zeros(int(PAUSE_S[pause_kind] * sample_rate), dtype=np.float32))

    if not pieces:
        return np.zeros(int(0.2 * sample_rate), dtype=np.float32), sample_rate

    joined = np.concatenate(pieces)
    # A short lead-in stops the very first phoneme being clipped by
    # playback devices that ramp up on stream start.
    return _normalize_peak(
        np.concatenate([np.zeros(int(0.04 * sample_rate), dtype=np.float32), joined]),
    ), sample_rate


class SynthesizeRequest(BaseModel):
    text: str
    lang: str = "bn"
    speaker: str | None = None
    speed: float | None = None      # length_scale; >1 slower
    pauses: bool = True


class UnsupportedLanguage(Exception):
    pass


@app.get("/health")
def health():
    return {
        "status": "ok",
        "languages": list(SYNTHESIZERS.keys()),
        "speaker": DEFAULT_SPEAKER,
        "sample_rates": SAMPLE_RATES,
        "length_scale": DEFAULT_LENGTH_SCALE,
    }


@app.post("/synthesize")
def synthesize(req: SynthesizeRequest):
    if req.lang not in SYNTHESIZERS:
        # Fail loudly and specifically rather than silently falling back
        # to Bengali -- a caller who switched to Hindi does not want an
        # answer read out in the wrong language because the router or the
        # caller passed an unrecognised code.
        return Response(
            content=f'{{"error":"unsupported lang {req.lang!r}, have {list(SYNTHESIZERS)}"}}',
            media_type="application/json", status_code=422,
        )
    wav, sample_rate = _render(
        req.lang, req.text,
        req.speaker or DEFAULT_SPEAKER,
        req.speed or DEFAULT_LENGTH_SCALE,
        req.pauses,
    )
    buf = io.BytesIO()
    sf.write(buf, wav, sample_rate, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return Response(content=buf.read(), media_type="audio/wav")
