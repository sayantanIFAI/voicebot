"""Bengali and Hindi speech recognition as its own HTTP service, run as SEVERAL worker processes.

Load test 2026-10-25 (24 simultaneous calls): the recogniser lived inside the one process that also handles every call's websocket, voice-activity checks and replies. Its decoder works a clip
at a time in Python (the "greedy" RNNT strategy agent/asr.py needs for this checkpoint), so 24 callers finishing together waited 12-28 s in one queue while the GPU sat 95% idle. Here the same
TurnASR class (same decoders, same dual CTC/RNNT agreement, same long-clip chunking, plus its own batching) is loaded once per worker process; `uvicorn --workers N` then spreads the callers
over N independent recognisers, each with its own interpreter lock. Same contract as english_asr_server.py: POST /transcribe with a WAV *path* (both processes read /workspace and /tmp)
returns the ASRResult as JSON.

    uvicorn indic_asr_server:app --port 8004 --workers 4
"""

from __future__ import annotations

import dataclasses
import os

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

app = FastAPI()
_engines: dict = {}
_lid = None


class TranscribeRequest(BaseModel):
    wav_path: str
    lang: str = "bn"


@app.on_event("startup")
def _load() -> None:
    from agent.asr import TurnASR

    for lang in [x.strip() for x in os.environ.get("INDIC_ASR_LANGS", "bn,hi").split(",") if x.strip()]:
        _engines[lang] = TurnASR(None, lang)
    global _lid
    if os.environ.get("INDIC_ASR_LID", "1") == "1":  # each worker also holds a language-ID model (see agent/lid.RemoteLID)
        from agent.lid import SpeechBrainVoxLingua107LID

        lid = SpeechBrainVoxLingua107LID(device=os.environ.get("LID_DEVICE", "cuda"))
        lid.load()
        _lid = lid


@app.get("/health")
def health():
    return {"status": "ok", "languages": sorted(_engines), "pid": os.getpid()}


class LidRequest(BaseModel):
    wav_path: str


@app.post("/lid")
async def identify(req: LidRequest):
    if _lid is None:
        return JSONResponse({"error": "language ID not loaded"}, status_code=503)
    import asyncio

    r = await asyncio.to_thread(_lid.identify_path, req.wav_path)
    return {"language": r.language, "confidence": r.confidence, "scores": r.scores}


@app.post("/transcribe")
async def transcribe(req: TranscribeRequest):
    engine = _engines.get(req.lang)
    if engine is None:
        return JSONResponse({"error": f"no engine for {req.lang!r}", "have": sorted(_engines)}, status_code=422)
    result = await engine.transcribe_utterance(req.wav_path)
    return dataclasses.asdict(result)
