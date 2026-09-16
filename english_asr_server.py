"""English ASR as its own HTTP service, in its own Python environment.

Pilot single-L4 architecture (docs/adr/0001-pilot-single-l4-architecture.md):
Bengali and Hindi share the AI4Bharat NeMo fork and load in-process inside
the main orchestrator (agent/asr.py). English needs mainline NVIDIA NeMo
instead -- nvidia/stt_en_fastconformer_hybrid_large_streaming_multi -- and
the two NeMo distributions cannot coexist in one Python environment (see
agent/asr.py's module docstring for exactly why the AI4Bharat fork exists
at all: mainline NeMo cannot even load the IndicConformer checkpoint).
So English ASR runs as this separate process, under
/workspace/venv-en-nemo, reached over HTTP by agent/asr_router.py's
English engine adapter (ENGLISH_ASR_URL in deploy/env.sh).

Same contract as agent/asr.py's TurnASR, deliberately: POST /transcribe
with a WAV file path (NOT the audio itself -- both processes share
/workspace, so passing a path avoids a redundant multi-megabyte copy over
localhost for every single turn) returns the same ASRResult shape as JSON,
so agent/asr_router.py's ASREngine Protocol is satisfied identically
whether the underlying engine is in-process (bn/hi) or HTTP (en).

UNVERIFIED beyond a basic load test at the time this was written: this
mirrors agent/asr.py's proven CTC/RNNT dual-decode pattern for the English
FastConformer checkpoint, but that pattern was proven against the
AI4Bharat Bengali checkpoint specifically. FastConformer's hybrid
CTC/Transducer head is architecturally similar but NOT the same codebase
as IndicConformer; confirm decoding strategy defaults (the same
"greedy_batch silently returns empty text" class of bug agent/asr.py
already hit once) against this exact checkpoint before trusting it in the
call path -- see docs/adr/0001 section 8 for what "verified" means here.
"""
from __future__ import annotations

import asyncio
import dataclasses
import os

import torch
import nemo.collections.asr as nemo_asr
from fastapi import FastAPI
from pydantic import BaseModel

NEMO_FILE = os.environ.get(
    "VOICE_AGENT_NEMO_FILE_EN",
    "/workspace/.cache/huggingface/hub/fastconformer_en/snap/"
    "stt_en_fastconformer_hybrid_large_streaming_multi.nemo",
)

app = FastAPI()
_model = None
_device = None


@app.on_event("startup")
def _load_model():
    global _model, _device
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _model = nemo_asr.models.ASRModel.restore_from(restore_path=NEMO_FILE)
    _model = _model.to(_device)
    _model.freeze()
    # Same proven fix as agent/asr.py's TurnASR -- do not assume the
    # library default decoding strategy is safe without checking it
    # against this checkpoint (see module docstring).
    if hasattr(_model, "change_decoding_strategy"):
        try:
            from nemo.collections.asr.parts.submodules.rnnt_decoding import RNNTDecodingConfig
            from omegaconf import OmegaConf
            rnnt_cfg = OmegaConf.structured(RNNTDecodingConfig(strategy="greedy"))
            _model.change_decoding_strategy(rnnt_cfg, decoder_type="rnnt")
        except Exception:  # noqa: BLE001 - best-effort; log and continue on CTC-only models
            pass


def _first_text(texts) -> str:
    if not texts:
        return ""
    item = texts[0]
    while isinstance(item, (list, tuple)):
        if not item:
            return ""
        item = item[0]
    return (item or "").strip()


@dataclasses.dataclass
class ASRResult:
    text: str
    decoder_used: str
    decoder_agreement: float = 1.0


def _transcribe_sync(wav_path: str) -> ASRResult:
    texts = _model.transcribe([wav_path], batch_size=1)
    text = _first_text(texts)
    return ASRResult(text=text, decoder_used="fastconformer", decoder_agreement=1.0)


class TranscribeRequest(BaseModel):
    wav_path: str


@app.get("/health")
def health():
    return {"status": "ok" if _model is not None else "loading", "device": str(_device)}


@app.post("/transcribe")
async def transcribe(req: TranscribeRequest):
    result = await asyncio.to_thread(_transcribe_sync, req.wav_path)
    return dataclasses.asdict(result)
