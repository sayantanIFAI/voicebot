"""How the caller SOUNDS (angry / sad), read from the audio, run beside the recogniser (owner, 2026-10-27).

Why: a caller who is upset or frightened should hear a calm, short, one-question-at-a-time agent that says it understands first -- that is Appendix C's "distressed" row (agent/speech_policy.py) and
its fixed acknowledgement (agent/acknowledgement.py). Until now nothing could put a call into that state from the VOICE; only words could (agent/emergency.py for a medical emergency, which is
unchanged and still the only thing that can start the emergency notice).

WHERE IT RUNS: IN PARALLEL with speech recognition, never before it. main.py starts the classification the moment the utterance is cut, lets recognition and language ID run alongside, and
takes the verdict at the end under a time budget. A slow or failed classifier just means "no verdict this turn"; it can never delay or fail a reply.

WHAT IT FEEDS: the call state ("distressed"), which changes HOW the agent speaks -- slower, shorter, one question, and the approved acknowledgement before the answer. It does NOT feed the language
model, and the agent's words are not generated from the emotion: replies stay fixed templates (CLAUDE.md truth boundary), so a model can never invent comfort, a diagnosis or a promise.

THE MODEL: SpeechBrain `emotion-recognition-wav2vec2-IEMOCAP` (wav2vec2-base, 4 classes: neutral, angry, happy, sad). LICENCE CAVEAT, owner to decide: the model card says Apache-2.0, but it was
trained on IEMOCAP, whose release form says the data is not shared for commercial purposes; whether the trained weights may be used in a commercial product is NOT settled by the card.
`EMOTION_DETECT=off` switches the whole feature off, and `Classifier.MODEL_SOURCE` is the one line to change for a model with a verified commercial licence.

LIMITS, REASONED NOT MEASURED: it was trained on acted English speech. It has not been checked on Bengali or Hindi phone audio, and it has no "fear" class (fear tends to read as sad or angry).
So it is deliberately conservative: two stressed clips out of the last three are needed (one clip never decides, however sure the model sounds -- it is often near 100% sure), and it switches off again
after three calm ones. A native-speaker listening pass is the way to tune the numbers below.
"""

from __future__ import annotations

import math
import threading
from collections import deque
from dataclasses import dataclass, field

STRESSED_LABELS = frozenset({"ang", "sad"})
MIN_CLIP_S = 1.2  # a one-word "yes" says nothing about mood
MIN_SCORE = 0.75  # probability the top class needs before the clip counts as stressed
WINDOW = 3  # stressed clips are counted over the last this-many scored clips
NEEDED = 2
CALM_TO_CLEAR = 3  # this many calm clips in a row end the state


@dataclass(frozen=True)
class EmotionResult:
    label: str  # neu / ang / hap / sad, or "skip" when the clip was too short to judge
    score: float
    scores: dict[str, float] = field(default_factory=dict)


SKIPPED = EmotionResult("skip", 0.0)


class EmotionEvidence:
    """Turns per-clip verdicts into an on/off decision for the call. add() returns "on", "off" or None (no change)."""

    def __init__(self) -> None:
        self._recent: deque[bool] = deque(maxlen=WINDOW)
        self._calm_run = 0
        self.active = False

    def add(self, result: EmotionResult | None) -> str | None:
        if result is None or result.label == "skip":
            return None
        stressed = result.label in STRESSED_LABELS and result.score >= MIN_SCORE
        self._recent.append(stressed)
        self._calm_run = 0 if stressed else self._calm_run + 1
        if not self.active:
            if sum(self._recent) >= NEEDED:
                self.active = True
                return "on"
        elif self._calm_run >= CALM_TO_CLEAR:
            self.active = False
            self._recent.clear()
            return "off"
        return None


class Classifier:
    """The model itself. Loaded once per recogniser worker process (indic_asr_server.py, POST /emotion); main.py talks to it through RemoteEmotion."""

    MODEL_SOURCE = "speechbrain/emotion-recognition-wav2vec2-IEMOCAP"

    def __init__(self, device: str = "cpu", savedir: str | None = None):
        self.device = device
        self.savedir = savedir
        self._model = None
        self._lock = threading.Lock()

    def load(self) -> None:
        import os

        from speechbrain.inference.interfaces import foreign_class

        # The model's own recipe keeps its wav2vec2 base weights in "wav2vec2_checkpoints" RELATIVE to the working directory; loaded from the savedir, they land (and are found) beside the rest of the model.
        here = os.getcwd()
        if self.savedir:
            os.makedirs(self.savedir, exist_ok=True)
            os.chdir(self.savedir)
        try:
            self._model = foreign_class(
                source=self.MODEL_SOURCE,
                pymodule_file="custom_interface.py",
                classname="CustomEncoderWav2vec2Classifier",
                savedir=self.savedir or ".",
                run_opts={"device": self.device},
            )
        finally:
            os.chdir(here)
        self._labels = {int(i): str(lab) for lab, i in self._model.hparams.label_encoder.lab2ind.items()}

    def classify_path(self, wav_path: str) -> EmotionResult:
        import soundfile as sf
        import torch

        if self._model is None:
            raise RuntimeError("the emotion model is not loaded")
        data, rate = sf.read(wav_path, dtype="float32", always_2d=True)
        if data.shape[0] / float(rate) < MIN_CLIP_S:
            return SKIPPED
        wav = torch.from_numpy(data.mean(axis=1))
        if rate != 16000:
            import torchaudio

            wav = torchaudio.functional.resample(wav, rate, 16000)
        with self._lock, torch.inference_mode():
            out_prob, _score, _index, _text = self._model.classify_batch(wav.unsqueeze(0).to(self.device))
        return result_from_output(out_prob[0].detach().float().cpu().tolist(), self._labels)


def result_from_output(values: list[float], labels: dict[int, str]) -> EmotionResult:
    """The model's output row -> probabilities. Log-probabilities (all <= 0) are exponentiated, a row that already sums to 1 is kept, anything else is read as raw scores and softmaxed."""
    if all(v <= 0.0 for v in values):
        probs = [math.exp(v) for v in values]
    elif all(0.0 <= v <= 1.0 for v in values) and abs(sum(values) - 1.0) < 1e-3:
        probs = list(values)
    else:
        top = max(values)
        probs = [math.exp(v - top) for v in values]
    total = sum(probs) or 1.0
    scores = {labels.get(i, str(i)): p / total for i, p in enumerate(probs)}
    label = max(scores, key=scores.get)
    return EmotionResult(label, scores[label], scores)


class RemoteEmotion:
    """The same classifier served by the indic recogniser service's worker processes (indic_asr_server.py, POST /emotion), so the call process holds no extra model and the clips are judged side by side."""

    def __init__(self, base_url: str, timeout_s: float = 4.0):
        import httpx

        self._client = httpx.Client(
            base_url=base_url.rstrip("/"), timeout=timeout_s, limits=httpx.Limits(max_connections=64, max_keepalive_connections=32)
        )

    def classify_path(self, wav_path: str) -> EmotionResult:
        r = self._client.post("/emotion", json={"wav_path": wav_path})
        r.raise_for_status()
        d = r.json()
        return EmotionResult(str(d["label"]), float(d["score"]), {k: float(v) for k, v in (d.get("scores") or {}).items()})
