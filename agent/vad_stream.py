"""Turn-taking VAD for a *live* WebSocket call.

voice-to-rx-repo/voicerx/vad.py already wraps Silero VAD, but it is built
for an offline consultation recording: load the whole file once, return
every segment. That is the wrong shape here. A live caller needs the
opposite question answered continuously: "has the caller stopped talking
*yet*?" -- asked every ~500ms against a buffer that is still growing.

Rather than switch to Silero's separate low-level streaming API (new
integration surface, new failure modes, unproven on this stack), this
reuses the exact call this codebase already trusts --
`get_speech_timestamps()`, the same function voicerx/vad.py calls -- but
runs it repeatedly against only the *unprocessed tail* of the buffer, the
same fix server.py's `_process_slice()` already had to make:

    "The first version re-ran VAD over the whole growing file each chunk
    ... Silero's boundaries MOVE as more audio arrives, so a segment that
    straddled processed_until_s was excluded on every subsequent pass and
    never processed at all. A 3-minute live recording produced 2 segments."

The caller (main.py) is responsible for passing only the unprocessed tail
slice into `poll()` -- this module treats index 0 of whatever tensor it's
given as "now", and never looks at audio before that.
"""

from __future__ import annotations

import copy
import itertools
import os
import sys
import threading
import time

import torch

from agent.endpointing import EndpointConfig, TurnDecision, decide

_LOCAL_SILERO_REPO = os.environ.get("SILERO_VAD_REPO", "/workspace/silero-vad")

# Back-compat alias: main.py and older tests read `.utterance_end_s` and
# `.had_any_speech`, both still present on TurnDecision.
TurnResult = TurnDecision


class TurnDetector:
    """One instance per process, shared across calls (holds no per-call
    state itself -- main.py's CallSession tracks processed_until_s).

    This class only answers "where is the speech?" (Silero). Whether that
    means the caller has FINISHED is agent/endpointing.decide -- the guards
    (KCD-046), the semantic threshold (KCD-048) and the wake-up schedule
    (KCD-049) all live there, documented and tested without torch."""

    def __init__(
        self,
        silence_confirm_s: float = 1.0,
        tail_guard_s: float = 0.3,
        min_speech_s: float = 0.35,
        max_utterance_s: float = 45.0,
        config: EndpointConfig | None = None,
    ):
        # silence_confirm_s was raised from an initial 0.8s after real
        # testing showed that was cutting callers off mid-sentence. The
        # remaining values and their reasons are documented on EndpointConfig.
        self.config = config or EndpointConfig(
            silence_confirm_s=silence_confirm_s,
            tail_guard_s=tail_guard_s,
            min_speech_s=min_speech_s,
            max_utterance_s=max_utterance_s,
        )

        if os.path.isdir(_LOCAL_SILERO_REPO):
            self.model, utils = torch.hub.load(
                repo_or_dir=_LOCAL_SILERO_REPO,
                source="local",
                model="silero_vad",
                force_reload=False,
                onnx=False,
                trust_repo=True,
            )
        else:
            self.model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                onnx=False,
                trust_repo=True,
            )
        self._get_speech_timestamps = utils[0]
        # Real load test 2026-10-25: 24 calls at once CRASHED the whole agent (segmentation fault). Silero's model is STATEFUL (it resets and carries its recurrent state while it runs), and every
        # call's poll runs `spans` on its own thread against this one shared model. A small POOL of independent copies, each behind its own lock, makes that safe: a poll takes a free copy
        # (or waits for one), never shares one with another thread. VAD_POOL sets the number of copies (the model is ~2 MB).
        self._models = [self.model]
        for _ in range(max(0, int(os.environ.get("VAD_POOL", "6")) - 1)):
            m, _u = torch.hub.load(
                repo_or_dir=_LOCAL_SILERO_REPO if os.path.isdir(_LOCAL_SILERO_REPO) else "snakers4/silero-vad",
                source="local" if os.path.isdir(_LOCAL_SILERO_REPO) else "github",
                model="silero_vad", force_reload=False, onnx=False, trust_repo=True,
            )
            self._models.append(m)
        self._locks = [threading.Lock() for _ in self._models]
        self._next = itertools.count()
        # STREAMING (load test 2026-10-25, profile of 24 simultaneous calls: ~2.7 CPU cores spent in this detector). Every poll used to run Silero over the call's WHOLE unprocessed tail
        # again, 5 times a second. The detector is a recurrent model that reads 32 ms windows one after another, so a call keeps its OWN copy of the model and its list of per-window
        # speech probabilities; a poll then runs only the windows that arrived since the last one, and turns the probabilities into spans with Silero's own
        # `get_speech_timestamps_from_probs` (the same segmentation, so the spans are the same). The state starts afresh whenever the tail starts somewhere new (after a turn is consumed).
        self._from_probs = getattr(sys.modules.get(self._get_speech_timestamps.__module__), "get_speech_timestamps_from_probs", None)
        self._template = None
        self._streams: dict = {}
        self._streams_lock = threading.Lock()
        if self._from_probs is not None:
            self._template, _u = torch.hub.load(
                repo_or_dir=_LOCAL_SILERO_REPO if os.path.isdir(_LOCAL_SILERO_REPO) else "snakers4/silero-vad",
                source="local" if os.path.isdir(_LOCAL_SILERO_REPO) else "github",
                model="silero_vad", force_reload=False, onnx=False, trust_repo=True,
            )

    def release(self, key) -> None:
        """A call is over: forget its streaming state."""
        with self._streams_lock:
            self._streams.pop(key, None)

    def _stream_spans(self, key, wav_tensor: torch.Tensor, start_abs: float) -> list[dict]:
        window = 512
        n_full = wav_tensor.shape[-1] // window
        with self._streams_lock:
            st = self._streams.get(key)
            if st is None or abs(st["start"] - start_abs) > 1e-3 or st["windows"] > n_full:
                model = copy.deepcopy(self._template)
                model.reset_states()
                st = {"start": start_abs, "windows": 0, "probs": [], "model": model, "lock": threading.Lock(), "used": time.monotonic()}
                self._streams[key] = st
                if len(self._streams) > 64:  # forgotten calls
                    for k in [k for k, v in self._streams.items() if time.monotonic() - v["used"] > 300]:
                        self._streams.pop(k, None)
        with st["lock"], torch.no_grad():
            flat = wav_tensor.reshape(-1)
            for w in range(st["windows"], n_full):
                st["probs"].append(st["model"](flat[w * window:(w + 1) * window], 16000).item())
            st["windows"] = n_full
            st["used"] = time.monotonic()
            probs = list(st["probs"])
        return self._from_probs(probs, sampling_rate=16000, return_seconds=True, audio_length_samples=n_full * window)

    def spans(self, wav_tensor: torch.Tensor, sr: int, key=None, start_abs: float | None = None) -> tuple[list[dict], float]:
        """Speech spans (seconds, relative to the tensor) and its duration. With `key` (the call) and `start_abs` (where the tensor starts on the call's timeline) only the audio
        that has arrived since the last poll is run through the model."""
        if key is not None and start_abs is not None and sr == 16000 and self._template is not None:
            duration_s = wav_tensor.shape[-1] / sr
            if duration_s < 0.2:
                return [], duration_s
            return self._stream_spans(key, wav_tensor, float(start_abs)), duration_s
        if sr != 16000:
            wav_tensor = torch.nn.functional.interpolate(
                wav_tensor.view(1, 1, -1),
                scale_factor=16000 / sr,
                mode="linear",
                align_corners=False,
            ).view(-1)
            sr = 16000
        duration_s = wav_tensor.shape[-1] / sr
        if duration_s < 0.2:
            return [], duration_s
        start = next(self._next) % len(self._models)
        for k in range(len(self._models)):  # a free copy if there is one...
            j = (start + k) % len(self._models)
            if self._locks[j].acquire(blocking=False):
                try:
                    return self._get_speech_timestamps(wav_tensor, self._models[j], sampling_rate=sr, return_seconds=True), duration_s
                finally:
                    self._locks[j].release()
        with self._locks[start]:  # ...else wait for one
            return self._get_speech_timestamps(wav_tensor, self._models[start], sampling_rate=sr, return_seconds=True), duration_s

    def poll(
        self, wav_tensor: torch.Tensor, sr: int, completeness: str | None = None, semantic: bool = False
    ) -> TurnResult:
        """wav_tensor: the UNPROCESSED TAIL of the call's buffer only --
        i.e. audio already consumed by a prior completed turn must not be
        included. Index 0 is treated as "now".

        `completeness` is the verdict on a partial transcript of that tail
        (agent/endpointing.classify_completeness), or None if there is none;
        `semantic` turns the shortened/lengthened threshold on."""
        spans, duration_s = self.spans(wav_tensor, sr)
        return decide(spans, duration_s, self.config, completeness=completeness, semantic=semantic)
