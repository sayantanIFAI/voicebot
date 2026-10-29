"""KCD-051/052 glue: one object per call that turns "microphone in, agent audio
sent" into "cleaned microphone out, and a barge-in event when the caller talks
over the agent".

It composes agent/echo_cancel.EchoPipeline and agent/barge_in.BargeInDetector
and adds the two policies neither should own:

  * The agent's speech is placed on the microphone timeline WHERE THE CLIENT
    WILL PLAY IT: clips play back to back, so a clip starts at
    max(now, end of the previous clip). `place_reference()` does that
    arithmetic; the residual (network + output buffering) is what the
    canceller's delay estimator measures.
  * Barge-in is only ACTED ON while the agent is speaking and only once the
    canceller has converged. Before convergence the residual echo is as loud as
    a caller and cannot be told apart, so events are ignored rather than
    guessed; the detector is still fed, so its noise floor keeps tracking.

Nothing here touches a socket or a model: it is fed numpy arrays, so the whole
of it runs and is tested off-pod.
"""

from __future__ import annotations

import dataclasses
import io
import os
import wave

import numpy as np

from agent.barge_in import BargeInConfig, BargeInDetector, BargeInEvent
from agent.echo_cancel import EchoPipeline

# ERLE (dB) the canceller must have reached before a barge-in event is trusted.
# MEASURED on synthetic scenes (12 callers over echo, 24 echo-only calls): with no
# trust rule, 44 false events in 21 of 24 echo-only calls (the residual echo of an
# unconverged filter is as loud as a caller); at >= 0 dB, 6 events in 5 calls;
# at >= 4 dB, none, with every caller still detected; at >= 8 dB one caller in
# twelve is missed. 5 dB keeps a margin above the zero-false-event point. Real
# handsets are untested -- tools/echo_eval.py measures this on recorded audio.
MIN_TRUSTED_ERLE_DB = 5.0
# A telephone leg has no speaker-to-microphone path of its own: the canceller has no echo to remove, so its ERLE stays at 0 dB for the whole call and the 5 dB rule above would discard EVERY
# interruption (caller report 2026-10-27: "even if I talk it does not listen"). On a phone line the rule is therefore dropped and the detector's own tests (level, not echo-shaped, voiced) are
# kept, with more evidence required (PHONE_MIN_SPEECH_S, against 0.11 s). The same idea was tried for BROWSER calls on 2026-10-15 and reverted: a browser leaks faint, non-linear echo that the
# estimator cannot see, and the agent interrupted itself. So this applies to phone-line calls only (for_phone_line), never to the browser.
PHONE_MIN_SPEECH_S = 0.30
# Live test, 2026-10-27, headphones mode: the agent interrupted ITSELF 0.6 s after starting a reply (5 s of silence followed in the log). With no echo found, the detector scores the agent's own
# faint voice in the microphone as a caller ("strong" evidence, because it is voiced and well above the noise floor). The caller speaks into the microphone; the agent's voice only leaks
# into it. So in the no-echo modes a frame must also be LOUD in absolute terms: the leak, tens of dB below the caller, stays under these floors. REASONED, NOT MEASURED: tune on the pod
# with the environment variables if a quiet speaker is ignored (lower it) or the agent still interrupts itself (raise it).
HEADPHONES_FLOOR_DBFS = float(os.environ.get("HEADPHONES_BARGE_FLOOR_DBFS", "-34"))
HEADPHONES_MIN_SPEECH_S = float(os.environ.get("HEADPHONES_BARGE_MIN_SPEECH_S", "0.45"))
PHONE_FLOOR_DBFS = float(os.environ.get("PHONE_BARGE_FLOOR_DBFS", "-40"))
# The agent is treated as still audible this long after the last reference
# sample (playback buffer, room reverberation).
TAIL_S = 0.4
# After the agent's VOICE changes (a different TTS voice: the caller switched language) the
# filter has never seen the newly excited frequencies and cancels poorly for a few seconds,
# and the residual is indistinguishable from a caller. MEASURED on synthetic scenes: every
# remaining false trigger in 200 calls fell within 1.3 s of a voice change, and cancellation
# recovers within about 2 s. Barge-in is not acted on until then; the caller still has the
# manual interrupt.
VOICE_CHANGE_INHIBIT_S = 2.0


@dataclasses.dataclass
class DuplexResult:
    cleaned: np.ndarray
    barge_in: BargeInEvent | None  # only ever set when it is trusted and acted on
    suppressed_barge_in: bool  # an event fired but was ignored (agent silent / not converged)
    agent_speaking: bool


class FullDuplexProcessor:
    def __init__(
        self,
        sr: int = 16000,
        min_trusted_erle_db: float = MIN_TRUSTED_ERLE_DB,
        barge_config: BargeInConfig | None = None,
        assume_echo_free: bool = False,
        **echo_kwargs,
    ):
        self.sr = sr
        self.min_trusted_erle_db = min_trusted_erle_db
        self.assume_echo_free = assume_echo_free  # a phone line: no acoustic echo path, so the ERLE rule is not applied
        self.skip_cancel = False  # AEC_MODE=phone_light on a phone line: the canceller's maths is not run (see for_phone_line)
        self.echo = EchoPipeline(sr=sr, **echo_kwargs)
        self.detector = BargeInDetector(sr=sr, config=barge_config)
        self._mic_pos = 0  # samples of microphone consumed so far
        self._ref_end = 0  # end of the last agent clip on the mic timeline
        self._voice: str | None = None
        self._inhibit_until = 0  # mic sample before which barge-in is not acted on
        self.barge_ins = 0
        self.suppressed = 0

    @classmethod
    def for_phone_line(cls, sr: int = 16000) -> "FullDuplexProcessor":
        """The processor for a telephone call (Twilio): barge-in is acted on without the ERLE rule, but only on PHONE_MIN_SPEECH_S of net voiced evidence."""
        p = cls(sr=sr, barge_config=BargeInConfig(min_speech_s=PHONE_MIN_SPEECH_S, absolute_floor_dbfs=PHONE_FLOOR_DBFS), assume_echo_free=True)
        # Owner, 2026-10-29: the echo canceller has TWO settings, chosen by the environment variable AEC_MODE (deploy/env.sh):
        #   full        (default, for browser testing until production): the canceller runs on every call, phone or browser.
        #   phone_light (for production): a PHONE-LINE call skips the canceller's maths -- the SIP leg has no acoustic echo, the canceller's ERLE stays at 0 dB there (it removes nothing) and its
        #               maths was about half of the agent's CPU at 24 calls (load test 2026-10-29). Barge-in detection stays; browser calls are not affected.
        p.skip_cancel = os.environ.get("AEC_MODE", "full").strip().lower() == "phone_light"
        return p

    def enable_echo_free(self, min_speech_s: float = HEADPHONES_MIN_SPEECH_S, floor_dbfs: float = HEADPHONES_FLOOR_DBFS) -> None:
        """The caller says the line has no echo (headphones in the browser test page, owner 2026-10-27): the same rule as a phone line, switched on for this call only. Never inferred by the
        agent -- the 2026-10-15 attempt to infer it from the microphone made the agent interrupt itself."""
        self.assume_echo_free = True
        self.detector = BargeInDetector(sr=self.sr, config=BargeInConfig(min_speech_s=min_speech_s, absolute_floor_dbfs=floor_dbfs))

    # -- agent side ---------------------------------------------------
    def place_reference(self, pcm: np.ndarray, voice: str | None = None) -> int:
        """Register a clip the agent is about to send. Returns where on the
        microphone timeline it is expected to start playing. `voice` identifies the
        TTS voice (e.g. the language): when it changes, barge-in is held off for
        VOICE_CHANGE_INHIBIT_S from the moment the new voice starts."""
        start = max(self._mic_pos, self._ref_end)
        if voice is not None:
            if self._voice is not None and voice != self._voice:
                self._inhibit_until = max(self._inhibit_until, start + int(VOICE_CHANGE_INHIBIT_S * self.sr))
            self._voice = voice
        self.echo.feed_reference(np.asarray(pcm, np.float32), start)
        self._ref_end = start + len(pcm)
        return start

    def agent_speaking(self) -> bool:
        return self._mic_pos < self._ref_end + int(TAIL_S * self.sr)

    def cancel_pending(self) -> None:
        """The caller interrupted: whatever was queued after now will not be
        played, so it must not be expected as echo either."""
        self._ref_end = min(self._ref_end, self._mic_pos)
        self.detector.reset()

    # -- caller side --------------------------------------------------
    def process(self, mic: np.ndarray) -> DuplexResult:
        speaking = self.agent_speaking()
        # Agent silent (and past the echo tail): there is no echo to cancel and a barge-in cannot be acted on -- the caller's audio goes straight through.
        cleaned, echo = self.echo.process_ex(mic) if (speaking and not self.skip_cancel) else self.echo.passthrough(mic)
        self._mic_pos += len(mic)
        speaking = speaking or self.agent_speaking()
        event = self.detector.process(cleaned, echo) if (cleaned.size and speaking) else None
        acted = None
        suppressed = False
        if event is not None:
            trusted = (
                speaking
                and (self.assume_echo_free or self.echo.canceller.erle_db >= self.min_trusted_erle_db)
                and self._mic_pos >= self._inhibit_until
            )
            if trusted:
                acted = event
                self.barge_ins += 1
            else:
                suppressed = True
                self.suppressed += 1
        return DuplexResult(cleaned, acted, suppressed, speaking)


def wav_to_pcm16k(wav_bytes: bytes, sr_out: int = 16000) -> np.ndarray:
    """Decode a WAV (the TTS output, typically 22.05 kHz) to float32 mono at the
    microphone's rate, for use as the echo canceller's far-end reference. FFT
    resampling: exact band-limiting, and clips are short. The reference only
    needs the spectral shape of what the caller will hear, not audiophile
    fidelity, and errors here are absorbed by the adaptive filter."""
    with wave.open(io.BytesIO(wav_bytes), "rb") as w:
        sr, ch, width, frames = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.readframes(w.getnframes())
    if width != 2:
        raise ValueError(f"unsupported sample width {width}")
    x = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    if sr == sr_out or x.size == 0:
        return x
    n_out = int(round(x.size * sr_out / sr))
    spec = np.fft.rfft(x)
    out_bins = n_out // 2 + 1
    resized = np.zeros(out_bins, dtype=np.complex128)
    k = min(out_bins, spec.size)
    resized[:k] = spec[:k]
    return (np.fft.irfft(resized, n=n_out) * (n_out / x.size)).astype(np.float32)
