"""Twilio phone-call transport for the PCM voice agent.

sourav, 2026-10-21: WHY THIS FILE EXISTS
----------------------------------------
Until now the agent could only be reached from a browser page (/ws/audio, raw 16 kHz int16 PCM frames in, one WAV
per reply out). The owner wants to ring it from an ordinary mobile phone:

    caller's phone --PSTN--> Twilio number --webhook--> POST /twilio/voice  (this file: returns <Connect><Stream>)
                                           --wss------> /twilio/stream      (this file: Media Streams WebSocket)
                                                              |
                                                     the SAME ws_audio() turn loop the browser uses

THE DESIGN CHOICE THAT KEEPS EXISTING BEHAVIOUR SAFE
----------------------------------------------------
Nothing in the turn logic (VAD, ASR, intent, booking, TTS, half-duplex gate, barge-in) is copied or changed. Instead
`TwilioStreamAdapter` makes a Twilio stream LOOK LIKE the browser WebSocket that ws_audio() already expects:

    ws_audio calls                      the adapter does
    -----------------------------       -------------------------------------------------------------------------
    await ws.accept()                   waits for Twilio's "start" event, checks the signed stream token
    await ws.receive()  (bytes)         Twilio "media" (8 kHz mu-law) -> 16 kHz int16 PCM, exactly what the browser sent
    await ws.receive()  (text)          Twilio "mark" echo -> {"type":"playback_done"}, the message the browser sent
    await ws.send_bytes(wav)            WAV -> 8 kHz mu-law -> Twilio "media" frames, then a "mark" to learn when played
    await ws.send_text('{"type":"stop_playback"}')   Twilio "clear" (drops queued audio: barge-in)
    await ws.close()                    lets queued audio finish (bounded), then closes the stream (call ends)

SECURITY (the endpoint is public, so it is not left open)
---------------------------------------------------------
* POST /twilio/voice checks Twilio's X-Twilio-Signature (HMAC-SHA1 of the public URL + sorted form params, keyed with
  the account Auth Token). A forged request gets 403 before anything is returned.
* The stream URL carries a short-lived token (HMAC of the CallSid) that /twilio/stream re-checks, so nobody can open a
  media stream by guessing the URL.
* TWILIO_ALLOWED_CALLERS (optional, comma-separated numbers) restricts who may reach the agent at all; empty = anyone.
* The Auth Token is read from the environment or a .env file and is NEVER logged.

CONFIGURATION (all optional except the token)
---------------------------------------------
  TWILIO_AUTH_TOKEN (or AuthToken / TWILIO_TOKEN)   the account Auth Token
  TWILIO_PUBLIC_BASE_URL     e.g. https://<POD_ID>-8888.proxy.runpod.net -- the URL Twilio calls. If unset it is derived
                             from X-Forwarded-Proto / X-Forwarded-Host / Host (the gateway forwards them).
  TWILIO_VALIDATE_SIGNATURE  "0" to switch the webhook signature check off (local testing ONLY)
  TWILIO_ALLOWED_CALLERS     "+919999999999,+918888888888"
  TWILIO_BRIDGE              "0" to not register the routes at all
"""

import asyncio
import base64
import collections
import contextlib
import hashlib
import hmac
import io
import json
import logging
import os
import pathlib
import re
import time
import wave
from urllib.parse import parse_qsl
from xml.sax.saxutils import quoteattr

import numpy as np

try:  # fastapi is always present in the running app; the guard only keeps the pure codec importable on its own
    from fastapi import Request, Response, WebSocket
    from starlette.websockets import WebSocketDisconnect
except ImportError:  # pragma: no cover
    Request = Response = WebSocket = None  # type: ignore[assignment,misc]

    class WebSocketDisconnect(Exception):  # type: ignore[no-redef]
        pass


logger = logging.getLogger("twilio_bridge")

TWILIO_RATE = 8000  # Twilio Media Streams are always 8 kHz mu-law
AGENT_RATE = 16000  # the PCM agent (agent/pcm_buffer.py SAMPLE_RATE) listens at 16 kHz int16
OUT_CHUNK_BYTES = 800  # 100 ms of 8 kHz mu-law per outgoing "media" message (Twilio accepts any size and buffers in order)
START_TIMEOUT_S = 10.0  # how long a new stream may take to send its "start" event
TOKEN_MAX_AGE_S = 120  # a stream token is only good for this long after the webhook issued it
DRAIN_CLOSE_S = 2.0  # close() waits at most this long for queued audio to finish playing
_CONVERT_CACHE_MAX = 128  # converted reply clips kept (pinned catalogue answers repeat constantly)

_TOKEN_ENV_NAMES = ("TWILIO_AUTH_TOKEN", "AuthToken", "TWILIO_TOKEN")


# =============================================================================================== audio codec (numpy only)

_MULAW_BIAS = 0x84
_MULAW_CLIP = 32635


def _build_mulaw_decode_table() -> np.ndarray:
    """sourav, 2026-10-21: G.711 mu-law byte -> int16 sample lookup (256 entries), built once at import so decoding a
    20 ms Twilio frame is one numpy index. Flow: invert the byte, split sign/exponent/mantissa, rebuild the magnitude."""
    table = np.empty(256, dtype=np.int16)
    for i in range(256):
        u = ~i & 0xFF
        exponent = (u >> 4) & 0x07
        mantissa = u & 0x0F
        sample = (((mantissa << 3) + _MULAW_BIAS) << exponent) - _MULAW_BIAS
        table[i] = -sample if (u & 0x80) else sample
    return table


_MULAW_DECODE = _build_mulaw_decode_table()


def mulaw_decode(data: bytes) -> np.ndarray:
    """sourav, 2026-10-21: Twilio audio in. 8 kHz mu-law bytes -> int16 samples. Flow: bytes -> table lookup."""
    return _MULAW_DECODE[np.frombuffer(data, dtype=np.uint8)]


def mulaw_encode(pcm: np.ndarray) -> bytes:
    """sourav, 2026-10-21: Twilio audio out. int16 samples -> G.711 mu-law bytes, vectorised (a 5 s reply is 40,000
    samples). Flow: take sign, clip + add the G.711 bias, find the segment (exponent) from the highest set bit, keep
    4 mantissa bits, then invert the byte as the standard requires."""
    x = np.asarray(pcm, dtype=np.int32)
    sign = np.where(x < 0, 0x80, 0)
    x = np.minimum(np.abs(x), _MULAW_CLIP) + _MULAW_BIAS
    exponent = np.floor(np.log2(x)).astype(np.int32) - 7
    mantissa = (x >> (exponent + 3)) & 0x0F
    return ((~(sign | (exponent << 4) | mantissa)) & 0xFF).astype(np.uint8).tobytes()


class Upsampler8to16:
    """sourav, 2026-10-21; owner, 2026-10-26: why: the agent's buffer and ASR expect 16 kHz but a phone line is 8 kHz.
    Flow: zero-stuff each 8 kHz sample with a following zero, then run the same Blackman-windowed-sinc low-pass this
    file already uses outbound (resample_to_8k, below) to remove the spectral image the zero-stuffing creates --
    the standard way to interpolate, and consistent with how this file treats the other direction. The filter's
    state (its last ntaps-1 input-rate samples) is carried across 20 ms frame boundaries exactly as the previous
    single-sample carry was, so there is still no discontinuity at a frame edge. Cutoff matches resample_to_8k's own
    3600 Hz (the line is band-limited under 4 kHz anyway; taps are scaled by the upsample factor, 2, since
    zero-stuffing halves the signal's DC energy that an ordinary unity-gain filter would otherwise lose)."""

    _NTAPS = 31  # group delay (NTAPS-1)/2 = 15 samples @ 16 kHz =~ 0.9 ms: a one-time constant, not per-frame, cost

    def __init__(self) -> None:
        self._taps = _lowpass_taps(3600.0, 16000, self._NTAPS) * 2.0
        self._carry = np.zeros(self._NTAPS - 1, dtype=np.float32)

    def process(self, samples: np.ndarray) -> np.ndarray:
        if samples.size == 0:
            return np.empty(0, dtype=np.int16)
        zero_stuffed = np.zeros(2 * samples.size, dtype=np.float32)
        zero_stuffed[0::2] = samples.astype(np.float32)
        padded = np.concatenate((self._carry, zero_stuffed))
        out = np.convolve(padded, self._taps, mode="valid")
        self._carry = padded[-(self._NTAPS - 1):]
        return np.clip(np.rint(out), -32768, 32767).astype(np.int16)


def _lowpass_taps(cutoff_hz: float, rate: int, ntaps: int = 127) -> np.ndarray:
    """sourav, 2026-10-21: a Blackman-windowed sinc low-pass (unity gain at DC), used before dropping to 8 kHz so
    energy above 4 kHz does not fold back into the speech band as hiss."""
    fc = cutoff_hz / float(rate)
    n = np.arange(ntaps) - (ntaps - 1) / 2.0
    h = 2.0 * fc * np.sinc(2.0 * fc * n) * np.blackman(ntaps)
    return (h / h.sum()).astype(np.float32)


def _sinc_interp(x: np.ndarray, positions: np.ndarray, half_width: int = 8) -> np.ndarray:
    """owner, 2026-10-26: band-limited (windowed-sinc) interpolation of `x` at fractional sample `positions`,
    Blackman-windowed over +/- half_width input samples either side of each position -- the standard way to
    resample at an arbitrary (non-integer) ratio, replacing plain straight-line interpolation's smearing of the
    high end of the band with the same kind of window _lowpass_taps already uses, just evaluated continuously
    instead of at fixed taps. Pure numpy, no new dependency; vectorised over every output position at once (one
    reply's worth of samples at a time, not per 20 ms frame, so this has no real-time budget to keep)."""
    n = x.size
    idx0 = np.floor(positions).astype(np.int64)
    frac = (positions - idx0).astype(np.float64)
    offsets = np.arange(-half_width + 1, half_width + 1, dtype=np.float64)
    d = frac[:, None] - offsets[None, :]  # (n_out, 2*half_width): each tap's distance from the fractional position
    t = (d + half_width) / (2.0 * half_width)
    window = np.where((t >= 0.0) & (t <= 1.0), 0.42 - 0.5 * np.cos(2 * np.pi * t) + 0.08 * np.cos(4 * np.pi * t), 0.0)
    weights = np.sinc(d) * window
    weights /= np.where(weights.sum(axis=1, keepdims=True) != 0, weights.sum(axis=1, keepdims=True), 1.0)  # unity gain
    tap_idx = idx0[:, None] + offsets[None, :].astype(np.int64)
    valid = (tap_idx >= 0) & (tap_idx < n)
    samples = x[np.clip(tap_idx, 0, n - 1)] * valid
    return (samples * weights).sum(axis=1).astype(np.float32)


def resample_to_8k(samples: np.ndarray, src_rate: int) -> np.ndarray:
    """sourav, 2026-10-21; owner, 2026-10-26: TTS replies are 22.05/24/16 kHz; the phone wants 8 kHz. Flow: low-pass
    at 3.6 kHz at the source rate (anti-aliasing), then a windowed-sinc interpolation onto the 8 kHz grid -- matching
    the same filtering standard this file's inbound path (Upsampler8to16) already uses, instead of the straight-line
    interpolation that used to do this grid mapping. Pure numpy so the bridge adds no new dependency to the pod.
    Returns float32 samples."""
    x = np.asarray(samples, dtype=np.float32)
    if src_rate == TWILIO_RATE or x.size == 0:
        return x
    if src_rate > TWILIO_RATE:
        x = np.convolve(x, _lowpass_taps(3600.0, src_rate), mode="same")
    n_out = int(x.size * TWILIO_RATE / float(src_rate))
    positions = np.arange(n_out) * (src_rate / float(TWILIO_RATE))
    return _sinc_interp(x, positions)


def wav_to_mulaw_8k(wav_bytes: bytes) -> bytes:
    """sourav, 2026-10-21: every reply the agent speaks is a WAV (agent/tts.py). Flow: parse the WAV header, take the
    samples (8/16/32-bit PCM, any channel count -> mono), resample to 8 kHz, encode as mu-law. Raises ValueError /
    wave.Error for anything it cannot read, and the caller turns that into 'skip this clip, release the gate'."""
    with contextlib.closing(wave.open(io.BytesIO(wav_bytes), "rb")) as w:
        channels, width, rate, frames = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
        raw = w.readframes(frames)
    if width == 2:
        x = np.frombuffer(raw, dtype="<i2").astype(np.float32)
    elif width == 1:
        x = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) * 256.0
    elif width == 4:
        x = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 65536.0
    else:
        raise ValueError(f"unsupported WAV sample width {width}")
    if channels > 1:
        x = x[: x.size - x.size % channels].reshape(-1, channels).mean(axis=1)
    y = resample_to_8k(x, rate)
    return mulaw_encode(np.clip(np.rint(y), -32768, 32767).astype(np.int16))


_convert_cache: "collections.OrderedDict[str, bytes]" = collections.OrderedDict()


def wav_to_mulaw_8k_cached(wav_bytes: bytes) -> bytes:
    """sourav, 2026-10-21: why: the catalogue answers (prices, prep, FAQ) are pre-rendered and pinned in the TTS cache so
    they cost 0 ms to synthesise; converting the same bytes again on every call would throw that saving away. Flow:
    hash the WAV, return the stored mu-law if seen, else convert and remember (bounded, oldest evicted)."""
    key = hashlib.sha1(wav_bytes).hexdigest()
    hit = _convert_cache.get(key)
    if hit is not None:
        _convert_cache.move_to_end(key)
        return hit
    out = wav_to_mulaw_8k(wav_bytes)
    _convert_cache[key] = out
    while len(_convert_cache) > _CONVERT_CACHE_MAX:
        _convert_cache.popitem(last=False)
    return out


# ============================================================================== configuration, signature, stream token


def _read_dotenv(path: pathlib.Path) -> dict:
    """sourav, 2026-10-21: the owner keeps the Twilio token in a .env file. Flow: read KEY=VALUE lines, skip blanks and
    comments, strip optional quotes and an 'export ' prefix. No dependency on python-dotenv; never logs values."""
    out: dict = {}
    try:
        text = path.read_text(encoding="utf-8-sig")
    except OSError:
        return out
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().removeprefix("export ").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        out[key] = value
    return out


def get_auth_token() -> str | None:
    """sourav, 2026-10-21: the secret that proves a request really came from Twilio. Flow: the process environment wins
    (the pod's env); otherwise a .env file next to the repo, one folder above it, or in the working directory. Returns
    None when it is not configured, and the webhook then refuses to serve rather than run unprotected."""
    for name in _TOKEN_ENV_NAMES:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    here = pathlib.Path(__file__).resolve().parent.parent
    for candidate in (here / ".env", here.parent / ".env", pathlib.Path.cwd() / ".env"):
        values = _read_dotenv(candidate)
        for name in _TOKEN_ENV_NAMES:
            if values.get(name):
                return values[name]
    return None


def signature_check_enabled() -> bool:
    """sourav, 2026-10-21: the webhook signature check is ON unless TWILIO_VALIDATE_SIGNATURE is explicitly 0/false/no/off
    (a local-testing escape hatch only)."""
    return os.environ.get("TWILIO_VALIDATE_SIGNATURE", "1").strip().lower() not in ("0", "false", "no", "off")


def _digits(number: str) -> str:
    return re.sub(r"\D", "", number or "")


def allowed_callers() -> set:
    """sourav, 2026-10-21: optional allow-list so a stranger who finds the number cannot spend Twilio minutes and GPU.
    Flow: split TWILIO_ALLOWED_CALLERS on commas, keep the digits of each. Empty set means everyone is allowed."""
    raw = os.environ.get("TWILIO_ALLOWED_CALLERS", "")
    return {d for d in (_digits(part) for part in raw.split(",")) if d}


def _mask(number: str) -> str:
    d = _digits(number)
    return ("*" * max(0, len(d) - 4)) + d[-4:] if d else "unknown"


def public_base_url(headers) -> str:
    """sourav, 2026-10-21: the URL Twilio actually called (needed for the signature and to build the wss:// stream URL).
    Behind RunPod's proxy + the pod's nginx the app only sees an internal address, so: TWILIO_PUBLIC_BASE_URL if set,
    else X-Forwarded-Proto/Host (set by the gateway), else Host, assuming https."""
    configured = os.environ.get("TWILIO_PUBLIC_BASE_URL", "").strip().rstrip("/")
    if configured:
        return configured
    host = (headers.get("x-forwarded-host") or headers.get("host") or "").split(",")[0].strip()
    proto = (headers.get("x-forwarded-proto") or "https").split(",")[0].strip() or "https"
    return f"{proto}://{host}"


def compute_signature(auth_token: str, url: str, params) -> str:
    """sourav, 2026-10-21: Twilio's request signature. Flow: URL + every form parameter as name+value, sorted by name,
    HMAC-SHA1 with the Auth Token, base64. `params` is a list of (name, value) pairs."""
    data = url + "".join(k + v for k, v in sorted(params))
    mac = hmac.new(auth_token.encode("utf-8"), data.encode("utf-8"), hashlib.sha1).digest()
    return base64.b64encode(mac).decode("ascii")


def _url_variants(url: str) -> list:
    """Twilio signs the URL exactly as configured, with or without the default :443 port; accept either spelling."""
    out = [url]
    m = re.match(r"^(https://[^/:?#]+)(:443)?(.*)$", url)
    if m:
        out.append(m.group(1) + (m.group(3) if m.group(2) else ":443" + m.group(3)))
    return out


def verify_signature(auth_token: str, url: str, params, header: str) -> bool:
    """sourav, 2026-10-21: constant-time comparison of the X-Twilio-Signature header with what we compute, for each
    accepted spelling of the URL. A missing header never matches."""
    if not header:
        return False
    return any(hmac.compare_digest(compute_signature(auth_token, u, params), header) for u in _url_variants(url))


def make_stream_token(auth_token: str, call_sid: str, now: float | None = None) -> str:
    """sourav, 2026-10-21: why: /twilio/stream is a public WebSocket; without a secret anyone could open one. The webhook
    puts this token in the <Stream> parameters and the stream re-checks it. Flow: timestamp + HMAC(auth token,
    'stream|CallSid|timestamp'), so a token is bound to one call and expires."""
    ts = int(now if now is not None else time.time())
    sig = hmac.new(auth_token.encode("utf-8"), f"stream|{call_sid}|{ts}".encode("utf-8"), hashlib.sha256).hexdigest()[:32]
    return f"{ts}.{sig}"


def verify_stream_token(auth_token: str, call_sid: str, token: str, now: float | None = None, max_age_s: int = TOKEN_MAX_AGE_S) -> bool:
    """sourav, 2026-10-21: counterpart of make_stream_token: right shape, not expired, signature matches this CallSid."""
    try:
        ts_text, _, sig = (token or "").partition(".")
        ts = int(ts_text)
    except ValueError:
        return False
    current = now if now is not None else time.time()
    if not sig or ts > current + 5 or current - ts > max_age_s:
        return False
    expected = make_stream_token(auth_token, call_sid, ts).partition(".")[2]
    return hmac.compare_digest(expected, sig)


def build_stream_twiml(stream_url: str, token: str) -> str:
    """sourav, 2026-10-21: the answer to Twilio's webhook: connect the call's audio to our WebSocket (bidirectional,
    which is what <Connect><Stream> is). Values are XML-escaped; the token rides along as a custom parameter."""
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        f"<Response><Connect><Stream url={quoteattr(stream_url)}>"
        f'<Parameter name="token" value={quoteattr(token)}/>'
        "</Stream></Connect></Response>"
    )


REJECT_TWIML = '<?xml version="1.0" encoding="UTF-8"?><Response><Reject reason="rejected"/></Response>'


# ====================================================================================== WebSocket adapter for ws_audio


class TwilioStreamRejected(Exception):
    """sourav, 2026-10-21: raised when a stream must not be served (no start event, bad token, wrong audio format)."""


_DISCONNECT = {"type": "websocket.disconnect"}


class TwilioStreamAdapter:
    """sourav, 2026-10-21: makes one Twilio Media Stream look like the browser WebSocket that ws_audio() already uses
    (see the table in the module docstring). One adapter per phone call. Only the methods ws_audio and CallSession
    call are provided: accept / receive / send_bytes / send_text / close.

    Inbound: a background reader task decodes Twilio's JSON events into the same dict shapes Starlette's receive()
    returns, via a queue, so receive() can also deliver synthetic events (a clip we could not convert still releases
    the playback gate). Outbound: sends are serialised with a lock because several tasks speak on one call."""

    phone_line = True  # a telephone leg, not a browser: the agent trusts barge-in without the echo-canceller convergence rule (agent/full_duplex.py)

    def __init__(self, ws, auth_token: str, *, check_token: bool = True, start_timeout_s: float = START_TIMEOUT_S) -> None:
        self._ws = ws
        self._auth_token = auth_token
        self._check_token = check_token
        self._start_timeout_s = start_timeout_s
        self.stream_sid: str | None = None
        self.call_sid: str | None = None
        self._up = Upsampler8to16()
        self._inbox: asyncio.Queue = asyncio.Queue()
        self._reader: asyncio.Task | None = None
        self._send_lock = asyncio.Lock()
        self._pending_marks: set = set()
        self._mark_seq = 0
        self._closed = False
        self.frames_in = 0
        self.frames_out = 0

    # -- handshake ---------------------------------------------------------------------------------------------

    async def accept(self) -> None:
        """sourav, 2026-10-21: ws_audio's first line. The real WebSocket is already accepted by the route; what remains
        is Twilio's handshake. Flow: read events until "start" (ignore "connected"), require 8 kHz mu-law, verify the
        token for this CallSid, then start the reader. Anything wrong -> TwilioStreamRejected (the route closes it)."""
        deadline = time.monotonic() + self._start_timeout_s
        start = None
        while start is None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TwilioStreamRejected("no start event from Twilio in time")
            try:
                message = await asyncio.wait_for(self._ws.receive(), timeout=remaining)
            except (asyncio.TimeoutError, TimeoutError):
                raise TwilioStreamRejected("no start event from Twilio in time") from None
            if message.get("type") == "websocket.disconnect":
                raise TwilioStreamRejected("stream closed before start")
            try:
                event = json.loads(message.get("text") or "")
            except (TypeError, ValueError):
                continue
            if event.get("event") == "start":
                start = event.get("start") or {}
        fmt = start.get("mediaFormat") or {}
        if fmt and (str(fmt.get("encoding", "")).lower() != "audio/x-mulaw" or int(fmt.get("sampleRate", TWILIO_RATE)) != TWILIO_RATE):
            raise TwilioStreamRejected(f"unsupported media format {fmt!r}")
        self.stream_sid = start.get("streamSid") or (event.get("streamSid") if isinstance(event, dict) else None)
        self.call_sid = start.get("callSid") or ""
        token = (start.get("customParameters") or {}).get("token", "")
        if self._check_token and not verify_stream_token(self._auth_token, self.call_sid, token):
            raise TwilioStreamRejected("stream token missing, expired or wrong")
        if not self.stream_sid:
            raise TwilioStreamRejected("start event carried no streamSid")
        logger.info("twilio stream started: call %s...%s", self.call_sid[:4], self.call_sid[-4:])
        self._reader = asyncio.ensure_future(self._read_loop())

    # -- inbound -----------------------------------------------------------------------------------------------

    async def _read_loop(self) -> None:
        """sourav, 2026-10-21: why: receive() must be able to return synthetic events too, so Twilio's socket is read
        in one place. Flow: loop on the socket; media -> 16 kHz PCM bytes message; mark -> playback_done when the last
        queued clip has finished; stop/close -> disconnect. Always ends by queueing a disconnect."""
        try:
            while True:
                message = await self._ws.receive()
                if message.get("type") == "websocket.disconnect":
                    break
                try:
                    event = json.loads(message.get("text") or "")
                except (TypeError, ValueError):
                    continue
                kind = event.get("event")
                if kind == "media":
                    media = event.get("media") or {}
                    if media.get("track", "inbound") != "inbound":
                        continue
                    try:
                        pcm8 = mulaw_decode(base64.b64decode(media.get("payload", "")))
                    except (ValueError, TypeError):
                        continue
                    pcm16 = self._up.process(pcm8)
                    if pcm16.size:
                        self.frames_in += 1
                        self._inbox.put_nowait({"type": "websocket.receive", "bytes": pcm16.tobytes()})
                elif kind == "mark":
                    name = (event.get("mark") or {}).get("name")
                    if name in self._pending_marks:
                        self._pending_marks.discard(name)
                        if not self._pending_marks:
                            self._inbox.put_nowait({"type": "websocket.receive", "text": json.dumps({"type": "playback_done"})})
                elif kind == "stop":
                    break
        except (WebSocketDisconnect, RuntimeError):
            pass
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - the call must end cleanly, not leave the turn loop waiting forever
            logger.exception("twilio stream reader failed")
        finally:
            self._closed = True
            self._inbox.put_nowait(_DISCONNECT)

    async def receive(self) -> dict:
        """sourav, 2026-10-21: what ws_audio's loop awaits. Returns the next queued message; once the stream has ended
        it keeps returning a disconnect so the loop exits."""
        if self._closed and self._inbox.empty():
            return _DISCONNECT
        return await self._inbox.get()

    # -- outbound ----------------------------------------------------------------------------------------------

    async def _send(self, payload: dict) -> bool:
        async with self._send_lock:
            if self._closed:
                return False
            try:
                await self._ws.send_text(json.dumps(payload, separators=(",", ":")))
                return True
            except Exception:  # noqa: BLE001 - a dead socket means the caller hung up
                self._closed = True
                return False

    def _release_gate_now(self) -> None:
        """A reply that could not be played must not leave the agent waiting for a 'finished playing' that never comes."""
        self._inbox.put_nowait({"type": "websocket.receive", "text": json.dumps({"type": "playback_done"})})

    async def send_bytes(self, wav_bytes: bytes) -> None:
        """sourav, 2026-10-21: where every spoken reply leaves. Flow: WAV -> 8 kHz mu-law (in a worker thread, off the
        event loop, cached for repeat clips) -> 100 ms "media" messages in order -> one "mark" so Twilio tells us when
        the clip has actually played. An unreadable clip is logged and the gate released instead of stalling the call."""
        if self._closed or not self.stream_sid or not wav_bytes:
            return
        try:
            mulaw = await asyncio.to_thread(wav_to_mulaw_8k_cached, wav_bytes)
        except Exception as e:  # noqa: BLE001 - wave.Error, ValueError, anything odd in a fallback clip
            logger.warning("could not convert a reply clip for Twilio (%s) -- skipped", e)
            self._release_gate_now()
            return
        if not mulaw:
            self._release_gate_now()
            return
        for offset in range(0, len(mulaw), OUT_CHUNK_BYTES):
            chunk = mulaw[offset : offset + OUT_CHUNK_BYTES]
            ok = await self._send(
                {"event": "media", "streamSid": self.stream_sid, "media": {"payload": base64.b64encode(chunk).decode("ascii")}}
            )
            if not ok:
                return
            self.frames_out += 1
        self._mark_seq += 1
        name = f"clip{self._mark_seq}"
        self._pending_marks.add(name)
        await self._send({"event": "mark", "streamSid": self.stream_sid, "mark": {"name": name}})

    async def send_text(self, text: str) -> None:
        """sourav, 2026-10-21: the browser protocol also carries JSON text (transcript lines, pings, config, handoff
        notices) that mean nothing on a phone and are dropped. The one that matters is stop_playback (the caller talked
        over the agent): it becomes Twilio's "clear", which empties the audio still queued at Twilio."""
        try:
            kind = json.loads(text).get("type")
        except (TypeError, ValueError, AttributeError):
            return
        if kind == "stop_playback" and self.stream_sid:
            self._pending_marks.clear()
            await self._send({"event": "clear", "streamSid": self.stream_sid})

    async def close(self, code: int = 1000) -> None:
        """sourav, 2026-10-21: the agent hangs up (goodbye said, idle, handed to a human). The callers of close() already
        wait for their own audio to finish; this adds a short bounded wait for Twilio's last mark, then closes, which
        ends the <Connect> and so the phone call. Safe to call more than once."""
        deadline = time.monotonic() + DRAIN_CLOSE_S
        while self._pending_marks and not self._closed and time.monotonic() < deadline:
            await asyncio.sleep(0.05)
        await self.aclose(code)

    async def aclose(self, code: int = 1000) -> None:
        """sourav, 2026-10-21: immediate, idempotent teardown used by the route's finally block: stop the reader, make
        any receive() return a disconnect, close the real socket."""
        first = not self._closed
        self._closed = True
        if first:
            self._inbox.put_nowait(_DISCONNECT)
        if self._reader is not None and not self._reader.done():
            self._reader.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._reader
        with contextlib.suppress(Exception):
            await self._ws.close(code)


# ============================================================================================================ routes


def install(app, ws_audio_handler) -> None:
    """sourav, 2026-10-21: registers the three Twilio routes on the FastAPI app. Called once from main_pcm.py (before
    the static-files mount, which would otherwise swallow these paths), passing the agent's own ws_audio so a phone
    call runs the identical turn loop. TWILIO_BRIDGE=0 skips it. Routes:
      POST /twilio/voice   Twilio's incoming-call webhook  -> TwiML <Connect><Stream>
      WS   /twilio/stream  the media stream                -> ws_audio(adapter)
      GET  /twilio/health  is the bridge configured (never reveals the token)"""
    if os.environ.get("TWILIO_BRIDGE", "1").strip().lower() in ("0", "false", "no", "off"):
        logger.info("Twilio bridge disabled by TWILIO_BRIDGE")
        return

    @app.post("/twilio/voice")
    async def twilio_voice(request: Request):
        # sourav, 2026-10-21: flow: need the token -> parse the form body -> verify Twilio's signature -> optional caller
        # allow-list -> mint a stream token for this CallSid -> answer with <Connect><Stream url=wss://same-host/twilio/stream>.
        auth_token = get_auth_token()
        if not auth_token:
            logger.error("Twilio webhook hit but no Auth Token is configured (TWILIO_AUTH_TOKEN / .env)")
            return Response("Twilio Auth Token is not configured", status_code=503, media_type="text/plain")
        params = parse_qsl((await request.body()).decode("utf-8", "replace"), keep_blank_values=True)
        base = public_base_url(request.headers)
        if signature_check_enabled():
            query = request.url.query
            url = base + request.url.path + (("?" + query) if query else "")
            if not verify_signature(auth_token, url, params, request.headers.get("x-twilio-signature", "")):
                logger.warning("Twilio webhook refused: bad signature (URL used for the check: %s -- set TWILIO_PUBLIC_BASE_URL if wrong)", url)
                return Response("invalid signature", status_code=403, media_type="text/plain")
        else:
            logger.warning("Twilio webhook signature check is OFF (TWILIO_VALIDATE_SIGNATURE=0) -- testing only")
        form = dict(params)
        caller = form.get("From", "")
        allow = allowed_callers()
        if allow and _digits(caller) not in allow:
            logger.warning("Twilio call from %s rejected: not in TWILIO_ALLOWED_CALLERS", _mask(caller))
            return Response(REJECT_TWIML, media_type="text/xml")
        call_sid = form.get("CallSid", "")
        stream_url = "wss://" + base.split("://", 1)[-1] + "/twilio/stream"
        logger.info("Twilio call %s from %s -> stream", call_sid[:4] + "..." + call_sid[-4:], _mask(caller))
        return Response(build_stream_twiml(stream_url, make_stream_token(auth_token, call_sid)), media_type="text/xml")

    @app.get("/twilio/health")
    async def twilio_health():
        # sourav, 2026-10-21: a no-secret status check for the operator: is the token present, is the signature check on.
        return {
            "bridge": "on",
            "auth_token_configured": get_auth_token() is not None,
            "signature_check": signature_check_enabled(),
            "caller_allow_list_entries": len(allowed_callers()),
            "public_base_url_configured": bool(os.environ.get("TWILIO_PUBLIC_BASE_URL", "").strip()),
        }

    @app.websocket("/twilio/stream")
    async def twilio_stream(ws: WebSocket):
        # sourav, 2026-10-21: flow: accept the socket -> wrap it in the adapter -> run the agent's normal ws_audio on it.
        # ws_audio's own first line (adapter.accept) does Twilio's handshake and the token check; a rejected stream is
        # closed with 1008 (policy violation). The finally block always tears the adapter down.
        await ws.accept()
        auth_token = get_auth_token()
        if not auth_token:
            await ws.close(code=1011)
            return
        adapter = TwilioStreamAdapter(ws, auth_token)
        try:
            await ws_audio_handler(adapter)
        except TwilioStreamRejected as e:
            logger.warning("Twilio stream refused: %s", e)
            await adapter.aclose(1008)
        except Exception:  # noqa: BLE001
            logger.exception("Twilio stream crashed")
        finally:
            await adapter.aclose()
            logger.info("twilio stream ended (frames in %d, out %d)", adapter.frames_in, adapter.frames_out)

    logger.info("Twilio bridge routes registered: POST /twilio/voice, WS /twilio/stream, GET /twilio/health")
