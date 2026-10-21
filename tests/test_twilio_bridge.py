"""sourav, 2026-10-21: tests for agent/twilio_bridge.py (the Twilio phone-call bridge).
Why: the bridge sits between the public internet and the agent, so codec maths, signature/token checks and the
WebSocket handshake are all verified without a real call. Flow: codec -> security -> webhook -> stream handshake.
Async parts run through asyncio.run so no pytest-asyncio mode is required."""
import asyncio
import audioop
import base64
import io
import json
import wave

import numpy as np
import pytest

pytest.importorskip("fastapi")
from fastapi import FastAPI  # noqa: E402
from starlette.testclient import TestClient  # noqa: E402

from agent import twilio_bridge as tb  # noqa: E402

TOKEN = "unit-test-token"


# ------------------------------------------------------------------------------------------------ codec
def test_mulaw_decode_matches_audioop():
    data = bytes(range(256))
    ref = np.frombuffer(audioop.ulaw2lin(data, 2), dtype="<i2")
    assert np.array_equal(tb.mulaw_decode(data), ref)


def test_mulaw_encode_matches_audioop_and_roundtrips():
    rng = np.random.default_rng(1)
    pcm = rng.integers(-32768, 32767, 4000).astype("<i2")
    ours = tb.mulaw_encode(pcm)
    ref = audioop.lin2ulaw(pcm.tobytes(), 2)
    # audioop truncates a 14-bit sample, ours is the CCITT G.711 reference (bias 0x84): codes may differ by one step.
    diff = np.abs(np.frombuffer(ours, np.uint8).astype(int) - np.frombuffer(ref, np.uint8).astype(int))
    assert diff.max() <= 1 and (diff != 0).mean() < 0.02
    back = tb.mulaw_decode(ours).astype(np.int32)
    assert np.max(np.abs(back - pcm.astype(np.int32))) <= 1100  # largest mu-law step near full scale


def test_upsampler_doubles_length_and_is_continuous():
    up = tb.Upsampler8to16()
    x = (np.sin(np.arange(320) * 0.05) * 8000).astype("<i2")
    whole = up.process(x)
    up2 = tb.Upsampler8to16()
    split = np.concatenate([up2.process(x[:100]), up2.process(x[100:])])
    assert whole.size == 640 and np.array_equal(whole, split)


def _wav(rate, seconds=0.5, freq=440, channels=1, width=2):
    t = np.arange(int(rate * seconds)) / rate
    s = (np.sin(2 * np.pi * freq * t) * 12000).astype("<i2")
    if channels == 2:
        s = np.repeat(s, 2)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(width)
        w.setframerate(rate)
        w.writeframes(s.tobytes())
    return buf.getvalue()


@pytest.mark.parametrize("rate", [8000, 16000, 22050, 24000])
def test_wav_to_mulaw_length_and_tone(rate):
    out = tb.wav_to_mulaw_8k(_wav(rate))
    assert abs(len(out) - 4000) <= 8  # 0.5 s at 8 kHz, 1 byte per sample
    pcm = tb.mulaw_decode(out).astype(float)
    spec = np.abs(np.fft.rfft(pcm * np.hanning(pcm.size)))
    assert abs(np.argmax(spec) * 8000 / pcm.size - 440) < 20


def test_resample_removes_alias_above_nyquist():
    t = np.arange(24000) / 24000
    x = (np.sin(2 * np.pi * 6000 * t) * 12000).astype("<i2")  # 6 kHz cannot exist at 8 kHz rate
    y = tb.resample_to_8k(x, 24000)
    assert np.sqrt(np.mean(np.asarray(y, float) ** 2)) < 0.05 * 12000


def test_wav_stereo_and_cache():
    a = tb.wav_to_mulaw_8k(_wav(16000, channels=2))
    assert len(a) > 0
    w = _wav(16000)
    assert tb.wav_to_mulaw_8k_cached(w) == tb.wav_to_mulaw_8k_cached(w)


# ------------------------------------------------------------------------------------------------ security
def test_twilio_documented_signature_vector():
    params = [("CallSid", "CA1234567890ABCDE"), ("Caller", "+14158675310"), ("Digits", "1234"),
              ("From", "+14158675310"), ("To", "+18005551212")]
    sig = tb.compute_signature("12345", "https://mycompany.com/myapp.php?foo=1&bar=2", params)
    assert sig == "GvWf1cFY/Q7PnoempGyD5oXAezc="
    assert tb.verify_signature("12345", "https://mycompany.com/myapp.php?foo=1&bar=2", params, sig)
    assert not tb.verify_signature("12345", "https://mycompany.com/myapp.php?foo=1&bar=2", params, "")
    assert not tb.verify_signature("wrong", "https://mycompany.com/myapp.php?foo=1&bar=2", params, sig)


def test_stream_token_roundtrip_expiry_tamper():
    t = tb.make_stream_token(TOKEN, "CA1", now=1000)
    assert tb.verify_stream_token(TOKEN, "CA1", t, now=1010)
    assert not tb.verify_stream_token(TOKEN, "CA2", t, now=1010)          # other call
    assert not tb.verify_stream_token(TOKEN, "CA1", t, now=1000 + 10_000)  # expired
    assert not tb.verify_stream_token("other", "CA1", t, now=1010)
    assert not tb.verify_stream_token(TOKEN, "CA1", "garbage", now=1010)
    assert not tb.verify_stream_token(TOKEN, "CA1", "", now=1010)


def test_twiml_escapes_values():
    x = tb.build_stream_twiml('wss://h/x?a=1&b="2"', "t<>&")
    assert "&amp;" in x and "<Connect><Stream" in x and "t<>&" not in x


# ------------------------------------------------------------------------------------------------ webhook
@pytest.fixture
def env(monkeypatch):
    for k in ("TWILIO_AUTH_TOKEN", "AuthToken", "TWILIO_TOKEN", "TWILIO_ALLOWED_CALLERS", "TWILIO_VALIDATE_SIGNATURE", "TWILIO_BRIDGE"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", TOKEN)
    monkeypatch.setenv("TWILIO_PUBLIC_BASE_URL", "https://pod-8888.proxy.runpod.net")
    return monkeypatch


def _app(handler=None):
    app = FastAPI()

    async def default_handler(adapter):
        await adapter.accept()
        while True:
            m = await adapter.receive()
            if m["type"] == "websocket.disconnect":
                return

    tb.install(app, handler or default_handler)
    return app


FORM = {"CallSid": "CA00112233445566", "From": "+919830012345", "To": "+14157970932"}


def _post(client, form=FORM, token=TOKEN, sign=True):
    url = "https://pod-8888.proxy.runpod.net/twilio/voice"
    headers = {"X-Twilio-Signature": tb.compute_signature(token, url, list(form.items()))} if sign else {}
    return client.post("/twilio/voice", data=form, headers=headers)


def test_webhook_valid_signature_returns_stream_twiml(env):
    r = _post(TestClient(_app()))
    assert r.status_code == 200 and "<Connect><Stream" in r.text
    assert 'url="wss://pod-8888.proxy.runpod.net/twilio/stream"' in r.text
    assert TOKEN not in r.text


def test_webhook_bad_or_missing_signature_403(env):
    c = TestClient(_app())
    assert _post(c, token="nope").status_code == 403
    assert _post(c, sign=False).status_code == 403


def test_webhook_no_token_503(env):
    env.delenv("TWILIO_AUTH_TOKEN")
    env.setattr(tb, "_read_dotenv", lambda p: {})
    assert _post(TestClient(_app())).status_code == 503


def test_webhook_allow_list(env):
    env.setenv("TWILIO_ALLOWED_CALLERS", "+91 98300 12345")
    c = TestClient(_app())
    assert "<Connect>" in _post(c).text
    other = dict(FORM, From="+911111111111")
    assert "<Reject" in _post(c, form=other).text


def test_health_hides_secret(env):
    r = TestClient(_app()).get("/twilio/health")
    assert r.status_code == 200 and r.json()["auth_token_configured"] is True and TOKEN not in r.text


def test_bridge_can_be_disabled(env):
    env.setenv("TWILIO_BRIDGE", "0")
    assert TestClient(_app()).get("/twilio/health").status_code == 404


# ------------------------------------------------------------------------------------------------ stream
def _start(token, sid="CA00112233445566"):
    return json.dumps({"event": "start", "streamSid": "MZ1", "start": {
        "streamSid": "MZ1", "callSid": sid, "customParameters": {"token": token},
        "mediaFormat": {"encoding": "audio/x-mulaw", "sampleRate": 8000, "channels": 1}}})


def test_stream_rejects_bad_token_with_1008(env):
    c = TestClient(_app())
    with c.websocket_connect("/twilio/stream") as ws:
        ws.send_text(json.dumps({"event": "connected"}))
        ws.send_text(_start("1.bad"))
        msg = ws.receive()
        assert msg["type"] == "websocket.close" and msg["code"] == 1008


def test_stream_full_turn_media_mark_and_clear(env):
    seen = {}

    async def handler(adapter):
        await adapter.accept()
        got = await adapter.receive()
        seen["bytes"] = len(got["bytes"])
        await adapter.send_bytes(_wav(16000, 0.3))
        done = await adapter.receive()
        seen["done"] = json.loads(done["text"])["type"]
        await adapter.send_text(json.dumps({"type": "stop_playback"}))
        await adapter.send_text(json.dumps({"type": "transcript", "text": "ignored"}))
        end = await adapter.receive()
        seen["end"] = end["type"]
        await adapter.close()

    c = TestClient(_app(handler))
    with c.websocket_connect("/twilio/stream") as ws:
        ws.send_text(json.dumps({"event": "connected"}))
        ws.send_text(_start(tb.make_stream_token(TOKEN, "CA00112233445566")))
        ws.send_text(json.dumps({"event": "media", "media": {"track": "inbound", "payload": base64.b64encode(b"\xff" * 160).decode()}}))
        events = []
        while True:
            e = json.loads(ws.receive_text())
            events.append(e)
            if e["event"] == "mark":
                break
        assert all(x["streamSid"] == "MZ1" for x in events)
        assert sum(1 for x in events if x["event"] == "media") == 3  # 0.3 s = 2400 bytes = 3 x 800
        ws.send_text(json.dumps({"event": "mark", "mark": {"name": events[-1]["mark"]["name"]}}))
        assert json.loads(ws.receive_text())["event"] == "clear"
        ws.send_text(json.dumps({"event": "stop"}))
    assert seen == {"bytes": 640, "done": "playback_done", "end": "websocket.disconnect"}


def test_unconvertible_clip_releases_gate_instead_of_stalling():
    class FakeWS:
        async def send_text(self, t): pass
        async def close(self, c=1000): pass
        async def receive(self): await asyncio.sleep(3600)

    async def run():
        a = tb.TwilioStreamAdapter(FakeWS(), TOKEN)
        a.stream_sid = "MZ1"
        await a.send_bytes(b"not a wav file")
        m = await asyncio.wait_for(a.receive(), 1)
        await a.aclose()
        await a.aclose()  # idempotent
        return json.loads(m["text"])["type"]

    assert asyncio.run(run()) == "playback_done"
