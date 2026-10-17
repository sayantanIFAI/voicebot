"""The SMS / WhatsApp and payment hooks: placeholders by default (owner, 2026-10-17), real the moment an endpoint is configured."""
import importlib
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "clinic-api"))
notifications = importlib.import_module("notifications")
payments = importlib.import_module("payments")


def test_without_a_provider_a_message_stays_queued(monkeypatch):
    monkeypatch.delenv("NOTIFY_PROVIDER_URL", raising=False)
    for channel in ("sms", "whatsapp"):
        assert notifications.send(channel, "9000000001", "hello", "KCD-1")["status"] == "queued"


def test_an_unknown_channel_fails_quietly():
    assert notifications.send("fax", "9000000001", "x")["status"] == "failed"


def test_with_a_provider_the_message_is_posted(monkeypatch):
    seen = {}

    class R:
        is_success = True
        status_code = 200

    def fake_post(url, json=None, headers=None, timeout=None):
        seen.update(url=url, json=json)
        return R()

    monkeypatch.setenv("NOTIFY_PROVIDER_URL", "http://provider.test/send")
    monkeypatch.setattr(notifications.httpx, "post", fake_post)
    out = notifications.send("whatsapp", "9000000001", "pay here", "KCD-9")
    assert out["status"] == "sent" and seen["json"] == {"channel": "whatsapp", "to": "9000000001", "message": "pay here", "reference": "KCD-9"}


def test_a_provider_that_is_down_never_raises(monkeypatch):
    monkeypatch.setenv("NOTIFY_PROVIDER_URL", "http://provider.test/send")

    def boom(*a, **k):
        raise notifications.httpx.ConnectError("down")

    monkeypatch.setattr(notifications.httpx, "post", boom)
    assert notifications.send("sms", "9000000001", "x")["status"] == "failed"


def test_the_payment_link_is_the_placeholder_until_a_gateway_is_set(monkeypatch):
    monkeypatch.delenv("PAYMENT_LINK_URL", raising=False)
    link = payments.create_payment_link("KCD-1", 300)
    assert link["url"] == "https://pay.kolkata-care-voice-agent.invalid/KCD-1" and link["provider"] == "placeholder"
