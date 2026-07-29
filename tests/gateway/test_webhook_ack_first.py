"""``ack_first`` routes answer before running their script.

A route ``script:`` normally runs to completion inside the request handler, so a
script slower than the sender's timeout loses the delivery outright — GitHub
allows 10s and does not retry. These tests pin the contract that fixes it:

  * ack_first answers 202 promptly even when the script is slow, and the script
    still runs (and still dispatches) afterwards;
  * a script that ignores the event still suppresses the agent, it just cannot
    be reported in the response any more;
  * the delivery is deduped BEFORE the ack, so a retry of a delivery we already
    promised to handle cannot start a second run;
  * without ack_first, nothing changes — the response still reports the script
    verdict.
"""

import asyncio
import hashlib
import hmac
import json
import time

import pytest

pytest.importorskip("aiohttp")

from gateway.config import PlatformConfig
from gateway.platforms.webhook import WebhookAdapter

SECRET = "test-secret"


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))


def _adapter(route):
    return WebhookAdapter(
        PlatformConfig(enabled=True, extra={"routes": {"r": route}})
    )


class _Req:
    """Minimal aiohttp-request stand-in for _handle_webhook."""

    def __init__(self, body: bytes, event: str = "push", delivery: str = "d1"):
        self._body = body
        self.method = "POST"
        self.match_info = {"route_name": "r"}
        self.content_length = len(body)
        self.remote = "127.0.0.1"
        self.path = "/webhooks/r"
        sig = hmac.new(SECRET.encode(), body, hashlib.sha256).hexdigest()
        self.headers = {
            "X-GitHub-Event": event,
            "X-GitHub-Delivery": delivery,
            "X-Hub-Signature-256": f"sha256={sig}",
            "Content-Length": str(len(body)),
        }

    async def read(self):
        return self._body


def _post(adapter, **kw):
    return asyncio.run(adapter._handle_webhook(_Req(json.dumps({"a": 1}).encode(), **kw)))


def _install_script(adapter, *, delay=0.0, keep=True, seen=None):
    """Replace the route processor's script runner with a controllable stub."""

    def run(script_value, payload):
        if delay:
            time.sleep(delay)
        if seen is not None:
            seen.append(payload)
        return (keep, {**payload, "scripted": True}) if keep else (False, None)

    adapter._route_processor.run_route_script = run


def _capture_dispatch(adapter):
    """Record dispatched prompts instead of running an agent."""
    dispatched = []

    async def handle_message(event):
        dispatched.append(event.text)

    adapter.handle_message = handle_message
    return dispatched


class TestAckFirst:
    def test_answers_202_without_waiting_for_a_slow_script(self):
        adapter = _adapter(
            {"secret": SECRET, "script": "s.sh", "prompt": "p", "ack_first": True}
        )
        _install_script(adapter, delay=0.6)
        _capture_dispatch(adapter)

        # Time the HANDLER, not asyncio.run(): loop teardown joins the
        # to_thread worker, so measuring from outside would always include the
        # script's sleep and the test would pass for the wrong reason.
        async def drive():
            started = time.monotonic()
            resp = await adapter._handle_webhook(
                _Req(json.dumps({"a": 1}).encode())
            )
            elapsed = time.monotonic() - started
            await asyncio.gather(*adapter._background_tasks)
            return resp, elapsed

        resp, elapsed = asyncio.run(drive())

        assert resp.status == 202
        # The whole point: the response must not carry the script's cost.
        assert elapsed < 0.3, f"ack took {elapsed:.2f}s — the script was awaited"

    def test_script_and_dispatch_still_happen_after_the_ack(self):
        adapter = _adapter(
            {"secret": SECRET, "script": "s.sh", "prompt": "{scripted}", "ack_first": True}
        )
        seen = []
        _install_script(adapter, seen=seen)
        dispatched = _capture_dispatch(adapter)

        async def drive():
            resp = await adapter._handle_webhook(
                _Req(json.dumps({"a": 1}).encode())
            )
            # Let the background task run to completion.
            await asyncio.gather(*adapter._background_tasks)
            return resp

        resp = asyncio.run(drive())
        assert resp.status == 202
        assert seen, "the route script never ran"
        assert dispatched == ["True"], dispatched

    def test_script_ignore_still_suppresses_the_agent(self):
        adapter = _adapter(
            {"secret": SECRET, "script": "s.sh", "prompt": "p", "ack_first": True}
        )
        _install_script(adapter, keep=False)
        dispatched = _capture_dispatch(adapter)

        async def drive():
            resp = await adapter._handle_webhook(
                _Req(json.dumps({"a": 1}).encode())
            )
            await asyncio.gather(*adapter._background_tasks)
            return resp

        resp = asyncio.run(drive())
        # Already acked, so the ignore cannot be reported — but it must still
        # be honoured.
        assert resp.status == 202
        assert dispatched == []

    def test_duplicate_delivery_is_dropped_before_the_script_runs(self):
        adapter = _adapter(
            {"secret": SECRET, "script": "s.sh", "prompt": "p", "ack_first": True}
        )
        seen = []
        _install_script(adapter, seen=seen)
        _capture_dispatch(adapter)

        async def drive():
            first = await adapter._handle_webhook(
                _Req(json.dumps({"a": 1}).encode(), delivery="same")
            )
            await asyncio.gather(*adapter._background_tasks)
            second = await adapter._handle_webhook(
                _Req(json.dumps({"a": 1}).encode(), delivery="same")
            )
            if adapter._background_tasks:
                await asyncio.gather(*adapter._background_tasks)
            return first, second

        first, second = asyncio.run(drive())
        assert first.status == 202
        assert second.status == 200
        assert json.loads(second.body)["status"] == "duplicate"
        assert len(seen) == 1, "a retry re-ran the script"

    def test_ack_first_without_a_script_is_inert(self):
        # No script means nothing slow to defer; the route behaves normally.
        adapter = _adapter({"secret": SECRET, "prompt": "p", "ack_first": True})
        dispatched = _capture_dispatch(adapter)

        async def drive():
            resp = await adapter._handle_webhook(
                _Req(json.dumps({"a": 1}).encode())
            )
            await asyncio.gather(*adapter._background_tasks)
            return resp

        resp = asyncio.run(drive())
        assert resp.status == 202
        assert dispatched == ["p"]


class TestDefaultModeUnchanged:
    def test_script_verdict_is_still_reported_without_ack_first(self):
        adapter = _adapter({"secret": SECRET, "script": "s.sh", "prompt": "p"})
        _install_script(adapter, keep=False)
        dispatched = _capture_dispatch(adapter)

        resp = _post(adapter)
        body = json.loads(resp.body)
        assert body["status"] == "ignored"
        assert body["reason"] == "script"
        assert dispatched == []

    def test_kept_script_still_dispatches_inline(self):
        adapter = _adapter(
            {"secret": SECRET, "script": "s.sh", "prompt": "{scripted}"}
        )
        _install_script(adapter)
        dispatched = _capture_dispatch(adapter)

        async def drive():
            resp = await adapter._handle_webhook(
                _Req(json.dumps({"a": 1}).encode())
            )
            await asyncio.gather(*adapter._background_tasks)
            return resp

        resp = asyncio.run(drive())
        assert resp.status == 202
        assert dispatched == ["True"]
