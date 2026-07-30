"""Wall-clock cap on a gateway agent run (``agent.gateway_max_run_seconds``).

``agent.gateway_timeout`` is an INACTIVITY timeout — gateway/run.py says so:
"the agent can run for hours if it's actively calling tools". A run that is busy
being wrong therefore has no ceiling at all, which matters most for unattended
runs (a webhook-triggered PR review) where nobody is watching the clock.

These tests drive the real predicate rather than a copy of the poll loop, and
pin the two properties that make it safe: unlimited by default, and monotonic
(so an NTP step or a sleep/wake cannot fire it spuriously).
"""

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gateway.run import _walltime_budget_exceeded


class TestWalltimeBudget:
    def test_unlimited_by_default(self):
        # None is what a missing/zero config resolves to. Never fires.
        assert _walltime_budget_exceeded(time.monotonic() - 10_000, None) is False

    def test_not_exceeded_within_budget(self):
        assert _walltime_budget_exceeded(time.monotonic(), 60.0) is False

    def test_exceeded_past_budget(self):
        assert _walltime_budget_exceeded(time.monotonic() - 61.0, 60.0) is True

    def test_boundary_is_inclusive(self):
        # >= not >, so a budget of 0.0 that somehow reaches here fires at once
        # rather than running forever one poll at a time.
        assert _walltime_budget_exceeded(time.monotonic(), 0.0) is True

    def test_uses_monotonic_not_wall_clock(self, monkeypatch):
        """A system-clock step must not trip the cap.

        Keyed on time.time(), an NTP correction or a laptop resuming from sleep
        would look like hours of elapsed run time and kill a healthy agent.
        """
        fake_now = [1000.0]
        monkeypatch.setattr(time, "monotonic", lambda: fake_now[0])
        # Pretend the wall clock jumped forward a day; monotonic did not.
        monkeypatch.setattr(time, "time", lambda: 10**9)

        started = 1000.0
        assert _walltime_budget_exceeded(started, 60.0) is False
        fake_now[0] = 1059.9
        assert _walltime_budget_exceeded(started, 60.0) is False
        fake_now[0] = 1060.0
        assert _walltime_budget_exceeded(started, 60.0) is True


class TestPollLoopIntegration:
    """The predicate must actually stop a busy run.

    Mirrors the poll-loop shape used by the sibling inactivity-timeout suite,
    with a fake agent that stays ACTIVE throughout — the case the inactivity
    clock can never catch.
    """

    def test_busy_agent_is_stopped_by_the_cap(self):
        import concurrent.futures

        class BusyAgent:
            """Never idle, never finishes within the test's patience."""

            def __init__(self):
                self.interrupted = False

            def get_activity_summary(self):
                return {"seconds_since_activity": 0.0, "api_call_count": 3,
                        "max_iterations": 90, "current_tool": "terminal",
                        "last_activity_desc": "tool_call"}

            def interrupt(self, msg):
                self.interrupted = True

            def run_conversation(self, prompt):
                time.sleep(3.0)
                return {"final_response": "too late", "messages": []}

        agent = BusyAgent()
        budget = 0.3
        inactivity_timeout = 30.0
        poll = 0.05

        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = pool.submit(agent.run_conversation, "x")
        started = time.monotonic()
        walltime_timeout = False
        inactivity_fired = False

        while True:
            done, _ = concurrent.futures.wait({future}, timeout=poll)
            if done:
                future.result()
                break
            idle = agent.get_activity_summary()["seconds_since_activity"]
            if idle >= inactivity_timeout:
                inactivity_fired = True
                break
            if _walltime_budget_exceeded(started, budget):
                walltime_timeout = True
                break

        if walltime_timeout:
            agent.interrupt("timeout")
        pool.shutdown(wait=False, cancel_futures=True)

        assert walltime_timeout, "cap did not stop a busy run"
        assert not inactivity_fired, "inactivity clock must not fire on a busy agent"
        assert agent.interrupted, "the run was not interrupted"

    def test_no_cap_lets_a_busy_agent_finish(self):
        import concurrent.futures

        class QuickAgent:
            def get_activity_summary(self):
                return {"seconds_since_activity": 0.0}

            def run_conversation(self, prompt):
                time.sleep(0.2)
                return {"final_response": "done", "messages": []}

        agent = QuickAgent()
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = pool.submit(agent.run_conversation, "x")
        started = time.monotonic()
        walltime_timeout = False
        result = None

        while True:
            done, _ = concurrent.futures.wait({future}, timeout=0.05)
            if done:
                result = future.result()
                break
            if _walltime_budget_exceeded(started, None):
                walltime_timeout = True
                break

        pool.shutdown(wait=False, cancel_futures=True)
        assert not walltime_timeout
        assert result["final_response"] == "done"
