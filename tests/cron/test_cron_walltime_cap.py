"""Wall-clock cap on a cron job run (``cron.max_run_seconds``).

``HERMES_CRON_TIMEOUT`` is an INACTIVITY limit — scheduler.py says so: "the job
can run for hours if it's actively calling tools". A job that is busy being
wrong therefore had no ceiling at all. That matters now that the PR-review
webhook dispatches its rounds through cron rather than running them inline: a
runaway round holds the pipeline's dispatch lease for as long as it lives.

These tests pin the resolver (opt-in, env-over-config, fails open on garbage)
and the loop behaviour that the cap must produce, including the branch that
previously did not poll at all.
"""

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cron.scheduler import _resolve_cron_max_run_seconds
from hermes_time import walltime_budget_exceeded


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("HERMES_CRON_MAX_RUN_SECONDS", raising=False)


def _patch_config(monkeypatch, cfg):
    monkeypatch.setattr("cron.scheduler.load_config", lambda: cfg)


class TestResolveCronMaxRunSeconds:
    def test_unlimited_by_default(self, monkeypatch):
        """Opt-in: absent config must not invent a cap."""
        _patch_config(monkeypatch, {})
        assert _resolve_cron_max_run_seconds() is None

    def test_zero_means_unlimited(self, monkeypatch):
        _patch_config(monkeypatch, {"cron": {"max_run_seconds": 0}})
        assert _resolve_cron_max_run_seconds() is None

    def test_config_value(self, monkeypatch):
        _patch_config(monkeypatch, {"cron": {"max_run_seconds": 1800}})
        assert _resolve_cron_max_run_seconds() == 1800.0

    def test_env_wins_over_config(self, monkeypatch):
        _patch_config(monkeypatch, {"cron": {"max_run_seconds": 1800}})
        monkeypatch.setenv("HERMES_CRON_MAX_RUN_SECONDS", "60")
        assert _resolve_cron_max_run_seconds() == 60.0

    def test_garbage_fails_open(self, monkeypatch):
        """A bad value must never impose a cap that kills healthy jobs."""
        _patch_config(monkeypatch, {})
        monkeypatch.setenv("HERMES_CRON_MAX_RUN_SECONDS", "not-a-number")
        assert _resolve_cron_max_run_seconds() is None

    def test_negative_is_unlimited(self, monkeypatch):
        _patch_config(monkeypatch, {"cron": {"max_run_seconds": -5}})
        assert _resolve_cron_max_run_seconds() is None

    def test_unreadable_config_is_unlimited(self, monkeypatch):
        def _boom():
            raise RuntimeError("config on fire")
        monkeypatch.setattr("cron.scheduler.load_config", _boom)
        assert _resolve_cron_max_run_seconds() is None


class TestCronPollLoopBehaviour:
    """The cap must stop a BUSY job — the case inactivity can never catch.

    Mirrors the scheduler's loop shape (concurrent.futures + _POLL_INTERVAL)
    with a fake agent that stays active throughout.
    """

    def _run(self, *, budget, inactivity, idle_secs, run_seconds):
        import concurrent.futures

        class FakeAgent:
            def __init__(self):
                self.interrupted_with = None

            def get_activity_summary(self):
                return {"seconds_since_activity": idle_secs, "api_call_count": 4,
                        "max_iterations": 90, "current_tool": "terminal",
                        "last_activity_desc": "tool_call"}

            def interrupt(self, msg):
                self.interrupted_with = msg

            def run_conversation(self, prompt):
                time.sleep(run_seconds)
                return {"final_response": "done"}

        agent = FakeAgent()
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        fut = pool.submit(agent.run_conversation, "x")
        started = time.monotonic()
        inactivity_timeout = False
        walltime_timeout = False
        result = None

        while True:
            done, _ = concurrent.futures.wait({fut}, timeout=0.05)
            if done:
                result = fut.result()
                break
            if inactivity is not None:
                idle = agent.get_activity_summary()["seconds_since_activity"]
                if idle >= inactivity:
                    inactivity_timeout = True
                    break
            if walltime_budget_exceeded(started, budget):
                walltime_timeout = True
                break

        if walltime_timeout:
            agent.interrupt("Cron job timed out (wall clock)")
        elif inactivity_timeout:
            agent.interrupt("Cron job timed out (inactivity)")
        pool.shutdown(wait=False, cancel_futures=True)
        return agent, inactivity_timeout, walltime_timeout, result

    def test_busy_job_is_stopped_by_the_cap(self):
        agent, inact, wall, _ = self._run(
            budget=0.3, inactivity=30.0, idle_secs=0.0, run_seconds=3.0,
        )
        assert wall, "cap did not stop a busy job"
        assert not inact, "inactivity must not fire on a busy job"
        assert agent.interrupted_with == "Cron job timed out (wall clock)"

    def test_inactivity_still_fires_when_idle(self):
        """The cap must not shadow the existing watchdog."""
        agent, inact, wall, _ = self._run(
            budget=30.0, inactivity=0.5, idle_secs=5.0, run_seconds=3.0,
        )
        assert inact and not wall
        assert agent.interrupted_with == "Cron job timed out (inactivity)"

    def test_no_cap_lets_the_job_finish(self):
        agent, inact, wall, result = self._run(
            budget=None, inactivity=30.0, idle_secs=0.0, run_seconds=0.2,
        )
        assert not wall and not inact
        assert result["final_response"] == "done"
        assert agent.interrupted_with is None

    def test_cap_applies_with_inactivity_unlimited(self):
        """'No inactivity limit' must not mean 'no limit at all'.

        This is the branch that previously called future.result() and blocked
        with no poll, so a cap there needs the loop to exist at all.
        """
        agent, inact, wall, _ = self._run(
            budget=0.3, inactivity=None, idle_secs=0.0, run_seconds=3.0,
        )
        assert wall
        assert not inact
        assert agent.interrupted_with == "Cron job timed out (wall clock)"
