"""Approval policy for sessions with no human listener.

A webhook delivery is machine-to-machine: nothing is "in the chat" to answer an
approval prompt. Before this policy existed, such a session took the interactive
gateway branch and — with no notify callback registered — left a pending
approval in a queue nobody would ever read, while telling the agent its action
was awaiting approval. Observed live: `workflow-launch:pr-review-loop-3` during
an unattended PR-review round.

Cron already had the equivalent answer (`approvals.cron_mode`); these tests pin
the same treatment for unattended gateway platforms.
"""

import pytest

from tools import approval


@pytest.fixture(autouse=True)
def _clear_config_cache(monkeypatch):
    """Drive policy from an in-memory config rather than the user's real one."""
    cfg = {}

    def _fake_load_config():
        return cfg

    monkeypatch.setattr("hermes_cli.config.load_config", _fake_load_config)
    return cfg


def _set_platform(monkeypatch, platform: str):
    monkeypatch.setattr(approval, "_get_session_platform", lambda: platform)


class TestIsUnattendedSession:
    def test_webhook_is_unattended_by_default(self, monkeypatch):
        _set_platform(monkeypatch, "webhook")
        assert approval._is_unattended_session() is True

    def test_telegram_is_attended(self, monkeypatch):
        _set_platform(monkeypatch, "telegram")
        assert approval._is_unattended_session() is False

    def test_no_platform_is_attended(self, monkeypatch):
        _set_platform(monkeypatch, "")
        assert approval._is_unattended_session() is False

    def test_case_and_whitespace_tolerated(self, monkeypatch):
        _set_platform(monkeypatch, "  WebHook ")
        assert approval._is_unattended_session() is True

    def test_config_can_add_a_platform(self, monkeypatch, _clear_config_cache):
        _clear_config_cache["approvals"] = {"unattended_platforms": ["telegram"]}
        _set_platform(monkeypatch, "telegram")
        assert approval._is_unattended_session() is True
        # And the default is REPLACED, not merged — an explicit list is the list.
        _set_platform(monkeypatch, "webhook")
        assert approval._is_unattended_session() is False

    def test_scalar_config_value_is_accepted(self, monkeypatch, _clear_config_cache):
        _clear_config_cache["approvals"] = {"unattended_platforms": "webhook"}
        _set_platform(monkeypatch, "webhook")
        assert approval._is_unattended_session() is True

    def test_garbage_config_falls_back_to_default(self, monkeypatch, _clear_config_cache):
        _clear_config_cache["approvals"] = {"unattended_platforms": 42}
        _set_platform(monkeypatch, "webhook")
        assert approval._is_unattended_session() is True


class TestUnattendedMode:
    def test_defaults_to_deny(self, _clear_config_cache):
        assert approval._get_unattended_approval_mode() == "deny"

    @pytest.mark.parametrize("value", ["approve", "off", "allow", "yes", "APPROVE"])
    def test_permissive_synonyms(self, _clear_config_cache, value):
        _clear_config_cache["approvals"] = {"unattended_mode": value}
        assert approval._get_unattended_approval_mode() == "approve"

    @pytest.mark.parametrize("value", ["deny", "block", "", "nonsense"])
    def test_anything_else_denies(self, _clear_config_cache, value):
        _clear_config_cache["approvals"] = {"unattended_mode": value}
        assert approval._get_unattended_approval_mode() == "deny"


class TestUnattendedDenial:
    def test_denies_on_an_unattended_platform(self, monkeypatch, _clear_config_cache):
        _set_platform(monkeypatch, "webhook")
        out = approval._unattended_denial("workflow_launch", "Launch dynamic workflow")
        assert out is not None
        assert out["approved"] is False
        assert out["user_consent"] is False
        assert out["pattern_key"] == "workflow_launch"
        # The message must tell the model to stop trying, not to wait.
        assert "Do NOT retry" in out["message"]
        assert "no user to ask" in out["message"]

    def test_falls_through_on_an_attended_platform(self, monkeypatch, _clear_config_cache):
        _set_platform(monkeypatch, "telegram")
        assert approval._unattended_denial("k", "d") is None

    def test_falls_through_when_mode_is_approve(self, monkeypatch, _clear_config_cache):
        _clear_config_cache["approvals"] = {"unattended_mode": "approve"}
        _set_platform(monkeypatch, "webhook")
        assert approval._unattended_denial("k", "d") is None

    def test_command_shape_carries_a_status(self, monkeypatch, _clear_config_cache):
        _set_platform(monkeypatch, "webhook")
        plain = approval._unattended_denial("k", "d")
        cmd = approval._unattended_denial("k", "d", status_shape="command")
        assert "status" not in plain
        assert cmd["status"] == "denied"
        # Never "pending_approval" — that is the leak this replaces.
        assert cmd["status"] != "pending_approval"

    def test_no_pending_approval_is_submitted(self, monkeypatch, _clear_config_cache):
        """The whole point: nothing is queued for a human who will never look."""
        submitted = []
        monkeypatch.setattr(
            approval, "submit_pending",
            lambda key, data: submitted.append((key, data)),
        )
        _set_platform(monkeypatch, "webhook")
        approval._unattended_denial("k", "d")
        assert submitted == []
