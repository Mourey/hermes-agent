"""Spec 042 Phase D — omp as a kanban card runner + ``kanban.default_runner``.

Covers the omp spawn leg in ``_default_spawn`` (argv shape with default
``kimi-code/k3`` + ``--thinking max``, model/thinking overrides, --max-time
wiring, env pins, pre-flight failure accounting), the default-runner
resolution order (card pin > ``kanban.default_runner`` config > hermes),
and ``swarm_preset`` prompt rendering per runner.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    (home / "profiles" / "elias").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.fixture
def fake_omp_binary(tmp_path, monkeypatch):
    """A stand-in omp CLI that answers ``--version`` successfully."""
    binary = tmp_path / "omp"
    binary.write_text("#!/bin/sh\necho omp/0.0.0-test\n", encoding="utf-8")
    binary.chmod(0o755)
    monkeypatch.setenv(kb.OMP_BINARY_PATH_ENV, str(binary))
    return binary


def _make_omp_task(kb_mod, **overrides):
    fields = {
        "id": "t_omp",
        "title": "omp card",
        "body": None,
        "assignee": "elias",
        "status": "running",
        "priority": 0,
        "created_by": "test",
        "created_at": 1,
        "started_at": None,
        "completed_at": None,
        "workspace_kind": "dir",
        "workspace_path": None,
        "claim_lock": "lock",
        "claim_expires": None,
        "tenant": None,
        "current_run_id": 7,
        "runner": "omp",
    }
    fields.update(overrides)
    return kb_mod.Task(**fields)


class _FakeProc:
    """Popen stand-in compatible with both the fire-and-forget spawn and
    the ``subprocess.run`` context-manager protocol the omp ``--version``
    pre-flight uses."""

    pid = 4245
    returncode = 0

    def __init__(self, cmd):
        self.args = cmd

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def poll(self):
        return self.returncode

    def communicate(self, *args, **kwargs):
        return (b"", b"")


def _capture_popen(monkeypatch) -> list:
    """Patch subprocess.Popen; return a list of captured invocations. The
    omp pre-flight's ``--version`` call lands first, the worker spawn last."""
    calls = []

    def fake_popen(cmd, *args, **kwargs):
        calls.append({"cmd": list(cmd), "kwargs": kwargs})
        return _FakeProc(list(cmd))

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    return calls


# ---------------------------------------------------------------------------
# omp spawn leg
# ---------------------------------------------------------------------------


def test_omp_spawn_argv_defaults_model_thinking(
    kanban_home, fake_omp_binary, monkeypatch, tmp_path
):
    """NULL model_override → operator default kimi-code/k3 with thinking max."""
    calls = _capture_popen(monkeypatch)

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    task = _make_omp_task(kb)
    pid = kb._default_spawn(task, str(workspace))

    assert pid == 4245
    # First call is the --version pre-flight against the configured binary.
    assert calls[0]["cmd"] == [str(fake_omp_binary), "--version"]
    captured = calls[-1]
    cmd = captured["cmd"]
    assert cmd[0] == str(fake_omp_binary)
    assert cmd[1] == "-p"
    assert cmd[2] == kb.render_worker_prompt(task, str(workspace))
    assert "hermes kanban complete t_omp --summary" in cmd[2]
    assert cmd[3:] == [
        "--no-session",
        "--model", "kimi-code/k3",
        "--thinking", "max",
    ]
    # Same env pins the hermes/kimi legs set.
    env = dict(captured["kwargs"].get("env") or {})
    assert env["HERMES_KANBAN_TASK"] == "t_omp"
    assert env["HERMES_KANBAN_RUN_ID"] == "7"
    assert env["HERMES_KANBAN_CLAIM_LOCK"] == "lock"
    assert env["HERMES_KANBAN_BOARD"]
    assert env["HERMES_KANBAN_DB"]
    assert env["HERMES_KANBAN_WORKSPACES_ROOT"]
    assert env["HERMES_SESSION_SOURCE"] == "kanban"
    assert captured["kwargs"].get("cwd") == str(workspace)
    log_path = kb.worker_logs_dir() / "t_omp.log"
    assert log_path.is_file()


def test_omp_spawn_model_thinking_max_time_overrides(
    kanban_home, fake_omp_binary, monkeypatch, tmp_path
):
    calls = _capture_popen(monkeypatch)

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    task = _make_omp_task(
        kb,
        model_override="kimi-code/k3-turbo",
        reasoning_effort="high",
        max_runtime_seconds=600,
    )
    kb._default_spawn(task, str(workspace))

    cmd = calls[-1]["cmd"]
    assert cmd[3:] == [
        "--no-session",
        "--model", "kimi-code/k3-turbo",
        "--thinking", "high",
        "--max-time", "600",
    ]


def test_omp_spawn_reasoning_none_maps_to_off(
    kanban_home, fake_omp_binary, monkeypatch, tmp_path
):
    calls = _capture_popen(monkeypatch)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    kb._default_spawn(
        _make_omp_task(kb, reasoning_effort="none"), str(workspace)
    )
    cmd = calls[-1]["cmd"]
    assert cmd[cmd.index("--thinking") + 1] == "off"


def test_omp_preflight_missing_binary_raises_runtime_error(
    kanban_home, monkeypatch, tmp_path
):
    monkeypatch.setenv(kb.OMP_BINARY_PATH_ENV, str(tmp_path / "no-such-omp"))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with pytest.raises(RuntimeError, match="omp` executable not found"):
        kb._default_spawn(_make_omp_task(kb), str(workspace))


def test_omp_preflight_failure_trips_same_spawn_failure_accounting(
    kanban_home, monkeypatch, tmp_path
):
    """An omp pre-flight failure goes through ``_record_spawn_failure``
    exactly like the hermes/kimi legs: with ``failure_limit=1`` the first
    dispatch tick auto-blocks the card."""
    monkeypatch.setenv(kb.OMP_BINARY_PATH_ENV, str(tmp_path / "no-such-omp"))
    with kb.connect_closing() as conn:
        tid = kb.create_task(conn, title="omp card", assignee="elias", runner="omp")
        res = kb.dispatch_once(conn, failure_limit=1)
        task = kb.get_task(conn, tid)
    assert tid in res.auto_blocked
    assert task.status == "blocked"
    assert "omp" in (task.last_failure_error or "")


# ---------------------------------------------------------------------------
# kanban.default_runner resolution
# ---------------------------------------------------------------------------


def _write_config(home: Path, text: str) -> None:
    home.joinpath("config.yaml").write_text(text, encoding="utf-8")


def test_default_runner_unset_config_falls_back_to_hermes(kanban_home):
    task = _make_omp_task(kb, runner=None)
    assert kb.task_runner(task) == "hermes"
    # NULL runner + NULL template renders the historical literal.
    assert kb.render_worker_prompt(task, "/tmp/ws") == "work kanban task t_omp"


def test_default_runner_from_config(kanban_home):
    _write_config(kanban_home, "kanban:\n  default_runner: omp\n")
    task = _make_omp_task(kb, runner=None)
    assert kb.task_runner(task) == "omp"
    # The rendered prompt follows the configured runner's default template.
    prompt = kb.render_worker_prompt(task, "/tmp/ws")
    assert "hermes kanban complete t_omp --summary" in prompt
    assert prompt != "work kanban task t_omp"


def test_default_runner_task_pin_beats_config(kanban_home):
    _write_config(kanban_home, "kanban:\n  default_runner: omp\n")
    assert kb.task_runner(_make_omp_task(kb, runner="kimi")) == "kimi"
    assert kb.task_runner(_make_omp_task(kb, runner="hermes")) == "hermes"


def test_default_runner_invalid_config_falls_back_to_hermes(kanban_home):
    _write_config(kanban_home, "kanban:\n  default_runner: claude\n")
    assert kb.task_runner(_make_omp_task(kb, runner=None)) == "hermes"


def test_hermes_pin_ignores_configured_default(kanban_home):
    """A card pinned to hermes keeps its byte-identical literal prompt even
    when the dispatcher default is omp."""
    _write_config(kanban_home, "kanban:\n  default_runner: omp\n")
    task = _make_omp_task(kb, runner="hermes")
    assert kb.render_worker_prompt(task, "/tmp/ws") == "work kanban task t_omp"


# ---------------------------------------------------------------------------
# swarm_preset
# ---------------------------------------------------------------------------


def test_swarm_preset_round_trip_via_cli(kanban_home):
    out = kc.run_slash('create "swarm card" --runner omp --swarm bees --json')
    created = json.loads(out)
    assert created["swarm_preset"] == "bees"
    assert created["routed_by"] == "operator"

    shown = json.loads(kc.run_slash(f"show {created['id']} --json"))
    assert shown["task"]["swarm_preset"] == "bees"
    text = kc.run_slash(f"show {created['id']}")
    assert "swarm:     bees" in text


def test_swarm_preset_prefixes_kimi_prompt(kanban_home):
    task = _make_omp_task(kb, runner="kimi", swarm_preset="bees")
    prompt = kb.render_worker_prompt(task, "/tmp/ws")
    assert prompt.startswith("/swarm ")
    assert "hermes kanban complete t_omp --summary" in prompt


def test_swarm_preset_instruction_line_for_omp(kanban_home):
    task = _make_omp_task(kb, runner="omp", swarm_preset="bees")
    prompt = kb.render_worker_prompt(task, "/tmp/ws")
    assert prompt.startswith(
        "Execute with parallel sub-agents (swarm preset: bees).\n\n"
    )
    assert "hermes kanban complete t_omp --summary" in prompt


def test_swarm_preset_unset_or_hermes_no_prompt_change(kanban_home):
    plain = _make_omp_task(kb, runner=None)
    assert kb.render_worker_prompt(plain, "/tmp/ws") == "work kanban task t_omp"
    hermes_swarm = _make_omp_task(kb, runner="hermes", swarm_preset="bees")
    assert (
        kb.render_worker_prompt(hermes_swarm, "/tmp/ws")
        == "work kanban task t_omp"
    )


def test_omp_default_prompt_carries_lifecycle_contract(kanban_home):
    with kb.connect_closing() as conn:
        tid = kb.create_task(conn, title="t", assignee="elias", runner="omp")
        task = kb.get_task(conn, tid)
    prompt = kb.render_worker_prompt(task, "/tmp/ws")
    assert f"hermes kanban show {tid} --json" in prompt
    assert f"hermes kanban complete {tid} --summary" in prompt
    assert f"hermes kanban block {tid} --kind" in prompt
    assert "counted as crashed" in prompt
    assert "{{" not in prompt
