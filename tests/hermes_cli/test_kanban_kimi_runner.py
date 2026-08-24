"""Spec 042 Phase A-lite + B — kimi as a first-class kanban card runner.

Covers the additive execution-contract columns
(``runner`` / ``prompt_template`` / ``permission_mode`` / ``routed_by``), the
``kanban create`` / ``kanban show`` round-trip, kickoff prompt rendering
(hermes default stays byte-identical to the historical literal), and the
kimi spawn leg in ``_default_spawn`` (argv shape, env pins, pre-flight
failure accounting).
"""

from __future__ import annotations

import json
import sqlite3
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
def fake_kimi_binary(tmp_path, monkeypatch):
    """A stand-in kimi CLI that answers ``--version`` successfully."""
    binary = tmp_path / "kimi"
    binary.write_text("#!/bin/sh\necho kimi 0.0.0-test\n", encoding="utf-8")
    binary.chmod(0o755)
    monkeypatch.setenv(kb.KIMI_BINARY_PATH_ENV, str(binary))
    return binary


def _legacy_db_without_spec042_columns(path: Path) -> None:
    """Write a kanban DB whose ``tasks`` table predates spec 042."""
    conn = sqlite3.connect(str(path))
    conn.execute("""
        CREATE TABLE tasks (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            body TEXT,
            assignee TEXT,
            status TEXT NOT NULL,
            priority INTEGER DEFAULT 0,
            created_by TEXT,
            created_at INTEGER NOT NULL,
            started_at INTEGER,
            completed_at INTEGER,
            workspace_kind TEXT NOT NULL DEFAULT 'scratch',
            workspace_path TEXT,
            claim_lock TEXT,
            claim_expires INTEGER,
            tenant TEXT,
            result TEXT,
            idempotency_key TEXT,
            spawn_failures INTEGER NOT NULL DEFAULT 0,
            worker_pid INTEGER,
            last_spawn_error TEXT
        )
    """)
    # task_events is required: _migrate_add_optional_columns also runs a
    # PRAGMA on it to back-fill the run_id column and raises
    # OperationalError if the table is absent.
    conn.execute("""
        CREATE TABLE task_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id TEXT NOT NULL,
            kind TEXT NOT NULL,
            payload TEXT,
            created_at INTEGER NOT NULL
        )
    """)
    conn.execute(
        "INSERT INTO tasks (id, title, status, created_at) "
        "VALUES ('legacy', 'old card', 'ready', 1)"
    )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Schema (Phase A-lite)
# ---------------------------------------------------------------------------


def test_spec042_columns_migrated_onto_legacy_board(tmp_path):
    """Legacy boards gain the execution columns as NULL — existing cards
    keep their exact pre-spec behaviour (NULL runner → hermes leg)."""
    db_path = tmp_path / "legacy.db"
    _legacy_db_without_spec042_columns(db_path)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    kb._migrate_add_optional_columns(conn)

    cols = {r[1] for r in conn.execute("PRAGMA table_info(tasks)")}
    for col in ("runner", "prompt_template", "permission_mode", "routed_by"):
        assert col in cols, f"migration must add tasks.{col}"

    row = conn.execute("SELECT * FROM tasks WHERE id = 'legacy'").fetchone()
    task = kb.Task.from_row(row)
    assert task.runner is None
    assert task.prompt_template is None
    assert task.permission_mode is None
    assert task.routed_by is None

    # Idempotent second run must not raise.
    kb._migrate_add_optional_columns(conn)
    conn.close()


def test_create_and_show_round_trip_execution_fields(kanban_home):
    """`kanban create --runner/--prompt/--yolo` persists the execution
    contract and `kanban show --json` exposes it."""
    out = kc.run_slash(
        'create "kimi card" --runner kimi --yolo '
        '--prompt "work {{task_id}} in {{workspace_path}}" --json'
    )
    created = json.loads(out)
    assert created["runner"] == "kimi"
    assert created["permission_mode"] == "yolo"
    assert created["prompt_template"] == "work {{task_id}} in {{workspace_path}}"
    # Pinning execution fields at filing stamps routed_by=operator so later
    # curator routing treats them as operator-decided.
    assert created["routed_by"] == "operator"

    shown = json.loads(kc.run_slash(f"show {created['id']} --json"))
    task = shown["task"]
    assert task["runner"] == "kimi"
    assert task["permission_mode"] == "yolo"
    assert task["prompt_template"] == "work {{task_id}} in {{workspace_path}}"
    assert task["routed_by"] == "operator"


def test_show_text_prints_execution_block(kanban_home):
    out = kc.run_slash('create "plain card" --runner kimi --yolo')
    tid = out.split()[1]
    text = kc.run_slash(f"show {tid}")
    assert "runner:    kimi" in text
    assert "permission: yolo" in text
    assert "routed-by: operator" in text

    out2 = kc.run_slash('create "hermes card"')
    tid2 = out2.split()[1]
    text2 = kc.run_slash(f"show {tid2}")
    assert "runner:    hermes (default)" in text2
    assert "  permission:" not in text2


def test_create_rejects_unknown_runner(kanban_home):
    out = kc.run_slash('create "bad card" --runner bogus')
    assert "usage error" in out
    with kb.connect_closing() as conn:
        assert kb.list_tasks(conn) == []


def test_create_task_rejects_unknown_runner(kanban_home):
    with kb.connect_closing() as conn:
        with pytest.raises(ValueError, match="runner must be one of"):
            kb.create_task(conn, title="bad", runner="claude")
        with pytest.raises(ValueError, match="permission_mode must be one of"):
            kb.create_task(conn, title="bad", permission_mode="unsafe")


# ---------------------------------------------------------------------------
# Prompt templates (spec 042 §4)
# ---------------------------------------------------------------------------


def test_hermes_default_prompt_renders_byte_identical(kanban_home):
    """A NULL prompt_template on a hermes card must render exactly the
    historical literal — pre-spec cards observe no change."""
    with kb.connect_closing() as conn:
        tid = kb.create_task(conn, title="t", assignee="elias")
        task = kb.get_task(conn, tid)
    assert task.runner is None
    assert kb.render_worker_prompt(task, "/tmp/ws") == f"work kanban task {tid}"

    task_explicit = kb.Task(
        id="t_x", title="t", body=None, assignee="elias", status="ready",
        priority=0, created_by="test", created_at=1, started_at=None,
        completed_at=None, workspace_kind="dir", workspace_path=None,
        claim_lock=None, claim_expires=None, tenant=None, runner="hermes",
    )
    assert kb.render_worker_prompt(task_explicit, "/tmp/ws") == "work kanban task t_x"


def test_kimi_default_prompt_carries_lifecycle_contract(kanban_home):
    """The kimi default template must spell out the whole lifecycle the
    in-process kanban toolset would otherwise carry: read the card first,
    work in the launch cwd, and end with a terminal CLI call — the signal
    ``detect_crashed_workers`` reads (task leaves ``running``)."""
    with kb.connect_closing() as conn:
        tid = kb.create_task(conn, title="t", assignee="elias", runner="kimi")
        task = kb.get_task(conn, tid)
    prompt = kb.render_worker_prompt(task, "/tmp/ws")
    assert f"hermes kanban show {tid} --json" in prompt
    assert f"hermes kanban complete {tid} --summary" in prompt
    assert f"hermes kanban block {tid} --kind" in prompt
    assert "counted as crashed" in prompt
    assert "{{" not in prompt  # every placeholder substituted


def test_prompt_template_substitutes_all_fields(kanban_home):
    with kb.connect_closing() as conn:
        tid = kb.create_task(
            conn,
            title="the title",
            body="the body",
            assignee="elias",
            runner="kimi",
            prompt_template=(
                "{{task_id}}|{{title}}|{{body}}|{{branch}}|{{workspace_path}}"
            ),
        )
        task = kb.get_task(conn, tid)
        task.branch_name = "wt/x"
    prompt = kb.render_worker_prompt(task, "/tmp/ws")
    assert prompt == f"{tid}|the title|the body|wt/x|/tmp/ws"


# ---------------------------------------------------------------------------
# Kimi spawn leg (Phase B)
# ---------------------------------------------------------------------------


def _make_kimi_task(kb_mod, **overrides):
    fields = {
        "id": "t_kimi",
        "title": "kimi card",
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
        "runner": "kimi",
    }
    fields.update(overrides)
    return kb_mod.Task(**fields)


class _FakeProc:
    """Popen stand-in compatible with both the fire-and-forget spawn and
    the ``subprocess.run`` context-manager protocol the kimi ``--version``
    pre-flight uses."""

    pid = 4243
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
    kimi pre-flight's ``--version`` call lands first, the worker spawn last."""
    calls = []

    def fake_popen(cmd, *args, **kwargs):
        calls.append({"cmd": list(cmd), "kwargs": kwargs})
        return _FakeProc(list(cmd))

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    return calls


def test_kimi_spawn_argv_env_cwd_and_log(
    kanban_home, fake_kimi_binary, monkeypatch, tmp_path
):
    calls = _capture_popen(monkeypatch)

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    task = _make_kimi_task(kb, model_override="kimi-for-coding")
    pid = kb._default_spawn(task, str(workspace))

    assert pid == 4243
    # First call is the --version pre-flight against the configured binary.
    assert calls[0]["cmd"] == [str(fake_kimi_binary), "--version"]
    captured = calls[-1]
    cmd = captured["cmd"]
    assert cmd[0] == str(fake_kimi_binary)
    assert cmd[1] == "-p"
    # Default kimi prompt for the card, rendered with the task id.
    assert cmd[2] == kb.render_worker_prompt(task, str(workspace))
    assert "hermes kanban complete t_kimi --summary" in cmd[2]
    assert cmd[3:5] == ["--output-format=stream-json", "--model"]
    assert cmd[5] == "kimi-for-coding"
    # Same env pins the hermes leg sets — the worker's terminal
    # `hermes kanban complete/block` calls resolve board + run from these.
    env = dict(captured["kwargs"].get("env") or {})
    assert env["HERMES_KANBAN_TASK"] == "t_kimi"
    assert env["HERMES_KANBAN_RUN_ID"] == "7"
    assert env["HERMES_KANBAN_CLAIM_LOCK"] == "lock"
    assert env["HERMES_KANBAN_BOARD"]
    assert env["HERMES_KANBAN_DB"]
    assert env["HERMES_KANBAN_WORKSPACES_ROOT"]
    assert env["HERMES_SESSION_SOURCE"] == "kanban"
    assert captured["kwargs"].get("cwd") == str(workspace)
    # Output still lands in the per-task board log.
    log_path = kb.worker_logs_dir() / "t_kimi.log"
    assert log_path.is_file()


def test_kimi_spawn_omits_model_flag_without_override(
    kanban_home, fake_kimi_binary, monkeypatch, tmp_path
):
    calls = _capture_popen(monkeypatch)

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    kb._default_spawn(_make_kimi_task(kb), str(workspace))

    cmd = calls[-1]["cmd"]
    assert cmd[0] == str(fake_kimi_binary)
    assert cmd[1] == "-p"
    assert cmd[3:] == ["--output-format=stream-json"]


def test_kimi_preflight_missing_binary_raises_runtime_error(
    kanban_home, monkeypatch, tmp_path
):
    monkeypatch.setenv(kb.KIMI_BINARY_PATH_ENV, str(tmp_path / "no-such-kimi"))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with pytest.raises(RuntimeError, match="kimi` executable not found"):
        kb._default_spawn(_make_kimi_task(kb), str(workspace))


def test_kimi_preflight_prompt_over_argv_budget(
    kanban_home, fake_kimi_binary, tmp_path
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    task = _make_kimi_task(
        kb, prompt_template="x" * (kb.WORKER_PROMPT_ARGV_MAX_BYTES + 1)
    )
    with pytest.raises(RuntimeError, match="argv budget"):
        kb._default_spawn(task, str(workspace))


def test_kimi_preflight_failure_trips_same_spawn_failure_accounting(
    kanban_home, monkeypatch, tmp_path
):
    """A kimi pre-flight failure goes through ``_record_spawn_failure``
    exactly like the hermes leg's missing-binary RuntimeError: with
    ``failure_limit=1`` the first dispatch tick auto-blocks the card."""
    monkeypatch.setenv(kb.KIMI_BINARY_PATH_ENV, str(tmp_path / "no-such-kimi"))
    with kb.connect_closing() as conn:
        tid = kb.create_task(conn, title="kimi card", assignee="elias", runner="kimi")
        res = kb.dispatch_once(conn, failure_limit=1)
        task = kb.get_task(conn, tid)
    assert tid in res.auto_blocked
    assert task.status == "blocked"
    assert "kimi" in (task.last_failure_error or "")
