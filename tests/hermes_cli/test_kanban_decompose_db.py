"""Tests for kb.decompose_triage_task — the DB-layer atomic fan-out
from the triage column. LLM-free by design.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.fixture
def worktree_lane(kanban_home):
    """Install a fake profile whose manifest declares workspace_requires.

    The lane→workspace policy reads the requirement from the profile's
    distribution.yaml, so a test that wants the policy to fire has to put one
    on disk under the (tmp) profiles root. Returns the lane name.

    Clears the lru_cache on both sides of the test: the resolver memoizes per
    assignee, and a cached "no requirement" from an earlier test would silently
    disable the policy here (and vice versa).
    """
    kb._lane_workspace_requirement.cache_clear()
    lane_dir = kanban_home / "profiles" / "heavy-lane"
    lane_dir.mkdir(parents=True)
    (lane_dir / "distribution.yaml").write_text(
        "name: heavy-lane\nversion: 0.1.0\nworkspace_requires: worktree\n",
        encoding="utf-8",
    )
    yield "heavy-lane"
    kb._lane_workspace_requirement.cache_clear()


def test_decompose_child_on_code_lane_does_not_inherit_scratch(
    kanban_home, worktree_lane
):
    """A child routed to a worktree-requiring lane must not be born scratch.

    THE regression test for the 2026-07-28 blocked-card cascade: the decomposer
    picks a lane per child but inherited the triage root's scratch kind AND its
    path, so every code-lane child was born into a workspace its own lane
    refuses at spawn — and four siblings shared one directory.
    """
    with kb.connect() as conn:
        tid = _create_triage(conn, title="root")          # scratch by default
        child_ids = kb.decompose_triage_task(
            conn, tid, root_assignee="orchestrator",
            children=[
                {"title": "impl", "assignee": worktree_lane},
                {"title": "docs", "assignee": worktree_lane},
            ],
            author="decomposer",
        )
    with kb.connect() as conn:
        kids = [kb.get_task(conn, c) for c in child_ids]
    for t in kids:
        assert t.workspace_kind == "worktree"   # was 'scratch'
        assert t.workspace_path is None         # was the root's shared dir
    # Siblings must not collide: the path is derived per card at dispatch.
    assert len({t.id for t in kids}) == 2


def test_decompose_child_on_exempt_lane_stays_scratch(kanban_home):
    """A lane that declares nothing keeps the inherited scratch (builder)."""
    kb._lane_workspace_requirement.cache_clear()
    with kb.connect() as conn:
        tid = _create_triage(conn, title="root")
        child_ids = kb.decompose_triage_task(
            conn, tid, root_assignee="orchestrator",
            children=[{"title": "ops", "assignee": "cheap-ops"}],
            author="decomposer",
        )
        t = kb.get_task(conn, child_ids[0])
    assert t.workspace_kind == "scratch"
    kb._lane_workspace_requirement.cache_clear()


def test_decompose_explicit_child_workspace_beats_lane_policy(
    kanban_home, worktree_lane
):
    """Explicit per-child intent still wins over the lane policy."""
    with kb.connect() as conn:
        tid = _create_triage(conn, title="root")
        child_ids = kb.decompose_triage_task(
            conn, tid, root_assignee="orchestrator",
            children=[{"title": "pinned", "assignee": worktree_lane,
                       "workspace_kind": "dir",
                       "workspace_path": "/other/repo"}],
            author="decomposer",
        )
        t = kb.get_task(conn, child_ids[0])
    assert t.workspace_kind == "dir"
    assert t.workspace_path == "/other/repo"


def _create_triage(conn, title="rough idea", body=None, assignee=None, tenant=None):
    return kb.create_task(
        conn,
        title=title,
        body=body,
        assignee=assignee,
        tenant=tenant,
        triage=True,
    )


def test_decompose_creates_children_and_promotes_root(kanban_home):
    with kb.connect() as conn:
        tid = _create_triage(conn, title="ship a feature")
        assert kb.get_task(conn, tid).status == "triage"

    children = [
        {"title": "research", "body": "look at prior art", "assignee": "researcher", "parents": []},
        {"title": "build it", "body": "write code", "assignee": "engineer", "parents": [0]},
    ]
    with kb.connect() as conn:
        child_ids = kb.decompose_triage_task(
            conn,
            tid,
            root_assignee="orchestrator",
            children=children,
            author="decomposer",
        )
    assert child_ids is not None
    assert len(child_ids) == 2

    with kb.connect() as conn:
        root = kb.get_task(conn, tid)
        c0 = kb.get_task(conn, child_ids[0])
        c1 = kb.get_task(conn, child_ids[1])

    # Root flipped to todo with orchestrator assignee, gated by children.
    assert root.status == "todo"
    assert root.assignee == "orchestrator"
    # First child has no internal parents → ready on recompute_ready.
    assert c0.status == "ready"
    assert c0.assignee == "researcher"
    # Second child has parents=[0] → stays in todo until c0 completes.
    assert c1.status == "todo"
    assert c1.assignee == "engineer"


def test_decompose_records_audit_comment_and_event(kanban_home):
    with kb.connect() as conn:
        tid = _create_triage(conn)
        child_ids = kb.decompose_triage_task(
            conn,
            tid,
            root_assignee="orch",
            children=[{"title": "task A", "assignee": "researcher"}],
            author="alice",
        )
    assert child_ids is not None

    with kb.connect() as conn:
        comments = kb.list_comments(conn, tid)
        events = kb.list_events(conn, tid)

    assert any("Decomposed into" in (c.body or "") for c in comments)
    assert any(ev.kind == "decomposed" for ev in events)




