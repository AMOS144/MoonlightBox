from datetime import date

import pytest
from moonlightbox.jobs.models import Job
from moonlightbox.node_investigation.schemas import InputArgs, ScopeArgs
from moonlightbox.node_investigation.tools import InvestigationTools
from sqlalchemy.orm import Session
from test_investigation import candidate
from test_investigation import setup as _source_setup

node_setup = _source_setup


@pytest.fixture
def setup(node_setup):
    return node_setup


def test_explicit_cursor_replay_does_not_skip_unseen_page(setup):
    _, store = setup
    job = store.get().state["job_id"]
    tools = InvestigationTools(store, job, None)
    first = tools.read(cursor=0, limit=2)
    # 模拟工具写进度后、LangGraph 存回执前重启：重放原参数仍得到原页。
    restarted = InvestigationTools(store, job, None)
    assert restarted.read(cursor=0, limit=2) == first
    assert store.view()["read_count"] == 2
    assert restarted.read(cursor=2)["messages"][0]["message_ref"] == "m2"


def test_scope_reports_skips_and_actual_coverage(setup):
    database, store = setup
    with Session(database.engine) as session:
        session.get(Job, store.get().state["job_id"]).status = "cancelled"
        session.commit()
    store.mutate(lambda state, _: state.update(status="paused"))
    store.set_scope(ScopeArgs(start_date=date(2026, 5, 6), end_date=date(2026, 5, 7)))
    view = store.view()
    assert view["skipped_intervals"] == [[0, 2]]
    assert view["read_count"] == 0 and view["range_count"] == 2
    tools = InvestigationTools(store, view["job_id"], None)
    assert tools.read(cursor=2)["end_of_record"]
    assert store.view()["read_count"] == 2


def test_unread_user_correction_cannot_be_overwritten(setup):
    _, store = setup
    tools = InvestigationTools(store, store.get().state["job_id"], None)
    c = tools.upsert(**candidate())
    store.add_input(
        InputArgs(request_id="correction", kind="correction", text="并非旅行，是回老家")
    )
    with pytest.raises(Exception, match="尚未读到"):
        tools.upsert(**candidate(candidate_ref=c["id"]))
    assert store.view()["candidates"][0]["revision"] == 1
