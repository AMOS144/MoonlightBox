from pathlib import Path

from moonlightbox.db import Database
from moonlightbox.events.schemas import ReviewedEvent
from sqlalchemy.orm import Session


def test_event_edits_create_revision_history(tmp_path: Path) -> None:
    from moonlightbox.events.models import AnalysisRevision, EventNode
    from moonlightbox.events.service import EventService

    database = Database(f"sqlite:///{tmp_path / 'events.db'}")
    EventNode.metadata.create_all(database.engine)
    reviewed = ReviewedEvent(
        type="conflict",
        start_message_id="m1",
        end_message_id="m2",
        before_state="正常交流",
        after_state="关系紧张",
        emotion_labels=["生气"],
        topic="误解",
        conflict_level=4,
        importance=0.8,
        reason="语气升级",
        evidence_ids=["m1", "m2"],
    )

    with Session(database.engine) as session:
        service = EventService(session)
        event = service.create("project-1", reviewed, "analysis-v1", "prompt-v1")
        revised = service.revise(event.id, {"type": "cold_war"}, "人工修改")
        history = service.revisions(event.id)

    assert revised.type == "cold_war"
    assert [item.revision_number for item in history] == [1, 2]
    assert all(isinstance(item, AnalysisRevision) for item in history)
