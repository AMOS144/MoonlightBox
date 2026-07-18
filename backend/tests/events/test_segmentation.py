from datetime import datetime, timedelta

from moonlightbox.imports.types import ImportedMessage, MessageKind


def make_message(message_id: str, hour: int) -> ImportedMessage:
    return ImportedMessage(
        source_id=message_id,
        timestamp=datetime(2026, 1, 1, hour),
        sender="甲",
        kind=MessageKind.TEXT,
        content=message_id,
        raw={},
    )


def test_long_gap_creates_stable_episode_boundary() -> None:
    from moonlightbox.events.segmentation import segment

    episodes = segment(
        [
            make_message("m1", 1),
            make_message("m2", 2),
            make_message("m3", 10),
            make_message("m4", 11),
        ],
        gap=timedelta(hours=6),
    )

    assert [episode.message_ids for episode in episodes] == [
        ["m1", "m2"],
        ["m3", "m4"],
    ]
    assert episodes[1].boundary_reasons == ["time_gap"]
