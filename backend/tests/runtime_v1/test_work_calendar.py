from datetime import UTC, date, datetime, timedelta

import httpx
from moonlightbox.runtime_v1.work_calendar import ChinaWorkCalendarService

_PAPER = "https://www.gov.cn/zhengce/zhengceku/202511/content_7047091.htm"


def _calendar_response() -> httpx.Response:
    return httpx.Response(
        200,
        headers={"etag": '"2026-v1"'},
        json={
            "year": 2026,
            "papers": [_PAPER],
            "days": [
                {"name": "劳动节", "date": "2026-05-04", "isOffDay": True},
                {"name": "劳动节", "date": "2026-05-09", "isOffDay": False},
            ],
        },
    )


def test_online_calendar_distinguishes_adjusted_workday_from_weekday(tmp_path) -> None:
    requests = 0

    def respond(_: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return _calendar_response()

    now = datetime(2026, 9, 9, tzinfo=UTC)
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        calendar = ChinaWorkCalendarService(
            cache_dir=tmp_path,
            client=client,
            wall_now=lambda: now,
        )
        adjusted = calendar.day_features(date(2026, 5, 9))
        # 同一个实例再次读取时命中年度缓存，不重复访问网络。
        holiday = calendar.day_features(date(2026, 5, 4))

    assert requests == 1
    assert adjusted["weekday_name"] == "Saturday"
    assert adjusted["is_weekday"] is False
    assert adjusted["is_workday"] is True
    assert adjusted["day_type"] == "adjusted_workday"
    assert adjusted["calendar_verified"] is True
    assert holiday["weekday_name"] == "Monday"
    assert holiday["is_workday"] is False
    assert holiday["day_type"] == "public_holiday"
    assert (tmp_path / "2026.json").is_file()


def test_stale_valid_cache_survives_online_failure(tmp_path) -> None:
    first_now = datetime(2026, 9, 1, tzinfo=UTC)
    with httpx.Client(
        transport=httpx.MockTransport(lambda _: _calendar_response())
    ) as client:
        ChinaWorkCalendarService(
            cache_dir=tmp_path,
            client=client,
            wall_now=lambda: first_now,
        ).day_features(date(2026, 5, 9))

    def offline(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline", request=request)

    with httpx.Client(transport=httpx.MockTransport(offline)) as client:
        result = ChinaWorkCalendarService(
            cache_dir=tmp_path,
            client=client,
            refresh_interval=timedelta(hours=1),
            wall_now=lambda: first_now + timedelta(days=2),
        ).day_features(date(2026, 5, 9))

    assert result["is_workday"] is True
    assert result["calendar_verified"] is True
    assert result["calendar_stale"] is True


def test_no_cache_and_online_failure_is_explicitly_unverified(tmp_path) -> None:
    def offline(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline", request=request)

    with httpx.Client(transport=httpx.MockTransport(offline)) as client:
        result = ChinaWorkCalendarService(cache_dir=tmp_path, client=client).day_features(
            date(2027, 1, 4)
        )

    assert result["weekday_name"] == "Monday"
    assert result["is_weekday"] is True
    assert result["is_workday"] is None
    assert result["day_type"] == "unverified"
    assert result["calendar_reason"] == "calendar_unavailable"
