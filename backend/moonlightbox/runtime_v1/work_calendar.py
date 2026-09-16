"""中国工作日历：在线年度数据、持久缓存与保守降级。

日历是 Runtime 的确定性基础设施，不是交给 LLM 自主调用的工具。服务只访问代码内
允许的 holiday-cn 静态地址，校验年度 JSON 后缓存；网络失败时复用最后一次有效缓存，
没有可信数据时明确返回 ``unverified``，绝不把周一至周五冒充成法定工作日。
"""

from __future__ import annotations

import os
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

import httpx
from pydantic import BaseModel, ConfigDict, Field

from moonlightbox.config import Settings

_CACHE_PROTOCOL = "moonlightbox-work-calendar-cache-v1"
_MAX_DOCUMENT_BYTES = 256 * 1024
_HOLIDAY_CN_URLS = (
    "https://cdn.jsdelivr.net/gh/NateScarlet/holiday-cn@master/{year}.json",
    "https://raw.githubusercontent.com/NateScarlet/holiday-cn/master/{year}.json",
)
_WEEKDAY_NAMES = (
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
)


class WorkCalendar(Protocol):
    """供上下文组装器消费的最小日历接口。"""

    def day_features(self, target_date: date) -> dict[str, Any]: ...


class HolidayCnDay(BaseModel):
    """holiday-cn 的单日覆盖项；false 表示周末调休上班。"""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    name: str = Field(min_length=1, max_length=80)
    date: date
    is_off_day: bool = Field(alias="isOffDay")


class HolidayCnDocument(BaseModel):
    """仅接收规划需要的年度字段，忽略上游的 JSON Schema 元数据。"""

    model_config = ConfigDict(extra="ignore")

    year: int = Field(ge=2000, le=2200)
    papers: list[str] = Field(default_factory=list, max_length=8)
    days: list[HolidayCnDay] = Field(default_factory=list, max_length=400)


class CalendarCache(BaseModel):
    """带来源与刷新元数据的本地缓存信封。"""

    model_config = ConfigDict(extra="forbid")

    protocol_version: str
    fetched_at: datetime
    dataset_url: str
    etag: str | None = None
    last_modified: str | None = None
    document: HolidayCnDocument


class UnverifiedWorkCalendar:
    """未配置在线日历时的安全实现，只暴露自然星期，不断言工作状态。"""

    def day_features(self, target_date: date) -> dict[str, Any]:
        return _unverified_features(target_date, reason="calendar_not_configured")


class ChinaWorkCalendarService:
    """从 holiday-cn 获取国务院节假日覆盖项并按年度缓存。"""

    def __init__(
        self,
        *,
        cache_dir: Path,
        enabled: bool = True,
        refresh_interval: timedelta = timedelta(hours=24),
        timeout_seconds: float = 5.0,
        client: httpx.Client | None = None,
        wall_now: Callable[[], datetime] | None = None,
    ) -> None:
        self.cache_dir = cache_dir
        self.enabled = enabled
        self.refresh_interval = refresh_interval
        self.timeout_seconds = timeout_seconds
        self._client = client
        self._wall_now = wall_now or (lambda: datetime.now(UTC))

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        client: httpx.Client | None = None,
    ) -> ChinaWorkCalendarService:
        cache_dir = settings.work_calendar_cache_dir
        if cache_dir is None:
            cache_dir = settings.data_dir / "calendar-cache" / "cn"
        return cls(
            cache_dir=cache_dir,
            enabled=settings.work_calendar_enabled,
            refresh_interval=timedelta(hours=settings.work_calendar_refresh_hours),
            timeout_seconds=settings.work_calendar_timeout_seconds,
            client=client,
        )

    def day_features(self, target_date: date) -> dict[str, Any]:
        if not self.enabled:
            return _unverified_features(target_date, reason="calendar_disabled")
        cached = self._load_cache(target_date.year)
        now = _as_utc(self._wall_now())
        if cached is not None and now - _as_utc(cached.fetched_at) < self.refresh_interval:
            return _resolved_features(target_date, cached, stale=False)
        refreshed = self._refresh(target_date.year, cached, now)
        if refreshed is not None:
            return _resolved_features(target_date, refreshed, stale=False)
        if cached is not None:
            return _resolved_features(target_date, cached, stale=True)
        return _unverified_features(target_date, reason="calendar_unavailable")

    def _cache_path(self, year: int) -> Path:
        return self.cache_dir / f"{year}.json"

    def _load_cache(self, year: int) -> CalendarCache | None:
        try:
            cache = CalendarCache.model_validate_json(
                self._cache_path(year).read_text(encoding="utf-8")
            )
            if cache.protocol_version != _CACHE_PROTOCOL:
                raise ValueError("工作日历缓存协议不受支持")
            _validate_document(cache.document, year)
        except (OSError, ValueError):
            return None
        return cache

    def _refresh(
        self,
        year: int,
        cached: CalendarCache | None,
        now: datetime,
    ) -> CalendarCache | None:
        owns_client = self._client is None
        client = self._client or httpx.Client(timeout=self.timeout_seconds)
        try:
            for template in _HOLIDAY_CN_URLS:
                url = template.format(year=year)
                headers = _conditional_headers(cached, url)
                try:
                    response = client.get(url, headers=headers, timeout=self.timeout_seconds)
                    if response.status_code == 304 and cached is not None:
                        refreshed = cached.model_copy(update={"fetched_at": now})
                    else:
                        response.raise_for_status()
                        if len(response.content) > _MAX_DOCUMENT_BYTES:
                            continue
                        document = HolidayCnDocument.model_validate_json(response.content)
                        _validate_document(document, year)
                        refreshed = CalendarCache(
                            protocol_version=_CACHE_PROTOCOL,
                            fetched_at=now,
                            dataset_url=url,
                            etag=response.headers.get("etag"),
                            last_modified=response.headers.get("last-modified"),
                            document=document,
                        )
                except (httpx.HTTPError, ValueError):
                    continue
                self._store_cache(refreshed)
                return refreshed
        finally:
            if owns_client:
                client.close()
        return None

    def _store_cache(self, cache: CalendarCache) -> None:
        path = self._cache_path(cache.document.year)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary.write_text(cache.model_dump_json(indent=2, by_alias=True), encoding="utf-8")
            os.replace(temporary, path)
        except OSError:
            # 只读部署仍可使用本次在线结果；缓存失败不能把日历降级成不可用。
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def _conditional_headers(cached: CalendarCache | None, url: str) -> dict[str, str]:
    if cached is None or cached.dataset_url != url:
        return {}
    headers: dict[str, str] = {}
    if cached.etag:
        headers["If-None-Match"] = cached.etag
    if cached.last_modified:
        headers["If-Modified-Since"] = cached.last_modified
    return headers


def _validate_document(document: HolidayCnDocument, expected_year: int) -> None:
    if document.year != expected_year:
        raise ValueError("工作日历年份与请求不一致")
    dates: set[date] = set()
    for item in document.days:
        if item.date.year != expected_year or item.date in dates:
            raise ValueError("工作日历包含越界或重复日期")
        dates.add(item.date)
    for paper in document.papers:
        try:
            url = httpx.URL(paper)
        except httpx.InvalidURL:
            raise ValueError("工作日历公告来源地址无效") from None
        host = (url.host or "").lower()
        if url.scheme != "https" or (host != "gov.cn" and not host.endswith(".gov.cn")):
            raise ValueError("工作日历公告来源不是允许的政府域名")


def _resolved_features(
    target_date: date,
    cache: CalendarCache,
    *,
    stale: bool,
) -> dict[str, Any]:
    override = next((item for item in cache.document.days if item.date == target_date), None)
    natural_weekday = target_date.weekday() < 5
    if override is not None and override.is_off_day:
        is_workday = False
        day_type = "public_holiday"
    elif override is not None:
        is_workday = True
        day_type = "adjusted_workday"
    elif natural_weekday:
        is_workday = True
        day_type = "regular_workday"
    else:
        is_workday = False
        day_type = "regular_rest_day"
    return {
        "date": target_date.isoformat(),
        "weekday": target_date.weekday(),
        "weekday_name": _WEEKDAY_NAMES[target_date.weekday()],
        "is_weekday": natural_weekday,
        "is_workday": is_workday,
        "day_type": day_type,
        "holiday": override.name if override is not None and override.is_off_day else None,
        "holiday_name": override.name if override is not None else None,
        "calendar_region": "CN",
        "calendar_verified": True,
        "calendar_stale": stale,
        "calendar_reason": None,
        "calendar_source": {
            "provider": "holiday-cn",
            "dataset_url": cache.dataset_url,
            "papers": list(cache.document.papers),
            "fetched_at": _as_utc(cache.fetched_at).isoformat(),
            "etag": cache.etag,
        },
    }


def _unverified_features(target_date: date, *, reason: str) -> dict[str, Any]:
    return {
        "date": target_date.isoformat(),
        "weekday": target_date.weekday(),
        "weekday_name": _WEEKDAY_NAMES[target_date.weekday()],
        "is_weekday": target_date.weekday() < 5,
        "is_workday": None,
        "day_type": "unverified",
        "holiday": None,
        "holiday_name": None,
        "calendar_region": "CN",
        "calendar_verified": False,
        "calendar_stale": False,
        "calendar_reason": reason,
        "calendar_source": None,
    }


def _as_utc(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
