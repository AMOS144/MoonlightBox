import hashlib
import json
from dataclasses import dataclass, field
from typing import Protocol

import httpx


@dataclass(frozen=True, slots=True)
class Coordinate:
    longitude: float
    latitude: float
    crs: str


@dataclass(frozen=True, slots=True)
class MapPlace:
    provider: str
    provider_place_id: str
    name: str
    address: str | None
    category: str | None
    coordinate: Coordinate | None
    administrative: dict[str, object] = field(default_factory=dict)
    parent_place_id: str | None = None
    raw_hash: str | None = None


@dataclass(frozen=True, slots=True)
class TravelMeasurement:
    origin: Coordinate
    destination: Coordinate
    mode: str
    distance_meters: float | None
    duration_seconds: float | None


class MapProvider(Protocol):
    name: str

    def search_poi(
        self,
        query: str,
        *,
        city_hint: str | None,
        adcode_hint: str | None,
        near: Coordinate | None,
        category: str | None,
        limit: int,
    ) -> list[MapPlace]: ...

    def geocode(
        self,
        address: str,
        *,
        city_hint: str | None,
        limit: int,
    ) -> list[MapPlace]: ...

    def measure(
        self,
        origins: list[Coordinate],
        destination: Coordinate,
        *,
        mode: str,
    ) -> list[TravelMeasurement]: ...

    def close(self) -> None: ...


class NullMapProvider:
    name = "none"

    def search_poi(
        self,
        query: str,
        *,
        city_hint: str | None,
        adcode_hint: str | None,
        near: Coordinate | None,
        category: str | None,
        limit: int,
    ) -> list[MapPlace]:
        return []

    def geocode(
        self,
        address: str,
        *,
        city_hint: str | None,
        limit: int,
    ) -> list[MapPlace]:
        return []

    def measure(
        self,
        origins: list[Coordinate],
        destination: Coordinate,
        *,
        mode: str,
    ) -> list[TravelMeasurement]:
        return []

    def close(self) -> None:
        return None


class AmapWebServiceProvider:
    name = "amap"

    def __init__(
        self,
        key: str,
        *,
        base_url: str = "https://restapi.amap.com",
        timeout_seconds: float = 15.0,
        client: httpx.Client | None = None,
    ) -> None:
        if not key.strip():
            raise ValueError("高德 Web 服务 key 不能为空")
        self._key = key
        self._base_url = base_url.rstrip("/")
        self._owns_client = client is None
        self._client = client or httpx.Client(timeout=timeout_seconds)

    def search_poi(
        self,
        query: str,
        *,
        city_hint: str | None,
        adcode_hint: str | None,
        near: Coordinate | None,
        category: str | None,
        limit: int,
    ) -> list[MapPlace]:
        if not query.strip() or limit <= 0:
            return []
        city = adcode_hint or city_hint
        params: dict[str, str | int] = {
            "key": self._key,
            "keywords": query.strip(),
            "offset": min(limit, 25),
            "page": 1,
            "extensions": "all",
        }
        if city:
            params.update({"city": city, "citylimit": "true"})
        if category:
            params["types"] = category
        path = "/v3/place/text"
        if near is not None and near.crs.upper() == "GCJ-02":
            path = "/v3/place/around"
            params["location"] = f"{near.longitude:.6f},{near.latitude:.6f}"
            params["radius"] = 20_000
            params["sortrule"] = "weight"
        payload = self._get(path, params)
        pois = payload.get("pois")
        if not isinstance(pois, list):
            return []
        return [place for item in pois[:limit] if (place := _parse_amap_poi(item)) is not None]

    def geocode(
        self,
        address: str,
        *,
        city_hint: str | None,
        limit: int,
    ) -> list[MapPlace]:
        if not address.strip() or limit <= 0:
            return []
        params: dict[str, str | int] = {
            "key": self._key,
            "address": address.strip(),
        }
        if city_hint:
            params["city"] = city_hint
        payload = self._get("/v3/geocode/geo", params)
        geocodes = payload.get("geocodes")
        if not isinstance(geocodes, list):
            return []
        places: list[MapPlace] = []
        for item in geocodes[:limit]:
            if not isinstance(item, dict):
                continue
            coordinate = _parse_coordinate(item.get("location"))
            formatted_address = _string(item.get("formatted_address"))
            if coordinate is None or formatted_address is None:
                continue
            raw_hash = _payload_hash(item)
            places.append(
                MapPlace(
                    provider=self.name,
                    provider_place_id=f"geocode:{raw_hash}",
                    name=formatted_address,
                    address=formatted_address,
                    category="address",
                    coordinate=coordinate,
                    administrative={
                        key: value
                        for key in ("country", "province", "city", "district", "adcode")
                        if (value := _string(item.get(key))) is not None
                    },
                    raw_hash=raw_hash,
                )
            )
        return places

    def measure(
        self,
        origins: list[Coordinate],
        destination: Coordinate,
        *,
        mode: str,
    ) -> list[TravelMeasurement]:
        compatible = [item for item in origins if item.crs.upper() == "GCJ-02"]
        if not compatible or destination.crs.upper() != "GCJ-02":
            return []
        params: dict[str, str | int] = {
            "key": self._key,
            "origins": "|".join(
                f"{item.longitude:.6f},{item.latitude:.6f}" for item in compatible[:100]
            ),
            "destination": f"{destination.longitude:.6f},{destination.latitude:.6f}",
            "type": 0 if mode == "straight" else 3 if mode == "walking" else 1,
        }
        payload = self._get("/v3/distance", params)
        results = payload.get("results")
        if not isinstance(results, list):
            return []
        measurements: list[TravelMeasurement] = []
        for origin, item in zip(compatible, results, strict=False):
            if not isinstance(item, dict):
                continue
            measurements.append(
                TravelMeasurement(
                    origin=origin,
                    destination=destination,
                    mode=mode,
                    distance_meters=_float(item.get("distance")),
                    duration_seconds=_float(item.get("duration")),
                )
            )
        return measurements

    def _get(self, path: str, params: dict[str, str | int]) -> dict[str, object]:
        response = self._client.get(f"{self._base_url}{path}", params=params)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict) or str(payload.get("status")) != "1":
            info = (
                str(payload.get("info", "地图服务返回异常"))
                if isinstance(payload, dict)
                else "地图服务返回异常"
            )
            raise RuntimeError(info)
        return payload

    def close(self) -> None:
        if self._owns_client:
            self._client.close()


def _parse_amap_poi(value: object) -> MapPlace | None:
    if not isinstance(value, dict):
        return None
    place_id = _string(value.get("id"))
    name = _string(value.get("name"))
    if place_id is None or name is None:
        return None
    return MapPlace(
        provider="amap",
        provider_place_id=place_id,
        name=name,
        address=_string(value.get("address")),
        category=_string(value.get("type")),
        coordinate=_parse_coordinate(value.get("location")),
        administrative={
            key: item
            for key in ("pcode", "pname", "citycode", "cityname", "adcode", "adname")
            if (item := _string(value.get(key))) is not None
        },
        parent_place_id=_string(value.get("parent")),
        raw_hash=_payload_hash(value),
    )


def _parse_coordinate(value: object) -> Coordinate | None:
    if not isinstance(value, str):
        return None
    parts = value.split(",")
    if len(parts) != 2:
        return None
    try:
        longitude, latitude = (float(item) for item in parts)
    except ValueError:
        return None
    return Coordinate(longitude=longitude, latitude=latitude, crs="GCJ-02")


def _payload_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _string(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _float(value: object) -> float | None:
    if not isinstance(value, str | int | float):
        return None
    try:
        return float(value) if value != "" else None
    except (TypeError, ValueError):
        return None
