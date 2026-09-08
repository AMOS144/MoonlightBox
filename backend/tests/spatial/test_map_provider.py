import httpx
from moonlightbox.spatial.map_provider import AmapWebServiceProvider, Coordinate


def test_amap_provider_parses_poi_and_uses_city_constraint() -> None:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(
            200,
            json={
                "status": "1",
                "pois": [
                    {
                        "id": "B001",
                        "name": "前滩太古里",
                        "address": "东育路500弄",
                        "type": "购物服务",
                        "location": "121.491000,31.153000",
                        "adcode": "310115",
                        "adname": "浦东新区",
                    }
                ],
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = AmapWebServiceProvider("test-key", client=client)

    places = provider.search_poi(
        "前滩太古里",
        city_hint="上海",
        adcode_hint=None,
        near=None,
        category=None,
        limit=3,
    )

    assert len(places) == 1
    assert places[0].provider_place_id == "B001"
    assert places[0].coordinate == Coordinate(121.491, 31.153, "GCJ-02")
    assert captured[0].url.params["city"] == "上海"
    assert captured[0].url.params["citylimit"] == "true"
    assert "test-key" not in repr(places[0])


def test_amap_provider_rejects_provider_error_without_leaking_key() -> None:
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(
                200,
                json={"status": "0", "info": "INVALID_USER_KEY"},
            )
        )
    )
    provider = AmapWebServiceProvider("secret-key", client=client)

    try:
        provider.geocode("上海市浦东新区", city_hint="上海", limit=1)
    except RuntimeError as error:
        assert str(error) == "INVALID_USER_KEY"
        assert "secret-key" not in str(error)
    else:
        raise AssertionError("地图服务错误不应被忽略")
