from fastapi.testclient import TestClient


def event_payload() -> dict[str, object]:
    return {
        "type": "conflict",
        "start_message_id": "m1",
        "end_message_id": "m2",
        "before_state": "正常交流",
        "after_state": "关系紧张",
        "emotion_labels": ["生气"],
        "topic": "误解",
        "conflict_level": 4,
        "importance": 0.8,
        "reason": "语气升级",
        "evidence_ids": ["m1", "m2"],
    }


def test_event_nodes_can_be_created_and_revised(client: TestClient) -> None:
    project = client.post("/api/projects", json={"name": "节点项目"}).json()

    created = client.post(
        f"/api/projects/{project['id']}/events",
        json=event_payload(),
    )
    assert created.status_code == 201
    event_id = created.json()["id"]

    revised = client.patch(
        f"/api/projects/{project['id']}/events/{event_id}",
        json={"changes": {"type": "cold_war"}, "reason": "人工纠正"},
    )
    listed = client.get(f"/api/projects/{project['id']}/events")

    assert revised.status_code == 200
    assert revised.json()["type"] == "cold_war"
    assert listed.json()[0]["id"] == event_id
