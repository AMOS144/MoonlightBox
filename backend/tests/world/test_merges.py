from moonlightbox.world.client import LightRAGEntity
from moonlightbox.world.merges import (
    MergeCandidate,
    _description_marks_alias,
    _attach_original_evidence,
    _explicit_identity_candidates,
    _filter_relation_target_conflicts,
    _sanitize_candidates,
    person_entities,
)


def entity(name: str, description: str, entity_type: str = "person") -> LightRAGEntity:
    return LightRAGEntity(
        entity_name=name,
        graph_data={"entity_type": entity_type, "description": description},
    )


def test_person_entities_filters_technical_ids_without_name_blacklist() -> None:
    result = person_entities(
        [
            entity("洪欣羽", "聊天参与者"),
            entity("wxid_m6t7zrkzb5hf22", "微信账号", "person"),
            entity("zrkzb5hf22", "微信账号", "person"),
            entity("女同事", "第三方女性", "person"),
            entity("上班", "日常活动", "routine"),
        ]
    )

    assert [item.entity_name for item in result] == ["洪欣羽", "女同事"]


def test_alias_detection_uses_description_signal() -> None:
    assert _description_marks_alias({"description": "洪欣羽的暱稱，在对话中使用简称"})
    assert not _description_marks_alias({"description": "洪欣羽的女性同事"})


def test_explicit_nickname_statement_creates_review_candidate() -> None:
    nodes = [
        {
            "entity": "jxy",
            "description": "发言者提及的车辆代号",
            "graph_data": {"entity_type": "person", "source_id": "bundle-a"},
        },
        {
            "entity": "江昕岳",
            "description": "洪欣羽提到的车主，昵称为jxy",
            "graph_data": {"entity_type": "person", "source_id": "bundle-b"},
        },
    ]

    result = _explicit_identity_candidates(
        nodes,
        canonical_names={"江昕岳"},
        subject_name="洪欣羽",
    )

    assert [(item.source_entities, item.target_entity) for item in result] == [
        (["jxy"], "江昕岳")
    ]


def test_subject_nickname_requires_direct_alias_wording() -> None:
    nodes = [
        {
            "entity": "羽",
            "description": "洪欣羽的暱稱，在對話開頭使用這個簡稱來稱呼洪欣羽",
            "graph_data": {"entity_type": "person"},
        },
        {
            "entity": "我",
            "description": "洪欣羽的昵称，男友称她为笨入",
            "graph_data": {"entity_type": "person"},
        },
        {
            "entity": "洪欣羽",
            "description": "聊天参与者",
            "graph_data": {"entity_type": "person"},
        },
    ]

    result = _explicit_identity_candidates(
        nodes,
        canonical_names={"洪欣羽"},
        subject_name="洪欣羽",
    )

    assert [(item.source_entities, item.target_entity) for item in result] == [
        (["羽"], "洪欣羽")
    ]


def test_sanitize_requires_stable_identity_target() -> None:
    candidates = [
        MergeCandidate(
            source_entities=["jxy"],
            target_entity="女同事",
            reason="关系相似",
        ),
        MergeCandidate(
            source_entities=["jxy"],
            target_entity="江昕岳",
            reason="节点描述明确写有昵称为 jxy",
        ),
    ]

    result = _sanitize_candidates(
        candidates,
        {"jxy", "女同事", "江昕岳"},
        canonical_names={"江昕岳"},
    )

    assert [(item.source_entities, item.target_entity) for item in result] == [
        (["jxy"], "江昕岳")
    ]


def test_relation_only_target_is_not_identity_evidence() -> None:
    candidate = MergeCandidate(
        source_entities=["瑞哥"],
        target_entity="余熠",
        reason="两人都是洪欣羽的恋人",
    )
    nodes = [
        {"entity": "瑞哥", "description": "余熠的恋人，天天汇报股票"},
        {"entity": "余熠", "description": "洪欣羽的恋人，男性"},
    ]

    assert _filter_relation_target_conflicts([candidate], nodes, []) == []


def test_original_tool_match_is_attached_when_model_omits_message_id() -> None:
    candidate = MergeCandidate(
        source_entities=["jxy"],
        target_entity="江昕岳",
        reason="昵称证据",
    )

    result = _attach_original_evidence(
        [candidate],
        [
            {
                "message_id": "message-1",
                "document_id": "bundle-1",
                "participant": "洪欣羽",
                "content": "洪欣羽说 jxy 的车很特别",
            }
        ],
    )

    assert result[0].evidence[0].message_id == "message-1"
    assert result[0].evidence[0].source_document_id == "bundle-1"
