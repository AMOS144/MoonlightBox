"""已退役启发式候选的测试改由统一 Agent 冒烟覆盖；保留图节点类型过滤。"""

from moonlightbox.world.client import LightRAGEntity
from moonlightbox.world.merges import person_entities


def test_person_entities_filters_technical_ids_without_name_blacklist():
    entities = [
        LightRAGEntity(entity_name="洪欣羽", graph_data={"entity_type": "person"}),
        LightRAGEntity(entity_name="wxid_m6t7zrkzb5hf22", graph_data={"entity_type": "person"}),
        LightRAGEntity(entity_name="女同事", graph_data={"entity_type": "person"}),
        LightRAGEntity(entity_name="上班", graph_data={"entity_type": "routine"}),
    ]
    assert [item.entity_name for item in person_entities(entities)] == ["洪欣羽", "女同事"]
