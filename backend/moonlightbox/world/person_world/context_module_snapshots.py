"""候选模块快照和字段依赖；内容只在回合边界替换，不共享可变 ORM 对象。"""

from copy import deepcopy
from dataclasses import dataclass, field
from uuid import uuid4

from .contracts.context_modules import MODULE_MODELS


def read_path(value, path):
    current = value
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            raise ValueError(f"不存在的模块字段: {path}")
        current = current[part]
    return deepcopy(current)


@dataclass(frozen=True)
class ModuleSnapshot:
    id: str
    project_id: str
    availability: str
    source: str
    # JSON 字符串保证冻结对象内部也不可被调用者修改。
    payload_json: str

    @classmethod
    def create(cls, project_id, modules=(), *, availability="ready", source="candidate"):
        import json

        return cls(
            str(uuid4()),
            project_id,
            availability,
            source,
            json.dumps(list(modules), ensure_ascii=False),
        )

    @property
    def modules(self):
        import json

        return json.loads(self.payload_json)

    def envelope(self):
        return {"snapshot_id": self.id, "availability": self.availability, "source": self.source}

    def to_dict(self):
        return {**self.envelope(), "project_id": self.project_id, "modules": self.modules}

    @classmethod
    def from_dict(cls, value):
        """恢复原快照身份，不为同一轮调查重新分配模块快照 ID。"""
        import json

        return cls(
            value["snapshot_id"],
            value["project_id"],
            value["availability"],
            value["source"],
            json.dumps(value["modules"], ensure_ascii=False),
        )


@dataclass
class ModuleReadTracker:
    section: str
    snapshot: ModuleSnapshot
    dependencies: list[dict] = field(default_factory=list)

    def record(self, module=None, paths=(), *, directory=False, entry_id=None):
        item = {
            "consuming_section": self.section,
            "entry_id": entry_id,
            "snapshot_id": self.snapshot.id,
            "module_id": module["id"] if module else None,
            "revision": module["revision"] if module else None,
            "field_paths": list(paths),
            "directory": directory,
        }
        if item not in self.dependencies:
            self.dependencies.append(item)


def accept_modules(modules, previous=()):
    """分配稳定 ID/revision，禁止伪造现有实例或原地换类型。标题修改保持 ID。"""
    old = {item["id"]: item for item in previous}
    accepted, seen = [], set()
    for raw in modules:
        item = deepcopy(raw)
        key = item.get("id")
        if key and key not in old:
            raise ValueError("模块 ID 不属于本次快照")
        if key in seen:
            raise ValueError("同一个模块不能出现两次")
        if key:
            seen.add(key)
            before = old[key]
            if before["kind"] != item["kind"]:
                raise ValueError("纠正类型须新建实例并移除旧实例")

            def content(value):
                return {k: v for k, v in value.items() if k != "revision"}

            item["revision"] = before["revision"] + int(content(before) != content(item))
        else:
            item["id"], item["revision"] = str(uuid4()), 1
        item["schema_version"] = f"{item['kind']}_v1"
        accepted.append(MODULE_MODELS[item["kind"]].model_validate(item).model_dump(mode="json"))
    ids = {item["id"] for item in accepted}
    if any(set(item["related_module_ids"]) - ids for item in accepted):
        raise ValueError("相关模块引用不属于当前候选")
    return accepted


def assign_entry_ids(value, previous=None):
    """固定字段按路径延续身份；模型不能自行决定持久化条目 ID。"""
    if isinstance(value, dict):
        old = previous if isinstance(previous, dict) else {}
        if "dimension_id" in value:
            value["id"] = old.get("id") or str(uuid4())
        if "relationship_scope" in value and "profile" in value:
            value["id"] = old.get("id") or str(uuid4())
        for key, child in value.items():
            assign_entry_ids(child, old.get(key))
    elif isinstance(value, list):
        old_items = previous if isinstance(previous, list) else []
        for child in value:
            old = next(
                (
                    item
                    for item in old_items
                    if isinstance(item, dict)
                    and isinstance(child, dict)
                    and (
                        (
                            child.get("dimension_id")
                            and item.get("dimension_id") == child["dimension_id"]
                        )
                        or (child.get("id") and item.get("id") == child["id"])
                    )
                ),
                None,
            )
            assign_entry_ids(child, old)


def bind_owned_module_references(section):
    """life_context 同次提交的综述引用绑定到同次模块结果。

    模块 revision 由后端分配，作者 Agent 不应猜测递增后的版本号。
    此操作仅用于模块所有者本次整体提交，不能替其他栏目的旧推断推进版本。
    """
    modules = {item["id"]: item for item in section["context_modules"]}

    def visit(value):
        if isinstance(value, dict):
            for ref in value.get("context_module_refs", []):
                module = modules.get(ref["module_id"])
                if module is None:
                    raise ValueError("生活情境综述引用了已移除或不存在的模块")
                for path in ref["field_paths"]:
                    read_path(module, path)
                ref["revision"] = module["revision"]
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(section)


def validate_references(value, snapshot, available_message_ids=None, tracker=None):
    """引用可为空；给出的 ID/版本/路径必须真实，不检查心理证据数量。"""
    if isinstance(value, dict):
        if (
            available_message_ids is not None
            and set(value.get("reference_message_ids", [])) - available_message_ids
        ):
            raise ValueError("参考消息 ID 不在本次可访问的资料中")
        by_id = {item["id"]: item for item in snapshot.modules}
        for ref in value.get("context_module_refs", []):
            module = by_id.get(ref["module_id"])
            if module is None or module["revision"] != ref["revision"]:
                raise ValueError("模块引用已过期或不属于本次快照")
            for path in ref["field_paths"]:
                read_path(module, path)
            if tracker:
                tracker.record(module, ref["field_paths"], entry_id=value.get("id"))
        for child in value.values():
            validate_references(child, snapshot, available_message_ids, tracker)
    elif isinstance(value, list):
        for child in value:
            validate_references(child, snapshot, available_message_ids, tracker)


def affected_sections(before, after, dependencies):
    """只依据已读字段计算可能影响，不用中文关键词或人格打分。"""
    old, new = ({m["id"]: m for m in items} for items in (before, after))
    added_removed = set(old) ^ set(new)
    affected = set()
    for dep in dependencies:
        if added_removed and dep.get("directory"):
            affected.add(dep["consuming_section"])
        key = dep.get("module_id")
        if not key:
            continue
        if key in added_removed:
            affected.add(dep["consuming_section"])
        elif key in old and key in new:
            paths = dep.get("field_paths") or [
                "kind",
                "status",
                "period",
                "summary",
                "details",
                "basis",
                "related_module_ids",
            ]
            if any(
                read_path(old[key], p) != read_path(new[key], p)
                for p in paths
                if p not in {"title", "revision"}
            ):
                affected.add(dep["consuming_section"])
    # 新增背景也可能改变没有显式引用过模块的主要消费者。
    if added_removed:
        affected.update({"practices", "agency", "identity"})
    return affected - {"life_context"}


def rebase_unchanged_references(value, before, after):
    """标题等非语义变动只推进未改变字段的引用版本，不重写 Agent 结论。"""
    old, new = ({item["id"]: item for item in modules} for modules in (before, after))

    def visit(node):
        if isinstance(node, dict):
            for ref in node.get("context_module_refs", []):
                key = ref["module_id"]
                if key not in old or key not in new or ref["revision"] != old[key]["revision"]:
                    continue
                paths = ref["field_paths"] or [
                    "kind",
                    "status",
                    "period",
                    "summary",
                    "details",
                    "basis",
                ]
                if all(
                    read_path(old[key], path) == read_path(new[key], path)
                    for path in paths
                    if path not in {"title", "revision"}
                ):
                    ref["revision"] = new[key]["revision"]
            for child in node.values():
                visit(child)
        elif isinstance(node, list):
            for child in node:
                visit(child)

    visit(value)
