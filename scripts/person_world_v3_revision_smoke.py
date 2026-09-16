"""在临时数据库副本中测试真实工作安排纠正，只输出候选 Diff，绝不批准或写图。

场景是测试输入，不代表用户确认了目标人物的真实工作安排。
临时副本退出即移除；项目仍只使用原来的共享数据库。
"""

import argparse
import json
import sqlite3
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from dotenv import load_dotenv
from moonlightbox.config import Settings
from moonlightbox.db import Database
from moonlightbox.events.cloud_client import NodeAnalysisCloudClient
from moonlightbox.imports.models import Participant  # noqa: F401
from moonlightbox.observability import initialize_phoenix, shutdown_phoenix
from moonlightbox.projects.models import Project  # noqa: F401
from moonlightbox.world.client import LightRAGSidecarClient
from moonlightbox.world.models import (
    PersonWorldAgentRun,
    PersonWorldProfile,
    PersonWorldRevisionSession,
    WorldGraphVersion,
)
from moonlightbox.world.person_world.review.profile_v3 import build_merged_preview
from sqlalchemy import select
from sqlalchemy.orm import Session


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", required=True)
    parser.add_argument("--input", type=Path, required=True, help="编译烟测产出的 JSON")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenario", required=True, help="明确标记为模拟纠正的测试输入")
    parser.add_argument(
        "--replay-result", type=Path, help="重放已成功的真实栏目输出，只验收确定性组装，不调用模型"
    )
    args = parser.parse_args()
    load_dotenv(args.env_file)
    settings = Settings(phoenix_enabled=True, phoenix_capture_content=True)
    document = json.loads(args.input.read_text(encoding="utf-8"))
    modules = document["profile"]["life_context"]["context_modules"]
    module = next(
        (item for item in modules if item["kind"] == "employment" and item["status"] == "current"),
        None,
    )
    if module is None:
        raise ValueError("候选画像没有当前工作模块，不能执行工作安排纠正烟测")
    initialize_phoenix(settings, service_name="personworld-v3-revision-smoke")
    source = Database(settings.database_url)
    compiler = NodeAnalysisCloudClient(
        enabled=settings.node_analysis_enabled,
        endpoint=settings.node_analysis_endpoint,
        model=settings.node_analysis_model,
        api_key=settings.node_analysis_api_key,
        timeout_seconds=settings.world_compiler_timeout_seconds,
        max_retries=0,
        response_format=settings.node_analysis_response_format,
        thinking_mode=settings.node_analysis_thinking_mode,
        max_output_tokens=settings.node_analysis_max_output_tokens,
    )
    sidecar = LightRAGSidecarClient(
        settings.lightrag_sidecar_url,
        settings.lightrag_sidecar_token.get_secret_value(),
        timeout_seconds=settings.lightrag_timeout_seconds,
    )
    try:
        with TemporaryDirectory(prefix="moonlightbox-revision-smoke-") as temp:
            copy_path = Path(temp) / "isolated.db"
            connection = source.engine.raw_connection()
            try:
                with sqlite3.connect(copy_path) as destination:
                    connection.driver_connection.backup(destination)
            finally:
                connection.close()
            database = Database(f"sqlite:///{copy_path}")
            try:
                with Session(database.engine, expire_on_commit=False) as session:
                    run = session.get(PersonWorldAgentRun, document["run_id"])
                    if run is None:
                        raise ValueError("编译烟测运行不存在")
                    graph = session.get(WorldGraphVersion, run.graph_version_id)
                    profile = session.scalar(
                        select(PersonWorldProfile).where(
                            PersonWorldProfile.graph_version_id == graph.id
                        )
                    )
                    if profile is None:
                        raise ValueError("本烟测需要已有基础 Profile 的图版本")
                    # 只在临时副本中装入新稿。正式库的旧 Profile 始终不改。
                    profile.profile_v3 = document["profile"]
                    profile.profile_schema_version = "v3"
                    profile.generation_summary = document["generation_summary"]
                    revision = PersonWorldRevisionSession(
                        project_id=graph.project_id,
                        base_graph_version_id=graph.id,
                        base_profile_id=profile.id,
                        status="understanding_ready",
                        input_revision=1,
                        scope={
                            "smoke_only": True,
                            "selected_statements": [
                                {
                                    "section": "life_context",
                                    "module_id": module["id"],
                                    "field_path": "details.working_arrangement.schedule",
                                }
                            ],
                        },
                        understanding_payload={
                            "summary_for_user": "模拟验收，非真实用户纠正",
                            "wrong_interpretation": "需要在烟测中更改的工作安排",
                            "corrected_interpretation": args.scenario,
                            "affected_dimensions": ["time"],
                            "affected_profile_sections": ["life_context"],
                            "source_message_ids": [],
                            "graph_change_requested": False,
                        },
                    )
                    session.add(revision)
                    session.commit()
                    replay = (
                        replay_successful_sections(args.replay_result, args.scenario)
                        if args.replay_result
                        else nullcontext()
                    )
                    with replay:
                        profile_patch = build_merged_preview(
                            SimpleNamespace(session=session, compiler=compiler, lightrag=sidecar),
                            revision,
                            profile,
                            graph,
                        )
                    output = {
                        "smoke_only": True,
                        "scenario": args.scenario,
                        "published": False,
                        "graph_operations": [],
                        "profile_patch": profile_patch,
                        "preview": revision.scope["v3_preview"],
                        "assembly_replay_from": str(args.replay_result)
                        if args.replay_result
                        else None,
                        "llm_called_in_this_invocation": args.replay_result is None,
                    }
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    args.output.write_text(
                        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
                    )
                    print(
                        json.dumps(
                            {
                                "output": str(args.output),
                                "changed_sections": [item["section"] for item in profile_patch],
                                "published": False,
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
            finally:
                database.close()
    finally:
        compiler.close()
        sidecar.close()
        source.close()
        shutdown_phoenix()


def replay_successful_sections(path, scenario):
    """仅烟测可重放。沿用真实 Phoenix 关联，不伪造一次新的模型调用或运行轨迹。"""
    from moonlightbox.world.person_world.contracts.profile_v3 import SECTION_RESULT_MODELS
    from moonlightbox.world.person_world.section_agent_v3 import SectionExecutionV3

    source = json.loads(path.read_text(encoding="utf-8"))
    if not source.get("smoke_only") or source.get("published") or source.get("graph_operations"):
        raise ValueError("只允许重放未发布、无图操作的烟测候选")
    if source["scenario"] != scenario:
        raise ValueError("重放不能改变原来确认的模拟输入")
    outputs = {item["section"]: item["value"] for item in source["profile_patch"]}
    attempts = {item["section"]: item for item in source["preview"]["attempts"]}

    def run_section(**kwargs):
        section = kwargs["definition"].name
        attempt = attempts[section]
        if attempt["status"] != "succeeded":
            raise ValueError("不能把失败生成重放成成功")
        data = deepcopy(outputs.get(section, kwargs["context"]["previous_section"]))
        if section == "identity":
            data["overview"] = outputs.get("overview", "")
        if section == "life_context":
            data["module_assessment"] = {
                "status": "identified" if data["context_modules"] else "not_identified",
                "explanation": "重放真实成功输出，仅重新检验组装，不是新的模型判断",
            }
        tracker = kwargs["tracker"]
        for dependency in source["preview"]["dependencies"]:
            if dependency["consuming_section"] == section:
                tracker.dependencies.append({**dependency, "snapshot_id": tracker.snapshot.id})
        return SectionExecutionV3(
            section,
            SECTION_RESULT_MODELS[section].model_validate(data),
            "succeeded",
            None,
            attempt["phoenix_execution_id"],
            (),
            (),
            tracker.dependencies,
        )

    return patch("moonlightbox.world.person_world.review.profile_v3.run_section_v3", run_section)


if __name__ == "__main__":
    main()
