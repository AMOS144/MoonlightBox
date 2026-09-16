"""运行真实人物关系栏目 Agent，输出表达资料；业务数据库只读，不发布产物。"""

import argparse
import json
from pathlib import Path
from uuid import uuid4

from moonlightbox.agent_runtime.policy import section_policy
from moonlightbox.config import Settings
from moonlightbox.events.cloud_client import NodeAnalysisCloudClient
from moonlightbox.imports.models import Participant
from moonlightbox.model_registry import register_models
from moonlightbox.runtime_v1.db_models import RuntimeSnapshotRow
from moonlightbox.world.client import LightRAGSidecarClient
from moonlightbox.world.models import WorldGraphVersion
from moonlightbox.world.person_world.context_module_snapshots import (
    ModuleReadTracker,
    ModuleSnapshot,
)
from moonlightbox.world.person_world.investigation_artifacts import InvestigationArtifactStore
from moonlightbox.world.person_world.prompt_loader import load_section_prompt_v3
from moonlightbox.world.person_world.section_agent_v3 import run_section_v3
from moonlightbox.world.person_world.tools.context_modules import build_context_module_tools
from moonlightbox.world.person_world.tools.section_tools import build_section_tools
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--branch-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    settings = Settings(_env_file=args.env_file)
    register_models()
    # 明确只读业务库；本次独立检查点和输出放在结果目录。
    engine = create_engine(f"sqlite:///file:{args.database.resolve()}?mode=ro&uri=true")
    args.output.parent.mkdir(parents=True, exist_ok=True)
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
        with Session(engine) as session:
            origin = session.scalar(
                select(RuntimeSnapshotRow).where(RuntimeSnapshotRow.branch_id == args.branch_id)
            )
            if origin is None:
                raise ValueError("分支背景不存在")
            graph = session.get(WorldGraphVersion, origin.graph_version_id)
            people = list(
                session.scalars(
                    select(Participant).where(Participant.project_id == graph.project_id)
                )
            )
            target = next(p for p in people if p.role == "target")
            user = next(p for p in people if p.role == "self")
            profile = origin.profile
            section = "relationship_with_user"
            snapshot = ModuleSnapshot.create(
                graph.project_id,
                profile.get("life_context", {}).get("context_modules", []),
                availability="ready",
                source="published",
            )
            tracker = ModuleReadTracker(section, snapshot)
            artifacts = InvestigationArtifactStore()
            tools = build_section_tools(
                session,
                graph=graph,
                lightrag=sidecar,
                top_k=30,
                chunk_top_k=12,
                max_total_tokens=16000,
                correction_change_set_ids=set(),
                artifacts=artifacts,
            )
            tools.update(build_context_module_tools(tracker, project_id=graph.project_id))
            definition = load_section_prompt_v3(section)
            context = {
                "section": section,
                "phase": "enrichment",
                "target_person": target.name,
                "user": user.name,
                "timezone": origin.timezone,
                "module_snapshot": snapshot.envelope(),
                "previous_section": profile.get(section),
                "profile_snapshot": profile,
                "section_summaries": {
                    k: v.get("summary", "") for k, v in profile.items() if isinstance(v, dict)
                },
                "instruction": "补全本栏目，尤其是尚未编译的 expression_profile。"
                "按栏目要求查询真实互动，"
                "总结常用称呼、语气词、分句、标点、玩笑与场景差异，提交完整栏目。",
            }
            print(
                json.dumps(
                    {
                        "stage": "running",
                        "model": settings.node_analysis_model,
                        "target": target.name,
                        "graph_id": graph.id,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            result = run_section_v3(
                definition=definition,
                tools={n: tools[n] for n in definition.tool_names},
                compiler=compiler,
                context=context,
                tracker=tracker,
                artifacts=artifacts,
                policy=section_policy(settings),
                owner_id=f"expression-preview:{uuid4()}",
                project_id=graph.project_id,
                graph_id=graph.id,
                target_id=target.id,
                checkpoint_path=str(args.output.with_suffix(".sqlite")),
                on_status=lambda status: print(
                    json.dumps(status, ensure_ascii=False, default=str), flush=True
                ),
            )
            value = result.result.model_dump(mode="json") if result.result is not None else None
            document = {
                "status": result.status,
                "reason": result.reason,
                "execution_id": result.execution_id,
                "profile_id": origin.profile_id,
                "expression_profile": value.get("expression_profile") if value else None,
                "relationship_with_user": value,
            }
            args.output.write_text(
                json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(
                json.dumps({"stage": "saved", "status": result.status, "path": str(args.output)}),
                flush=True,
            )
    finally:
        compiler.close()
        sidecar.close()
        engine.dispose()


if __name__ == "__main__":
    main()
