"""对既有图谱运行 v3 编译烟测，保存 JSON；不覆盖旧 Profile、不批准或修改图谱。

使用示例：PYTHONPATH=backend .venv/bin/python scripts/person_world_v3_smoke.py
  --env-file ../moonlight-box-implementation/.env --graph-id <现有图版本>
  --output ../.runtime-data/results/person-world-v3.json
"""

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv
from moonlightbox.config import Settings
from moonlightbox.db import Database
from moonlightbox.events.cloud_client import NodeAnalysisCloudClient
from moonlightbox.imports.models import Participant
from moonlightbox.observability import initialize_phoenix, shutdown_phoenix
from moonlightbox.projects.models import Project  # noqa: F401
from moonlightbox.world.client import LightRAGSidecarClient
from moonlightbox.world.models import PersonWorldAgentRun, PersonWorldSectionTask, WorldGraphVersion
from moonlightbox.world.person_world.coordinator_v3 import PersonWorldCoordinatorV3
from moonlightbox.agent_runtime.policy import SectionAgentRuntimePolicy
from sqlalchemy import select
from sqlalchemy.orm import Session


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", required=True)
    parser.add_argument("--graph-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume-run-id", help="只补跑已结束烟测的失败栏目，保留成功草稿")
    parser.add_argument("--refresh-sections", nargs="+", help="开发烟测中显式重做指定栏目，须与 resume-run-id 同用")
    args = parser.parse_args()
    load_dotenv(args.env_file)
    settings = Settings(phoenix_enabled=True, phoenix_capture_content=True)
    initialize_phoenix(settings, service_name="personworld-v3-smoke")
    database = Database(settings.database_url)
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
        with Session(database.engine, expire_on_commit=False) as session:
            graph = session.get(WorldGraphVersion, args.graph_id)
            if graph is None:
                raise ValueError("指定图版本不存在")
            people = list(
                session.scalars(
                    select(Participant).where(Participant.project_id == graph.project_id)
                )
            )
            target = next(person for person in people if person.role == "target")
            user = next(person for person in people if person.role == "self")
            print(
                json.dumps(
                    {
                        "stage": "started",
                        "model": settings.node_analysis_model,
                        "graph_id": graph.id,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            coordinator = PersonWorldCoordinatorV3(
                session=session,
                graph=graph,
                lightrag=sidecar,
                compiler=compiler,
                subject_name=target.name,
                user_name=user.name,
                target_participant_id=target.id,
                user_participant_id=user.id,
                section_concurrency=settings.person_world_section_concurrency,
                section_runtime_policy=SectionAgentRuntimePolicy(
                    max_tool_calls=settings.person_world_section_max_tool_calls,
                    deadline_seconds=settings.person_world_section_deadline_seconds,
                    max_stalled_cycles=settings.person_world_section_max_stalled_cycles,
                    model_request_timeout_seconds=settings.person_world_section_model_timeout_seconds,
                ),
                progress=lambda stage, completed, total: print(
                    json.dumps({"stage": stage, "completed": completed, "total": total}), flush=True
                ),
            )
            result = (
                resume_smoke(coordinator, args.resume_run_id, args.refresh_sections)
                if args.resume_run_id
                else coordinator.run(mode="smoke_v3")
            )
            document = {
                "run_id": result.run_id,
                "profile": result.profile_v3.model_dump(mode="json"),
                "investigation": result.investigation_report.model_dump(mode="json"),
                "generation_summary": result.generation_summary,
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(
                json.dumps(
                    {
                        "stage": "saved",
                        "output": str(args.output),
                        "failed_sections": result.generation_summary["failed_sections"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            if result.generation_summary["failed_sections"]:
                raise SystemExit(2)
    finally:
        compiler.close()
        sidecar.close()
        shutdown_phoenix()


def resume_smoke(coordinator, run_id, refresh_sections=None):
    """开发烟测恢复：不重跑已成功调查，不接触线上 Profile 或 Publication。"""
    from copy import deepcopy

    from moonlightbox.world.person_world.context_module_snapshots import ModuleSnapshot

    session = coordinator.session
    run = session.get(PersonWorldAgentRun, run_id)
    if (
        run is None
        or run.graph_version_id != coordinator.graph.id
        or run.mode != "smoke_v3"
        or run.status != "awaiting_review"
    ):
        raise ValueError("只能恢复同一图版本下已结束、尚待审核的 v3 烟测")
    errors = dict(run.state.get("errors", {}))
    from moonlightbox.world.person_world.contracts.world_dimensions import SECTION_DIMENSIONS
    selected = list(dict.fromkeys(refresh_sections or errors))
    if not selected or set(selected) - set(SECTION_DIMENSIONS):
        raise ValueError("没有失败栏目需要补跑")
    coordinator.run_row, coordinator._v3_run_id = run, run.id
    coordinator.results = deepcopy(run.state.get("section_drafts", {}))
    coordinator.baseline = deepcopy(coordinator.results)
    coordinator.errors, coordinator.executions = errors, {}
    coordinator.overview = run.state.get("overview", "")
    coordinator.dependencies = deepcopy(run.state.get("dependencies", []))
    coordinator.attempts = deepcopy(run.state.get("attempts", []))
    coordinator.tasks = {
        task.section: task
        for task in session.scalars(
            select(PersonWorldSectionTask).where(PersonWorldSectionTask.agent_run_id == run.id)
        )
    }
    modules = coordinator.results.get("life_context", {}).get("context_modules", [])
    coordinator.module_snapshot = ModuleSnapshot.create(
        coordinator.graph.project_id,
        modules,
        availability="ready"
        if "life_context" in coordinator.results and "life_context" not in errors
        else "pending",
    )
    run.status = "researching"
    session.commit()
    retry_modules = "life_context" in selected
    coordinator._batch(selected, "retry")
    if retry_modules:
        coordinator._enrich({})
    coordinator._assemble({})
    return coordinator.output


if __name__ == "__main__":
    main()
