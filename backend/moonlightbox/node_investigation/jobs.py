"""Job 只负责排队和恢复；起点调查由 Agent 自主完成。"""

from sqlalchemy.orm import Session

from moonlightbox.agent_runtime.persistence import checkpoint_path
from moonlightbox.agent_runtime.sensitive_mask import MaskRegistry, SensitiveContentHandler
from moonlightbox.events.cloud_client import NodeAnalysisCloudClient
from moonlightbox.jobs.models import Job
from moonlightbox.jobs.registry import JobHandlerError
from moonlightbox.world.client import LightRAGSidecarClient

from .agent import run_investigation
from .store import InvestigationStore, activity, queue_turn
from .tools import InvestigationTools


def create_investigation_handler(settings, *, model=None, retrieval_client=None):
    def handler(service, job):
        store = InvestigationStore(
            service.session.get_bind(), job.payload["project_id"], job.payload["investigation_id"]
        )
        store.worker_token = job.worker_token
        if store.get().state["status"] not in {"queued", "running"}:
            return

        def cancelled():
            with Session(store.engine) as session:
                current = session.get(Job, job.id)
                return (
                    not current
                    or current.status != "running"
                    or current.worker_token != job.worker_token
                    or store.get().state["status"] not in {"queued", "running"}
                )

        store.mutate(lambda s, _: s.update(status="running"), job_id=job.id)
        owned_model = None
        owned_rag = None
        try:
            sensitive_handler = None
            if model is None:
                owned_model = NodeAnalysisCloudClient(
                    enabled=settings.node_analysis_enabled,
                    endpoint=settings.node_analysis_endpoint,
                    model=settings.node_analysis_model,
                    api_key=settings.node_analysis_api_key,
                    timeout_seconds=settings.world_compiler_timeout_seconds,
                    max_retries=settings.node_analysis_max_retries,
                    backoff_seconds=settings.node_analysis_backoff_seconds,
                    max_backoff_seconds=settings.node_analysis_max_backoff_seconds,
                    max_retry_after_seconds=settings.node_analysis_max_retry_after_seconds,
                    response_format=settings.node_analysis_response_format,
                    thinking_mode=settings.node_analysis_thinking_mode,
                    max_output_tokens=settings.node_analysis_max_output_tokens,
                )
                registry = MaskRegistry(service.session.get_bind(), store.project_id)

                def on_mask(refs):
                    store.mutate(
                        lambda state, _: activity(
                            state,
                            "masked",
                            f"{len(refs)} 条消息因内容策略未送云端，仅保留时间与发送人",
                        ),
                        job_id=job.id,
                    )

                sensitive_handler = SensitiveContentHandler(
                    registry, source="node_investigation", on_mask=on_mask
                )
            if retrieval_client is None and settings.lightrag_enabled:
                owned_rag = LightRAGSidecarClient(
                    settings.lightrag_sidecar_url,
                    settings.lightrag_sidecar_token.get_secret_value(),
                )
            tools = InvestigationTools(store, job.id, retrieval_client or owned_rag)
            result = run_investigation(
                tools,
                model or owned_model.create_agent_chat_model(sensitive_handler=sensitive_handler),
                checkpoint_path=checkpoint_path(service.session),
                cancelled=cancelled,
            )
            if result.value is None:
                raise JobHandlerError(
                    result.terminal_reason,
                    f"节点调查暂未完成：{result.terminal_reason}；已保存的阅读和候选仍保留",
                )

            def finish(state, session):
                state.update(summary=result.value.summary, error=None, seen_input=tools.seen_input)
                # 第一版不把调查问题交给用户编辑。即使旧模型仍提交了 question，
                # 也把它作为未决信息留在摘要中，不阻塞自动调查。全范围尚未
                # 顺序读完时自动开启下一回合，候选在真正完成前不交给用户选择。
                state["pending_question"] = None
                scope_complete = state["cursor"] >= state.get(
                    "range_end", len(state["message_ids"])
                )
                has_new_input = len(state["inputs"]) > tools.seen_input
                if not scope_complete or has_new_input:
                    activity(
                        state,
                        "continuing",
                        "Agent 已保存当前候选，继续完成剩余记录调查。",
                    )
                    queue_turn(session, store.id, store.project_id, state)
                else:
                    state["status"] = "completed"
                    activity(state, "completed", result.value.summary)

            if not cancelled():
                store.mutate(finish, job_id=job.id)
        except Exception as error:
            if not cancelled():
                code = str(getattr(error, "code", "investigation_failed"))
                store.mutate(
                    lambda s, _: (
                        s.update(status="failed", error=code),
                        activity(s, "error", f"调查暂停：{code}；已保存进展可恢复"),
                    ),
                    job_id=job.id,
                )
            if isinstance(error, JobHandlerError):
                raise
            raise JobHandlerError(
                str(getattr(error, "code", "investigation_failed")),
                "节点调查未完成，已保存进展；请查看 Phoenix 或恢复任务",
            ) from error
        finally:
            if owned_model:
                owned_model.close()
            if owned_rag:
                owned_rag.close()

    return handler
