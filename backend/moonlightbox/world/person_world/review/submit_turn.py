"""提交用户可见的纠正回合；批准和图谱写入仍由 RevisionService 阶段门保护。"""

from moonlightbox.agent_runtime.input_contracts import input_model
from moonlightbox.agent_runtime.submission import result_submission_tool

from ..schemas import RevisionAgentTurn, RevisionQuestionOption


def build_submit_revision_turn():
    projected = input_model(
        RevisionAgentTurn,
        omit=lambda cls, name, field: cls is RevisionQuestionOption and name == "id",
    )

    def bind(value):
        data = value.model_dump(mode="json")
        for index, option in enumerate((data.get("question") or {}).get("options", []), 1):
            option["id"] = f"option-{index}"
        return RevisionAgentTurn.model_validate(data)

    return result_submission_tool(
        "submit_revision_turn",
        projected,
        result_adapter=bind,
        completion_status=lambda value: (
            "waiting_for_user" if value.kind == "question" else "succeeded"
        ),
    )
