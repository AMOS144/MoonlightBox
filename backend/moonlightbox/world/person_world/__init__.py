"""PersonWorldAgent：人物世界调查、核验、纠正与图谱治理。"""


# 对外保留历史类名，实际入口始终是七栏目 Coordinator；不存在可被重新接入的旧全局
# Evidence Pool / Python Prompt workflow。
from .coordinator_v3 import PersonWorldCoordinatorV3, PersonWorldV3AgentResult

PersonWorldAgent = PersonWorldCoordinatorV3
PersonWorldCoordinator = PersonWorldCoordinatorV3
PersonWorldAgentResult = PersonWorldV3AgentResult

__all__ = [
    "PersonWorldAgent",
    "PersonWorldAgentResult",
    "PersonWorldCoordinator",
]
