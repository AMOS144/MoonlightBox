"""跨进程冒烟：不用供应商模型，验证真正退出后从文件存档继续。"""

import os
import subprocess
import sys

import pytest

SCRIPT = r"""
import os
import sys
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel
from moonlightbox.agent_runtime import AgentLoopController, AgentSpec, AgentBudgetPolicy, RunScope
from moonlightbox.agent_runtime.contracts import AgentExecutionRequest, RegisteredTool, ToolContract
from moonlightbox.agent_runtime.submission import result_submission_tool
from moonlightbox.world.person_world.investigation_artifacts import InvestigationArtifactStore

class Result(BaseModel):
    text: str

path, mode, terminal = sys.argv[1:]
store = InvestigationArtifactStore()
def search():
    assert mode == 'interrupt', '恢复时不应重复查询'
    retrieval = store.add_retrieval(
        question='工作时间', mode='mix', references=[{'content':'材料'}])
    key = store.add_evidence_set(
        [{'message_id':'m1','content':'恢复正文'}], retrieval_id=retrieval.retrieval_id)
    return {'evidence_set_id':key}

class Model:
    def invoke(self, messages):
        assert mode != 'replay', '已接受的结果不应重新调用模型'
        has_tool = any(isinstance(m, ToolMessage) for m in messages)
        if not has_tool:
            return AIMessage(content='',tool_calls=[{'name':'search','id':'read','args':{}}])
        if mode == 'interrupt':
            os._exit(17)  # 模拟断电：不运行清理、异常处理或业务提交。
        assert store.evidence_set_rows('evidence-1')[0]['content'] == '恢复正文'
        assert store.get_retrieval('retrieval-1').question == '工作时间'
        new = store.add_evidence_set([{'message_id':'m2'}], retrieval_id=None)
        assert new == 'evidence-2'
        return AIMessage(content='',tool_calls=[{
            'name':'finish','id':'submit','args':{'result':{'text':'完成'}}}])

tool = StructuredTool.from_function(search, description='测试检索')
spec = AgentSpec(name='restart',prompt_version='v1',
    budget=AgentBudgetPolicy(max_wall_seconds=30,max_tool_result_chars=100000),
    submission_tool_name='finish',
    tools=(RegisteredTool(tool,ToolContract(name='search')),
           result_submission_tool('finish',Result,completion_status=lambda value:terminal)),
    snapshot_work_state=store.snapshot,restore_work_state=store.restore)
request = AgentExecutionRequest(owner_type='person_world',owner_id='stable-job',
    scope=RunScope(project_id='p'),messages=(HumanMessage(content='调查'),),checkpoint_path=path)
result = AgentLoopController().run(spec=spec,request=request,model=Model())
assert result.status == terminal
assert result.value.text == '完成'
assert store.evidence_set_rows('evidence-1')[0]['message_id'] == 'm1'
print('restored',flush=True)
"""


@pytest.mark.parametrize("terminal", ["succeeded", "waiting_for_user"])
def test_real_process_restart_restores_materials_and_accepted_result(tmp_path, terminal):
    path = str(tmp_path / "agent.sqlite")
    env = {**os.environ, "OTEL_SDK_DISABLED": "true"}
    for mode, expected in [("interrupt", 17), ("resume", 0), ("replay", 0)]:
        result = subprocess.run(
            [sys.executable, "-c", SCRIPT, path, mode, terminal],
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == expected, result.stdout + result.stderr


def test_material_snapshot_preserves_reference_ids():
    from moonlightbox.world.person_world.investigation_artifacts import InvestigationArtifactStore

    first = InvestigationArtifactStore()
    first.add_retrieval(question="q", mode="mix", references=[])
    first.add_evidence_set([{"message_id": "m1"}], retrieval_id="retrieval-1")
    restored = InvestigationArtifactStore()
    restored.restore(first.snapshot())
    assert restored.evidence_set_retrieval_id("evidence-1") == "retrieval-1"
    assert (
        restored.add_retrieval(question="next", mode="mix", references=[]).retrieval_id
        == "retrieval-2"
    )
