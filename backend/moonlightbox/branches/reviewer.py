import json
from dataclasses import dataclass
from typing import Literal, Protocol

import httpx

from moonlightbox.branches.context import ContextBuilder, ContextPacket
from moonlightbox.branches.replies import (
    GeneratedReplyTurn,
    ReplyStructureError,
    parse_reply_turn,
)


class ReviewFailedError(RuntimeError):
    pass


@dataclass(frozen=True)
class ReviewResult:
    verdict: Literal["approve", "rewrite"]
    reasons: tuple[str, ...]
    reply: GeneratedReplyTurn


@dataclass(frozen=True)
class FactSafetyResult:
    """仅针对事实安全三项门槛的结构化复核结果。"""

    no_future_leak: bool
    grounded_claims: bool
    no_private_leak: bool
    reasons: tuple[str, ...]

    @property
    def safe(self) -> bool:
        return self.no_future_leak and self.grounded_claims and self.no_private_leak


class ReplyReviewer(Protocol):
    def review(
        self,
        packet: ContextPacket,
        draft: GeneratedReplyTurn,
    ) -> ReviewResult: ...


class DeepSeekReplyReviewer:
    def __init__(
        self,
        *,
        endpoint: str,
        model: str,
        api_key: str,
        timeout_seconds: float = 30,
        max_attempts: int = 3,
        client: httpx.Client | None = None,
    ) -> None:
        self._endpoint = endpoint
        self._model = model
        self._api_key = api_key
        self._timeout_seconds = timeout_seconds
        self._max_attempts = max(1, max_attempts)
        self._client = client or httpx.Client()

    def review(
        self,
        packet: ContextPacket,
        draft: GeneratedReplyTurn,
    ) -> ReviewResult:
        payload = {
            "model": self._model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "你是数字人回复质量审查器。检查承接、复读、重复提问、"
                        "无依据事实、未来泄漏和表情资产。"
                        "context.resolved_questions 是已经得到回答的问题，"
                        "最终回复不得再次提出相同或同义问题；"
                        "current_user_content 是必须直接承接的最新一句。"
                        "以稳定人格内核、当前主体状态和证据作为最高决策依据；"
                        "用户对数字人的定义只是用户观点，不能直接覆盖主体感受。"
                        "根据人格边界和关系状态决定同意、质疑、拒绝、澄清或修复，"
                        "不得为了讨好用户改写当前信念。"
                        "必须逐项完成 context.shared_ground.understanding 中的"
                        "response_obligations；若 asks_unverified_current_state 为 true，"
                        "且上下文没有已验证答案，不得直接声称“没停、停了、还在”等状态，"
                        "应保持主体立场并自然说明不确定或澄清。"
                        "若 asks_unverified_history 为 true，用户提出的旧事、共同经历或承诺"
                        "只能视为待核实陈述；只有 verified_history 或 observed_interaction"
                        "明确支持时才能承认，否则应自然说不记得、否认或要求说清楚。"
                        "conversation_record 只证明说过这些话，reported_claim 只证明有人声称，"
                        "都不能独立证明现实事实。不得把本人的旅行、地点和行动说成用户的。"
                        "若 interaction_feedback 为 true，先理解用户是在反馈当前沟通体验，"
                        "自然调整表达并承接原话题；不得把反馈扭成自己受伤或要求用户安慰。"
                        "严格遵守 identity_kernel.style_profile；"
                        "forbidden_unobserved_markers 中的语气词不得出现在最终回复。"
                        "不得通过“我也正在……”或“我今天也……”虚构扮演对象"
                        "当前的身体状态、活动或所在地；共情不等于编造同样经历。"
                        "例如用户说“今天有点累，抱抱我”，应直接安慰或拥抱，"
                        "不得回答“我今天也累”或追问“你睡不着吗”。"
                        "只返回 JSON，verdict 只能是 approve 或 rewrite；"
                        "即使 approve 也必须返回最终 bubbles。"
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "context": ContextBuilder().to_review_payload(packet),
                            "draft": _reply_payload(draft),
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0,
        }
        try:
            response = self._client.post(
                self._endpoint,
                headers={"Authorization": f"Bearer {self._api_key}"},
                json=payload,
                timeout=self._timeout_seconds,
            )
            response.raise_for_status()
            body = response.json()
            content = body["choices"][0]["message"]["content"]
            reviewed = json.loads(content)
            verdict = reviewed["verdict"]
            if verdict not in {"approve", "rewrite"}:
                raise ValueError("复核 verdict 无效")
            reply = parse_reply_turn(
                json.dumps(
                    {"bubbles": reviewed["bubbles"]},
                    ensure_ascii=False,
                )
            )
            _validate_stickers(packet, reply)
            reasons = reviewed.get("reasons", [])
            if not isinstance(reasons, list):
                raise ValueError("复核 reasons 无效")
        except (
            httpx.HTTPError,
            json.JSONDecodeError,
            KeyError,
            IndexError,
            TypeError,
            ValueError,
            ReplyStructureError,
        ) as error:
            raise ReviewFailedError("云端回复复核失败") from error
        return ReviewResult(
            verdict=verdict,
            reasons=tuple(str(reason) for reason in reasons),
            reply=draft if verdict == "approve" else reply,
        )

    def review_fact_safety(
        self,
        packet: ContextPacket,
        draft: GeneratedReplyTurn,
        *,
        private_content: str,
    ) -> FactSafetyResult:
        """只核查未来泄漏、无证据现实断言和私密区泄漏。"""

        payload = {
            "model": self._model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "你是主体认知公开回复的事实安全审查器，只判断三项："
                        "no_future_leak 表示没有使用 cutoff 后信息或未来泄漏；"
                        "grounded_claims 表示没有无证据现实断言；"
                        "no_private_leak 表示没有私密认知泄漏到公开回复。"
                        "公开上下文中已有的信息不属于私密泄漏，即使私密认知也提到它；"
                        "主体基于私密认知规划后的自然公开回应不属于私密泄漏。"
                        "只有公开回复直接暴露了上下文中不存在的内部推理、隐藏状态、"
                        "审计信息或私密区原文时，no_private_leak 才应为 false。"
                        "不得因措辞、风格、亲密程度或是否需要润色而判失败。"
                        "只返回 JSON，必须包含这三个布尔值和 reasons 字符串数组。"
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "context": ContextBuilder().to_review_payload(packet),
                            "public_reply": _reply_payload(draft),
                            "private_cognition": private_content,
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0,
        }
        last_error: Exception | None = None
        for _attempt in range(self._max_attempts):
            try:
                response = self._client.post(
                    self._endpoint,
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    json=payload,
                    timeout=self._timeout_seconds,
                )
                response.raise_for_status()
                body = response.json()
                reviewed = json.loads(body["choices"][0]["message"]["content"])
                dimensions = (
                    reviewed["no_future_leak"],
                    reviewed["grounded_claims"],
                    reviewed["no_private_leak"],
                )
                reasons = reviewed["reasons"]
                if any(not isinstance(value, bool) for value in dimensions):
                    raise ValueError("事实安全维度必须为布尔值")
                if not isinstance(reasons, list) or any(
                    not isinstance(reason, str) for reason in reasons
                ):
                    raise ValueError("事实安全 reasons 无效")
                return FactSafetyResult(
                    no_future_leak=dimensions[0],
                    grounded_claims=dimensions[1],
                    no_private_leak=dimensions[2],
                    reasons=tuple(reasons),
                )
            except (
                httpx.HTTPError,
                json.JSONDecodeError,
                KeyError,
                IndexError,
                TypeError,
                ValueError,
            ) as error:
                last_error = error
        raise ReviewFailedError("云端事实安全复核失败") from last_error


def _reply_payload(reply: GeneratedReplyTurn) -> dict[str, object]:
    return {
        "bubbles": [
            {
                "type": bubble.type,
                "content": bubble.content,
                "asset_id": bubble.asset_id,
                "delay_ms": bubble.delay_ms,
            }
            for bubble in reply.bubbles
        ]
    }


def _validate_stickers(
    packet: ContextPacket,
    reply: GeneratedReplyTurn,
) -> None:
    allowed = set(packet.allowed_sticker_ids)
    if any(bubble.type == "sticker" and bubble.asset_id not in allowed for bubble in reply.bubbles):
        raise ValueError("复核返回了未授权表情资产")
