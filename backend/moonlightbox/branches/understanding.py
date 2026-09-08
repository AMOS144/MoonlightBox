import re
from collections.abc import Sequence

from moonlightbox.branches.context import ContextMemory, ContextPacket
from moonlightbox.branches.memory_policy import (
    is_current_state_question,
    retrieval_policy,
)
from moonlightbox.branches.replies import GeneratedReplyTurn
from moonlightbox.branches.situational_state import (
    active_situational_state,
    natural_situational_context,
)
from moonlightbox.training.style_profile import style_violations

_PLACE_PATTERN = re.compile(
    r"(?:北京|上海|广州|深圳|杭州|成都|重庆|武汉|南京|苏州|西安|长沙|天津|"
    r"香港|澳门|台湾|韩国|日本|首尔|东京|大阪|济州岛|釜山|美国|英国|法国|"
    r"德国|意大利|加拿大|澳大利亚|新加坡|泰国|越南|"
    r"[\u4e00-\u9fff]{2,8}(?:省|市|县|机场|医院|公司|学校))"
)


def requires_cloud_review(packet: ContextPacket) -> bool:
    if packet.conversation_mode == "proactive" or packet.contested_beliefs:
        return True
    understanding = packet.shared_ground.get("understanding")
    if not isinstance(understanding, dict):
        return False
    risky_state_question = (
        understanding.get("asks_unverified_current_state") is True
        and active_situational_state(packet.branch_state) is None
    )
    return risky_state_question or understanding.get("asks_unverified_history") is True or any(
        understanding.get(key) is True
        for key in ("commitment_conflict", "relationship_conflict", "interaction_feedback")
    )


def understand_contributions(content: str) -> dict[str, object]:
    concern = any(marker in content for marker in ("担心", "害怕", "不放心"))
    uncertainty = any(
        marker in content
        for marker in ("纠结", "不知道", "不确定", "拿不定", "没想好", "选择困难")
    )
    commitment_conflict = any(
        marker in content
        for marker in ("明明保证", "明明答应", "之前保证", "之前答应", "说好了")
    )
    asks_reason = any(marker in content for marker in ("为什么", "为何", "怎么不"))
    asks_current_state = is_current_state_question(content)
    asks_unverified_history = bool(
        any(marker in content for marker in ("以前", "上次", "那次", "记得", "去过"))
        and content.rstrip().endswith(("吗", "呢", "？", "?"))
    )
    conflict = any(
        marker in content
        for marker in ("不乖", "没听", "生气", "臭", "骗我")
    )
    interaction_feedback = any(
        marker in content
        for marker in ("说话非常冷漠", "说话很冷漠", "太冷漠", "很敷衍", "太敷衍")
    )
    obligations: list[str] = []
    if concern:
        obligations.append("回应担心")
    if commitment_conflict:
        obligations.append("处理违背承诺")
    if asks_reason:
        obligations.append("回答原因问题")
    if conflict:
        obligations.append("承接关系冲突")
    if asks_current_state:
        obligations.append("核实当前状态后再回答")
    if asks_unverified_history:
        obligations.append("核实对方提出的旧事后再回答")
    if interaction_feedback:
        obligations.append("修复当前沟通方式")
    if uncertainty:
        obligations.append("承接犹豫")
    emotions: list[str] = []
    if concern:
        emotions.append("担心")
    if uncertainty:
        emotions.append("纠结")
    return {
        "emotions": emotions,
        "decision_uncertainty": uncertainty,
        "commitment_conflict": commitment_conflict,
        "relationship_conflict": conflict,
        "asks_reason": asks_reason,
        "asks_unverified_current_state": asks_current_state,
        "asks_unverified_history": asks_unverified_history,
        "interaction_feedback": interaction_feedback,
        "response_obligations": obligations,
    }


def active_belief_content_draft(
    active_beliefs: Sequence[ContextMemory],
    current_user_content: str,
) -> str | None:
    """Compile a trusted relationship stance before persona style generation."""

    if retrieval_policy(current_user_content).intent != "relationship":
        return None
    relationship_markers = (
        "关系",
        "分手",
        "和好",
        "距离",
        "联系",
        "冷静",
        "重新开始",
        "继续",
    )
    relevant = next(
        (
            memory.content.strip()
            for memory in reversed(active_beliefs)
            if memory.authority == "subjective"
            and any(marker in memory.content for marker in relationship_markers)
        ),
        "",
    )
    if not relevant:
        return None
    draft = re.sub(r"^本人目前", "我现在", relevant)
    draft = re.sub(r"^本人当前", "我现在", draft)
    draft = re.sub(r"^本人", "我", draft)
    return draft.strip() or None


def validate_grounded_reply(
    packet: ContextPacket,
    reply: GeneratedReplyTurn,
) -> None:
    validate_fact_grounded_reply(packet, reply)
    content = "\n".join(bubble.content or "" for bubble in reply.bubbles)
    profile = packet.identity_kernel.get("style_profile")
    violations = style_violations(
        content,
        profile if isinstance(profile, dict) else None,
    )
    if violations:
        raise ValueError("回复使用了真实语料中未观察到的语气标记")


def validate_fact_grounded_reply(
    packet: ContextPacket,
    reply: GeneratedReplyTurn,
) -> None:
    content = "\n".join(bubble.content or "" for bubble in reply.bubbles)
    intent = retrieval_policy(packet.current_user_content).intent
    understanding = packet.shared_ground.get("understanding")
    unverified_state = (
        isinstance(understanding, dict)
        and understanding.get("asks_unverified_current_state") is True
    )
    active_state = active_situational_state(packet.branch_state)
    objective_evidence_text = "\n".join(
        [
            *(
                memory.content
                for memory in (*packet.memories, *packet.continuity_memories)
                if memory.authority in {"verified_history", "observed_interaction"}
            ),
            *natural_situational_context(active_state),
        ]
    )
    utterance_evidence_text = "\n".join(
        [
            *(
                bubble.content or ""
                for turn in packet.history
                if turn.role == "assistant"
                for bubble in turn.bubbles
            ),
            *(
                memory.content
                for memory in (*packet.memories, *packet.continuity_memories)
                if memory.authority == "conversation_record"
            ),
        ]
    )
    objective_evidence_text = objective_evidence_text.replace(packet.persona, "本人")
    utterance_evidence_text = utterance_evidence_text.replace(packet.persona, "本人")
    claim_evidence_text = objective_evidence_text + "\n" + utterance_evidence_text
    unsupported_third_party_claim = _unsupported_third_party_claim(
        content,
        claim_evidence_text,
    )
    if unsupported_third_party_claim is not None:
        raise ValueError("回复包含无证据第三方说法：" + unsupported_third_party_claim)
    unsupported_identity = _unsupported_person_identity(
        content,
        claim_evidence_text,
    )
    if unsupported_identity is not None:
        raise ValueError("回复包含无证据人物身份：" + unsupported_identity)
    if intent == "preference":
        expressed_preferences = packet.identity_kernel.get("expressed_preferences")
        preference_sources = [
            *(
                item
                for item in (
                    expressed_preferences
                    if isinstance(expressed_preferences, list)
                    else []
                )
                if isinstance(item, str) and item.strip()
            ),
            *(memory.content for memory in packet.active_beliefs),
            *(
                memory.content
                for memory in (*packet.memories, *packet.continuity_memories)
                if memory.authority
                in {
                    "subjective",
                    "conversation_record",
                    "observed_interaction",
                    "verified_history",
                }
            ),
        ]
        preference_evidence = bool(
            isinstance(expressed_preferences, list)
            and any(
                isinstance(item, str) and item.strip()
                for item in expressed_preferences
            )
        ) or any(
            memory.authority
            in {
                "subjective",
                "conversation_record",
                "observed_interaction",
                "verified_history",
            }
            for memory in (*packet.memories, *packet.continuity_memories)
        ) or bool(packet.active_beliefs)
        preference_uncertainty = any(
            marker in content
            for marker in (
                "不知道",
                "不确定",
                "没什么特别",
                "没有特别",
                "都行",
                "一般",
                "还好",
                "你呢",
                "你猜",
                "？",
                "?",
            )
        )
        preference_denial = bool(
            re.search(r"(?:不喜欢|不讨厌|没习惯|没有|不是)", content)
        )
        if (
            content.strip()
            and not preference_evidence
            and not preference_uncertainty
            and not preference_denial
        ):
            raise ValueError("回复包含无证据稳定偏好")
        preference_payloads = _preference_payloads(content)
        normalized_preference_evidence = tuple(
            payload
            for source in preference_sources
            for payload in _preference_payloads(source)
        )
        if (
            preference_evidence
            and preference_payloads
            and any(
                not any(
                    preference_payload in evidence_payload
                    or evidence_payload in preference_payload
                    for evidence_payload in normalized_preference_evidence
                )
                for preference_payload in preference_payloads
            )
            and not _relationship_belief_paraphrase_matches(
                content,
                tuple(memory.content for memory in packet.active_beliefs),
            )
            and not preference_uncertainty
            and not preference_denial
        ):
            raise ValueError("偏好回答超出证据")
    if intent == "relationship" and packet.active_beliefs:
        relationship_evidence = "\n".join(
            memory.content for memory in packet.active_beliefs
        )
        normalized_relationship_reply = _normalize_state_evidence(content)
        normalized_relationship_evidence = _normalize_state_evidence(
            relationship_evidence
        )
        directly_grounded = bool(
            normalized_relationship_reply
            and normalized_relationship_evidence
            and (
                normalized_relationship_reply in normalized_relationship_evidence
                or normalized_relationship_evidence in normalized_relationship_reply
            )
        )
        if not directly_grounded and not _relationship_belief_paraphrase_matches(
            content,
            tuple(memory.content for memory in packet.active_beliefs),
        ):
            raise ValueError("关系回答没有承接当前激活信念")
    uncertainty_markers = (
        "不知道",
        "不清楚",
        "不确定",
        "不记得",
        "记不清",
        "说不好",
        "等会",
        "晚点",
        "你猜",
        "不告诉你",
        "干嘛",
        "怎么了",
        "秘密",
        "？",
        "?",
    )
    normalized_content = _normalize_state_evidence(content)
    normalized_evidence = _normalize_state_evidence(objective_evidence_text)
    grounded_in_general_evidence = bool(
        normalized_content
        and normalized_evidence
        and (
            normalized_content in normalized_evidence
            or normalized_evidence in normalized_content
        )
    )
    state_values = active_state.get("values") if active_state is not None else None
    grounded_in_current_state = False
    if isinstance(state_values, dict):
        grounded_in_current_state = any(
            normalized_value
            and normalized_value in normalized_content
            for value in state_values.values()
            if isinstance(value, str)
            and (normalized_value := _normalize_state_evidence(value))
        )
        grounded_in_current_state = grounded_in_current_state or _situational_paraphrase_matches(
            content,
            state_values,
        )
    grounded_in_evidence = (
        grounded_in_current_state if unverified_state else grounded_in_general_evidence
    )
    grounded_in_evidence = grounded_in_evidence or _grounded_by_current_interaction(content)
    if (
        unverified_state
        and not grounded_in_evidence
        and re.search(
            r"(没停|停了|已经停|早不碰|不碰了|没再碰|"
            r"还在(?:炒|做|吃|看|上班|外面)|"
            r"我(?:现在|今天|刚刚|刚才|目前)(?:在|正|已经))",
            content,
        )
    ):
        raise ValueError("回复包含无证据当前状态")
    if (
        unverified_state
        and not grounded_in_evidence
        and not any(marker in content for marker in uncertainty_markers)
    ):
        raise ValueError("回复包含无证据当前状态")
    unsupported_activity = re.search(
        r"(我(?:正在|刚|刚刚|刚才|现在|今天|目前|还在|已经)"
        r"[^，。！？\n]{0,12}"
        r"(?:吃|看|上班|下班|工作|开会|开车|洗澡|炒股|睡觉|休息|"
        r"生气|难过|开心|哭|忙|等人|回家|出门|在路上|在公司|在学校|"
        r"在医院|在家|睡不着|累))",
        content,
    )
    if (
        unsupported_activity
        and _normalize_state_evidence(unsupported_activity.group(1))
        not in "".join(
            _normalize_state_evidence(value)
            for value in (state_values or {}).values()
            if isinstance(value, str)
        )
    ):
        raise ValueError("回复包含无证据当前状态")
    unsupported_shared_history = re.search(
        r"((?:我们|咱们|你和我|我和你|跟你|和你|一起)"
        r"[^。！？\n]{0,24}"
        r"(?:去过|见过|做过|走过|吃过|看过|住过|玩过|经历过|那天))",
        content,
    )
    if (
        unsupported_shared_history
        and _normalize_state_evidence(unsupported_shared_history.group(1))
        not in normalized_evidence
    ):
        raise ValueError("回复包含无证据共同经历")
    unsupported_acquaintance = re.search(
        r"((?:是|就在|在)?[^，。！？\n]{0,16}"
        r"(?<!没)(?<!不)(?:遇见|碰见|遇到|碰到|认识|见到)(?:过|的)?)",
        content,
    )
    if (
        unsupported_acquaintance
        and _normalize_claim_evidence(unsupported_acquaintance.group(1))
        not in _normalize_claim_evidence(
            objective_evidence_text + "\n" + utterance_evidence_text
        )
        and not unsupported_acquaintance.group(1).rstrip().endswith(
            ("吗", "呢", "哪", "哪里", "？", "?")
        )
    ):
        raise ValueError("回复包含无证据相识经过")
    environment_values = "".join(
        str(value)
        for key, value in (state_values or {}).items()
        if key == "environment"
    )
    if (
        re.search(r"(?:天气|下雨|下雪|太阳|刮风).{0,8}(?:不错|很好|好|冷|热|大)", content)
        and not environment_values
    ):
        raise ValueError("回复包含无证据外部环境")
    unsupported_schedule = re.search(
        r"((?:今天|明天|等会|一会|待会|这次|马上|就要|准备|打算)"
        r"[^。！？\n]{0,24}(?:去|飞|回|住|待|呆|带|买|见|出发))",
        content,
    )
    if (
        unsupported_schedule
        and _normalize_state_evidence(unsupported_schedule.group(1))
        not in normalized_evidence
    ):
        raise ValueError("回复包含无证据时间或计划")
    unsupported_places = tuple(
        place
        for place in dict.fromkeys(_PLACE_PATTERN.findall(content))
        if _normalize_state_evidence(place) not in normalized_evidence
        and not _place_is_nonassertive_response(
            place,
            content,
            packet.current_user_content,
        )
    )
    if unsupported_places:
        raise ValueError("回复包含无证据地点：" + "、".join(unsupported_places))
    unsupported_commitment = re.search(
        r"((?:(?:我)?(?:之前|当时|明明)?"
        r"(?:答应(?:过|了)|保证(?:过|了)|承诺(?:过|了))"
        r"|我(?:之前|当时|明明)?说过)"
        r"[^。！？\n]{0,24})",
        content,
    )
    if (
        unsupported_commitment
        and not _commitment_supported(
            unsupported_commitment.group(1),
            objective_evidence_text + "\n" + utterance_evidence_text,
            packet.current_user_content,
        )
    ):
        raise ValueError("回复承认了无证据历史承诺")
    unsupported_interaction = re.search(
        r"((?:你|对方)[^，。！？\n]{0,12}(?:没|没有|不)(?:回|回复|接|理|联系)"
        r"|我(?:就|所以|后来|已经)[^，。！？\n]{0,16}"
        r"(?:发|打|改成|换成)[^，。！？\n]{0,6}(?:语音|电话|视频|消息))",
        content,
    )
    interaction_evidence = (
        objective_evidence_text
        + "\n"
        + utterance_evidence_text
        + "\n"
        + packet.current_user_content
    )
    if (
        intent in {"commitment", "prior_utterance"}
        and unsupported_interaction
        and _normalize_state_evidence(unsupported_interaction.group(1))
        not in _normalize_state_evidence(interaction_evidence)
    ):
        raise ValueError("回复包含无证据互动经过")
    for place in _PLACE_PATTERN.findall(content):
        if (
            re.search(rf"你[^。！？\n]{{0,16}}{re.escape(place)}", content)
            and re.search(
                rf"本人[^。！？\n]{{0,24}}{re.escape(place)}",
                objective_evidence_text,
            )
            and not re.search(
                rf"用户[^。！？\n]{{0,24}}{re.escape(place)}",
                objective_evidence_text,
            )
        ):
            raise ValueError("回复混淆了本人和用户的经历归属")


def grounded_retry_instruction(
    error: ValueError,
    packet: ContextPacket | None = None,
) -> str:
    """为本地二次生成提供事实约束，不让外部模型代写人物语气。"""

    if "无证据当前状态" in str(error):
        trusted_state = (
            natural_situational_context(active_situational_state(packet.branch_state))
            if packet is not None
            else ()
        )
        if trusted_state:
            return (
                "上一版回答与本轮可信状态冲突。重新回复最新消息："
                "只能按这条可信状态回答，不得反转，也不得增加原因、地点或后续事件："
                + "；".join(trusted_state)
                + "。不要解释规则，仍只用本人平时的微信表达。"
            )
        return (
            "上一版凭空声称了你此刻的活动、地点或情绪。重新回复最新消息："
            "只能回应对方已经说出的内容；证据没有明确写出的本人当前状态一律不要说。"
            "不要回答是或否，可以用本人平时的语气反问对方为什么突然问，"
            "或者简短说你猜、干嘛。"
            "不要解释规则，仍只用本人平时的微信表达。"
        )
    if any(
        marker in str(error)
        for marker in (
            "无证据共同经历",
            "无证据外部环境",
            "无证据时间或计划",
            "无证据地点",
            "无证据相识经过",
        )
    ):
        return (
            "上一版凭空增加了共同经历、时间计划或此刻外部环境。重新回复最新消息："
            "只承接背景明确写出的经历；没有证据的地点、天气、时间、行程长度"
            "和一起做过的事情不要说。"
            "对方追问不存在的旧事时，可以自然说不记得、记不清或反问是哪次。"
            "不要解释规则，仍只用本人平时的微信表达。"
        )
    if "无证据历史承诺" in str(error) or "混淆了本人和用户" in str(error):
        return (
            "上一版承认了证据中不存在的承诺，或把本人经历说成了对方经历。"
            "重新回复最新消息：不接受对方擅自定义旧承诺；分清本人和对方，"
            "可以自然质疑、否认或让对方说清楚。不要解释规则。"
        )
    if "无证据稳定偏好" in str(error) or "偏好回答超出证据" in str(error):
        return (
            "上一版临时创造了本人没有证据的固定喜好或习惯。重新回复最新消息："
            "只沿用背景中本人真实说过的偏好；没有证据时可以自然说没有特别偏好、"
            "不确定、你猜，或者反问对方。不要解释规则。"
        )
    if "无证据第三方说法" in str(error) or "无证据人物身份" in str(error):
        return (
            "上一版替第三个人补出了证据里没有的评价、原话、职业或关系。"
            "重新回复最新消息：只有聊天记录明确出现的第三人说法和身份才能确认；"
            "没有证据时只能自然说不知道、没听他说、没见过，或反问对方是谁。"
            "不要猜测，不要解释规则，仍只用本人平时的微信表达。"
        )
    return (
        "上一版用了本人真实语料中没有的套话。重新回复最新消息，"
        "事实和态度不变，只用本人语料里出现过的自然微信表达；不要解释规则。"
    )


def _normalize_state_evidence(content: str) -> str:
    normalized = re.sub(
        r"(?:我|现在|正在|正|刚刚|刚才|刚|目前|今天|还在|已经)",
        "",
        content,
    )
    return re.sub(r"[\s，。！？!?、；;：:]", "", normalized)


def _normalize_claim_evidence(content: str) -> str:
    return re.sub(r"[\s，。！？!?、；;：:\"'“”‘’（）()]", "", content)


def _unsupported_third_party_claim(content: str, evidence: str) -> str | None:
    """Return an attributed claim that the persona never actually observed saying."""

    normalized_evidence = _normalize_claim_evidence(evidence)
    pattern = re.compile(
        r"(?:他|她|那个人|这个人|对方)[^，。！？\n]{0,10}"
        r"(?:说|觉得|认为|评价(?:是|为)?)"
        r"(?P<claim>[^，。！？\n]{1,28})"
    )
    for match in pattern.finditer(content):
        full_claim = match.group(0).strip()
        payload = match.group("claim").strip()
        if re.search(r"(?:没|没有|从没|未曾)(?:跟我)?(?:说|提|评价)", full_claim):
            continue
        if payload.startswith(("什么", "啥", "怎么", "如何", "哪", "谁")):
            continue
        if full_claim.rstrip().endswith(("吗", "呢", "？", "?")):
            continue
        normalized_payload = _normalize_claim_evidence(payload)
        if len(normalized_payload) < 2:
            continue
        if normalized_payload not in normalized_evidence:
            return full_claim
    return None


def _unsupported_person_identity(content: str, evidence: str) -> str | None:
    """Reject newly invented third-person jobs and relationship identities."""

    identity_pattern = re.compile(
        r"(?P<claim>(?:他|她|那个人|这个人|对方)?"
        r"(?<!不)(?<!没)(?:就是|是)"
        r"(?:公司(?:里|新来的)?|新来的|我的|你们?的)?"
        r"(?P<role>实习生|同事|同学|朋友|前任|亲戚|家人|客户|老板|"
        r"老师|学生|室友|邻居|对象|男朋友|女朋友|新人))"
    )
    normalized_evidence = _normalize_claim_evidence(evidence)
    for match in identity_pattern.finditer(content):
        full_claim = match.group("claim").strip()
        role = match.group("role")
        if full_claim.rstrip().endswith(("吗", "呢", "？", "?")):
            continue
        if role not in normalized_evidence:
            return full_claim
    return None


def _preference_payloads(content: str) -> tuple[str, ...]:
    payloads: list[str] = []
    for clause in re.split(r"[，。！？!?、；;：:\n]", content):
        normalized = re.sub(
            r"(?:本人|目标|用户|我|你|最|一直|挺|很|比较|不太|"
            r"喜欢|爱吃|爱喝|爱看|讨厌|不喜欢|习惯|以前|说过|"
            r"就是|这个|那个|呀|啊|呢|啦|哦)",
            "",
            clause,
        )
        normalized = re.sub(r"\s+", "", normalized)
        if len(normalized) >= 2:
            payloads.append(normalized)
    return tuple(payloads)


def _relationship_belief_paraphrase_matches(
    reply: str,
    active_beliefs: tuple[str, ...],
) -> bool:
    """Accept ordinary paraphrases of an explicit active relationship stance."""

    evidence = "\n".join(active_beliefs)
    if not evidence:
        return False
    distance_evidence_markers = (
        "保持距离",
        "需要距离",
        "先冷静",
        "冷静一段",
        "少联系",
        "别联系",
        "不联系",
        "缓一缓",
        "需要空间",
    )
    distance_reply_markers = (
        "距离",
        "冷静一段",
        "少联系",
        "别联系",
        "不联系",
        "缓一缓",
        "需要空间",
    )
    repair_evidence_markers = (
        "修复关系",
        "愿意修复",
        "想修复",
        "愿意和好",
        "想和好",
        "重新开始",
        "继续这段关系",
    )
    repair_reply_markers = ("修复", "和好", "重新开始", "继续这段关系")
    return (
        any(marker in evidence for marker in distance_evidence_markers)
        and any(marker in reply for marker in distance_reply_markers)
    ) or (
        any(marker in evidence for marker in repair_evidence_markers)
        and any(marker in reply for marker in repair_reply_markers)
    )


def _commitment_supported(claim: str, evidence: str, question: str) -> bool:
    strip_markers = re.compile(
        r"(?:本人|用户|我|你|之前|当时|明明|已经|曾经|确实|"
        r"答应(?:过|了)?|保证(?:过|了)?|承诺(?:过|了)?|说过|说好|"
        r"说[:：]?|永远|吗|呢|呀)"
    )
    proposition = strip_markers.sub("", _normalize_state_evidence(claim))
    if len(proposition) < 2:
        proposition = strip_markers.sub("", _normalize_state_evidence(question))
    normalized_evidence = strip_markers.sub("", _normalize_state_evidence(evidence))
    if "说过" in claim and not re.search(r"(?:答应|保证|承诺|说好)", claim):
        return len(proposition) >= 2 and proposition in normalized_evidence
    if not re.search(r"(?:答应|保证|承诺|说好)", evidence):
        return False
    return len(proposition) >= 2 and proposition in normalized_evidence


def _place_is_nonassertive_response(
    place: str,
    reply: str,
    current_user_content: str,
) -> bool:
    """Allow denying or questioning a place the user just introduced."""

    if place not in current_user_content:
        return False
    position = reply.find(place)
    window = reply[max(0, position - 12) : position + len(place) + 12]
    return any(
        marker in window
        for marker in (
            "没去过",
            "没有去过",
            "不记得",
            "记不清",
            "哪次",
            "什么时候",
            "什么",
            "？",
            "?",
        )
    )


def _grounded_by_current_interaction(content: str) -> bool:
    return any(
        marker in content
        for marker in (
            "回你消息",
            "跟你聊天",
            "和你聊天",
            "等你发消息",
            "等你回消息",
            "在等你",
            "等你呢",
            "看你消息",
        )
    )


def _situational_paraphrase_matches(
    content: str,
    values: dict[str, object],
) -> bool:
    availability = values.get("availability")
    if isinstance(availability, str):
        unavailable = any(
            marker in availability for marker in ("不方便", "不可以", "不能", "不接")
        )
        if unavailable and any(
            marker in content for marker in ("不方便", "不可以", "不能", "不接")
        ):
            return True
        available = any(marker in availability for marker in ("方便", "可以", "能接"))
        if available and not unavailable and any(
            marker in content for marker in ("方便", "可以", "能接")
        ):
            return True
    return False
