"""提交工具与事务层共用的计划参数规则，不依赖模型或数据库写入。"""


def validate_plan_structure(proposal):
    cursor = "00:00"
    for index, block in enumerate(proposal.blocks):
        if block.start != cursor or block.end <= block.start:
            raise ValueError(f"blocks[{index}] 时间不连续或倒序；start 应为 {cursor}")
        if int(block.start[-2:]) % 15 or int(block.end[-2:]) % 15:
            raise ValueError(f"blocks[{index}].start/end 必须按 15 分钟对齐")
        if (
            block.basis
            in {
                "branch_commitment",
                "snapshot_routine",
                "historical_pattern",
                "profile_inference",
            }
            and not block.evidence_ids
        ):
            raise ValueError(f"blocks[{index}].evidence_ids 不能为空：非兜底计划需引用已有来源")
        cursor = block.end
    if cursor != "24:00":
        raise ValueError("计划必须连续覆盖 00:00–24:00")
