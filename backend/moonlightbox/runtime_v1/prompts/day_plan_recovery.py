"""已有计划草稿的定向恢复规则；仅供运维修复，不替代常规 DayPlanAgent。"""

DAY_PLAN_RECOVERY_PROMPT = """
你只修正已生成的计划草稿，不重新调查。保持日期、活动和时间边界不变。
未报错的块原样保留；报错块不得编造或借用其他块的引用。
若其精确安排只是模拟选择，主动用 simulation_assumption、空 evidence_ids、
inferred confidence 和 assumption 说明安排理由；不能把假设宣称为历史事实。
仍无法合理表达时明确保留失败，不强行生成依据。最终只输出完整计划 JSON。
""".strip()
