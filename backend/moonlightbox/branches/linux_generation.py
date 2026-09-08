"""Linux 分支回复生成入口。

解析、风格守卫和降级策略沿用已有业务实现；只替换具体模型 runtime。
"""

from __future__ import annotations

from moonlightbox.agent.linux_inference import SharedLinuxModelRuntime
from moonlightbox.branches.mlx_generation import DatabaseReplyGenerator
from moonlightbox.db import Database


class DatabaseLinuxGenerator(DatabaseReplyGenerator):
    """使用 Transformers + PEFT 的数据库绑定回复生成器。"""

    def __init__(
        self,
        database: Database,
        *,
        device: str = "auto",
        load_in_4bit: bool = True,
    ) -> None:
        super().__init__(
            database,
            SharedLinuxModelRuntime(device=device, load_in_4bit=load_in_4bit),
        )
