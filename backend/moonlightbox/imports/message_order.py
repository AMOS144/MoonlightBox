"""起点边界与来源索引共用的同秒顺序，不用随机 UUID 代替导入顺序。"""


def message_order_key(timestamp, import_id: str, source_id: str, message_id: str):
    return (
        timestamp,
        import_id,
        (0, int(source_id)) if source_id.isdecimal() else (1, source_id),
        message_id,
    )
