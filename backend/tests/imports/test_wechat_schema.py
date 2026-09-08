import sqlite3
from pathlib import Path


def test_schema_adapter_detects_media_columns_without_fixed_table_name(
    tmp_path: Path,
) -> None:
    from moonlightbox.imports.wechat_schema import WechatSchemaAdapter

    database_path = tmp_path / "message.db"
    connection = sqlite3.connect(database_path)
    connection.execute(
        "CREATE TABLE strange_resource_table (local_id INTEGER, emoji_md5 TEXT, local_path TEXT)"
    )
    connection.commit()

    schema = WechatSchemaAdapter.detect(connection)

    assert schema is not None
    assert schema.table == "strange_resource_table"
    assert schema.message_id_column == "local_id"
    assert schema.md5_column == "emoji_md5"
    assert schema.path_column == "local_path"
