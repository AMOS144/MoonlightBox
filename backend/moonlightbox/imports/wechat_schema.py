import sqlite3
from dataclasses import dataclass


@dataclass(frozen=True)
class WechatMediaSchema:
    table: str
    message_id_column: str
    md5_column: str | None
    path_column: str | None


class WechatSchemaAdapter:
    @staticmethod
    def detect(connection: sqlite3.Connection) -> WechatMediaSchema | None:
        tables = [
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
            )
        ]
        for table in tables:
            columns = [
                str(row[1])
                for row in connection.execute(f"PRAGMA table_info({_quote_identifier(table)})")
            ]
            by_key = {_column_key(column): column for column in columns}
            message_id = _first_column(
                by_key,
                ("localid", "msgid", "messageid", "localmsgid"),
            )
            md5 = _first_column(
                by_key,
                ("md5", "emojimd5", "resourcemd5", "imagemd5"),
            )
            path = _first_column(
                by_key,
                ("localpath", "filepath", "path", "thumbpath", "resourcepath"),
            )
            if message_id is not None and (md5 is not None or path is not None):
                return WechatMediaSchema(
                    table=table,
                    message_id_column=message_id,
                    md5_column=md5,
                    path_column=path,
                )
        return None


def _column_key(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())


def _first_column(
    columns: dict[str, str],
    aliases: tuple[str, ...],
) -> str | None:
    return next((columns[alias] for alias in aliases if alias in columns), None)


def _quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'
