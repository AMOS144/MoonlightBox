from collections.abc import Iterator

from sqlalchemy import Engine, create_engine, event, text
from sqlalchemy.engine.interfaces import DBAPIConnection
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sqlalchemy.pool import ConnectionPoolEntry


class Base(DeclarativeBase):
    pass


def _configure_sqlite_connection(
    connection: DBAPIConnection,
    _: ConnectionPoolEntry,
    *,
    busy_timeout_ms: int,
    enable_wal: bool,
) -> None:
    cursor = connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute(f"PRAGMA busy_timeout={busy_timeout_ms}")
    if enable_wal:
        cursor.execute("PRAGMA journal_mode=WAL")
    cursor.close()


class Database:
    def __init__(
        self,
        database_url: str,
        *,
        busy_timeout_ms: int = 1000,
        lock_retries: int = 2,
    ) -> None:
        if busy_timeout_ms < 0:
            raise ValueError("busy_timeout_ms 不能为负数")
        if lock_retries < 0:
            raise ValueError("lock_retries 不能为负数")
        is_sqlite = database_url.startswith("sqlite")
        connect_args = (
            {
                "check_same_thread": False,
                "timeout": busy_timeout_ms / 1000,
            }
            if is_sqlite
            else {}
        )
        self.engine: Engine = create_engine(database_url, connect_args=connect_args)
        self.lock_retries = lock_retries
        if is_sqlite:
            is_memory = ":memory:" in database_url or "mode=memory" in database_url

            def configure(
                connection: DBAPIConnection,
                record: ConnectionPoolEntry,
            ) -> None:
                _configure_sqlite_connection(
                    connection,
                    record,
                    busy_timeout_ms=busy_timeout_ms,
                    enable_wal=not is_memory,
                )

            event.listen(self.engine, "connect", configure)
        self._session_factory = sessionmaker(bind=self.engine, expire_on_commit=False)

    def create_schema(self) -> None:
        Base.metadata.create_all(self.engine)

    def session(self) -> Iterator[Session]:
        with self._session_factory() as session:
            yield session

    def check_health(self) -> bool:
        try:
            with self.engine.connect() as connection:
                connection.execute(text("SELECT 1"))
        except Exception:
            return False
        return True

    def close(self) -> None:
        self.engine.dispose()
