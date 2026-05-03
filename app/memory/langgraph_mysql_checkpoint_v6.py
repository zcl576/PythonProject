import aiomysql
from langgraph.checkpoint.mysql.aio import AIOMySQLSaver
from loguru import logger as log

from app.config import get_settings

_conn: aiomysql.Connection | None = None
_checkpointer: AIOMySQLSaver | None = None


async def get_v6_mysql_checkpointer() -> AIOMySQLSaver:
    global _conn, _checkpointer
    if _checkpointer is not None:
        return _checkpointer

    settings = get_settings()
    if not settings.mysql_host or not settings.mysql_user or not settings.mysql_db:
        raise ValueError("MySQL settings are incomplete")

    conn = await aiomysql.connect(
        host=settings.mysql_host,
        port=settings.mysql_port,
        user=settings.mysql_user,
        password=settings.mysql_password or "",
        db=settings.mysql_db,
        autocommit=True,
    )

    checkpointer = AIOMySQLSaver(conn)
    try:
        await checkpointer.setup()
    except Exception:
        log.error("failed to initialize LangGraph MySQL checkpointer", exc_info=True)
        conn.close()
        raise

    _conn = conn
    _checkpointer = checkpointer
    return checkpointer
