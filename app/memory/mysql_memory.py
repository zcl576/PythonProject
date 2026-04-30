import aiomysql
from langgraph.checkpoint.mysql.aio import AIOMySQLSaver
from loguru import logger as log

from app.config import get_settings

_conn: aiomysql.Connection | None = None
_saver: AIOMySQLSaver | None = None


async def get_mysql_saver() -> AIOMySQLSaver:
    global _conn, _saver
    if _saver is not None:
        return _saver

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

    saver = AIOMySQLSaver(conn)
    try:
        await saver.setup()
    except Exception:
        log.error("mysql操作失败", exc_info=True)
        conn.close()
        raise

    _conn = conn
    _saver = saver
    return saver
