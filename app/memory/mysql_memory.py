from contextlib import ExitStack
from functools import lru_cache

import pymysql
from langgraph.checkpoint.mysql.pymysql import PyMySQLSaver

from app.config import get_settings

_stack = ExitStack()


@lru_cache(maxsize=1)
def get_mysql_saver() -> PyMySQLSaver:
    settings = get_settings()
    if not settings.mysql_host or not settings.mysql_user or not settings.mysql_db:
        raise ValueError("MySQL settings are incomplete")

    conn = pymysql.connect(
        host=settings.mysql_host,
        port=settings.mysql_port,
        user=settings.mysql_user,
        password=settings.mysql_password or "",
        database=settings.mysql_db,
        autocommit=True,
    )

    saver = PyMySQLSaver(conn)
    try:
        saver.setup()
    except Exception:
        conn.close()
        raise

    _stack.callback(conn.close)
    return saver
