from contextlib import ExitStack
from functools import lru_cache

from langgraph.checkpoint.redis import RedisSaver
from redis import Redis
from loguru import logger as log
from app.config import get_settings

_stack = ExitStack()


def _close_redis_client(client: Redis) -> None:
    client.close()
    client.connection_pool.disconnect()


@lru_cache(maxsize=1)
def get_redis_saver() -> RedisSaver:
    settings = get_settings()
    if not settings.redis_host:
        raise ValueError("Redis settings are incomplete")

    client = Redis(
        host=settings.redis_host,
        port=settings.redis_port,
        password=settings.redis_password,
        ssl=settings.redis_ssl,
        ssl_cert_reqs=settings.redis_ssl_cert_reqs,
        decode_responses=False,
    )

    saver = RedisSaver(redis_client=client)
    try:
        saver.setup()
    except Exception:
        log.error("redis操作失败", exc_info=True)
        _close_redis_client(client)
        raise

    _stack.callback(_close_redis_client, client)
    return saver
