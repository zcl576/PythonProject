from redis import Redis
from langgraph.checkpoint.redis import RedisSaver

from app.config import get_settings
from functools import lru_cache


@lru_cache
def get_redis_saver() -> RedisSaver:
    settings = get_settings()

    client = Redis(
        host=settings.redis_host,
        port=settings.redis_port,
        password=settings.redis_password,
        ssl=settings.redis_ssl,
        ssl_cert_reqs=settings.redis_ssl_cert_reqs,
        decode_responses=False,
    )

    saver = RedisSaver(redis_client=client)
    saver.setup()
    return saver