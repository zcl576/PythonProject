from langgraph.checkpoint.memory import InMemorySaver

from functools import lru_cache


@lru_cache
def get_inmemory_session() -> InMemorySaver:
    return InMemorySaver()