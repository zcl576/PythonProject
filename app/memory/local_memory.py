from functools import lru_cache

from langgraph.checkpoint.memory import InMemorySaver

from app.memory.memory import InMemorySessionStore


@lru_cache
def get_inmemory_session() -> InMemorySessionStore:
    return InMemorySessionStore()


@lru_cache
def get_inmemory_saver() -> InMemorySaver:
    return InMemorySaver()
