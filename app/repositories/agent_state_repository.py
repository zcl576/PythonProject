from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any

import aiomysql

from app.config import get_settings


class AgentStateRepository:
    def __init__(self) -> None:
        self._ready = False

    async def setup(self) -> None:
        if self._ready:
            return
        conn = await self._connection()
        try:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS agent_threads (
                        session_id VARCHAR(128) PRIMARY KEY,
                        project_id BIGINT NOT NULL,
                        user_id VARCHAR(128) NULL,
                        status VARCHAR(32) NOT NULL DEFAULT 'active',
                        current_state VARCHAR(64) NOT NULL DEFAULT 'idle',
                        created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
                            ON UPDATE CURRENT_TIMESTAMP,
                        INDEX idx_project_user (project_id, user_id),
                        INDEX idx_status_updated (status, updated_at)
                    )
                    """
                )
                await cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS agent_state_snapshots (
                        session_id VARCHAR(128) PRIMARY KEY,
                        state VARCHAR(64) NOT NULL,
                        pending_action VARCHAR(128) NULL,
                        slots JSON NULL,
                        choices JSON NULL,
                        confirm JSON NULL,
                        metadata JSON NULL,
                        expires_at DATETIME NULL,
                        version INT NOT NULL DEFAULT 1,
                        updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
                            ON UPDATE CURRENT_TIMESTAMP
                    )
                    """
                )
            await conn.commit()
        finally:
            conn.close()
        self._ready = True

    async def ensure_thread(self, session_id: str, project_id: int) -> None:
        await self.setup()
        conn = await self._connection()
        try:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO agent_threads (session_id, project_id, current_state)
                    VALUES (%s, %s, 'idle')
                    ON DUPLICATE KEY UPDATE
                        project_id = VALUES(project_id),
                        updated_at = CURRENT_TIMESTAMP
                    """,
                    (session_id, project_id),
                )
            await conn.commit()
        finally:
            conn.close()

    async def save_snapshot(
        self,
        *,
        session_id: str,
        state: str,
        pending_action: str | None = None,
        slots: dict[str, Any] | None = None,
        choices: list[dict[str, Any]] | None = None,
        confirm: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        expires_in_minutes: int = 30,
    ) -> None:
        await self.setup()
        expires_at = datetime.now() + timedelta(minutes=expires_in_minutes)
        conn = await self._connection()
        try:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO agent_state_snapshots
                        (session_id, state, pending_action, slots, choices, confirm,
                         metadata, expires_at, version)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 1)
                    ON DUPLICATE KEY UPDATE
                        state = VALUES(state),
                        pending_action = VALUES(pending_action),
                        slots = VALUES(slots),
                        choices = VALUES(choices),
                        confirm = VALUES(confirm),
                        metadata = VALUES(metadata),
                        expires_at = VALUES(expires_at),
                        version = version + 1
                    """,
                    (
                        session_id,
                        state,
                        pending_action,
                        self._dump(slots or {}),
                        self._dump(choices or []),
                        self._dump(confirm),
                        self._dump(metadata or {}),
                        expires_at,
                    ),
                )
                await cur.execute(
                    """
                    UPDATE agent_threads
                    SET current_state = %s, updated_at = CURRENT_TIMESTAMP
                    WHERE session_id = %s
                    """,
                    (state, session_id),
                )
            await conn.commit()
        finally:
            conn.close()

    async def load_snapshot(self, session_id: str) -> dict[str, Any] | None:
        await self.setup()
        conn = await self._connection()
        try:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                await cur.execute(
                    """
                    SELECT session_id, state, pending_action, slots, choices, confirm,
                           metadata, expires_at, version
                    FROM agent_state_snapshots
                    WHERE session_id = %s
                    """,
                    (session_id,),
                )
                row = await cur.fetchone()
        finally:
            conn.close()
        if row is None:
            return None
        expires_at = row.get("expires_at")
        if expires_at and expires_at < datetime.now():
            return None
        return {
            "session_id": row["session_id"],
            "state": row["state"],
            "pending_action": row.get("pending_action"),
            "slots": self._load(row.get("slots"), {}),
            "choices": self._load(row.get("choices"), []),
            "confirm": self._load(row.get("confirm"), None),
            "metadata": self._load(row.get("metadata"), {}),
            "expires_at": expires_at,
            "version": row.get("version"),
        }

    async def clear_snapshot(self, session_id: str, state: str = "done") -> None:
        await self.setup()
        conn = await self._connection()
        try:
            async with conn.cursor() as cur:
                await cur.execute(
                    "DELETE FROM agent_state_snapshots WHERE session_id = %s",
                    (session_id,),
                )
                await cur.execute(
                    """
                    UPDATE agent_threads
                    SET current_state = %s, updated_at = CURRENT_TIMESTAMP
                    WHERE session_id = %s
                    """,
                    (state, session_id),
                )
            await conn.commit()
        finally:
            conn.close()

    async def _connection(self) -> aiomysql.Connection:
        settings = get_settings()
        if not settings.mysql_host or not settings.mysql_user or not settings.mysql_db:
            raise ValueError("MySQL settings are incomplete")
        return await aiomysql.connect(
            host=settings.mysql_host,
            port=settings.mysql_port,
            user=settings.mysql_user,
            password=settings.mysql_password or "",
            db=settings.mysql_db,
            autocommit=False,
        )

    @staticmethod
    def _dump(value: Any) -> str | None:
        if value is None:
            return None
        return json.dumps(value, ensure_ascii=False)

    @staticmethod
    def _load(value: Any, default: Any) -> Any:
        if value in (None, ""):
            return default
        if isinstance(value, (dict, list)):
            return value
        return json.loads(value)
