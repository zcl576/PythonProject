from typing import Any

class InMemorySessionStore:
    """内存会话存储
    
    在内存中存储会话数据，用于在多次请求之间保留和合并会话上下文。
    """
    def __init__(self) -> None:
        """初始化内存会话存储
        
        创建一个空的会话字典，用于存储会话数据。
        """
        # 会话字典，键为会话ID，值为会话数据字典
        self._sessions: dict[str, dict[str, Any]] = {}

    def merge(self, session_id: str, current: dict[str, Any]) -> dict[str, Any]:
        """合并会话数据
        
        将当前数据与存储的会话数据合并，非空值会覆盖原有值，空值只会在不存在时添加。
        
        Args:
            session_id: 会话ID
            current: 当前会话数据
            
        Returns:
            dict[str, Any]: 合并后的会话数据
        """
        # 获取之前的会话数据，如果不存在则使用空字典
        previous = self._sessions.get(session_id) or {}
        # 复制之前的会话数据，避免直接修改原数据
        merged = previous.copy()
        
        # 遍历当前数据的所有键值对
        for key, value in current.items():
            if value:
                # 如果值非空，覆盖原有值
                merged[key] = value
            else:
                # 如果值为空，只在键不存在时添加
                merged.setdefault(key, value)
        
        # 更新存储的会话数据
        self._sessions[session_id] = merged
        # 返回合并后的数据
        return merged

    def get(self, session_id: str) -> dict[str, Any] | None:
        """获取会话数据
        
        根据会话ID获取存储的会话数据。
        
        Args:
            session_id: 会话ID
            
        Returns:
            dict[str, Any] | None: 会话数据，如果不存在则返回None
        """
        return self._sessions.get(session_id)