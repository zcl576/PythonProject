from __future__ import annotations

from time import perf_counter
from typing import Any

from pydantic import BaseModel, Field


class AgentTraceEvent(BaseModel):
    """代理追踪事件模型
    
    表示代理执行过程中的单个事件，包含事件的阶段、消息、耗时和元数据。
    """
    stage: str  # 事件阶段名称
    message: str  # 事件描述消息
    elapsed_ms: int  # 从追踪开始到该事件的耗时（毫秒）
    metadata: dict[str, Any] = Field(default_factory=dict)  # 事件相关的元数据


class AgentTrace(BaseModel):
    """代理追踪模型
    
    记录代理执行的完整追踪信息，包含追踪ID和所有事件。
    """
    trace_id: str  # 追踪唯一标识符
    events: list[AgentTraceEvent] = Field(default_factory=list)  # 追踪事件列表

    @classmethod
    def start(cls, trace_id: str) -> "AgentTrace":
        """创建新的追踪实例
        
        Args:
            trace_id: 追踪唯一标识符
            
        Returns:
            AgentTrace: 新创建的追踪实例
        """
        return cls(trace_id=trace_id)

    def add(self, stage: str, message: str, metadata: dict[str, Any] | None = None) -> None:
        """添加追踪事件
        
        Args:
            stage: 事件阶段名称
            message: 事件描述消息
            metadata: 事件相关的元数据，默认为空字典
        """
        self.events.append(
            AgentTraceEvent(
                stage=stage,
                message=message,
                elapsed_ms=0,
                metadata=metadata or {},
            )
        )


class TraceRecorder:
    """追踪记录器
    
    用于记录代理执行过程中的事件和时间，提供完整的执行追踪功能。
    """
    def __init__(self, trace_id: str) -> None:
        """初始化追踪记录器
        
        Args:
            trace_id: 追踪唯一标识符
        """
        self._started_at = perf_counter()  # 记录开始时间
        self.trace = AgentTrace.start(trace_id)  # 创建追踪实例

    def add(self, stage: str, message: str, metadata: dict[str, Any] | None = None) -> None:
        """添加追踪事件
        
        Args:
            stage: 事件阶段名称
            message: 事件描述消息
            metadata: 事件相关的元数据，默认为空字典
        """
        self.trace.events.append(
            AgentTraceEvent(
                stage=stage,
                message=message,
                elapsed_ms=int((perf_counter() - self._started_at) * 1000),  # 计算耗时
                metadata=metadata or {},
            )
        )

    def steps(self) -> list[str]:
        """获取所有追踪事件的消息列表
        
        Returns:
            list[str]: 所有事件的消息列表
        """
        return [event.message for event in self.trace.events]