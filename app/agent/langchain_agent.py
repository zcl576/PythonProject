from __future__ import annotations

import json
import os
from app.memory.redis_memory import get_redis_saver
from functools import lru_cache
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import BaseMessage
from langchain_deepseek import ChatDeepSeek
from pydantic import SecretStr

from app.config import get_settings
from app.tools.estate_ai_tool import EstateAITool


class LangChainAgentResult:
    """LangChain Agent 执行结果
    
    该类封装了 LangChain Agent 执行后的结果，包括最终答案、工具输出和对话消息。
    """
    def __init__(self, answer: str, tool_outputs: list[dict[str, Any]], messages: list[BaseMessage]) -> None:
        """初始化结果对象
        
        Args:
            answer: 最终的答案文本
            tool_outputs: 工具调用的输出结果列表
            messages: 交互过程中的消息列表
        """
        self.answer = answer  # 最终生成的答案文本
        self.tool_outputs = tool_outputs  # 工具调用产生的输出结果
        self.messages = messages  # Agent交互过程中产生的消息列表


class LangChainAccessAgent:
    """基于 LangChain 的门禁访问 AI 代理
    
    该代理利用大语言模型和工具调用能力，智能化地处理门禁诊断请求。
    支持自主规划工具调用顺序，以获取最准确的诊断结果。
    """
    def __init__(self) -> None:
        """初始化 LangChain 门禁代理
        
        从系统设置中加载 LLM 配置，并初始化所需的组件，
        包括模型配置、API密钥和可用工具列表。
        """
        settings = get_settings()
        self._enabled = settings.llm_enabled and bool(settings.llm_api_key)  # 标记代理是否启用
        self._model = settings.llm_model  # LLM 模型名称
        self._api_key = settings.llm_api_key  # LLM API 密钥
        self._base_url = settings.llm_base_url.rstrip("/")  # LLM API 基础URL
        self._tools = EstateAITool().tools  # 加载门禁诊断相关的工具列表
        self._system_prompt = load_access_agent_system_prompt()

    @property
    def enabled(self) -> bool:
        """获取代理启用状态
        
        Returns:
            bool: 如果代理已启用返回 True，否则返回 False
        """
        return self._enabled

    async def run(self, project_id: int, question: str, known_fields: dict[str, Any], session_id: str) -> LangChainAgentResult:
        """运行 LangChain Agent 执行门禁诊断
        
        该方法创建并运行一个 AI Agent，让它根据提供的信息自主调用工具进行门禁诊断。
        Agent 会根据系统提示词和用户问题，智能地选择合适的工具并处理结果。
        
        Args:
            project_id: 项目ID，用于区分不同的物业项目
            question: 用户提出的问题
            known_fields: 已知的字段信息，如手机号、卡号、人员ID、设备ID等
            
        Returns:
            LangChainAgentResult: 包含答案、工具输出和消息的执行结果
            
        Raises:
            RuntimeError: 当代理未启用时抛出异常
        """
        if not self._enabled:
            raise RuntimeError("LangChain agent is disabled")

        # 创建 DeepSeek 模型实例
        model = ChatDeepSeek(
            model=self._model,  # 使用配置的模型名称
            api_key=SecretStr(self._api_key or ""),  # 使用配置的API密钥
            base_url=self._base_url,  # 使用配置的基础URL
            temperature=0,  # 设置温度为0，确保输出一致性
            max_retries=2,  # 设置最大重试次数
        )
        
        # 创建 LangChain Agent
        agent = create_agent(
            model=model,  # 使用上述创建的模型
            tools=self._tools,  # 使用门禁诊断相关的工具
            checkpointer=get_redis_saver(),
            system_prompt=self._system_prompt,  # 系统提示词，定义AI行为
        )
        
        # 构建请求负载
        payload = {
            "project_id": project_id,  # 项目ID
            "question": question,  # 用户问题
            "known_fields": known_fields,  # 已知字段
        }
        
        # 异步调用 Agent
        response = await agent.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",  # 用户角色
                        "content": json.dumps(payload, ensure_ascii=False),  # 序列化后的请求负载
                    }
                ]
            },
            config={"configurable": {"thread_id": session_id}}
        )
        
        # 提取消息和答案
        messages = response["messages"]  # 获取交互消息
        answer = messages[-1].content if messages else ""  # 获取最后一条消息的内容作为答案
        return LangChainAgentResult(
            answer=str(answer),  # 最终答案
            tool_outputs=self._extract_tool_outputs(messages),  # 提取的工具输出
            messages=messages,  # 完整的消息列表
        )

    def _extract_tool_outputs(self, messages: list[BaseMessage]) -> list[dict[str, Any]]:
        """从消息列表中提取工具输出
        
        该方法遍历所有消息，识别出工具调用类型的消息，并提取其内容。
        工具输出可能是字典格式或JSON字符串格式，该方法会处理这两种情况。
        
        Args:
            messages: 包含各种类型消息的列表
            
        Returns:
            list[dict[str, Any]]: 提取出的工具输出字典列表
        """
        outputs: list[dict[str, Any]] = []
        for message in messages:
            # 检查消息类型是否为工具调用
            if getattr(message, "type", None) != "tool":
                continue
            content = message.content
            # 如果内容已经是字典格式，直接添加
            if isinstance(content, dict):
                outputs.append(content)
                continue
            # 如果内容不是字符串格式，跳过
            if not isinstance(content, str):
                continue
            # 尝试解析JSON字符串
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                continue
            # 如果解析结果是字典格式，添加到输出列表
            if isinstance(parsed, dict):
                outputs.append(parsed)
        return outputs


@lru_cache
def load_access_agent_system_prompt() -> str:
    prompt_path = os.path.join(get_settings().base_dir, "prompts", "access_agent_system_v1.md")
    with open(prompt_path, "r", encoding="utf-8") as file:
        return file.read()
