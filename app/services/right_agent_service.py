from langchain_core.messages import HumanMessage, SystemMessage
from langchain_deepseek import ChatDeepSeek
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import SecretStr

import json
import os.path
import uuid
from app.config import get_settings
from app.memory.mysql_memory import get_mysql_saver
from app.schemas.right_schema import AgentResponse, AgentRequest
from app.tools.right_tool import get_person_info
from functools import lru_cache
from typing import TypedDict


class StateData(TypedDict,total=False):
    question: str
    llm_unstarted_result: str
    tool_result: str
    llm_answer: str


def _route_unstarted(state_data:StateData) -> str:
    llm_unstarted_result = state_data.get("llm_unstarted_result")
    llm_result = json.loads(llm_unstarted_result);
    flag = all(not item for item in llm_result.values())
    if flag:
        return "answer"
    else:
        return "call_tool"


class RightAgentService:
    def __init__(self):
        self._tool = [get_person_info]
        self._settings = get_settings()

    async def do_execute(self,request:AgentRequest,project_id:str) -> AgentResponse:

        thread_id = request.session_id if request.session_id else uuid.uuid4().hex
        config = {"configurable": {"thread_id": thread_id}}
        graph = StateGraph(StateData)
        graph.add_node("unstarted",self.unstarted)
        graph.add_node("answer",self.answer)
        graph.add_node("call_tool",self.call_tool)

        graph.add_edge(START,"unstarted")
        graph.add_conditional_edges("unstarted", _route_unstarted,{"answer":"answer","call_tool":"call_tool"})
        graph.add_edge("call_tool", "answer")
        graph.add_edge("answer",END)
        graph_compile:CompiledStateGraph = graph.compile(checkpointer=await get_mysql_saver())
        ainvoke = await graph_compile.ainvoke(input={"question":request.question}, config=config)
        return AgentResponse(answer=ainvoke.get("llm_answer", ""),status="",follow_up_question="")

    async def unstarted(self,state_data:StateData) -> StateData:
        question = state_data.get("question")
        result = await self._json_chat(question)
        return StateData(llm_unstarted_result=result)

    async def answer(self,state_data:StateData) -> StateData:
        data = state_data.get("tool_result") if state_data.get("tool_result") else state_data.get("question")
        chat = await self._text_chat(data)
        return StateData(llm_answer=chat)

    async def call_tool(self,state_data:StateData) -> StateData:
        return state_data

    async def _text_chat(self, question: str) -> str:
        """与 LLM 服务进行文本对话

        向 LLM 服务发送请求并获取文本响应，不要求 JSON 格式。

        Args:
            messages: 消息列表，包含系统消息和用户消息
            temperature: 生成文本的随机性，0 表示确定性输出

        Returns:
            str: LLM 生成的响应内容
        """
        model = ChatDeepSeek(
            model=self._settings.llm_model,
            temperature=0,
            api_key=SecretStr(self._settings.llm_api_key or ""),
            base_url=self._settings.llm_base_url,
            timeout=self._settings.llm_timeout_seconds,
            max_retries=2,
        )
        response = await model.ainvoke(input=[SystemMessage(content="你是门禁权限运营助手。请根据给定工具结果生成一句面向用户的中文回答。"
                    "只能使用输入中提供的信息，不要编造权限状态或操作结果。"
                    "不要暴露 project_id、session_id、trace_id、原始 JSON 或内部编排字段。"
                    "只有 write_result 明确表示成功时，才能说已经完成写操作。"), HumanMessage(content=question)])
        return str(response.content)

    async def _json_chat(self, question:str) -> str:
        """与 LLM 服务进行对话

        向 LLM 服务发送请求并获取响应。

        Args:
            messages: 消息列表，包含系统消息和用户消息
            temperature: 生成文本的随机性，0 表示确定性输出

        Returns:
            str: LLM 生成的响应内容
        """
        model = ChatDeepSeek(
            model=self._settings.llm_model,
            temperature=0,
            api_key=SecretStr(self._settings.llm_api_key or ""),
            base_url=self._settings.llm_base_url,
            timeout=self._settings.llm_timeout_seconds,
            max_retries=2,
            model_kwargs={"response_format": {"type": "json_object"}},
        )
        response = await model.ainvoke(input=[SystemMessage(content=self._get_system_prompt()),HumanMessage(content=question)])
        return str(response.content)

    @lru_cache
    def _get_system_prompt(self) -> str:
        os_path_join = os.path.join(self._settings.base_dir, "prompts", "field_extraction_system_v1.md")
        with open(os_path_join, "r", encoding="utf-8") as file:
            return file.read()

    @lru_cache
    def _get_result_prompt(self) -> str:
        os_path_join = os.path.join(self._settings.base_dir, "prompts", "access_agent_system_v1.md")
        with open(os_path_join, "r", encoding="utf-8") as file:
            return file.read()