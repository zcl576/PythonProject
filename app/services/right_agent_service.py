import json
import os.path
import uuid
from functools import lru_cache

from langchain.agents import create_agent
from langchain_deepseek import ChatDeepSeek
from pydantic import SecretStr

from app.config import get_settings
from app.memory.mysql_memory import get_mysql_saver
from app.schemas.right_schema import AgentResponse, AgentRequest
from app.schemas.user_context_schema import UserContext
from app.tools.right_tool import get_person_info
from langchain.messages import HumanMessage


class RightAgentService:
    def __init__(self):
        self._tool = [get_person_info]
        self._settings = get_settings()

    async def do_execute(self,request:AgentRequest,project_id:str) -> AgentResponse:
        llm = ChatDeepSeek(
            model=self._settings.llm_model,
            api_key=SecretStr(self._settings.llm_api_key or ""),
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )

        thread_id = request.session_id if request.session_id else uuid.uuid4().hex
        agent = create_agent(
            model=llm,
            tools=self._tool,
            system_prompt=self._get_system_prompt(),
            checkpointer=await get_mysql_saver(),
        )
        human_message = HumanMessage(content=request.question)
        agent_invoke = await agent.ainvoke(input={"messages": [human_message]}
                                           , config={"configurable": {"thread_id": thread_id}}
                                           , context=UserContext(project_id=project_id))
        return AgentResponse(answer=agent_invoke["messages"][-1].content, status="ok",follow_up_question="hao")


    @lru_cache
    def _get_system_prompt(self) -> str:
        os_path_join = os.path.join(self._settings.base_dir, "prompts", "access_agent_system_v1.md")
        with open(os_path_join, "r", encoding="utf-8") as file:
            return file.read()