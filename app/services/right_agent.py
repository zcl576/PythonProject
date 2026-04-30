from langchain.agents import create_agent
from langchain_deepseek import ChatDeepSeek

from app.schemas.right import AgentDiagnosisRequest, AgentDiagnosisResponse

content = "请在 PROJECT-ID 请求头中提供项目ID，我再继续诊断。"


class RightAgent:
    async def do_execute(
        self,
        request: AgentDiagnosisRequest,
        project_id: str) -> AgentDiagnosisResponse:


        model = ChatDeepSeek(
            "gemini-3.1-pro-preview",
            model_provider="google-genai",
            temperature=0.5,
            timeout=600,
            max_tokens=25000,
            streaming=True,
        )
        agent = create_agent(
            model=model,
            tools=[fetch_text_from_url],
            system_prompt=SYSTEM_PROMPT,
            checkpointer=checkpointer,
        )



