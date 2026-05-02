from typing import Any

from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from pydantic import Field, BaseModel
from app.schemas.user_context_schema import UserContext
from app.agent.tools import AccessDiagnosisToolExecutor
from app.clients.estate_ai_client import EstateAiClient

executor = AccessDiagnosisToolExecutor(EstateAiClient())

# @tool
# async def get_person_info(
#     project_id: int,
#     telephone: str | None = None,
#     personName: str | None = None,
#     ) -> dict[str, Any]:
#     """
#     通过手机号或人员姓名查询人员信息
#
#     Args:
#         project_id: 项目ID，用于区分不同物业项目
#         telephone: 手机号（可选）
#         personName: 人员姓名（可选）
#
#     Returns:
#         dict: 包含诊断结果的字典
#
#     Note:
#         telephone、personName至少需要提供一个
#     """
#     normalized = {
#         "telephone": telephone,
#         "personName": personName,
#     }
#     result = await executor.run_diagnosis(project_id, normalized)
#     return result.output

class PersonInfoRequest(BaseModel):
    telephone: str | None=Field(None,description="手机号")
    personName: str | None=Field(None,description="人员姓名")


@tool(args_schema=PersonInfoRequest)
async def get_person_info(
    tool_runtime: ToolRuntime,
    telephone: str | None = None,
    personName: str | None = None,
    ) -> dict[str, Any]:
    """
    通过手机号或人员姓名查询人员信息

    Args:
        telephone: 手机号（可选）
        personName: 人员姓名（可选）

    Returns:
        dict: 包含诊断结果的字典

    Note:
        telephone、personName至少需要提供一个
    """
    # print(tool_runtime.context.project_id)
    print(tool_runtime)
    result = {"content":f"我是{personName}，我的手机号是{telephone}，19岁，在上海念大学，毕业后是程序员"}
    return result.get("content")