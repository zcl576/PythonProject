from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from app.agent.tools import AccessDiagnosisToolExecutor
from app.clients.estate_ai_client import EstateAiClient


class AccessDiagnosisInput(BaseModel):
    """
    门禁诊断工具的输入参数模型
    
    该模型定义了门禁诊断功能所需的参数，包括项目标识和用户/设备标识信息。
    """
    project_id: int = Field(description="项目 ID")
    personId: str | None = Field(default=None, description="人员 ID")
    telephone: str | None = Field(default=None, description="手机号")
    cardNo: str | None = Field(default=None, description="卡号")
    deviceId: str | None = Field(default=None, description="设备 ID")
    deviceName: str | None = Field(default=None, description="设备名称")
    deviceSn: str | None = Field(default=None, description="设备 SN")
    personName: str | None = Field(default=None, description="人员姓名")


class EstateAITool:
    """
    门禁诊断工具类
    
    该类封装了门禁诊断功能，将其作为 LangChain 工具提供给 AI 代理使用。
    支持根据人员标识（如手机号、卡号、人员ID）或设备ID进行门禁异常诊断。
    """
    def __init__(self, client: EstateAiClient | None = None) -> None:
        """
        初始化门禁诊断工具
        
        Args:
            client: 物业AI客户端实例，如果未提供则使用默认实例
        """
        executor = AccessDiagnosisToolExecutor(client or EstateAiClient())

        async def get_diagnose(
            project_id: int,
            personId: str | None = None,
            telephone: str | None = None,
            cardNo: str | None = None,
            deviceId: str | None = None,
            deviceName: str | None = None,
            deviceSn: str | None = None,
            personName: str | None = None,
        ) -> dict[str, Any]:
            """
            诊断门禁打不开、刷不开、进不去等异常原因
            
            该函数是工具的核心处理逻辑，接收用户标识信息并执行诊断
            
            Args:
                project_id: 项目ID，用于区分不同物业项目
                personId: 人员ID（可选）
                telephone: 手机号（可选）
                cardNo: 卡号（可选）
                deviceId: 设备ID（可选）
                deviceName: 设备名称（可选）
                deviceSn: 设备SN（可选）
                personName: 人员姓名（可选）
                
            Returns:
                dict: 包含诊断结果的字典
                
            Note:
                personId、telephone、cardNo、personName、deviceId、deviceName、deviceSn 至少需要提供一个
            """
            normalized = {
                "personId": personId,
                "telephone": telephone,
                "cardNo": cardNo,
                "deviceId": deviceId,
                "deviceName": deviceName,
                "deviceSn": deviceSn,
                "personName": personName,
            }
            result = await executor.run_diagnosis(project_id, normalized)
            return result.output

        self.tools = [
            StructuredTool.from_function(
                coroutine=get_diagnose,
                name="get_diagnose",
                description=(
                    "当用户询问门禁打不开、刷不开、进不去、开不了门等异常原因时使用。"
                    "需要 project_id，以及 personId、telephone、cardNo、personName、deviceId、deviceName、deviceSn 中至少一个。"
                ),
                args_schema=AccessDiagnosisInput,
            )
        ]
