from fastapi import APIRouter, Request

from app.agent.runtime import AgentRuntime
from app.schemas.contracts import AgentInvokeRequest, AgentInvokeResponse, AgentValidationErrorResponse

router = APIRouter(prefix="/api/v1/agent", tags=["agent"])


@router.post(
    "/invoke",
    response_model=AgentInvokeResponse,
    responses={422: {"model": AgentValidationErrorResponse, "description": "请求格式或 Agent 输入无效"}},
)
async def invoke_agent(request: AgentInvokeRequest, http_request: Request) -> AgentInvokeResponse:
    """按请求中的 agentCode 调用已注册的 Agent。"""
    runtime: AgentRuntime = http_request.app.state.agent_runtime
    result = await runtime.invoke(request.agent_code, request.input, request.context)
    return AgentInvokeResponse(requestId=request.context.request_id, data=result.model_dump(mode="json"))
