from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.agent.exceptions import AgentInputError, AgentNotFoundError
from app.llm.deepseek import DeepSeekInvocationError


async def agent_not_found_handler(request: Request, exc: AgentNotFoundError) -> JSONResponse:
    return JSONResponse(
        status_code=404,
        content={
            "success": False,
            "error": {
                "code": "AGENT_NOT_FOUND",
                "message": str(exc),
                "agentCode": exc.agent_code,
            },
        },
    )


async def request_validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "error": {
                "code": "INVALID_REQUEST",
                "message": "请求格式无效",
                "details": [{"loc": error["loc"], "type": error["type"]} for error in exc.errors()],
            },
        },
    )


async def agent_input_handler(request: Request, exc: AgentInputError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "error": {
                "code": "INVALID_AGENT_INPUT",
                "message": str(exc),
                "details": [
                    {"loc": ["input", *error["loc"]], "type": error["type"]} for error in exc.details
                ],
            },
        },
    )


async def deepseek_invocation_handler(request: Request, exc: DeepSeekInvocationError) -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content={"success": False, "error": {"code": "LLM_UNAVAILABLE", "message": str(exc)}},
    )
